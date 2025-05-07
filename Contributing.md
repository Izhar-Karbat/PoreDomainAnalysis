# **Pore Analysis Suite**

A comprehensive toolkit for analyzing ion channel structures and dynamics from molecular dynamics trajectory data.

## **Database-Centric Refactoring**

This codebase has been refactored to use a database-centric approach with the following key improvements:

1.  **Separation of Computation from Visualization**: Each analysis module now clearly separates calculation logic from visualization, enabling more flexible pipelines.
2.  **Database Integration**: All analysis products (output files), metrics (numerical results), configuration parameters, and metadata are registered in a SQLite database (`analysis_registry.db`) within each simulation's run directory. This facilitates tracking, reproducibility, and downstream comparative analysis.
3.  **Module Status Tracking**: The execution status (running, success, failed) of each analysis module is tracked in the database, enabling dependency management and incremental analysis.
4.  **Efficient Trajectory Handling**: The MDAnalysis `Universe` object (representing the trajectory) is loaded once in the main script and passed to computation modules, improving performance.
5.  **Trajectory Slicing**: Allows analysis of specific frame ranges using `--start_frame` and `--end_frame` command-line arguments.
6.  **Cross-Simulation Analysis**: Storing standardized metrics in the database makes it easier to compare results across different simulation runs.

## **Database Schema**

The database (SQLite) includes the following tables:

* `simulation_metadata`: General information about the simulation and analysis (including frame ranges used).
* `analysis_modules`: Tracks which analysis modules have been run, their status, parameters, and errors.
* `analysis_products`: Central registry of all files produced by the analysis (CSVs, PNGs, etc.), linked to the generating module.
* `metrics`: Stores numerical metrics (e.g., mean distances, counts) with units, linked to the generating module.
* `config_parameters`: Stores the values of parameters from `core/config.py` used for the run.
* `dependencies`: (Future use) Tracks dependencies between analysis modules.
* `product_relationships`: (Future use) Tracks relationships between analysis products (e.g., a plot derived from a CSV).

*(See `database.md` for full schema details)*

## **Module Structure**

The package is organized into the following structure:

```
pore_analysis/
├── core/                  # Core utilities (DB, config, filtering, logging, utils)
│   ├── database.py        # <-- Central database interaction logic
│   ├── config.py          # <-- Configuration constants (thresholds, etc.)
│   ├── filtering.py       # <-- Data filtering (PBC correction, smoothing)
│   ├── logging.py         # <-- Logging setup
│   └── utils.py           # <-- General helper functions
├── modules/               # Specific analysis tasks
│   ├── core_analysis/     # <-- Basic G-G/COM distance analysis
│   │   ├── computation.py # Calculation part (Accepts Universe)
│   │   └── visualization.py # Plotting part (Reads from DB)
│   ├── ion_analysis/      # Ion tracking and conduction (Accepts Universe)
│   ├── inner_vestibule_analysis/  # Water analysis (Accepts Universe)
│   ├── dw_gate_analysis/  # DW gate dynamics (Accepts Universe)
│   ├── tyrosine_analysis/ # SF tyrosine rotamers (Accepts Universe)
│   ├── orientation_contacts/ # Toxin orientation (Accepts Universe)
│   ├── gyration_analysis/ # Gyration radius analysis (Accepts Universe)
│   └── pocket_analysis/   # Pocket water analysis (Accepts Universe)
├── templates/             # HTML report templates
│   └── report_template.html # Main template & includes
├── main.py                # <-- Main script to run workflow (Loads Universe)
├── summary.py             # <-- Generates JSON summary from DB
├── html.py                # <-- Generates HTML report from summary/DB
├── plots_dict.json       # <-- Defines plots for HTML report
└── __init__.py            # Package indicator
```

## **Using the Refactored Modules**

Each analysis module follows a consistent pattern:

1.  **Loading (`main.py`)**: The main script loads the `MDAnalysis.Universe` object once and determines the frame range (full or sliced based on `--start_frame`/`--end_frame`).
2.  **Computation (`computation.py`)**: Accepts the `Universe` object and frame range (`start_frame`, `end_frame`). Performs calculations over the specified frames, saves results to files (e.g., CSV), registers itself and its products in the database, and stores key metrics.
3.  **Visualization (`visualization.py`)**: Retrieves necessary data file paths from the database, generates plots (e.g., PNG), saves them, and registers the plot files in the database.

### **Example: Core Analysis Workflow**

```python
# In main.py (simplified)
import MDAnalysis as mda
from pore_analysis.modules.core_analysis import analyze_trajectory, filter_and_save_data
# ... other imports ...

# Load Universe and determine frame range
universe = mda.Universe(psf_file, dcd_file)
n_frames_total = len(universe.trajectory)
actual_start = args.start_frame if args.start_frame is not None else 0
actual_end = n_frames_total if args.end_frame is None else min(args.end_frame, n_frames_total)
# ... add validation ...

# Run the analysis computation (pass Universe object and range)
results = analyze_trajectory(
    run_dir,
    universe=universe,
    start_frame=actual_start,
    end_frame=actual_end,
    db_conn=db_conn
)

# Filter the data (pass calculated data and potentially range info)
if results['status'] == 'success':
    filtered_results = filter_and_save_data(
        run_dir,
        # Pass data arrays from results['data']
        dist_ac=results['data']['dist_ac'],
        dist_bd=results['data']['dist_bd'],
        com_distances=results['data']['com_distances'],
        time_points=results['data']['time_points'],
        # Pass frame range for context if needed by the function
        start_frame=actual_start,
        end_frame=actual_end,
        # Other args...
        db_conn=db_conn,
        is_control_system=results['metadata']['is_control_system']
    )

# Visualization step (unchanged - retrieves data from DB)
from pore_analysis.modules.core_analysis import plot_distances, plot_kde_analysis

g_g_plots = plot_distances(run_dir, db_conn=db_conn, is_gg=True)
# ... etc ...
```

## **Command Line Interface**

The analysis workflow is orchestrated by main.py:

```bash
# Run all available/implemented analysis modules on the full trajectory
python -m pore_analysis.main --folder /path/to/simulation

# Run analysis only on frames 1000 to 4999 (exclusive end frame)
python -m pore_analysis.main --folder /path/to/simulation --start_frame 1000 --end_frame 5000

# Run specific analysis modules (e.g., core and ions) on a slice
python -m pore_analysis.main --folder /path/to/simulation --start_frame 500 --end_frame 2500 --core --ions

# Generate HTML report using existing database results (no re-analysis)
python -m pore_analysis.main --folder /path/to/simulation --summary-only --report

# Force rerun of core analysis for frames 0-999, even if DB shows 'success'
python -m pore_analysis.main --folder /path/to/simulation --end_frame 1000 --core --force-rerun

# Reinitialize database (WARNING: Deletes previous results for this run)
python -m pore_analysis.main --folder /path/to/simulation --reinit-db --core
```

## **Cross-Simulation Analysis**

The database-centric approach makes comparing metrics across simulations straightforward by querying the individual analysis_registry.db files:

```python
import os
import sqlite3
import pandas as pd

# List of simulation directories
sim_dirs = ['/path/to/sim1', '/path/to/sim2', '/path/to/sim3']

# Collect specific metrics (e.g., filtered COM Mean)
metrics_data = []
for sim_dir in sim_dirs:
    db_path = os.path.join(sim_dir, 'analysis_registry.db')
    if not os.path.exists(db_path):
        print(f"Warning: DB not found for {sim_dir}")
        continue

    conn = sqlite3.connect(db_path)
    # Use dict_factory for easy row access if defined in core.database
    # Assuming dict_factory is defined or imported
    conn.row_factory = sqlite3.Row # Or your dict_factory
    cursor = conn.cursor()

    try:
        # Get simulation name (assuming it's stored)
        cursor.execute("SELECT value FROM simulation_metadata WHERE key='run_name'")
        sim_name_row = cursor.fetchone()
        sim_name = sim_name_row['value'] if sim_name_row else os.path.basename(sim_dir) # Access by key

        # Get specific metrics
        cursor.execute("""
            SELECT metric_name, value, units
            FROM metrics
            WHERE metric_name LIKE '%Mean_Filt' OR metric_name LIKE '%Min_Filt'
        """) # Example query
        for row in cursor.fetchall():
            metrics_data.append({
                'simulation': sim_name,
                'metric_name': row['metric_name'], # Access by key
                'value': row['value'],
                'units': row['units']
            })
    except Exception as e:
        print(f"Error querying DB for {sim_dir}: {e}")
    finally:
        conn.close()

# Create DataFrame for analysis
if metrics_data:
    metrics_df = pd.DataFrame(metrics_data)
    print(metrics_df.head())
else:
    print("No metrics collected.")
```

## **Development Guidelines**

This section outlines the conventions and workflow for developing new analysis modules for the suite.

### **Database Interaction Rules**

#### **Module Status:**
- Register module start using `register_module(conn, module_name, status='running')`.
- Update status reliably upon completion (`update_module_status(..., status='success')`) or failure (`update_module_status(..., status='failed', error_message=...)`).

#### **Product Registration (`register_product`):**
- **Mandatory**: Register every generated file (CSV, JSON, PNG, etc.) intended for later use or tracking.
- **Strict Parameter Order**: Use the exact signature: `register_product(db_conn, module_name, product_type, category, relative_path, subcategory, description)`.
- **relative_path**: Must be relative to the run_dir (e.g., `module_subdir/output_file.csv`). Use `os.path.relpath(full_path, run_dir)`.
- **subcategory**: Use clear, consistent, and unique names for distinct products, especially plots (e.g., `filtered_distances`, `subunit_comparison`, `kde_analysis`). This is the primary identifier used by `html.py` via the plot configuration file.

#### **Metric Storage (`store_metric`):**
- **Mandatory**: Store all key numerical results for summary tables or comparisons.
- **Naming**: Use descriptive `metric_names`, indicating source if relevant (e.g., `G_G_AC_Mean_Filt`).
- **Units**: Provide correct units ("Å", "ns", "°", "count", etc.). Use `None` or `""` for unitless metrics.
- **Source**: Calculate metrics from the appropriate data (e.g., filtered data, data corresponding to the analyzed frame slice).

#### **Data Retrieval (`get_product_path`):**
- Visualization/reporting code must retrieve file paths from the database first.
- Use specific `product_type`, `category`, and `subcategory` arguments to find the correct product.
- Implement file system fallbacks cautiously, logging warnings.

#### **HTML Integration (via `plots_dict.json`):**
- The list of plots included in the HTML report is now defined in `pore_analysis/plots_dict.json`.
- This JSON file contains a list of objects, each defining a plot to be included. The required keys for each object are: `template_key`, `product_type`, `category`, `subcategory`, and `module_name`.
- `html.py` reads this file using the `load_plot_queries` function to determine which plots registered in the database should be fetched and embedded.
- Crucially, the `subcategory` in a `plots_dict.json` entry must exactly match the `subcategory` used during `register_product` in the visualization script that generates the corresponding plot.

### **Plotting Conventions**

#### **Titles:**
- **No Python Titles**: Do not add titles within the Python plotting code (`plt.title`, `ax.set_title`, `fig.suptitle`).
- **HTML Titles**: Use HTML structure (e.g., `<h3>`, `<h4>`) within the template (`templates/_tab_*.html`) for overall plot titles.
- **Multi-Panel Figures**: If a single image contains multiple conceptual panels (like the G-G plot), add descriptive labels in the HTML template above or beside the image (e.g., using flexbox divs) to identify the panels. Avoid overly complex figures if HTML layout can achieve the goal.

#### **Layout & Content:**
- **Target Context**: Design plots considering their final HTML display size (e.g., full-width, half-width column). Ensure legibility.
- **Completeness**: Verify all intended visual elements (lines, histograms, KDEs, legends, axis labels) are present and correct in the saved plot.
- **Simplicity**: Prefer generating simpler, single-purpose plots arranged using HTML layout (two-column class) over complex multi-panel images, unless essential for direct comparison within one image.

#### **Styling:**
- Adhere strictly to the `STYLE` dictionary defined in `pore_analysis/core/plotting_style.py` for colors, lines, markers, fonts.
- Use the `setup_style()` function (also defined in `pore_analysis/core/plotting_style.py`) within each visualization script to apply these settings.
- Be prepared to iteratively adjust `STYLE` values (e.g., font sizes) based on feedback for optimal legibility.

### **Testing**

- **Framework**: Tests are written using the pytest framework and are located in the `tests/` directory.
- **Dependencies**: To run the tests, ensure you have installed the development dependencies:

```bash
pip install -e .[dev]
```

- **Execution**: Run the test suite from the project root directory (the directory containing setup.py):

```bash
pytest
```

Or run with verbose output:

```bash
pytest -v
```

- **Coverage**: The current test suite covers core functionality and report generation. Contributions that add new analysis modules should ideally include corresponding tests (unit tests for utilities and integration tests for the module workflow if possible).

### **Module Development Workflow & Checklist**

1. **Define Goal**: Specify the analysis and desired outputs (data files, metrics, plots).
2. **Implement Computation (`computation.py`):**
   - Add calculation logic.
   - Accept `universe`, `start_frame`, `end_frame` arguments in function signature (instead of `psf_file`, `dcd_file`).
   - Remove internal `mda.Universe()` loading.
   - Iterate over the trajectory slice passed via `universe.trajectory[start_frame:end_frame:stride]`.
   - Adapt array initialization and indexing to use the slice length and relative frame indices.
   - Calculate time points based on original frame indices collected from the slice.
   - Save data file(s) (e.g., to `run_dir/module_name/`).
   - `register_module()` start (status='running').
   - `register_product()` for data file(s) (use correct `relative_path`, `subcategory`).
   - Calculate required metrics based on the analyzed slice.
   - `store_metric()` for each metric (use correct `metric_name`, `units`).
   - `update_module_status()` end (status='success' or 'failed').
3. **Implement Visualization (`visualization.py`):**
   - `get_product_path()` to retrieve data file path (using correct `subcategory`).
   - Load data (which now corresponds to the potentially sliced data).
   - Generate plot(s) (adhere to styling, omit Python titles). Ensure axes reflect the analyzed time range.
   - Save plot file(s) (e.g., to `run_dir/module_name/`).
   - `register_product()` for each plot file (use correct `relative_path`, unique `subcategory`).
4. **Update Reporting (`plots_dict.json` & `templates/`):**
   - (`plots_dict.json`): Add/modify JSON object(s) for the plot(s). Ensure `template_key` is unique and `product_type`, `category`, `subcategory`, `module_name` exactly match plot registration.
   - (`templates/`): Add/modify the relevant `_tab_*.html`. Add HTML structure (`<h3>`, `plot-container`, etc.) for titles/layout. Use `{% if plots.get('your_template_key') %}` to display images. Add/modify tables for new metrics.
5. **Add Tests (`tests/`):**
   - Add relevant unit or integration tests for the new module's functionality.
   - Update tests to handle the new function signatures (e.g., pass mock Universe objects instead of mocking file loading).
6. **Test & Verify:**
   - Run the full pipeline (`python -m pore_analysis.main ... --report`). Test with and without frame slicing flags.
   - Run the test suite (`pytest`).
   - Check the database (`analysis_registry.db`) for correct registration and metadata (including frame range info).
   - Verify the HTML report: Plots displayed correctly for full/sliced data? Metrics correct? Layout/styling okay? Titles correct?

## **Index of Database Contents**

### **Index of Common Metadata Keys (simulation_metadata table)**

- `schema_version`: Version of the database schema (e.g., 1.0.0).
- `db_init_timestamp`: ISO timestamp when the database was initialized.
- `run_name`: Name of the simulation run directory (e.g., R6).
- `system_name`: Name of the system directory (parent of run_name) (e.g., toxin).
- `analysis_start_time`: ISO timestamp when the main.py script started.
- `analysis_end_time`: ISO timestamp when the main.py script finished.
- `analysis_status`: Overall status of the analysis run (success, failed).
- `psf_file`: Basename of the PSF file used (e.g., step5_input.psf).
- `dcd_file`: Basename of the DCD file used (e.g., MD_Aligned.dcd).
- `analysis_version`: Version of the Pore Analysis Suite used (e.g., 2.0.0).
- `is_control_system`: Boolean ('True'/'False') indicating if toxin is absent.
- `filter_residues_dict`: JSON string representation of the filter residue dictionary ({'PROA': [63, 64, 65, 66, 67], ...}).
- `analysis_start_frame`: Starting frame index used for the analysis run.
- `analysis_end_frame`: Ending frame index (exclusive) used for the analysis run.
- `analysis_frames_total`: Total number of frames analyzed in the run (end_frame - start_frame).
- `trajectory_total_frames`: Total number of frames in the original DCD file.

### **Index of Analysis Modules (analysis_modules table)**

- `core_analysis`: Raw trajectory reading (G-G, COM distances).
- `core_analysis_filtering`: Filtering of core distance data.
- `core_analysis_visualization_g_g`: Plotting for G-G distances.
- `core_analysis_visualization_com`: Plotting for COM distance/KDE.
- `orientation_analysis`: Toxin orientation/contact computation.
- `orientation_analysis_visualization`: Toxin orientation/contact plotting.
- `ion_analysis`: Ion analysis computation (tracking, sites, occupancy, HMM).
- `ion_analysis_visualization`: Ion analysis plotting.
- `inner_vestibule_analysis`: Inner vestibule water computation.
- `inner_vestibule_analysis_visualization`: Inner vestibule water plotting.
- `gyration_analysis`: Carbonyl Gyration computation.
- `gyration_analysis_visualization`: Carbonyl Gyration plotting.
- `tyrosine_analysis`: SF Tyrosine rotamer computation (HMM-based).
- `tyrosine_analysis_visualization`: SF Tyrosine rotamer plotting.
- `dw_gate_analysis`: DW-Gate computation.
- `dw_gate_analysis_visualization`: DW-Gate plotting.
- `pocket_analysis`: Peripheral pocket water analysis computation (ML-based).
- `pocket_analysis_visualization`: Peripheral pocket water analysis plotting.
- `report_generation`: HTML report generation process.
- `print_report_generation`: Print-friendly report generation process.
- (Future modules...)

### **Index of Analysis Products (analysis_products table)**

(This section lists typical files generated, identified by their subcategory. Add entries for new modules)

#### **Module: core_analysis**
- Category: data, Subcategory: raw_distances, Type: csv, Path: core_analysis/Raw_Distances.csv

#### **Module: core_analysis_filtering**
- Category: data, Subcategory: g_g_distance_filtered, Type: csv, Path: core_analysis/G_G_Distance_Filtered.csv
- Category: data, Subcategory: com_stability_filtered, Type: csv, Path: core_analysis/COM_Stability_Filtered.csv

#### **Module: core_analysis_visualization_g_g**
- Category: plot, Subcategory: subunit_comparison, Type: png, Path: core_analysis/G_G_Distance_Subunit_Comparison.png

#### **Module: core_analysis_visualization_com**
- Category: plot, Subcategory: comparison, Type: png, Path: core_analysis/COM_Stability_Comparison.png
- Category: plot, Subcategory: kde_analysis, Type: png, Path: core_analysis/COM_Stability_KDE_Analysis.png

#### **Module: orientation_analysis**
- Category: data, Subcategory: orientation_timeseries, Type: csv, Path: orientation_contacts/orientation_data.csv
- Category: data, Subcategory: residue_frequency, Type: csv, Path: orientation_contacts/residue_contact_frequency.csv

#### **Module: orientation_analysis_visualization**
- Category: plot, Subcategory: orientation_angle, Type: png, Path: orientation_contacts/Toxin_Orientation_Angle.png
- Category: plot, Subcategory: rotation_components, Type: png, Path: orientation_contacts/Toxin_Rotation_Components.png
- Category: plot, Subcategory: channel_contacts, Type: png, Path: orientation_contacts/Toxin_Channel_Contacts.png
- Category: plot, Subcategory: contact_map_focused, Type: png, Path: orientation_contacts/Toxin_Channel_Residue_Contact_Map_Focused.png

#### **Module: ion_analysis**
- Category: data, Subcategory: ion_positions_g1_centric, Type: csv, Path: ion_analysis/ion_positions_g1_centric.csv
- Category: data, Subcategory: ion_positions_absolute, Type: csv, Path: ion_analysis/ion_positions_absolute.csv
- Category: data, Subcategory: ion_filter_presence, Type: csv, Path: ion_analysis/ion_filter_presence.csv
- Category: definition, Subcategory: binding_sites_definition, Type: txt, Path: ion_analysis/binding_site_positions_g1_centric.txt
- Category: plot, Subcategory: site_optimization_plot, Type: png, Path: ion_analysis/binding_site_optimization.png
- Category: data, Subcategory: ion_occupancy_per_frame, Type: csv, Path: ion_analysis/ion_occupancy_per_frame.csv
- Category: data, Subcategory: ion_site_statistics, Type: csv, Path: ion_analysis/ion_site_statistics.csv
- Category: data, Subcategory: ion_hmm_dwell_events, Type: csv, Path: ion_analysis/ion_hmm_dwell_events.csv
- Category: data, Subcategory: ion_hmm_quality_data, Type: csv, Path: ion_analysis/ion_hmm_quality_data.csv
- Category: data, Subcategory: ion_hmm_conduction_events, Type: csv, Path: ion_analysis/ion_hmm_conduction_events.csv

#### **Module: ion_analysis_visualization**
- Category: plot, Subcategory: binding_sites_visualization, Type: png, Path: ion_analysis/binding_sites_g1_centric_visualization.png
- Category: plot, Subcategory: combined_plot, Type: png, Path: ion_analysis/K_Ion_Combined_Plot.png
- Category: plot, Subcategory: occupancy_heatmap, Type: png, Path: ion_analysis/K_Ion_Occupancy_Heatmap.png
- Category: plot, Subcategory: average_occupancy, Type: png, Path: ion_analysis/K_Ion_Average_Occupancy.png
- Category: plot, Subcategory: hmm_transitions_plot, Type: png, Path: ion_analysis/ion_transitions_hmm.png
- Category: plot, Subcategory: quality_metrics_plot, Type: png, Path: ion_analysis/ion_quality_metrics.png

#### **Module: inner_vestibule_analysis**
- Category: data, Subcategory: occupancy_per_frame, Type: csv, Path: inner_vestibule_analysis/inner_vestibule_occupancy.csv
- Category: data, Subcategory: residence_times, Type: json, Path: inner_vestibule_analysis/inner_vestibule_residence_times.json

#### **Module: inner_vestibule_analysis_visualization**
- Category: plot, Subcategory: count_plot, Type: png, Path: inner_vestibule_analysis/inner_vestibule_count_plot.png
- Category: plot, Subcategory: residence_hist, Type: png, Path: inner_vestibule_analysis/inner_vestibule_residence_hist.png

#### **Module: gyration_analysis**
- Category: data, Subcategory: g1_gyration_data, Type: csv, Path: gyration_analysis/G1_gyration_radii.csv
- Category: data, Subcategory: y_gyration_data, Type: csv, Path: gyration_analysis/Y_gyration_radii.csv
- Category: data, Subcategory: gyration_flip_events, Type: csv, Path: gyration_analysis/gyration_flip_events.csv

#### **Module: gyration_analysis_visualization**
- Category: plot, Subcategory: g1_gyration, Type: png, Path: gyration_analysis/G1_gyration_radii_stacked.png
- Category: plot, Subcategory: y_gyration, Type: png, Path: gyration_analysis/Y_gyration_radii_stacked.png
- Category: plot, Subcategory: flip_duration, Type: png, Path: gyration_analysis/Flip_Duration_Distribution.png

#### **Module: tyrosine_analysis**
- Category: data, Subcategory: raw_dihedrals, Type: csv, Path: tyrosine_analysis/sf_tyrosine_raw_dihedrals.csv
- Category: data, Subcategory: hmm_dwell_events, Type: csv, Path: tyrosine_analysis/sf_tyrosine_hmm_dwell_events.csv
- Category: data, Subcategory: hmm_state_path, Type: csv, Path: tyrosine_analysis/sf_tyrosine_hmm_state_path.csv

#### **Module: tyrosine_analysis_visualization**
- Category: plot, Subcategory: dihedrals_chi1, Type: png, Path: tyrosine_analysis/SF_Tyrosine_Chi1_Dihedrals_HMM.png
- Category: plot, Subcategory: dihedrals_chi2, Type: png, Path: tyrosine_analysis/SF_Tyrosine_Chi2_Dihedrals_HMM.png
- Category: plot, Subcategory: rotamer_scatter, Type: png, Path: tyrosine_analysis/SF_Tyrosine_Rotamer_Scatter.png
- Category: plot, Subcategory: rotamer_population, Type: png, Path: tyrosine_analysis/SF_Tyrosine_Rotamer_Population_HMM.png

#### **Module: dw_gate_analysis**
- Category: data, Subcategory: raw_dw_distances, Type: csv, Path: dw_gate_analysis/dw_gate_raw_distances.csv
- Category: data, Subcategory: kde_plot_data, Type: json, Path: dw_gate_analysis/dw_gate_kde_plot_data.json
- Category: data, Subcategory: debounced_states, Type: csv, Path: dw_gate_analysis/dw_gate_debounced_states.csv
- Category: data, Subcategory: dw_events, Type: csv, Path: dw_gate_analysis/dw_gate_events.csv
- Category: data, Subcategory: dw_summary_stats, Type: csv, Path: dw_gate_analysis/dw_gate_summary_stats.csv
- Category: data, Subcategory: dw_probabilities, Type: csv, Path: dw_gate_analysis/dw_gate_probabilities.csv

#### **Module: dw_gate_analysis_visualization**
- Category: plot, Subcategory: distance_distribution, Type: png, Path: dw_gate_analysis/dw_gate_distance_distribution.png
- Category: plot, Subcategory: distance_vs_state, Type: png, Path: dw_gate_analysis/dw_gate_distance_vs_state.png
- Category: plot, Subcategory: open_probability, Type: png, Path: dw_gate_analysis/dw_gate_open_probability.png
- Category: plot, Subcategory: state_heatmap, Type: png, Path: dw_gate_analysis/dw_gate_state_heatmap.png
- Category: plot, Subcategory: duration_distributions, Type: png, Path: dw_gate_analysis/dw_gate_duration_distribution.png

#### **Module: pocket_analysis**
- Category: data, Subcategory: pocket_occupancy_timeseries, Type: csv, Path: pocket_analysis/pocket_occupancy.csv
- Category: data, Subcategory: pocket_residence_stats, Type: json, Path: pocket_analysis/pocket_residence_stats.json
- Category: data, Subcategory: pocket_assignments_per_frame, Type: pkl, Path: pocket_analysis/pocket_assignments.pkl
- Category: data, Subcategory: pocket_imbalance_metrics, Type: csv, Path: pocket_analysis/pocket_imbalance_metrics.csv
- Category: summary, Subcategory: pocket_analysis_summary, Type: txt, Path: pocket_analysis/pocket_summary.txt

#### **Module: pocket_analysis_visualization**
- Category: plot, Subcategory: pocket_occupancy_plot, Type: png, Path: pocket_analysis/pocket_occupancy_plot.png
- Category: plot, Subcategory: pocket_residence_histogram, Type: png, Path: pocket_analysis/pocket_residence_distribution.png
- Category: plot, Subcategory: pocket_residence_analysis, Type: png, Path: pocket_analysis/pocket_residence_analysis.png

#### **Module: summary (or main)**
- Category: summary, Subcategory: analysis_summary, Type: json, Path: analysis_summary.json

#### **Module: report_generation**
- Category: report, Subcategory: analysis_report, Type: html, Path: data_analysis_report.html

#### **Module: print_report_generation**
- Category: report, Subcategory: print_report, Type: html, Path: print_report.html
- Category: report, Subcategory: print_report_pdf, Type: pdf, Path: print_report.pdf (If PDF generation enabled and successful)

### **Index of Metrics (metrics table)**

(This section lists typical metrics stored, identified by metric_name. Add entries for new modules)

#### **Module: core_analysis_filtering**
- COM_Max_Filt, COM_Mean_Filt, COM_Min_Filt, COM_Std_Filt (Units: Å)
- G_G_AC_Max_Filt, G_G_AC_Mean_Filt, G_G_AC_Min_Filt, G_G_AC_Std_Filt (Units: Å)
- G_G_BD_Max_Filt, G_G_BD_Mean_Filt, G_G_BD_Min_Filt, G_G_BD_Std_Filt (Units: Å)
- (Percentile metrics like G_G_AC_Pctl90_Filt, COM_Pctl50_Filt also stored)

#### **Module: orientation_analysis**
- Orient_Angle_Mean, Orient_Angle_Std (Units: °)
- Orient_Contacts_Mean, Orient_Contacts_Std (Units: count)
- Orient_RotX_Mean, Orient_RotX_Std (Units: °)
- Orient_RotY_Mean, Orient_RotY_Std (Units: °)
- Orient_RotZ_Mean, Orient_RotZ_Std (Units: °)

#### **Module: ion_analysis (Occupancy and HMM Stats)**
- Ion_AvgOcc_{Site}, Ion_MaxOcc_{Site} (Units: count)
- Ion_PctTimeOcc_{Site} (Units: %)
- Ion_HMM_TransitionEvents_Total, _Upward, _Downward (Units: count)
- Ion_HMM_Transition_{S1}_{S2} (Units: count)
- Ion_HMM_ConductionEvents_Total, _Outward, _Inward (Units: count)
- Ion_HMM_Conduction_MeanTransitTime_ns, _MedianTransitTime_ns, _StdTransitTime_ns (Units: ns)

#### **Module: inner_vestibule_analysis**
- InnerVestibule_AvgResidenceTime_ns, InnerVestibule_MedianResidenceTime_ns (Units: ns)
- InnerVestibule_ExchangeRatePerNs (Units: rate (ns^-1))
- InnerVestibule_MeanOcc, InnerVestibule_StdOcc (Units: count)
- InnerVestibule_TotalExitEvents (Units: count)

#### **Module: gyration_analysis**
- Gyration_G1_Mean, Gyration_G1_Std (Units: Å)
- Gyration_Y_Mean, Gyration_Y_Std (Units: Å)
- Gyration_G1_OnFlips, Gyration_G1_OffFlips (Units: count)
- Gyration_Y_OnFlips, Gyration_Y_OffFlips (Units: count)
- Gyration_G1_MeanDuration_ns, Gyration_G1_StdDuration_ns, Gyration_G1_MaxDuration_ns (Units: ns)
- Gyration_Y_MeanDuration_ns, Gyration_Y_StdDuration_ns, Gyration_Y_MaxDuration_ns (Units: ns)

#### **Module: tyrosine_analysis**
- Tyr_HMM_TotalTransitions (Units: count)
- Tyr_HMM_MeanDwell_{state}, Tyr_HMM_MedianDwell_{state}, Tyr_HMM_StdDwell_{state} (Units: ns) (e.g., Tyr_HMM_MeanDwell_pp)
- Tyr_HMM_Population_{state} (Units: %) (e.g., Tyr_HMM_Population_mt)
- Config_TyrHMM_...: Parameters used (EmissionSigma, SelfTransitionP, Epsilon, FlickerNs)

#### **Module: dw_gate_analysis**
- DW_{Chain}_{State}_Mean_ns, _Median_ns, _Std_Dev_ns (Units: ns) (e.g., DW_PROA_open_Mean_ns)
- DW_{Chain}_{State}_Count (Units: count) (e.g., DW_PROB_closed_Count)
- DW_{Chain}_Closed_Fraction, DW_{Chain}_Open_Fraction (Units: %)
- DW_RefDist_Closed_Used, DW_RefDist_Open_Used (Units: Å)
- DW_StateDurationVsChain_Chi2_pvalue, DW_OpenDurationVsChain_Kruskal_pvalue (Units: p-value)

#### **Module: pocket_analysis**
- Pocket{A/B/C/D}_MeanOccupancy, _OccupancyStd (Units: count)
- Pocket{A/B/C/D}_MeanResidence_ns, _MedianResidence_ns, _MaxResidence_ns (Units: ns)
- Pocket{A/B/C/D}_ResidencePeriods (Units: count)
- Pocket{A/B/C/D}_RTSkewness (Units: )
- Pocket{A/B/C/D}_ShortLivedPct, _LongLivedPct (Units: %)
- PocketWater_OccupancyRatio, CV_of_Mean_Residence_Times, Gini_Coefficient_TotalTime, Entropy_TotalTime, Normalized_Range_Median_RTs (Units: )
- PocketWater_KS_{P1}_{P2}, Max_Pairwise_KS_Statistic (Units: KS stat)
