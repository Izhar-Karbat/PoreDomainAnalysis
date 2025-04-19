# Pore Analysis Suite (v1.6.1)

**Version:** 1.6.1 # Update Date Here

## Overview

This suite provides a set of Python scripts for analyzing molecular dynamics (MD) simulations, primarily focused on toxin-channel complexes (e.g., Cs1 toxin with K+ channel tetramers). It processes a **single simulation run directory** at a time, extracting key metrics related to structural stability, pore dynamics, toxin orientation, interface contacts, ion permeation, and water behavior within the channel cavity.

The analysis includes specialized filtering routines designed to correct artifacts in distance measurements arising from Periodic Boundary Conditions (PBC), including standard unwrapping/smoothing and an advanced multi-level correction method based on Kernel Density Estimation (KDE).

The codebase has been refactored into logical modules for improved maintainability and extensibility, packaged as `pore_analysis`.

## Features

* K+ Channel Pore Dynamics: Analyzes Glycine-Glycine (G-G) Cα distances between opposing subunits (A-C, B-D) as a proxy for pore diameter.
* Toxin-Channel Stability: Calculates the Center-of-Mass (COM) distance between the toxin and the channel complex over time.
* Distance Filtering: Implements robust filtering for G-G and COM distances to correct:
    * Standard PBC jumps (unwrapping).
    * Statistical spikes/outliers.
    * Multi-level artifacts using KDE-based detection and normalization (with quality control).
    * Automatic selection between standard and multi-level filtering based on data characteristics.
* Toxin Orientation: Calculates the angle between toxin/channel principal axes and the toxin's rotation relative to its starting frame (Euler angles).
* Interface Contacts:
    * Calculates the total number of atomic contacts (< 3.5 Å default) between toxin and channel over time.
    * Generates average residue-residue contact frequency maps (full and focused on top interactions).
* K+ Ion Dynamics:
    * Identifies selectivity filter residues (TVGYG motif).
    * Calculates binding site positions (S0-S4, Cavity) relative to the G1 Cα plane.
    * Tracks K+ ions near the filter region (using carbonyl oxygen distances for accuracy).
    * Analyzes site occupancy over time (heatmap, average bar chart) and calculates site statistics.
    * **Analyzes site-to-site ion transitions and full conduction events.**
* Cavity Water Dynamics:
    * Tracks water molecules within the cavity region below the selectivity filter (S4).
    * Calculates water occupancy over time.
    * Calculates water residence times using an exit buffer condition (>5 frames outside cavity).
* **SF Carbonyl Gyration:** Analyzes the gyration radius of SF carbonyl groups to detect conformational flips.
* **SF Tyrosine Rotamers:** Analyzes the rotameric states of key tyrosine residues in the selectivity filter.
* Reporting:
    * Generates a comprehensive HTML report per run (configurable).
    * Saves detailed analysis results to CSV and JSON files within the processed run directory.

## Dependencies

* Python 3.8+
* MDAnalysis (>= 2.0.0 recommended)
* NumPy
* Pandas
* SciPy
* Matplotlib
* Seaborn
* Jinja2 (for HTML report generation)
* tqdm (for progress bars)

## Installation

1.  **Clone the repository** (or download the source code).
2.  **Create a Python environment** (recommended):
    ```bash
    python -m venv pore_analysis_env
    source pore_analysis_env/bin/activate # Linux/macOS
    # pore_analysis_env\Scripts\activate # Windows
    ```
3.  **Install the package:** Navigate to the directory containing `setup.py` and run:
    ```bash
    pip install .
    ```
    *(This installs the `pore_analysis` package and its dependencies as listed in `setup.py`)*

## Project Structure

The analysis suite is organized as follows (see `Code_Structure.md` for more detail):

```
.
├── setup.py                  # Package installation and dependencies
├── config.py                 # Global configuration (can be overridden)
├── README.md                 # This file
├── Code_Structure.md         # Detailed structure and guidelines
├── Aggregate_Summaries.py    # Utility for combining results from multiple runs
├── pore_analysis/              # Main Python package source code
│   ├── __init__.py
│   ├── main.py               # Main execution script (entry point)
│   ├── core/                 # Core utilities, config loading, logging
│   │   ├── __init__.py
│   │   ├── logging.py
│   │   └── utils.py
│   ├── modules/              # Specific analysis modules
│   │   ├── __init__.py
│   │   ├── core_analysis/
│   │   ├── orientation_contacts/
│   │   ├── ion_analysis/         # Includes tracking, coordination, conduction
│   │   ├── inner_vestibule_analysis/
│   │   ├── gyration_analysis/
│   │   └── tyrosine_analysis/
│   └── reporting/            # Report generation and summary logic
│       ├── __init__.py
│       ├── summary.py
│       ├── html.py
│       └── templates/
└── ... (other files like .gitignore, LICENSE)
```

**Expected Input Data Structure:**

The script operates on a single run directory specified via the `--folder` argument. This directory should ideally contain:

`<run_directory_path>/`
├── `step5_input.psf`     # Topology file (or similar name)
├── `MD_Aligned.dcd`      # Trajectory file (or similar name)
└── ... (other simulation files)

The script attempts to infer a "System Name" from the parent directory of the run folder (e.g., if the path is `/path/to/SystemName/RunName`), otherwise, it uses the run folder name as the system name.

## Usage (`python -m pore_analysis.main`)

Execute the main analysis module from the command line. The primary argument is `--folder`, which specifies the path to the simulation run directory you want to analyze.

The script operates in two primary modes based on the flags provided:

### 1. Full Analysis (Default Workflow)

This mode runs all available analysis modules and is the recommended way for a standard analysis.

*   **Command:**
    ```bash
    python -m pore_analysis.main --folder /path/to/your/run_directory
    ```
*   **Action:** Runs all analysis modules (G-G Distance, COM Distance, Orientation/Contacts, Ion Tracking, Ion Coordination, Ion Conduction, Cavity Water, Gyration, Tyrosine). It checks for an existing, valid `analysis_summary.json` from a previous successful run with the same script version; if found, the run is skipped to save time. An HTML report (`report.html`) summarizing all results is generated by default.
*   **Suppressing the Report:** If you want to run all analyses but *not* generate the HTML report:
    ```bash
    python -m pore_analysis.main --folder /path/to/run --no-report
    ```
*   **Forcing a Rerun:** If you need to re-process the run completely, ignoring any existing summary file and overwriting all previous results:
    ```bash
    python -m pore_analysis.main --folder /path/to/run --force_rerun
    ```
    This also generates the HTML report by default. To force a rerun *without* the report, combine flags:
    ```bash
    python -m pore_analysis.main --folder /path/to/run --force_rerun --no-report
    ```

### 2. Selective Analysis (Targeted Workflow)

This mode allows you to run only specific analysis modules. This is useful for debugging, development, or re-running only a part of the analysis.

*   **Command:**
    ```bash
    # Example: Run only Ion Tracking, Coordination, and Conduction
    python -m pore_analysis.main --folder /path/to/run --ions --conduction
    ```
    *(Note: `--ions` implicitly covers tracking and coordination needed by conduction)*
*   **Action:** Executes *only* the analysis modules corresponding to the specified flags (`--ions`, `--conduction` in the example). These modules are **always executed** when flagged, overwriting any previous output files generated by them. The check for an existing `analysis_summary.json` is bypassed for these modules.
*   **Reporting:** An HTML report is **NOT** generated by default in this mode.
*   **Generating a Report:** If you want an HTML report containing the results from *only* the selected analyses run:
    ```bash
    python -m pore_analysis.main --folder /path/to/run --ions --conduction --report
    ```

### Available Analysis Flags for Selective Mode:

*   `--GG`: G-G distance analysis.
*   `--COM`: COM distance analysis.
*   `--orientation`: Toxin orientation and contact analysis.
*   `--ions`: K+ ion tracking and coordination analysis.
*   `--water`: Cavity Water analysis.
*   `--gyration`: Carbonyl Gyration analysis.
*   `--tyrosine`: SF Tyrosine rotamer analysis.
*   `--conduction`: Ion Conduction/Transition analysis (requires ion tracking data).

### Other Options:

*   `--box_z <value>`: Provide estimated box Z-dimension (Angstroms) for multi-level COM filter.
*   `--log_level <LEVEL>`: Set logging detail (DEBUG, INFO, WARNING, ERROR). Default is INFO.
*   `--report`: Generate HTML report when running selective analyses.
*   `--no-report`: Suppress HTML report generation when running full analysis (default or --force_rerun).

## Output Files

Outputs are generated within the processed run directory specified via `--folder` (or `--output` in trajectory mode).

**1. Common Files:**
*   `analysis_summary.json`: Machine-readable JSON summary of key results and run status. **Crucial for later aggregation.**
*   `report.html`: User-friendly HTML report (if generated based on flags).
*   `<RunName>_analysis.log`: Detailed log file specific to the analysis of this run.

**2. Module-Specific Outputs (Examples):**
*   **`core_analysis/`**: CSV files for raw and filtered distances (`G_G_Distance_*.csv`, `COM_Stability_*.csv`), plots (PNG).
*   **`orientation_contacts/`**: CSV files (`Toxin_Orientation.csv`, `Toxin_Channel_Residue_Contacts.csv`), plots (PNG, contact maps).
*   **`ion_analysis/`**: CSV files (`K_Ion_*.csv` for positions, occupancy, stats, transitions, conduction), text files (`binding_site_positions*.txt`), plots (PNG, position/density, heatmap, overlayed transitions).
*   **`inner_vestibule_analysis/`**: CSV/JSON files (`Cavity_Water_*.csv/json`), plots (PNG).
*   **`gyration_analysis/`**: CSV files (`Gyration_States.csv`, `Gyration_Summary.csv`), plots (PNG).
*   **`tyrosine_analysis/`**: CSV file (`Tyrosine_Rotamer_States.csv`), plots (PNG).

*(Refer to individual module code/docstrings for precise output filenames)*

## Configuration

Key parameters can be adjusted in `config.py` located in the project root directory:

* `FRAMES_PER_NS`: Frames per nanosecond in your trajectory (affects time axis).
* `DEFAULT_CUTOFF`: Default distance (Å) for atom contacts.
* `EXIT_BUFFER_FRAMES`: Frames water must be outside cavity for residence time calculation.
* Filter parameters, Gyration thresholds, Ion transition tolerance, etc.

## Logging

* All log messages related to the analysis of a specific run are consolidated into a single file named `<RunName>_analysis.log` located directly **inside the run directory** specified by `--folder`.
* There is no separate main log file created in the script's execution directory.
* Logging level to the console and file can be controlled with `--log_level`.

## Advanced Usage / Workflow

* **Running Multiple Simulations:** Since the script processes only one folder at a time, use a simple shell script or a workflow manager to iterate over multiple run directories.
    * **Example Shell Script:**
        ```bash
        #!/bin/bash
        # Define the base directory containing system folders
        BASE_DATA_DIR="/path/to/all/systems"

        # Find all Run directories (e.g., System*/Run*) and process them
        find "$BASE_DATA_DIR" -mindepth 2 -maxdepth 2 -type d -print0 | while IFS= read -r -d $'\0' run_folder; do
            echo "Processing: $run_folder"
            python -m pore_analysis.main --folder "$run_folder" # Run full analysis + report
            # Or use flags for selective analysis: python -m pore_analysis.main --folder "$run_folder" --ions --conduction
            echo "----------------------------------------"
        done
        echo "All processing finished."
        ```
* **Aggregation:** After processing multiple run directories, use the separate `Aggregate_Summaries.py` script to parse all generated `analysis_summary.json` files and create a master CSV summary table across all runs.

    ```bash
    python Aggregate_Summaries.py --input_dirs /path/to/system1 /path/to/system2 ... --output aggregated_results.csv
    # Or using wildcards (shell dependent)
    python Aggregate_Summaries.py --input_dirs /path/to/simulations/*/run_* --output aggregated_results.csv
    ```

## Development

Please refer to `Code_Structure.md` for details on the code organization, development guidelines, and integration procedures, especially when working with AI assistants.

## License

MIT

## Contact

izhar.karbat@weizmann.ac.il