Markdown

# Molecular Dynamics Simulation Analysis Suite (v1.5.0)

**Version:** 1.4.0 (April 14, 2025)

## Overview

This suite provides a set of Python scripts for analyzing molecular dynamics (MD) simulations, primarily focused on toxin-channel complexes (e.g., Cs1 toxin with K+ channel tetramers). It extracts key metrics related to structural stability, pore dynamics, toxin orientation, interface contacts, ion permeation, and water behavior within the channel cavity.

The analysis includes specialized filtering routines designed to correct artifacts in distance measurements arising from Periodic Boundary Conditions (PBC), including standard unwrapping/smoothing and an advanced multi-level correction method based on Kernel Density Estimation (KDE).

The codebase has been refactored into logical modules for improved maintainability and extensibility.

## Features

* **K+ Channel Pore Dynamics:** Analyzes Glycine-Glycine (G-G) Cα distances between opposing subunits (A-C, B-D) as a proxy for pore diameter.
* **Toxin-Channel Stability:** Calculates the Center-of-Mass (COM) distance between the toxin and the channel complex over time.
* **Distance Filtering:** Implements robust filtering for G-G and COM distances to correct:
    * Standard PBC jumps (unwrapping).
    * Statistical spikes/outliers.
    * Multi-level artifacts using KDE-based detection and normalization (with quality control).
    * Automatic selection between standard and multi-level filtering based on data characteristics.
* **Toxin Orientation:** Calculates the angle between toxin/channel principal axes and the toxin's rotation relative to its starting frame (Euler angles).
* **Interface Contacts:**
    * Calculates the total number of atomic contacts (< 3.5 Å default) between toxin and channel over time.
    * Generates average residue-residue contact frequency maps (full and focused on top interactions).
* **K+ Ion Dynamics:**
    * Identifies selectivity filter residues (TVGYG motif).
    * Calculates binding site positions (S0-S4, Cavity) relative to the G1 Cα plane.
    * Tracks K+ ions near the filter region (using carbonyl oxygen distances for accuracy).
    * Analyzes site occupancy over time (heatmap, average bar chart) and calculates site statistics.
* **Cavity Water Dynamics:**
    * Tracks water molecules within the cavity region below the selectivity filter (S4).
    * Calculates water occupancy over time.
    * Calculates water residence times using an exit buffer condition (>5 frames outside cavity).
* **Reporting:**
    * Generates a comprehensive HTML report per run (when running all analyses) with embedded plots and statistics.
    * Generates a summary PowerPoint presentation aggregating key plots across multiple runs (optional).
    * Saves detailed analysis results to CSV and JSON files within each run directory.

## Dependencies

* Python 3.8+
* MDAnalysis (>= 2.0.0 recommended)
* NumPy
* Pandas
* SciPy
* Matplotlib
* Seaborn
* python-pptx (for PowerPoint generation)
* Jinja2 (for HTML report generation)
* tqdm (for progress bars)

## Installation

1.  **Clone the repository** (or download the script files).
2.  **Create a Python environment** (recommended):
    ```bash
    python -m venv md_analysis_env
    source md_analysis_env/bin/activate # Linux/macOS
    # md_analysis_env\Scripts\activate # Windows
    ```
3.  **Install dependencies:** Create a `requirements.txt` file with the packages listed above (pin versions as needed) and run:
    ```bash
    pip install -r requirements.txt
    ```
    Alternatively, install them individually:
    ```bash
    pip install MDAnalysis numpy pandas scipy matplotlib seaborn python-pptx Jinja2 tqdm
    ```

## Project Structure

The analysis suite is organized into modules:

* `main_analyzer.py`: Main script execution and orchestration.
* `Analysis/` : Directory containing the analysis library modules.
    * `__init__.py`: Makes `Analysis` a Python package.
    * `config.py`: Global configuration parameters (version, FRAMES_PER_NS, cutoffs).
    * `utils.py`: General utility functions (AA conversion, time conversion, JSON cleaning).
    * `logger_setup.py`: Functions for configuring logging.
    * `filtering.py`: Distance data filtering algorithms (PBC correction, smoothing).
    * `core_analysis.py`: Basic trajectory reading (raw distances), application of filtering, basic distance plots.
    * `orientation_contacts.py`: Toxin orientation, rotation, and interface contact analysis.
    * `ion_analysis.py`: K+ ion tracking, binding site calculation, occupancy analysis.
    * `water_analysis.py`: Cavity water occupancy and residence time analysis.
    * `summary.py`: Calculation and saving of the final `analysis_summary.json` file per run.
    * `reporting.py`: Generation of HTML and PowerPoint reports.

**Expected Input Data Structure:**

The script generally expects input data organized as follows (relative to the `--base_dir` used for batch processing):

<base_dir>/
├── System_Name_1/
│   ├── Run_1/
│   │   ├── step5_input.psf     # Topology file
│   │   ├── MD_Aligned.dcd      # Trajectory file (or similar name)
│   │   └── ... (other simulation files)
│   ├── Run_2/
│   │   └── ...
│   └── ...
├── System_Name_2/
│   ├── Run_1/
│   │   └── ...
│   └── ...


## Usage

Execute the main script (`main_analyzer.py`) from the command line in the directory *containing* `main_analyzer.py` and the `Analysis/` subdirectory.

**1. Batch Processing (Recommended for multiple runs):**

* Navigate to the directory containing `main_analyzer.py` and the `Analysis/` folder. Your simulation data should be in subdirectories relative to this location (or specify with `--base_dir`).
* Run all analyses and generate individual summaries (HTML report generated):
    ```bash
    python main_analyzer.py # Runs all analyses by default if no flags given
    ```
    or explicitly:
    ```bash
    python main_analyzer.py --all
    ```
* Run only specific analyses (e.g., COM distance and Ion tracking). HTML report will be **skipped**:
    ```bash
    python main_analyzer.py --COM --ions
    ```

**2. Single Folder Analysis:**

* Analyze a specific run directory (e.g., `/path/to/System_1/Run_1`). The script assumes `main_analyzer.py` is run from its own directory:
    ```bash
    python main_analyzer.py --folder /path/to/System_1/Run_1
    ```
* Analyze a specific run directory with specific analyses (HTML report skipped):
    ```bash
    python main_analyzer.py --folder /path/to/System_1/Run_1 --ions --water
    ```

**3. Generate PowerPoint Summary Only:**

* Collect results from existing `analysis_summary.json` files under the current directory (or specified `--base_dir`) and create a presentation:
    ```bash
    python main_analyzer.py --pptx [--base_dir /path/to/results/base]
    ```
    *(Requires analysis to have been run previously)*

**4. Other Options:**

* `--force_rerun`: Reprocess runs even if a summary file exists and matches the current script version.
* `--box_z <value>`: Provide an estimate for the simulation box Z-dimension (in Å) to aid COM distance filtering.
* `--log_level <LEVEL>`: Set logging verbosity (DEBUG, INFO, WARNING, ERROR). Default: INFO.
* `--base_dir <path>`: Specify the base directory containing `System/Run` folders if not running the script from there directly. Also affects where the main log and PPT are saved.

**5. Single Trajectory Mode (Discouraged):**

* Use specific topology and trajectory files. Requires `--topology`. Output goes to trajectory directory unless `--output` is specified.
    ```bash
    python main_analyzer.py --trajectory traj.dcd --topology topo.psf [--output ./analysis_out] [--all | analysis_flags]
    ```

## Output Files

Outputs are generated within each processed run directory (`<base_dir>/System/Run/` or the specific folder provided via `--folder` or `--output`).

**1. CSV Files:**

* `G_G_Distance_Raw.csv`: Raw G-G distances per frame.
* `G_G_Distance_Filtered.csv`: Raw and filtered G-G distances.
* `COM_Stability_Raw.csv`: Raw Toxin-Channel COM distances per frame.
* `COM_Stability_Filtered.csv`: Raw and filtered COM distances.
* `Toxin_Orientation.csv`: Toxin orientation angle, rotation components, and total atom contacts per analyzed frame.
* `Toxin_Channel_Residue_Contacts.csv`: Average contact frequency between each Toxin-Channel residue pair (Note: uses simplified labels, see Focused Map).
* `K_Ion_Z_Positions_Absolute.csv`: Absolute Z-positions of tracked K+ ions.
* `K_Ion_Z_Positions_G1Centric.csv`: Z-positions relative to G1 Cα plane.
* `K_Ion_Occupancy_Per_Frame.csv`: Number of ions in each binding site per frame.
* `K_Ion_Site_Statistics.csv`: Summary statistics (Mean Occ, Max Occ, % Time Occ) for each binding site.
* `Cavity_Water_Occupancy.csv`: Water count and indices in the cavity per frame.

**2. JSON Files:**

* `analysis_summary.json`: Overall summary statistics for the run (means, stds, flags, status). **Crucial for aggregation.**
* `Cavity_Water_ResidenceTimes.json`: List of calculated water residence times (ns) and buffer used.

**3. Text Files:**

* `binding_site_positions_g1_centric.txt`: Calculated Z-positions of binding sites relative to G1 Cα.
* `K_Ion_Coordinate_Reference.txt`: Information about the G1 Cα reference plane.
* `<RunName>_analysis.log`: Detailed log file specific to the analysis of this run.

**4. Visualizations (PNG):**

* G-G distance plots (Raw, Filtered, Comparison).
* COM distance plots (Raw, Filtered, Comparison, KDE Analysis).
* Toxin Orientation plots (Angle, Rotation Components, Atom Contacts).
* Residue Contact Maps (Full, Focused).
* K+ Ion plots (Binding Site Schematic, Combined Position/Density, Occupancy Heatmap, Average Occupancy).
* Cavity Water plots (Count vs Time, Residence Time Histogram).

**5. HTML Reports:**

* `<RunName>_analysis_report.html`: Comprehensive report with embedded plots and statistics. *(Generated only when running in 'all analyses' mode)*.

**6. PowerPoint Summary (Optional):**

* `MD_Analysis_Summary.pptx`: Generated in the directory where the script is run using the `--pptx` flag (or `--base_dir`). Aggregates key plots from multiple runs.

## Configuration

Key parameters can be adjusted in `Analysis/config.py`:

* `FRAMES_PER_NS`: Frames per nanosecond in your trajectory (affects time axis).
* `DEFAULT_CUTOFF`: Default distance (Å) for atom contacts.
* `EXIT_BUFFER_FRAMES`: Frames water must be outside cavity for residence time calculation.

## Logging

* A main log file (`md_analysis_main_*.log`) is created in the base directory (`--base_dir` or CWD).
* Each run directory gets a specific log file (`<RunName>_analysis.log`).
* Logging level can be controlled with `--log_level`.

## Advanced Usage

* **Distributed Computing:** The script (especially in single-folder mode via `--folder`) can be submitted as individual jobs on computing clusters (e.g., using SLURM). Each job processes one run folder.
* **Aggregation:** After batch processing, use the separate `aggregate_summaries.py` script (if provided/created) to parse all generated `analysis_summary.json` files and create a master CSV summary table across all runs.

## License

MIT

## Contact

izhar.karbat@weizmann.ac.il