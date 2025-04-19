# Molecular Dynamics Simulation Analysis Suite (v1.5.0)

**Version:** 1.5.0 (April 14, 2025)

## Overview

This suite provides a set of Python scripts for analyzing molecular dynamics (MD) simulations, primarily focused on toxin-channel complexes (e.g., Cs1 toxin with K+ channel tetramers). It processes a **single simulation run directory** at a time, extracting key metrics related to structural stability, pore dynamics, toxin orientation, interface contacts, ion permeation, and water behavior within the channel cavity.

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

1.  **Clone the repository** (or download the script files).
2.  **Create a Python environment** (recommended):
    ```bash
    python -m venv pore_analysis_env
    source pore_analysis_env/bin/activate # Linux/macOS
    # pore_analysis_env\\Scripts\\activate # Windows
    ```
3.  **Install dependencies:** Create a `requirements.txt` file with the packages listed above (pin versions as needed) and run:
    ```bash
    pip install -r requirements.txt
    ```
    Alternatively, install them individually:
    ```bash
    pip install MDAnalysis numpy pandas scipy matplotlib seaborn Jinja2 tqdm
    ```
    Or, if `setup.py` is present and configured:
    ```bash
    pip install .
    ```

## Project Structure

The analysis suite is organized into modules:

* `main_analyzer.py`: Main script execution and orchestration.
* `pore_analysis/` : Directory containing the analysis library modules.
    * `__init__.py`: Makes `pore_analysis` a Python package.
    * `config.py`: Global configuration parameters (version, FRAMES_PER_NS, cutoffs).
    * `core/`:
        * `utils.py`: General utility functions (AA conversion, time conversion, JSON cleaning).
        * `logging.py`: Functions for configuring logging (if `logger_setup.py` is moved here).
    * `modules/`:
        * `core_analysis/`: Basic trajectory reading (raw distances), filtering, basic distance plots.
        * `orientation_contacts/`: Toxin orientation, rotation, and interface contact analysis.
        * `ion_analysis/`: K+ ion tracking, binding site calculation, occupancy analysis.
        * `inner_vestibule_analysis/`: Cavity water occupancy and residence time analysis (renamed from `water_analysis`).
        * `gyration_analysis/`: Carbonyl gyration analysis.
    * `reporting/`:
        * `summary.py`: Calculation and saving of the final `analysis_summary.json` file per run.
        * `html.py`: Generation of the HTML report.
        * `templates/`: Jinja2 templates for the HTML report.
* `logger_setup.py`: Functions for configuring logging (can be moved into `core`).
* `config.py`: Global configuration parameters (can be moved into `pore_analysis/core`).

**Expected Input Data Structure:**

The script operates on a single run directory specified via the `--folder` argument. This directory should ideally contain:

<run_directory_path>/
├── step5_input.psf     # Topology file (or similar name)
├── MD_Aligned.dcd      # Trajectory file (or similar name)
└── ... (other simulation files)

The script attempts to infer a "System Name" from the parent directory of the run folder (e.g., if the path is `/path/to/SystemName/RunName`), otherwise, it uses the run folder name as the system name.

## Usage

Execute the main script (`main_analyzer.py`) from the command line. The primary argument is `--folder`, which specifies the path to the simulation run directory you want to analyze.

**1. Run All Analyses (Default):**

* This is the standard mode and generates the HTML report.
    ```bash
    python main_analyzer.py --folder /path/to/your/run_directory
    ```
    or explicitly:
    ```bash
    python main_analyzer.py --folder /path/to/your/run_directory --all
    ```

**2. Run Specific Analyses Only:**

* If you only want to run specific parts of the analysis (e.g., Ion and Water analysis), use the corresponding flags. Note that the HTML report will **not** be generated in this case, only the specific output files and the summary JSON.
    ```bash
    python main_analyzer.py --folder /path/to/your/run_directory --ions --water
    ```
* Available analysis flags:
    * `--GG`: G-G distance (pore diameter) analysis.
    * `--COM`: COM distance (toxin stability) analysis.
    * `--orientation`: Toxin orientation and contact analysis.
    * `--ions`: K+ ion tracking and coordination analysis.
    * `--water`: Cavity Water analysis.
    * `--gyration`: Carbonyl Gyration analysis.

**3. Other Options:**

* `--force_rerun`: Reprocess the run even if a summary file exists and matches the current script version. Useful if input data or code relevant to the analysis has changed.
* `--box_z <value>`: Provide an estimate for the simulation box Z-dimension (in Å) to aid COM distance filtering, particularly the multi-level KDE filter.
* `--log_level <LEVEL>`: Set logging verbosity (DEBUG, INFO, WARNING, ERROR). Default: INFO.

**4. Single Trajectory Mode (Discouraged):**

* This mode is less recommended as it bypasses the organized folder structure. It requires specific topology and trajectory files. Output goes to the trajectory's directory unless `--output` is specified.
    ```bash
    python main_analyzer.py --trajectory traj.dcd --topology topo.psf [--output ./analysis_out] [--all | analysis_flags]
    ```

## Output Files

Outputs are generated within the processed run directory specified via `--folder` (or `--output` in trajectory mode).

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
* `Gyration_States.csv`: Frame-by-frame state ('On'/'Off') for G1 and Y residues based on gyration.
* `Gyration_Summary.csv`: Summary statistics for gyration analysis (mean radius, flips, durations).

**2. JSON Files:**

* `analysis_summary.json`: Overall summary statistics for the run (means, stds, flags, status). **Crucial for later aggregation.**
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
* Carbonyl Gyration plots (Radius vs Time, State vs Time).

**5. HTML Reports:**

* `<RunName>_analysis_report.html`: Comprehensive report with embedded plots and statistics. *(Generated only when running with `--all` or no specific analysis flags)*.

## Configuration

Key parameters can be adjusted in `config.py` (or its new location, e.g., `pore_analysis/core/config.py`):

* `FRAMES_PER_NS`: Frames per nanosecond in your trajectory (affects time axis).
* `DEFAULT_CUTOFF`: Default distance (Å) for atom contacts.
* `EXIT_BUFFER_FRAMES`: Frames water must be outside cavity for residence time calculation.
* Filter parameters, Gyration thresholds, etc.

## Logging

* A main log file (`pore_analysis_main_*.log`) is created in the directory where the script is run.
* Each processed run directory gets a specific log file (`<RunName>_analysis.log`).
* Logging level can be controlled with `--log_level`.

## Advanced Usage / Workflow

* **Running Multiple Simulations:** Since the script now processes only one folder at a time, you can run multiple instances in parallel or sequentially using a simple shell script or a workflow manager.
    * **Example Shell Script:**
        ```bash
        #!/bin/bash
        ANALYSIS_SCRIPT="/path/to/main_analyzer.py"
        BASE_DATA_DIR="/path/to/all/systems"

        # Find all Run directories (e.g., System*/Run*)
        find "$BASE_DATA_DIR" -mindepth 2 -maxdepth 2 -type d -print0 | while IFS= read -r -d $'\0' run_folder; do
            echo "Processing: $run_folder"
            python "$ANALYSIS_SCRIPT" --folder "$run_folder" --all
            echo "----------------------------------------"
        done
        echo "All processing finished."
        ```
* **Aggregation:** After processing multiple run directories, use the separate `Aggregate_Summaries.py` script (if provided/created) to parse all generated `analysis_summary.json` files and create a master CSV summary table across all runs.

## License

MIT

## Contact

izhar.karbat@weizmann.ac.il