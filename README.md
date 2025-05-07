# Pore Analysis Suite (v2.0.0)

**Version:** 2.0.0

## Overview

This suite provides a set of Python scripts for analyzing molecular dynamics (MD) simulations, primarily focused on toxin-channel complexes (e.g., Cs1 toxin with K+ channel tetramers). It processes a **single simulation run directory** at a time, extracting key metrics related to structural stability, pore dynamics, toxin orientation, interface contacts, ion permeation, and water behavior within the channel cavity.

The analysis includes specialized filtering routines designed to correct artifacts in distance measurements arising from Periodic Boundary Conditions (PBC), including standard unwrapping/smoothing and an advanced multi-level correction method based on Kernel Density Estimation (KDE).

**This version (v2.0.0) features significant refactoring** including a database-centric approach for improved modularity/traceability, centralized trajectory loading for efficiency, and the ability to analyze specific trajectory slices.

## Features (v2.0.0)

* **Database-Centric:** All analysis results (files, metrics, parameters, status) are stored in a per-run SQLite database (`analysis_registry.db`). See `database.md` for schema details.
* **Modular Design:** Analysis tasks are separated into distinct modules (e.g., `core_analysis`, `ion_analysis`) with clear separation between computation and visualization steps.
* **Efficient Trajectory Handling:** MDAnalysis Universe object is loaded once per run, improving performance.
* **Trajectory Slicing:** Analyze specific portions of trajectories using `--start_frame` and `--end_frame` options.
* **K+ Channel Pore Dynamics:** Analyzes Glycine-Glycine (G-G) Cα distances between opposing subunits (A-C, B-D).
* **Toxin-Channel Stability:** Calculates the Center-of-Mass (COM) distance between the toxin and the channel complex.
* **Distance Filtering:** Implements robust filtering for G-G and COM distances (Standard PBC, Spike Correction, optional Multi-level KDE-based correction). Automatic filter selection based on data characteristics.
* **Toxin Orientation & Contacts:** Calculates toxin orientation angle, rotation, atom contacts, and residue contact frequencies.
* **K+ Ion Dynamics (HMM-based):**
    * Identifies selectivity filter residues (TVGYG motif).
    * Calculates and optimizes binding site positions (S0-S4, Cavity) relative to the G1 Cα plane using ion density.
    * Tracks K+ ions near the filter region.
    * Analyzes site occupancy over time (heatmap, average bar chart).
    * Analyzes site-to-site ion transitions and full conduction events using a **Hidden Markov Model (HMM)** for improved state assignment and transition detection.
* **Cavity Water Dynamics:** Tracks water molecules within the cavity, calculates occupancy, and residence times.
* **SF Carbonyl Gyration:** Analyzes the gyration radius of SF carbonyl groups to detect conformational flips.
* **SF Tyrosine Rotamers (HMM-based):** Analyzes the rotameric states (Chi1/Chi2) of key tyrosine residues in the selectivity filter using HMM.
* **DW Gate Dynamics:** Analyzes the open/closed state dynamics of the Asp-Trp gate near the extracellular entrance.
* **Peripheral Pocket Water Analysis (Optional, ML-based):** Uses a pre-trained Equivariant Transformer model to classify water molecules in peripheral pockets, analyzing occupancy, residence times, and imbalance metrics (requires `torch`, `torchmd-net`, and CUDA setup - see Installation).
* **Reporting:**
    * Generates a comprehensive HTML report per run using data queried from the database.
    * Saves detailed analysis results (data, metrics) to the database and associated files (CSVs, JSON, PNGs) within module-specific subdirectories in the run folder.
    * Creates a summary JSON file (`analysis_summary.json`).

## Dependencies

* Python 3.9+
* MDAnalysis (>= 2.0.0 recommended)
* NumPy
* Pandas
* SciPy
* scikit-learn
* Matplotlib
* Seaborn
* Jinja2 (for HTML report generation)
* tqdm (for progress bars)

**Optional Dependencies (for Pocket Analysis):**

* PyTorch (`torch`)
* PyTorch Geometric (`torch_geometric`)
* TorchMD-Net (`torchmd-net`)
* CUDA-enabled GPU and corresponding CUDA Toolkit (recommended for performance)

*(See `setup.py` for specific versions and development dependencies like `pytest`, `Pillow`)*

## Installation

1.  **Clone the repository** (or download the source code).
2.  **Create a Python environment** (recommended):
    ```bash
    # Using conda/mamba (recommended for managing PyTorch/CUDA)
    mamba create -n poreanalysis python=3.9 -c conda-forge -y
    mamba activate poreanalysis

    # Or using venv
    # python -m venv pore_env
    # source pore_env/bin/activate # Linux/macOS
    # # pore_env\Scripts\activate # Windows
    ```
3.  **Install the package:** Navigate to the directory containing `setup.py`.
    * **Minimal Installation (without Pocket Analysis):**
        ```bash
        pip install -e .
        ```
    * **Full Installation (including Pocket Analysis):** This requires PyTorch, TorchMD-Net, and potentially a specific CUDA setup first (see `INSTALL_ADVANCED.md` for details). Once prerequisites are met, run:
        ```bash
        pip install -e .[pocket]
        # Or potentially: pip install -e .[all]
        ```

## Basic Usage

The analysis suite is run from the command line, targeting a specific simulation run directory containing your trajectory files (e.g., `step5_input.psf`, `MD_Aligned.dcd`).

```bash
# Example run directory structure:
# /path/to/your/simulations/
# ├── system1/
# │   ├── run1/
# │   │   ├── step5_input.psf
# │   │   ├── MD_Aligned.dcd
# │   │   └── (other simulation files...)
# │   └── run2/
# │       └── ...
# └── system2/
#     └── ...

# --- Run Analysis ---

# Analyze the FULL trajectory for run1 in system1
python -m pore_analysis.main --folder /path/to/your/simulations/system1/run1

# Analyze only frames 1000 to 4999 (exclusive end) for run1
python -m pore_analysis.main --folder /path/to/your/simulations/system1/run1 --start_frame 1000 --end_frame 5000

# Analyze only the first 500 frames (0-499)
python -m pore_analysis.main --folder /path/to/your/simulations/system1/run1 --end_frame 500

# Analyze only the ion and water modules for frames 500 onwards
python -m pore_analysis.main --folder /path/to/your/simulations/system1/run1 --start_frame 500 --ions --water

# Force re-analysis of the DW Gate module for the full trajectory
python -m pore_analysis.main --folder /path/to/your/simulations/system1/run1 --dwgates --force-rerun

# --- Generate Reports ---

# Generate reports from existing analysis data in the database
python -m pore_analysis.main --folder /path/to/your/simulations/system1/run1 --summary-only --report --no-pdf
```

Analysis results (data files, plots) are saved in subdirectories within the specified run folder (e.g., core_analysis/, ion_analysis/).
A database file analysis_registry.db is created/updated in the run folder to track progress and results.
An HTML report (data_analysis_report.html) and a summary (analysis_summary.json) are generated by default unless suppressed.
See Contributing.md for developer guidelines and database.md for database schema details.

## Project Structure

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
