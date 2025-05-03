# Pore Analysis Suite (v2.0.0)

**Version:** 2.0.0

## Overview

This suite provides a set of Python scripts for analyzing molecular dynamics (MD) simulations, primarily focused on toxin-channel complexes (e.g., Cs1 toxin with K+ channel tetramers). It processes a **single simulation run directory** at a time, extracting key metrics related to structural stability, pore dynamics, toxin orientation, interface contacts, ion permeation, and water behavior within the channel cavity.

The analysis includes specialized filtering routines designed to correct artifacts in distance measurements arising from Periodic Boundary Conditions (PBC), including standard unwrapping/smoothing and an advanced multi-level correction method based on Kernel Density Estimation (KDE).

**This version (v2.0.0) features a significant refactoring** to a database-centric approach for improved modularity, traceability, and cross-simulation analysis capabilities.

## Features (v2.0.0)

* **Database-Centric:** All analysis results (files, metrics, parameters, status) are stored in a per-run SQLite database (`analysis_registry.db`). See `database.md` for schema details.
* **Modular Design:** Analysis tasks are separated into distinct modules (e.g., `core_analysis`, `ion_analysis`) with clear separation between computation and visualization steps.
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

## Project Structure