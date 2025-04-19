# MD Analysis Suite - Code Structure & Development Guidelines

**Version:** (Reflects state as of YYYY-MM-DD)

## 1. Purpose & Philosophy

This document outlines the structure, core logic, conventions, and development practices for the MD Analysis Suite. Its primary goal is to ensure code contributions (from humans or AI assistants) maintain the integrity, consistency, and intended workflow of the suite, preventing integration errors and promoting maintainability.

**Core Philosophy:**

*   **Modularity:** Code is organized into distinct modules with specific responsibilities (e.g., ion analysis, orientation analysis, reporting).
*   **Single Run Focus:** The primary operational mode processes a single simulation run directory (`--folder`). Batch processing capabilities have been removed.
*   **Clear Data Flow:** Analysis results are managed within a defined workflow, primarily using a central dictionary within the main analysis function.
*   **Standardized Outputs:** Output files (CSV, JSON, PNG, logs) follow consistent naming and location conventions within the processed run directory.

## 2. Project Structure Overview

```
.
├── setup.py                  # Package installation and dependencies
├── config.py                 # Global configuration (consider moving to md_analysis/core)
├── README.md                 # Project overview and usage
├── Code_Structure.md         # This file
├── Aggregate_Summaries.py    # Utility for combining results from multiple runs
├── md_analysis/              # Main Python package source code
│   ├── __init__.py
│   ├── main.py               # Main execution script (entry point for python -m)
│   ├── core/                 # Core utilities, config, logging
│   │   ├── __init__.py
│   │   ├── config.py         # (If moved here)
│   │   ├── logging.py        # Logger setup function (setup_analysis_logger)
│   │   └── utils.py
│   ├── modules/              # Specific analysis modules
│   │   ├── __init__.py
│   │   ├── core_analysis/
│   │   ├── orientation_contacts/
│   │   ├── ion_analysis/
│   │   ├── inner_vestibule_analysis/
│   │   ├── gyration_analysis/
│   │   └── tyrosine_analysis/
│   └── reporting/            # Report generation and summary logic
│       ├── __init__.py
│       ├── summary.py
│       ├── html.py
│       └── templates/
└── ... (other files like .gitignore, LICENSE)
# NOTE: main_analyzer.py and logger_setup.py in root are obsolete and should be deleted.
```

## 3. Execution Flow: `md_analysis/main.py`

The primary way to run the analysis is via `python -m md_analysis.main --folder <path> ...`.

1.  **Argument Parsing:** Uses `argparse` in `main()` to handle command-line arguments.
    *   `--folder` (Required): Specifies the path to the single run directory to process.
    *   Analysis Flags (`--GG`, `--COM`, `--ions`, etc.): Select specific analyses. If none are provided, `--all` is assumed.
    *   Other flags (`--force_rerun`, `--box_z`, `--log_level`).
2.  **Determine Analyses:** Sets boolean flags (`run_gg`, `run_com`, etc.) based on user arguments or the default `--all`. Determines if the final HTML report should be generated (`generate_html`).
3.  **Handle Single Trajectory Mode:** (Discouraged) If `--trajectory` is used, determines paths, sets up the logger using `setup_analysis_logger` from `md_analysis.core.logging` (directing output to the `--output` dir), logs initial info, calls `_run_analysis_workflow`, and exits.
4.  **Single Folder Mode Execution:** (Primary Path)
    *   Validates the `--folder` path.
    *   Determines `system_name` and `run_name`.
    *   **Sets up Logger:** Calls `setup_analysis_logger` from `md_analysis.core.logging`, passing the `folder_path`, `run_name`, and log level. This configures the root logger.
    *   **Logs Initial Info:** Writes startup messages, command, analysis plan etc. to console and the run-specific log file.
    *   Checks for required input files (`MD_Aligned.dcd`, `step5_input.psf`).
    *   Calls the internal `_run_analysis_workflow` function, passing the run details and analysis flags.
5.  **Workflow Execution (`_run_analysis_workflow`)**:
    *   **Check Skip Logic:** If `--force_rerun` is not set, checks for an existing `analysis_summary.json` with a matching version and "Success" status to potentially skip the run.
    *   **Initialize `results` Dictionary:** Creates an empty dictionary to hold data passed between analysis steps within this function.
    *   **Execute Analyses Sequentially:** Calls specific analysis functions from the `modules/` directory based on the boolean flags (`run_gg`, `run_com`, etc.).
        *   Each analysis function typically takes `run_dir`, `psf_file`, `dcd_file`, and potentially data from previous steps (passed explicitly or retrieved from the `results` dictionary) as input.
        *   Results from analysis functions (e.g., data arrays, statistics dictionaries) are stored back into the `results` dictionary (e.g., `results['ion_indices'] = ion_indices`).
        *   Analysis functions may also have side effects like saving plots or intermediate CSV files within designated subdirectories (e.g., `ion_analysis/`).
    *   **Save Summary:** Calls `calculate_and_save_run_summary` from `reporting/summary.py`, passing necessary data accumulated in the `results` dictionary. This creates `analysis_summary.json`.
    *   **Generate HTML Report:** If `generate_html` is true, calls `generate_html_report` from `reporting/html.py`, passing the `run_dir` and the just-loaded `run_summary`.
    *   Uses the root logger configured in the previous step for all its logging.

## 4. Key Data Structures: The `results` Dictionary

*   **Purpose:** Acts as a central, in-memory carrier for data passed between different analysis steps *within* the `_run_analysis_workflow` function in `main_analyzer.py`.
*   **Usage:**
    *   Initialized as an empty dictionary at the start of the workflow.
    *   Populated by the return values of core analysis functions (e.g., `analyze_trajectory`, `track_potassium_ions`).
    *   Data from it is read by subsequent analysis steps or by the final `calculate_and_save_run_summary` function.
*   **Scope:** Primarily intended for use *inside* `_run_analysis_workflow`. Analysis functions within modules should generally *return* their results rather than directly modifying this dictionary (though they read from it via arguments).
*   **Example Keys:** `dist_ac`, `com_distances`, `time_points`, `ion_indices`, `filter_sites`, `g1_reference`, `cavity_water_stats`, `gyration_stats`, `is_control_system`.

## 5. Output Conventions

*   **Location:** All primary outputs (CSVs, JSONs, PNGs, run-specific logs) are saved *within* the processed `--folder` directory or its subdirectories.
    *   Specific modules often create their own subdirectories (e.g., `ion_analysis/`, `orientation_contacts/`, `gyration_analysis/`). **Convention:** New analysis modules should save their outputs within a dedicated subdirectory named after the analysis type inside the main run folder.
*   **Naming:**
    *   Files often incorporate the analysis type (e.g., `K_Ion_Site_Statistics.csv`, `COM_Stability_Filtered.csv`).
    *   The run-specific log is named `<RunName>_analysis.log`.
    *   The HTML report is named `<RunName>_analysis_report.html`.
*   **File Types:**
    *   **CSV:** Detailed frame-by-frame data or tabular statistics.
    *   **JSON:** Structured summary data (`analysis_summary.json`), sometimes lists (`Cavity_Water_ResidenceTimes.json`).
    *   **PNG:** Plots and visualizations.
    *   **TXT:** Human-readable info like binding site positions.
    *   **LOG:** Detailed execution logs.
    *   **HTML:** Final user-facing report.

## 6. The `analysis_summary.json` File

*   **Purpose:** Provides a single, machine-readable summary of the key quantitative results and status for a single analysis run. It is the **primary input** for any downstream aggregation or cross-run comparison (like `Aggregate_Summaries.py`).
*   **Generation:** Created at the end of `_run_analysis_workflow` by `calculate_and_save_run_summary` (in `reporting/summary.py`).
*   **Content:** Contains key calculated statistics (means, std devs, counts), configuration parameters used, analysis status (Success, Failed, Success_With_Issues), script version, and timestamps.
*   **Update Policy:** This file should only be updated by `calculate_and_save_run_summary`. Individual analysis modules should return their data to `_run_analysis_workflow` to be included, rather than modifying this file directly. Adding new summary statistics requires updating `calculate_and_save_run_summary`.

## 7. HTML Report

*   **Purpose:** Provides a user-friendly, visual summary of the analysis results for a single run.
*   **Generation:** Created by `generate_html_report` (in `reporting/html.py`) using Jinja2 templates located in `reporting/templates/`.
*   **Content:** Uses data primarily loaded from `analysis_summary.json` and embeds PNG plots (often read from the run directory and encoded in base64).
*   **Structure:** Defined by the Jinja2 templates (e.g., `base_template.html`, `_tab_overview.html`, `_tab_pore_ions.html`, etc.). Tabs organize different analysis sections.
*   **Update Policy:**
    *   To add new plots or tables, modify the relevant template file (e.g., `_tab_new_analysis.html`).
    *   Ensure the necessary data keys are present in `run_summary` (populated by `calculate_and_save_run_summary`).
    *   Ensure the corresponding plot PNG files are generated by the analysis module in the expected location (or modify `generate_html_report` to handle data differently if needed).
    *   Add the new tab include statement to `base_template.html`.
*   **Generation Condition:** Only generated if `generate_html` is `True` in `main_analyzer.py`, which usually requires all analysis flags (`run_gg`, `run_com`, etc.) to be true (i.e., `--all` mode or no specific flags selected).

## 8. Configuration (`config.py`)

*   Contains global constants and parameters used across different modules.
*   **Policy:** Define parameters here if they are likely to be used in multiple places or might need user adjustment later. Avoid hardcoding values.
*   **Location:** Currently in root directory. Consider moving to `md_analysis/core/` for better package structure.

## 9. Logging (`md_analysis/core/logging.py`)

*   Provides the `setup_analysis_logger` function.
*   **Behavior:** This function configures the **root logger** when called by `main()`. It removes any previous handlers and sets up two new ones:
    1.  A `StreamHandler` writing formatted messages to the console (stdout).
    2.  A `FileHandler` writing formatted messages to a single log file named `<RunName>_analysis.log` located directly **inside the run directory** (`--folder` path or `--output` path in trajectory mode).
*   **Consolidation:** There is no separate main log file created in the script's execution directory. All logs related to processing a specific run are contained within that run's folder.
*   **Policy:** Use standard Python `logging` within modules (`logger = logging.getLogger(__name__)`). The `setup_analysis_logger` function configures the root logger, and module loggers will inherit this configuration automatically. Use appropriate levels (DEBUG, INFO, WARNING, ERROR).

## 10. Development Guidelines & Best Practices

*   **Modularity & Single Responsibility:** Keep functions focused.
*   **Docstrings:** Document purpose, Args, Returns, and **Side Effects** (like file saving).
*   **Type Hinting:** Use type hints for clarity and static analysis.
*   **Testing:**
    *   **Unit Tests:** Add tests for new or modified analysis functions to verify correctness in isolation.
    *   **Integration Tests:** Crucial for this workflow. Test the `_run_analysis_workflow` or even `main` with sample data. Verify that:
        *   The expected analysis steps run based on flags.
        *   The `results` dictionary is populated correctly.
        *   `analysis_summary.json` contains the expected keys and plausible values.
        *   Output files (CSVs, plots) are created in the correct locations with expected content (even if just checking existence and basic format).
*   **Version Control (Git):** Use branches, commit frequently with clear messages.
*   **Working with AI Assistants (CRITICAL for Integration):**
    *   **Be Explicit:** Clearly state the goal, the specific function to modify/create, its inputs (names, types), outputs (return value, files saved), and *exactly where it fits in the workflow*.
    *   **Provide Context:** Show the AI the calling function (e.g., `_run_analysis_workflow`) and the function being called (if modifying an existing one).
    *   **Specify Integration:** Do not just ask the AI to "add feature X". Instruct it *how* to integrate it.
        *   **Example:** "Modify the `_run_analysis_workflow` function in `main_analyzer.py`. After the section calling `analyze_ion_coordination` (around line XXX), add a call to the new function `analyze_ion_conduction` (which you will define in `md_analysis/modules/ion_analysis/ion_conduction.py`). This new function requires `run_dir`, `time_points_ions`, `ions_z_abs`, `ion_indices`, `filter_sites`, and `g1_reference` as input. These variables should be available from the `results` dictionary or previous steps. Store the dictionary returned by `analyze_ion_conduction` into `results['conduction_stats']`. Ensure the `run_conduction` flag controls whether this call happens."
    *   **Review Carefully:** Always review AI-generated code for correctness, adherence to conventions, and proper integration *before* running it. Check function calls, argument passing, and file paths.
    *   **Incremental Steps:** Ask for the core function first, review it, then ask for the integration code separately.

## 11. Maintaining This Document

Keep this document updated as the codebase evolves. Significant changes to workflow, structure, or conventions should be reflected here. 