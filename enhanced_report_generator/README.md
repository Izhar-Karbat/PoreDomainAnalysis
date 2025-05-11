# Enhanced Report Generator

A tool for generating comprehensive comparative reports from MD simulations analysis data.

## Overview

The Enhanced Report Generator is designed to produce detailed comparative reports between toxin-bound and toxin-free (control) channel simulations. It works by:

1. Aggregating metrics from individual simulation analyses into a central database
2. Performing statistical comparisons between toxin and control groups
3. Generating HTML reports with visualizations and AI-powered interpretations

The system follows a two-stage approach:
- **Stage 1**: Data aggregation and generation of a minimal "System Overview" report
- **Stage 2**: Human-guided enhancement to produce a detailed comparative report

## Installation

```bash
# Install from local directory
pip install -e .
```

## Usage

### Basic usage

```bash
# Stage 1: Data aggregation and system overview
generate-report --sim_roots /path/to/toxin_systems /path/to/control_systems \
                --output_dir ./my_report \
                --title "My Analysis" \
                --stage overview

# Stage 2: Detailed report (after running Stage 1)
generate-report --output_dir ./my_report \
                --title "My Analysis" \
                --stage detailed
```

### Command-line options

```
--output_dir: Directory to save the generated report and plots
--aggregated_db_name: Name of the SQLite database file within the output directory
--title: Base title for the generated report
--stage: Pipeline stage (aggregate, overview, or detailed)
--sim_roots: Root directories containing simulation runs (each with analysis_registry.db)
--force_aggregate: Force re-aggregation of data
--log_level: Set the logging level (DEBUG, INFO, WARNING, ERROR)
```

## Input Data Structure

The tool expects individual simulation run folders, each containing an `analysis_registry.db` file with:

1. A `simulation_metadata` table with:
   - `run_name`: Unique identifier for the run
   - `system_name`: Description of the system
   - `is_control_system`: Whether this is a control (toxin-free) system
   - `analysis_start_frame`, `analysis_end_frame`, `trajectory_total_frames`: Frame information

2. A `metrics` table with:
   - `metric_name`: Name of the metric (e.g., "DW_PROA_Open_Fraction")
   - `value`: Numerical value of the metric
   - `units`: Units of measurement (e.g., "%", "Å")
   - `module_name`: Source module that generated the metric

## Testing

Generate test data and run the report generator:

```bash
# Run test script
python tests/test_report_generator.py
```

## Project Structure

```
enhanced_report_generator/
├── core/
│   ├── data_models.py            # Core data structures
├── components/
│   ├── data_aggregator.py        # Collects metrics from individual DBs
│   ├── data_access.py            # DB interaction and metric comparison
│   ├── analysis_selection.py     # Filtering and sectioning of metrics
│   ├── visualization.py          # Plotting and visualization
│   ├── ai_insights.py            # AI-powered interpretations
│   ├── rendering.py              # HTML/PDF report generation
├── templates/
│   ├── system_overview_template.html  # Template for system overview
│   ├── report_template.html      # Template for detailed report
├── config/
│   ├── glossary_mapping.csv      # Maps metric prefixes to sections
├── assets/
│   ├── style.css                 # CSS for reports
├── main.py                       # Main entry point
```

## Future Enhancements

- Integration with real AI/LLM services for advanced insights
- Additional visualization types
- PDF export functionality
- Interactive Streamlit dashboard output