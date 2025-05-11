# Enhanced Cross-Simulation Analysis Tool

This tool generates comprehensive comparative reports between toxin-bound and toxin-free (control) channel simulations.

## Overview

The Enhanced Report Generator aggregates metrics from individual simulation analyses, performs statistical comparisons, and generates HTML reports with visualizations and data-driven interpretations.

## Usage

### Single Command Execution

The most straightforward way to run the tool is with a single command:

```bash
python -m enhanced_report_generator.main \
  --toxin-dir /home/labs/bmeitan/karbati/rCs1/toxin/toxin \
  --control-dir /home/labs/bmeitan/karbati/rCs1/control/control \
  --output-dir /home/labs/bmeitan/karbati/rCs1/enhanced_report_results \
  --title "rCs1 Toxin Analysis"
```

This will:
1. Aggregate metrics from both directories
2. Generate a system overview report
3. Generate a detailed comparative report

### Advanced Usage

You can also run the tool with more control over the process:

#### Skip Data Aggregation

If you've already aggregated the data and want to regenerate reports:

```bash
python -m enhanced_report_generator.main \
  --output-dir /home/labs/bmeitan/karbati/rCs1/enhanced_report_results \
  --title "rCs1 Toxin Analysis" \
  --skip-aggregation
```

#### Generate Only System Overview

```bash
python -m enhanced_report_generator.main \
  --toxin-dir /home/labs/bmeitan/karbati/rCs1/toxin/toxin \
  --control-dir /home/labs/bmeitan/karbati/rCs1/control/control \
  --output-dir /home/labs/bmeitan/karbati/rCs1/enhanced_report_results \
  --title "rCs1 Toxin Analysis" \
  --overview-only
```

#### Generate Only Detailed Report

```bash
python -m enhanced_report_generator.main \
  --output-dir /home/labs/bmeitan/karbati/rCs1/enhanced_report_results \
  --title "rCs1 Toxin Analysis" \
  --detailed-only \
  --skip-aggregation
```

## Generated Reports

The reports will be saved in the specified output directory:

- System Overview Report: `/home/labs/bmeitan/karbati/rCs1/enhanced_report_results/rcs1_toxin_analysis_system_overview.html`
- Detailed Report: `/home/labs/bmeitan/karbati/rCs1/enhanced_report_results/rcs1_toxin_analysis_detailed_report.html`
- Plots: `/home/labs/bmeitan/karbati/rCs1/enhanced_report_results/plots/`

## Analysis Process

The tool performs these steps:

1. **Data Aggregation**:
   - Collects metrics from individual `analysis_registry.db` files
   - Stores them in a central database for comparative analysis

2. **System Overview Report**:
   - Provides basic statistics about the datasets
   - Shows high-level summary of the systems

3. **Detailed Comparative Report**:
   - Performs statistical comparison between toxin and control groups
   - Visualizes key metrics with plots
   - Groups metrics into thematic sections
   - Provides data-driven interpretations of the results

## Command Line Options

```
--toxin-dir: Directory containing toxin simulation data
--control-dir: Directory containing control simulation data
--output-dir: Directory to save the generated reports and plots
--aggregated-db-name: Name of the SQLite database file
--title: Base title for the generated reports
--skip-aggregation: Skip data aggregation and use existing database
--force-aggregate: Force re-aggregation of data even if database exists
--overview-only: Generate only the system overview report
--detailed-only: Generate only the detailed report
--log-level: Set the logging level (DEBUG, INFO, WARNING, ERROR)
```