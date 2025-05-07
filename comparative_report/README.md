# Pore Cross-Analysis Suite

A comprehensive toolkit for comparative analysis of toxin-bound and control ion channel systems from molecular dynamics simulations.

## Overview

The Pore Cross-Analysis Suite enables systematic comparisons between toxin-bound and control systems by:

1. Extracting metrics from individual system databases
2. Performing statistical comparisons between system types
3. Generating visualizations of key differences
4. Providing AI-powered insights (optional)
5. Creating comprehensive HTML reports

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd comparative_report

# Install the package
pip install -e .
```

## Usage

### Basic Usage

```bash
# Run the complete analysis pipeline
python cross_analysis_main.py --toxin-dir /path/to/toxin --control-dir /path/to/control --output-dir results

# Extract metrics only without analysis
python cross_analysis_main.py --toxin-dir /path/to/toxin --control-dir /path/to/control --output-dir results --extract-only

# Run analysis on specific metric categories
python cross_analysis_main.py --toxin-dir /path/to/toxin --control-dir /path/to/control --output-dir results --categories structure ion water
```

### With AI Integration

```bash
# Enable AI-assisted analysis (requires Claude API key)
python cross_analysis_main.py --toxin-dir /path/to/toxin --control-dir /path/to/control --output-dir results --enable-ai --api-key your_api_key

# Use environment variable for API key
export CLAUDE_API_KEY=your_api_key
python cross_analysis_main.py --toxin-dir /path/to/toxin --control-dir /path/to/control --output-dir results --enable-ai
```

## Directory Structure

The tool expects system directories to contain the following structure:

```
system_dir/
└── */
    └── */
        └── R*/
            └── analysis_registry.db
```

The tool will recursively search for `analysis_registry.db` files in both the toxin and control directories.

## Output

The tool creates the following outputs in the specified output directory:

1. `cross_analysis.db`: Meta-database containing aggregated metrics and analysis results
2. `cross_analysis_report.html`: Comprehensive HTML report of the comparison
3. Visualization plots for key metrics and comparisons
4. Log files detailing the analysis process

## Features

- **Metrics Extraction**: Automatically extracts metrics from individual system databases
- **Statistical Analysis**: Performs statistical tests to identify significant differences
- **Visualization**: Creates various plots for comparing metrics across systems
- **AI Integration**: Generates insights and hypotheses based on the observed patterns
- **HTML Reporting**: Creates a comprehensive, interactive HTML report

## Requirements

- Python 3.7+
- NumPy, Pandas, Matplotlib, Seaborn
- SciPy, StatsModels
- Jinja2, Markdown
- Requests (for AI integration)