# **Pore Analysis Suite**

A comprehensive toolkit for analyzing ion channel structures and dynamics from molecular dynamics trajectory data.

## **Database-Centric Refactoring**

This codebase has been refactored to use a database-centric approach with the following key improvements:

1.  **Separation of Computation from Visualization**: Each analysis module now clearly separates calculation logic from visualization, enabling more flexible pipelines.
2.  **Database Integration**: All analysis products (output files), metrics (numerical results), and metadata are registered in a SQLite database (analysis\_registry.db) within each simulation's run directory. This facilitates tracking, reproducibility, and downstream comparative analysis.
3.  **Module Status Tracking**: The execution status (running, success, failed) of each analysis module is tracked in the database, enabling dependency management and incremental analysis.
4.  **Cross-Simulation Analysis**: Storing standardized metrics in the database makes it easier to compare results across different simulation runs.

## **Database Schema**

The database (SQLite) includes the following tables:

* simulation\_metadata: General information about the simulation and analysis.
* analysis\_modules: Tracks which analysis modules have been run, their status, parameters, and errors.
* analysis\_products: Central registry of all files produced by the analysis (CSVs, PNGs, etc.), linked to the generating module.
* metrics: Stores numerical metrics (e.g., mean distances, counts) with units, linked to the generating module.
* dependencies: (Future use) Tracks dependencies between analysis modules.
* product\_relationships: (Future use) Tracks relationships between analysis products (e.g., a plot derived from a CSV).

## **Module Structure**

The package is organized into the following structure:

pore\_analysis/
├── core/                  \# Core utilities (DB, config, filtering, logging, utils)
│   ├── database.py        \# \<-- Central database interaction logic
│   ├── config.py          \# \<-- Configuration constants (thresholds, etc.)
│   ├── filtering.py       \# \<-- Data filtering (PBC correction, smoothing)
│   ├── logging.py         \# \<-- Logging setup
│   └── utils.py           \# \<-- General helper functions
├── modules/               \# Specific analysis tasks
│   ├── core\_analysis/     \# \<-- Basic G-G/COM distance analysis
│   │   ├── computation.py \# Calculation part
│   │   └── visualization.py \# Plotting part
│   ├── ion\_analysis/      \# Ion tracking and conduction
│   ├── inner\_vestibule\_analysis/  \# Water analysis
│   ├── dw\_gate\_analysis/  \# DW gate dynamics
│   ├── tyrosine\_analysis/ \# SF tyrosine rotamers
│   ├── orientation\_contacts/ \# Toxin orientation
│   └── gyration\_analysis/ \# Gyration radius analysis
├── templates/             \# HTML report templates
│   └── report\_template.html \# Main template & includes
├── main.py                \# \<-- Main script to run the analysis workflow
├── summary.py             \# \<-- Generates JSON summary from DB
├── html.py                \# \<-- Generates HTML report from summary/DB
├── plots\_dict.json       \# \<-- **NEW: Defines plots for HTML report**
└── \_\_init\_\_.py            \# Package indicator

## **Using the Refactored Modules**

Each analysis module follows a consistent pattern:

1.  **Computation (computation.py)**: Performs calculations, saves results to files (e.g., CSV), registers itself and its products in the database, and stores key metrics.
2.  **Visualization (visualization.py)**: Retrieves necessary data file paths from the database, generates plots (e.g., PNG), saves them, and registers the plot files in the database.

### **Example: Core Analysis**

\# Computational step
from pore\_analysis.modules.core\_analysis import analyze\_trajectory, filter\_and\_save\_data

\# Run the analysis (calculates raw data, saves CSV, registers in DB)
results \= analyze\_trajectory(run\_dir, psf\_file, dcd\_file)

\# Filter the data (applies filtering, saves filtered CSV, stores metrics in DB)
if results\['status'\] \== 'success':
    filtered\_results \= filter\_and\_save\_data(
        run\_dir,
        results\['data'\]\['dist\_ac'\],
        results\['data'\]\['dist\_bd'\],
        results\['data'\]\['com\_distances'\],
        results\['data'\]\['time\_points'\],
        is\_control\_system=results\['metadata'\]\['is\_control\_system'\]
    )

\# Visualization step
from pore\_analysis.modules.core\_analysis import plot\_distances, plot\_kde\_analysis

\# Create plots (looks up CSV paths from DB, saves PNGs, registers plots in DB)
g\_g\_plots \= plot\_distances(run\_dir, is\_gg=True)
com\_plots \= plot\_distances(run\_dir, is\_gg=False) \# Only generates comparison plot now
kde\_plot \= plot\_kde\_analysis(run\_dir) \# Generates KDE distribution plot

## **Command Line Interface**

The analysis workflow is orchestrated by main.py:

\# Run all available/implemented analysis modules
python \-m pore\_analysis.main /path/to/simulation

\# Run specific analysis modules (e.g., core and orientation analysis)
python \-m pore\_analysis.main /path/to/simulation \--core --orientation

\# Generate HTML report using existing database results (no re-analysis)
python \-m pore\_analysis.main /path/to/simulation \--summary-only --report

\# Force rerun of all specified analyses, even if DB shows 'success'
python \-m pore\_analysis.main /path/to/simulation --orientation --force-rerun

\# Reinitialize database (WARNING: Deletes previous results for this run)
python \-m pore\_analysis.main /path/to/simulation \--reinit-db \--core

## **Cross-Simulation Analysis**

The database-centric approach makes comparing metrics across simulations straightforward by querying the individual analysis\_registry.db files:

import os
import sqlite3
import pandas as pd

\# List of simulation directories
sim\_dirs \= \['/path/to/sim1', '/path/to/sim2', '/path/to/sim3'\]

\# Collect specific metrics (e.g., filtered COM Mean)
metrics\_data \= \[\]
for sim\_dir in sim\_dirs:
    db\_path \= os.path.join(sim\_dir, 'analysis\_registry.db')
    if not os.path.exists(db\_path):
        print(f"Warning: DB not found for {sim\_dir}")
        continue

    conn \= sqlite3.connect(db\_path)
    \# Use dict\_factory for easy row access if defined in core.database
    conn.row\_factory \= sqlite3.Row # Or your dict\_factory
    cursor \= conn.cursor()

    try:
        \# Get simulation name (assuming it's stored)
        cursor.execute("SELECT value FROM simulation\_metadata WHERE key='run\_name'")
        sim\_name\_row \= cursor.fetchone()
        sim\_name \= sim\_name\_row\['value'\] if sim\_name\_row else os.path.basename(sim\_dir) # Access by key

        \# Get specific metrics
        cursor.execute("""
            SELECT metric\_name, value, units
            FROM metrics
            WHERE metric\_name LIKE '%Mean\_Filt' OR metric\_name LIKE '%Min\_Filt'
        """) # Example query
        for row in cursor.fetchall():
            metrics\_data.append({
                'simulation': sim\_name,
                'metric\_name': row\['metric_name'\], # Access by key
                'value': row\['value'\],
                'units': row\['units'\]
            })
    except Exception as e:
        print(f"Error querying DB for {sim\_dir}: {e}")
    finally:
        conn.close()

\# Create DataFrame for analysis
if metrics\_data:
    metrics\_df \= pd.DataFrame(metrics\_data)
    print(metrics\_df.head())
else:
    print("No metrics collected.")

## **Development Guidelines**

This section outlines the conventions and workflow for developing new analysis modules for the suite.

### **Database Interaction Rules**

1.  **Module Status:**
    * Register module start using register\_module(conn, module\_name, status='running').
    * Update status reliably upon completion (update\_module\_status(conn, module\_name, status='success')) or failure (update\_module\_status(conn, module\_name, status='failed', error\_message=...)).
2.  **Product Registration (register\_product):**
    * **Mandatory:** Register *every* generated file (CSV, JSON, PNG, etc.) intended for later use or tracking.
    * **Strict Parameter Order:** Use the exact signature: register\_product(db\_conn, module\_name, product\_type, category, relative\_path, subcategory, description).
    * **relative\_path:** Must be relative to the run\_dir (e.g., module\_subdir/output\_file.csv). Use os.path.relpath(full\_path, run\_dir).
    * **subcategory:** Use clear, consistent, and *unique* names for distinct products, especially plots (e.g., 'filtered\_distances', 'subunit\_comparison', 'kde\_analysis'). This is the primary identifier used by html.py via the plot configuration file.
3.  **Metric Storage (store\_metric):**
    * **Mandatory:** Store all key numerical results for summary tables or comparisons.
    * **Naming:** Use descriptive metric\_names, indicating source if relevant (e.g., G\_G\_AC\_Mean\_Filt).
    * **Units:** Provide correct units ("Å", "ns", "°", "count", etc.).
    * **Source:** Calculate metrics from the appropriate data (e.g., filtered data).
4.  **Data Retrieval (get\_product\_path):**
    * Visualization/reporting code must retrieve file paths from the database first.
    * Use specific product\_type, category, and subcategory arguments to find the correct product.
    * Implement file system fallbacks cautiously, logging warnings.
5.  **HTML Integration (via plots\_dict.json):**
    * The list of plots included in the HTML report is now defined in `pore_analysis/plots_dict.json`.
    * This JSON file contains a list of objects, each defining a plot to be included. The required keys for each object are: `template_key`, `product_type`, `category`, `subcategory`, and `module_name`.
    * `html.py` reads this file using the `load_plot_queries` function to determine which plots registered in the database should be fetched and embedded.
    * **Crucially, the `subcategory` in a `plots_dict.json` entry *must exactly match* the `subcategory` used during `register_product` in the visualization script that generates the corresponding plot.**

### **Plotting Conventions**

1.  **Titles:**
    * **No Python Titles:** Do *not* add titles within the Python plotting code (`plt.title`, `ax.set_title`, `fig.suptitle`).
    * **HTML Titles:** Use HTML structure (e.g., `<h3>`, `<h4>`) within the template for overall plot titles.
    * **Multi-Panel Figures:** If a single image contains multiple conceptual panels (like the G-G plot), add descriptive labels *in the HTML template* above or beside the image (e.g., using flexbox divs) to identify the panels. Avoid overly complex figures if HTML layout can achieve the goal.
2.  **Layout & Content:**
    * **Target Context:** Design plots considering their final HTML display size (e.g., full-width, half-width column). Ensure legibility.
    * **Completeness:** Verify all intended visual elements (lines, histograms, KDEs, legends, axis labels) are present and correct in the saved plot.
    * **Simplicity:** Prefer generating simpler, single-purpose plots arranged using HTML layout (`two-column` class) over complex multi-panel images, unless essential for direct comparison within one image.
3.  **Styling:**
    * Adhere strictly to the `STYLE` dictionary defined in `pore_analysis/core/plotting_style.py` for colors, lines, markers, fonts.
    * Use the `setup_style()` function (also defined in `pore_analysis/core/plotting_style.py`) within each visualization script to apply these settings.
    * Be prepared to iteratively adjust `STYLE` values (e.g., font sizes) based on feedback for optimal legibility.

### **Module Development Workflow & Checklist**

1.  **Define Goal:** Specify the analysis and desired outputs (data files, metrics, plots).
2.  **Implement Computation (computation.py):**
    * Add calculation logic.
    * Save data file(s) (e.g., to run\_dir/module\_name/).
    * register\_module() start (status='running').
    * register\_product() for data file(s) (use correct relative\_path, subcategory).
    * Calculate required metrics.
    * store\_metric() for each metric (use correct metric\_name, units).
    * update\_module\_status() end (status='success' or 'failed').
3.  **Implement Visualization (visualization.py):**
    * get\_product\_path() to retrieve data file path (using correct subcategory).
    * Load data.
    * Generate plot(s) (adhere to styling, *omit Python titles*).
    * Save plot file(s) (e.g., to run\_dir/module\_name/).
    * register\_product() for *each* plot file (use correct relative\_path, *unique subcategory*).
4.  **Update HTML Reporting (`plots_dict.json` & `templates/`):**
    * **(`plots_dict.json`):** Add a new JSON object for the plot. Ensure `template_key` is unique and sensible, and `product_type`, `category`, `subcategory`, `module_name` exactly match how the plot was registered in step 3.
    * **(`templates/`):** Add/modify the relevant section in the appropriate \_tab\_\*.html. Add HTML structure (\<h3\>, plot-container, etc.) for titles and layout. Use {% if plots.get('your\_template\_key') %} to display the image (using the `template_key` from `plots_dict.json`). Add/modify tables to display new metrics using the `metric_name` from step 2.
5.  **Test & Verify:**
    * Run the full pipeline (main.py \--\[module\] --report).
    * Check the database (inspect\_db.py) for correct registration (paths, subcategories, metrics).
    * Verify the HTML report: Plots displayed? Metrics correct? Layout/styling okay? Titles correct?

## **Future Enhancements**

Planned improvements for the refactored codebase:

1.  **API Documentation**: Generate comprehensive API documentation (e.g., using Sphinx).
2.  **Web Interface**: Create a web-based interface for browsing analysis results across multiple runs.
3.  **Parallel Processing**: Enable parallel execution of independent analysis modules.
4.  **Version Tracking**: Track software and analysis version numbers in the database.
5.  **More Visualization Options**: Add interactive visualizations (e.g., using Plotly or Bokeh).
6.  **Dependency Tracking**: Fully implement the dependencies and product\_relationships tables for robust workflow management.