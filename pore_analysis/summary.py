# filename: pore_analysis/summary.py
"""
Summary generation module for pore analysis results.

This module creates a summary of analysis results from the database,
consolidating key metrics, statistics, and file references.
It now uses plots_dict.json to determine which plots to query from the database.
"""

import os
import json
import logging
from datetime import datetime
import sqlite3
from typing import Dict, List, Any, Optional

# Import necessary database functions
from pore_analysis.core.database import (
    connect_db,
    get_simulation_metadata,
    list_modules,
    get_product_path,
    register_product # Keep register_product if calculate_summary uses it
)
# Import the plot query loader function from html.py
# (Ideally, move this function to core.utils later)
try:
    from pore_analysis.html import load_plot_queries
except ImportError as e:
    print(f"Error importing load_plot_queries from html.py: {e}")
    # Define a dummy function if import fails, so the rest of the code doesn't break immediately
    def load_plot_queries(*args, **kwargs) -> List[Dict[str, str]]:
        logging.error("load_plot_queries function is unavailable. Cannot load plot definitions.")
        return []

# Import logging setup if calculate_summary needs it (optional)
# from pore_analysis.core.logging import setup_system_logger

# Import get_all_metrics (assuming it's defined correctly, possibly in database.py now or still needed here)
# Copied definition from previous context for self-containment
def get_all_metrics(conn: sqlite3.Connection) -> Dict[str, Dict[str, Any]]:
    """
    Get all metrics with their units from the database.

    Args:
        conn: Database connection

    Returns:
        Dictionary mapping metric_name to {'value': value, 'units': units} dict
    """
    metrics_dict: Dict[str, Dict[str, Any]] = {}
    logger = logging.getLogger(__name__) # Use local logger instance
    try:
        # Ensure row factory is set for dictionary access
        original_factory = conn.row_factory
        # Use standard Row factory which allows access by index and name
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        # Select required columns explicitly
        cursor.execute("SELECT metric_name, value, units FROM metrics")
        results = cursor.fetchall()
        conn.row_factory = original_factory  # Restore original factory

        for result in results:
            try:
                # Access by column name using sqlite3.Row
                metric_name = result['metric_name']
                value = result['value']
                # Handle potential None for units
                units = result['units'] if result['units'] is not None else ''
                metrics_dict[metric_name] = {'value': value, 'units': units}
            except (IndexError, KeyError, TypeError) as e: # Catch potential errors during row processing
                logger.warning(f"Could not parse metric row: {dict(result) if isinstance(result, sqlite3.Row) else result} due to error: {e}") # Log the row content if possible

        return metrics_dict
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}", exc_info=True)
        # Restore factory even on error
        if 'original_factory' in locals() and hasattr(conn, 'row_factory'):
            conn.row_factory = original_factory
        return {}  # Return empty dict on error


logger = logging.getLogger(__name__)

# Deprecating calculate_summary in favor of generate_summary_from_database
# as main.py now handles saving/registration differently.
# Keeping it for potential standalone use or reference.
def calculate_summary(run_dir):
    """
    Calculate a summary of analysis results from the database.
    [Note: This function might be deprecated as main.py workflow evolves.]

    Args:
        run_dir (str): Path to the simulation directory

    Returns:
        dict: Summary dictionary with analysis results
    """
    logger.warning("Function 'calculate_summary' may be deprecated. Use 'generate_summary_from_database' called by main.py workflow.")
    db_conn = connect_db(run_dir)
    if db_conn is None:
        logger.error(f"Failed to connect to database for {run_dir} in calculate_summary")
        return {}

    summary = generate_summary_from_database(run_dir, db_conn) # Call the main generator

    # Save summary to a JSON file (original logic kept here for now)
    if summary: # Only save if summary generation succeeded
        try:
            summary_file = os.path.join(run_dir, "analysis_summary.json")
            with open(summary_file, 'w') as f:
                # Use sort_keys for consistent output, handle potential non-serializable data if clean_json_data wasn't used
                json.dump(summary, f, indent=4, sort_keys=True, default=str)
            logger.info(f"Saved analysis summary (via calculate_summary) to {summary_file}")

            # Register the summary in the database
            # Use a specific module name like "summary_generation"
            register_product(db_conn, "summary_generation", "json", "summary",
                             os.path.relpath(summary_file, run_dir), # Use relative path
                             subcategory="analysis_summary", # Consistent subcategory
                             description="Analysis summary JSON file")
        except Exception as e:
            logger.error(f"Failed to save or register analysis summary JSON: {e}")
        finally:
             if db_conn: db_conn.close()
    elif db_conn:
         db_conn.close() # Close connection even if summary failed

    return summary

# Keeping get_summary as it provides a simple loading interface
def get_summary(run_dir):
    """
    Get the analysis summary, loading from analysis_summary.json file.
    [Note: calculation logic is now primarily handled by generate_summary_from_database]

    Args:
        run_dir (str): Path to the simulation directory

    Returns:
        dict: Summary dictionary with analysis results or empty dict if file not found/invalid.
    """
    summary_file = os.path.join(run_dir, "analysis_summary.json")
    if os.path.exists(summary_file):
        try:
            with open(summary_file, 'r') as f:
                summary = json.load(f)
            logger.info(f"Loaded analysis summary from {summary_file}")
            return summary
        except Exception as e:
            logger.warning(f"Failed to load summary file {summary_file}: {e}")
            return {} # Return empty if loading fails
    else:
        logger.warning(f"Summary file not found: {summary_file}")
        return {} # Return empty if file doesn't exist


# --- generate_summary_from_database (REVISED) ---
def generate_summary_from_database(run_dir: str, db_conn: sqlite3.Connection) -> dict:
    """
    Generate a summary of analysis results directly from the database connection.
    Uses plots_dict.json to determine which plots to query.
    Does not save the summary to a file or register it itself.

    Args:
        run_dir (str): Path to the simulation directory.
        db_conn (sqlite3.Connection): Active database connection.

    Returns:
        dict: Summary dictionary with analysis results, or empty dict on error.
    """
    logger.info(f"Generating summary directly from database for {run_dir}")

    if db_conn is None:
        logger.error(f"Invalid database connection provided for {run_dir}")
        return {}

    # Initialize summary dictionary
    summary = {
        'run_dir': run_dir,
        'run_name': os.path.basename(run_dir),
        'analysis_timestamp': datetime.now().isoformat(), # Timestamp of generation
        'module_status': {},
        'metrics': {},
        'key_plots': {}, # Initialize as empty dict
        'metadata': {}
    }

    # Get data from database
    try:
        # Get metadata from database
        # Ensure row factory allows dict access if needed, or use index
        original_factory = db_conn.row_factory
        db_conn.row_factory = sqlite3.Row
        cursor = db_conn.cursor()
        cursor.execute("SELECT key, value FROM simulation_metadata")
        meta_rows = cursor.fetchall()
        for row in meta_rows:
            summary['metadata'][row['key']] = row['value']
        # Update run_name from metadata if available
        summary['run_name'] = summary['metadata'].get('run_name', summary['run_name'])
        db_conn.row_factory = original_factory # Restore

        # Determine if it's a control system from metadata
        summary['is_control_system'] = summary['metadata'].get('is_control_system') == 'True'

        # Get modules and their status
        modules = list_modules(db_conn) # Assumes list_modules returns list of dicts
        for module in modules:
            summary['module_status'][module['module_name']] = module['status']

        # Get all metrics
        metrics = get_all_metrics(db_conn) # Uses the function defined above
        summary['metrics'] = metrics

        # --- Fetch Key Plot Paths using plots_dict.json ---
        logger.debug("Loading plot queries from plots_dict.json for summary...")
        plot_queries = load_plot_queries() # Load the definitions

        if not plot_queries:
            logger.warning("No plot queries loaded. Summary will not contain plot paths.")
        else:
            logger.debug(f"Attempting to find {len(plot_queries)} plots defined in plots_dict.json")
            found_plot_count = 0
            for plot_config in plot_queries:
                key = plot_config['template_key']
                ptype = plot_config['product_type']
                cat = plot_config['category']
                subcat = plot_config['subcategory']
                mod_name = plot_config['module_name']

                # Query the database using the details from plots_dict.json
                plot_path = get_product_path(db_conn, ptype, cat, subcat, mod_name)

                if plot_path:
                    summary['key_plots'][key] = plot_path
                    found_plot_count += 1
                    logger.debug(f"Found plot for key='{key}': Path='{plot_path}' (Module='{mod_name}', Subcat='{subcat}')")
                else:
                    logger.debug(f"Plot NOT FOUND for key='{key}' (Module='{mod_name}', Subcat='{subcat}')")
            logger.info(f"Found {found_plot_count} out of {len(plot_queries)} plots defined in plots_dict.json for summary.")
            # Check specifically for the plots that were previously missing
            if "g_g_distances" not in summary['key_plots']: logger.warning("G-G distance plot ('subunit_comparison') still not found in summary key_plots.")
            if "com_distances" not in summary['key_plots']: logger.warning("COM distance plot ('comparison') still not found in summary key_plots.")


        # Final check if essential parts are missing
        if not summary['metadata'] or not summary['module_status']:
            logger.warning("Summary generation resulted in missing metadata or module status.")

        logger.info(f"Successfully generated summary dictionary from database for {run_dir}")
        return summary

    except Exception as e:
        logger.error(f"Error generating summary from database for {run_dir}: {e}", exc_info=True)
        # Ensure factory is restored on error
        if 'original_factory' in locals() and hasattr(db_conn, 'row_factory'):
            db_conn.row_factory = original_factory
        return {} # Return empty dict on error