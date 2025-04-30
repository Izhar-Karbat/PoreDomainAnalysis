# pore_analysis/html.py
"""
HTML report generation module for pore analysis results.

This module creates an HTML report of analysis results, with embedded plots
and tables of key metrics. It relies on the database for tracking analysis products
and configuration parameters used for the run.
"""

import os
import base64
import logging
from datetime import datetime
import json
import sqlite3 # Import for type hinting
from typing import Dict, Any, Optional, List, Type # Added List, Type
from jinja2 import Environment, FileSystemLoader, select_autoescape

# Import necessary functions from the database module
from pore_analysis.core.database import (
    connect_db,
    get_product_path,
    register_product,
    register_module,       # Needed for registering report_generation module status
    update_module_status,  # Needed for updating report_generation module status
    get_config_parameters,
    get_simulation_metadata # Need this specific one too
)

# Make sure get_all_metrics is available, either imported or defined here
# Copied definition from summary.py for self-containment, assuming it's accurate
def get_all_metrics(conn: sqlite3.Connection) -> Dict[str, Dict[str, Any]]:
    """
    Get all metrics with their units from the database.

    Args:
        conn: Database connection

    Returns:
        Dictionary mapping metric_name to {'value': value, 'units': units} dict
    """
    metrics_dict: Dict[str, Dict[str, Any]] = {}
    try:
        # Ensure row factory is set for dictionary access
        original_factory = conn.row_factory
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT metric_name, value, units FROM metrics")
        results = cursor.fetchall()
        conn.row_factory = original_factory  # Restore original factory

        for result in results:
            try:
                # Access by column name
                metric_name = result['metric_name']
                value = result['value']
                units = result['units'] if result['units'] is not None else ''  # Use empty string if units are None
                metrics_dict[metric_name] = {'value': value, 'units': units}
            except (IndexError, KeyError, TypeError) as e:  # Added TypeError
                logger.warning(f"Could not parse metric row: {result} due to error: {e}")

        return metrics_dict
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}", exc_info=True)
        # Restore factory even on error
        if 'original_factory' in locals() and hasattr(conn, 'row_factory'):
            conn.row_factory = original_factory
        return {}  # Return empty dict on error


logger = logging.getLogger(__name__)

# Function to load plot queries from JSON
def load_plot_queries(config_path: str = "plots_dict.json") -> List[Dict[str, str]]:
    """Loads plot query definitions from a JSON file."""
    script_dir = os.path.dirname(__file__)
    full_config_path = os.path.join(script_dir, config_path)

    if not os.path.exists(full_config_path):
        logger.error(f"Plot configuration file not found: {full_config_path}")
        return []
    try:
        with open(full_config_path, 'r') as f:
            plot_queries_list = json.load(f)
        # Basic validation
        if not isinstance(plot_queries_list, list):
            raise TypeError("Plot config is not a list.")
        for item in plot_queries_list:
            if not isinstance(item, dict) or not all(k in item for k in [
                "template_key", "product_type", "category", "subcategory", "module_name"]):
                raise ValueError(f"Invalid item format in plot config: {item}")
        logger.info(f"Successfully loaded {len(plot_queries_list)} plot queries from {full_config_path}")
        return plot_queries_list
    except (json.JSONDecodeError, TypeError, ValueError, Exception) as e:
        logger.error(f"Failed to load or parse plot configuration file {full_config_path}: {e}", exc_info=True)
        return []


def generate_html_report(run_dir: str, summary: Optional[dict] = None) -> Optional[str]:
    """
    Generate an HTML report for the analysis results.

    Args:
        run_dir (str): Path to the simulation directory
        summary (dict, optional): Analysis summary dictionary. If None, it will be loaded/calculated.

    Returns:
        str: Path to the generated HTML file, or None on failure.
    """
    logger.info(f"Generating HTML report for {run_dir}")

    # Connect to the database for plot retrieval and metrics
    db_conn = connect_db(run_dir)
    if not db_conn:
        logger.error("Failed to connect to database for HTML report")
        return None

    # Get summary if not provided
    if summary is None:
        # Assuming generate_summary_from_database exists and works as intended
        try:
            from pore_analysis.summary import generate_summary_from_database
            summary = generate_summary_from_database(run_dir, db_conn)
        except ImportError:
            logger.error("Could not import generate_summary_from_database. Summary generation skipped.")
            summary = {} # Initialize empty summary

        if not summary:
            logger.error("Failed to load or generate summary data for HTML report.")
            if db_conn:
                db_conn.close()
            return None

    # Set up Jinja2 environment
    template_dir = os.path.join(os.path.dirname(__file__), 'templates')
    try:
        env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=select_autoescape(['html', 'xml'])
        )
        template = env.get_template('report_template.html')
    except Exception as e:
        logger.error(f"Failed to load Jinja2 environment or template: {e}")
        if db_conn:
            db_conn.close()
        return None

    # Retrieve plot paths and convert to base64
    plots: Dict[str, str] = {}
    plot_queries_config = load_plot_queries()
    if not plot_queries_config:
        logger.warning("No plot queries loaded from config. Report may be missing plots.")

    for plot_config in plot_queries_config:
        plot_key = plot_config['template_key']
        product_type = plot_config['product_type']
        category = plot_config['category']
        subcategory = plot_config['subcategory']
        module_name = plot_config['module_name']

        if plot_key not in plots: # Avoid redundant lookups if key already found
            plot_path_rel = get_product_path(
                db_conn,
                product_type=product_type,
                category=category,
                subcategory=subcategory,
                module_name=module_name
            )
            if plot_path_rel:
                full_path = os.path.join(run_dir, plot_path_rel)
                if os.path.exists(full_path):
                    try:
                        with open(full_path, 'rb') as f:
                            plots[plot_key] = base64.b64encode(f.read()).decode('utf-8')
                        logger.debug(f"Loaded plot '{plot_key}' from {full_path} (Module: {module_name}, Subcat: {subcategory})")
                    except Exception as e:
                        logger.warning(f"Failed to load/encode plot '{plot_key}' from {full_path}: {e}")
                else:
                    logger.warning(f"Plot file '{plot_key}' not found at registered path: {full_path} (Module: {module_name}, Subcat: {subcategory})")


    # Fetch and Prepare Config Parameters
    config_params = get_config_parameters(db_conn) # Returns {name: {'value':.., 'type':.., 'description':..}}

    def get_config_val(name: str, default: Any, target_type: Type) -> Any:
        """Safely retrieves and converts config value, returning default on error."""
        param_info = config_params.get(name)
        if param_info and param_info.get('value') is not None:
            try:
                if target_type == bool: value_str = param_info['value'].lower(); return value_str == 'true'
                elif target_type in (list, dict) and param_info.get('type') in ('list', 'dict'): return json.loads(param_info['value'])
                else: return target_type(param_info['value'])
            except (ValueError, TypeError, json.JSONDecodeError) as e:
                logger.warning(f"Could not convert config param '{name}' value '{param_info['value']}' to {target_type.__name__}, using default {default}. Error: {e}")
                return default
        elif name not in config_params: logger.warning(f"Config param '{name}' not found in database, using default {default}.")
        return default

    gyration_threshold = get_config_val('GYRATION_FLIP_THRESHOLD', default=4.5, target_type=float)
    gyration_tolerance = get_config_val('GYRATION_FLIP_TOLERANCE_FRAMES', default=5, target_type=int)
    frames_per_ns = get_config_val('FRAMES_PER_NS', default=10.0, target_type=float)
    if frames_per_ns == 0: logger.warning("FRAMES_PER_NS retrieved from DB is 0, using default 10.0 for template calculations."); frames_per_ns = 10.0

    # --- MODIFICATION: Fetch ALL Metadata ---
    all_metadata = {}
    try:
        # Use dict_factory temporarily for this query
        original_factory = db_conn.row_factory
        db_conn.row_factory = sqlite3.Row # Use standard Row factory for dict access
        cursor = db_conn.cursor()
        cursor.execute("SELECT key, value FROM simulation_metadata")
        rows = cursor.fetchall()
        db_conn.row_factory = original_factory  # Restore original factory

        for row in rows:
            try:
                # Access by column name using sqlite3.Row
                key = row['key']
                value = row['value']
                all_metadata[key] = value
            except (TypeError, KeyError, IndexError) as e:
                logger.warning(f"Could not parse metadata row: {row}. Error: {e}")

        logger.debug(f"Fetched {len(all_metadata)} metadata key-value pairs.")
    except Exception as e:
        logger.error(f"Failed to fetch all simulation metadata: {e}", exc_info=True)
        if 'original_factory' in locals() and hasattr(db_conn, 'row_factory'): # Ensure factory is restored on error
             db_conn.row_factory = original_factory
        # Continue with potentially empty metadata dict
    # --- END MODIFICATION ---

    # Get metrics with units using the database connection
    metrics = get_all_metrics(db_conn)

    # Prepare data for the template rendering
    report_data = {
        'run_name': all_metadata.get('run_name', os.path.basename(run_dir)), # Use fetched metadata
        'run_dir': run_dir,
        'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'analysis_timestamp': datetime.now().isoformat(),
        'is_control_system': all_metadata.get('is_control_system') == 'True', # Use fetched metadata
        'module_status': summary.get('module_status', {}), # Still get this from summary or recalculate
        'metrics': metrics, # Pass the metrics dict
        'plots': plots,     # Pass the plots dict
        'metadata': all_metadata, # Pass the FULL metadata dictionary
        # Pass config values fetched earlier...
        'GYRATION_FLIP_THRESHOLD': gyration_threshold,
        'GYRATION_FLIP_TOLERANCE_FRAMES': gyration_tolerance,
        'FRAMES_PER_NS': frames_per_ns
    }

    logger.debug(f"Report data prepared: {len(plots)} plots loaded, {len(metrics)} metrics retrieved.")
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Available plots keys for template: {list(plots.keys())}")
        logger.debug(f"Available metadata keys for template: {list(all_metadata.keys())}") # Log metadata keys
        logger.debug(f"Config values passed to template: "
                     f"GYRATION_FLIP_THRESHOLD={report_data['GYRATION_FLIP_THRESHOLD']}, "
                     f"GYRATION_FLIP_TOLERANCE_FRAMES={report_data['GYRATION_FLIP_TOLERANCE_FRAMES']}, "
                     f"FRAMES_PER_NS={report_data['FRAMES_PER_NS']}")

    # Render the HTML template
    try:
        html_content = template.render(**report_data)
    except Exception as e:
        logger.error(f"Failed to render HTML template: {e}", exc_info=True)
        if db_conn: db_conn.close()
        return None

    # Save the rendered HTML content
    html_file = os.path.join(run_dir, "data_analysis_report.html")
    try:
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        logger.info(f"Saved HTML report to {html_file}")
    except Exception as e:
        logger.error(f"Failed to save HTML report to {html_file}: {e}")
        if db_conn: db_conn.close()
        return None

    # Register the HTML report itself as a product and track its generation status
    report_module_name = "report_generation"
    try:
        register_module(db_conn, report_module_name, status='running')
        register_product(
            conn=db_conn,
            module_name=report_module_name,
            product_type="html",
            category="report",
            relative_path=os.path.relpath(html_file, run_dir),
            subcategory="analysis_report",
            description="Analysis HTML report"
        )
        update_module_status(db_conn, report_module_name, 'success')
    except Exception as e:
        logger.error(f"Failed to register HTML report product or update status in database: {e}")
        try:
            update_module_status(db_conn, report_module_name, 'failed', error_message=f"DB registration/update failed: {e}")
        except Exception as e_update:
             logger.error(f"Failed even to update report generation status to failed: {e_update}")

    # Close the database connection
    if db_conn:
        db_conn.close()

    # Return the path to the generated HTML file
    return html_file