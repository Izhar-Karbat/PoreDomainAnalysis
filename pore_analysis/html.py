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
import traceback # Import traceback for detailed error messages
from typing import Dict, Any, Optional, List, Type # Added List, Type
from jinja2 import Environment, FileSystemLoader, select_autoescape, TemplateNotFound # Import TemplateNotFound

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

# (get_all_metrics function remains unchanged)
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

# Function to load plot queries from JSON (unchanged)
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


# --- generate_html_report (Revised) ---
def generate_html_report(run_dir: str, summary: Optional[dict] = None) -> Optional[str]:
    """
    Generate an HTML report for the analysis results.

    Ensures that an HTML file is always created at the target path and the path
    is returned, even if errors occur during generation (in which case a minimal
    error page is written).

    Args:
        run_dir (str): Path to the simulation directory
        summary (dict, optional): Analysis summary dictionary. If None, it will be loaded/calculated.

    Returns:
        str: Absolute path to the generated HTML file (data_analysis_report.html).
             Returns the path even if only an error page could be written.
             Returns None only if the database connection fails initially.
    """
    logger.info(f"Attempting to generate HTML report for {run_dir}")
    report_module_name = "report_generation"
    html_file_path = os.path.join(run_dir, "data_analysis_report.html") # Define path upfront

    db_conn: Optional[sqlite3.Connection] = None # Initialize
    report_generation_successful = False
    error_during_generation = None

    try:
        # Connect to the database for plot retrieval and metrics
        db_conn = connect_db(run_dir)
        if not db_conn:
            logger.error("Failed to connect to database for HTML report generation.")
            # Cannot proceed without DB, cannot even register failure status
            return None # Return None only in this critical DB connection failure case

        register_module(db_conn, report_module_name, status='running')

        # Get summary if not provided
        if summary is None:
            try:
                from pore_analysis.summary import generate_summary_from_database
                summary = generate_summary_from_database(run_dir, db_conn)
            except ImportError:
                logger.error("Could not import generate_summary_from_database.")
                summary = {} # Proceed with empty summary

            if not summary:
                logger.warning("Failed to load or generate summary data for HTML report.")
                summary = {} # Use empty dict to allow minimal report generation attempt

        # Set up Jinja2 environment
        template_dir = os.path.join(os.path.dirname(__file__), 'templates')
        env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=select_autoescape(['html', 'xml'])
        )
        template = env.get_template('report_template.html')

        # Retrieve plot paths and convert to base64
        plots: Dict[str, str] = {}
        plot_queries_config = load_plot_queries()
        if not plot_queries_config:
            logger.warning("No plot queries loaded from config. Report may be missing plots.")

        for plot_config in plot_queries_config:
            plot_key = plot_config['template_key']
            # Check if plot already loaded (avoid redundant lookups)
            if plot_key in plots: continue

            product_type = plot_config['product_type']
            category = plot_config['category']
            subcategory = plot_config['subcategory']
            module_name = plot_config['module_name']

            plot_path_rel = get_product_path(
                db_conn, product_type, category, subcategory, module_name
            )
            if plot_path_rel:
                full_path = os.path.join(run_dir, plot_path_rel)
                if os.path.exists(full_path):
                    try:
                        with open(full_path, 'rb') as f:
                            plots[plot_key] = base64.b64encode(f.read()).decode('utf-8')
                        logger.debug(f"Loaded plot '{plot_key}'")
                    except Exception as e_encode:
                        logger.warning(f"Failed to load/encode plot '{plot_key}' from {full_path}: {e_encode}")
                else:
                    logger.warning(f"Plot file '{plot_key}' not found at registered path: {full_path}")
            # else: logger.debug(f"Plot path not found in DB for key '{plot_key}'") # Optional: Log missing DB entries


        # Fetch Config Parameters (logic unchanged)
        config_params = get_config_parameters(db_conn)
        def get_config_val(name: str, default: Any, target_type: Type) -> Any:
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

        # Fetch ALL Metadata (logic unchanged)
        all_metadata = {}
        original_factory = db_conn.row_factory
        try:
            db_conn.row_factory = sqlite3.Row
            cursor = db_conn.cursor()
            cursor.execute("SELECT key, value FROM simulation_metadata")
            rows = cursor.fetchall()
            for row in rows:
                all_metadata[row['key']] = row['value']
        except Exception as e_meta:
            logger.error(f"Failed to fetch all simulation metadata: {e_meta}", exc_info=True)
        finally:
             if original_factory: db_conn.row_factory = original_factory # Restore


        # Get metrics with units (logic unchanged)
        metrics = get_all_metrics(db_conn)

        # Get pocket analysis parameters from config
        pocket_residence_thresh = get_config_val('POCKET_ANALYSIS_RESIDENCE_THRESHOLD', default=10, target_type=int)
        pocket_short_lived_thresh = get_config_val('POCKET_ANALYSIS_SHORT_LIVED_THRESH_NS', default=5.0, target_type=float)
        pocket_long_lived_thresh = get_config_val('POCKET_ANALYSIS_LONG_LIVED_THRESH_NS', default=10.0, target_type=float)
        
        # Prepare data for the template rendering
        report_data = {
            'run_name': all_metadata.get('run_name', os.path.basename(run_dir)),
            'run_dir': run_dir,
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'analysis_timestamp': datetime.now().isoformat(),
            'is_control_system': all_metadata.get('is_control_system') == 'True',
            'module_status': summary.get('module_status', {}),
            'metrics': metrics,
            'plots': plots,
            'metadata': all_metadata,
            'GYRATION_FLIP_THRESHOLD': gyration_threshold,
            'GYRATION_FLIP_TOLERANCE_FRAMES': gyration_tolerance,
            'FRAMES_PER_NS': frames_per_ns,
            # Add pocket analysis parameters
            'POCKET_ANALYSIS_RESIDENCE_THRESHOLD': pocket_residence_thresh,
            'POCKET_ANALYSIS_SHORT_LIVED_THRESH_NS': pocket_short_lived_thresh,
            'POCKET_ANALYSIS_LONG_LIVED_THRESH_NS': pocket_long_lived_thresh
        }

        # Render the HTML template
        html_content = template.render(**report_data)

        # Save the rendered HTML content
        with open(html_file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        logger.info(f"Successfully saved HTML report to {html_file_path}")
        report_generation_successful = True # Mark as successful

    except TemplateNotFound as e_tpl:
        error_during_generation = f"HTML template not found: {e_tpl}"
        logger.error(error_during_generation)
    except Exception as e_main:
        error_during_generation = f"Error during report generation: {e_main}"
        logger.error(error_during_generation, exc_info=True)
        # Attempt to write fallback error page
        try:
            error_html = f"""<!DOCTYPE html>
<html>
<head><title>Report Generation Error</title></head>
<body>
<h1>Report Generation Error</h1>
<p>An error occurred while generating the analysis report for {run_dir}.</p>
<p><strong>Error:</strong></p>
<pre>{e_main}</pre>
<p><strong>Traceback:</strong></p>
<pre>{traceback.format_exc()}</pre>
</body>
</html>"""
            with open(html_file_path, 'w', encoding='utf-8') as f_err:
                f_err.write(error_html)
            logger.info(f"Wrote minimal error page to {html_file_path}")
        except Exception as e_write_err:
            logger.error(f"Failed even to write the error page to {html_file_path}: {e_write_err}")
            # In this extreme case, the file might not exist or be empty, but we still return the path

    finally:
        # --- Database updates moved to finally block ---
        final_status = 'success' if report_generation_successful else 'failed'
        error_msg = error_during_generation if not report_generation_successful else None
        if db_conn:
            try:
                # Register the HTML report product regardless of success/failure (as file path is always returned)
                register_product(
                    conn=db_conn,
                    module_name=report_module_name,
                    product_type="html",
                    category="report",
                    relative_path=os.path.relpath(html_file_path, run_dir),
                    subcategory="analysis_report",
                    description=f"Analysis HTML report ({'Generated successfully' if final_status == 'success' else 'Generation failed - see content'})"
                )
                # Update the module status
                update_module_status(db_conn, report_module_name, final_status, error_message=error_msg)
                db_conn.commit() # Commit final status updates
            except Exception as e_db:
                logger.error(f"Failed to register HTML report product or update status in database: {e_db}")
            finally:
                # Close the database connection
                db_conn.close()
                logger.debug("Database connection closed.")

    # --- Always return the path ---
    return html_file_path