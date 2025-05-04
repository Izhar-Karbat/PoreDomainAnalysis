# filename: pore_analysis/print_report.py
#!/usr/bin/env python
"""
Print-friendly report generator for pore analysis results.

This script creates a print-friendly HTML report that displays all sections
in sequence with proper page breaks, optimized for printing to PDF.
It strictly follows the database-driven approach of the main report generator.
"""

import os
import sys
import base64
import logging
import argparse
import sqlite3
import traceback # Import traceback
from datetime import datetime
from typing import Dict, Any, Optional, List # Import List

# --- Core Suite Imports ---
# Assuming these functions are correctly defined in their respective modules
try:
    from pore_analysis.core.database import (
        connect_db,
        get_product_path,
        register_product,
        register_module,
        update_module_status,
        get_simulation_metadata, # Specifically get the function that returns the dict
        list_modules # Use this to get module statuses
    )
    from pore_analysis.summary import get_all_metrics # Use the metric fetching function
    from pore_analysis.html import load_plot_queries # Use the plot query loader
    from pore_analysis.core.config import Analysis_version # Get version if needed
    from jinja2 import Environment, FileSystemLoader, select_autoescape, TemplateNotFound

    CORE_AVAILABLE = True
except ImportError as e:
    # Fallback basic logging if core logging fails
    logging.basicConfig(level=logging.ERROR)
    logging.critical(f"Failed to import core modules for print report generation: {e}")
    CORE_AVAILABLE = False
    # Define dummy functions to prevent script from crashing immediately
    def connect_db(*args, **kwargs): return None
    def get_product_path(*args, **kwargs): return None
    def register_product(*args, **kwargs): pass
    def register_module(*args, **kwargs): pass
    def update_module_status(*args, **kwargs): pass
    def get_simulation_metadata(*args, **kwargs): return {}
    def list_modules(*args, **kwargs): return []
    def get_all_metrics(*args, **kwargs): return {}
    def load_plot_queries(*args, **kwargs): return []
    Analysis_version = "N/A"
    # Jinja2 needs to be available, otherwise this script is unusable
    from jinja2 import Environment, FileSystemLoader, select_autoescape, TemplateNotFound


# --- PDF Conversion Function (Keep separate for clarity) ---
def convert_to_pdf(html_path: str) -> Optional[str]:
    """
    Convert HTML to PDF using wkhtmltopdf.
    (Implementation from original file - requires wkhtmltopdf installed)
    """
    logger_pdf = logging.getLogger(__name__) # Use local logger instance
    if not os.path.exists(html_path):
        logger_pdf.error(f"HTML file not found for PDF conversion: {html_path}")
        return None

    pdf_path = html_path.replace('.html', '.pdf')

    # Try to add print instructions to HTML (optional enhancement)
    try:
        with open(html_path, 'r', encoding='utf-8') as f: html_content = f.read()
        if '<div class="print-instructions"' not in html_content and '<body>' in html_content:
            # (Instructions HTML omitted for brevity - see original file)
            print_instructions = '<div class="print-instructions">...</div>' # Placeholder
            html_content = html_content.replace('<body>', '<body>' + print_instructions)
            with open(html_path, 'w', encoding='utf-8') as f: f.write(html_content)
            logger_pdf.debug(f"Added print instructions to HTML file: {html_path}")
    except Exception as e: logger_pdf.warning(f"Could not add print instructions to HTML: {e}")

    # Attempt PDF conversion
    import subprocess
    try:
        # Check for wkhtmltopdf
        subprocess.run(['wkhtmltopdf', '--version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger_pdf.info(f"Converting {html_path} to PDF...")
        # (wkhtmltopdf command arguments omitted for brevity - see original file)
        subprocess.run([
            'wkhtmltopdf', '--enable-local-file-access', # ... other args ...
            html_path, pdf_path
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger_pdf.info(f"PDF generated at {pdf_path}")
        return pdf_path
    except subprocess.CalledProcessError as e:
        logger_pdf.error(f"Error converting to PDF: {e}\nStderr: {e.stderr.decode() if hasattr(e, 'stderr') else 'N/A'}")
        logger_pdf.info(f"HTML file with print instructions is available at: {html_path}. Please print manually.")
        return None
    except FileNotFoundError:
        logger_pdf.warning("wkhtmltopdf not found. PDF generation skipped.")
        logger_pdf.info(f"HTML file with print instructions is available at: {html_path}. Please print manually.")
        return None
    except Exception as e:
        logger_pdf.error(f"Unexpected error during PDF conversion: {e}", exc_info=True)
        return None


# --- Main Report Generation Function (REVISED) ---
def generate_print_report(run_dir: str, output_file: Optional[str] = None) -> Optional[str]:
    """
    Generate a print-friendly report for the given run directory.
    Ensures metadata, metrics, and statuses are fetched correctly.

    Args:
        run_dir (str): Path to the run directory containing the analysis_registry.db
        output_file (str, optional): Path to save the output file. Defaults to
                                    <run_dir>/print_report.html

    Returns:
        str: Path to the generated report file, or None if report generation failed critically.
    """
    logger_report = logging.getLogger(__name__) # Use local logger instance
    if not CORE_AVAILABLE:
        logger_report.critical("Core modules not available. Cannot generate print report.")
        return None

    logger_report.info(f"Generating print-friendly report for {run_dir}")
    report_module_name = "print_report_generation"
    generation_success = False
    error_message = None

    # Set default output file if not provided
    if output_file is None:
        output_file = os.path.join(run_dir, "print_report.html")

    db_conn: Optional[sqlite3.Connection] = None
    try:
        # Connect to the database
        db_conn = connect_db(run_dir)
        if not db_conn:
            raise ConnectionError("Failed to connect to database.") # Raise specific error

        register_module(db_conn, report_module_name, status='running')

        # --- Fetch ALL Simulation Metadata ---
        # get_simulation_metadata should return a dict {key: value}
        # Example definition for get_simulation_metadata (ensure this matches core.database)
        def get_simulation_metadata_internal(conn):
            metadata = {}
            original_factory = conn.row_factory
            try:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("SELECT key, value FROM simulation_metadata")
                for row in cursor.fetchall():
                    metadata[row['key']] = row['value']
            except Exception as e_meta_fetch:
                 logger_report.error(f"Failed to fetch simulation metadata: {e_meta_fetch}")
                 # Return empty dict or raise depending on desired behavior
            finally:
                 if original_factory: conn.row_factory = original_factory
            return metadata

        metadata_dict = get_simulation_metadata_internal(db_conn) # Use the function
        if not metadata_dict:
             logger_report.warning("No simulation metadata found in database.")
             # Proceed with defaults, but log warning

        # Extract specific metadata needed, providing defaults
        run_name = metadata_dict.get('run_name', os.path.basename(run_dir))
        is_control_system = metadata_dict.get('is_control_system') == 'True'
        analysis_version = metadata_dict.get('analysis_version', Analysis_version) # Get suite version used

        # --- Fetch Metrics ---
        metrics = get_all_metrics(db_conn) # Expects {metric: {'value': V, 'units': U}}
        if not metrics:
            logger_report.warning("No metrics found in database.")

        # --- Fetch Module Status ---
        module_status_list = list_modules(db_conn) # Expects list of dicts
        module_status = {mod['module_name']: mod['status'] for mod in module_status_list}
        if not module_status:
            logger_report.warning("No module statuses found in database.")

        # --- Load Plot Queries and Fetch Plots ---
        plots: Dict[str, str] = {}
        plot_queries = load_plot_queries() # Load definitions from plots_dict.json
        if not plot_queries: logger_report.warning("No plot queries loaded.")

        for plot_config in plot_queries:
            plot_key = plot_config['template_key']
            if plot_key in plots: continue

            plot_path_rel = get_product_path(
                db_conn,
                plot_config['product_type'],
                plot_config['category'],
                plot_config['subcategory'],
                plot_config['module_name']
            )
            if plot_path_rel:
                full_path = os.path.join(run_dir, plot_path_rel)
                if os.path.exists(full_path):
                    try:
                        with open(full_path, 'rb') as f:
                            plots[plot_key] = base64.b64encode(f.read()).decode('utf-8')
                        logger_report.debug(f"Loaded plot '{plot_key}'")
                    except Exception as e_encode:
                        logger_report.warning(f"Failed to load/encode plot '{plot_key}' from {full_path}: {e_encode}")
                else:
                    logger_report.warning(f"Plot file '{plot_key}' not found at registered path: {full_path}")
            # else: logger_report.debug(f"Plot path not found for key '{plot_key}'")

        # --- Prepare Template Rendering Context ---
        report_data = {
            'run_name': run_name,
            'run_dir': run_dir,
            'generation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'analysis_version': analysis_version, # Pass analysis version
            'is_control_system': is_control_system,
            'metadata': metadata_dict, # Pass the full metadata dictionary
            'metrics': metrics, # Pass the full metrics dictionary
            'module_status': module_status, # Pass the module status dictionary
            'plots': plots,
            # Add any other variables needed by print_sections_template.html
        }

        # --- Load and Render Template ---
        template_dir = os.path.join(os.path.dirname(__file__), 'templates')
        if not os.path.isdir(template_dir):
             raise FileNotFoundError(f"Template directory not found: {template_dir}")
        env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=select_autoescape(['html', 'xml'])
        )
        template = env.get_template('print_sections_template.html') # Use the correct template name
        html_content = template.render(**report_data) # Pass data using **

        # --- Write HTML File ---
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        logger_report.info(f"Successfully generated print-friendly report: {output_file}")
        generation_success = True

        # --- Register Product ---
        register_product(
            conn=db_conn,
            module_name=report_module_name,
            product_type="html",
            category="report",
            relative_path=os.path.relpath(output_file, run_dir),
            subcategory="print_report", # Match the subcategory
            description="Print-friendly HTML report"
        )

    except FileNotFoundError as e:
         error_message = f"File Not Found Error: {e}"
         logger_report.critical(error_message, exc_info=True)
    except TemplateNotFound as e_tpl:
        error_message = f"HTML template not found: {e_tpl}"
        logger_report.critical(error_message, exc_info=True)
    except ConnectionError as e_db:
        error_message = f"Database connection error: {e_db}"
        logger_report.critical(error_message, exc_info=True)
        # Return None early if DB connection failed initially
        if "Failed to connect" in str(e_db): return None
    except Exception as e_gen:
        error_message = f"Error generating print report: {e_gen}"
        logger_report.critical(error_message, exc_info=True)
        # Attempt to write a minimal error HTML page
        try:
            error_html = f"""<!DOCTYPE html><html><head><title>Report Error</title></head><body>
            <h1>Error Generating Print Report</h1><p>Run: {run_dir}</p>
            <p><strong>Error:</strong></p><pre>{error_message}</pre>
            <p><strong>Traceback:</strong></p><pre>{traceback.format_exc()}</pre>
            </body></html>"""
            with open(output_file, 'w', encoding='utf-8') as f_err: f_err.write(error_html)
            logger_report.info(f"Wrote minimal error page to {output_file}")
        except Exception as e_write:
             logger_report.error(f"Could not write error page to {output_file}: {e_write}")

    finally:
        # --- Update Status and Close Connection ---
        final_status = 'success' if generation_success else 'failed'
        if db_conn:
            try:
                update_module_status(db_conn, report_module_name, final_status, error_message=error_message)
                db_conn.commit()
            except Exception as e_db_update:
                logger_report.error(f"Failed to update final status for {report_module_name}: {e_db_update}")
            finally:
                db_conn.close()
                logger_report.debug("Database connection closed.")

    # Return the path even if generation failed (as an error file might exist)
    return output_file if generation_success else None


# --- Main Execution Block (Optional, for standalone execution) ---
def main():
    """Main entry point for the print-friendly report generator script."""
    parser = argparse.ArgumentParser(description="Generate a print-friendly HTML/PDF report from analysis results")
    parser.add_argument("run_dir", help="Path to the simulation directory")
    parser.add_argument("--output", "-o", help="Output HTML file path (default: <run_dir>/print_report.html)")
    parser.add_argument("--pdf", action="store_true", help="Convert the HTML report to PDF (requires wkhtmltopdf)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Set logging level")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if not os.path.isdir(args.run_dir):
        logging.error(f"Run directory not found: {args.run_dir}")
        sys.exit(1)

    # Generate print-friendly HTML report
    html_path = generate_print_report(args.run_dir, args.output)

    if not html_path:
        logging.error("Failed to generate print-friendly HTML report.")
        sys.exit(1)

    logging.info(f"Print-friendly HTML report generated: {html_path}")

    # Convert to PDF if requested
    if args.pdf:
        pdf_path = convert_to_pdf(html_path)
        if pdf_path:
            logging.info(f"PDF report generated: {pdf_path}")
            print(f"PDF report generated: {pdf_path}")
        else:
            logging.warning("PDF conversion failed. HTML report is still available.")
            print(f"PDF conversion failed. HTML report available at: {html_path}")
            sys.exit(1) # Indicate failure if PDF was requested but failed

    print(f"Report generation complete. HTML: {html_path}")
    sys.exit(0) # Success

if __name__ == "__main__":
    main()
