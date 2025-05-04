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
from datetime import datetime
from typing import Dict, Any, Optional

# Import necessary functions from the database module
from pore_analysis.core.database import (
    connect_db,
    get_product_path,
    register_product,
    register_module,
    update_module_status,
    get_simulation_metadata
)

# Import JSON loading function for plot configuration
from pore_analysis.html import load_plot_queries

# Set up logging
logger = logging.getLogger(__name__)

def generate_print_report(run_dir: str, output_file: Optional[str] = None) -> Optional[str]:
    """
    Generate a print-friendly report for the given run directory.
    
    Args:
        run_dir (str): Path to the run directory containing the analysis_registry.db
        output_file (str, optional): Path to save the output file. If not provided,
                                    defaults to <run_dir>/print_report.html
    
    Returns:
        str: Path to the generated report file, or None if report generation failed
    """
    logger.info(f"Generating print-friendly report for {run_dir}")
    report_module_name = "print_report_generation"
    
    # Set default output file if not provided
    if output_file is None:
        output_file = os.path.join(run_dir, "print_report.html")
    
    # Connect to the database
    db_conn = connect_db(run_dir)
    if not db_conn:
        logger.error(f"Failed to connect to database for print report generation: {run_dir}")
        return None
    
    try:
        # Register the module as running
        register_module(db_conn, report_module_name, status='running')
        
        # Get simulation metadata
        run_name = get_simulation_metadata(db_conn, 'run_name')
        if not run_name:
            run_name = os.path.basename(run_dir)
        
        is_control_system = get_simulation_metadata(db_conn, 'is_control_system') == 'True'
        
        # Load plot queries from configuration file
        plot_queries = load_plot_queries()
        if not plot_queries:
            logger.warning("No plot queries loaded from config. Report may be missing plots.")
        
        # Retrieve plot paths and convert to base64
        plots: Dict[str, str] = {}
        
        for plot_config in plot_queries:
            plot_key = plot_config['template_key']
            # Skip if already loaded
            if plot_key in plots:
                continue
                
            product_type = plot_config['product_type']
            category = plot_config['category']
            subcategory = plot_config['subcategory']
            module_name = plot_config['module_name']
            
            # Get the plot path from the database
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
                    except Exception as e:
                        logger.warning(f"Failed to load/encode plot '{plot_key}' from {full_path}: {e}")
                else:
                    logger.warning(f"Plot file '{plot_key}' not found at registered path: {full_path}")
        
        # Fetch metrics with units
        metrics = {}
        try:
            cursor = db_conn.cursor()
            cursor.execute("SELECT metric_name, value, units FROM metrics")
            for row in cursor.fetchall():
                try:
                    metric_name = row[0]
                    value = row[1]
                    units = row[2] if row[2] is not None else ''
                    metrics[metric_name] = {'value': value, 'units': units}
                except Exception as e:
                    logger.warning(f"Error processing metric: {e}")
        except Exception as e:
            logger.error(f"Failed to fetch metrics: {e}")
        
        # Get module status information
        module_status = {}
        try:
            cursor = db_conn.cursor()
            cursor.execute("SELECT module_name, status FROM analysis_modules")
            for row in cursor.fetchall():
                module_status[row[0]] = row[1]
        except Exception as e:
            logger.error(f"Failed to fetch module status: {e}")
        
        # Load print report template
        from jinja2 import Environment, FileSystemLoader, select_autoescape
        
        template_dir = os.path.join(os.path.dirname(__file__), 'templates')
        env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=select_autoescape(['html', 'xml'])
        )
        
        template = env.get_template('print_sections_template.html')
        
        # Prepare data for the template
        report_data = {
            'run_name': run_name,
            'run_dir': run_dir,
            'generation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'is_control_system': is_control_system,
            'plots': plots,
            'metrics': metrics,
            'module_status': module_status
        }
        
        # Render the template
        html_content = template.render(**report_data)
        
        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # Register the product in the database
        register_product(
            conn=db_conn,
            module_name=report_module_name,
            product_type="html",
            category="report",
            relative_path=os.path.relpath(output_file, run_dir),
            subcategory="print_report",
            description="Print-friendly HTML report"
        )
        
        # Update module status
        update_module_status(db_conn, report_module_name, 'success')
        
        logger.info(f"Successfully generated print-friendly report: {output_file}")
        return output_file
        
    except Exception as e:
        logger.error(f"Error generating print-friendly report: {e}", exc_info=True)
        update_module_status(db_conn, report_module_name, 'failed', error_message=str(e))
        return None
        
    finally:
        if db_conn:
            db_conn.close()

def convert_to_pdf(html_path: str) -> Optional[str]:
    """
    Convert HTML to PDF using wkhtmltopdf.
    
    Args:
        html_path (str): Path to the HTML file
        
    Returns:
        str: Path to the generated PDF file or None if conversion failed
    """
    if not os.path.exists(html_path):
        logger.error(f"HTML file not found: {html_path}")
        return None
    
    pdf_path = html_path.replace('.html', '.pdf')
    
    # Add a message at the top of the HTML file to help with browser printing
    try:
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Add print instructions after the body tag if not already present
        if '<div class="print-instructions"' not in html_content:
            print_instructions = '''
            <div class="print-instructions" style="background-color: #f0f8ff; padding: 10px; margin: 10px 0; border-left: 5px solid #0056b3; display: block;">
                <h3>Print Instructions</h3>
                <p>This is a print-friendly version of the report. To save as PDF:</p>
                <ol>
                    <li>Open this HTML file in Chrome or another modern browser</li>
                    <li>Press Ctrl+P (or Cmd+P on Mac) to open the print dialog</li>
                    <li>Change the destination to "Save as PDF"</li>
                    <li>Set layout to "Portrait"</li>
                    <li>Set margins to "Default" or "None"</li>
                    <li>Click "Save" or "Print"</li>
                </ol>
                <p><em>This message will not appear in the printed/PDF version.</em></p>
                <style media="print">
                    .print-instructions { display: none !important; }
                </style>
            </div>
            '''
            
            # Insert after the opening body tag
            if '<body>' in html_content:
                html_content = html_content.replace('<body>', '<body>' + print_instructions)
            
            # Write back the modified content
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"Added print instructions to HTML file: {html_path}")
    except Exception as e:
        logger.warning(f"Failed to add print instructions to HTML: {e}")
    
    # Try to find wkhtmltopdf in the system
    import subprocess
    
    try:
        # Check if wkhtmltopdf is available
        subprocess.run(['wkhtmltopdf', '--version'], 
                       check=True, 
                       stdout=subprocess.PIPE, 
                       stderr=subprocess.PIPE)
        
        # Convert HTML to PDF
        logger.info(f"Converting {html_path} to PDF...")
        result = subprocess.run([
            'wkhtmltopdf',
            '--enable-local-file-access',
            '--page-size', 'A4',
            '--margin-top', '15mm',
            '--margin-bottom', '15mm',
            '--margin-left', '15mm',
            '--margin-right', '15mm',
            '--footer-center', 'Page [page] of [topage]',
            '--footer-font-size', '8',
            # Settings for embedded image and font support
            '--encoding', 'utf-8',
            '--image-quality', '100',
            '--image-dpi', '300',
            '--disable-smart-shrinking',
            '--print-media-type',
            html_path, pdf_path
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        logger.info(f"PDF generated at {pdf_path}")
        return pdf_path
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Error converting to PDF: {e}")
        logger.error(f"Error output: {e.stderr.decode() if hasattr(e, 'stderr') else 'No stderr'}")
        logger.info(f"HTML file with print instructions is available at: {html_path}")
        logger.info("Please open this HTML file in a browser and use the print function to save as PDF manually.")
        return None
        
    except FileNotFoundError:
        logger.warning("wkhtmltopdf not found. PDF generation skipped.")
        logger.info(f"HTML file with print instructions is available at: {html_path}")
        logger.info("Please open this HTML file in a browser and use the print function to save as PDF manually.")
        logger.info("To enable automatic PDF generation, install wkhtmltopdf")
        return None

def main():
    """Main entry point for the print-friendly report generator."""
    parser = argparse.ArgumentParser(description="Generate a print-friendly PDF report from analysis results")
    parser.add_argument("run_dir", help="Path to the simulation directory")
    parser.add_argument("--output", "-o", help="Output file path (default: <run_dir>/print_report.html)")
    parser.add_argument("--html-only", action="store_true", help="Only generate HTML, skip PDF conversion")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Validate run directory
    if not os.path.isdir(args.run_dir):
        logger.error(f"Run directory not found: {args.run_dir}")
        sys.exit(1)
    
    # Generate print-friendly HTML report
    html_path = generate_print_report(args.run_dir, args.output)
    
    if not html_path:
        logger.error("Failed to generate HTML report")
        sys.exit(1)
    
    logger.info(f"Print-friendly HTML report generated: {html_path}")
    
    # Convert to PDF if requested
    if not args.html_only:
        pdf_path = convert_to_pdf(html_path)
        if pdf_path:
            logger.info(f"PDF report generated: {pdf_path}")
            print(f"PDF report generated: {pdf_path}")
        else:
            logger.warning("PDF conversion failed. HTML report is still available.")
            print(f"PDF conversion failed. HTML report is available at: {html_path}")
    else:
        logger.info("Skipping PDF conversion as requested")
        print(f"HTML report generated: {html_path}")
    
    # Success
    sys.exit(0)

if __name__ == "__main__":
    main()