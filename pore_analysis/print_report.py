#!/usr/bin/env python
"""
Print-friendly report generator for pore analysis results.

This script creates a print-friendly HTML report and converts it to PDF.
It reuses most of the logic from the html.py module but uses a different template
that displays all sections continuously for better printing.
"""

import os
import sys
import argparse
import logging
import subprocess
from typing import Optional

# Import HTML generation functionality
from pore_analysis.html import (
    connect_db,
    get_product_path,
    register_product,
    register_module,
    update_module_status,
    get_config_parameters,
    get_all_metrics,
    load_plot_queries,
    # Don't import generate_html_report as we'll reimplement it with our template
)

from pore_analysis.html import generate_html_report as _original_generate_html_report

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def generate_print_report(run_dir: str, summary: Optional[dict] = None) -> Optional[str]:
    """
    Generate a print-friendly report by modifying the template path.
    
    This function calls the original generate_html_report but modifies the template
    to use the print-friendly version.
    
    Args:
        run_dir (str): Path to the simulation directory
        summary (dict, optional): Analysis summary dictionary. If None, it will be loaded/calculated.
        
    Returns:
        str: Absolute path to the generated print-friendly HTML file
    """
    # Core implementation is in the original generate_html_report
    # We'll override the template path before calling it
    import pore_analysis.html as html_module
    
    # Override the template path in the module's environment
    # Save the original template
    original_template_path = 'report_template.html'
    html_module.ENV_TEMPLATE_PATH = 'print_report_template.html'
    
    # Set the output file path for the print-friendly report
    output_path = os.path.join(run_dir, "print_analysis_report.html")
    
    try:
        # Create a custom version of generate_html_report that uses our template
        # This is done by monkey-patching the html_module's internal workings
        
        # Call the implementation from html.py with our overrides
        html_path = _original_generate_html_report(run_dir, summary)
        
        # If the original function succeeded but wrote to the wrong path,
        # we need to rename the file
        if html_path and html_path != output_path:
            try:
                # Ensure the original path exists
                if os.path.exists(html_path):
                    # Read the file
                    with open(html_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Write to our desired path
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    logger.info(f"Copied print report to {output_path}")
                    
                    # Return our output path
                    return output_path
                else:
                    logger.warning(f"Original HTML path {html_path} doesn't exist")
                    return None
            except Exception as e:
                logger.error(f"Error copying HTML from {html_path} to {output_path}: {e}")
                return html_path  # Return the original path as fallback
        
        return html_path
        
    finally:
        # Restore the original template path
        html_module.ENV_TEMPLATE_PATH = original_template_path

def convert_to_pdf(html_path: str) -> Optional[str]:
    """
    Convert HTML report to PDF using wkhtmltopdf if available.
    If wkhtmltopdf is not available, provides instructions for manual conversion.
    
    Args:
        html_path (str): Path to the HTML file
        
    Returns:
        str: Path to the generated PDF file or None if conversion failed
    """
    if not html_path or not os.path.exists(html_path):
        logger.error(f"HTML file not found: {html_path}")
        return None
    
    pdf_path = html_path.replace('.html', '.pdf')
    
    # Add a message at the top of the HTML file to help with browser printing if wkhtmltopdf is not available
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
            
            # Insert after the opening body or container div
            if '<body>' in html_content:
                html_content = html_content.replace('<body>', '<body>' + print_instructions)
            elif '<div class="container">' in html_content:
                html_content = html_content.replace('<div class="container">', '<div class="container">' + print_instructions)
            
            # Write back the modified content
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"Added print instructions to HTML file: {html_path}")
    except Exception as e:
        logger.warning(f"Failed to add print instructions to HTML: {e}")
    
    # Try to find wkhtmltopdf in the system
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
            '--margin-top', '10mm',
            '--margin-bottom', '10mm',
            '--margin-left', '10mm',
            '--margin-right', '10mm',
            '--footer-center', 'Page [page] of [topage]',
            '--footer-font-size', '8',
            # Settings for embedded image and font support
            '--encoding', 'utf-8',
            '--image-quality', '100',
            '--image-dpi', '300',
            '--disable-smart-shrinking',
            '--enable-local-file-access',
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
        logger.info("To enable automatic PDF generation, install wkhtmltopdf:\n" +
                    "  - On Ubuntu/Debian: sudo apt install wkhtmltopdf\n" +
                    "  - On CentOS/RHEL: sudo yum install wkhtmltopdf\n" +
                    "  - On macOS: brew install wkhtmltopdf\n" +
                    "  - Or with conda: conda install -c conda-forge wkhtmltopdf")
        return None

def main():
    """Main entry point for the print-friendly report generator."""
    parser = argparse.ArgumentParser(description="Generate a print-friendly PDF report from analysis results")
    parser.add_argument("run_dir", help="Path to the simulation directory")
    parser.add_argument("--html-only", action="store_true", help="Only generate HTML, skip PDF conversion")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate run directory
    if not os.path.isdir(args.run_dir):
        logger.error(f"Run directory not found: {args.run_dir}")
        sys.exit(1)
    
    # Generate print-friendly HTML report
    logger.info(f"Generating print-friendly report for {args.run_dir}")
    html_path = generate_print_report(args.run_dir)
    
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