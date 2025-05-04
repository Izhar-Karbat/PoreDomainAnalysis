"""
PDF conversion utilities for pore analysis reports.

This module provides functions for converting HTML reports to PDF using wkhtmltopdf.
"""

import os
import logging
import subprocess
from typing import Optional

# Set up logging
logger = logging.getLogger(__name__)

def convert_html_to_pdf(html_path: str, pdf_path: Optional[str] = None) -> str:
    """
    Convert HTML to PDF using wkhtmltopdf.
    
    Args:
        html_path (str): Path to the HTML file
        pdf_path (str, optional): Output path for the PDF file. If not provided,
                                  replaces .html extension with .pdf
    
    Returns:
        str: Path to the generated PDF file
        
    Raises:
        FileNotFoundError: If wkhtmltopdf is not installed
        subprocess.CalledProcessError: If wkhtmltopdf fails
    """
    if not os.path.exists(html_path):
        raise FileNotFoundError(f"HTML file not found: {html_path}")
    
    if pdf_path is None:
        pdf_path = os.path.splitext(html_path)[0] + ".pdf"
    
    logger.info(f"Converting {html_path} to PDF...")
    
    # Try to find wkhtmltopdf in the system
    try:
        # Check if wkhtmltopdf is available
        subprocess.run(
            ['wkhtmltopdf', '--version'],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Convert HTML to PDF
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
        
    except FileNotFoundError:
        logger.error("wkhtmltopdf not found. Please install it to enable PDF conversion.")
        raise
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Error during PDF conversion: {e}")
        logger.error(f"Error output: {e.stderr.decode() if hasattr(e, 'stderr') else 'No stderr'}")
        raise