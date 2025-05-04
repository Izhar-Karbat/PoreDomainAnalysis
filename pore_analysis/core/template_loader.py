"""
Template loading utilities for pore analysis.

This module provides functions for loading Jinja2 templates from the templates directory.
"""

import os
import logging
from jinja2 import Environment, FileSystemLoader, select_autoescape, Template, TemplateNotFound

# Set up logging
logger = logging.getLogger(__name__)

def get_template(template_name: str) -> Template:
    """
    Load a Jinja2 template from the templates directory.
    
    Args:
        template_name (str): Name of the template file
        
    Returns:
        Template: Jinja2 template object
        
    Raises:
        TemplateNotFound: If the template file is not found
    """
    # Get the pore_analysis directory path
    pore_analysis_dir = os.path.dirname(os.path.dirname(__file__))  # core/../ = pore_analysis/
    templates_dir = os.path.join(pore_analysis_dir, 'templates')
    
    if not os.path.isdir(templates_dir):
        logger.error(f"Templates directory not found: {templates_dir}")
        raise TemplateNotFound(f"Templates directory not found: {templates_dir}")
    
    env = Environment(
        loader=FileSystemLoader(templates_dir),
        autoescape=select_autoescape(['html', 'xml']),
        trim_blocks=True,
        lstrip_blocks=True
    )
    
    try:
        template = env.get_template(template_name)
        logger.debug(f"Successfully loaded template: {template_name}")
        return template
    except TemplateNotFound:
        logger.error(f"Template not found: {template_name}")
        raise