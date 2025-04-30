"""
Summary generation module for pore analysis results.

This module creates a summary of analysis results from the database,
consolidating key metrics, statistics, and file references.
"""

import os
import json
import logging
from datetime import datetime
import sqlite3
from typing import Dict, List, Any, Optional

from pore_analysis.core.database import (
    connect_db, get_simulation_metadata, 
    list_modules, get_product_path, register_product, get_all_products
)
from pore_analysis.core.logging import setup_system_logger

logger = logging.getLogger(__name__)

def get_all_metrics(conn: sqlite3.Connection, module_name: str = None) -> Dict[str, Dict[str, Any]]:
    """
    Get all metrics with their units for a module, or all metrics if module_name is None.
    
    Args:
        conn: Database connection
        module_name: Name of the module that produced the metrics (optional)
        
    Returns:
        Dictionary mapping metric_name to {'value': value, 'units': units} dict
    """
    try:
        cursor = conn.cursor()
        
        if module_name:
            cursor.execute(
                """
                SELECT metric_name, value, units FROM metrics 
                WHERE module_id = (SELECT module_id FROM analysis_modules WHERE module_name = ?)
                """,
                (module_name,)
            )
        else:
            cursor.execute("SELECT metric_name, value, units FROM metrics")
        
        results = cursor.fetchall()
        metrics_dict = {}
        for result in results:
            metrics_dict[result['metric_name']] = {
                'value': result['value'],
                'units': result['units'] if result['units'] else 'N/A'
            }
        return metrics_dict
    except Exception as e:
        logger.error(f"Failed to get metrics with units: {e}")
        return {}

def calculate_summary(run_dir):
    """
    Calculate a summary of analysis results from the database.
    
    Args:
        run_dir (str): Path to the simulation directory
        
    Returns:
        dict: Summary dictionary with analysis results
    """
    logger.info(f"Generating analysis summary for {run_dir}")
    
    # Connect to the database
    db_conn = connect_db(run_dir)
    if db_conn is None:
        logger.error(f"Failed to connect to database for {run_dir}")
        return {}
    
    # Initialize summary dictionary
    summary = {
        'run_dir': run_dir,
        'run_name': os.path.basename(run_dir),
        'analysis_timestamp': datetime.now().isoformat(),
        'module_status': {},
        'metrics': {},
        'key_plots': {},
        'metadata': {}
    }
    
    # Get metadata from database
    for key in ['run_name', 'analysis_start_time', 'analysis_end_time', 'analysis_status',
                'psf_file', 'dcd_file']:
        value = get_simulation_metadata(db_conn, key)
        if value is not None:
            summary['metadata'][key] = value
    
    # Determine if it's a control system
    is_control = get_simulation_metadata(db_conn, 'is_control_system')
    summary['is_control_system'] = is_control == 'True' if is_control is not None else None
    
    # Get modules and their status
    modules = list_modules(db_conn)
    for module in modules:
        summary['module_status'][module['module_name']] = module['status']
    
    # Get all metrics
    metrics = get_all_metrics(db_conn)
    summary['metrics'] = metrics
    
    # Build a list of key plots
    # Core analysis plots
    g_g_plot = get_product_path(db_conn, 'png', 'plot', 'filtered_distances', module_name='core_analysis_visualization_g_g')
    if g_g_plot:
        summary['key_plots']['g_g_distances'] = g_g_plot
    
    com_plot = get_product_path(db_conn, 'png', 'plot', 'filtered_distances', module_name='core_analysis_visualization_com')
    if com_plot:
        summary['key_plots']['com_distances'] = com_plot
    
    # Add more plot paths as modules are implemented
    
    # Save summary to a JSON file
    try:
        summary_file = os.path.join(run_dir, "analysis_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=4, sort_keys=True)
        logger.info(f"Saved analysis summary to {summary_file}")
        
        # Register the summary in the database
        register_product(db_conn, "summary", "json", "summary", "analysis_summary", 
                        "analysis_summary.json", "Analysis summary JSON file")
    except Exception as e:
        logger.error(f"Failed to save analysis summary: {e}")
    
    return summary

def get_summary(run_dir):
    """
    Get the analysis summary, either by loading from a file or calculating it.
    
    Args:
        run_dir (str): Path to the simulation directory
        
    Returns:
        dict: Summary dictionary with analysis results
    """
    # Try to load existing summary first
    summary_file = os.path.join(run_dir, "analysis_summary.json")
    if os.path.exists(summary_file):
        try:
            with open(summary_file, 'r') as f:
                summary = json.load(f)
            logger.info(f"Loaded existing analysis summary from {summary_file}")
            return summary
        except Exception as e:
            logger.warning(f"Failed to load existing summary, will regenerate: {e}")
    
    # Calculate if no file exists or loading failed
    return calculate_summary(run_dir)

def generate_summary_from_database(run_dir: str, db_conn: sqlite3.Connection) -> dict:
    """
    Generate a summary of analysis results directly from the database connection.
    Does not save the summary to a file or register it.

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
        'key_plots': {},
        'metadata': {}
    }

    # Get data from database
    try:
        # Get metadata from database
        for key in ['run_name', 'analysis_start_time', 'analysis_end_time', 'analysis_status',
                    'psf_file', 'dcd_file']:
            value = get_simulation_metadata(db_conn, key)
            if value is not None:
                summary['metadata'][key] = value

        # Determine if it's a control system
        is_control = get_simulation_metadata(db_conn, 'is_control_system')
        summary['is_control_system'] = is_control == 'True' if is_control is not None else None

        # Get modules and their status
        modules = list_modules(db_conn)
        for module in modules:
            summary['module_status'][module['module_name']] = module['status']

        # Get all metrics
        metrics = get_all_metrics(db_conn)
        summary['metrics'] = metrics

        # Build a list of key plots by querying the database
        # Define potential plots and the modules/categories they belong to
        plot_keys = {
            'g_g_distances': [ # Try specific visualization module first, then base
                ('core_analysis_visualization_g_g', 'filtered_distances'),
                ('core_analysis_filtering', 'filtered_distances'), # Fallback if viz module didn't register
                ('core_analysis', 'filtered_distances') # Broader fallback
            ],
            'com_distances': [
                 ('core_analysis_visualization_com', 'filtered_distances'),
                 ('core_analysis_filtering', 'filtered_distances'),
                 ('core_analysis', 'filtered_distances')
            ],
            'com_kde': [
                ('core_analysis_visualization_com', 'kde_analysis'),
                ('core_analysis_filtering', 'kde_analysis'),
                ('core_analysis', 'kde_analysis')
            ],
            # Add expected plot keys and their potential (module_name, subcategory) pairs here
            # Example:
            # 'ion_density': [('ion_analysis_visualization', 'density_plot'), ('ion_analysis', 'density_plot')],
        }

        for plot_name, potential_sources in plot_keys.items():
            found_plot = False
            for module_name, subcategory in potential_sources:
                # Attempt to find the plot using this combination
                plot_path = get_product_path(db_conn, product_type='png', category='plot',
                                             subcategory=subcategory, module_name=module_name)
                if plot_path:
                    summary['key_plots'][plot_name] = plot_path
                    logger.debug(f"Found plot '{plot_name}' from module '{module_name}', subcategory '{subcategory}'")
                    found_plot = True
                    break # Stop searching once found
            if not found_plot:
                 logger.debug(f"Could not find registered plot for '{plot_name}'")

        logger.debug(f"Generated summary from database for {run_dir}")
        return summary

    except Exception as e:
        logger.error(f"Error generating summary from database for {run_dir}: {e}", exc_info=True)
        return {}