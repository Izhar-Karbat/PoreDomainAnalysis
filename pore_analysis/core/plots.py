"""
Core plotting utilities for pore analysis.

This module provides functions for loading plot configurations and retrieving
plot data from the database.
"""

import os
import json
import logging
import sqlite3
from typing import Dict, List, Optional, Any

# Set up logging
logger = logging.getLogger(__name__)

def load_plot_queries(config_path: str = "plots_dict.json") -> List[Dict[str, str]]:
    """
    Loads plot query definitions from a JSON file.
    
    Args:
        config_path (str): Path to the JSON configuration file, relative to the pore_analysis directory
                         
    Returns:
        List[Dict[str, str]]: List of plot query configurations
    """
    # Get the pore_analysis directory path
    script_dir = os.path.dirname(os.path.dirname(__file__))  # core/../ = pore_analysis/
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

def fetch_plot_blob(
    conn: sqlite3.Connection,
    product_type: str,
    category: str,
    subcategory: str,
    module_name: str,
    **kwargs
) -> Optional[bytes]:
    """
    Fetch a plot binary data from the database or file system.
    
    Args:
        conn (sqlite3.Connection): Database connection
        product_type (str): Type of product (e.g., 'png', 'svg')
        category (str): Category of the product (e.g., 'plot')
        subcategory (str): Subcategory of the product
        module_name (str): Name of the module that generated the product
        **kwargs: Additional keyword arguments (ignored)
        
    Returns:
        Optional[bytes]: Binary data of the plot, or None if not found
    """
    from pore_analysis.core.database import get_product_path
    
    try:
        # Get the relative path from the database
        relative_path = get_product_path(
            conn, product_type, category, subcategory, module_name
        )
        
        if not relative_path:
            logger.warning(
                f"No product found for {product_type}/{category}/{subcategory} from {module_name}"
            )
            return None
        
        # Determine the run directory from the database connection
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM simulation_metadata WHERE key='run_dir'")
        result = cursor.fetchone()
        
        if result and result[0]:
            run_dir = result[0]
        else:
            cursor.execute("PRAGMA database_list")
            db_path = cursor.fetchone()[2]  # Get the database file path
            run_dir = os.path.dirname(db_path)
        
        # Build the full path
        full_path = os.path.join(run_dir, relative_path)
        
        # Check if the file exists
        if not os.path.exists(full_path):
            logger.warning(f"Plot file not found at: {full_path}")
            return None
        
        # Read the binary data
        with open(full_path, 'rb') as f:
            blob_data = f.read()
        
        logger.debug(f"Successfully read plot from {full_path} ({len(blob_data)} bytes)")
        return blob_data
    
    except Exception as e:
        logger.error(f"Error fetching plot blob: {e}", exc_info=True)
        return None