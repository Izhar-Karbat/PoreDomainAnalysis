"""
Data extraction module for cross-analysis suite.

This module handles scanning for system directories, connecting to their databases,
and extracting metrics for comparative analysis.
"""

import os
import sqlite3
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import glob

# Configure logging
logger = logging.getLogger(__name__)

class MetricsExtractor:
    """
    Extracts metrics from individual system databases for cross-system analysis.
    """
    
    def __init__(self, toxin_dir: str, control_dir: str, meta_db_path: str):
        """
        Initialize the metrics extractor.
        
        Args:
            toxin_dir: Directory containing toxin systems
            control_dir: Directory containing control systems
            meta_db_path: Path to the meta-database file
        """
        self.toxin_dir = os.path.abspath(toxin_dir)
        self.control_dir = os.path.abspath(control_dir)
        self.meta_db_path = meta_db_path
        self.meta_conn = None
        self.systems_info = []
        
    def connect_meta_db(self) -> bool:
        """
        Connects to the meta-database, creating it if it doesn't exist.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.meta_db_path), exist_ok=True)
            
            # Connect to database
            self.meta_conn = sqlite3.connect(self.meta_db_path)
            self.meta_conn.row_factory = sqlite3.Row
            
            # Create tables if they don't exist
            self._create_meta_db_schema()
            
            logger.info(f"Connected to meta-database: {self.meta_db_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to meta-database: {e}")
            return False
            
    def _create_meta_db_schema(self):
        """Creates the meta-database schema if it doesn't exist."""
        cursor = self.meta_conn.cursor()
        
        # Create systems table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS systems (
            system_id INTEGER PRIMARY KEY,
            system_name TEXT NOT NULL,
            system_type TEXT NOT NULL,
            run_name TEXT NOT NULL,
            db_path TEXT NOT NULL,
            analysis_status TEXT,
            frame_count INTEGER,
            time_ns REAL,
            is_included BOOLEAN DEFAULT 1,
            metadata TEXT
        )
        ''')
        
        # Create aggregated_metrics table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS aggregated_metrics (
            metric_id INTEGER PRIMARY KEY,
            system_id INTEGER,
            metric_name TEXT NOT NULL,
            metric_category TEXT,
            value REAL,
            units TEXT,
            FOREIGN KEY (system_id) REFERENCES systems(system_id)
        )
        ''')
        
        # Create comparative_analysis table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS comparative_analysis (
            analysis_id INTEGER PRIMARY KEY,
            analysis_name TEXT NOT NULL,
            analysis_type TEXT NOT NULL,
            p_value REAL,
            effect_size REAL,
            details TEXT,
            timestamp TEXT,
            parameters TEXT
        )
        ''')
        
        # Create comparative_plots table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS comparative_plots (
            plot_id INTEGER PRIMARY KEY,
            plot_name TEXT NOT NULL,
            plot_type TEXT NOT NULL,
            category TEXT NOT NULL,
            subcategory TEXT,
            relative_path TEXT NOT NULL,
            analysis_id INTEGER,
            parameters TEXT,
            timestamp TEXT,
            FOREIGN KEY (analysis_id) REFERENCES comparative_analysis(analysis_id)
        )
        ''')
        
        # Create ai_insights table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS ai_insights (
            insight_id INTEGER PRIMARY KEY,
            timestamp TEXT NOT NULL,
            insight_type TEXT NOT NULL,
            description TEXT NOT NULL,
            confidence REAL,
            metrics_involved TEXT,
            systems_involved TEXT,
            related_plots TEXT,
            analysis_ids TEXT,
            is_validated BOOLEAN DEFAULT 0,
            validator_notes TEXT
        )
        ''')
        
        self.meta_conn.commit()
    
    def scan_systems(self) -> List[Dict[str, Any]]:
        """
        Scans the directories for toxin and control systems.
        
        Returns:
            List of dictionaries containing system information
        """
        systems_info = []
        
        # Function to recursively search for database files
        def find_db_files(base_dir, system_type):
            if not os.path.exists(base_dir):
                logger.error(f"Directory does not exist: {base_dir}")
                return []
            
            found_systems = []
            
            # Walk through directory tree
            for root, dirs, files in os.walk(base_dir):
                # Check if analysis_registry.db exists in this directory
                db_path = os.path.join(root, 'analysis_registry.db')
                if os.path.exists(db_path):
                    # Get relative path components for system name
                    rel_path = os.path.relpath(root, base_dir)
                    components = rel_path.split(os.sep)
                    
                    # Use last directory as run name
                    run_name = os.path.basename(root)
                    
                    # Construct system name
                    if len(components) > 1:
                        system_name = os.path.join(system_type, *components)
                    else:
                        system_name = os.path.join(system_type, run_name)
                    
                    found_systems.append({
                        'system_name': system_name,
                        'system_type': system_type,
                        'run_name': run_name,
                        'db_path': db_path
                    })
            
            return found_systems
        
        # Find toxin systems
        toxin_systems = find_db_files(self.toxin_dir, 'toxin')
        systems_info.extend(toxin_systems)
        
        # Find control systems
        control_systems = find_db_files(self.control_dir, 'control')
        systems_info.extend(control_systems)
        
        self.systems_info = systems_info
        logger.info(f"Found {len(systems_info)} systems ({len(toxin_systems)} toxin, {len(control_systems)} control)")
        return systems_info
    
    def register_systems(self) -> int:
        """
        Registers systems in the meta-database.
        
        Returns:
            int: Number of systems registered
        """
        if not self.meta_conn:
            logger.error("Meta-database not connected")
            return 0
        
        if not self.systems_info:
            logger.warning("No systems scanned yet")
            return 0
        
        cursor = self.meta_conn.cursor()
        count = 0
        
        for system in self.systems_info:
            # Check if system already exists
            cursor.execute(
                "SELECT system_id FROM systems WHERE system_name=?", 
                (system['system_name'],)
            )
            result = cursor.fetchone()
            
            if result:
                # Update existing system
                system_id = result[0]
                cursor.execute(
                    """UPDATE systems SET 
                       system_type=?, run_name=?, db_path=?, 
                       is_included=1
                       WHERE system_id=?""",
                    (system['system_type'], system['run_name'], 
                     system['db_path'], system_id)
                )
                logger.debug(f"Updated existing system: {system['system_name']}")
            else:
                # Insert new system
                cursor.execute(
                    """INSERT INTO systems 
                       (system_name, system_type, run_name, db_path, is_included) 
                       VALUES (?, ?, ?, ?, 1)""",
                    (system['system_name'], system['system_type'], 
                     system['run_name'], system['db_path'])
                )
                system_id = cursor.lastrowid
                logger.debug(f"Registered new system: {system['system_name']}")
            
            # Update system metadata from its database
            self._update_system_metadata(system_id, system['db_path'])
            count += 1
        
        self.meta_conn.commit()
        logger.info(f"Registered {count} systems in meta-database")
        return count
    
    def _update_system_metadata(self, system_id: int, db_path: str):
        """
        Updates system metadata from its database.
        
        Args:
            system_id: ID of the system in the meta-database
            db_path: Path to the system's database
        """
        try:
            # Connect to the system database
            sys_conn = sqlite3.connect(db_path)
            sys_conn.row_factory = sqlite3.Row
            cursor = sys_conn.cursor()
            
            # Get metadata
            cursor.execute("SELECT key, value FROM simulation_metadata")
            rows = cursor.fetchall()
            metadata = {row['key']: row['value'] for row in rows}
            
            # Get frame count and time
            frame_count = int(metadata.get('total_frames', 0))
            frames_per_ns = float(metadata.get('frames_per_ns', 10.0))
            time_ns = frame_count / frames_per_ns if frames_per_ns > 0 else 0
            
            # Update in meta-database
            meta_cursor = self.meta_conn.cursor()
            meta_cursor.execute(
                """UPDATE systems SET 
                   analysis_status=?, frame_count=?, time_ns=?, metadata=?
                   WHERE system_id=?""",
                (metadata.get('analysis_status', 'unknown'), 
                 frame_count, time_ns, json.dumps(metadata), system_id)
            )
            
            sys_conn.close()
            logger.debug(f"Updated metadata for system_id={system_id}")
        except Exception as e:
            logger.error(f"Failed to update metadata for system_id={system_id}: {e}")
    
    def extract_metrics(self, metric_categories: Optional[List[str]] = None) -> int:
        """
        Extracts metrics from all registered systems.
        
        Args:
            metric_categories: Optional list of metric categories to extract
                              (if None, extracts all metrics)
        
        Returns:
            int: Number of metrics extracted
        """
        if not self.meta_conn:
            logger.error("Meta-database not connected")
            return 0
        
        cursor = self.meta_conn.cursor()
        cursor.execute("SELECT system_id, system_name, db_path FROM systems WHERE is_included=1")
        systems = cursor.fetchall()
        
        if not systems:
            logger.warning("No systems registered or all are excluded")
            return 0
        
        # Clear existing metrics if we're doing a full extraction
        if not metric_categories:
            cursor.execute("DELETE FROM aggregated_metrics")
            logger.info("Cleared existing metrics for full extraction")
        
        total_metrics = 0
        for system in systems:
            system_id, system_name, db_path = system['system_id'], system['system_name'], system['db_path']
            
            try:
                metrics_count = self._extract_system_metrics(system_id, db_path, metric_categories)
                total_metrics += metrics_count
                logger.info(f"Extracted {metrics_count} metrics from {system_name}")
            except Exception as e:
                logger.error(f"Failed to extract metrics from {system_name}: {e}")
        
        self.meta_conn.commit()
        logger.info(f"Extracted a total of {total_metrics} metrics")
        return total_metrics
    
    def _extract_system_metrics(self, system_id: int, db_path: str, 
                               metric_categories: Optional[List[str]] = None) -> int:
        """
        Extracts metrics from a single system database.
        
        Args:
            system_id: ID of the system in the meta-database
            db_path: Path to the system's database
            metric_categories: Optional list of metric categories to extract
        
        Returns:
            int: Number of metrics extracted
        """
        # Connect to the system database
        sys_conn = sqlite3.connect(db_path)
        sys_conn.row_factory = sqlite3.Row
        sys_cursor = sys_conn.cursor()
        
        # Get metrics
        if metric_categories:
            placeholders = ', '.join(['?'] * len(metric_categories))
            query = f"""
                SELECT m.metric_name, m.value, m.units, am.module_name 
                FROM metrics m
                JOIN analysis_modules am ON m.module_id = am.module_id
                WHERE am.module_name IN ({placeholders})
            """
            sys_cursor.execute(query, metric_categories)
        else:
            sys_cursor.execute("""
                SELECT m.metric_name, m.value, m.units, am.module_name 
                FROM metrics m
                JOIN analysis_modules am ON m.module_id = am.module_id
            """)
        
        metrics = sys_cursor.fetchall()
        sys_conn.close()
        
        # Map module names to categories
        category_mapping = {
            'core_analysis': 'structure',
            'core_analysis_filtering': 'structure',
            'orientation_analysis': 'toxin',
            'ion_analysis': 'ion',
            'inner_vestibule_analysis': 'water',
            'gyration_analysis': 'carbonyl',
            'tyrosine_analysis': 'tyrosine',
            'dw_gate_analysis': 'dw_gate',
            'pocket_analysis': 'pocket'
        }
        
        # Insert metrics into meta-database
        meta_cursor = self.meta_conn.cursor()
        
        # If specific categories are provided, delete only those categories for this system
        if metric_categories:
            for category in metric_categories:
                meta_cursor.execute(
                    "DELETE FROM aggregated_metrics WHERE system_id=? AND metric_category=?",
                    (system_id, category_mapping.get(category, category))
                )
        else:
            # Delete all metrics for this system if doing a full extraction
            meta_cursor.execute(
                "DELETE FROM aggregated_metrics WHERE system_id=?", 
                (system_id,)
            )
        
        # Insert new metrics
        for metric in metrics:
            metric_name = metric['metric_name']
            module_name = metric['module_name']
            category = category_mapping.get(module_name, module_name.split('_')[0])
            
            meta_cursor.execute(
                """INSERT INTO aggregated_metrics 
                   (system_id, metric_name, metric_category, value, units) 
                   VALUES (?, ?, ?, ?, ?)""",
                (system_id, metric_name, category, metric['value'], metric['units'])
            )
        
        return len(metrics)
    
    def get_metrics_dataframe(self, metric_names: Optional[List[str]] = None, 
                             categories: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Retrieves metrics as a pandas DataFrame for analysis.
        
        Args:
            metric_names: Optional list of specific metric names to include
            categories: Optional list of metric categories to include
        
        Returns:
            pd.DataFrame: DataFrame with metrics data
        """
        if not self.meta_conn:
            logger.error("Meta-database not connected")
            return pd.DataFrame()
        
        cursor = self.meta_conn.cursor()
        
        # Build the query based on parameters
        query = """
            SELECT s.system_id, s.system_name, s.system_type, s.run_name, 
                   m.metric_name, m.metric_category, m.value, m.units
            FROM systems s
            JOIN aggregated_metrics m ON s.system_id = m.system_id
            WHERE s.is_included = 1
        """
        params = []
        
        if metric_names:
            placeholders = ', '.join(['?'] * len(metric_names))
            query += f" AND m.metric_name IN ({placeholders})"
            params.extend(metric_names)
        
        if categories:
            placeholders = ', '.join(['?'] * len(categories))
            query += f" AND m.metric_category IN ({placeholders})"
            params.extend(categories)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        # Convert to DataFrame
        data = []
        for row in rows:
            data.append({
                'system_id': row['system_id'],
                'system_name': row['system_name'],
                'system_type': row['system_type'],
                'run_name': row['run_name'],
                'metric_name': row['metric_name'],
                'category': row['metric_category'],
                'value': row['value'],
                'units': row['units']
            })
        
        df = pd.DataFrame(data)
        
        # Pivot to get systems as rows and metrics as columns
        if not df.empty:
            pivot_df = df.pivot_table(
                index=['system_id', 'system_name', 'system_type', 'run_name'],
                columns=['metric_name'], 
                values='value',
                aggfunc='first'
            ).reset_index()
            
            # Add units as a separate dataframe or attribute
            units_dict = df.groupby('metric_name')['units'].first().to_dict()
            pivot_df.attrs['units'] = units_dict
            
            return pivot_df
        
        return df
    
    def close(self):
        """Closes the database connection."""
        if self.meta_conn:
            self.meta_conn.close()
            logger.info("Closed meta-database connection")