"""
Data aggregator component for the enhanced report generator.

This module is responsible for:
1. Discovering individual analysis_registry.db files from simulation runs
2. Extracting metrics and metadata from these databases
3. Aggregating the data into a central enhanced_cross_analysis.db database
"""

import sqlite3
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)

# Define the schema for the aggregated database
AGGREGATED_DB_SCHEMA = {
    "systems": """
        CREATE TABLE IF NOT EXISTS systems (
            system_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_name TEXT NOT NULL UNIQUE,
            group_type TEXT,
            original_db_path TEXT,
            psf_file TEXT,
            dcd_file TEXT,
            analysis_start_frame INTEGER,
            analysis_end_frame INTEGER,
            trajectory_total_frames INTEGER,
            is_control_system BOOLEAN
        )
    """,
    "aggregated_metrics": """
        CREATE TABLE IF NOT EXISTS aggregated_metrics (
            metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
            system_id INTEGER NOT NULL,
            metric_name TEXT NOT NULL,
            value REAL,
            units TEXT,
            source_module TEXT,
            FOREIGN KEY (system_id) REFERENCES systems (system_id)
        )
    """
}


class DataAggregator:
    """
    Discovers, extracts, and aggregates data from individual analysis_registry.db files
    into a central enhanced_cross_analysis.db database.
    """
    
    def __init__(self, aggregated_db_path: Path):
        """
        Initialize the DataAggregator with the path to the aggregated database.
        
        Args:
            aggregated_db_path: Path to the enhanced_cross_analysis.db file
        """
        self.aggregated_db_path = aggregated_db_path
        self.conn = None
        self.cursor = None

    def _connect_db(self):
        """Connects to the aggregated database and creates tables if they don't exist."""
        self.conn = sqlite3.connect(self.aggregated_db_path)
        self.conn.row_factory = sqlite3.Row  # Access columns by name
        self.cursor = self.conn.cursor()
        
        for table_name, schema_sql in AGGREGATED_DB_SCHEMA.items():
            try:
                self.cursor.execute(schema_sql)
            except sqlite3.Error as e:
                logger.error(f"Error creating table {table_name} in {self.aggregated_db_path}: {e}")
                raise
                
        self.conn.commit()
        logger.info(f"Successfully connected to and ensured schema in {self.aggregated_db_path}")

    def _close_db(self):
        """Commits changes and closes the database connection."""
        if self.conn:
            self.conn.commit()
            self.conn.close()
            self.conn = None
            self.cursor = None

    def discover_individual_dbs(self, root_dirs: List[Path], db_filename: str = "analysis_registry.db") -> List[Tuple[Path, str, str]]:
        """
        Discovers individual analysis_registry.db files in the provided directories.

        Args:
            root_dirs: List of root directories to search in
            db_filename: Name of the database files to look for (default: "analysis_registry.db")

        Returns:
            List of tuples (db_path, group_type, run_id_prefix) for discovered databases
        """
        db_paths = []
        for root_dir in root_dirs:
            if not root_dir.is_dir():
                logger.warning(f"Root directory {root_dir} not found or not a directory. Skipping.")
                continue

            # Determine group type from directory path
            group_type = "unknown"
            # Create a unique prefix for run IDs from this root directory to avoid conflicts
            root_prefix = ""

            root_str = str(root_dir).lower()
            if "control" in root_str:
                group_type = "toxin-free"
                root_prefix = "control_"
            elif "toxin" in root_str:
                group_type = "toxin-bound"
                root_prefix = "toxin_"

            logger.info(f"Searching for databases in {root_dir} (group_type={group_type}, prefix={root_prefix})")

            for db_path in root_dir.rglob(f"**/{db_filename}"):
                if db_path.is_file():
                    db_paths.append((db_path, group_type, root_prefix))

        logger.info(f"Discovered {len(db_paths)} individual database files.")
        return db_paths

    def _extract_metadata_from_db(self, individual_db_conn: sqlite3.Connection) -> Dict[str, Any]:
        """
        Extracts relevant metadata from a single analysis_registry.db.
        
        Args:
            individual_db_conn: Connection to the individual database
            
        Returns:
            Dictionary of metadata key-value pairs
        """
        meta_cursor = individual_db_conn.cursor()
        metadata = {}
        
        try:
            # Fields to extract from simulation_metadata table
            keys_to_extract = [
                'run_name', 'system_name', 'psf_file', 'dcd_file',
                'analysis_start_frame', 'analysis_end_frame',
                'trajectory_total_frames', 'is_control_system'
            ]
            
            for key in keys_to_extract:
                meta_cursor.execute("SELECT value FROM simulation_metadata WHERE key=?", (key,))
                row = meta_cursor.fetchone()
                
                if row:
                    # Handle data type conversions
                    if key == 'is_control_system':
                        metadata[key] = str(row['value']).lower() == 'true'
                    elif key in ['analysis_start_frame', 'analysis_end_frame', 'trajectory_total_frames']:
                        metadata[key] = int(row['value']) if row['value'] is not None else None
                    else:
                        metadata[key] = row['value']
                else:
                    metadata[key] = None
                    
            # Group type will be set based on the root directory in aggregate_data

        except sqlite3.Error as e:
            logger.error(f"Error extracting metadata: {e}")
            
        return metadata

    def _extract_metrics_from_db(self, individual_db_conn: sqlite3.Connection) -> List[Tuple[str, float, str, str]]:
        """
        Extracts metrics from a single analysis_registry.db.
        Handles different schema versions of the metrics table.
        
        Args:
            individual_db_conn: Connection to the individual database
            
        Returns:
            List of tuples (metric_name, value, units, source_module)
        """
        metrics_cursor = individual_db_conn.cursor()
        metrics_data = []
        
        # Get table schema to adapt to different database structures
        try:
            # Check if module_name column exists
            metrics_cursor.execute("PRAGMA table_info(metrics)")
            columns = {row['name'] for row in metrics_cursor.fetchall()}
            
            if 'module_name' in columns:
                # Use schema with module_name
                try:
                    metrics_cursor.execute("SELECT metric_name, value, units, module_name FROM metrics")
                    for row in metrics_cursor.fetchall():
                        metrics_data.append((row['metric_name'], row['value'], row['units'], row['module_name']))
                except sqlite3.Error as e:
                    logger.error(f"Error querying metrics with module_name: {e}")
            else:
                # Use schema without module_name
                try:
                    metrics_cursor.execute("SELECT metric_name, value, units FROM metrics")
                    for row in metrics_cursor.fetchall():
                        # Use a default module_name or derive it from metric_name prefix
                        module_name = self._derive_module_from_metric(row['metric_name'])
                        metrics_data.append((row['metric_name'], row['value'], row['units'], module_name))
                except sqlite3.Error as e:
                    logger.error(f"Error querying metrics without module_name: {e}")
                    
                    # Last resort: try minimal schema (just metric_name and value)
                    try:
                        metrics_cursor.execute("SELECT metric_name, value FROM metrics")
                        for row in metrics_cursor.fetchall():
                            module_name = self._derive_module_from_metric(row['metric_name'])
                            metrics_data.append((row['metric_name'], row['value'], "", module_name))
                    except sqlite3.Error as e2:
                        logger.error(f"Error querying minimal metrics schema: {e2}")
                        
        except sqlite3.Error as e:
            logger.error(f"Error determining metrics table schema: {e}")
            
        return metrics_data

    def _derive_module_from_metric(self, metric_name: str) -> str:
        """
        Derives a module name from a metric name prefix.
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            Derived module name
        """
        prefixes = {
            "DW_": "dw_gate_analysis",
            "Ion_": "ion_analysis",
            "Tyr_": "tyrosine_analysis",
            "COM_": "core_analysis",
            "Gyr_": "gyration_analysis",
            "IV_": "inner_vestibule_analysis",
            "Pocket_": "pocket_analysis",
            "Orient_": "orientation_contacts"
        }
        
        for prefix, module in prefixes.items():
            if metric_name.startswith(prefix):
                return module
                
        return "unknown_module"

    def aggregate_data(self, individual_db_info: List[Tuple[Path, str, str]], clear_existing_aggregated_data: bool = False):
        """
        Aggregates data from individual DBs into the central aggregated DB.

        Args:
            individual_db_info: List of tuples (db_path, group_type, run_id_prefix) for individual databases
            clear_existing_aggregated_data: Whether to clear existing data before aggregation
        """
        if not self.conn or not self.cursor:
            self._connect_db()

        if clear_existing_aggregated_data:
            logger.info("Clearing existing data from aggregated_metrics and systems tables.")
            self.cursor.execute("DELETE FROM aggregated_metrics")
            self.cursor.execute("DELETE FROM systems")
            self.conn.commit()

        # Track group counts to ensure we have data for both groups
        group_counts = {"toxin-bound": 0, "toxin-free": 0, "unknown": 0}

        for db_path, default_group_type, run_id_prefix in individual_db_info:
            logger.info(f"Processing individual database: {db_path} (group: {default_group_type}, prefix: {run_id_prefix})")

            try:
                ind_conn = sqlite3.connect(db_path)
                ind_conn.row_factory = sqlite3.Row

                # Extract metadata
                metadata = self._extract_metadata_from_db(ind_conn)
                metadata['original_db_path'] = str(db_path)

                # Set group type from the directory structure (more reliable)
                metadata['group_type'] = default_group_type

                # Update group counts
                group_type = metadata.get('group_type', 'unknown')
                group_counts[group_type] = group_counts.get(group_type, 0) + 1

                run_name = metadata.get('run_name')
                if not run_name:
                    # Use directory name as fallback for run_name
                    run_name = db_path.parent.name
                    metadata['run_name'] = run_name
                    logger.warning(f"No run_name found in metadata for {db_path}. Using directory name: {run_name}")

                # Add prefix to run_name to distinguish between toxin and control systems with same base name
                prefixed_run_name = f"{run_id_prefix}{run_name}"

                # Check if system already exists in the aggregated DB
                self.cursor.execute("SELECT system_id FROM systems WHERE run_name = ?", (prefixed_run_name,))
                system_row = self.cursor.fetchone()
                system_id = None

                system_data_tuple = (
                    prefixed_run_name, metadata.get('group_type'), str(db_path),
                    metadata.get('psf_file'), metadata.get('dcd_file'),
                    metadata.get('analysis_start_frame'), metadata.get('analysis_end_frame'),
                    metadata.get('trajectory_total_frames'), metadata.get('is_control_system')
                )

                if system_row:
                    # System already exists, get its ID
                    system_id = system_row['system_id']
                    logger.debug(f"System '{prefixed_run_name}' already exists with ID {system_id}. Metrics will be added/updated.")
                else:
                    # Insert new system
                    self.cursor.execute("""
                        INSERT INTO systems (
                            run_name, group_type, original_db_path, psf_file, dcd_file,
                            analysis_start_frame, analysis_end_frame, trajectory_total_frames,
                            is_control_system
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, system_data_tuple)
                    system_id = self.cursor.lastrowid
                    logger.info(f"Inserted new system '{prefixed_run_name}' with ID {system_id}.")

                # Extract and insert metrics
                metrics = self._extract_metrics_from_db(ind_conn)
                metrics_to_insert = []
                
                for metric_name, value, units, source_module in metrics:
                    metrics_to_insert.append((system_id, metric_name, value, units, source_module))

                if metrics_to_insert:
                    # Delete existing metrics for this system
                    self.cursor.execute("DELETE FROM aggregated_metrics WHERE system_id = ?", (system_id,))
                    
                    # Insert all metrics for this system
                    self.cursor.executemany("""
                        INSERT INTO aggregated_metrics (system_id, metric_name, value, units, source_module)
                        VALUES (?, ?, ?, ?, ?)
                    """, metrics_to_insert)
                    
                    logger.info(f"Inserted {len(metrics_to_insert)} metrics for system ID {system_id} ('{run_name}').")
                else:
                    logger.warning(f"No metrics extracted for system '{run_name}'. This may indicate a schema issue.")

                ind_conn.close()
                self.conn.commit()

            except sqlite3.Error as e:
                logger.error(f"SQLite error processing {db_path}: {e}")
            except Exception as e:
                logger.error(f"General error processing {db_path}: {e}")
                
        # Log group counts to help with debugging
        logger.info(f"Aggregated systems by group: toxin-bound={group_counts['toxin-bound']}, toxin-free={group_counts['toxin-free']}, unknown={group_counts['unknown']}")
        if group_counts['toxin-bound'] == 0 or group_counts['toxin-free'] == 0:
            logger.warning("Missing data for one or both groups. This will prevent comparative analysis.")
            
        logger.info("Data aggregation complete.")

    def run(self, root_dirs_for_discovery: List[Path], clear_existing: bool = True):
        """
        Main method to run the aggregator.

        Args:
            root_dirs_for_discovery: List of root directories to search for DBs
            clear_existing: Whether to clear existing data before aggregation
        """
        self._connect_db()

        try:
            individual_db_info = self.discover_individual_dbs(root_dirs_for_discovery)

            if individual_db_info:
                self.aggregate_data(individual_db_info, clear_existing_aggregated_data=clear_existing)
            else:
                logger.warning("No individual databases found. Aggregation skipped.")
        finally:
            self._close_db()


if __name__ == '__main__':
    # Example usage for testing
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    
    # These paths would be configured by the user
    sim_root_dirs = [
        Path("./example_sim_runs/toxin_systems"),
        Path("./example_sim_runs/control_systems")
    ]
    
    # Create and run the aggregator
    aggregator = DataAggregator(aggregated_db_path=Path("./enhanced_cross_analysis.db"))
    aggregator.run(root_dirs_for_discovery=sim_root_dirs, clear_existing=True)
    logger.info(f"Aggregated data written to {aggregator.aggregated_db_path}")