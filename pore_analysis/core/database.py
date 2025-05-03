# pore_analysis/core/database.py

import os
import json
import sqlite3
import logging
import re # For parsing config file
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union # Added Optional, Tuple, Union, Any

logger = logging.getLogger(__name__)

# Database schema version
DB_SCHEMA_VERSION = "1.0.0" # Make sure this is defined or imported

# --- Helper Functions ---

def dict_factory(cursor, row):
    """
    Convert SQLite row to dictionary for easier access.
    Used as row_factory for connections.
    """
    # Check if cursor.description is available
    if cursor.description:
        return {col[0]: row[idx] for idx, col in enumerate(cursor.description)}
    else:
        # Fallback or error handling if description is not available
        # This might happen for certain PRAGMA statements or if the query doesn't return rows
        # For simplicity, return an empty dict or handle as appropriate for your needs
        # If rows are simple tuples without named columns, you might return the tuple directly
        # return row # Or handle differently
        return {}


def get_db_path(run_dir: str) -> str:
    """Get the path to the SQLite database file for a simulation."""
    return os.path.join(run_dir, "analysis_registry.db")

def connect_db(run_dir: str) -> Optional[sqlite3.Connection]:
    """Connect to the SQLite database for a simulation."""
    db_path = get_db_path(run_dir)
    try:
        conn = sqlite3.connect(db_path, timeout=10) # Added timeout
        # Use dict_factory for easier row access by column name
        conn.row_factory = dict_factory
        conn.execute("PRAGMA foreign_keys = ON;") # Ensure foreign keys are enabled
        return conn
    except sqlite3.Error as e:
        logger.error(f"Failed to connect to database at {db_path}: {e}")
        return None

# --- Database Initialization ---

def init_db(run_dir: str, force_recreate: bool = False) -> Optional[sqlite3.Connection]:
    """
    Initialize the database for a simulation. Creates tables if they don't exist.

    Args:
        run_dir: The simulation directory path
        force_recreate: If True, drop existing tables and recreate

    Returns:
        SQLite connection object or None on failure.
    """
    db_path = get_db_path(run_dir)

    # Check if database exists and force_recreate is False
    if os.path.exists(db_path) and not force_recreate:
        logger.info(f"Database already exists at {db_path}. Connecting to existing database.")
        conn = connect_db(run_dir)
        if conn is None: return None # Connection failed

        # Check schema version
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT value FROM simulation_metadata WHERE key = 'schema_version'")
            result = cursor.fetchone()
            if result and result.get('value') != DB_SCHEMA_VERSION: # Use .get() for safety
                logger.warning(f"Database schema version mismatch. Expected {DB_SCHEMA_VERSION}, found {result.get('value')}.")
            elif not result:
                 logger.warning("Schema version not found in existing database.")
        except sqlite3.Error as e: # Catch specific sqlite errors
            logger.warning(f"Could not verify schema version (DB Error: {e}). Database may be corrupted or from an older version.")
        except Exception as e:
             logger.warning(f"Error checking schema version: {e}")


        # Check if config_parameters table exists and has the description column
        try:
            cursor.execute("PRAGMA table_info(config_parameters)")
            columns = [col['name'] for col in cursor.fetchall()]
            if not columns: # Table doesn't exist yet
                 logger.info("'config_parameters' table not found in existing database. It will be created if needed by store_config_parameters.")
            elif 'param_description' not in columns:
                logger.warning("Existing 'config_parameters' table is missing 'param_description' column. Consider re-initializing with --reinit-db if needed.")
        except sqlite3.Error as e: # Catch specific sqlite errors
             logger.warning(f"'config_parameters' table check failed (DB Error: {e}). It may not exist yet.")
        except Exception as e:
             logger.warning(f"Error checking config_parameters table structure: {e}")

        return conn

    # Create new database
    if os.path.exists(db_path) and force_recreate:
        logger.warning(f"Recreating database at {db_path}")
        try:
            # Ensure connection is closed before removing if it exists
            temp_conn = sqlite3.connect(db_path)
            temp_conn.close()
            os.remove(db_path)
        except sqlite3.Error as e_con:
             logger.warning(f"Could not close connection before removing DB: {e_con}")
             # Attempt removal anyway
             try: os.remove(db_path)
             except OSError as e_rem: logger.error(f"Failed to remove existing database: {e_rem}")
        except OSError as e:
            logger.error(f"Failed to remove existing database: {e}")
            # Continue anyway, will try to connect and recreate tables

    conn = connect_db(run_dir)
    if conn is None: return None # Connection failed

    cursor = conn.cursor()

    # Create tables
    try:
        cursor.executescript("""
        PRAGMA foreign_keys = ON; -- Recommended to enforce foreign key constraints

        -- simulation_metadata table
        CREATE TABLE IF NOT EXISTS simulation_metadata (
            metadata_id INTEGER PRIMARY KEY,
            key TEXT NOT NULL UNIQUE, -- Added UNIQUE constraint
            value TEXT
        );

        -- analysis_modules table
        CREATE TABLE IF NOT EXISTS analysis_modules (
            module_id INTEGER PRIMARY KEY,
            module_name TEXT NOT NULL UNIQUE, -- Added UNIQUE constraint
            status TEXT NOT NULL,
            execution_time REAL,
            start_timestamp TEXT,
            end_timestamp TEXT,
            parameters TEXT, -- Storing parameters as JSON string
            error_message TEXT
        );

        -- analysis_products table
        CREATE TABLE IF NOT EXISTS analysis_products (
            product_id INTEGER PRIMARY KEY,
            module_id INTEGER NOT NULL,
            product_type TEXT NOT NULL,
            category TEXT NOT NULL,
            subcategory TEXT,
            relative_path TEXT NOT NULL UNIQUE, -- Added UNIQUE constraint
            description TEXT,
            generation_timestamp TEXT,
            metadata TEXT, -- Storing metadata as JSON string
            FOREIGN KEY(module_id) REFERENCES analysis_modules(module_id) ON DELETE CASCADE -- Added ON DELETE CASCADE
        );

        -- metrics table
        CREATE TABLE IF NOT EXISTS metrics (
            metric_id INTEGER PRIMARY KEY,
            module_id INTEGER NOT NULL,
            metric_name TEXT NOT NULL,
            value REAL, -- Storing numerical values directly
            units TEXT,
            description TEXT,
            FOREIGN KEY(module_id) REFERENCES analysis_modules(module_id) ON DELETE CASCADE, -- Added ON DELETE CASCADE
            UNIQUE(module_id, metric_name) -- Ensure metric name is unique per module
        );

        -- config_parameters table (NEW)
        CREATE TABLE IF NOT EXISTS config_parameters (
            param_id INTEGER PRIMARY KEY,
            param_name TEXT NOT NULL UNIQUE, -- Parameter name from config.py
            param_value TEXT, -- Value stored as string (or JSON string for complex types)
            param_type TEXT, -- Python type name (e.g., 'int', 'float', 'str', 'list')
            param_description TEXT -- Description parsed from comments
        );

        -- dependencies table
        CREATE TABLE IF NOT EXISTS dependencies (
            dependency_id INTEGER PRIMARY KEY,
            dependent_module_id INTEGER NOT NULL,
            required_module_id INTEGER NOT NULL,
            FOREIGN KEY(dependent_module_id) REFERENCES analysis_modules(module_id) ON DELETE CASCADE,
            FOREIGN KEY(required_module_id) REFERENCES analysis_modules(module_id) ON DELETE CASCADE,
            UNIQUE(dependent_module_id, required_module_id)
        );

        -- product_relationships table
        CREATE TABLE IF NOT EXISTS product_relationships (
            relationship_id INTEGER PRIMARY KEY,
            source_product_id INTEGER NOT NULL,
            derived_product_id INTEGER NOT NULL,
            relationship_type TEXT NOT NULL,
            FOREIGN KEY(source_product_id) REFERENCES analysis_products(product_id) ON DELETE CASCADE,
            FOREIGN KEY(derived_product_id) REFERENCES analysis_products(product_id) ON DELETE CASCADE,
            UNIQUE(source_product_id, derived_product_id, relationship_type)
        );

        -- Create indices for faster lookups
        CREATE INDEX IF NOT EXISTS idx_products_by_module ON analysis_products(module_id);
        CREATE INDEX IF NOT EXISTS idx_products_by_type ON analysis_products(product_type, category, subcategory); -- Added subcategory
        CREATE INDEX IF NOT EXISTS idx_metrics_by_module ON metrics(module_id);
        CREATE INDEX IF NOT EXISTS idx_metrics_by_name ON metrics(metric_name);
        """)

        # Store schema version
        cursor.execute(
            "INSERT OR REPLACE INTO simulation_metadata (key, value) VALUES (?, ?)",
            ("schema_version", DB_SCHEMA_VERSION)
        )

        # Store initialization timestamp
        cursor.execute(
            "INSERT OR REPLACE INTO simulation_metadata (key, value) VALUES (?, ?)",
            ("db_init_timestamp", datetime.now().isoformat())
        )

        conn.commit()
        logger.info(f"Initialized database at {db_path} with schema version {DB_SCHEMA_VERSION}")
        return conn

    except sqlite3.Error as e:
        logger.error(f"Failed to execute schema script: {e}")
        conn.rollback() # Roll back changes on error
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during DB initialization: {e}")
        conn.rollback()
        return None

# --- Config Parameter Handling ---

def store_config_parameters(conn: sqlite3.Connection, config_module: Any):
    """
    Parses the config module's source file to extract parameter names,
    values, types, and preceding comments (as descriptions), storing
    them in the config_parameters table.

    Args:
        conn: Active database connection.
        config_module: The imported config module object (e.g., from `import pore_analysis.core.config`).
    """
    if not hasattr(config_module, '__file__') or not config_module.__file__:
         logger.error("Cannot parse config module: __file__ attribute not found.")
         return

    logger.info(f"Attempting to parse and store parameters from {config_module.__file__}")
    params_to_store = []
    try:
        config_filepath = config_module.__file__
        if not os.path.exists(config_filepath):
            logger.error(f"Config file not found at: {config_filepath}")
            return

        with open(config_filepath, 'r', encoding='utf-8') as f: # Specify encoding
            lines = f.readlines()

        last_comment = None
        # Regex to match typical variable assignments (allows basic types, strings, lists, dicts)
        assignment_re = re.compile(r"^\s*([A-Z_][A-Z0-9_]*)\s*[:=]\s*(.*)") # Allow type hints with ':'

        for line in lines:
            stripped_line = line.strip()

            if not stripped_line: # Skip blank lines
                # Don't reset last_comment on blank lines immediately after a comment
                continue

            comment_match = re.match(r"^\s*#\s*(.*)", stripped_line)
            # Use regex on stripped line for assignment detection after comment check
            assignment_match = assignment_re.match(stripped_line)

            if comment_match:
                # Store the most recent comment line
                last_comment = comment_match.group(1).strip()
                # logger.debug(f"Found comment: {last_comment}")
            elif assignment_match:
                param_name = assignment_match.group(1)
                # Safely get the actual value and type from the imported module object
                # Safely get the actual value and type from the imported module object
                if hasattr(config_module, param_name):
                    param_value = getattr(config_module, param_name) # Correct variable holding the value
                    # Skip if value is a module or function (or other non-data type)
                    if callable(param_value) or type(param_value).__name__ == 'module':
                         last_comment = None # Reset comment as this wasn't a data param
                         continue

                    param_type = type(param_value).__name__
                    # Use json.dumps for lists/dicts/tuples, str() otherwise
                    try:
                        # --- FIX: Use param_value instead of value ---
                        if isinstance(param_value, (list, dict, tuple)):
                             param_value_str = json.dumps(param_value)
                        else:
                             param_value_str = str(param_value)
                        # --- END FIX ---
                    except Exception as e_json:
                         logger.warning(f"Could not serialize value for param '{param_name}' to string/JSON: {e_json}. Storing as repr().")
                         # --- FIX: Use param_value in fallback too ---
                         param_value_str = repr(param_value) # Fallback to repr
                         # --- END FIX ---

                    param_desc = last_comment if last_comment else 'No description found'

                    params_to_store.append({
                        'name': param_name,
                        'value': param_value_str, # Use the generated string
                        'type': param_type,
                        'description': param_desc
                    })
                    # logger.debug(f"Found param: {param_name}, Type: {param_type}, Desc: {param_desc}")

                else:
                     logger.warning(f"Variable '{param_name}' found in file but not in imported module, skipping.")
                last_comment = None # Reset comment after processing an assignment

            else:
                # Reset comment if line is neither comment nor assignment
                last_comment = None

    except FileNotFoundError:
         logger.error(f"Config file not found during parsing: {config_filepath}")
         return
    except Exception as e:
        logger.error(f"Error parsing config file {config_filepath}: {e}", exc_info=True)
        return # Abort storing if parsing fails

    # Store collected parameters in the database
    if params_to_store:
        cursor = conn.cursor()
        stored_count = 0
        skipped_count = 0
        try:
            # Use executemany for potentially better performance
            cursor.executemany(
                """
                INSERT OR REPLACE INTO config_parameters
                (param_name, param_value, param_type, param_description)
                VALUES (:name, :value, :type, :description)
                """,
                params_to_store # List of dictionaries
            )
            conn.commit()
            stored_count = len(params_to_store) # All attempted were successful if no exception
            logger.info(f"Stored/Replaced {stored_count} parsed config parameters.")
        except sqlite3.Error as e_db:
            logger.error(f"Database error storing config parameters: {e_db}")
            conn.rollback() # Rollback on error
        except Exception as e_exec:
             logger.error(f"Unexpected error storing config parameters: {e_exec}")
             conn.rollback()
    else:
        logger.warning("No parameters extracted from config file to store.")


def get_config_parameters(conn: sqlite3.Connection) -> Dict[str, Dict[str, str]]:
    """
    Retrieves stored configuration parameters from the database.

    Args:
        conn: Active database connection.

    Returns:
        Dictionary mapping parameter names to {'value': str, 'type': str, 'description': str}.
    """
    params: Dict[str, Dict[str, str]] = {}
    try:
        # Ensure row factory is set correctly for this operation
        original_factory = conn.row_factory
        conn.row_factory = dict_factory # Use the function defined above
        cursor = conn.cursor()
        cursor.execute("SELECT param_name, param_value, param_type, param_description FROM config_parameters")
        rows = cursor.fetchall() # Fetch all results
        conn.row_factory = original_factory # Restore original factory

        for row in rows:
            # Check if row is a dictionary (due to dict_factory)
            if isinstance(row, dict):
                params[row['param_name']] = {
                    'value': row.get('param_value'), # Use .get() for safety
                    'type': row.get('param_type'),
                    'description': row.get('param_description')
                }
            else:
                 logger.warning(f"Unexpected row format from get_config_parameters query: {row}")

    except sqlite3.Error as e_db:
        logger.error(f"Database error retrieving config parameters: {e_db}")
        if 'original_factory' in locals(): conn.row_factory = original_factory # Ensure restore on error
    except Exception as e:
        logger.error(f"Failed to retrieve config parameters: {e}", exc_info=True)
        if 'original_factory' in locals(): conn.row_factory = original_factory # Ensure restore on error
    return params


# --- Metadata Handling ---

def set_simulation_metadata(conn: sqlite3.Connection, key: str, value: Any) -> bool:
    """Set a metadata value for the simulation."""
    try:
        cursor = conn.cursor()
        str_value = str(value)
        cursor.execute(
            "INSERT OR REPLACE INTO simulation_metadata (key, value) VALUES (?, ?)",
            (key, str_value)
        )
        conn.commit()
        return True
    except sqlite3.Error as e:
        logger.error(f"Failed to set simulation metadata '{key}': {e}")
        conn.rollback()
        return False
    except Exception as e:
        logger.error(f"Unexpected error setting simulation metadata '{key}': {e}")
        conn.rollback()
        return False


def get_simulation_metadata(conn: sqlite3.Connection, key: str) -> Optional[str]:
    """Get a metadata value for the simulation."""
    try:
        # Ensure row factory is set for dict access
        original_factory = conn.row_factory
        conn.row_factory = dict_factory
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM simulation_metadata WHERE key = ?", (key,))
        result = cursor.fetchone()
        conn.row_factory = original_factory # Restore
        return result['value'] if result else None
    except sqlite3.Error as e:
        logger.error(f"Failed to get simulation metadata '{key}': {e}")
        if 'original_factory' in locals(): conn.row_factory = original_factory
        return None
    except Exception as e:
        logger.error(f"Unexpected error getting simulation metadata '{key}': {e}")
        if 'original_factory' in locals(): conn.row_factory = original_factory
        return None


# --- Module Status Handling ---

def register_module(
    conn: sqlite3.Connection,
    module_name: str,
    status: str = "pending",
    parameters: Optional[Dict[str, Any]] = None
) -> Optional[int]:
    """Register or update an analysis module entry."""
    try:
        cursor = conn.cursor()
        param_json = json.dumps(parameters) if parameters else None
        now = datetime.now().isoformat()

        # Check if module exists
        cursor.execute("SELECT module_id FROM analysis_modules WHERE module_name = ?", (module_name,))
        result = cursor.fetchone()

        if result:
            # Update existing module status and start time if 'running' or 'pending'
            module_id = result['module_id']
            if status in ('running', 'pending'):
                cursor.execute(
                    """
                    UPDATE analysis_modules
                    SET status = ?, start_timestamp = ?, parameters = ?, end_timestamp = NULL, error_message = NULL, execution_time = NULL
                    WHERE module_id = ?
                    """,
                    (status, now, param_json, module_id)
                )
            else: # If setting to success/failed/skipped, only update status for now (end time set by update_module_status)
                 cursor.execute(
                    """
                    UPDATE analysis_modules SET status = ? WHERE module_id = ?
                    """, (status, module_id)
                 )
        else:
            # Insert new module
            cursor.execute(
                """
                INSERT INTO analysis_modules
                (module_name, status, start_timestamp, parameters)
                VALUES (?, ?, ?, ?)
                """,
                (module_name, status, now, param_json)
            )
            module_id = cursor.lastrowid

        conn.commit()
        return module_id
    except sqlite3.Error as e:
        logger.error(f"Failed to register/update module '{module_name}': {e}")
        conn.rollback()
        return None
    except Exception as e:
         logger.error(f"Unexpected error registering/updating module '{module_name}': {e}")
         conn.rollback()
         return None


def update_module_status(
    conn: sqlite3.Connection,
    module_name: str,
    status: str,
    execution_time: Optional[float] = None,
    error_message: Optional[str] = None
) -> bool:
    """Update the status, end time, and execution time of an analysis module."""
    try:
        cursor = conn.cursor()
        now = datetime.now().isoformat()

        # Find module_id first
        cursor.execute("SELECT module_id FROM analysis_modules WHERE module_name = ?", (module_name,))
        result = cursor.fetchone()

        if not result:
            logger.warning(f"Module '{module_name}' not found for status update. Registering first.")
            # Register with the final status, but times will be approximate
            register_module(conn, module_name, status=status, parameters={'note': 'Registered during status update'})
            # Re-fetch the ID
            cursor.execute("SELECT module_id FROM analysis_modules WHERE module_name = ?", (module_name,))
            result = cursor.fetchone()
            if not result:
                 logger.error(f"Failed to register and find module '{module_name}' during status update.")
                 return False

        module_id = result['module_id']

        # Prepare update parameters
        update_params = {'status': status}
        if status in ('success', 'failed', 'skipped'):
            update_params['end_timestamp'] = now
        if execution_time is not None:
            update_params['execution_time'] = execution_time
        if error_message is not None:
            update_params['error_message'] = error_message
        else:
            # Clear error message if status is success or skipped
            if status in ('success', 'skipped'):
                 update_params['error_message'] = None


        # Build query dynamically
        set_clauses = ", ".join([f"{key} = ?" for key in update_params.keys()])
        values = list(update_params.values())
        values.append(module_id) # For the WHERE clause

        sql = f"UPDATE analysis_modules SET {set_clauses} WHERE module_id = ?"
        cursor.execute(sql, tuple(values))

        conn.commit()
        return True
    except sqlite3.Error as e:
        logger.error(f"Failed to update module status for '{module_name}': {e}")
        conn.rollback()
        return False
    except Exception as e:
         logger.error(f"Unexpected error updating module status for '{module_name}': {e}")
         conn.rollback()
         return False


def get_module_status(conn: sqlite3.Connection, module_name: str) -> Optional[str]:
    """Get the status of an analysis module."""
    try:
        # Ensure row factory is set for dict access
        original_factory = conn.row_factory
        conn.row_factory = dict_factory
        cursor = conn.cursor()
        cursor.execute("SELECT status FROM analysis_modules WHERE module_name = ?", (module_name,))
        result = cursor.fetchone()
        conn.row_factory = original_factory # Restore
        return result['status'] if result else None
    except sqlite3.Error as e:
        logger.error(f"Failed to get status for module '{module_name}': {e}")
        if 'original_factory' in locals(): conn.row_factory = original_factory
        return None
    except Exception as e:
         logger.error(f"Unexpected error getting status for module '{module_name}': {e}")
         if 'original_factory' in locals(): conn.row_factory = original_factory
         return None

def list_modules(conn: sqlite3.Connection, status: Optional[str] = None) -> List[Dict[str, Any]]:
    """List modules, optionally filtered by status."""
    modules = []
    try:
        original_factory = conn.row_factory
        conn.row_factory = dict_factory
        cursor = conn.cursor()
        if status:
            cursor.execute("SELECT * FROM analysis_modules WHERE status = ?", (status,))
        else:
            cursor.execute("SELECT * FROM analysis_modules ORDER BY module_id")
        rows = cursor.fetchall()
        conn.row_factory = original_factory # Restore

        for row in rows:
             module_data = dict(row) # Convert row object if needed
             # Parse JSON parameters if present
             if module_data.get('parameters'):
                 try:
                     module_data['parameters'] = json.loads(module_data['parameters'])
                 except (json.JSONDecodeError, TypeError):
                     logger.warning(f"Could not decode parameters JSON for module {module_data.get('module_name')}")
                     # Keep as string if parsing fails
             modules.append(module_data)

    except sqlite3.Error as e:
        logger.error(f"Failed to list modules: {e}")
        if 'original_factory' in locals(): conn.row_factory = original_factory
    except Exception as e:
         logger.error(f"Unexpected error listing modules: {e}")
         if 'original_factory' in locals(): conn.row_factory = original_factory
    return modules


# --- Product Handling ---

def register_product(
    conn: sqlite3.Connection,
    module_name: str,
    product_type: str,
    category: str,
    relative_path: str, # Changed order to match Contributing.md
    subcategory: Optional[str] = None,
    description: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Optional[int]:
    """Register or update an analysis product."""
    try:
        cursor = conn.cursor()

        # Get module_id
        cursor.execute("SELECT module_id FROM analysis_modules WHERE module_name = ?", (module_name,))
        result = cursor.fetchone()

        if not result:
            logger.warning(f"Module '{module_name}' not found for product registration. Registering module.")
            module_id = register_module(conn, module_name, status="success") # Assume success if product generated
            if not module_id:
                logger.error(f"Failed to register module '{module_name}' for product '{relative_path}'")
                return None
        else:
            module_id = result['module_id']

        meta_json = json.dumps(metadata) if metadata else None
        now = datetime.now().isoformat()

        # Use INSERT OR REPLACE based on the unique relative_path
        cursor.execute(
            """
            INSERT OR REPLACE INTO analysis_products
            (module_id, product_type, category, subcategory, relative_path,
             description, generation_timestamp, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (module_id, product_type, category, subcategory, relative_path,
             description, now, meta_json)
        )
        product_id = cursor.lastrowid # Get ID of inserted/replaced row

        conn.commit()
        # logger.debug(f"Registered/Updated product: {relative_path} (ID: {product_id})")
        return product_id
    except sqlite3.Error as e:
        logger.error(f"Failed to register product '{relative_path}': {e}")
        conn.rollback()
        return None
    except Exception as e:
         logger.error(f"Unexpected error registering product '{relative_path}': {e}")
         conn.rollback()
         return None


def get_product_path(
    conn: sqlite3.Connection,
    product_type: Optional[str] = None,
    category: Optional[str] = None,
    subcategory: Optional[str] = None,
    module_name: Optional[str] = None
) -> Optional[str]:
    """Get the relative path to a product matching the criteria."""
    try:
        original_factory = conn.row_factory
        conn.row_factory = dict_factory
        cursor = conn.cursor()

        # Build query
        query = "SELECT p.relative_path FROM analysis_products p"
        params = []
        joins = ""
        conditions = []

        if module_name:
            joins += " JOIN analysis_modules m ON p.module_id = m.module_id"
            conditions.append("m.module_name = ?")
            params.append(module_name)

        if product_type:
            conditions.append("p.product_type = ?")
            params.append(product_type)
        if category:
            conditions.append("p.category = ?")
            params.append(category)
        if subcategory:
            conditions.append("p.subcategory = ?")
            params.append(subcategory)

        if conditions:
            query += joins + " WHERE " + " AND ".join(conditions)
        else:
             query += joins # Include join even if no other conditions

        # Order by newest first in case multiple matches exist (prefer latest)
        query += " ORDER BY p.generation_timestamp DESC LIMIT 1"

        cursor.execute(query, tuple(params))
        result = cursor.fetchone()
        conn.row_factory = original_factory # Restore

        if result:
            return result['relative_path']
        else:
            logger.debug(f"No product found matching type={product_type}, cat={category}, subcat={subcategory}, mod={module_name}")
            return None
    except sqlite3.Error as e:
        logger.error(f"Failed to get product path: {e}")
        if 'original_factory' in locals(): conn.row_factory = original_factory
        return None
    except Exception as e:
         logger.error(f"Unexpected error getting product path: {e}")
         if 'original_factory' in locals(): conn.row_factory = original_factory
         return None

def get_all_products(
    conn: sqlite3.Connection,
    module_name: Optional[str] = None,
    product_type: Optional[str] = None,
    category: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Get all products matching the specified criteria."""
    products = []
    try:
        original_factory = conn.row_factory
        conn.row_factory = dict_factory
        cursor = conn.cursor()

        query = "SELECT p.*, m.module_name FROM analysis_products p JOIN analysis_modules m ON p.module_id = m.module_id"
        params = []
        conditions = []

        if module_name:
            conditions.append("m.module_name = ?")
            params.append(module_name)
        if product_type:
            conditions.append("p.product_type = ?")
            params.append(product_type)
        if category:
            conditions.append("p.category = ?")
            params.append(category)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY p.generation_timestamp DESC"

        cursor.execute(query, tuple(params))
        rows = cursor.fetchall()
        conn.row_factory = original_factory # Restore

        for row in rows:
            product_data = dict(row)
            # Parse JSON metadata if present
            if product_data.get('metadata'):
                try:
                    product_data['metadata'] = json.loads(product_data['metadata'])
                except (json.JSONDecodeError, TypeError):
                    logger.warning(f"Could not decode metadata JSON for product {product_data.get('relative_path')}")
                    # Keep as string if parsing fails
            products.append(product_data)

    except sqlite3.Error as e:
        logger.error(f"Failed to get products: {e}")
        if 'original_factory' in locals(): conn.row_factory = original_factory
    except Exception as e:
         logger.error(f"Unexpected error getting products: {e}")
         if 'original_factory' in locals(): conn.row_factory = original_factory
    return products

def check_product_exists(conn: sqlite3.Connection, relative_path: str) -> bool:
    """Check if a product exists by its relative path."""
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM analysis_products WHERE relative_path = ? LIMIT 1", (relative_path,))
        result = cursor.fetchone()
        return result is not None
    except sqlite3.Error as e:
        logger.error(f"Failed to check product existence for '{relative_path}': {e}")
        return False
    except Exception as e:
         logger.error(f"Unexpected error checking product existence for '{relative_path}': {e}")
         return False

# --- Metric Handling ---

def store_metric(
    conn: sqlite3.Connection,
    module_name: str,
    metric_name: str,
    value: Optional[Union[float, int, str]], # Allow storing strings if needed
    units: Optional[str] = None,
    description: Optional[str] = None
) -> bool:
    """Store or update a numerical metric."""
    try:
        cursor = conn.cursor()

        # Get module_id
        cursor.execute("SELECT module_id FROM analysis_modules WHERE module_name = ?", (module_name,))
        result = cursor.fetchone()

        if not result:
            logger.warning(f"Module '{module_name}' not found for metric storage. Registering module.")
            module_id = register_module(conn, module_name, status="success")
            if not module_id:
                logger.error(f"Failed to register module '{module_name}' for metric '{metric_name}'")
                return False
        else:
            module_id = result['module_id']

        # Handle value: Store None as NULL, attempt float conversion otherwise
        numeric_value = None
        if value is not None:
             try:
                 # Check for NaN/Inf explicitly before float conversion
                 if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                     numeric_value = None # Store as NULL
                 else:
                     numeric_value = float(value)
             except (ValueError, TypeError):
                 logger.warning(f"Metric '{metric_name}' has non-numeric value: '{value}'. Storing as NULL.")
                 numeric_value = None # Store non-convertible as NULL

        # Use empty string for units if None
        units_str = units if units is not None else ''

        # Use INSERT OR REPLACE based on the unique (module_id, metric_name) constraint
        cursor.execute(
            """
            INSERT OR REPLACE INTO metrics
            (module_id, metric_name, value, units, description)
            VALUES (?, ?, ?, ?, ?)
            """,
            (module_id, metric_name, numeric_value, units_str, description)
        )
        conn.commit()
        # logger.debug(f"Stored metric: {module_name}/{metric_name} = {numeric_value} {units_str}")
        return True
    except sqlite3.Error as e:
        logger.error(f"Failed to store metric '{metric_name}': {e}")
        conn.rollback()
        return False
    except Exception as e:
         logger.error(f"Unexpected error storing metric '{metric_name}': {e}")
         conn.rollback()
         return False


def get_metric_value(
    conn: sqlite3.Connection,
    metric_name: str,
    module_name: Optional[str] = None
) -> Optional[float]:
    """Get the value of a metric."""
    try:
        original_factory = conn.row_factory
        conn.row_factory = dict_factory
        cursor = conn.cursor()

        if module_name:
            cursor.execute(
                """
                SELECT m.value FROM metrics m
                JOIN analysis_modules mod ON m.module_id = mod.module_id
                WHERE m.metric_name = ? AND mod.module_name = ?
                """,
                (metric_name, module_name)
            )
        else:
            # If module not specified, get the latest one (highest metric_id)
            cursor.execute(
                """
                SELECT value FROM metrics
                WHERE metric_name = ?
                ORDER BY metric_id DESC LIMIT 1
                """,
                (metric_name,)
            )

        result = cursor.fetchone()
        conn.row_factory = original_factory # Restore
        # Value is stored as REAL, return directly (could be None if stored as NULL)
        return result['value'] if result else None
    except sqlite3.Error as e:
        logger.error(f"Failed to get metric value for '{metric_name}': {e}")
        if 'original_factory' in locals(): conn.row_factory = original_factory
        return None
    except Exception as e:
         logger.error(f"Unexpected error getting metric value for '{metric_name}': {e}")
         if 'original_factory' in locals(): conn.row_factory = original_factory
         return None


def get_all_metrics(conn: sqlite3.Connection, module_name: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """Get all metrics for a module, or all metrics if module_name is None."""
    metrics_dict: Dict[str, Dict[str, Any]] = {}
    try:
        original_factory = conn.row_factory
        conn.row_factory = dict_factory
        cursor = conn.cursor()

        if module_name:
            cursor.execute(
                """
                SELECT m.metric_name, m.value, m.units, m.description
                FROM metrics m
                JOIN analysis_modules mod ON m.module_id = mod.module_id
                WHERE mod.module_name = ?
                ORDER BY m.metric_name
                """,
                (module_name,)
            )
        else:
            cursor.execute(
                """
                SELECT m.metric_name, m.value, m.units, m.description, mod.module_name
                FROM metrics m
                JOIN analysis_modules mod ON m.module_id = mod.module_id
                ORDER BY mod.module_name, m.metric_name
                """
            )

        rows = cursor.fetchall()
        conn.row_factory = original_factory # Restore

        for row in rows:
             metric_name = row['metric_name']
             # If querying all modules, maybe prefix with module name?
             # key = f"{row['module_name']}.{metric_name}" if module_name is None else metric_name
             key = metric_name # Keep key simple for now
             metrics_dict[key] = {
                 'value': row['value'], # Stored as REAL, could be None
                 'units': row.get('units', ''), # Use .get() for safety
                 'description': row.get('description')
             }
             # Optionally add module name if querying all
             if module_name is None:
                 metrics_dict[key]['module_name'] = row.get('module_name')


    except sqlite3.Error as e:
        logger.error(f"Failed to get metrics: {e}")
        if 'original_factory' in locals(): conn.row_factory = original_factory
    except Exception as e:
         logger.error(f"Unexpected error getting metrics: {e}")
         if 'original_factory' in locals(): conn.row_factory = original_factory
    return metrics_dict
