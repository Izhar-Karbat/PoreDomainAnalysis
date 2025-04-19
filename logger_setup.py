# logger_setup.py
"""
Functions for setting up logging for the MD Analysis Script.
"""

import logging
import os
import sys
from datetime import datetime

# Define standard log format
LOG_FORMAT = '%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s] - %(message)s'
# Alternative shorter format:
# LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'


def setup_analysis_logger(run_dir, run_name, log_level=logging.INFO):
    """
    Configures the root logger to write ONLY to a log file within the specific run directory.

    Removes any existing handlers from the root logger and sets up a new
    FileHandler pointing to '<run_dir>/<run_name>_analysis.log'.
    Also maintains console logging.

    Args:
        run_dir (str): Path to the specific run directory where the log file will be saved.
        run_name (str): The name of the run (used for the log filename).
        log_level (int): The logging level (e.g., logging.INFO, logging.DEBUG).

    Returns:
        str | None: The path to the analysis log file created, or None on error.
    """
    if not run_dir or not os.path.isdir(run_dir):
        print(f"ERROR: Invalid run directory provided for logger setup: {run_dir}", file=sys.stderr)
        return None
    if not run_name:
        print(f"ERROR: Invalid run_name provided for logger setup.", file=sys.stderr)
        return None

    log_file_name = f"{run_name}_analysis.log"
    log_file_path = os.path.join(run_dir, log_file_name)

    # Get the root logger
    root_logger = logging.getLogger()

    # --- Critical: Remove existing handlers --- #
    # This prevents duplicate logs if the function is called again or if basicConfig was used.
    if root_logger.hasHandlers():
        # Use list comprehension to avoid modifying the list while iterating
        for handler in list(root_logger.handlers):
            root_logger.removeHandler(handler)

    # --- Configure Handlers --- #
    # Configure file logging (writes to run_dir)
    try:
        file_handler = logging.FileHandler(log_file_path, mode='w') # Use mode 'w' to overwrite previous logs for this run
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        file_handler.setLevel(log_level)
    except Exception as e:
        print(f"ERROR: Failed to create log file handler at {log_file_path}: {e}", file=sys.stderr)
        return None

    # Configure basic console logging (still useful for immediate feedback)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    console_handler.setLevel(log_level) # Can set a different level for console if desired

    # Add handlers to the root logger
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger.setLevel(log_level) # Set the overall level for the logger

    # --- Initial Log Messages --- #
    # Now log using the configured root logger
    root_logger.info(f"Logger configured. Level: {log_level}")
    root_logger.info(f"Logging to Console and File: {log_file_path}")
    root_logger.info(f"Python version: {sys.version}")
    root_logger.info(f"Script execution started at: {datetime.now().isoformat()}")

    return log_file_path
