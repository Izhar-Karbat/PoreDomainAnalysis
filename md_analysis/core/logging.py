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
    if root_logger.hasHandlers():
        for handler in list(root_logger.handlers):
            root_logger.removeHandler(handler)

    # --- Configure Handlers --- #
    try:
        file_handler = logging.FileHandler(log_file_path, mode='w') # Use mode 'w' to overwrite
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        file_handler.setLevel(log_level)
    except Exception as e:
        print(f"ERROR: Failed to create log file handler at {log_file_path}: {e}", file=sys.stderr)
        return None

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    console_handler.setLevel(log_level)

    # Add handlers to the root logger
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger.setLevel(log_level)

    # --- Initial Log Messages --- #
    root_logger.info(f"Logger configured. Level: {logging.getLevelName(log_level)}") # Use getLevelName
    root_logger.info(f"Logging to Console and File: {log_file_path}")
    root_logger.info(f"Python version: {sys.version}")
    root_logger.info(f"Script execution started at: {datetime.now().isoformat()}")

    return log_file_path


def setup_system_logger(run_dir, log_level=logging.INFO):
    """
    Sets up a system-specific logger that writes logs to the specified run directory.

    Prevents propagation to the root logger to avoid duplicate messages in the main log.

    Args:
        run_dir (str): Path to the specific run directory where the log file will be saved.
        log_level (int): The logging level for this specific logger.

    Returns:
        logging.Logger: The configured system-specific logger instance.
                        Returns None if run_dir is invalid.
    """
    if not run_dir or not os.path.isdir(run_dir):
        logging.error(f"Invalid run directory provided for logger setup: {run_dir}")
        return None

    try:
        # Create a unique logger name based on the directory structure
        run_name = os.path.basename(run_dir)
        parent_dir = os.path.dirname(run_dir)
        system_name = os.path.basename(parent_dir) if parent_dir else run_name # Handle edge case
        logger_name = f"{system_name}_{run_name}" # e.g., "SystemX_R1"

        # Get the logger instance
        logger = logging.getLogger(logger_name)

        # --- Crucial: Prevent adding handlers if they already exist for this logger ---
        # This avoids duplicate logs if this function is somehow called multiple times
        # for the same run_dir within a single script execution.
        if not logger.handlers:
            # Don't propagate messages to the root logger (avoids duplicates in main log)
            logger.propagate = False

            # Set the logging level for this specific logger
            logger.setLevel(log_level)

            # Create log file path within the run directory
            log_file_name = f"{run_name}_analysis.log"
            log_file_path = os.path.join(run_dir, log_file_name)

            # Create file handler for this specific logger
            file_handler = logging.FileHandler(log_file_path)
            file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
            file_handler.setLevel(log_level) # Use the specified level

            # Add the handler to this specific logger
            logger.addHandler(file_handler)

            logger.info(f"System-specific logger '{logger_name}' configured.")
            logger.info(f"Log file created at: {log_file_path}")

            # Also log to main logger that we're starting analysis for this system
            main_logger = logging.getLogger() # Get root logger
            main_logger.info(f"--- Starting analysis for {system_name}/{run_name} (Log: {log_file_path}) ---")

        else:
            # Logger already has handlers, likely setup previously in this run
            logger.debug(f"Logger '{logger_name}' already has handlers. Skipping setup.")
            # Log to main logger anyway to indicate processing start
            main_logger = logging.getLogger() # Get root logger
            main_logger.info(f"--- Processing (re-entry?) {system_name}/{run_name} ---")


        return logger

    except Exception as e:
        # Log error using the root logger as the specific logger might have failed
        logging.error(f"Failed to set up system logger for {run_dir}: {e}", exc_info=True)
        return None # Indicate failure
