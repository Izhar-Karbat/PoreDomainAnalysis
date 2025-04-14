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


def setup_root_logger(log_level=logging.INFO, log_dir=None):
    """
    Configures the root logger for the main script execution.

    Sets up basic console logging and a file handler for the main execution log.

    Args:
        log_level (int): The logging level (e.g., logging.INFO, logging.DEBUG).
        log_dir (str, optional): Directory to save the main log file.
                                 Defaults to the current working directory.

    Returns:
        str: The path to the main log file created.
    """
    if log_dir is None:
        log_dir = os.getcwd()
    os.makedirs(log_dir, exist_ok=True) # Ensure log directory exists

    # Create the main log file path
    main_log_filename = f"md_analysis_main_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    main_log_path = os.path.join(log_dir, main_log_filename)

    # Get the root logger
    root_logger = logging.getLogger()

    # Check if handlers already exist (e.g., if called multiple times)
    # Avoid adding duplicate handlers
    if not root_logger.handlers:
        # Configure basic console logging
        console_handler = logging.StreamHandler(sys.stdout) # Use stdout
        console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        console_handler.setLevel(log_level) # Console level can be same or different

        # Configure file logging
        file_handler = logging.FileHandler(main_log_path)
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        file_handler.setLevel(log_level) # Log level for the file

        # Add handlers to the root logger
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)
        root_logger.setLevel(log_level) # Set the overall level for the logger

        root_logger.info(f"Root logger configured. Console Level: {log_level}, File Level: {log_level}")
        root_logger.info(f"Main execution log file: {main_log_path}")
        root_logger.info(f"Python version: {sys.version}")
        root_logger.info(f"Current working directory: {os.getcwd()}")
    else:
         root_logger.info("Root logger already configured.")

    return main_log_path


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
