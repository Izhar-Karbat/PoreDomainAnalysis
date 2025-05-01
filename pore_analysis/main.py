# filename: pore_analysis/main.py
"""
Main orchestration module for pore analysis.

This module coordinates the entire analysis workflow, managing the execution of various
analysis modules and ensuring proper sequencing of dependencies. It uses a
database-centric approach for tracking analysis products and status.
"""

import os
import sys
import logging
import argparse
import time
from datetime import datetime
import sqlite3
from typing import Optional, Dict, Any # Added Dict, Any
import json # Import json for error summary
import traceback # Import traceback for detailed error logging

# --- Import analysis modules with error handling ---
# (Imports remain the same as in the original file provided)
try:
    # Core Analysis
    from pore_analysis.modules.core_analysis import (
        analyze_trajectory, filter_and_save_data, plot_distances, plot_kde_analysis
    )
    core_analysis_available = True
except ImportError:
    core_analysis_available = False
    # Define dummy functions if module not found, allows script to run partially
    def analyze_trajectory(*args, **kwargs): logging.warning("Core analysis module not found."); return {'status': 'skipped', 'error': 'Module not found'}
    def filter_and_save_data(*args, **kwargs): logging.warning("Core analysis module not found."); return {'status': 'skipped', 'error': 'Module not found'}
    def plot_distances(*args, **kwargs): logging.warning("Core analysis module not found."); return {}
    def plot_kde_analysis(*args, **kwargs): logging.warning("Core analysis module not found."); return None

try:
    # Orientation Analysis
    from pore_analysis.modules.orientation_contacts import (
        run_orientation_analysis, generate_orientation_plots
    )
    orientation_analysis_available = True
except ImportError:
    orientation_analysis_available = False
    def run_orientation_analysis(*args, **kwargs): logging.warning("Orientation analysis module not found."); return {'status': 'skipped', 'error': 'Module not found'}
    def generate_orientation_plots(*args, **kwargs): logging.warning("Orientation analysis module not found."); return {'status': 'skipped', 'error': 'Module not found'}

try:
    # Ion Analysis
    from pore_analysis.modules.ion_analysis import (
        run_ion_analysis, generate_ion_plots
    )
    ion_analysis_available = True
except ImportError:
    ion_analysis_available = False
    def run_ion_analysis(*args, **kwargs): logging.warning("Ion analysis module not found."); return {'status': 'skipped', 'error': 'Module not found'}
    def generate_ion_plots(*args, **kwargs): logging.warning("Ion analysis module not found."); return {'status': 'skipped', 'error': 'Module not found'}

try:
    # Inner Vestibule Analysis
    from pore_analysis.modules.inner_vestibule_analysis import (
        run_inner_vestibule_analysis, generate_inner_vestibule_plots
    )
    inner_vestibule_analysis_available = True
except ImportError:
    inner_vestibule_analysis_available = False
    def run_inner_vestibule_analysis(*args, **kwargs): logging.warning("Inner vestibule analysis module not found."); return {'status': 'skipped', 'error': 'Module not found'}
    def generate_inner_vestibule_plots(*args, **kwargs): logging.warning("Inner vestibule analysis module not found."); return {'status': 'skipped', 'error': 'Module not found'}

try:
    # Gyration Analysis
    from pore_analysis.modules.gyration_analysis import (
        run_gyration_analysis, generate_gyration_plots
    )
    gyration_analysis_available = True
except ImportError:
    gyration_analysis_available = False
    def run_gyration_analysis(*args, **kwargs): logging.warning("Gyration analysis module not found."); return {'status': 'skipped', 'error': 'Module not found'}
    def generate_gyration_plots(*args, **kwargs): logging.warning("Gyration analysis module not found."); return {'status': 'skipped', 'error': 'Module not found'}

try:
    # Tyrosine Analysis
    from pore_analysis.modules.tyrosine_analysis import (
         run_tyrosine_analysis, generate_tyrosine_plots
    )
    tyrosine_analysis_available = True
except ImportError:
    tyrosine_analysis_available = False
    # Define dummy functions if module not found
    def run_tyrosine_analysis(*args, **kwargs): logging.warning("Tyrosine analysis module not found."); return {'status': 'skipped', 'error': 'Module not found'}
    def generate_tyrosine_plots(*args, **kwargs): logging.warning("Tyrosine analysis module not found."); return {'status': 'skipped', 'error': 'Module not found'}

try:
    # DW Gate modules
    from pore_analysis.modules.dw_gate_analysis import (
        run_dw_gate_analysis, generate_dw_gate_plots
    )
    dw_gate_analysis_available = True
except ImportError:
    dw_gate_analysis_available = False
    def run_dw_gate_analysis(*args, **kwargs): logging.warning("DW Gate analysis module not found."); return {'status': 'skipped', 'error': 'Module not found'}
    def generate_dw_gate_plots(*args, **kwargs): logging.warning("DW Gate analysis module not found."); return {'status': 'skipped', 'error': 'Module not found'}

try:
    # Core Utilities
    from pore_analysis.core.config import Analysis_version # Import version
    from pore_analysis.core.logging import setup_analysis_logger
    from pore_analysis.core.database import (
        init_db, connect_db, set_simulation_metadata, get_module_status,
        get_product_path, list_modules, get_simulation_metadata, update_module_status,
        store_config_parameters # Import function to store config
    )
    from pore_analysis.core.utils import clean_json_data # Import for error summary
    # Summary and Report Generation
    from pore_analysis.summary import calculate_summary, generate_summary_from_database
    from pore_analysis.html import generate_html_report
    # Import the config module itself to pass to store_config_parameters
    from pore_analysis.core import config as core_config_module
    core_utils_available = True
except ImportError as e:
    # Use standard print as logging might not be set up
    print(f"Error importing CORE pore analysis modules/utilities: {e}", file=sys.stderr)
    sys.exit(1)

# Configure logger
logger = logging.getLogger(__name__) # Get logger instance


# --- Argument Parsing (Based on Original Logic) ---
# (parse_arguments function remains unchanged)
def parse_arguments():
    """Parse command line arguments based on the original script's logic."""
    parser = argparse.ArgumentParser(
        description=f"Pore Analysis Suite (Database Refactor v{Analysis_version}). Processes trajectories for a SINGLE run folder.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    # --- Input/Output & Mode Flags ---
    parser.add_argument("--folder", required=True, help="Path to the specific run folder containing PSF/DCD (required).")
    parser.add_argument('--psf', help='Override default PSF file path (step5_input.psf in run folder)')
    parser.add_argument('--dcd', help='Override default DCD file path (MD_Aligned.dcd in run folder)')
    # Processing options
    parser.add_argument("--force-rerun", action="store_true", help="Force reprocessing of ALL specified or default modules, ignoring database status.")
    parser.add_argument('--reinit-db', action='store_true', help='Reinitialize database (will lose previous analysis tracking)')
    parser.add_argument('--summary-only', action='store_true', help='Only generate summary/report from DB without running analysis')

    # --- Analysis Selection Flags ---
    analysis_group = parser.add_argument_group('Selective Analysis Flags (run ONLY specified modules)')
    analysis_group.add_argument("--core", action="store_true", help="Run Core distance & filtering analysis.")
    analysis_group.add_argument("--orientation", action="store_true", help="Run Toxin orientation and contact analysis.")
    analysis_group.add_argument("--ions", action="store_true", help="Run K+ ion tracking, occupancy, and conduction analysis.")
    analysis_group.add_argument("--water", action="store_true", help="Run Cavity Water analysis.")
    analysis_group.add_argument("--gyration", action="store_true", help="Run Carbonyl Gyration analysis.")
    analysis_group.add_argument("--tyrosine", action="store_true", help="Run SF Tyrosine rotamer analysis.")
    analysis_group.add_argument("--dwgates", action="store_true", help="Run DW-Gate analysis.")

    # --- Other Options / Report Control ---
    other_group = parser.add_argument_group('Other Options')
    other_group.add_argument("--box_z", type=float, default=None, help="Provide estimated box Z-dimension (Angstroms) for multi-level COM filter.")
    other_group.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Set the logging level.")
    other_group.add_argument("--report", action="store_true", help="Generate HTML report when running selective analyses.")
    other_group.add_argument("--no-report", action="store_true", help="Suppress HTML report generation when running full analysis (default or --force-rerun).")
    other_group.add_argument("--no-plots", action="store_true", help="Skip generating plots for selected analyses.")

    return parser.parse_args()


# --- Error Summary Helper (Unchanged) ---
# (_save_error_summary function remains unchanged)
def _save_error_summary(run_dir, system_name, run_name, error_message):
    """Helper to save a minimal summary JSON when a critical error occurs."""
    summary_file_path = os.path.join(run_dir, 'analysis_summary.json')
    error_summary = {
        'SystemName': system_name,
        'RunName': run_name,
        'RunPath': run_dir,
        'AnalysisStatus': f'FAILED: {str(error_message)[:150]}', # Truncate long errors
        'AnalysisScriptVersion': Analysis_version,
        'AnalysisTimestamp': datetime.now().isoformat()
    }
    try:
        # Use clean_json_data if available
        cleaned_summary = clean_json_data(error_summary) if 'clean_json_data' in globals() else error_summary
        with open(summary_file_path, 'w') as f_json:
            json.dump(cleaned_summary, f_json, indent=4)
        logger.info(f"Saved error status to {summary_file_path}")
    except Exception as e_save:
        logger.error(f"Failed to save error summary JSON to {summary_file_path}: {e_save}")


# --- Main Workflow Function (Revised with try/except/finally) ---
def _run_analysis_workflow(args):
    """
    Main analysis workflow function using database tracking.
    Applies logic based on original script's flag handling.
    Includes try/except/finally block for robust status updates.
    """
    run_dir = os.path.abspath(args.folder)
    if not os.path.isdir(run_dir):
        logger.error(f"Run directory does not exist: {run_dir}")
        return None

    psf_file = args.psf if args.psf else os.path.join(run_dir, "step5_input.psf")
    dcd_file = args.dcd if args.dcd else os.path.join(run_dir, "MD_Aligned.dcd")

    if not os.path.exists(psf_file):
        logger.error(f"PSF file not found: {psf_file}")
        return None
    if not os.path.exists(dcd_file):
        logger.error(f"DCD file not found: {dcd_file}")
        return None

    db_conn: Optional[sqlite3.Connection] = None # Initialize connection variable
    summary = {} # Initialize summary dictionary
    overall_success = False # Assume failure until proven otherwise

    try:
        # --- Initialize Database ---
        db_conn = init_db(run_dir, force_recreate=args.reinit_db)
        if db_conn is None:
            # Error already logged by init_db
            return None # Cannot proceed without DB

        run_name = os.path.basename(run_dir)
        parent_dir = os.path.dirname(run_dir)
        system_name = os.path.basename(parent_dir) if parent_dir and os.path.basename(parent_dir) != '.' else run_name

        # --- Set Initial Metadata, Status, and Store Config ---
        set_simulation_metadata(db_conn, "run_name", run_name)
        set_simulation_metadata(db_conn, "system_name", system_name)
        set_simulation_metadata(db_conn, "analysis_start_time", datetime.now().isoformat())
        set_simulation_metadata(db_conn, "psf_file", os.path.basename(psf_file))
        set_simulation_metadata(db_conn, "dcd_file", os.path.basename(dcd_file))
        set_simulation_metadata(db_conn, "analysis_version", Analysis_version)
        set_simulation_metadata(db_conn, "analysis_status", "running") # Mark as running
        db_conn.commit() # Commit initial metadata

        if 'core_config_module' in globals():
            store_config_parameters(db_conn, core_config_module)
            logger.info("Stored configuration parameters in database.")
        else:
            logger.warning("core_config module not imported, cannot store parameters.")

        # --- Summary-Only Mode ---
        if args.summary_only:
            logger.info("Summary-only mode. Skipping analysis execution.")
            summary = generate_summary_from_database(run_dir, db_conn)
            if not summary:
                 logger.error("Failed to generate summary from database.")
                 raise RuntimeError("Failed to generate summary") # Treat as failure

            if args.report:
                logger.info("Generating HTML report...")
                report_path = generate_html_report(run_dir, summary=summary)
                if not report_path:
                    logger.warning("HTML report generation failed or returned no path")
                    # Do not treat report failure as overall failure in summary-only
                else:
                    logger.info(f"HTML report generated: {report_path}")
            else:
                logger.info("HTML report generation skipped in summary-only mode.")

            overall_success = True # Summary generation is considered success here
            return summary # Exit workflow early

        # --- Determine which modules to run (Logic unchanged) ---
        specific_flags_set = args.core or args.orientation or args.ions or args.water or args.gyration or args.tyrosine or args.dwgates
        run_all_initially = not specific_flags_set

        run_core = (args.core or run_all_initially) and core_analysis_available
        run_orientation = (args.orientation or run_all_initially) and orientation_analysis_available
        run_ions = (args.ions or run_all_initially) and ion_analysis_available
        run_water = (args.water or run_all_initially) and inner_vestibule_analysis_available
        run_gyration = (args.gyration or run_all_initially) and gyration_analysis_available
        run_tyrosine = (args.tyrosine or run_all_initially) and tyrosine_analysis_available
        run_dwgates = (args.dwgates or run_all_initially) and dw_gate_analysis_available

        generate_plots = not args.no_plots

        if specific_flags_set:
            generate_html = args.report
        else: # Run all default case
            generate_html = not args.no_report

        # Log module availability warnings (Logic unchanged)
        if args.core and not core_analysis_available: logger.warning("Core analysis requested but module is not available/imported.")
        if args.orientation and not orientation_analysis_available: logger.warning("Orientation analysis requested but module is not available/imported.")
        if args.ions and not ion_analysis_available: logger.warning("Ion analysis requested but module is not available/imported.")
        if args.water and not inner_vestibule_analysis_available: logger.warning("Inner Vestibule analysis requested but module is not available/imported.")
        if args.gyration and not gyration_analysis_available: logger.warning("Gyration analysis requested but module is not available/imported.")
        if args.tyrosine and not tyrosine_analysis_available: logger.warning("Tyrosine analysis requested but module is not available/imported.")
        if args.dwgates and not dw_gate_analysis_available: logger.warning("DW Gate analysis requested but module is not available/imported.")

        logger.info(f"Analysis Plan: Core={run_core}, Orientation={run_orientation}, Ions={run_ions}, Water={run_water}, Gyration={run_gyration}, Tyrosine={run_tyrosine}, DWGates={run_dwgates}")
        logger.info(f"Generate Plots: {generate_plots}")
        logger.info(f"Generate HTML Report: {generate_html}")
        logger.info(f"Force Rerun: {args.force_rerun}")

        # ===========================================
        # --- Start of Analysis Module Execution ----
        # ===========================================
        # Initialize success flag for the main analysis part
        current_step_success = True

        # --- Core Analysis Module ---
        core_analysis_done = False
        core_filter_status = get_module_status(db_conn, "core_analysis_filtering")
        core_comp_status = get_module_status(db_conn, "core_analysis")
        needs_core_run = args.force_rerun or core_comp_status != "success" or core_filter_status != "success"

        if run_core and needs_core_run:
            logger.info("Running core distance computation and filtering...")
            start_time_core = time.time()
            core_results = analyze_trajectory(run_dir, psf_file, dcd_file)
            if core_results and core_results['status'] == 'success':
                set_simulation_metadata(db_conn, "is_control_system", str(core_results['metadata']['is_control_system']))
                filtering_results = filter_and_save_data(
                    run_dir, core_results['data']['dist_ac'], core_results['data']['dist_bd'],
                    core_results['data']['com_distances'], core_results['data']['time_points'],
                    is_control_system=core_results['metadata']['is_control_system'],
                    db_conn=db_conn, box_z=args.box_z
                )
                if filtering_results and filtering_results['status'] == 'success':
                    core_analysis_done = True
                else:
                    logger.error(f"Core: Distance filtering failed: {filtering_results.get('error', 'Unknown')}")
                    current_step_success = False
            elif core_results and core_results['status'] == 'skipped':
                 logger.warning("Core analysis computation skipped (module not found?)")
                 current_step_success = False # Consider unavailability as failure if needed
            else:
                logger.error(f"Core: Trajectory analysis failed: {core_results.get('error', 'Unknown') if core_results else 'No results returned'}")
                current_step_success = False
            logger.info(f"Core computation/filtering finished in {time.time() - start_time_core:.2f} seconds.")
            # --- Raise error if core step failed critically ---
            if not current_step_success: raise RuntimeError("Core analysis computation/filtering failed.")
        elif run_core:
            logger.info("Skipping core computation/filtering - already completed successfully.")
            core_analysis_done = True
        else: # Not running core
            core_analysis_done = core_filter_status == "success" # Check if previously done

        # Fetch control status AFTER potential core run
        is_control_system = get_simulation_metadata(db_conn, 'is_control_system') == 'True'

        # Core Viz (Run if needed and computation was done or already successful)
        core_viz_gg_status = get_module_status(db_conn, "core_analysis_visualization_g_g")
        core_viz_com_status = get_module_status(db_conn, "core_analysis_visualization_com")
        needs_core_viz = generate_plots and (args.force_rerun or core_viz_gg_status != "success" or (not is_control_system and core_viz_com_status != "success"))
        if core_analysis_done and needs_core_viz and core_analysis_available:
            logger.info("Generating core analysis plots...")
            plot_distances(run_dir, is_gg=True, db_conn=db_conn)
            if not is_control_system:
                plot_distances(run_dir, is_gg=False, db_conn=db_conn)
                plot_kde_analysis(run_dir, db_conn=db_conn, box_z=args.box_z)
        # (Logging for skipping viz remains the same)

        # --- Other Analysis Modules ---
        # (Existing logic for Orientation, Ion, Vestibule, Gyration, Tyrosine, DWGate remains largely the same)
        # Key change: If a module fails, set current_step_success = False and raise RuntimeError
        # Example for Orientation:

        # --- Orientation Analysis Module ---
        orientation_analysis_done = False
        orient_comp_status = get_module_status(db_conn, "orientation_analysis")
        orient_viz_status = get_module_status(db_conn, "orientation_analysis_visualization")

        if run_orientation:
            if is_control_system:
                # (Skip logic remains the same)
                orientation_analysis_done = True
            else:
                needs_orient_comp_run = args.force_rerun or orient_comp_status != "success"
                needs_orient_viz_run = generate_plots and (args.force_rerun or orient_viz_status != "success")

                if needs_orient_comp_run or (needs_orient_viz_run and orient_comp_status != 'success'):
                    logger.info(f"Running orientation analysis computation (Needed: {needs_orient_comp_run})...")
                    start_time_orient_comp = time.time()
                    orient_comp_results = run_orientation_analysis(run_dir, psf_file, dcd_file, db_conn=db_conn)
                    orientation_analysis_done = orient_comp_results.get('status') == 'success'
                    logger.info(f"Orientation computation completed in {time.time() - start_time_orient_comp:.2f} seconds. Success: {orientation_analysis_done}")
                    if not orientation_analysis_done:
                        logger.error(f"Orientation computation failed: {orient_comp_results.get('error', 'Unknown')}")
                        current_step_success = False
                        raise RuntimeError("Orientation analysis computation failed.")
                else:
                    # (Skip logic remains the same)
                    orientation_analysis_done = True

                if orientation_analysis_done and needs_orient_viz_run and orientation_analysis_available:
                    logger.info("Generating orientation analysis plots...")
                    start_time_orient_viz = time.time()
                    viz_results = generate_orientation_plots(run_dir, db_conn=db_conn)
                    if viz_results.get('status') != 'success': logger.error("Orientation visualization failed.") # Log error but might not be critical failure
                    logger.info(f"Orientation visualization completed in {time.time() - start_time_orient_viz:.2f} seconds.")
                # (Logging for skipping viz remains the same)

        # --- Ion Analysis Module ---
        ion_analysis_done = False
        ion_comp_status = get_module_status(db_conn, "ion_analysis")
        ion_viz_status = get_module_status(db_conn, "ion_analysis_visualization")

        if run_ions:
            needs_ion_comp_run = args.force_rerun or ion_comp_status != "success"
            needs_ion_viz_run = generate_plots and (args.force_rerun or ion_viz_status != "success")

            if needs_ion_comp_run or (needs_ion_viz_run and ion_comp_status != 'success'):
                logger.info(f"Running ion analysis computation (Needed: {needs_ion_comp_run})...")
                start_time_ion_comp = time.time()
                ion_comp_results = run_ion_analysis(run_dir, psf_file, dcd_file, db_conn=db_conn)
                ion_analysis_done = ion_comp_results.get('status') == 'success'
                logger.info(f"Ion analysis computation completed in {time.time() - start_time_ion_comp:.2f} seconds. Success: {ion_analysis_done}")
                if not ion_analysis_done:
                    logger.error(f"Ion analysis computation failed: {ion_comp_results.get('error', 'Unknown')}")
                    current_step_success = False
                    raise RuntimeError("Ion analysis computation failed.")
            else:
                # (Skip logic remains the same)
                ion_analysis_done = True

            if ion_analysis_done and needs_ion_viz_run and ion_analysis_available:
                logger.info("Generating ion analysis plots...")
                start_time_ion_viz = time.time()
                viz_results = generate_ion_plots(run_dir, db_conn=db_conn)
                if viz_results.get('status') != 'success': logger.error("Ion analysis visualization failed.")
                logger.info(f"Ion analysis visualization completed in {time.time() - start_time_ion_viz:.2f} seconds.")
            # (Logging for skipping viz remains the same)

        # --- Inner Vestibule Analysis Module ---
        inner_vestibule_analysis_done = False
        vestibule_comp_status = get_module_status(db_conn, "inner_vestibule_analysis")
        vestibule_viz_status = get_module_status(db_conn, "inner_vestibule_analysis_visualization")

        if run_water:
            needs_vestibule_comp_run = args.force_rerun or vestibule_comp_status != "success"
            needs_vestibule_viz_run = generate_plots and (args.force_rerun or vestibule_viz_status != "success")

            if needs_vestibule_comp_run or (needs_vestibule_viz_run and vestibule_comp_status != 'success'):
                logger.info(f"Running inner vestibule analysis computation (Needed: {needs_vestibule_comp_run})...")
                start_time_vestibule_comp = time.time()
                vestibule_comp_results = run_inner_vestibule_analysis(run_dir, psf_file, dcd_file, db_conn=db_conn)
                inner_vestibule_analysis_done = vestibule_comp_results.get('status') == 'success'
                logger.info(f"Inner vestibule computation completed in {time.time() - start_time_vestibule_comp:.2f} seconds. Success: {inner_vestibule_analysis_done}")
                if not inner_vestibule_analysis_done:
                    logger.error(f"Inner vestibule computation failed: {vestibule_comp_results.get('error', 'Unknown')}")
                    current_step_success = False
                    raise RuntimeError("Inner vestibule analysis computation failed.")
            else:
                # (Skip logic remains the same)
                inner_vestibule_analysis_done = True

            if inner_vestibule_analysis_done and needs_vestibule_viz_run and inner_vestibule_analysis_available:
                logger.info("Generating inner vestibule analysis plots...")
                start_time_vestibule_viz = time.time()
                viz_results = generate_inner_vestibule_plots(run_dir, db_conn=db_conn)
                if viz_results.get('status') != 'success': logger.error("Inner vestibule visualization failed.")
                logger.info(f"Inner vestibule visualization completed in {time.time() - start_time_vestibule_viz:.2f} seconds.")
            # (Logging for skipping viz remains the same)

        # --- Gyration Analysis Module ---
        gyration_analysis_done = False
        gyration_comp_status = get_module_status(db_conn, "gyration_analysis")
        gyration_viz_status = get_module_status(db_conn, "gyration_analysis_visualization")

        if run_gyration:
            needs_gyration_comp_run = args.force_rerun or gyration_comp_status != "success"
            needs_gyration_viz_run = generate_plots and (args.force_rerun or gyration_viz_status != "success")

            if not ion_analysis_done: # Check dependency
                 logger.warning("Cannot run Gyration analysis: Ion analysis did not complete successfully.")
                 if gyration_comp_status != 'skipped': update_module_status(db_conn, "gyration_analysis", 'skipped', error_message="Dependency failed: ion_analysis")
                 if gyration_viz_status != 'skipped': update_module_status(db_conn, "gyration_analysis_visualization", 'skipped', error_message="Dependency failed: ion_analysis")
                 # Treat dependency failure as skip, not critical failure for overall run unless specifically desired
            elif needs_gyration_comp_run or (needs_gyration_viz_run and gyration_comp_status != 'success'):
                logger.info(f"Running gyration analysis computation (Needed: {needs_gyration_comp_run})...")
                start_time_gyration_comp = time.time()
                gyration_comp_results = run_gyration_analysis(run_dir, psf_file, dcd_file, db_conn=db_conn)
                gyration_analysis_done = gyration_comp_results.get('status') == 'success'
                logger.info(f"Gyration analysis computation completed in {time.time() - start_time_gyration_comp:.2f} seconds. Success: {gyration_analysis_done}")
                if not gyration_analysis_done:
                    logger.error(f"Gyration analysis computation failed: {gyration_comp_results.get('error', 'Unknown')}")
                    current_step_success = False
                    raise RuntimeError("Gyration analysis computation failed.")
            else:
                # (Skip logic remains the same)
                gyration_analysis_done = True

            if gyration_analysis_done and needs_gyration_viz_run and gyration_analysis_available:
                logger.info("Generating gyration analysis plots...")
                start_time_gyration_viz = time.time()
                viz_results = generate_gyration_plots(run_dir, db_conn=db_conn)
                if viz_results.get('status') != 'success': logger.error("Gyration analysis visualization failed.")
                logger.info(f"Gyration analysis visualization completed in {time.time() - start_time_gyration_viz:.2f} seconds.")
            # (Logging for skipping viz remains the same)

        # --- Tyrosine Analysis Module ---
        tyrosine_analysis_done = False
        tyrosine_comp_status = get_module_status(db_conn, "tyrosine_analysis")
        tyrosine_viz_status = get_module_status(db_conn, "tyrosine_analysis_visualization")

        if run_tyrosine:
            needs_tyrosine_comp_run = args.force_rerun or tyrosine_comp_status != "success"
            needs_tyrosine_viz_run = generate_plots and (args.force_rerun or tyrosine_viz_status != "success")

            if not ion_analysis_done: # Check dependency
                 logger.warning("Cannot run Tyrosine analysis: Ion analysis did not complete successfully.")
                 if tyrosine_comp_status != 'skipped': update_module_status(db_conn, "tyrosine_analysis", 'skipped', error_message="Dependency failed: ion_analysis")
                 if tyrosine_viz_status != 'skipped': update_module_status(db_conn, "tyrosine_analysis_visualization", 'skipped', error_message="Dependency failed: ion_analysis")
            elif needs_tyrosine_comp_run or (needs_tyrosine_viz_run and tyrosine_comp_status != 'success'):
                logger.info(f"Running tyrosine analysis computation (Needed: {needs_tyrosine_comp_run})...")
                start_time_tyrosine_comp = time.time()
                tyrosine_comp_results = run_tyrosine_analysis(run_dir, psf_file, dcd_file, db_conn=db_conn)
                tyrosine_analysis_done = tyrosine_comp_results.get('status') == 'success'
                logger.info(f"Tyrosine analysis computation completed in {time.time() - start_time_tyrosine_comp:.2f} seconds. Success: {tyrosine_analysis_done}")
                if not tyrosine_analysis_done:
                    logger.error(f"Tyrosine analysis computation failed: {tyrosine_comp_results.get('error', 'Unknown')}")
                    current_step_success = False
                    raise RuntimeError("Tyrosine analysis computation failed.")
            else:
                # (Skip logic remains the same)
                tyrosine_analysis_done = True

            if tyrosine_analysis_done and needs_tyrosine_viz_run and tyrosine_analysis_available:
                logger.info("Generating tyrosine analysis plots...")
                start_time_tyrosine_viz = time.time()
                viz_results = generate_tyrosine_plots(run_dir, db_conn=db_conn)
                if viz_results.get('status') != 'success': logger.error("Tyrosine analysis visualization failed.")
                logger.info(f"Tyrosine analysis visualization completed in {time.time() - start_time_tyrosine_viz:.2f} seconds.")
            # (Logging for skipping viz remains the same)

        # --- DW Gate Analysis Module ---
        dw_gate_analysis_done = False
        dw_gate_comp_status = get_module_status(db_conn, "dw_gate_analysis")
        dw_gate_viz_status = get_module_status(db_conn, "dw_gate_analysis_visualization")

        if run_dwgates:
            needs_dw_gate_comp_run = args.force_rerun or dw_gate_comp_status != "success"
            needs_dw_gate_viz_run = generate_plots and (args.force_rerun or dw_gate_viz_status != "success")

            filter_res_meta = get_simulation_metadata(db_conn, 'filter_residues_dict')
            if not filter_res_meta: # Check dependency
                 logger.warning("Cannot run DW Gate analysis: Filter residue dictionary not found in database metadata (dependency: ion_analysis).")
                 if dw_gate_comp_status != 'skipped': update_module_status(db_conn, "dw_gate_analysis", 'skipped', error_message="Dependency failed: filter_residues_dict")
                 if dw_gate_viz_status != 'skipped': update_module_status(db_conn, "dw_gate_analysis_visualization", 'skipped', error_message="Dependency failed: filter_residues_dict")
            elif needs_dw_gate_comp_run or (needs_dw_gate_viz_run and dw_gate_comp_status != 'success'):
                logger.info(f"Running DW Gate analysis computation (Needed: {needs_dw_gate_comp_run})...")
                start_time_dw_gate_comp = time.time()
                dw_gate_comp_results = run_dw_gate_analysis(run_dir, psf_file, dcd_file, db_conn=db_conn)
                dw_gate_analysis_done = dw_gate_comp_results.get('status') == 'success'
                logger.info(f"DW Gate analysis computation completed in {time.time() - start_time_dw_gate_comp:.2f} seconds. Success: {dw_gate_analysis_done}")
                if not dw_gate_analysis_done:
                    logger.error(f"DW Gate analysis computation failed: {dw_gate_comp_results.get('error', 'Unknown')}")
                    current_step_success = False
                    raise RuntimeError("DW Gate analysis computation failed.")
            else:
                # (Skip logic remains the same)
                dw_gate_analysis_done = True

            if dw_gate_analysis_done and needs_dw_gate_viz_run and dw_gate_analysis_available:
                logger.info("Generating DW Gate analysis plots...")
                start_time_dw_gate_viz = time.time()
                viz_results = generate_dw_gate_plots(run_dir, db_conn=db_conn)
                if viz_results.get('status') != 'success': logger.error("DW Gate analysis visualization failed.")
                logger.info(f"DW Gate analysis visualization completed in {time.time() - start_time_dw_gate_viz:.2f} seconds.")
            # (Logging for skipping viz remains the same)


        # ===========================================
        # --- End of Analysis Module Execution ------
        # ===========================================

        # --- Final Summary and Report ---
        logger.info("Generating final analysis summary...")
        summary = generate_summary_from_database(run_dir, db_conn)
        if not summary:
            logger.warning("Analysis summary generation returned empty result")
            # Consider if this should mark success as false?
            # For now, let previous step success dictate overall success

        if generate_html:
            if summary:
                logger.info("Generating HTML report...")
                report_path = generate_html_report(run_dir, summary=summary)
                if report_path:
                    logger.info(f"HTML report generated: {report_path}")
                else:
                    logger.warning("HTML report generation failed or returned no path")
                    # Do not mark overall as failure just for report unless required
            else:
                logger.error("Cannot generate HTML report: Failed to generate summary from database.")
                # Mark as failure if summary was needed for report?
                # current_step_success = False # Optional: Fail if summary fails when report needed
        else:
            logger.info("Skipping HTML report generation based on flags.")

        # --- Set Final Overall Success Status ---
        # Only set to success if all critical steps passed
        if current_step_success:
             overall_success = True
             set_simulation_metadata(db_conn, "analysis_status", "success")
             set_simulation_metadata(db_conn, "analysis_end_time", datetime.now().isoformat())
             db_conn.commit() # Commit success status
             logger.info(f"Overall analysis status for this run: success")
        else:
             # If current_step_success is False, an error was already logged and status set to failed by the failing step or the except block below will catch it.
             # No need to explicitly set failed here again unless we want to override a previous success state which shouldn't happen with this logic.
             logger.info(f"Overall analysis status for this run: failed (due to earlier step failure)")


    except Exception as e:
        # Catch any unhandled exceptions during the main workflow
        error_message = f"Unhandled Workflow Error: {e}"
        tb_str = traceback.format_exc() # Get detailed traceback
        logger.critical(f"{error_message}\n{tb_str}")
        overall_success = False # Ensure overall status is marked as failed
        # Try to update status and save error summary
        if db_conn:
            try:
                set_simulation_metadata(db_conn, "analysis_status", "failed")
                set_simulation_metadata(db_conn, "analysis_error", f"{error_message[:200]}...") # Store truncated error
                set_simulation_metadata(db_conn, "analysis_end_time", datetime.now().isoformat())
                db_conn.commit()
            except Exception as db_e:
                logger.error(f"Failed to update DB status to failed during critical error handling: {db_e}")
        # Save minimal error summary file
        _save_error_summary(run_dir, system_name, run_name, error_message)
        # Return None to indicate critical failure to the main() function
        return None

    finally:
        # --- Ensure DB connection is always closed ---
        if db_conn:
            try:
                # Final commit just in case (e.g., metadata updates)
                db_conn.commit()
                db_conn.close()
                logger.info("Database connection closed.")
            except Exception as db_e:
                logger.error(f"Error closing database connection: {db_e}")

    return summary if overall_success and summary else {} # Return summary only on success, else empty dict


# --- Main Execution Guard ---
def main():
    """Main entry point."""
    # Ensure core utilities were imported
    if not core_utils_available:
        # Use standard print before logging is configured
        print("Critical Error: Core utilities could not be imported. Cannot proceed.", file=sys.stderr)
        sys.exit(1)

    print(f"Starting Pore Analysis Suite v{Analysis_version}...")
    args = parse_arguments()

    # Set up logging
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    abs_run_dir = os.path.abspath(args.folder)
    run_name = os.path.basename(abs_run_dir)
    log_file = setup_analysis_logger(abs_run_dir, run_name, log_level)

    if log_file: logger.info(f"Logging to: {log_file}")
    else: print(f"Warning: Failed to create log file in {abs_run_dir}", file=sys.stderr) # Use print if logger failed

    logger.info(f"--- Analysis started for {abs_run_dir} at {datetime.now()} ---")
    logger.info(f"Command line arguments: {vars(args)}")
    start_run_time = time.time()

    # Run the workflow
    # _run_analysis_workflow now returns the summary dict on success, {} on handled failure, or None on critical failure
    results = _run_analysis_workflow(args)

    end_run_time = time.time()
    logger.info(f"--- Analysis finished for {abs_run_dir} at {datetime.now()} (Duration: {end_run_time - start_run_time:.2f} sec) ---")

    # Determine exit code based on the result
    if results is None:
        # Critical failure caught in the workflow's except block
        logger.error("Workflow failed critically.")
        return 1
    elif isinstance(results, dict) and results.get('metadata', {}).get('analysis_status') == 'success':
        # Workflow completed successfully (might have warnings, but no critical errors)
        logger.info("Workflow completed successfully.")
        return 0
    else:
        # Workflow completed but ended in a 'failed' state (e.g., a module failed but wasn't caught as critical)
        # Or summary generation failed after successful analysis steps.
        logger.error("Workflow completed, but the final status was 'failed' or summary was missing.")
        return 1


if __name__ == "__main__":
    sys.exit(main())