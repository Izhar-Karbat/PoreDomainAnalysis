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
from typing import Optional, Dict, Any
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
    # Pocket Analysis
    from pore_analysis.modules.pocket_analysis import (
        run_pocket_analysis, generate_pocket_plots
    )
    pocket_analysis_available = True
except ImportError as e_pocket:
    pocket_analysis_available = False
    # Capture specific error message for better diagnostics
    pocket_error_msg = str(e_pocket)
    # Check for the specific torchmdnet dependency error
    if "torchmdnet" in pocket_error_msg:
        pocket_unavailable_reason = f"Pocket analysis unavailable: Missing torchmdnet dependency. {pocket_error_msg}"
    else:
        pocket_unavailable_reason = f"Pocket analysis unavailable: {pocket_error_msg}"
    
    # Log at import time so it's always visible
    if 'logging' in globals():
        logging.warning(pocket_unavailable_reason)
    else:
        print(f"WARNING: {pocket_unavailable_reason}", file=sys.stderr)
    
    def run_pocket_analysis(*args, **kwargs): logging.warning(pocket_unavailable_reason); return {'status': 'skipped', 'error': pocket_unavailable_reason}
    def generate_pocket_plots(*args, **kwargs): logging.warning(pocket_unavailable_reason); return {'status': 'skipped', 'error': pocket_unavailable_reason}

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
    from pore_analysis.print_report import generate_print_report, convert_to_pdf
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
    analysis_group.add_argument("--pocket", action="store_true", help="Run Peripheral Pocket Water analysis (requires ML model).")

    # --- Other Options / Report Control ---
    other_group = parser.add_argument_group('Other Options')
    other_group.add_argument("--box_z", type=float, default=None, help="Provide estimated box Z-dimension (Angstroms) for multi-level COM filter.")
    other_group.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Set the logging level.")
    other_group.add_argument("--report", action="store_true", help="Generate HTML report when running selective analyses.")
    other_group.add_argument("--no-report", action="store_true", help="Suppress HTML report generation when running full analysis (default or --force-rerun).")
    other_group.add_argument("--no-print-report", action="store_true", help="Suppress print-friendly HTML report generation (generated by default).")
    other_group.add_argument("--no-pdf", action="store_true", help="Suppress PDF report generation (generated by default).")
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


# --- Main Workflow Function (Revised with Status Update Before Summary) ---
def _run_analysis_workflow(args):
    """
    Main analysis workflow function using database tracking.
    Applies logic based on original script's flag handling.
    Includes try/except/finally block for robust status updates.
    Sets final status BEFORE generating summary.
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
    # Assume success unless a computation step explicitly fails
    current_step_success = True
    final_status_for_db = "failed" # Default to failed unless explicitly set to success
    run_name = os.path.basename(run_dir) # Define here for use in except/finally
    parent_dir = os.path.dirname(run_dir)
    system_name = os.path.basename(parent_dir) if parent_dir and os.path.basename(parent_dir) != '.' else run_name

    try:
        # --- Initialize Database ---
        db_conn = init_db(run_dir, force_recreate=args.reinit_db)
        if db_conn is None:
            # Error already logged by init_db
            return None # Cannot proceed without DB

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
            
        # Default initialization of generate_html to prevent UnboundLocalError
        generate_html = False

        # --- Summary-Only Mode ---
        if args.summary_only:
            logger.info("Summary-only mode. Skipping analysis execution.")
            summary = generate_summary_from_database(run_dir, db_conn)
            if not summary:
                 logger.error("Failed to generate summary from database.")
                 current_step_success = False # Mark as failure if summary fails in this mode
                 raise RuntimeError("Failed to generate summary in summary-only mode.")
            
            # Set generate_html based on --report flag for summary-only mode
            generate_html = args.report

            if args.report:
                logger.info("Generating HTML report...")
                report_path = generate_html_report(run_dir, summary=summary)
                if not report_path:
                    logger.warning("HTML report generation failed or returned no path")
                else:
                    logger.info(f"HTML report generated: {report_path}")
            else:
                logger.info("HTML report generation skipped in summary-only mode.")
                
            if not args.no_print_report:
                logger.info("Generating print-friendly report...")
                print_report_path = generate_print_report(run_dir)
                if print_report_path:
                    logger.info(f"Print-friendly report generated: {print_report_path}")
                    
                    # Convert to PDF by default, unless explicitly suppressed
                    if not args.no_pdf:
                        logger.info("Converting report to PDF...")
                        pdf_path = convert_to_pdf(print_report_path)
                        if pdf_path:
                            logger.info(f"PDF report generated: {pdf_path}")
                        else:
                            logger.warning("PDF conversion failed. Print-friendly HTML report is still available.")
                else:
                    logger.warning("Print-friendly report generation failed or returned no path")

            current_step_success = True # Summary generation succeeded
            # Skip the rest of the analysis, jump towards finalization
        else:
            # --- Determine which modules to run ---
            specific_flags_set = (args.core or args.orientation or args.ions or 
                                args.water or args.gyration or args.tyrosine or 
                                args.dwgates or args.pocket) # Add args.pocket
            run_all_initially = not specific_flags_set

            run_core = (args.core or run_all_initially) and core_analysis_available
            run_orientation = (args.orientation or run_all_initially) and orientation_analysis_available
            run_ions = (args.ions or run_all_initially) and ion_analysis_available
            run_water = (args.water or run_all_initially) and inner_vestibule_analysis_available
            run_gyration = (args.gyration or run_all_initially) and gyration_analysis_available
            run_tyrosine = (args.tyrosine or run_all_initially) and tyrosine_analysis_available
            run_dwgates = (args.dwgates or run_all_initially) and dw_gate_analysis_available
            run_pocket = (args.pocket or run_all_initially) and pocket_analysis_available
            
            # Log availability status for modules when running all (explicitly note when a module is skipped)
            if run_all_initially:
                logger.info("No specific analysis modules requested - determining available modules...")
                if not core_analysis_available:
                    logger.warning("Core analysis module is NOT available. This module will be skipped.")
                if not orientation_analysis_available:
                    logger.warning("Orientation analysis module is NOT available. This module will be skipped.")
                if not ion_analysis_available:
                    logger.warning("Ion analysis module is NOT available. This module will be skipped.")
                if not inner_vestibule_analysis_available:
                    logger.warning("Inner vestibule analysis module is NOT available. This module will be skipped.")
                if not gyration_analysis_available:
                    logger.warning("Gyration analysis module is NOT available. This module will be skipped.")
                if not tyrosine_analysis_available:
                    logger.warning("Tyrosine analysis module is NOT available. This module will be skipped.")
                if not dw_gate_analysis_available:
                    logger.warning("DW-Gate analysis module is NOT available. This module will be skipped.")
                if not pocket_analysis_available:
                    if 'pocket_unavailable_reason' in globals():
                        logger.warning(f"{pocket_unavailable_reason} This module will be skipped.")
                    else:
                        logger.warning("Pocket analysis module is NOT available (check torch/torchmdnet installation?). This module will be skipped.")

            generate_plots = not args.no_plots

            if specific_flags_set:
                generate_html = args.report
            else: # Run all default case
                generate_html = not args.no_report

            # Log module availability warnings with specific reasons when possible
            if args.core and not core_analysis_available: 
                logger.warning("Core analysis requested but module is not available/imported.")
            if args.orientation and not orientation_analysis_available: 
                logger.warning("Orientation analysis requested but module is not available/imported.")
            if args.ions and not ion_analysis_available: 
                logger.warning("Ion analysis requested but module is not available/imported.")
            if args.water and not inner_vestibule_analysis_available: 
                logger.warning("Inner Vestibule analysis requested but module is not available/imported.")
            if args.gyration and not gyration_analysis_available: 
                logger.warning("Gyration analysis requested but module is not available/imported.")
            if args.tyrosine and not tyrosine_analysis_available: 
                logger.warning("Tyrosine analysis requested but module is not available/imported.")
            if args.dwgates and not dw_gate_analysis_available: 
                logger.warning("DW Gate analysis requested but module is not available/imported.")
            if args.pocket and not pocket_analysis_available: 
                if 'pocket_unavailable_reason' in globals():
                    logger.warning(pocket_unavailable_reason)
                else:
                    logger.warning("Pocket analysis requested but module is not available/imported (check torch/torchmdnet installation?).")


            logger.info(f"Analysis Plan: Core={run_core}, Orientation={run_orientation}, Ions={run_ions}, Water={run_water}, Gyration={run_gyration}, Tyrosine={run_tyrosine}, DWGates={run_dwgates}, Pocket={run_pocket}")
            logger.info(f"Generate Plots: {generate_plots}")
            logger.info(f"Generate HTML Report: {generate_html}")
            logger.info(f"Force Rerun: {args.force_rerun}")

            # ===========================================
            # --- Start of Analysis Module Execution ----
            # ===========================================
            # current_step_success is True initially

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
                        # Raise immediately on critical step failure
                        raise RuntimeError(f"Core analysis filtering failed: {filtering_results.get('error', 'Unknown')}")
                elif core_results and core_results['status'] == 'skipped':
                    logger.warning("Core analysis computation skipped (module not found?)")
                    current_step_success = False # Unavailable = failure for workflow
                    raise RuntimeError("Core analysis module is unavailable.")
                else:
                    logger.error(f"Core: Trajectory analysis failed: {core_results.get('error', 'Unknown') if core_results else 'No results returned'}")
                    current_step_success = False
                    raise RuntimeError(f"Core trajectory analysis failed: {core_results.get('error', 'Unknown') if core_results else 'No results returned'}")
                logger.info(f"Core computation/filtering finished in {time.time() - start_time_core:.2f} seconds.")
            elif run_core:
                logger.info("Skipping core computation/filtering - already completed successfully.")
                core_analysis_done = True # Already done
            else: # Not running core
                core_analysis_done = core_filter_status == "success" # Check if previously done

            is_control_system = get_simulation_metadata(db_conn, 'is_control_system') == 'True'

            # Core Viz (Run if needed and computation was done or already successful)
            core_viz_gg_status = get_module_status(db_conn, "core_analysis_visualization_g_g")
            core_viz_com_status = get_module_status(db_conn, "core_analysis_visualization_com")
            needs_core_viz = generate_plots and (args.force_rerun or core_viz_gg_status != "success" or (not is_control_system and core_viz_com_status != "success"))
            if core_analysis_done and needs_core_viz and core_analysis_available:
                logger.info("Generating core analysis plots...")
                try:
                    plot_distances(run_dir, is_gg=True, db_conn=db_conn)
                    if not is_control_system:
                        plot_distances(run_dir, is_gg=False, db_conn=db_conn)
                        plot_kde_analysis(run_dir, db_conn=db_conn, box_z=args.box_z)
                except Exception as e_plot_core:
                     logger.error(f"Core plotting failed: {e_plot_core}", exc_info=True)
                     # Log as error, but might not be a critical workflow failure
            elif core_analysis_done and generate_plots and not needs_core_viz:
                logger.info("Skipping core plot generation - already completed successfully.")
            elif not core_analysis_done and generate_plots:
                logger.warning("Cannot generate core plots - computation/filtering did not complete successfully.")

            # --- Run Other Modules (Condensed - Apply same pattern as Core/Orientation) ---
            # Helper function to run a module computation step
            def run_computation_step(module_flag, run_needed, module_name_comp, run_func, deps_ok=True, dep_err_msg="Dependency failed"):
                nonlocal current_step_success, db_conn, run_dir, psf_file, dcd_file
                step_done = False
                if module_flag:
                    if not deps_ok:
                         logger.warning(f"Cannot run {module_name_comp}: {dep_err_msg}")
                         if get_module_status(db_conn, module_name_comp) != 'skipped': update_module_status(db_conn, module_name_comp, 'skipped', error_message=dep_err_msg)
                         # Treat dependency failure as skip, not necessarily overall failure
                         step_done = get_module_status(db_conn, module_name_comp) == 'success' # Check if maybe done previously?
                    elif run_needed:
                         logger.info(f"Running {module_name_comp} computation...")
                         start_comp = time.time()
                         comp_results = run_func(run_dir, psf_file, dcd_file, db_conn=db_conn)
                         step_done = comp_results.get('status') == 'success'
                         logger.info(f"{module_name_comp} computation finished in {time.time() - start_comp:.2f} sec. Success: {step_done}")
                         if not step_done:
                              err = comp_results.get('error', 'Unknown')
                              logger.error(f"{module_name_comp} computation failed: {err}")
                              current_step_success = False
                              raise RuntimeError(f"{module_name_comp} computation failed: {err}")
                    else:
                         logger.info(f"Skipping {module_name_comp} computation - already completed successfully.")
                         step_done = True # Already done successfully
                else:
                    step_done = get_module_status(db_conn, module_name_comp) == 'success'
                return step_done

            # Helper function to run a module visualization step
            def run_visualization_step(module_flag, computation_done, run_needed, module_name_viz, gen_plots_func, module_available):
                 nonlocal generate_plots, db_conn, run_dir
                 if module_flag and computation_done and run_needed and module_available:
                     logger.info(f"Generating {module_name_viz} plots...")
                     start_viz = time.time()
                     viz_results = gen_plots_func(run_dir, db_conn=db_conn)
                     if viz_results.get('status') != 'success': logger.error(f"{module_name_viz} visualization failed: {viz_results.get('error', 'Unknown')}") # Log error but don't fail workflow
                     logger.info(f"{module_name_viz} visualization finished in {time.time() - start_viz:.2f} sec.")
                 elif module_flag and computation_done and generate_plots and not run_needed:
                      logger.info(f"Skipping {module_name_viz} plot generation - already completed successfully.")
                 elif module_flag and not computation_done and generate_plots:
                      logger.warning(f"Cannot generate {module_name_viz} plots - computation did not complete successfully.")

            # Orientation
            orient_comp_status = get_module_status(db_conn, "orientation_analysis")
            orient_viz_status = get_module_status(db_conn, "orientation_analysis_visualization")
            needs_orient_comp_run = args.force_rerun or orient_comp_status != "success"
            needs_orient_viz_run = generate_plots and (args.force_rerun or orient_viz_status != "success")
            orientation_analysis_done = run_computation_step(run_orientation, needs_orient_comp_run, "orientation_analysis", run_orientation_analysis, deps_ok=(not is_control_system), dep_err_msg="Control system")
            run_visualization_step(run_orientation, orientation_analysis_done, needs_orient_viz_run, "orientation_analysis_visualization", generate_orientation_plots, orientation_analysis_available)

            # Ion Analysis
            ion_comp_status = get_module_status(db_conn, "ion_analysis")
            ion_viz_status = get_module_status(db_conn, "ion_analysis_visualization")
            needs_ion_comp_run = args.force_rerun or ion_comp_status != "success"
            needs_ion_viz_run = generate_plots and (args.force_rerun or ion_viz_status != "success")
            ion_analysis_done = run_computation_step(run_ions, needs_ion_comp_run, "ion_analysis", run_ion_analysis)
            run_visualization_step(run_ions, ion_analysis_done, needs_ion_viz_run, "ion_analysis_visualization", generate_ion_plots, ion_analysis_available)

            # Inner Vestibule
            vestibule_comp_status = get_module_status(db_conn, "inner_vestibule_analysis")
            vestibule_viz_status = get_module_status(db_conn, "inner_vestibule_analysis_visualization")
            needs_vestibule_comp_run = args.force_rerun or vestibule_comp_status != "success"
            needs_vestibule_viz_run = generate_plots and (args.force_rerun or vestibule_viz_status != "success")
            # Dependency: Requires filter_residues_dict metadata from ion_analysis run
            filter_res_meta_for_vest = get_simulation_metadata(db_conn, 'filter_residues_dict')
            vestibule_analysis_done = run_computation_step(run_water, needs_vestibule_comp_run, "inner_vestibule_analysis", run_inner_vestibule_analysis, deps_ok=(filter_res_meta_for_vest is not None), dep_err_msg="filter_residues_dict missing")
            run_visualization_step(run_water, vestibule_analysis_done, needs_vestibule_viz_run, "inner_vestibule_analysis_visualization", generate_inner_vestibule_plots, inner_vestibule_analysis_available)

            # Gyration
            gyration_comp_status = get_module_status(db_conn, "gyration_analysis")
            gyration_viz_status = get_module_status(db_conn, "gyration_analysis_visualization")
            needs_gyration_comp_run = args.force_rerun or gyration_comp_status != "success"
            needs_gyration_viz_run = generate_plots and (args.force_rerun or gyration_viz_status != "success")
            # Dependency: Ion analysis computation must be done/successful
            gyration_analysis_done = run_computation_step(run_gyration, needs_gyration_comp_run, "gyration_analysis", run_gyration_analysis, deps_ok=ion_analysis_done, dep_err_msg="Ion analysis failed")
            run_visualization_step(run_gyration, gyration_analysis_done, needs_gyration_viz_run, "gyration_analysis_visualization", generate_gyration_plots, gyration_analysis_available)

            # Tyrosine
            tyrosine_comp_status = get_module_status(db_conn, "tyrosine_analysis")
            tyrosine_viz_status = get_module_status(db_conn, "tyrosine_analysis_visualization")
            needs_tyrosine_comp_run = args.force_rerun or tyrosine_comp_status != "success"
            needs_tyrosine_viz_run = generate_plots and (args.force_rerun or tyrosine_viz_status != "success")
            # Dependency: Ion analysis computation must be done/successful
            tyrosine_analysis_done = run_computation_step(run_tyrosine, needs_tyrosine_comp_run, "tyrosine_analysis", run_tyrosine_analysis, deps_ok=ion_analysis_done, dep_err_msg="Ion analysis failed")
            run_visualization_step(run_tyrosine, tyrosine_analysis_done, needs_tyrosine_viz_run, "tyrosine_analysis_visualization", generate_tyrosine_plots, tyrosine_analysis_available)

            # DW Gate
            dw_gate_comp_status = get_module_status(db_conn, "dw_gate_analysis")
            dw_gate_viz_status = get_module_status(db_conn, "dw_gate_analysis_visualization")
            needs_dw_gate_comp_run = args.force_rerun or dw_gate_comp_status != "success"
            needs_dw_gate_viz_run = generate_plots and (args.force_rerun or dw_gate_viz_status != "success")
            # Dependency: Filter residues metadata from ion_analysis
            filter_res_meta_for_dw = get_simulation_metadata(db_conn, 'filter_residues_dict')
            dw_gate_analysis_done = run_computation_step(run_dwgates, needs_dw_gate_comp_run, "dw_gate_analysis", run_dw_gate_analysis, deps_ok=(filter_res_meta_for_dw is not None), dep_err_msg="filter_residues_dict missing")
            run_visualization_step(run_dwgates, dw_gate_analysis_done, needs_dw_gate_viz_run, "dw_gate_analysis_visualization", generate_dw_gate_plots, dw_gate_analysis_available)

            # Pocket Analysis
            pocket_comp_status = get_module_status(db_conn, "pocket_analysis")
            pocket_viz_status = get_module_status(db_conn, "pocket_analysis_visualization")
            needs_pocket_comp_run = args.force_rerun or pocket_comp_status != "success"
            needs_pocket_viz_run = generate_plots and (args.force_rerun or pocket_viz_status != "success")
            # Dependency: Requires filter_residues_dict metadata from a successful ion_analysis run
            filter_res_meta_for_pocket = get_simulation_metadata(db_conn, 'filter_residues_dict')
            # Use the ion_analysis_done flag which should be set correctly by the ion_analysis block above
            pocket_deps_ok = ion_analysis_done and (filter_res_meta_for_pocket is not None)
            pocket_dep_err_msg = "Ion analysis failed or filter_residues_dict missing"
            pocket_analysis_done = run_computation_step(
                run_pocket, needs_pocket_comp_run, "pocket_analysis",
                run_pocket_analysis, deps_ok=pocket_deps_ok, dep_err_msg=pocket_dep_err_msg
            )
            run_visualization_step(
                run_pocket, pocket_analysis_done, needs_pocket_viz_run,
                "pocket_analysis_visualization", generate_pocket_plots,
                pocket_analysis_available
            )

            # ===========================================
            # --- End of Analysis Module Execution ------
            # ===========================================

        # --- Set Final Overall Status BEFORE Summary ---
        final_status_for_db = 'success' if current_step_success else 'failed'
        set_simulation_metadata(db_conn, "analysis_status", final_status_for_db)
        set_simulation_metadata(db_conn, "analysis_end_time", datetime.now().isoformat())
        db_conn.commit() # Commit final status *before* summary generation
        logger.info(f"Overall analysis status for this run set to: {final_status_for_db}")

        # --- Final Summary and Report ---
        logger.info("Generating final analysis summary...")
        summary = generate_summary_from_database(run_dir, db_conn)
        if not summary:
            logger.warning("Analysis summary generation returned empty result")
            summary = {} # Ensure summary is at least an empty dict

        if generate_html:
            if summary:
                logger.info("Generating HTML report...")
                report_path = generate_html_report(run_dir, summary=summary)
                if report_path:
                    logger.info(f"HTML report generated: {report_path}")
                else:
                    logger.warning("HTML report generation failed or returned no path")
            else:
                logger.error("Cannot generate HTML report: Failed to generate summary from database.")
        
        # Print-report generation (by default, unless explicitly suppressed)
        if not args.no_print_report:
            if summary:
                logger.info("Generating print-friendly report...")
                print_report_path = generate_print_report(run_dir)
                if print_report_path:
                    logger.info(f"Print-friendly report generated: {print_report_path}")
                    
                    # Convert to PDF by default, unless explicitly suppressed
                    if not args.no_pdf:
                        logger.info("Converting report to PDF...")
                        pdf_path = convert_to_pdf(print_report_path)
                        if pdf_path:
                            logger.info(f"PDF report generated: {pdf_path}")
                        else:
                            logger.warning("PDF conversion failed. Print-friendly HTML report is still available.")
                else:
                    logger.warning("Print-friendly report generation failed or returned no path")
            else:
                logger.error("Cannot generate print-friendly report: Failed to generate summary from database.")

    except Exception as e:
        # Catch any unhandled exceptions during the main workflow
        error_message = f"Unhandled Workflow Error: {e}"
        tb_str = traceback.format_exc() # Get detailed traceback
        logger.critical(f"{error_message}\n{tb_str}")
        current_step_success = False # Mark as failed
        final_status_for_db = "failed" # Ensure final status is marked failed
        # Try to update status and save error summary
        if db_conn:
            try:
                # Ensure status is marked failed even if it happened after modules
                set_simulation_metadata(db_conn, "analysis_status", "failed")
                set_simulation_metadata(db_conn, "analysis_error", f"{error_message[:200]}...") # Store truncated error
                set_simulation_metadata(db_conn, "analysis_end_time", datetime.now().isoformat())
                db_conn.commit()
            except Exception as db_e:
                logger.error(f"Failed to update DB status to failed during critical error handling: {db_e}")
        # Save minimal error summary file
        _save_error_summary(run_dir, system_name, run_name, error_message)
        summary = None # Indicate critical failure for return value

    finally:
        # --- Ensure DB connection is always closed ---
        if db_conn:
            try:
                db_conn.commit() # Final commit just in case
                db_conn.close()
                logger.info("Database connection closed.")
            except Exception as db_e:
                logger.error(f"Error closing database connection: {db_e}")

    # Return summary object only if the core analysis steps succeeded
    # and summary generation produced a result. Otherwise return {} or None.
    if final_status_for_db == 'success' and isinstance(summary, dict) and summary:
         return summary
    elif final_status_for_db == 'success' and (not isinstance(summary, dict) or not summary):
         logger.warning("Workflow computation succeeded, but summary generation failed or returned empty. Returning empty dict.")
         return {}
    else: # final_status_for_db was 'failed'
         logger.error(f"Workflow failed during computation steps. Final status: {final_status_for_db}. Returning None.")
         return None # Return None if a critical computation error occurred

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
    results = _run_analysis_workflow(args)

    end_run_time = time.time()
    logger.info(f"--- Analysis finished for {abs_run_dir} at {datetime.now()} (Duration: {end_run_time - start_run_time:.2f} sec) ---")

    # Determine exit code based on the return value
    if results is None:
        # Critical failure caught in the workflow's except block
        logger.error("Workflow failed critically.")
        return 1
    elif isinstance(results, dict) and results:
        # Workflow completed successfully AND summary was generated
        logger.info("Workflow completed successfully.")
        return 0
    elif isinstance(results, dict) and not results:
         # Workflow computation likely succeeded, but summary generation failed
         logger.error("Workflow computation finished, but summary generation failed or returned empty. Final status likely 'success' in DB, but returning error code.")
         return 1 # Return error code because summary step failed
    else:
         # Should not happen with current return logic, but catch-all
         logger.error("Workflow finished with an unexpected result state.")
         return 1


if __name__ == "__main__":
    sys.exit(main())