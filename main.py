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

# --- Import analysis modules with error handling ---
try:
    # Core Analysis
    from pore_analysis.modules.core_analysis import (
        analyze_trajectory, filter_and_save_data, plot_distances, plot_kde_analysis
    )
    core_analysis_available = True
except ImportError:
    core_analysis_available = False
    # Define dummy functions if module not found, allows script to run partially
    def analyze_trajectory(*args, **kwargs): logger.warning("Core analysis module not found."); return {'status': 'skipped', 'error': 'Module not found'}
    def filter_and_save_data(*args, **kwargs): logger.warning("Core analysis module not found."); return {'status': 'skipped', 'error': 'Module not found'}
    def plot_distances(*args, **kwargs): logger.warning("Core analysis module not found."); return {}
    def plot_kde_analysis(*args, **kwargs): logger.warning("Core analysis module not found."); return None

try:
    # Orientation Analysis
    from pore_analysis.modules.orientation_contacts import (
        run_orientation_analysis, generate_orientation_plots
    )
    orientation_analysis_available = True
except ImportError:
    orientation_analysis_available = False
    def run_orientation_analysis(*args, **kwargs): logger.warning("Orientation analysis module not found."); return {'status': 'skipped', 'error': 'Module not found'}
    def generate_orientation_plots(*args, **kwargs): logger.warning("Orientation analysis module not found."); return {'status': 'skipped', 'error': 'Module not found'}

try:
    # Ion Analysis
    from pore_analysis.modules.ion_analysis import (
        run_ion_analysis, generate_ion_plots
    )
    ion_analysis_available = True
except ImportError:
    ion_analysis_available = False
    def run_ion_analysis(*args, **kwargs): logger.warning("Ion analysis module not found."); return {'status': 'skipped', 'error': 'Module not found'}
    def generate_ion_plots(*args, **kwargs): logger.warning("Ion analysis module not found."); return {'status': 'skipped', 'error': 'Module not found'}

try:
    # Inner Vestibule Analysis
    from pore_analysis.modules.inner_vestibule_analysis import (
        run_inner_vestibule_analysis, generate_inner_vestibule_plots
    )
    inner_vestibule_analysis_available = True
except ImportError:
    inner_vestibule_analysis_available = False
    def run_inner_vestibule_analysis(*args, **kwargs): logger.warning("Inner vestibule analysis module not found."); return {'status': 'skipped', 'error': 'Module not found'}
    def generate_inner_vestibule_plots(*args, **kwargs): logger.warning("Inner vestibule analysis module not found."); return {'status': 'skipped', 'error': 'Module not found'}

try:
    # Gyration Analysis
    from pore_analysis.modules.gyration_analysis import (
        run_gyration_analysis, generate_gyration_plots
    )
    gyration_analysis_available = True
except ImportError:
    gyration_analysis_available = False
    def run_gyration_analysis(*args, **kwargs): logger.warning("Gyration analysis module not found."); return {'status': 'skipped', 'error': 'Module not found'}
    def generate_gyration_plots(*args, **kwargs): logger.warning("Gyration analysis module not found."); return {'status': 'skipped', 'error': 'Module not found'}

try:
    # Tyrosine Analysis
    from pore_analysis.modules.tyrosine_analysis import (
         run_tyrosine_analysis, generate_tyrosine_plots
    )
    tyrosine_analysis_available = True
except ImportError:
    tyrosine_analysis_available = False
    # Define dummy functions if module not found
    def run_tyrosine_analysis(*args, **kwargs): logger.warning("Tyrosine analysis module not found."); return {'status': 'skipped', 'error': 'Module not found'}
    def generate_tyrosine_plots(*args, **kwargs): logger.warning("Tyrosine analysis module not found."); return {'status': 'skipped', 'error': 'Module not found'}

# --- Import DW Gate modules --- # MODIFIED: Integrated DW Gate
try:
    from pore_analysis.modules.dw_gate_analysis import (
        run_dw_gate_analysis, generate_dw_gate_plots
    )
    dw_gate_analysis_available = True
except ImportError:
    dw_gate_analysis_available = False
    def run_dw_gate_analysis(*args, **kwargs): logger.warning("DW Gate analysis module not found."); return {'status': 'skipped', 'error': 'Module not found'}
    def generate_dw_gate_plots(*args, **kwargs): logger.warning("DW Gate analysis module not found."); return {'status': 'skipped', 'error': 'Module not found'}
# --- End DW Gate Integration --- #

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
    print(f"Error importing CORE pore analysis modules/utilities: {e}")
    sys.exit(1)

# Configure logger
logger = logging.getLogger(__name__) # Get logger instance


# --- Argument Parsing (Based on Original Logic) ---
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
    analysis_group.add_argument("--dwgates", action="store_true", help="Run DW-Gate analysis.") # MODIFIED: Added dwgates flag

    # --- Other Options / Report Control ---
    other_group = parser.add_argument_group('Other Options')
    other_group.add_argument("--box_z", type=float, default=None, help="Provide estimated box Z-dimension (Angstroms) for multi-level COM filter.")
    other_group.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Set the logging level.")
    other_group.add_argument("--report", action="store_true", help="Generate HTML report when running selective analyses.")
    other_group.add_argument("--no-report", action="store_true", help="Suppress HTML report generation when running full analysis (default or --force-rerun).")
    other_group.add_argument("--no-plots", action="store_true", help="Skip generating plots for selected analyses.")

    return parser.parse_args()


# --- Error Summary Helper (From Original) ---
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


# --- Main Workflow Function (Applying Correct Logic) ---
def _run_analysis_workflow(args):
    """
    Main analysis workflow function using database tracking.
    Applies logic based on original script's flag handling.
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

    # --- Initialize Database ---
    db_conn: Optional[sqlite3.Connection] = init_db(run_dir, force_recreate=args.reinit_db)
    if db_conn is None:
        logger.error("Failed to initialize or connect to the database")
        return None

    run_name = os.path.basename(run_dir)
    parent_dir = os.path.dirname(run_dir)
    system_name = os.path.basename(parent_dir) if parent_dir and os.path.basename(parent_dir) != '.' else run_name

    # --- Set Initial Metadata and Store Config ---
    try:
        set_simulation_metadata(db_conn, "run_name", run_name)
        set_simulation_metadata(db_conn, "system_name", system_name)
        set_simulation_metadata(db_conn, "analysis_start_time", datetime.now().isoformat())
        set_simulation_metadata(db_conn, "psf_file", os.path.basename(psf_file))
        set_simulation_metadata(db_conn, "dcd_file", os.path.basename(dcd_file))
        set_simulation_metadata(db_conn, "analysis_version", Analysis_version)
        # Store config parameters from the imported module
        if 'core_config_module' in globals():
            store_config_parameters(db_conn, core_config_module)
            logger.info("Stored configuration parameters in database.")
        else:
            logger.warning("core_config module not imported, cannot store parameters.")
    except Exception as e:
        logger.error(f"Failed to set initial metadata or store config in database: {e}")
        if db_conn: db_conn.close()
        return None

    # --- Summary-Only Mode ---
    if args.summary_only:
        logger.info("Summary-only mode. Skipping analysis execution.")
        summary = generate_summary_from_database(run_dir, db_conn)
        if args.report:
            if summary:
                logger.info("Generating HTML report...")
                report_path = generate_html_report(run_dir, summary=summary)
                if report_path: logger.info(f"HTML report generated: {report_path}")
                else: logger.warning("HTML report generation failed or returned no path")
            else: logger.error("Cannot generate HTML report: Failed to generate summary from database.")
        elif args.no_report or not args.report:
            logger.info("HTML report generation skipped in summary-only mode (use --report to generate).")

        if db_conn: db_conn.close()
        return summary if summary else None

    # --- Determine which modules to run ---
    specific_flags_set = args.core or args.orientation or args.ions or args.water or args.gyration or args.tyrosine or args.dwgates
    run_all_initially = not specific_flags_set

    # Check module availability when deciding to run
    run_core = (args.core or run_all_initially) and core_analysis_available
    run_orientation = (args.orientation or run_all_initially) and orientation_analysis_available
    run_ions = (args.ions or run_all_initially) and ion_analysis_available
    run_water = (args.water or run_all_initially) and inner_vestibule_analysis_available
    run_gyration = (args.gyration or run_all_initially) and gyration_analysis_available
    run_tyrosine = (args.tyrosine or run_all_initially) and tyrosine_analysis_available
    run_dwgates = (args.dwgates or run_all_initially) and dw_gate_analysis_available # MODIFIED: Added dwgates check

    generate_plots = not args.no_plots

    if args.force_rerun:
        logger.info("Overriding analysis flags: --force_rerun specified, running ALL selected or default modules.")
    else:
        logger.info("Running based on selected flags or default (all). Use --force_rerun to re-execute completed modules.")

    if specific_flags_set:
        generate_html = args.report
        if args.no_report: logger.warning("--no-report flag has no effect when running selective analyses.")
    else: # Run all default case
        generate_html = not args.no_report
        if args.report: logger.warning("--report flag is redundant when running full analysis (report is generated by default).")

    # Log module availability warnings if flags were set but module not found
    if args.core and not core_analysis_available: logger.warning("Core analysis requested but module is not available/imported.")
    if args.orientation and not orientation_analysis_available: logger.warning("Orientation analysis requested but module is not available/imported.")
    if args.ions and not ion_analysis_available: logger.warning("Ion analysis requested but module is not available/imported.")
    if args.water and not inner_vestibule_analysis_available: logger.warning("Inner Vestibule analysis requested but module is not available/imported.")
    if args.gyration and not gyration_analysis_available: logger.warning("Gyration analysis requested but module is not available/imported.")
    if args.tyrosine and not tyrosine_analysis_available: logger.warning("Tyrosine analysis requested but module is not available/imported.")
    if args.dwgates and not dw_gate_analysis_available: logger.warning("DW Gate analysis requested but module is not available/imported.") # MODIFIED: Added dwgates check


    logger.info(f"Analysis Plan: Core={run_core}, Orientation={run_orientation}, Ions={run_ions}, Water={run_water}, Gyration={run_gyration}, Tyrosine={run_tyrosine}, DWGates={run_dwgates}") # MODIFIED: Added dwgates log
    logger.info(f"Generate Plots: {generate_plots}")
    logger.info(f"Generate HTML Report: {generate_html}")
    logger.info(f"Force Rerun: {args.force_rerun}")

    overall_success = True
    try:
        # --- Core Analysis Module ---
        core_analysis_done = False
        core_comp_status = get_module_status(db_conn, "core_analysis")
        core_filter_status = get_module_status(db_conn, "core_analysis_filtering")
        core_viz_gg_status = get_module_status(db_conn, "core_analysis_visualization_g_g")
        core_viz_com_status = get_module_status(db_conn, "core_analysis_visualization_com")
        needs_core_run = args.force_rerun or core_comp_status != "success" or core_filter_status != "success"

        if run_core and needs_core_run:
            logger.info("Running core distance computation and filtering...")
            start_time_core = time.time()
            # Ensure dummy function doesn't break flow if module missing
            core_results = analyze_trajectory(run_dir, psf_file, dcd_file) if core_analysis_available else {'status': 'skipped'}
            if core_results and core_results['status'] == 'success':
                set_simulation_metadata(db_conn, "is_control_system", str(core_results['metadata']['is_control_system']))
                filtering_results = filter_and_save_data(
                    run_dir, core_results['data']['dist_ac'], core_results['data']['dist_bd'],
                    core_results['data']['com_distances'], core_results['data']['time_points'],
                    is_control_system=core_results['metadata']['is_control_system'],
                    db_conn=db_conn, box_z=args.box_z
                ) if core_analysis_available else {'status': 'skipped'}
                if filtering_results and filtering_results['status'] == 'success':
                    core_analysis_done = True
                else:
                    logger.error(f"Core: Distance filtering failed: {filtering_results.get('error', 'Unknown')}")
                    overall_success = False
            elif core_results and core_results['status'] == 'skipped':
                 logger.warning("Core analysis computation skipped (module not found?)")
                 # If core is skipped, likely cannot proceed with dependent modules
                 overall_success = False
            else:
                logger.error(f"Core: Trajectory analysis failed: {core_results.get('error', 'Unknown') if core_results else 'No results returned'}")
                overall_success = False
            logger.info(f"Core computation/filtering finished in {time.time() - start_time_core:.2f} seconds.")
        elif run_core:
            logger.info("Skipping core computation/filtering - already completed successfully.")
            core_analysis_done = True # Still done if skipping completed run
        else: # Not running core
            core_analysis_done = core_filter_status == "success" # Check if previously done

        is_control_system = get_simulation_metadata(db_conn, 'is_control_system') == 'True'

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
        elif core_analysis_done and generate_plots and not needs_core_viz:
            logger.info("Skipping core plot generation - already completed successfully.")
        elif not core_analysis_done and generate_plots:
             logger.warning("Cannot generate core plots - computation/filtering did not complete successfully.")


        # --- Orientation Analysis Module ---
        # (Keep existing Orientation logic - check flags, DB status, run comp/viz)
        orientation_analysis_done = False
        orient_comp_status = get_module_status(db_conn, "orientation_analysis")
        orient_viz_status = get_module_status(db_conn, "orientation_analysis_visualization")

        if run_orientation:
            if is_control_system:
                logger.info("Skipping orientation analysis - This is a control system.")
                if orient_comp_status != 'skipped': update_module_status(db_conn, "orientation_analysis", 'skipped', error_message="Control system")
                if orient_viz_status != 'skipped': update_module_status(db_conn, "orientation_analysis_visualization", 'skipped', error_message="Control system")
                orientation_analysis_done = True
            else:
                needs_orient_comp_run = args.force_rerun or orient_comp_status != "success"
                needs_orient_viz_run = generate_plots and (args.force_rerun or orient_viz_status != "success")

                if needs_orient_comp_run or (needs_orient_viz_run and orient_comp_status != 'success'):
                    logger.info(f"Running orientation analysis computation (Needed: {needs_orient_comp_run})...")
                    start_time_orient_comp = time.time()
                    orient_comp_results = run_orientation_analysis(run_dir, psf_file, dcd_file, db_conn=db_conn) if orientation_analysis_available else {'status': 'skipped'}
                    orientation_analysis_done = orient_comp_results.get('status') == 'success'
                    if not orientation_analysis_done:
                        logger.error(f"Orientation computation failed: {orient_comp_results.get('error', 'Unknown')}")
                        overall_success = False
                    logger.info(f"Orientation computation completed in {time.time() - start_time_orient_comp:.2f} seconds. Success: {orientation_analysis_done}")
                else:
                    logger.info("Skipping orientation computation - already completed successfully.")
                    orientation_analysis_done = True

                if orientation_analysis_done and needs_orient_viz_run and orientation_analysis_available:
                    logger.info("Generating orientation analysis plots...")
                    start_time_orient_viz = time.time()
                    viz_results = generate_orientation_plots(run_dir, db_conn=db_conn)
                    if viz_results.get('status') != 'success': logger.error("Orientation visualization failed.")
                    logger.info(f"Orientation visualization completed in {time.time() - start_time_orient_viz:.2f} seconds.")
                elif orientation_analysis_done and generate_plots and not needs_orient_viz_run:
                     logger.info("Skipping orientation plot generation - already completed successfully.")
                elif not orientation_analysis_done and generate_plots:
                     logger.warning("Cannot generate orientation plots - computation did not complete successfully.")


        # --- Ion Analysis Module ---
        # (Keep existing Ion logic - check flags, DB status, run comp/viz)
        ion_analysis_done = False
        ion_comp_status = get_module_status(db_conn, "ion_analysis")
        ion_viz_status = get_module_status(db_conn, "ion_analysis_visualization")

        if run_ions:
            needs_ion_comp_run = args.force_rerun or ion_comp_status != "success"
            needs_ion_viz_run = generate_plots and (args.force_rerun or ion_viz_status != "success")

            if needs_ion_comp_run or (needs_ion_viz_run and ion_comp_status != 'success'):
                logger.info(f"Running ion analysis computation (Needed: {needs_ion_comp_run})...")
                start_time_ion_comp = time.time()
                ion_comp_results = run_ion_analysis(run_dir, psf_file, dcd_file, db_conn=db_conn) if ion_analysis_available else {'status': 'skipped'}
                ion_analysis_done = ion_comp_results.get('status') == 'success'
                if not ion_analysis_done:
                    logger.error(f"Ion analysis computation failed: {ion_comp_results.get('error', 'Unknown')}")
                    overall_success = False
                logger.info(f"Ion analysis computation completed in {time.time() - start_time_ion_comp:.2f} seconds. Success: {ion_analysis_done}")
            else:
                logger.info("Skipping ion analysis computation - already completed successfully.")
                ion_analysis_done = True

            if ion_analysis_done and needs_ion_viz_run and ion_analysis_available:
                logger.info("Generating ion analysis plots...")
                start_time_ion_viz = time.time()
                viz_results = generate_ion_plots(run_dir, db_conn=db_conn)
                if viz_results.get('status') != 'success': logger.error("Ion analysis visualization failed.")
                logger.info(f"Ion analysis visualization completed in {time.time() - start_time_ion_viz:.2f} seconds.")
            elif ion_analysis_done and generate_plots and not needs_ion_viz_run:
                logger.info("Skipping ion analysis plot generation - already completed successfully.")
            elif not ion_analysis_done and generate_plots:
                logger.warning("Cannot generate ion analysis plots - computation did not complete successfully.")


        # --- Inner Vestibule Analysis Module ---
        # (Keep existing Inner Vestibule logic - check flags, DB status, run comp/viz)
        inner_vestibule_analysis_done = False
        vestibule_comp_status = get_module_status(db_conn, "inner_vestibule_analysis")
        vestibule_viz_status = get_module_status(db_conn, "inner_vestibule_analysis_visualization")

        if run_water:
            needs_vestibule_comp_run = args.force_rerun or vestibule_comp_status != "success"
            needs_vestibule_viz_run = generate_plots and (args.force_rerun or vestibule_viz_status != "success")

            if needs_vestibule_comp_run or (needs_vestibule_viz_run and vestibule_comp_status != 'success'):
                logger.info(f"Running inner vestibule analysis computation (Needed: {needs_vestibule_comp_run})...")
                start_time_vestibule_comp = time.time()
                vestibule_comp_results = run_inner_vestibule_analysis(run_dir, psf_file, dcd_file, db_conn=db_conn) if inner_vestibule_analysis_available else {'status': 'skipped'}
                inner_vestibule_analysis_done = vestibule_comp_results.get('status') == 'success'
                if not inner_vestibule_analysis_done:
                    logger.error(f"Inner vestibule computation failed: {vestibule_comp_results.get('error', 'Unknown')}")
                    overall_success = False
                logger.info(f"Inner vestibule computation completed in {time.time() - start_time_vestibule_comp:.2f} seconds. Success: {inner_vestibule_analysis_done}")
            else:
                logger.info("Skipping inner vestibule computation - already completed successfully.")
                inner_vestibule_analysis_done = True

            if inner_vestibule_analysis_done and needs_vestibule_viz_run and inner_vestibule_analysis_available:
                logger.info("Generating inner vestibule analysis plots...")
                start_time_vestibule_viz = time.time()
                viz_results = generate_inner_vestibule_plots(run_dir, db_conn=db_conn)
                if viz_results.get('status') != 'success': logger.error("Inner vestibule visualization failed.")
                logger.info(f"Inner vestibule visualization completed in {time.time() - start_time_vestibule_viz:.2f} seconds.")
            elif inner_vestibule_analysis_done and generate_plots and not needs_vestibule_viz_run:
                 logger.info("Skipping inner vestibule plot generation - already completed successfully.")
            elif not inner_vestibule_analysis_done and generate_plots:
                 logger.warning("Cannot generate inner vestibule plots - computation did not complete successfully.")


        # --- Gyration Analysis Module ---
        # (Keep existing Gyration logic - check flags, DB status, run comp/viz)
        gyration_analysis_done = False
        gyration_comp_status = get_module_status(db_conn, "gyration_analysis")
        gyration_viz_status = get_module_status(db_conn, "gyration_analysis_visualization")

        if run_gyration:
            needs_gyration_comp_run = args.force_rerun or gyration_comp_status != "success"
            needs_gyration_viz_run = generate_plots and (args.force_rerun or gyration_viz_status != "success")

            # Dependency Check: Gyration depends on ion_analysis for filter residues
            if not ion_analysis_done:
                logger.warning("Cannot run Gyration analysis: Ion analysis did not complete successfully.")
                if gyration_comp_status != 'skipped': update_module_status(db_conn, "gyration_analysis", 'skipped', error_message="Dependency failed: ion_analysis")
                if gyration_viz_status != 'skipped': update_module_status(db_conn, "gyration_analysis_visualization", 'skipped', error_message="Dependency failed: ion_analysis")
            elif needs_gyration_comp_run or (needs_gyration_viz_run and gyration_comp_status != 'success'):
                logger.info(f"Running gyration analysis computation (Needed: {needs_gyration_comp_run})...")
                start_time_gyration_comp = time.time()
                gyration_comp_results = run_gyration_analysis(run_dir, psf_file, dcd_file, db_conn=db_conn) if gyration_analysis_available else {'status': 'skipped'}
                gyration_analysis_done = gyration_comp_results.get('status') == 'success'
                if not gyration_analysis_done:
                    logger.error(f"Gyration analysis computation failed: {gyration_comp_results.get('error', 'Unknown')}")
                    overall_success = False
                logger.info(f"Gyration analysis computation completed in {time.time() - start_time_gyration_comp:.2f} seconds. Success: {gyration_analysis_done}")
            else:
                logger.info("Skipping gyration analysis computation - already completed successfully.")
                gyration_analysis_done = True # Already done

            if gyration_analysis_done and needs_gyration_viz_run and gyration_analysis_available:
                logger.info("Generating gyration analysis plots...")
                start_time_gyration_viz = time.time()
                viz_results = generate_gyration_plots(run_dir, db_conn=db_conn)
                if viz_results.get('status') != 'success': logger.error("Gyration analysis visualization failed.")
                logger.info(f"Gyration analysis visualization completed in {time.time() - start_time_gyration_viz:.2f} seconds.")
            elif gyration_analysis_done and generate_plots and not needs_gyration_viz_run:
                 logger.info("Skipping gyration plot generation - already completed successfully.")
            elif not gyration_analysis_done and generate_plots:
                 logger.warning("Cannot generate gyration plots - computation did not complete successfully.")


        # --- Tyrosine Analysis Module ---
        # (Keep existing Tyrosine logic)
        tyrosine_analysis_done = False
        tyrosine_comp_status = get_module_status(db_conn, "tyrosine_analysis")
        tyrosine_viz_status = get_module_status(db_conn, "tyrosine_analysis_visualization")

        if run_tyrosine:
            needs_tyrosine_comp_run = args.force_rerun or tyrosine_comp_status != "success"
            needs_tyrosine_viz_run = generate_plots and (args.force_rerun or tyrosine_viz_status != "success")

            # Dependency Check: Tyrosine depends on ion_analysis for filter residues
            if not ion_analysis_done:
                logger.warning("Cannot run Tyrosine analysis: Ion analysis did not complete successfully.")
                if tyrosine_comp_status != 'skipped': update_module_status(db_conn, "tyrosine_analysis", 'skipped', error_message="Dependency failed: ion_analysis")
                if tyrosine_viz_status != 'skipped': update_module_status(db_conn, "tyrosine_analysis_visualization", 'skipped', error_message="Dependency failed: ion_analysis")
            elif needs_tyrosine_comp_run or (needs_tyrosine_viz_run and tyrosine_comp_status != 'success'):
                logger.info(f"Running tyrosine analysis computation (Needed: {needs_tyrosine_comp_run})...")
                start_time_tyrosine_comp = time.time()
                tyrosine_comp_results = run_tyrosine_analysis(run_dir, psf_file, dcd_file, db_conn=db_conn) if tyrosine_analysis_available else {'status': 'skipped'}
                tyrosine_analysis_done = tyrosine_comp_results.get('status') == 'success'
                if not tyrosine_analysis_done:
                    logger.error(f"Tyrosine analysis computation failed: {tyrosine_comp_results.get('error', 'Unknown')}")
                    overall_success = False
                logger.info(f"Tyrosine analysis computation completed in {time.time() - start_time_tyrosine_comp:.2f} seconds. Success: {tyrosine_analysis_done}")
            else:
                logger.info("Skipping tyrosine analysis computation - already completed successfully.")
                tyrosine_analysis_done = True

            if tyrosine_analysis_done and needs_tyrosine_viz_run and tyrosine_analysis_available:
                logger.info("Generating tyrosine analysis plots...")
                start_time_tyrosine_viz = time.time()
                viz_results = generate_tyrosine_plots(run_dir, db_conn=db_conn)
                if viz_results.get('status') != 'success': logger.error("Tyrosine analysis visualization failed.")
                logger.info(f"Tyrosine analysis visualization completed in {time.time() - start_time_tyrosine_viz:.2f} seconds.")
            elif tyrosine_analysis_done and generate_plots and not needs_tyrosine_viz_run:
                 logger.info("Skipping tyrosine plot generation - already completed successfully.")
            elif not tyrosine_analysis_done and generate_plots:
                 logger.warning("Cannot generate tyrosine plots - computation did not complete successfully.")


        # --- DW Gate Analysis Module --- # MODIFIED: Added DW Gate execution
        dw_gate_analysis_done = False
        dw_gate_comp_status = get_module_status(db_conn, "dw_gate_analysis")
        dw_gate_viz_status = get_module_status(db_conn, "dw_gate_analysis_visualization")

        if run_dwgates:
            needs_dw_gate_comp_run = args.force_rerun or dw_gate_comp_status != "success"
            needs_dw_gate_viz_run = generate_plots and (args.force_rerun or dw_gate_viz_status != "success")

            # Dependency Check: DW Gate depends on ion_analysis for filter residues (via metadata)
            # Filter residues dict should be stored in metadata by ion_analysis
            filter_res_meta = get_simulation_metadata(db_conn, 'filter_residues_dict')
            if not filter_res_meta:
                logger.warning("Cannot run DW Gate analysis: Filter residue dictionary not found in database metadata (dependency: ion_analysis).")
                if dw_gate_comp_status != 'skipped': update_module_status(db_conn, "dw_gate_analysis", 'skipped', error_message="Dependency failed: filter_residues_dict")
                if dw_gate_viz_status != 'skipped': update_module_status(db_conn, "dw_gate_analysis_visualization", 'skipped', error_message="Dependency failed: filter_residues_dict")
            elif needs_dw_gate_comp_run or (needs_dw_gate_viz_run and dw_gate_comp_status != 'success'):
                logger.info(f"Running DW Gate analysis computation (Needed: {needs_dw_gate_comp_run})...")
                start_time_dw_gate_comp = time.time()
                dw_gate_comp_results = run_dw_gate_analysis(run_dir, psf_file, dcd_file, db_conn=db_conn) if dw_gate_analysis_available else {'status': 'skipped'}
                dw_gate_analysis_done = dw_gate_comp_results.get('status') == 'success'
                if not dw_gate_analysis_done:
                    logger.error(f"DW Gate analysis computation failed: {dw_gate_comp_results.get('error', 'Unknown')}")
                    overall_success = False
                logger.info(f"DW Gate analysis computation completed in {time.time() - start_time_dw_gate_comp:.2f} seconds. Success: {dw_gate_analysis_done}")
            else:
                logger.info("Skipping DW Gate analysis computation - already completed successfully.")
                dw_gate_analysis_done = True

            if dw_gate_analysis_done and needs_dw_gate_viz_run and dw_gate_analysis_available:
                logger.info("Generating DW Gate analysis plots...")
                start_time_dw_gate_viz = time.time()
                viz_results = generate_dw_gate_plots(run_dir, db_conn=db_conn)
                if viz_results.get('status') != 'success': logger.error("DW Gate analysis visualization failed.")
                logger.info(f"DW Gate analysis visualization completed in {time.time() - start_time_dw_gate_viz:.2f} seconds.")
            elif dw_gate_analysis_done and generate_plots and not needs_dw_gate_viz_run:
                logger.info("Skipping DW Gate plot generation - already completed successfully.")
            elif not dw_gate_analysis_done and generate_plots:
                logger.warning("Cannot generate DW Gate plots - computation did not complete successfully.")
        # --- End DW Gate Integration --- #


        # --- Finalization ---
        set_simulation_metadata(db_conn, "analysis_end_time", datetime.now().isoformat())
        logger.info("Generating final analysis summary...")
        summary = generate_summary_from_database(run_dir, db_conn)
        if not summary:
            logger.warning("Analysis summary generation returned empty result")
            # Decide if this should mark overall_success as False
            # overall_success = False

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
        else:
            logger.info("Skipping HTML report generation based on flags.")

        final_status = 'success' if overall_success else 'failed'
        set_simulation_metadata(db_conn, "analysis_status", final_status)
        logger.info(f"Overall analysis status for this run: {final_status}")

        if db_conn:
            db_conn.close()
        return summary if summary else {} # Return empty dict if summary failed

    except Exception as e:
        logger.critical(f"Unhandled error during analysis workflow: {e}", exc_info=True)
        _save_error_summary(run_dir, system_name, run_name, f"Unhandled Workflow Error: {e}")
        if db_conn:
            try:
                set_simulation_metadata(db_conn, "analysis_status", "failed")
                set_simulation_metadata(db_conn, "analysis_error", f"Unhandled Workflow Error: {str(e)[:200]}")
                db_conn.close()
            except Exception as db_e:
                logger.error(f"Failed to close database connection during critical error handling: {db_e}")
        return None

# --- Main Execution Guard ---
def main():
    """Main entry point."""
    # Ensure core utilities were imported
    if not core_utils_available:
        print("Critical Error: Core utilities could not be imported. Cannot proceed.", file=sys.stderr)
        sys.exit(1)

    print(f"Starting Pore Analysis Suite v{Analysis_version}...")
    args = parse_arguments()

    # Set up logging (using --folder as run_dir)
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    abs_run_dir = os.path.abspath(args.folder)
    run_name = os.path.basename(abs_run_dir)
    log_file = setup_analysis_logger(abs_run_dir, run_name, log_level)

    if log_file: logger.info(f"Logging to: {log_file}")
    else: print(f"Warning: Failed to create log file in {abs_run_dir}", file=sys.stderr)

    logger.info(f"--- Analysis started for {abs_run_dir} at {datetime.now()} ---")
    logger.info(f"Command line arguments: {vars(args)}")
    start_run_time = time.time()

    # Run the workflow
    results = _run_analysis_workflow(args) # results is the summary dict or None

    end_run_time = time.time()
    logger.info(f"--- Analysis finished for {abs_run_dir} at {datetime.now()} (Duration: {end_run_time - start_run_time:.2f} sec) ---")

    if results is not None:
        logger.info("Workflow completed.")
        return 0
    else:
        logger.error("Workflow failed critically.")
        return 1

if __name__ == "__main__":
    sys.exit(main())