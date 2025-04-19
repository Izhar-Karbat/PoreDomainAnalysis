#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Molecular Dynamics Simulation Analysis Script (Refactored)
=========================================================

Main orchestrator script for analyzing molecular dynamics simulations
of toxin-channel complexes. Reads trajectories, performs various analyses
(distances, filtering, orientation, contacts, ions, water), and generates
reports and summaries for a SINGLE simulation run folder.

(Docstring content from original script regarding PURPOSE, DATA PROBLEMS, etc.
 can be retained or summarized here if desired)

See individual modules (core_analysis.py, filtering.py, ion_analysis.py, etc.)
for details on specific analysis implementations.

USAGE:
------
1. Single Run Analysis (All Analyses - Default):
   python -m md_analysis.main --folder /path/to/run_directory [--all]

2. Selective Analysis (Single Run):
   python -m md_analysis.main --folder /path/to/run --FLAG [--FLAG ...]
   (e.g., --GG, --COM, --ions, --water, --tyrosine, --conduction)
   NOTE: HTML report only generated with --all or no flags.

3. Discouraged Single Trajectory Mode:
    python -m md_analysis.main --trajectory <path> --topology <path> [--output <dir>] [analysis_flags]

"""

import argparse
import os
import sys
import logging
import json
from datetime import datetime
from collections import defaultdict
import traceback # For more detailed error logging
import numpy as np
import MDAnalysis as mda

# --- Import constants and functions from refactored modules ---
try:
    from md_analysis.core.config import Analysis_version
    from md_analysis.core.utils import frames_to_time, clean_json_data
    from md_analysis.core.logging import setup_analysis_logger
    from md_analysis.modules.core_analysis.core import analyze_trajectory, filter_and_save_data
    from md_analysis.modules.orientation_contacts.orientation_contacts import analyze_toxin_orientation
    from md_analysis.modules.ion_analysis import track_potassium_ions, analyze_ion_coordination, analyze_ion_conduction
    from md_analysis.modules.inner_vestibule_analysis import analyze_inner_vestibule as analyze_cavity_water
    from md_analysis.reporting.summary import calculate_and_save_run_summary
    from md_analysis.reporting.html import generate_html_report
    from md_analysis.modules.gyration_analysis.gyration_core import analyze_carbonyl_gyration
    from md_analysis.modules.tyrosine_analysis import analyze_sf_tyrosine_rotamers

except ImportError as e:
    print(f"ERROR: Failed to import necessary modules: {e}", file=sys.stderr)
    print("Please ensure the md_analysis package is properly installed and structured.", file=sys.stderr)
    sys.exit(1)

def main():
    """
    Main execution function: parses arguments and orchestrates the analysis workflow.
    """
    parser = argparse.ArgumentParser(
        description=f"MD Simulation Analysis Script (v{Analysis_version}). Processes trajectories for a SINGLE run folder.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    # --- Input/Output & Mode Flags ---
    parser.add_argument("--trajectory", help="Path to a specific trajectory file (single mode - discouraged).")
    parser.add_argument("--topology", help="Path to a specific topology file (required for --trajectory mode).")
    parser.add_argument("--output", help="Output directory for results (used in --trajectory mode, defaults to trajectory dir).")
    parser.add_argument("--folder", required=True, help="Path to the specific run folder containing PSF/DCD (required).")
    parser.add_argument("--force_rerun", action="store_true", help="Force reprocessing of the run even if a successful summary exists.")

    # --- Analysis Selection Flags --- (Added tyrosine, conduction)
    analysis_group = parser.add_argument_group('Analysis Selection (runs all if none specified, skips HTML report if specific flags used)')
    analysis_group.add_argument("--all", action="store_true", help="Run all available analyses (default if no other analysis flag is set).")
    analysis_group.add_argument("--GG", action="store_true", help="Run G-G distance analysis.")
    analysis_group.add_argument("--COM", action="store_true", help="Run COM distance analysis.")
    analysis_group.add_argument("--orientation", action="store_true", help="Run Toxin orientation and contact analysis.")
    analysis_group.add_argument("--ions", action="store_true", help="Run K+ ion tracking and coordination analysis.")
    analysis_group.add_argument("--water", action="store_true", help="Run Cavity Water analysis.")
    analysis_group.add_argument("--gyration", action="store_true", help="Run Carbonyl Gyration analysis.")
    analysis_group.add_argument("--tyrosine", action="store_true", help="Run SF Tyrosine rotamer analysis.")
    analysis_group.add_argument("--conduction", action="store_true", help="Run Ion Conduction/Transition analysis.")

    # --- Other Options ---
    parser.add_argument("--box_z", type=float, default=None, help="Provide estimated box Z-dimension (Angstroms) for multi-level COM filter.")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Set the logging level.")

    args = parser.parse_args()

    # Logger setup is deferred until folder_path is known
    log_level_map = {"DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING, "ERROR": logging.ERROR}
    root_log_level = log_level_map.get(args.log_level.upper(), logging.INFO)

    # --- Determine Which Analyses to Run ---
    specific_flags_set = args.GG or args.COM or args.orientation or args.ions or args.water or args.gyration or args.tyrosine or args.conduction
    run_all_initially = args.all or not specific_flags_set

    run_gg = args.GG or run_all_initially
    run_com = args.COM or run_all_initially
    run_orientation = args.orientation or run_com
    run_ion_tracking = args.ions or args.water or args.conduction or run_all_initially
    run_ion_coordination = args.ions or run_all_initially
    run_water = args.water or run_all_initially
    run_gyration = args.gyration or run_all_initially
    run_tyrosine = args.tyrosine or run_all_initially
    run_conduction = args.conduction or run_all_initially

    if args.force_rerun:
        # Log attempt will likely go to stderr before logger setup
        print("WARNING: --force_rerun specified. Running ALL analyses.", file=sys.stderr)
        run_gg = True
        run_com = True
        run_orientation = True
        run_ion_tracking = True
        run_ion_coordination = True
        run_water = True
        run_gyration = True
        run_tyrosine = True
        run_conduction = True

    generate_html = (run_gg and run_com and run_orientation and run_ion_tracking and
                     run_ion_coordination and run_water and run_gyration and run_tyrosine and run_conduction)
    run_any_core_analysis = (run_gg or run_com or run_orientation or run_ion_tracking or
                             run_water or run_gyration or run_tyrosine or run_conduction)

    if not run_any_core_analysis and not args.trajectory:
         print("WARNING: No analysis flags selected. Defaulting to running all.", file=sys.stderr)
         run_gg = True
         run_com = True
         run_orientation = True
         run_ion_tracking = True
         run_ion_coordination = True
         run_water = True
         run_gyration = True
         run_tyrosine = True
         run_conduction = True
         generate_html = True

    # ============================
    # --- Single Trajectory Mode ---
    # ============================
    if args.trajectory:
        # ... (Argument validation) ...
        # --- Determine paths and names ---
        run_name = os.path.splitext(os.path.basename(args.trajectory))[0]
        output_dir = args.output if args.output else os.path.dirname(args.trajectory)
        os.makedirs(output_dir, exist_ok=True)
        parent_output_dir = os.path.dirname(output_dir)
        system_name = os.path.basename(parent_output_dir) if parent_output_dir and os.path.basename(parent_output_dir) else run_name

        # --- Setup Logger ---
        analysis_log_file = setup_analysis_logger(output_dir, run_name, root_log_level)
        if not analysis_log_file:
            return # Error printed by setup
        
        logging.info(f"--- MD Analysis Script v{Analysis_version} Started (Trajectory Mode) ---")
        # ... (Log command, paths, analysis plan) ...

        # --- Call Analysis Workflow --- 
        try:
            success = _run_analysis_workflow(
                run_dir=output_dir, # Pass output_dir as run_dir
                system_name=system_name,
                run_name=run_name,
                psf_file=args.topology,
                dcd_file=args.trajectory,
                run_gg=run_gg, run_com=run_com, run_orientation=run_orientation,
                run_ion_tracking=run_ion_tracking, run_ion_coordination=run_ion_coordination,
                run_water=run_water, run_gyration=run_gyration, run_tyrosine=run_tyrosine,
                run_conduction=run_conduction, # Pass conduction flag
                generate_html=generate_html,
                box_z=args.box_z,
                force_rerun=args.force_rerun
            )
            # ... (log success/failure) ...
        except Exception as e:
            # ... (log critical error)
            logging.critical(f"Unhandled error during trajectory processing: {e}", exc_info=True)
            _save_error_summary(output_dir, system_name, run_name, f"Unhandled Error: {e}")
        return # Ensure this is indented correctly

    # ========================
    # --- Single Folder Mode --- (Primary workflow)
    # ========================
    folder_path = os.path.abspath(args.folder)
    if not os.path.isdir(folder_path):
         print(f"ERROR: Specified folder does not exist: {folder_path}", file=sys.stderr)
         return

    run_name = os.path.basename(folder_path)
    parent_dir = os.path.dirname(folder_path)
    system_name = os.path.basename(parent_dir) if parent_dir and os.path.basename(parent_dir) else run_name

    # --- Setup Logger ---
    analysis_log_file = setup_analysis_logger(folder_path, run_name, root_log_level)
    if not analysis_log_file:
        return # Error printed by setup

    # --- Log initial info ---
    logging.info(f"--- MD Analysis Script v{Analysis_version} Started (Folder Mode) ---")
    # ... (Log command, system/run name, folder, analysis plan) ...

    # --- Prepare for workflow ---
    dcd_file = os.path.join(folder_path, "MD_Aligned.dcd")
    psf_file = os.path.join(folder_path, "step5_input.psf")

    if not os.path.exists(dcd_file) or not os.path.exists(psf_file):
        logging.error(f"Input files missing in {folder_path}. Aborting.")
        _save_error_summary(folder_path, system_name, run_name, "Input_File_Missing")
        return

    # --- Call Analysis Workflow ---
    try:
        success = _run_analysis_workflow(
            run_dir=folder_path,
            system_name=system_name,
            run_name=run_name,
            psf_file=psf_file,
            dcd_file=dcd_file,
            run_gg=run_gg, run_com=run_com, run_orientation=run_orientation,
            run_ion_tracking=run_ion_tracking, run_ion_coordination=run_ion_coordination,
            run_water=run_water, run_gyration=run_gyration, run_tyrosine=run_tyrosine,
            run_conduction=run_conduction, # Pass conduction flag
            generate_html=generate_html,
            box_z=args.box_z,
            force_rerun=args.force_rerun
        )
        # ... (log success/failure) ...
    except Exception as e:
        # ... (log critical error) ...
        logging.critical(f"Unhandled error during single folder processing: {e}", exc_info=True)
        _save_error_summary(folder_path, system_name, run_name, f"Unhandled Error: {e}")

def _run_analysis_workflow(run_dir, system_name, run_name, psf_file, dcd_file,
                           run_gg, run_com, run_orientation, run_ion_tracking,
                           run_ion_coordination, run_water, run_gyration, run_tyrosine,
                           run_conduction, # Added conduction
                           generate_html,
                           box_z=None, force_rerun=False):
    """
    Internal helper function to execute the analysis steps for a single run.
    (Ensure this function correctly handles the new flags and passes them down)
    """
    summary_file_path = os.path.join(run_dir, 'analysis_summary.json')

    # --- Skip Logic (if not forcing rerun) ---
    if not force_rerun and os.path.exists(summary_file_path):
        try:
            with open(summary_file_path, 'r') as f_check:
                existing_summary = json.load(f_check)
            previous_version = existing_summary.get('AnalysisScriptVersion')
            previous_status = existing_summary.get('AnalysisStatus', '')

            # Skip only if ALL analyses were requested originally (implied by generate_html=True)
            # AND previous run was successful AND versions match.
            # If specific flags are used now, we always rerun those specific parts.
            if generate_html and previous_status.startswith('Success') and previous_version == Analysis_version:
                logging.info(f"Skipping successfully processed run (v{Analysis_version}): {system_name}/{run_name}")
                return True # Indicate successful skip
            elif previous_version != Analysis_version:
                 logging.info(f"Reprocessing run: Version mismatch (Prev: {previous_version}, Curr: {Analysis_version})")
            elif not previous_status.startswith('Success'):
                 logging.info(f"Reprocessing run: Previous status was '{previous_status}'")
            # Else (generate_html is False): continue to run selected analyses

        except Exception as e_check:
            logging.warning(f"Could not read/parse existing summary {summary_file_path}, reprocessing. Error: {e_check}")

    # --- Initialize results --- (Add new keys)
    results = {
        'dist_ac': None, 'dist_bd': None, 'com_distances': None, 'time_points': None,
        'filtered_ac': None, 'filtered_bd': None, 'filtered_com': None,
        'filter_info_gg': {}, 'filter_info_com': {},
        'ions_z_g1_filtered': None, 'time_points_ions': None, 'ion_indices': [], # Updated ion keys
        'g1_reference': None, 'filter_sites': None, 'filter_residues': None,
        'cavity_water_stats': {}, 'ion_transit_stats': {}, # Added ion transit stats
        'orientation_rotation_stats': {}, 'raw_dist_stats': {}, 'percentile_stats': {},
        'com_analyzed': False, # Flag: Was COM calculated?
        'is_control_system': False, # Flag: Is this a control system?
        'gyration_stats': {},
        'tyrosine_stats': {},
        'conduction_stats': {} # New key for conduction results
    }

    # --- Execute Analyses --- 
    try:
        # 1. Core Trajectory Analysis (G-G, COM Raw) - Always run if any analysis needed
        logging.info("Running Core Trajectory Analysis...")
        # Unpack 6 values, including is_control_system
        results['dist_ac'], results['dist_bd'], results['com_distances'], results['time_points'], _, results['is_control_system'] = \
            analyze_trajectory(run_dir, psf_file=psf_file, dcd_file=dcd_file)
        results['com_analyzed'] = results['com_distances'] is not None

        # Update logs based on system type
        if results['is_control_system']:
            logging.info(f"CONTROL system detected (no toxin) for {system_name}/{run_name}")
        else:
            logging.info(f"Toxin-channel complex system detected for {system_name}/{run_name}")

        # Check if trajectory analysis returned valid time points
        if results['time_points'] is None or len(results['time_points']) == 0:
             raise ValueError("Trajectory analysis failed to produce time points.")

        # 2. Filtering (if requested)
        if run_gg or run_com:
            logging.info(f"Running Filtering and Saving (GG={run_gg}, COM={run_com})...")
            results['filtered_ac'], results['filtered_bd'], results['filtered_com'], \
            results['filter_info_gg'], results['filter_info_com'], \
            results['raw_dist_stats'], results['percentile_stats'] = \
                filter_and_save_data(
                    run_dir, results['dist_ac'], results['dist_bd'],
                    results['com_distances'], results['time_points'],
                    box_z=box_z,
                    is_control_system=results['is_control_system'] # Pass flag
                )

            # Update com_analyzed based on raw distance result
            results['com_analyzed'] = results['com_distances'] is not None

        # 3. Ion Tracking (if requested or needed by water)
        if run_ion_tracking:
            logging.info("Running K+ Ion Tracking...")
            # track_potassium_ions returns: ions_z_abs, time_points, ion_indices, g1_ref, sites, filter_residues
            ions_z_abs, time_points_ions, ion_indices, g1_ref, sites, filter_res_dict = \
                track_potassium_ions(run_dir, psf_file=psf_file, dcd_file=dcd_file)
            # === DEBUGGING ADDED ===
            logging.debug(f"track_potassium_ions returned:")
            logging.debug(f"  - ion_indices type: {type(ion_indices)}, len: {len(ion_indices) if ion_indices is not None else 'N/A'}")
            logging.debug(f"  - g1_ref type: {type(g1_ref)}, value: {g1_ref}")
            logging.debug(f"  - sites type: {type(sites)}, is_none: {sites is None}, keys: {list(sites.keys()) if isinstance(sites, dict) else 'N/A'}")
            logging.debug(f"  - filter_res_dict type: {type(filter_res_dict)}, is_none: {filter_res_dict is None}, keys: {list(filter_res_dict.keys()) if isinstance(filter_res_dict, dict) else 'N/A'}")
            # === END DEBUGGING ===
            # Assign to results dictionary
            results['ions_z_abs'] = ions_z_abs
            results['time_points_ions'] = time_points_ions
            results['ion_indices'] = ion_indices
            results['g1_reference'] = g1_ref
            results['filter_sites'] = sites
            results['filter_residues'] = filter_res_dict
            # NOTE: results['ion_transit_stats'] is NOT assigned here anymore

        # 4. Toxin Orientation/Contacts (if requested and NOT a control system)
        if run_orientation and not results['is_control_system']:
            logging.info("Running Toxin Orientation Analysis...")
            if results['com_analyzed']:
                # Call the comprehensive function from orientation_contacts.py
                # It handles calculations, plotting, and saving internally.
                # It returns 5 values: angles, euler_angles, contact_counts, contact_df, rotation_stats_dict
                orientation_angles, rotation_euler_angles, contact_counts, residue_contact_df, rotation_stats_dict = \
                    analyze_toxin_orientation(dcd_file, psf_file, run_dir)
                
                # Assign the returned dictionary directly to the results
                results['orientation_rotation_stats'] = rotation_stats_dict
                
                # Optional: could store other returned values in results if needed later
                # results['orientation_angles'] = orientation_angles
                # results['contact_counts'] = contact_counts

            else:
                # Should not happen if com_analyzed is False for control, but safety check
                logging.warning("Skipping Toxin Orientation analysis (run requested but COM not analyzed).")
                results['orientation_rotation_stats'] = {}
        elif run_orientation and results['is_control_system']:
            logging.info("Skipping Toxin Orientation analysis (control system without toxin).")
            # Set empty rotation stats for control systems for consistency
            results['orientation_rotation_stats'] = { # Set to None, not NaN, matching summary expectation
                'Orient_RotX_Mean': None, 'Orient_RotX_Std': None,
                'Orient_RotY_Mean': None, 'Orient_RotY_Std': None,
                'Orient_RotZ_Mean': None, 'Orient_RotZ_Std': None
            }
        elif not run_orientation:
             # If not run, ensure stats dict exists but is empty or nullified
             results['orientation_rotation_stats'] = { # Set to None
                'Orient_RotX_Mean': None, 'Orient_RotX_Std': None,
                'Orient_RotY_Mean': None, 'Orient_RotY_Std': None,
                'Orient_RotZ_Mean': None, 'Orient_RotZ_Std': None
             }

        # 5. Ion Coordination (if requested)
        if run_ion_coordination:
            logging.info("Running K+ Ion Coordination Analysis...")
            # === DEBUGGING ADDED - BEFORE CHECK ===
            logging.debug(f"BEFORE check: ion_indices is None = {results.get('ion_indices') is None}")
            logging.debug(f"BEFORE check: filter_residues is None = {results.get('filter_residues') is None}")
            logging.debug(f"BEFORE check: filter_residues is Falsy = {not results.get('filter_residues')}")
            # === END DEBUGGING ===
            if results.get('ion_indices') is not None and results.get('filter_residues'):
                try:
                    # Call the refactored function with CORRECT arguments
                    analyze_ion_coordination(
                        run_dir,
                        results['time_points_ions'],      # Pass correct time points
                        results['ions_z_abs'],           # Pass correct Z positions
                        results['ion_indices'],          # Pass correct indices
                        results['filter_sites'],         # Pass correct filter sites
                        results['g1_reference']          # Pass correct G1 reference
                    )
                except FileNotFoundError:
                    logging.error("Coordination: PSF or DCD file not found. Cannot load Universe.")
                except Exception as e_coord_setup:
                    logging.error(f"Coordination: Error during setup or execution: {e_coord_setup}", exc_info=True)
            else:
                logging.warning("Skipping Ion Coordination (missing prerequisites from ion tracking: indices or residues).")

        # 6. Cavity Water (if requested)
        if run_water:
            logging.info("Running Cavity Water Analysis...")
            # Check prerequisites (sites, reference, and the filter_residues dict)
            if results.get('filter_sites') and results.get('filter_residues') \
               and isinstance(results.get('g1_reference'), (float, int, np.number)):
                results['cavity_water_stats'] = analyze_cavity_water(
                    run_dir, psf_file, dcd_file,
                    results['filter_sites'], results['g1_reference'], results['filter_residues'] # Pass filter_residues
                )
            else:
                logging.warning("Skipping Cavity Water analysis (missing prerequisites: sites, g1_ref, or filter_residues).")

        # 7. Carbonyl Gyration Analysis (if requested)
        if run_gyration:
            logging.info("Running Carbonyl Gyration Analysis...")
            # Assume analyze_carbonyl_gyration is defined elsewhere and imported
            gyration_results = analyze_carbonyl_gyration(
                run_dir=run_dir,
                psf_file=psf_file,
                dcd_file=dcd_file,
                # Pass system type based on previously determined flag
                system_type="toxin" if not results.get('is_control_system', False) else "control"
            )

            # Store results in a variable for summary generation
            results['gyration_stats'] = gyration_results
            # Log completion with a key result if available
            flips = gyration_results.get('flips_detected', 'N/A')
            logging.info(f"Completed carbonyl gyration analysis: {flips} flips detected")
        else:
            logging.info("Skipping Carbonyl Gyration analysis (not requested)")
            results['gyration_stats'] = {} # Ensure key exists even if empty

        # 8. SF Tyrosine Rotamer Analysis (if requested) 
        if run_tyrosine:
            logging.info("Running SF Tyrosine Rotamer Analysis...")
            # Assumes analyze_sf_tyrosine_rotamers is imported
            tyrosine_results = analyze_sf_tyrosine_rotamers(
                run_dir=run_dir,
                psf_file=psf_file,
                dcd_file=dcd_file,
                filter_residues=results.get('filter_residues') # Pass detected residues if available
            )
            results['tyrosine_stats'] = tyrosine_results
        else:
            logging.info("Skipping SF Tyrosine analysis (not requested)")
            results['tyrosine_stats'] = {}

        # 9. Ion Conduction / Transition Analysis (if requested) 
        if run_conduction:
            logging.info("Running Ion Conduction / Transition Analysis...")
            # Check prerequisites from ion tracking
            if (results.get('time_points_ions') is not None and
                results.get('ions_z_abs') is not None and
                results.get('ion_indices') is not None and
                results.get('filter_sites') is not None and
                results.get('g1_reference') is not None):

                results['conduction_stats'] = analyze_ion_conduction(
                    run_dir=run_dir,
                    time_points=results['time_points_ions'],
                    ions_z_positions=results['ions_z_abs'],
                    ion_indices=results['ion_indices'],
                    filter_sites=results['filter_sites'],
                    g1_reference=results['g1_reference']
                )
                total_cond = results['conduction_stats'].get('Ion_ConductionEvents_Total', 0)
                total_trans = results['conduction_stats'].get('Ion_TransitionEvents_Total', 0)
                logging.info(f"Completed ion conduction analysis: {total_cond} conduction events, {total_trans} transitions.")
            else:
                logging.warning("Skipping Ion Conduction analysis (missing prerequisites from ion tracking). Ensure --ions was run.")
                results['conduction_stats'] = {} # Ensure key exists
        else:
             logging.info("Skipping Ion Conduction analysis (not requested)")
             results['conduction_stats'] = {}

        # --- Post-Analysis ---

        # 10. Calculate and Save Final Summary JSON (Pass new stats dicts)
        logging.info("Calculating and saving summary JSON...")
        calculate_and_save_run_summary(
            run_dir, system_name, run_name,
            results.get('com_analyzed', False), # Use .get with default
            results.get('filter_info_com', {}),
            results.get('ion_indices', []),
            results.get('cavity_water_stats', {}),
            raw_dist_stats=results.get('raw_dist_stats', {}), # Corrected from raw_stats
            percentile_stats=results.get('percentile_stats', {}),
            orientation_rotation_stats=results.get('orientation_rotation_stats', {}),
            ion_transit_stats=results.get('ion_transit_stats', {}),
            gyration_stats=results.get('gyration_stats', {}),
            tyrosine_stats=results.get('tyrosine_stats', {}),
            conduction_stats=results.get('conduction_stats', {}),
            is_control_system=results.get('is_control_system', False)
        )

        # 11. Generate HTML Report 
        if generate_html:
            logging.info("Generating HTML Report...")
            # Load the freshly saved summary to pass to HTML generator
            try:
                with open(summary_file_path, 'r') as f_sum:
                    run_summary_for_html = json.load(f_sum)
                generate_html_report(run_dir, run_summary_for_html)
            except FileNotFoundError:
                 logging.error(f"Could not find summary file {summary_file_path} to generate HTML report.")
            except Exception as e_html:
                 logging.error(f"Failed to generate HTML report: {e_html}", exc_info=True)
        else:
             logging.info("Skipping HTML report generation (specific analysis flags used).")

        return True # Success

    except Exception as e:
        logging.error(f"Workflow failed for {run_dir}: {e}", exc_info=True)
        _save_error_summary(run_dir, system_name, run_name, f"WorkflowError: {e}")
        return False # Failure

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
        cleaned_summary = clean_json_data(error_summary) # Clean just in case
        with open(summary_file_path, 'w') as f_json:
            json.dump(cleaned_summary, f_json, indent=4)
        logging.info(f"Saved error status to {summary_file_path}")
    except Exception as e_save:
        logging.error(f"Failed to save error summary JSON to {summary_file_path}: {e_save}")

if __name__ == "__main__":
    main()
    logging.info(f"--- MD Analysis Script v{Analysis_version} Finished ---")
