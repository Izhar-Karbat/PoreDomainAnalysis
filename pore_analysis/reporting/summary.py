# summary.py
"""
Functions for calculating and saving summary statistics for a single MD analysis run.
Reads output files from other analysis modules and aggregates key metrics into
a JSON file ('analysis_summary.json').
"""

import os
import logging
import numpy as np
import pandas as pd
import json
from datetime import datetime
import math # Keep math for potential future use, clean_json_data uses it too

# Import from other modules
try:
    from pore_analysis.core.utils import clean_json_data
    from pore_analysis.core.config import (Analysis_version,
                                         TYROSINE_ROTAMER_TOLERANCE_FRAMES,
                                         ION_TRANSITION_TOLERANCE_FRAMES) # Import ion tolerance
except ImportError as e:
    print(f"Error importing dependency modules in reporting/summary.py: {e}")
    raise

# Get a logger for this module
logger = logging.getLogger(__name__)


def calculate_and_save_run_summary(run_dir, system_name, run_name,
                                   com_analyzed, filter_info_com, ion_indices,
                                   cavity_water_stats={}, raw_dist_stats={},
                                   percentile_stats={}, orientation_rotation_stats={},
                                   ion_transit_stats={}, gyration_stats={}, tyrosine_stats={},
                                   conduction_stats={},
                                   is_control_system=False):
    """
    Calculates summary stats by reading analysis output files from a specific run
    (including ion, water, raw distance, percentile, orientation, ion transit, gyration, tyrosine, and conduction stats)
    and saves the combined summary to 'analysis_summary.json'. Handles control systems.

    Args:
        run_dir (str): Path to the specific run directory.
        system_name (str): Name of the system (parent directory).
        run_name (str): Name of the run (current directory).
        com_analyzed (bool): Flag indicating if COM distances were analyzed.
        filter_info_com (dict): Dictionary with COM filtering details.
        ion_indices (list | None): List of unique K+ ion indices tracked near the filter.
        cavity_water_stats (dict, optional): Dictionary with cavity water summary stats. Defaults to {}.
        raw_dist_stats (dict, optional): Dictionary with raw GG and COM distance stats (Mean, Std). Defaults to {}.
        percentile_stats (dict, optional): Dictionary with 10th/90th percentiles for raw/filtered GG and COM. Defaults to {}.
        orientation_rotation_stats (dict, optional): Dictionary with mean/std for toxin Euler rotation angles. Defaults to {}.
        ion_transit_stats (dict, optional): Dictionary with ion transit time statistics. Defaults to {}.
        gyration_stats (dict, optional): Dictionary with carbonyl gyration statistics. Defaults to {}.
        tyrosine_stats (dict, optional): Dictionary with tyrosine analysis statistics. Defaults to {}.
        conduction_stats (dict, optional): Dictionary with ion conduction statistics. Defaults to {}.
        is_control_system (bool, optional): Flag indicating if this is a control system (no toxin). Defaults to False.
    """
    logger.info(f"Calculating final summary statistics for {system_name}/{run_name}...")

    run_summary = {
        'SystemName': system_name,
        'RunName': run_name,
        'RunPath': run_dir,  # Store absolute path? Or relative? Keep as passed for now.
        'AnalysisStatus': 'Pending',  # Default status
        'AnalysisScriptVersion': Analysis_version,  # Add script version
        'AnalysisTimestamp': datetime.now().isoformat(),  # Timestamp of summary generation
        'IsControlSystem': is_control_system  # Add control system flag
    }
    stats_collection_errors = []  # Collect specific errors encountered

    # Helper function to safely get mean/std/min/max from series
    def safe_stat(series, stat_func):
        if series is None or series.empty or series.isnull().all():
            return np.nan
        try:
            # Ensure data is numeric before applying numpy functions
            numeric_series = pd.to_numeric(series, errors='coerce')
            finite_series = numeric_series.dropna()
            if finite_series.empty:
                return np.nan
            # Use nan-aware functions if appropriate, although dropna handles it
            if stat_func in [np.nanmean, np.nanstd, np.nanmin, np.nanmax]:
                return stat_func(numeric_series)
            else:
                return stat_func(finite_series)
        except Exception as e:
            logger.warning(f"Error calculating statistic ({stat_func.__name__}) on series: {e}")
            return np.nan

    try:
        # --- G-G Stats (Filtered) --- (Always calculated)
        gg_filt_path = os.path.join(run_dir, "core_analysis", "G_G_Distance_Filtered.csv")
        gg_ac_series = None
        gg_bd_series = None
        if os.path.exists(gg_filt_path):
            try:
                gg_df = pd.read_csv(gg_filt_path)
                gg_ac_series = gg_df.get('G_G_Distance_AC_Filt')
                gg_bd_series = gg_df.get('G_G_Distance_BD_Filt')
            except Exception as e:
                logger.warning(f"Error reading G-G filtered CSV ({gg_filt_path}): {e}")
                stats_collection_errors.append("G_G_Filtered_ReadError")
        else:
            logger.warning(f"G-G filtered CSV not found: {gg_filt_path}")
            stats_collection_errors.append("G_G_Filtered_Missing")

        run_summary['G_G_AC_Mean_Filt'] = safe_stat(gg_ac_series, np.nanmean)
        run_summary['G_G_AC_Std_Filt'] = safe_stat(gg_ac_series, np.nanstd)
        run_summary['G_G_AC_Min_Filt'] = safe_stat(gg_ac_series, np.nanmin)
        run_summary['G_G_BD_Mean_Filt'] = safe_stat(gg_bd_series, np.nanmean)
        run_summary['G_G_BD_Std_Filt'] = safe_stat(gg_bd_series, np.nanstd)
        run_summary['G_G_BD_Min_Filt'] = safe_stat(gg_bd_series, np.nanmin)

        # --- COM Stats (Filtered) --- (Set to None for control systems)
        if is_control_system:
            logger.debug("Control system - setting COM metrics to None")
            for metric in ['COM_Mean_Filt', 'COM_Std_Filt', 'COM_Max_Filt', 'COM_Min_Filt',
                           'COM_Filter_Type', 'COM_Filter_QualityIssue']:
                run_summary[metric] = None
        else:
            # Normal system with potential toxin
            com_filt_path = os.path.join(run_dir, "core_analysis", "COM_Stability_Filtered.csv")
            com_series = None
            if com_analyzed: # Check if COM analysis was *intended* (toxin present)
                if os.path.exists(com_filt_path):
                    try:
                        com_df = pd.read_csv(com_filt_path)
                        com_series = com_df.get('COM_Distance_Filt')
                    except Exception as e:
                        logger.warning(f"Error reading COM filtered CSV ({com_filt_path}): {e}")
                        stats_collection_errors.append("COM_Filtered_ReadError")
                else:
                    logger.warning(f"COM filtered CSV not found (but COM was analyzed): {com_filt_path}")
                    stats_collection_errors.append("COM_Filtered_Missing")
            # else: If com_analyzed is False (e.g., error in toxin detection), stats will be None anyway

            run_summary['COM_Mean_Filt'] = safe_stat(com_series, np.nanmean) if com_analyzed else None
            run_summary['COM_Std_Filt'] = safe_stat(com_series, np.nanstd) if com_analyzed else None
            run_summary['COM_Max_Filt'] = safe_stat(com_series, np.nanmax) if com_analyzed else None
            run_summary['COM_Min_Filt'] = safe_stat(com_series, np.nanmin) if com_analyzed else None
            # Get filter info from the passed dictionary
            com_filter_info = filter_info_com or {}
            run_summary['COM_Filter_Type'] = com_filter_info.get('method_applied', 'Unknown') if com_analyzed else None
            run_summary['COM_Filter_QualityIssue'] = bool(com_filter_info.get('quality_control_issues', False)) if com_analyzed else None


        # --- Orientation & Atom Contact Stats --- (Set to None for control systems)
        if is_control_system:
            logger.debug("Control system - setting orientation metrics to None")
            for metric in ['Orient_Angle_Mean', 'Orient_Angle_Std',
                           'Orient_Contacts_Mean', 'Orient_Contacts_Std']:
                run_summary[metric] = None
        else:
            orient_path = os.path.join(run_dir, "orientation_contacts", "Toxin_Orientation.csv")
            orient_angle_series = None
            orient_contacts_series = None
            if com_analyzed: # Orientation/Contacts only relevant if toxin was analyzed
                if os.path.exists(orient_path):
                    try:
                        orient_df = pd.read_csv(orient_path)
                        orient_angle_series = orient_df.get('Orientation_Angle')
                        # Handle potential column name variation for contacts
                        contacts_col = 'Total_Atom_Contacts'
                        if contacts_col not in orient_df.columns:
                            contacts_col = 'Contact_Count'
                        if contacts_col in orient_df.columns:
                            orient_contacts_series = orient_df.get(contacts_col)
                        else:
                            logger.warning(f"Could not find contact count column in {orient_path}")
                            stats_collection_errors.append("Orient_ContactsCol_Missing")
                    except Exception as e:
                        logger.warning(f"Error reading Orientation CSV ({orient_path}): {e}")
                        stats_collection_errors.append("Orientation_ReadError")
                else:
                    logger.warning(f"Orientation CSV not found (but COM was analyzed): {orient_path}")
                    stats_collection_errors.append("Orientation_Missing")
            # else: If com_analyzed is False, stats will be None anyway

            run_summary['Orient_Angle_Mean'] = safe_stat(orient_angle_series, np.nanmean) if com_analyzed else None
            run_summary['Orient_Angle_Std'] = safe_stat(orient_angle_series, np.nanstd) if com_analyzed else None
            run_summary['Orient_Contacts_Mean'] = safe_stat(orient_contacts_series, np.nanmean) if com_analyzed else None
            run_summary['Orient_Contacts_Std'] = safe_stat(orient_contacts_series, np.nanstd) if com_analyzed else None

        # --- Ion Stats --- (Always calculated)
        run_summary['Ion_Count'] = len(ion_indices) if ion_indices is not None else 0
        ion_stats_path = os.path.join(run_dir, "ion_analysis", "K_Ion_Site_Statistics.csv")
        ion_site_keys = ['S0', 'S1', 'S2', 'S3', 'S4', 'Cavity']  # Expected sites
        ion_stats_found = False
        if os.path.exists(ion_stats_path):
            try:
                ion_stats_df = pd.read_csv(ion_stats_path)
                # Updated required columns based on new ion analysis output
                required_cols = {'Site', 'Mean Occupancy', 'Max Occupancy', 'Occupancy > 0 (%)'}
                if required_cols.issubset(ion_stats_df.columns):
                    ion_stats_dict = ion_stats_df.set_index('Site').to_dict('index')
                    ion_stats_found = True
                else:
                    logger.warning(f"Ion stats CSV ({ion_stats_path}) missing required columns (Need: {required_cols}). Found: {list(ion_stats_df.columns)}")
                    stats_collection_errors.append("IonStats_MissingCols")
            except Exception as e:
                logger.warning(f"Could not read/process ion stats CSV ({ion_stats_path}): {e}")
                stats_collection_errors.append("IonStats_ReadError")
        else:
            logger.warning(f"Ion stats CSV not found: {ion_stats_path}")
            stats_collection_errors.append("IonStats_Missing")

        # Populate ion stats (use NaN if file/data missing)
        for site_key in ion_site_keys:
            if ion_stats_found:
                site_data = ion_stats_dict.get(site_key, {})  # Get data for the site, default to empty dict
                run_summary[f'Ion_AvgOcc_{site_key}'] = site_data.get('Mean Occupancy', np.nan)
                run_summary[f'Ion_MaxOcc_{site_key}'] = site_data.get('Max Occupancy', np.nan)
                run_summary[f'Ion_PctTimeOcc_{site_key}'] = site_data.get('Occupancy > 0 (%)', np.nan)
            else:  # Assign NaN if stats file wasn't read correctly
                run_summary[f'Ion_AvgOcc_{site_key}'] = np.nan
                run_summary[f'Ion_MaxOcc_{site_key}'] = np.nan
                run_summary[f'Ion_PctTimeOcc_{site_key}'] = np.nan

        # --- Cavity Water Stats --- (Always populated)
        # Renamed to Inner Vestibule Stats
        inner_vestibule_stat_keys = [
            'InnerVestibule_MeanOcc', 'InnerVestibule_StdOcc',
            'InnerVestibule_AvgResidenceTime_ns', 'InnerVestibule_TotalExitEvents',
            'InnerVestibule_ExchangeRatePerNs'
        ]

        # Check if the passed dictionary exists and has content
        if cavity_water_stats and isinstance(cavity_water_stats, dict) and cavity_water_stats:
            logger.debug("Incorporating provided inner vestibule stats into summary.")
            # Populate summary from the dictionary using the correct keys
            for key in inner_vestibule_stat_keys:
                run_summary[key] = cavity_water_stats.get(key, np.nan)
        else:
            logger.debug("Inner vestibule stats missing or empty, adding NaN placeholders.")
            # Only add error if water analysis was *expected* but failed
            # Assuming main analyzer handles this, just put NaN here
            stats_collection_errors.append("InnerVestibule_MissingOrEmpty") # Updated error code
            for key in inner_vestibule_stat_keys:
                run_summary[key] = np.nan

        # --- Add Raw Distance Stats --- (COM depends on control status)
        run_summary['G_G_AC_Std_Raw'] = raw_dist_stats.get('G_G_AC_Std_Raw', np.nan)
        run_summary['G_G_BD_Std_Raw'] = raw_dist_stats.get('G_G_BD_Std_Raw', np.nan)
        run_summary['COM_Std_Raw'] = raw_dist_stats.get('COM_Std_Raw', np.nan) if not is_control_system else None
        run_summary['G_G_AC_Mean_Raw'] = raw_dist_stats.get('G_G_AC_Mean_Raw', np.nan)
        run_summary['G_G_BD_Mean_Raw'] = raw_dist_stats.get('G_G_BD_Mean_Raw', np.nan)
        run_summary['COM_Mean_Raw'] = raw_dist_stats.get('COM_Mean_Raw', np.nan) if not is_control_system else None
        logger.debug(f"Incorporated raw distance stats: {raw_dist_stats}")

        # --- Add Percentile Stats --- (COM depends on control status)
        percentile_keys = [
            'G_G_AC_Pctl10_Raw', 'G_G_AC_Pctl90_Raw', 'G_G_BD_Pctl10_Raw', 'G_G_BD_Pctl90_Raw',
            'COM_Pctl10_Raw', 'COM_Pctl90_Raw', 'G_G_AC_Pctl10_Filt', 'G_G_AC_Pctl90_Filt',
            'G_G_BD_Pctl10_Filt', 'G_G_BD_Pctl90_Filt', 'COM_Pctl10_Filt', 'COM_Pctl90_Filt'
        ]
        for key in percentile_keys:
            # Check if the key exists in the passed dict *before* checking control status
            value = percentile_stats.get(key, np.nan) # Default to NaN if missing
            # COM percentiles are only relevant if not a control system
            if key.startswith('COM_') and is_control_system:
                run_summary[key] = None # Set to None for control
            else:
                run_summary[key] = value # Use retrieved value (could be NaN if calc failed)
        run_summary.update(percentile_stats) # Directly update from the dict
        logger.debug(f"Incorporated percentile stats: {percentile_stats}")

        # --- Add Orientation Rotation Stats --- (Use values from passed dict)
        run_summary['Orient_RotX_Mean'] = orientation_rotation_stats.get('Orient_RotX_Mean', None)
        run_summary['Orient_RotX_Std'] = orientation_rotation_stats.get('Orient_RotX_Std', None)
        run_summary['Orient_RotY_Mean'] = orientation_rotation_stats.get('Orient_RotY_Mean', None)
        run_summary['Orient_RotY_Std'] = orientation_rotation_stats.get('Orient_RotY_Std', None)
        run_summary['Orient_RotZ_Mean'] = orientation_rotation_stats.get('Orient_RotZ_Mean', None)
        run_summary['Orient_RotZ_Std'] = orientation_rotation_stats.get('Orient_RotZ_Std', None)
        logger.debug(f"Incorporated orientation stats: {orientation_rotation_stats}")

        # --- Add Gyration Stats --- (Explicitly map keys)
        logger.debug(f"Incorporating gyration stats: {gyration_stats}")
        if gyration_stats and isinstance(gyration_stats, dict):
            # Map keys from gyration_stats dict to run_summary keys
            run_summary['Gyration_G1_Mean'] = gyration_stats.get('mean_gyration_g1', np.nan)
            run_summary['Gyration_G1_Std'] = gyration_stats.get('std_gyration_g1', np.nan)
            run_summary['Gyration_Y_Mean'] = gyration_stats.get('mean_gyration_y', np.nan)
            run_summary['Gyration_Y_Std'] = gyration_stats.get('std_gyration_y', np.nan)
            run_summary['Gyration_G1_OnFlips'] = gyration_stats.get('g1_on_flips', 0)
            run_summary['Gyration_G1_OffFlips'] = gyration_stats.get('g1_off_flips', 0)
            run_summary['Gyration_G1_MeanDuration_ns'] = gyration_stats.get('g1_mean_flip_duration_ns', np.nan)
            run_summary['Gyration_G1_StdDuration_ns'] = gyration_stats.get('g1_std_flip_duration_ns', np.nan)
            run_summary['Gyration_Y_OnFlips'] = gyration_stats.get('y_on_flips', 0)
            run_summary['Gyration_Y_OffFlips'] = gyration_stats.get('y_off_flips', 0)
            run_summary['Gyration_Y_MeanDuration_ns'] = gyration_stats.get('y_mean_flip_duration_ns', np.nan)
            run_summary['Gyration_Y_StdDuration_ns'] = gyration_stats.get('y_std_flip_duration_ns', np.nan)
        else:
            logger.warning("Gyration stats missing or not a dict. Setting defaults.")
            # Set defaults if gyration_stats is missing/empty
            keys_to_default = ['Gyration_G1_Mean', 'Gyration_G1_Std', 'Gyration_Y_Mean', 'Gyration_Y_Std',
                               'Gyration_G1_MeanDuration_ns', 'Gyration_G1_StdDuration_ns',
                               'Gyration_Y_MeanDuration_ns', 'Gyration_Y_StdDuration_ns']
            count_keys_to_default = ['Gyration_G1_OnFlips', 'Gyration_G1_OffFlips',
                                   'Gyration_Y_OnFlips', 'Gyration_Y_OffFlips']
            for key in keys_to_default: run_summary[key] = np.nan
            for key in count_keys_to_default: run_summary[key] = 0

        # --- Add Tyrosine Stats --- (Explicitly map keys)
        logger.debug(f"Incorporating tyrosine stats: {tyrosine_stats}")
        tyrosine_stat_keys = {
            'Tyr_DominantRotamerOverall': ('Tyr_DominantRotamerOverall', 'N/A'),
            'Tyr_RotamerTransitions': ('Tyr_RotamerTransitions', 0),
            'Tyr_RotamerToleranceFrames': ('Tyr_RotamerToleranceFrames', TYROSINE_ROTAMER_TOLERANCE_FRAMES),
            'Tyr_NonDominant_MeanDuration_ns': ('Tyr_NonDominant_MeanDuration_ns', np.nan),
            'Tyr_NonDominant_StdDuration_ns': ('Tyr_NonDominant_StdDuration_ns', np.nan),
            'Tyr_NonDominant_EventCount': ('Tyr_NonDominant_EventCount', 0)
        }
        if tyrosine_stats and isinstance(tyrosine_stats, dict):
            for summary_key, (stats_key, default_val) in tyrosine_stat_keys.items():
                run_summary[summary_key] = tyrosine_stats.get(stats_key, default_val)
        else:
            logger.warning("Tyrosine stats missing or not a dict. Setting defaults.")
            for summary_key, (_, default_val) in tyrosine_stat_keys.items():
                run_summary[summary_key] = default_val

        # --- Add Ion Conduction / Transition Stats ---
        logger.debug(f"Incorporating conduction stats: {conduction_stats}")
        # Keys should now be calculated by ion_conduction.py
        # Use .get() with defaults just in case the dictionary is missing keys
        run_summary['Ion_ConductionEvents_Total'] = conduction_stats.get('Ion_ConductionEvents_Total', 0)
        run_summary['Ion_ConductionEvents_Outward'] = conduction_stats.get('Ion_ConductionEvents_Outward', 0)
        run_summary['Ion_ConductionEvents_Inward'] = conduction_stats.get('Ion_ConductionEvents_Inward', 0)
        run_summary['Ion_Conduction_MeanTransitTime_ns'] = conduction_stats.get('Ion_Conduction_MeanTransitTime_ns', np.nan)
        run_summary['Ion_Conduction_MedianTransitTime_ns'] = conduction_stats.get('Ion_Conduction_MedianTransitTime_ns', np.nan)
        run_summary['Ion_Conduction_StdTransitTime_ns'] = conduction_stats.get('Ion_Conduction_StdTransitTime_ns', np.nan)

        run_summary['Ion_TransitionEvents_Total'] = conduction_stats.get('Ion_TransitionEvents_Total', 0)
        run_summary['Ion_TransitionEvents_Upward'] = conduction_stats.get('Ion_TransitionEvents_Upward', 0)
        run_summary['Ion_TransitionEvents_Downward'] = conduction_stats.get('Ion_TransitionEvents_Downward', 0)

        # Get per-pair counts (expect keys like 'Ion_Transition_Cavity_S4')
        # Define expected pairs based on site order (adjust if needed)
        site_order = [s for s in ['Cavity', 'S4', 'S3', 'S2', 'S1', 'S0']]
        for s1, s2 in zip(site_order[:-1], site_order[1:]):
            key = f"Ion_Transition_{s1}_{s2}"
            run_summary[key] = conduction_stats.get(key, 0) # Default to 0 if key missing

        run_summary['Ion_Transition_ToleranceMode'] = conduction_stats.get('Ion_Transition_ToleranceMode', 'N/A')
        run_summary['Ion_Transition_ToleranceFrames'] = conduction_stats.get('Ion_Transition_ToleranceFrames', np.nan)

        # --- Add Config Constants Used (for report reference) ---
        # Explicitly add specific config values that the report might want to display
        try:
            from pore_analysis.core.config import GYRATION_FLIP_THRESHOLD, GYRATION_FLIP_TOLERANCE_FRAMES, FRAMES_PER_NS
            run_summary['Gyration_FlipThreshold'] = GYRATION_FLIP_THRESHOLD
            run_summary['Gyration_FlipToleranceFrames'] = GYRATION_FLIP_TOLERANCE_FRAMES
            run_summary['FRAMES_PER_NS'] = FRAMES_PER_NS
            run_summary['TYROSINE_ROTAMER_TOLERANCE_FRAMES'] = TYROSINE_ROTAMER_TOLERANCE_FRAMES # Also add here
            run_summary.setdefault('ION_TRANSITION_TOLERANCE_FRAMES', ION_TRANSITION_TOLERANCE_FRAMES) # Add ion tolerance here too
        except ImportError:
            logger.warning("Could not import/add config values to summary.")
            # Ensure keys exist even if import fails, using defaults or None
            run_summary.setdefault('GYRATION_FLIP_THRESHOLD', None)
            run_summary.setdefault('GYRATION_FLIP_TOLERANCE_FRAMES', None)
            run_summary.setdefault('FRAMES_PER_NS', None)
            run_summary.setdefault('TYROSINE_ROTAMER_TOLERANCE_FRAMES', TYROSINE_ROTAMER_TOLERANCE_FRAMES) # Use imported default
            run_summary.setdefault('ION_TRANSITION_TOLERANCE_FRAMES', ION_TRANSITION_TOLERANCE_FRAMES) # Add ion tolerance here too

        # --- Finalize Status ---
        # More granular status based on errors
        if not stats_collection_errors:
            run_summary['AnalysisStatus'] = 'Success'
            logger.info("Summary statistics calculation completed successfully.")
        else:
            # Check if critical files were missing vs. just read errors or missing optional data
            critical_missing = {'G_G_Filtered_Missing'}
            if not is_control_system:
                # Only add COM/orientation as critical for non-control systems
                critical_missing.update({'COM_Filtered_Missing', 'Orientation_Missing'})
            # NOTE: IonStats_Missing and InnerVestibule_MissingOrEmpty are NOT currently treated as critical failures by default.
            # Adjust 'critical_missing' set above if they should be treated as critical.

            is_critical_failure = any(err in critical_missing for err in stats_collection_errors)

            if is_critical_failure:
                run_summary['AnalysisStatus'] = 'Failed_Missing_Outputs'
                # Filter errors to show only the critical missing ones for the ERROR log
                critical_errors_found = [err for err in stats_collection_errors if err in critical_missing]
                logger.error(f"Summary calculation failed due to missing critical output files: {critical_errors_found}")
                # Log non-critical issues as warnings
                non_critical_issues = [err for err in stats_collection_errors if err not in critical_missing]
                if non_critical_issues:
                    logger.warning(f"Additional non-critical issues found during summary: {non_critical_issues}")
            else:
                # If no critical failures, it's Success_With_Issues
                error_str = "_".join(sorted(list(set(stats_collection_errors))))
                run_summary['AnalysisStatus'] = f"Success_With_Issues_{error_str}"
                logger.warning(f"Summary calculation finished with issues: {stats_collection_errors}")

    except Exception as e:
        logger.error(f"Critical error during statistics collection for {run_dir}: {e}", exc_info=True)
        run_summary['AnalysisStatus'] = 'Summary_Calculation_Error'
        # Try to include the error message in the summary if possible
        try: run_summary['ErrorDetails'] = str(e)
        except: pass
        stats_collection_errors.append("Summary_Calculation_Error")

    # --- Save the summary to JSON ---
    summary_file_path = os.path.join(run_dir, 'analysis_summary.json')
    logger.info(f"Attempting to save run summary to: {summary_file_path}")
    try:
        # Clean the data structure for JSON compatibility (handles NaN, numpy types)
        cleaned_summary = clean_json_data(run_summary)

        # Dump the cleaned dictionary
        with open(summary_file_path, 'w') as f_json:
            json.dump(cleaned_summary, f_json, indent=4)  # allow_nan=False should be safe after cleaning

        logger.info(f"Successfully saved run summary: {summary_file_path} (Status: {run_summary['AnalysisStatus']})")

    except TypeError as e:
        logger.error(f"JSON TypeError saving summary for {run_dir} (even after cleaning): {e}. Data sample: {str(cleaned_summary)[:500]}...", exc_info=True)
        run_summary['AnalysisStatus'] = 'JSON_Save_TypeError'  # Overwrite status
        # Try fallback save with repr for problematic types
        try:
            with open(summary_file_path, 'w') as f_json:
                json.dump(run_summary, f_json, indent=4, default=repr)
            logger.warning("Saved summary using fallback repr method due to TypeError.")
        except Exception as e_save:
            logger.error(f"Fallback summary save failed: {e_save}")
    except Exception as e:
        logger.error(f"Failed to save summary JSON to {summary_file_path}: {e}", exc_info=True)
        # Try saving a minimal error status
        error_summary = {k: run_summary.get(k) for k in ['SystemName', 'RunName', 'RunPath', 'AnalysisStatus', 'AnalysisScriptVersion', 'AnalysisTimestamp']}
        error_summary['AnalysisStatus'] = 'JSON_Save_Failed'
        error_summary['ErrorDetails'] = str(e)
        try:
            with open(summary_file_path, 'w') as f_json:
                json.dump(error_summary, f_json, indent=4)
        except Exception as e_save:
            logger.error(f"Failed even to save minimal error summary JSON: {e_save}")
