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
    from utils import clean_json_data
    from config import Analysis_version
except ImportError as e:
    print(f"Error importing dependency modules in summary.py: {e}")
    raise

# Get a logger for this module
logger = logging.getLogger(__name__)


def calculate_and_save_run_summary(run_dir, system_name, run_name,
                                   com_analyzed, filter_info_com, ion_indices,
                                   cavity_water_stats={}):
    """
    Calculates summary stats by reading analysis output files from a specific run
    (including ion and cavity water stats passed as arguments) and saves the
    combined summary to 'analysis_summary.json'.

    Args:
        run_dir (str): Path to the specific run directory.
        system_name (str): Name of the system (parent directory).
        run_name (str): Name of the run (current directory).
        com_analyzed (bool): Flag indicating if COM distances were analyzed (i.e., not a control run).
        filter_info_com (dict): Dictionary with COM filtering details (used for filter type).
        ion_indices (list | None): List of unique K+ ion indices tracked near the filter, or None.
        cavity_water_stats (dict, optional): Dictionary with cavity water summary stats. Defaults to {}.
    """
    logger.info(f"Calculating final summary statistics for {system_name}/{run_name}...")

    run_summary = {
        'SystemName': system_name,
        'RunName': run_name,
        'RunPath': run_dir, # Store absolute path? Or relative? Keep as passed for now.
        'AnalysisStatus': 'Pending', # Default status
        'AnalysisScriptVersion': Analysis_version, # Add script version
        'AnalysisTimestamp': datetime.now().isoformat() # Timestamp of summary generation
    }
    stats_collection_errors = [] # Collect specific errors encountered

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
            return stat_func(finite_series)
        except Exception as e:
            logger.warning(f"Error calculating statistic ({stat_func.__name__}) on series: {e}")
            return np.nan

    try:
        # --- G-G Stats (Filtered) ---
        gg_filt_path = os.path.join(run_dir, "G_G_Distance_Filtered.csv")
        gg_ac_series = None
        gg_bd_series = None
        if os.path.exists(gg_filt_path):
            try:
                gg_df = pd.read_csv(gg_filt_path)
                gg_ac_series = gg_df.get('A_C_Distance_Filtered')
                gg_bd_series = gg_df.get('B_D_Distance_Filtered')
            except Exception as e:
                logger.warning(f"Error reading G-G filtered CSV ({gg_filt_path}): {e}")
                stats_collection_errors.append("GG_Filtered_ReadError")
        else:
            logger.warning(f"G-G filtered CSV not found: {gg_filt_path}")
            stats_collection_errors.append("GG_Filtered_Missing")

        run_summary['GG_AC_Mean_Filt'] = safe_stat(gg_ac_series, np.mean)
        run_summary['GG_AC_Std_Filt'] = safe_stat(gg_ac_series, np.std)
        run_summary['GG_AC_Min_Filt'] = safe_stat(gg_ac_series, np.min)
        run_summary['GG_BD_Mean_Filt'] = safe_stat(gg_bd_series, np.mean)
        run_summary['GG_BD_Std_Filt'] = safe_stat(gg_bd_series, np.std)
        run_summary['GG_BD_Min_Filt'] = safe_stat(gg_bd_series, np.min)

        # --- COM Stats (Filtered) ---
        com_filt_path = os.path.join(run_dir, "COM_Stability_Filtered.csv")
        com_series = None
        if com_analyzed: # Only process if toxin was expected/analyzed
            if os.path.exists(com_filt_path):
                try:
                    com_df = pd.read_csv(com_filt_path)
                    com_series = com_df.get('COM_Distance_Filtered')
                except Exception as e:
                    logger.warning(f"Error reading COM filtered CSV ({com_filt_path}): {e}")
                    stats_collection_errors.append("COM_Filtered_ReadError")
            else:
                logger.warning(f"COM filtered CSV not found (but COM was analyzed): {com_filt_path}")
                stats_collection_errors.append("COM_Filtered_Missing")
        else: # If COM not analyzed (e.g., control), set stats to None
             logger.debug("COM analysis not performed for this run. Setting COM stats to None.")

        run_summary['COM_Mean_Filt'] = safe_stat(com_series, np.mean) if com_analyzed else None
        run_summary['COM_Std_Filt'] = safe_stat(com_series, np.std) if com_analyzed else None
        run_summary['COM_Max_Filt'] = safe_stat(com_series, np.max) if com_analyzed else None
        run_summary['COM_Min_Filt'] = safe_stat(com_series, np.min) if com_analyzed else None
        run_summary['COM_Filter_Type'] = filter_info_com.get('method_applied', 'Unknown') if com_analyzed else None # Use method applied key
        run_summary['COM_Filter_QualityIssue'] = bool(filter_info_com.get('quality_control_issues')) if com_analyzed else None # Use QC key

        # --- Orientation & Atom Contact Stats ---
        orient_path = os.path.join(run_dir, "Toxin_Orientation.csv")
        orient_angle_series = None
        orient_contacts_series = None
        if com_analyzed: # Orientation/Contacts only relevant if toxin present
            if os.path.exists(orient_path):
                try:
                    orient_df = pd.read_csv(orient_path)
                    orient_angle_series = orient_df.get('Orientation_Angle')
                    # Handle potential column name variation for contacts
                    contacts_col = 'Total_Atom_Contacts'
                    if contacts_col not in orient_df.columns: contacts_col = 'Contact_Count' # Fallback? Check actual output
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

        run_summary['Orient_Angle_Mean'] = safe_stat(orient_angle_series, np.mean) if com_analyzed else None
        run_summary['Orient_Angle_Std'] = safe_stat(orient_angle_series, np.std) if com_analyzed else None
        run_summary['Orient_Contacts_Mean'] = safe_stat(orient_contacts_series, np.mean) if com_analyzed else None
        run_summary['Orient_Contacts_Std'] = safe_stat(orient_contacts_series, np.std) if com_analyzed else None

        # --- Ion Stats ---
        run_summary['Ion_Count'] = len(ion_indices) if ion_indices is not None else 0
        ion_stats_path = os.path.join(run_dir, 'K_Ion_Site_Statistics.csv')
        ion_site_keys = ['S0', 'S1', 'S2', 'S3', 'S4', 'Cavity'] # Expected sites
        ion_stats_found = False
        if os.path.exists(ion_stats_path):
            try:
                ion_stats_df = pd.read_csv(ion_stats_path)
                required_cols = {'Site', 'Mean Occupancy', 'Max Occupancy', 'Occupancy > 0 (%)'}
                if required_cols.issubset(ion_stats_df.columns):
                    ion_stats_dict = ion_stats_df.set_index('Site').to_dict('index')
                    ion_stats_found = True
                else:
                    logger.warning(f"Ion stats CSV ({ion_stats_path}) missing required columns. Found: {list(ion_stats_df.columns)}")
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
                 site_data = ion_stats_dict.get(site_key, {}) # Get data for the site, default to empty dict
                 run_summary[f'Ion_AvgOcc_{site_key}'] = site_data.get('Mean Occupancy', np.nan)
                 run_summary[f'Ion_MaxOcc_{site_key}'] = site_data.get('Max Occupancy', np.nan)
                 run_summary[f'Ion_PctTimeOcc_{site_key}'] = site_data.get('Occupancy > 0 (%)', np.nan)
            else: # Assign NaN if stats file wasn't read correctly
                 run_summary[f'Ion_AvgOcc_{site_key}'] = np.nan
                 run_summary[f'Ion_MaxOcc_{site_key}'] = np.nan
                 run_summary[f'Ion_PctTimeOcc_{site_key}'] = np.nan

        # --- Cavity Water Stats ---
        # These are passed directly as a dictionary
        cavity_stat_keys = ['CavityWater_MeanOcc', 'CavityWater_StdOcc',
                            'CavityWater_AvgResidenceTime_ns', 'CavityWater_TotalExitEvents',
                            'CavityWater_ExchangeRatePerNs']
        if cavity_water_stats and isinstance(cavity_water_stats, dict) and cavity_water_stats: # Check not empty
            logger.debug("Incorporating provided cavity water stats into summary.")
            stats_collection_errors.append("CavityWater_DataFound") # Indicate data was processed
            for key in cavity_stat_keys:
                run_summary[key] = cavity_water_stats.get(key, np.nan) # Use .get() for safety
        else:
             logger.debug("Cavity water stats missing or empty, adding NaN placeholders.")
             stats_collection_errors.append("CavityWater_MissingOrEmpty") # Add specific status
             for key in cavity_stat_keys:
                 run_summary[key] = np.nan


        # --- Finalize Status ---
        # More granular status based on errors
        if not stats_collection_errors:
            run_summary['AnalysisStatus'] = 'Success'
            logger.info("Summary statistics calculation completed successfully.")
        else:
            # Check if critical files were missing vs. just read errors or missing optional data
            critical_missing = {'GG_Filtered_Missing'}
            if com_analyzed: critical_missing.update({'COM_Filtered_Missing', 'Orientation_Missing'})
            # IonStats_Missing is only critical if ions were expected/tracked > 0 ? Maybe not critical.
            # CavityWater_MissingOrEmpty might be expected for some runs.

            is_critical_failure = any(err in critical_missing for err in stats_collection_errors)

            if is_critical_failure:
                 run_summary['AnalysisStatus'] = 'Failed_Missing_Outputs'
                 logger.error(f"Summary calculation failed due to missing critical output files: {stats_collection_errors}")
            else:
                 error_str = "_".join(sorted(list(set(stats_collection_errors)))) # Summarize errors
                 run_summary['AnalysisStatus'] = f"Success_With_Issues_{error_str}"
                 logger.warning(f"Summary calculation finished with issues: {stats_collection_errors}")

    except Exception as e:
        logger.error(f"Critical error during statistics collection for {run_dir}: {e}", exc_info=True)
        run_summary['AnalysisStatus'] = 'Summary_Calculation_Error'
        stats_collection_errors.append("Summary_Calculation_Error")

    # --- Save the summary to JSON ---
    summary_file_path = os.path.join(run_dir, 'analysis_summary.json')
    logger.info(f"Attempting to save run summary to: {summary_file_path}")
    try:
        # Clean the data structure for JSON compatibility (handles NaN, numpy types)
        cleaned_summary = clean_json_data(run_summary)

        # Dump the cleaned dictionary
        with open(summary_file_path, 'w') as f_json:
            json.dump(cleaned_summary, f_json, indent=4) # allow_nan=False should be safe after cleaning

        logger.info(f"Successfully saved run summary: {summary_file_path} (Status: {run_summary['AnalysisStatus']})")

    except TypeError as e:
        logger.error(f"JSON TypeError saving summary for {run_dir} (even after cleaning): {e}. Data sample: {str(cleaned_summary)[:500]}...", exc_info=True)
        run_summary['AnalysisStatus'] = 'JSON_Save_TypeError' # Overwrite status
        # Try fallback save with repr for problematic types
        try:
             with open(summary_file_path, 'w') as f_json: json.dump(run_summary, f_json, indent=4, default=repr)
             logger.warning("Saved summary using fallback repr method due to TypeError.")
        except Exception as e_save: logger.error(f"Fallback summary save failed: {e_save}")
    except Exception as e:
        logger.error(f"Failed to save summary JSON to {summary_file_path}: {e}", exc_info=True)
        # Try saving a minimal error status
        error_summary = {k: run_summary.get(k) for k in ['SystemName', 'RunName', 'RunPath', 'AnalysisStatus', 'AnalysisScriptVersion', 'AnalysisTimestamp']}
        error_summary['AnalysisStatus'] = 'JSON_Save_Failed'
        error_summary['ErrorDetails'] = str(e)
        try:
             with open(summary_file_path, 'w') as f_json: json.dump(error_summary, f_json, indent=4)
        except Exception as e_save: logger.error(f"Failed even to save minimal error summary JSON: {e_save}")
