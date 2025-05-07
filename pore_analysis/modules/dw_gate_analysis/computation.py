# filename: pore_analysis/modules/dw_gate_analysis/computation.py
"""
Computation functions for DW-Gate analysis.

Orchestrates loading data, identifying residues, calculating distances,
processing signals, detecting states, building events, calculating statistics,
saving results (CSVs, stats DFs, KDE plot data), and storing metrics/products in the database.
"""

import os
import logging
import time
import sqlite3
import json # <<< Ensure json is imported >>>
from typing import Dict, Optional, Tuple, List, Any

import MDAnalysis as mda
import numpy as np
import pandas as pd
from tqdm import tqdm

# Import functions from local computational modules within dw_gate_analysis
from . import data_collection
from . import residue_identification
from . import signal_processing
from . import state_detection
from . import event_building
from . import statistics # Keep import for calling the function
from .utils import CLOSED_STATE, OPEN_STATE # Import state constants

# Import from core modules
try:
    # <<< Ensure clean_json_data is imported >>>
    from pore_analysis.core.utils import frames_to_time, clean_json_data
    from pore_analysis.core.database import (
        register_module, update_module_status, register_product, store_metric,
        get_simulation_metadata, get_config_parameters
    )
    from pore_analysis.core.logging import setup_system_logger
    # Import necessary config values using defaults from core.config as fallback
    from pore_analysis.core.config import (
        FRAMES_PER_NS as DEFAULT_FRAMES_PER_NS,
        DW_GATE_TOLERANCE_FRAMES as DEFAULT_TOLERANCE,
        DW_GATE_AUTO_DETECT_REFS as DEFAULT_AUTO_DETECT,
        DW_GATE_DEFAULT_CLOSED_REF_DIST as DEFAULT_CLOSED_REF,
        DW_GATE_DEFAULT_OPEN_REF_DIST as DEFAULT_OPEN_REF,
        #DEFAULT_CUTOFF as DEFAULT_DISTANCE_THRESHOLD # Not needed here anymore
    )
    # Import the function to find filter residues from ion_analysis module
    from pore_analysis.modules.ion_analysis.structure import find_filter_residues
    CORE_AVAILABLE = True
except ImportError as e:
    # Fallback if running standalone or core modules are missing
    logging.basicConfig(level=logging.INFO) # Basic logging config
    logger = logging.getLogger(__name__)
    logger.error(f"Critical Import Error - Core modules missing: {e}. Using dummy functions/defaults.")
    CORE_AVAILABLE = False
    # Define dummy DB functions
    def register_module(*args, **kwargs): pass
    def update_module_status(*args, **kwargs): pass
    def register_product(*args, **kwargs): pass
    def store_metric(*args, **kwargs): pass
    def get_simulation_metadata(*args, **kwargs): return None
    def get_config_parameters(*args, **kwargs): return {}
    # Define dummy filter residue function
    def find_filter_residues(*args, **kwargs): return {}
    # Define dummy logger setup
    def setup_system_logger(*args, **kwargs): return logging.getLogger(__name__) # Use default logger
    # Define frames_to_time dummy if core utils failed
    def frames_to_time(frames): logger.error("frames_to_time function unavailable!"); return np.array(frames) * 0.1 # Assume 10 frames/ns
    # Define clean_json_data dummy if core utils failed
    def clean_json_data(data): logger.error("clean_json_data function unavailable!"); return data
    # Use default config values directly
    DEFAULT_FRAMES_PER_NS = 10.0
    DEFAULT_TOLERANCE = 5
    DEFAULT_AUTO_DETECT = True
    DEFAULT_CLOSED_REF = 2.7
    DEFAULT_OPEN_REF = 4.7
    # DEFAULT_DISTANCE_THRESHOLD = 3.5 # Not needed here

logger = logging.getLogger(__name__)

# --- Internal Helper Functions ---

def _melt_distance_data(df_wide_dist_only, id_vars=['Frame', 'Time (ns)']):
    """Melts wide distance data (Dist_ChainX) to long format."""
    dist_cols = [col for col in df_wide_dist_only.columns if col.startswith("Dist_")]
    if not dist_cols:
        logger.error("No 'Dist_' columns found for melting distance data.")
        return None
    try:
        # Ensure id_vars exist in the dataframe before melting
        missing_ids = [v for v in id_vars if v not in df_wide_dist_only.columns]
        if missing_ids:
             logger.error(f"Missing required id_vars columns for melting distance data: {missing_ids}")
             return None

        df_dist_long = pd.melt(df_wide_dist_only, id_vars=id_vars, value_vars=dist_cols,
                               var_name="dist_col_name", value_name="distance")
        # Extract chain robustly, assuming format Dist_ChainID
        df_dist_long['chain'] = df_dist_long['dist_col_name'].str.replace("Dist_", "", n=1)
        return df_dist_long.drop(columns=["dist_col_name"])
    except Exception as e:
        logger.error(f"Error during distance data melting: {e}", exc_info=True)
        return None

def _melt_state_data_computation(df_wide_with_raw_states, id_vars=['Frame', 'Time (ns)']):
    """Melts wide distance and raw state data to long format."""
    dist_cols = sorted([col for col in df_wide_with_raw_states.columns if col.startswith("Dist_")])
    state_cols = sorted([col for col in df_wide_with_raw_states.columns if col.endswith("_state_raw")])

    if not dist_cols or not state_cols or len(dist_cols) != len(state_cols):
        logger.error(f"Mismatch or missing columns for melting state data. Dist: {len(dist_cols)}, State: {len(state_cols)}")
        return None

    try:
        # Ensure id_vars exist
        missing_ids = [v for v in id_vars if v not in df_wide_with_raw_states.columns]
        if missing_ids:
             logger.error(f"Missing required id_vars columns for melting state data: {missing_ids}")
             return None

        df_dist_long = pd.melt(df_wide_with_raw_states, id_vars=id_vars, value_vars=dist_cols, var_name="dist_col_name", value_name="distance")
        df_dist_long['chain'] = df_dist_long['dist_col_name'].str.replace("Dist_", "", n=1)

        df_state_long = pd.melt(df_wide_with_raw_states, id_vars=id_vars, value_vars=state_cols, var_name="state_col_name", value_name="state_raw")
        # Extract chain robustly, assuming format ChainID_state_raw
        df_state_long['chain'] = df_state_long['state_col_name'].str.rsplit('_', n=2).str[0]

        # Merge using all id_vars and chain
        merge_on = id_vars + ['chain']
        df_long = pd.merge(df_dist_long.drop(columns=['dist_col_name']),
                           df_state_long.drop(columns=['state_col_name']),
                           on=merge_on, how='inner') # Inner merge ensures consistency

        # Check if merge resulted in expected number of rows
        expected_rows = len(df_wide_with_raw_states) * len(dist_cols)
        if len(df_long) != expected_rows:
             logger.warning(f"Merged state data length ({len(df_long)}) does not match expected ({expected_rows}). Check merge keys and input data.")

        # Return sorted data with essential columns
        final_cols = id_vars + ['chain', 'distance', 'state_raw']
        return df_long[final_cols].sort_values(['chain'] + id_vars).reset_index(drop=True)

    except Exception as e:
        logger.error(f"Error during state data melting: {e}", exc_info=True)
        return None

def _save_stats_dataframes(
    output_dir: str,
    run_dir: str,
    stats_results: Dict[str, Any],
    db_conn: sqlite3.Connection,
    module_name: str
) -> Dict[str, Optional[str]]:
    """
    Saves the summary_stats_df and probability_df DataFrames to CSV files
    and registers them as products.

    Returns dictionary mapping subcategory ('dw_summary_stats', 'dw_probabilities')
    to the relative path of the saved file, or None if saving failed.
    """
    saved_paths = {'dw_summary_stats': None, 'dw_probabilities': None}

    # Save Summary Stats DF
    summary_df = stats_results.get('summary_stats_df')
    if isinstance(summary_df, pd.DataFrame) and not summary_df.empty:
        try:
            summary_csv_path = os.path.join(output_dir, "dw_gate_summary_stats.csv")
            summary_df.to_csv(summary_csv_path, index=False, float_format='%.4f', na_rep='NaN')
            logger.info(f"Saved DW Gate summary statistics to {summary_csv_path}")
            rel_path = os.path.relpath(summary_csv_path, run_dir)
            register_product(db_conn, module_name, "csv", "data", rel_path,
                             subcategory="dw_summary_stats",
                             description="Summary statistics (count, mean, median, etc.) for DW Gate states.")
            saved_paths['dw_summary_stats'] = rel_path
        except Exception as e:
            logger.error(f"Failed to save/register DW Gate summary statistics CSV: {e}")
    else:
        logger.warning("Summary statistics DataFrame not found or empty, skipping save.")

    # Save Probability DF
    prob_df = stats_results.get('probability_df')
    if isinstance(prob_df, pd.DataFrame) and not prob_df.empty:
        try:
            prob_csv_path = os.path.join(output_dir, "dw_gate_probabilities.csv")
            prob_df.to_csv(prob_csv_path, index=False, float_format='%.4f', na_rep='NaN')
            logger.info(f"Saved DW Gate state probabilities to {prob_csv_path}")
            rel_path = os.path.relpath(prob_csv_path, run_dir)
            register_product(db_conn, module_name, "csv", "data", rel_path,
                             subcategory="dw_probabilities",
                             description="Probability of open/closed states for DW Gate per chain.")
            saved_paths['dw_probabilities'] = rel_path
        except Exception as e:
            logger.error(f"Failed to save/register DW Gate probability CSV: {e}")
    else:
        logger.warning("Probability DataFrame not found or empty, skipping save.")

    return saved_paths


# --- Main Computation Function ---
def run_dw_gate_analysis(
    run_dir: str,
    universe=None,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    db_conn: sqlite3.Connection = None,
    psf_file: Optional[str] = None,
    dcd_file: Optional[str] = None
) -> Dict[str, any]:
    """
    Orchestrates the DW-Gate analysis computational workflow.
    Loads data, identifies residues relative to filter, calculates distances,
    determines states, builds events, calculates statistics, saves results
    (CSVs, stats DFs, KDE plot data), and stores metrics/products in the database.

    Args:
        run_dir: Path to the specific run directory.
        universe: MDAnalysis Universe object (if provided, psf_file and dcd_file are ignored).
        start_frame: Starting frame index for analysis (0-based).
        end_frame: Ending frame index for analysis (exclusive). If not specified, analyzes to the end.
        db_conn: Active database connection.
        psf_file: Path to the PSF topology file (used only if universe is not provided).
        dcd_file: Path to the DCD trajectory file (used only if universe is not provided).

    Returns:
        Dictionary containing status and error message if applicable.
    """
    module_name = "dw_gate_analysis"
    start_time = time.time()
    register_module(db_conn, module_name, status='running')
    logger_local = setup_system_logger(run_dir) # Use system logger
    if logger_local is None: logger_local = logger # Fallback

    results = {'status': 'failed', 'error': None} # Default status
    output_dir = os.path.join(run_dir, module_name)
    os.makedirs(output_dir, exist_ok=True)

    try:
        # --- 0. Fetch Configuration from DB ---
        logger_local.info("Fetching configuration parameters from database...")
        db_params = get_config_parameters(db_conn)
        def get_param(name, default, target_type):
            p_info = db_params.get(name)
            if p_info and p_info.get('value') is not None:
                try:
                     if target_type == bool: return str(p_info['value']).lower() == 'true'
                     else: return target_type(p_info['value'])
                except (ValueError, TypeError) as e:
                    logger_local.warning(f"Could not convert DB param '{name}' value '{p_info['value']}' to {target_type.__name__}. Using default: {default}. Error: {e}")
                    return default
            elif name in db_params and p_info.get('value') is None: logger_local.warning(f"DB param '{name}' found but value is None. Using default: {default}")
            return default

        frames_per_ns = get_param('FRAMES_PER_NS', DEFAULT_FRAMES_PER_NS, float)
        tolerance_frames = get_param('DW_GATE_TOLERANCE_FRAMES', DEFAULT_TOLERANCE, int)
        auto_detect_refs = get_param('DW_GATE_AUTO_DETECT_REFS', DEFAULT_AUTO_DETECT, bool)
        default_closed_ref = get_param('DW_GATE_DEFAULT_CLOSED_REF_DIST', DEFAULT_CLOSED_REF, float)
        default_open_ref = get_param('DW_GATE_DEFAULT_OPEN_REF_DIST', DEFAULT_OPEN_REF, float)
        # distance_threshold_for_kde_plot = get_param('DEFAULT_CUTOFF', DEFAULT_DISTANCE_THRESHOLD, float) # No longer needed here

        if frames_per_ns <= 0:
            logger_local.warning(f"FRAMES_PER_NS ({frames_per_ns}) is invalid, reset to default: {DEFAULT_FRAMES_PER_NS}")
            frames_per_ns = DEFAULT_FRAMES_PER_NS
        dt_ns = 1.0 / frames_per_ns
        logger_local.info(f"DW Gate Params: Tolerance={tolerance_frames} frames, AutoRefs={auto_detect_refs}, Defaults=[{default_closed_ref:.2f}, {default_open_ref:.2f}] Å, dt={dt_ns:.4f} ns")

        # --- 1. Load or Use Provided Universe ---
        logger_local.info("Step 1: Setting up Universe...")
        if universe is not None:
            u = universe
            logger_local.info("Using provided Universe object.")
        else:
            if psf_file is None or dcd_file is None:
                raise ValueError("If universe is not provided, both psf_file and dcd_file must be specified.")
            logger_local.info(f"Loading Universe from files: {psf_file}, {dcd_file}")
            u = data_collection.load_universe(psf_file, dcd_file)
            if u is None: raise ValueError("Universe loading failed.")
        
        # Validate and process frame range
        n_frames_total = len(u.trajectory)
        logger_local.info(f"Trajectory has {n_frames_total} frames total.")
        
        if start_frame < 0 or start_frame >= n_frames_total:
            error_msg = f"Invalid start_frame: {start_frame}. Must be between 0 and {n_frames_total-1}"
            logger_local.error(error_msg)
            raise ValueError(error_msg)
            
        actual_end = end_frame if end_frame is not None else n_frames_total
        if actual_end <= start_frame or actual_end > n_frames_total:
            error_msg = f"Invalid end_frame: {actual_end}. Must be between {start_frame+1} and {n_frames_total}"
            logger_local.error(error_msg)
            raise ValueError(error_msg)
            
        logger_local.info(f"Using frame range: {start_frame} to {actual_end} (analyzing {actual_end-start_frame} frames)")
        
        # Store frame range metrics
        store_metric(db_conn, module_name, "start_frame", start_frame, "frame", "Starting frame index for DW-gate analysis")
        store_metric(db_conn, module_name, "end_frame", actual_end, "frame", "Ending frame index for DW-gate analysis")
        store_metric(db_conn, module_name, "frames_analyzed", actual_end - start_frame, "frame", "Number of frames analyzed")
        
        if n_frames_total < 2: raise ValueError("Trajectory has insufficient frames (< 2).")

        # --- 2. Identify Filter & Gate Residues ---
        logger_local.info("Step 2: Identifying Filter and Gate Residues...")
        filter_res_json = get_simulation_metadata(db_conn, 'filter_residues_dict')
        if not filter_res_json: raise ValueError("filter_residues_dict not found in DB metadata. Run ion_analysis first.")
        try:
            filter_res_map = json.loads(filter_res_json)
            if not isinstance(filter_res_map, dict): raise TypeError("Parsed filter_residues is not a dictionary.")
        except (json.JSONDecodeError, TypeError) as e:
            raise ValueError(f"Failed to load/parse filter_residues_dict from DB: {e}")
        logger_local.info(f"Retrieved filter residues for chains: {list(filter_res_map.keys())}")

        chain_ids = sorted(list(filter_res_map.keys()))
        gate_residues = residue_identification.select_dw_residues(universe, chain_ids, filter_res_map)
        if not gate_residues: raise ValueError("DW-Gate residue selection failed for all chains.")
        valid_chain_ids = sorted(list(gate_residues.keys()))
        logger_local.info(f"Successfully identified DW-Gate residues for chains: {valid_chain_ids}")

        # --- 3. Calculate Raw Distances ---
        logger_local.info("Step 3: Calculating Raw Distances...")
        # Pass the universe and frame range to the distance calculation
        df_distances_wide = data_collection.calculate_dw_distances(
            universe=u, 
            gate_residues=gate_residues,
            start_frame=start_frame,
            end_frame=actual_end
        )
        if df_distances_wide is None or df_distances_wide.empty: raise ValueError("Raw distance calculation failed.")
        
        # The frame indices in the DataFrame are now local (0-based for the analyzed window)
        # We need to convert them back to global trajectory frames
        if 'Frame' not in df_distances_wide.columns:
            # If Frame is index, convert to column and add start_frame offset
            df_distances_wide['Frame'] = df_distances_wide.index + start_frame
        else:
            # If Frame is already a column, add start_frame offset
            df_distances_wide['Frame'] = df_distances_wide['Frame'] + start_frame
            
        if 'Time (ns)' not in df_distances_wide.columns:
            time_points = frames_to_time(df_distances_wide['Frame'].values)
            df_distances_wide.insert(0, 'Time (ns)', time_points)

        raw_dist_csv_path = os.path.join(output_dir, "dw_gate_raw_distances.csv")
        df_distances_wide.to_csv(raw_dist_csv_path, index=False, float_format='%.4f', na_rep='NaN')
        rel_raw_dist_path = os.path.relpath(raw_dist_csv_path, run_dir)
        register_product(db_conn, module_name, "csv", "data", rel_raw_dist_path,
                         subcategory="raw_dw_distances",
                         description="Raw minimum DW-Gate distances per chain per frame.")
        logger_local.info(f"Saved raw distances to {raw_dist_csv_path}")

        # --- 4. Determine Reference Distances & Save KDE Data ---
        logger_local.info("Step 4: Determining Reference Distances...")
        dist_cols_for_kde = [f'Dist_{ch}' for ch in valid_chain_ids]
        cols_present = ['Frame', 'Time (ns)'] + dist_cols_for_kde
        missing_melt_cols = [c for c in cols_present if c not in df_distances_wide.columns]
        if missing_melt_cols: raise ValueError(f"Missing columns for melting: {missing_melt_cols}")
        df_for_kde_melt = df_distances_wide[cols_present].copy()
        df_dist_long_for_kde = _melt_distance_data(df_for_kde_melt)
        if df_dist_long_for_kde is None: raise ValueError("Failed to melt distance data for KDE.")

        kde_plot_data = {} # Initialize
        if auto_detect_refs:
            # Capture kde_plot_data
            final_closed_ref, final_open_ref, kde_plot_data = signal_processing.determine_reference_distances(
                df_raw=df_dist_long_for_kde,
                default_closed_ref=default_closed_ref,
                default_open_ref=default_open_ref
                # Pass other KDE params if needed, e.g., kde_bw_method=...
            )
        else:
            final_closed_ref = default_closed_ref
            final_open_ref = default_open_ref
            logger_local.info("Using default reference distances.")
            # Populate kde_plot_data minimally if needed for plotting defaults
            kde_plot_data['all_distances'] = df_dist_long_for_kde['distance'].dropna().tolist()
            kde_plot_data['final_closed_ref'] = final_closed_ref
            kde_plot_data['final_open_ref'] = final_open_ref

        logger_local.info(f"Final Reference Distances: Closed={final_closed_ref:.3f} Å, Open={final_open_ref:.3f} Å")
        store_metric(db_conn, module_name, "DW_RefDist_Closed_Used", final_closed_ref, "Å", "Final Closed Reference Distance Used")
        store_metric(db_conn, module_name, "DW_RefDist_Open_Used", final_open_ref, "Å", "Final Open Reference Distance Used")
        store_metric(db_conn, module_name, "Config_DW_GATE_TOLERANCE_FRAMES", tolerance_frames, "frames", "DW Gate tolerance frames used")

        # <<< MODIFIED: Save kde_plot_data to JSON >>>
        kde_data_json_path = os.path.join(output_dir, "dw_gate_kde_plot_data.json")
        rel_kde_data_path = os.path.relpath(kde_data_json_path, run_dir)
        try:
            # Clean data for JSON serialization (handles NumPy arrays etc.)
            # Ensure clean_json_data is imported from core.utils
            cleaned_kde_data = clean_json_data(kde_plot_data)
            with open(kde_data_json_path, 'w') as f_json:
                json.dump(cleaned_kde_data, f_json, indent=2)
            logger_local.info(f"Saved KDE plot data to {kde_data_json_path}")
            register_product(db_conn, module_name, "json", "data", rel_kde_data_path,
                             subcategory="kde_plot_data", # NEW subcategory for the data
                             description="Data required for plotting DW Gate KDE distribution.")
        except Exception as e_save_kde:
            logger_local.error(f"Failed to save KDE plot data JSON: {e_save_kde}", exc_info=True)
            # Decide if this is critical? Plotting will fail later.
            # Maybe don't raise an error, but log it.
        # <<< END MODIFICATION >>>

        # --- 5. Assign Initial States ---
        logger_local.info("Step 5: Assigning Initial States...")
        df_states_wide_raw = state_detection.assign_initial_state(
            df_distances_wide, final_closed_ref, final_open_ref
        )
        if df_states_wide_raw is None or df_states_wide_raw.empty: raise ValueError("Initial state assignment failed.")

        # --- 6. Apply RLE Debouncing ---
        logger_local.info("Step 6: Applying RLE Debouncing...")
        df_long_raw_states = _melt_state_data_computation(df_states_wide_raw)
        if df_long_raw_states is None: raise ValueError("Melting data for debouncing failed.")
        df_states_long_debounced = signal_processing.apply_rle_debouncing(
            df_long_raw_states, tolerance_frames, state_col='state_raw', debounced_col='state'
        )
        if df_states_long_debounced is None or df_states_long_debounced.empty: raise ValueError("Debouncing failed.")

        debounced_csv_path = os.path.join(output_dir, "dw_gate_debounced_states.csv")
        cols_to_save_debounced = ['Frame', 'Time (ns)', 'chain', 'distance', 'state_raw', 'state']
        df_to_save_debounced = df_states_long_debounced[[c for c in cols_to_save_debounced if c in df_states_long_debounced.columns]]
        df_to_save_debounced.to_csv(debounced_csv_path, index=False, float_format='%.4f', na_rep='NaN')
        rel_debounced_path = os.path.relpath(debounced_csv_path, run_dir)
        register_product(db_conn, module_name, "csv", "data", rel_debounced_path,
                         subcategory="debounced_states",
                         description="Time series of DW-Gate distances and debounced states (long format).")
        logger_local.info(f"Saved debounced states to {debounced_csv_path}")

        # --- 7. Build Events ---
        logger_local.info("Step 7: Building Events...")
        if dt_ns <= 0: raise ValueError("Invalid dt_ns for event building.")
        events_df = event_building.build_events_from_states(df_states_long_debounced, dt_ns, state_col='state')
        if events_df is None: raise ValueError("Event building failed critically.")
        if events_df.empty: logger.warning("Event building resulted in an empty DataFrame.")
        # Save events and register product (even if empty)
        events_csv_path = os.path.join(output_dir, "dw_gate_events.csv")
        events_df.to_csv(events_csv_path, index=False, float_format='%.4f', na_rep='NaN')
        rel_events_path = os.path.relpath(events_csv_path, run_dir)
        register_product(db_conn, module_name, "csv", "data", rel_events_path,
                         subcategory="dw_events",
                         description="Processed DW-Gate state events (after debouncing).")
        logger_local.info(f"Saved events data to {events_csv_path}")

        # --- 8. Calculate Statistics ---
        logger_local.info("Step 8: Calculating Statistics...")
        stats_results = statistics.calculate_dw_statistics(events_df) # Pass df with standardized names
        if not stats_results or stats_results.get('Error'):
            logger.warning(f"Statistics calculation failed or returned error: {stats_results.get('Error', 'Unknown error')}")
        else:
            # --- 8a. Save Statistics DataFrames ---
            saved_stats_paths = _save_stats_dataframes(output_dir, run_dir, stats_results, db_conn, module_name)
            if not saved_stats_paths.get('dw_summary_stats') or not saved_stats_paths.get('dw_probabilities'):
                 logger_local.warning("Failed to save one or both DW Gate statistics DataFrames.")
                 # Decide if this should be a critical failure? For now, just warn.

            # --- 8b. Store Key Metrics (Probabilities/Counts/P-values) ---
            prob_df = stats_results.get('probability_df')
            if isinstance(prob_df, pd.DataFrame) and not prob_df.empty:
                for _, row in prob_df.iterrows():
                    chain = row['Chain'] # Use Standardized Name
                    store_metric(db_conn, module_name, f"DW_{chain}_Closed_Fraction", row.get(CLOSED_STATE, np.nan) * 100.0, "%", f"Fraction time Chain {chain} DW-Gate is Closed")
                    store_metric(db_conn, module_name, f"DW_{chain}_Open_Fraction", row.get(OPEN_STATE, np.nan) * 100.0, "%", f"Fraction time Chain {chain} DW-Gate is Open")

            summary_df = stats_results.get('summary_stats_df')
            if isinstance(summary_df, pd.DataFrame) and not summary_df.empty:
                 for _, row in summary_df.iterrows():
                    chain = row['Chain'] # Use Standardized Name
                    state = row['State'] # Use Standardized Name
                    if pd.notna(row.get('Mean_ns')): store_metric(db_conn, module_name, f"DW_{chain}_{state}_Mean_ns", row['Mean_ns'], "ns", f"Mean duration Chain {chain} in {state} state (ns)")
                    if pd.notna(row.get('Median_ns')): store_metric(db_conn, module_name, f"DW_{chain}_{state}_Median_ns", row['Median_ns'], "ns", f"Median duration Chain {chain} in {state} state (ns)")
                    if pd.notna(row.get('Count')): store_metric(db_conn, module_name, f"DW_{chain}_{state}_Count", row['Count'], "count", f"Number of {state} events for Chain {chain}")
                    if pd.notna(row.get('Std_Dev_ns')): store_metric(db_conn, module_name, f"DW_{chain}_{state}_Std_Dev_ns", row['Std_Dev_ns'], "ns", f"Std Dev duration Chain {chain} in {state} state (ns)")

            chi2_test_res = stats_results.get('chi2_test', {})
            p_value_chi2 = chi2_test_res.get('p_value')
            if p_value_chi2 is not None and pd.notna(p_value_chi2):
                 store_metric(db_conn, module_name, "DW_StateDurationVsChain_Chi2_pvalue", p_value_chi2, "p-value", "Chi2 test p-value (State Durations vs Chain)")

            kruskal_test_res = stats_results.get('kruskal_test', {})
            p_value_kruskal = kruskal_test_res.get('p_value')
            if p_value_kruskal is not None and pd.notna(p_value_kruskal):
                 store_metric(db_conn, module_name, "DW_OpenDurationVsChain_Kruskal_pvalue", p_value_kruskal, "p-value", "Kruskal-Wallis p-value (Open Durations vs Chain)")

        results['status'] = 'success'
        logger_local.info("DW-Gate computation steps completed successfully.")

    except Exception as e:
        error_msg = f"Error during DW-Gate analysis: {repr(e)}"
        logger_local.error(error_msg, exc_info=True)
        results['error'] = error_msg
        results['status'] = 'failed'
        try: update_module_status(db_conn, module_name, 'failed', error_message=str(e))
        except Exception as db_e: logger.error(f"Failed to update module status to failed: {db_e}")

    # --- Finalize ---
    exec_time = time.time() - start_time
    try: update_module_status(db_conn, module_name, results['status'], execution_time=exec_time, error_message=results['error'])
    except Exception as db_e: logger.error(f"Failed to update final module status: {db_e}")
    logger_local.info(f"--- DW-Gate Analysis computation finished in {exec_time:.2f} seconds (Status: {results['status']}) ---")

    return results