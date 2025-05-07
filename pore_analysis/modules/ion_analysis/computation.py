# pore_analysis/modules/ion_analysis/computation.py
"""
Ion Analysis: Orchestration of computational steps, including HMM-based transition analysis.
"""

import os
import logging
import time
import sqlite3
import numpy as np
import pandas as pd
import json
from typing import Dict, Optional, List, Any, Tuple
from collections import defaultdict

# Import functions from local computational modules
from .structure import find_filter_residues, calculate_g1_reference, calculate_binding_sites
from .tracking import track_ion_positions, _save_ion_position_csvs
from .occupancy import calculate_occupancy
from .hmm import build_transition_matrix, process_ion_with_hmm, warning_aggregator

# Import from core modules
try:
    from pore_analysis.core.database import (
        register_module, update_module_status, register_product, store_metric,
        set_simulation_metadata
    )
    from pore_analysis.core.config import (
        HMM_SELF_TRANSITION_P, HMM_EPSILON, HMM_EMISSION_SIGMA, HMM_FLICKER_NS
    )
    # <<< --- ADDED: clean_json_data import --- >>>
    from pore_analysis.core.utils import frames_to_time, clean_json_data
    # <<< --- END OF ADDED IMPORT --- >>>
except ImportError as e:
    print(f"Error importing dependency modules in ion_analysis/computation.py: {e}")
    raise

logger = logging.getLogger(__name__)

# --- Helper Functions to Save HMM Results ---
# ... ( _save_hmm_transition_data, _save_hmm_quality_data,
#       _calculate_and_store_hmm_stats, _calculate_and_save_hmm_conduction functions remain unchanged ) ...

def _save_hmm_transition_data(
    run_dir: str,
    all_ion_transitions: Dict[int, List[Dict]], # Dict keyed by ion_idx
    db_conn: sqlite3.Connection,
    module_name: str
) -> Optional[str]:
    """Saves HMM-derived transition events to CSV."""
    output_dir = os.path.join(run_dir, "ion_analysis")
    os.makedirs(output_dir, exist_ok=True)
    rel_path = None
    compiled_transitions = []

    for ion_idx, transitions in all_ion_transitions.items():
        for event in transitions:
            if event['site_label'] == "Outside": continue # Exclude 'Outside' dwells
            compiled_transitions.append({
                'ion_idx': ion_idx,
                'start_frame': event['start_frame'],
                'end_frame': event['end_frame'],
                'start_time': event['start_time'],
                'end_time': event['end_time'],
                'site_label': event['site_label'],
                'duration_ns': event['end_time'] - event['start_time']
            })

    if not compiled_transitions:
        logger.info("No HMM dwell events (excluding 'Outside') to save.")
        return None

    try:
        df = pd.DataFrame(compiled_transitions)
        df = df.sort_values(by=['ion_idx', 'start_frame'])
        csv_path = os.path.join(output_dir, 'ion_hmm_dwell_events.csv')
        df.to_csv(csv_path, index=False, float_format='%.4f')
        logger.info(f"Saved HMM ion dwell events to {csv_path}")
        rel_path = os.path.relpath(csv_path, run_dir)
        register_product(db_conn, module_name, "csv", "data", rel_path,
                         subcategory="ion_hmm_dwell_events",
                         description="HMM-derived ion dwell events per site (filtered, collapsed).")
    except Exception as e:
        logger.error(f"Failed to save/register HMM dwell events CSV: {e}", exc_info=True)
        rel_path = None
    return rel_path

def _save_hmm_quality_data(
    run_dir: str,
    all_quality_flags: Dict[int, np.ndarray],
    all_hmm_paths: Dict[int, np.ndarray],
    time_points: np.ndarray,
    site_names: List[str], # Site names from HMM
    db_conn: sqlite3.Connection,
    module_name: str
) -> Optional[str]:
    """Saves HMM quality flags and raw path indices to CSV."""
    output_dir = os.path.join(run_dir, "ion_analysis")
    rel_path = None
    ion_indices = sorted(all_quality_flags.keys())

    if not ion_indices:
        logger.info("No quality flag data to save.")
        return None

    n_frames = len(time_points)
    base_data = {'Frame': np.arange(n_frames), 'Time (ns)': time_points}
    df_quality = pd.DataFrame(base_data)

    for ion_idx in ion_indices:
        # Ensure data exists and has the correct length
        quality_flags = all_quality_flags.get(ion_idx)
        hmm_path = all_hmm_paths.get(ion_idx)

        has_quality = quality_flags is not None and len(quality_flags) == n_frames
        has_path = hmm_path is not None and len(hmm_path) == n_frames

        df_quality[f'Ion_{ion_idx}_SuspiciousFlag'] = quality_flags if has_quality else np.zeros(n_frames, dtype=bool)

        if has_path:
            state_names_col = [site_names[idx] if 0 <= idx < len(site_names) else 'Outside' for idx in hmm_path]
            df_quality[f'Ion_{ion_idx}_HMM_State'] = state_names_col
            df_quality[f'Ion_{ion_idx}_HMM_StateIdx'] = hmm_path
        else:
            df_quality[f'Ion_{ion_idx}_HMM_State'] = 'Unknown'
            df_quality[f'Ion_{ion_idx}_HMM_StateIdx'] = -2

    try:
        csv_path = os.path.join(output_dir, 'ion_hmm_quality_data.csv')
        df_quality.to_csv(csv_path, index=False, float_format='%.4f')
        logger.info(f"Saved HMM quality data to {csv_path}")
        rel_path = os.path.relpath(csv_path, run_dir)
        register_product(db_conn, module_name, "csv", "data", rel_path,
                         subcategory="ion_hmm_quality_data",
                         description="HMM raw state assignments and suspicious flags per frame.")
    except Exception as e:
        logger.error(f"Failed to save/register HMM quality data CSV: {e}", exc_info=True)
        rel_path = None
    return rel_path

def _calculate_and_store_hmm_stats(
    all_ion_transitions: Dict[int, List[Dict]], # HMM dwell events
    site_names: List[str], # Includes 'Outside' if used, or just binding sites
    db_conn: sqlite3.Connection,
    module_name: str):
    """Calculates summary stats from HMM dwell events and stores them."""
    logger.info("Calculating HMM transition statistics...")
    stats = defaultdict(int)
    site_order = [s for s in ['Cavity','S4','S3','S2','S1','S0'] if s in site_names] # Order for pairs

    total_transitions = 0
    upward_transitions = 0
    downward_transitions = 0
    per_pair_counts = defaultdict(int)

    for ion_idx, dwell_events in all_ion_transitions.items():
         sorted_events = sorted(dwell_events, key=lambda x: x['start_time'])
         for i in range(len(sorted_events) - 1):
             from_event = sorted_events[i]
             to_event = sorted_events[i+1]

             if from_event['site_label'] in site_order and to_event['site_label'] in site_order:
                 total_transitions += 1
                 try:
                     idx_from = site_order.index(from_event['site_label'])
                     idx_to = site_order.index(to_event['site_label'])
                     if idx_to > idx_from: downward_transitions += 1 # Towards Cavity
                     elif idx_to < idx_from: upward_transitions += 1 # Towards S0

                     # Count transitions between adjacent pairs only
                     if abs(idx_to - idx_from) == 1:
                          pair_key = tuple(sorted((from_event['site_label'], to_event['site_label'])))
                          per_pair_counts[pair_key] += 1
                 except ValueError: pass # Site not in standard order

    stats['Ion_HMM_TransitionEvents_Total'] = total_transitions
    stats['Ion_HMM_TransitionEvents_Upward'] = upward_transitions
    stats['Ion_HMM_TransitionEvents_Downward'] = downward_transitions

    for s1, s2 in zip(site_order[:-1], site_order[1:]):
         pair_key = tuple(sorted((s1, s2)))
         key = f"Ion_HMM_Transition_{s1}_{s2}"
         stats[key] = per_pair_counts.get(pair_key, 0) # Use .get with default 0

    # Store metrics
    try:
        logger.debug(f"Storing HMM transition metrics: {stats}")
        store_metric(db_conn, module_name, 'Ion_HMM_TransitionEvents_Total', stats['Ion_HMM_TransitionEvents_Total'], 'count', 'Total adjacent site transition events (HMM)')
        store_metric(db_conn, module_name, 'Ion_HMM_TransitionEvents_Upward', stats['Ion_HMM_TransitionEvents_Upward'], 'count', 'Upward transition events (HMM)')
        store_metric(db_conn, module_name, 'Ion_HMM_TransitionEvents_Downward', stats['Ion_HMM_TransitionEvents_Downward'], 'count', 'Downward transition events (HMM)')
        for s1, s2 in zip(site_order[:-1], site_order[1:]):
            key = f"Ion_HMM_Transition_{s1}_{s2}"
            desc = f"Transitions between {s1} and {s2} (HMM)"
            store_metric(db_conn, module_name, key, stats.get(key, 0), 'count', desc)
    except Exception as e: logger.error(f"Failed to store HMM transition metrics: {e}", exc_info=True)

def _calculate_and_save_hmm_conduction(
    run_dir: str,
    all_ion_transitions: Dict[int, List[Dict]], # HMM dwell events
    site_names: List[str],
    db_conn: sqlite3.Connection,
    module_name: str
) -> Optional[str]:
    """Calculates full conduction events based on HMM dwell events."""
    # --- Implementation unchanged ---
    logger.info("Calculating full conduction events from HMM paths...")
    output_dir = os.path.join(run_dir, "ion_analysis")
    rel_path = None
    conduction_events = []
    site_order_standard = ['Cavity','S4','S3','S2','S1','S0'] # Define standard order

    site_names_in_order = [s for s in site_order_standard if s in site_names]
    if len(site_names_in_order) < 2: # Need at least two standard sites
        logger.warning("Fewer than two standard sites found for conduction analysis.")
        return None

    entry_site = site_names_in_order[0] # e.g., Cavity
    exit_site = site_names_in_order[-1] # e.g., S0
    intermediate_sites_set = set(site_names_in_order[1:-1])

    for ion_idx, dwell_events in all_ion_transitions.items():
        sorted_events = sorted(dwell_events, key=lambda x: x['start_time'])
        i = 0
        while i < len(sorted_events):
            start_event = sorted_events[i]
            # Check for potential outward conduction start (entry_site)
            if start_event['site_label'] == entry_site:
                path_indices = [i]
                current_path_sites = {entry_site}
                for j in range(i + 1, len(sorted_events)):
                     current_event = sorted_events[j]
                     current_site = current_event['site_label']
                     path_indices.append(j)
                     current_path_sites.add(current_site)

                     if current_site == exit_site: # Reached the end
                          # Check if intermediate sites were visited
                          if not intermediate_sites_set.isdisjoint(current_path_sites):
                               # Valid outward conduction
                               conduction_events.append({
                                   'ion_idx': ion_idx, 'direction': 'outward',
                                   'entry_frame': start_event['start_frame'], 'exit_frame': current_event['end_frame'],
                                   'entry_time': start_event['start_time'], 'exit_time': current_event['end_time'],
                                   'transit_time': current_event['end_time'] - start_event['start_time'],
                                   'sites_visited': [sorted_events[k]['site_label'] for k in path_indices]
                               })
                               i = j # Move main loop index past this event
                               break # Stop searching for this conduction path
                          else: break # Reached exit without intermediates
                     elif current_site == entry_site: break # Returned to start

            # Check for potential inward conduction start (exit_site)
            elif start_event['site_label'] == exit_site:
                 path_indices = [i]
                 current_path_sites = {exit_site}
                 for j in range(i + 1, len(sorted_events)):
                      current_event = sorted_events[j]
                      current_site = current_event['site_label']
                      path_indices.append(j)
                      current_path_sites.add(current_site)

                      if current_site == entry_site: # Reached the end (cavity)
                           if not intermediate_sites_set.isdisjoint(current_path_sites):
                                # Valid inward conduction
                                conduction_events.append({
                                    'ion_idx': ion_idx, 'direction': 'inward',
                                    'entry_frame': start_event['start_frame'], 'exit_frame': current_event['end_frame'],
                                    'entry_time': start_event['start_time'], 'exit_time': current_event['end_time'],
                                    'transit_time': current_event['end_time'] - start_event['start_time'],
                                    'sites_visited': [sorted_events[k]['site_label'] for k in path_indices]
                                })
                                i = j
                                break
                           else: break
                      elif current_site == exit_site: break # Returned to start

            i += 1 # Move to next event to check as a potential start

    # Save conduction events and store metrics
    if conduction_events:
        logger.info(f"Found {len(conduction_events)} full conduction events from HMM analysis.")
        try:
            dfc = pd.DataFrame(conduction_events)
            path = os.path.join(output_dir, "ion_hmm_conduction_events.csv")
            dfc.to_csv(path, index=False, float_format="%.3f")
            logger.info(f"HMM Conduction events written to {path}")
            rel_path = os.path.relpath(path, run_dir)
            register_product(db_conn, module_name, "csv", "data", rel_path, subcategory="ion_hmm_conduction_events", description="Details of full K+ ion conduction events (HMM-derived).")

            stats_cond = {
                'Ion_HMM_ConductionEvents_Total': len(conduction_events),
                'Ion_HMM_ConductionEvents_Outward': sum(1 for e in conduction_events if e['direction']=='outward'),
                'Ion_HMM_ConductionEvents_Inward': sum(1 for e in conduction_events if e['direction']=='inward'),
            }
            times = [e['transit_time'] for e in conduction_events if 'transit_time' in e]
            stats_cond['Ion_HMM_Conduction_MeanTransitTime_ns'] = float(np.mean(times)) if times else np.nan
            stats_cond['Ion_HMM_Conduction_MedianTransitTime_ns'] = float(np.median(times)) if times else np.nan
            stats_cond['Ion_HMM_Conduction_StdTransitTime_ns'] = float(np.std(times)) if times else np.nan

            for key, val in stats_cond.items():
                 units = 'count' if 'Events' in key else ('ns' if 'Time' in key else None)
                 desc = key.replace('_', ' ').replace('Ion HMM ', '') + ' (HMM)'
                 # Ensure value is finite before storing
                 if np.isfinite(val):
                    store_metric(db_conn, module_name, key, float(val), units, desc) # Store as float
                 else:
                    logger.warning(f"HMM conduction metric '{key}' is non-finite, storing as NULL.")
                    store_metric(db_conn, module_name, key, None, units, desc) # Store None for non-finite

        except Exception as e:
            logger.error(f"Failed to save/register/store HMM conduction data/stats: {e}", exc_info=True)
            rel_path = None
    else:
        logger.info("No full conduction events found from HMM analysis.")
        store_metric(db_conn, module_name, 'Ion_HMM_ConductionEvents_Total', 0, 'count', 'Total full conduction events (HMM)')
        store_metric(db_conn, module_name, 'Ion_HMM_ConductionEvents_Outward', 0, 'count', 'Outward conduction events (HMM)')
        store_metric(db_conn, module_name, 'Ion_HMM_ConductionEvents_Inward', 0, 'count', 'Inward conduction events (HMM)')
        # Store time metrics as NaN (or None) if no events
        store_metric(db_conn, module_name, 'Ion_HMM_Conduction_MeanTransitTime_ns', None, 'ns', 'Mean full transit time (HMM)')
        store_metric(db_conn, module_name, 'Ion_HMM_Conduction_MedianTransitTime_ns', None, 'ns', 'Median full transit time (HMM)')
        store_metric(db_conn, module_name, 'Ion_HMM_Conduction_StdTransitTime_ns', None, 'ns', 'Std Dev full transit time (HMM)')

    return rel_path

# --- Main Computation Orchestration (Revised Order) ---
def run_ion_analysis(
    run_dir: str,
    universe=None,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    db_conn: sqlite3.Connection = None,
    psf_file: Optional[str] = None,
    dcd_file: Optional[str] = None
) -> Dict[str, any]:
    """
    Orchestrates the ion analysis computational workflow using HMM for transitions.
    Corrected order: Track -> Calc G1 Ref -> Save Positions -> Calc/Optimize Sites -> Occupancy -> HMM.

    Args:
        run_dir: Path to the specific run directory.
        universe: Pre-loaded MDAnalysis Universe object.
        start_frame: Starting frame index for analysis (0-based). Defaults to 0.
        end_frame: Ending frame index for analysis (exclusive). If None, goes to the end.
        db_conn: Active database connection.
        psf_file: Path to the PSF topology file. Only used if universe is not provided.
        dcd_file: Path to the DCD trajectory file. Only used if universe is not provided.

    Returns:
        Dictionary containing status and potentially paths to key results or errors.
    """
    module_name = "ion_analysis"
    start_time = time.time()
    register_module(db_conn, module_name, status='running')
    logger.info(f"--- Starting Ion Analysis Computation (HMM) for {run_dir} ---")

    results = {'status': 'failed', 'error': None, 'files': {}} # Default status

    # --- Universe Handling ---
    try:
        import MDAnalysis as mda
        if universe is not None:
            # Use the provided universe
            u = universe
            logger.info(f"{module_name}: Using provided Universe object")
        else:
            # Need to load the universe from files
            if psf_file is None or dcd_file is None:
                error_msg = "Neither universe nor psf_file/dcd_file were provided."
                logger.error(f"{module_name}: {error_msg}")
                update_module_status(db_conn, module_name, 'failed', error_message=error_msg)
                results['error'] = error_msg
                return results
                
            logger.info(f"{module_name}: Loading topology: {psf_file}")
            logger.info(f"{module_name}: Loading trajectory: {dcd_file}")
            u = mda.Universe(psf_file, dcd_file)
            logger.info(f"{module_name}: Universe loaded successfully")
            
        # Validate universe
        n_frames_total = len(u.trajectory)
        logger.info(f"{module_name}: Universe has {n_frames_total} frames total")
        
        if n_frames_total < 2:
            error_msg = "Trajectory has < 2 frames."
            logger.error(f"{module_name}: {error_msg}")
            update_module_status(db_conn, module_name, 'failed', error_message=error_msg)
            results['error'] = error_msg
            return results
             
        # Handle frame range
        if end_frame is None:
            end_frame = n_frames_total
            
        # Validate frame range
        if start_frame < 0 or start_frame >= n_frames_total:
            error_msg = f"Invalid start_frame: {start_frame}. Must be between 0 and {n_frames_total-1}"
            logger.error(f"{module_name}: {error_msg}")
            update_module_status(db_conn, module_name, 'failed', error_message=error_msg)
            results['error'] = error_msg
            return results
            
        if end_frame <= start_frame or end_frame > n_frames_total:
            error_msg = f"Invalid end_frame: {end_frame}. Must be between {start_frame+1} and {n_frames_total}"
            logger.error(f"{module_name}: {error_msg}")
            update_module_status(db_conn, module_name, 'failed', error_message=error_msg)
            results['error'] = error_msg
            return results
            
        # Store frame range info in the database for later reference
        store_metric(db_conn, module_name, "start_frame", start_frame, "frame", "Starting frame index for ion analysis")
        store_metric(db_conn, module_name, "end_frame", end_frame, "frame", "Ending frame index for ion analysis")
        store_metric(db_conn, module_name, "frames_analyzed", end_frame - start_frame, "frames", "Number of frames analyzed")
            
        logger.info(f"{module_name}: Analyzing frame range {start_frame} to {end_frame} (total: {end_frame - start_frame} frames)")
        
    except Exception as e:
        error_msg = f"Failed to load or validate Universe: {e}"
        logger.error(error_msg, exc_info=True)
        update_module_status(db_conn, module_name, 'failed', error_message=error_msg)
        results['error'] = error_msg
        return results

    # --- 1. Identify Filter Residues ---
    filter_residues = find_filter_residues(u)
    if filter_residues is None:
        error_msg = "Failed to identify filter residues."
        logger.error(error_msg)
        update_module_status(db_conn, module_name, 'failed', error_message=error_msg)
        results['error'] = error_msg
        return results
    else:
        # <<< --- MODIFIED: Store filter_residues in database metadata using clean_json_data --- >>>
        try:
            # Use clean_json_data from core.utils to handle NumPy types
            cleaned_residues = clean_json_data(filter_residues)
            filter_res_json = json.dumps(cleaned_residues) # Dump the cleaned data
            # Use set_simulation_metadata imported from core.database
            set_simulation_metadata(db_conn, 'filter_residues_dict', filter_res_json)
            logger.info("Stored filter residue dictionary in database metadata.")
        except Exception as e_meta:
            error_msg = f"Failed to store filter_residues in database metadata: {e_meta}"
            logger.error(error_msg)
            update_module_status(db_conn, module_name, 'failed', error_message=error_msg)
            results['error'] = error_msg
            return results
        # <<< --- END OF MODIFIED SECTION --- >>>


    # --- 2. Calculate G1 Reference Z ---
    g1_ref = calculate_g1_reference(u, filter_residues, start_frame=start_frame)
    if g1_ref is None:
        error_msg = "Failed to calculate G1 reference Z."
        logger.error(error_msg)
        update_module_status(db_conn, module_name, 'failed', error_message=error_msg)
        results['error'] = error_msg
        return results

    # --- 3. Ion Tracking ---
    ions_z_abs, time_points, ion_indices = track_ion_positions(
        u, filter_residues, run_dir, db_conn, module_name, 
        start_frame=start_frame, end_frame=end_frame
    )
    if ions_z_abs is None or time_points is None or ion_indices is None:
        error_msg = "Ion tracking failed."
        logger.error(error_msg)
        update_module_status(db_conn, module_name, 'failed', error_message=error_msg)
        results['error'] = error_msg
        return results

    # --- 4. Save Position Data (Now that G1 ref is known) ---
    pos_g1_path, pos_abs_path, presence_path = _save_ion_position_csvs(
        run_dir, time_points, ions_z_abs, ion_indices, g1_ref, db_conn, module_name
    )
    if not pos_g1_path or not pos_abs_path: # Check both essential files were saved/registered
         error_msg = "Saving ion position data (G1/Abs) failed."
         logger.error(error_msg)
         update_module_status(db_conn, module_name, 'failed', error_message=error_msg)
         results['error'] = error_msg
         return results
    # Store registered paths
    results['files']['positions_g1_centric'] = pos_g1_path
    results['files']['positions_absolute'] = pos_abs_path
    if presence_path: results['files']['filter_presence'] = presence_path

    # --- 5. Calculate/Optimize Binding Sites (Now that absolute positions CSV is registered) ---
    # Note: calculate_binding_sites now takes g1_ref as input
    optimized_sites_rel = calculate_binding_sites(g1_ref, run_dir, db_conn, module_name)
    if optimized_sites_rel is None:
        error_msg = "Failed to calculate/optimize binding sites."
        logger.error(error_msg)
        update_module_status(db_conn, module_name, 'failed', error_message=error_msg)
        results['error'] = error_msg
        return results
    # Store registered path (registration happens within the function)
    results['files']['sites_definition'] = 'ion_analysis/binding_site_positions_g1_centric.txt'
    results['files']['site_optimization_plot'] = 'ion_analysis/binding_site_optimization.png'


    # --- 6. Occupancy Analysis (Uses optimized sites) ---
    occ_csv_path, stats_csv_path = calculate_occupancy(
        run_dir, time_points, ions_z_abs, ion_indices, optimized_sites_rel, g1_ref, db_conn, module_name
    )
    if occ_csv_path is None or stats_csv_path is None:
        error_msg = "Occupancy analysis failed (data/stats saving)."
        logger.error(error_msg)
        # Log warning but allow HMM to continue
        logger.warning(f"{error_msg} - HMM analysis will proceed but visualization/metrics might be affected.")
        # Do not set overall failure status yet
    if occ_csv_path: results['files']['occupancy_per_frame'] = occ_csv_path
    if stats_csv_path: results['files']['site_statistics'] = stats_csv_path


    # --- 7. HMM Transition Analysis ---
    logger.info("Starting HMM transition analysis...")
    hmm_success = False
    try:
        # Prepare HMM inputs
        site_names_hmm = sorted(optimized_sites_rel.keys(), key=lambda s: optimized_sites_rel[s], reverse=True)
        site_centers_hmm = np.array([optimized_sites_rel[s] for s in site_names_hmm])
        n_states = len(site_names_hmm)
        log_pi = np.full(n_states, -np.log(n_states)) # Uniform initial
        A, logA, adjacency = build_transition_matrix(n_states, HMM_SELF_TRANSITION_P, HMM_EPSILON)
        hmm_params = {
            'emission_sigma': HMM_EMISSION_SIGMA,
            'flicker_ns': HMM_FLICKER_NS
        }

        all_final_transitions = {}
        all_full_hmm_paths = {}
        all_entry_exit_events = {}
        all_quality_flags = {}

        # Load G1-centric data (path already retrieved and checked)
        df_g1 = pd.read_csv(os.path.join(run_dir, pos_g1_path))

        for ion_idx in ion_indices:
             ion_name = f"Ion_{ion_idx}"
             ion_col_name = f"Ion_{ion_idx}_Z_G1Centric"
             if ion_col_name not in df_g1.columns: continue
             ion_z_g1_nan = df_g1[ion_col_name].values

             transitions, hmm_path, entry_exit, quality_flags = process_ion_with_hmm(
                 ion_idx, ion_name, ion_z_g1_nan, time_points,
                 site_names_hmm, site_centers_hmm, log_pi, logA, adjacency, hmm_params
             )
             if transitions is not None: # Store results if processing succeeded
                 all_final_transitions[ion_idx] = transitions
                 all_full_hmm_paths[ion_idx] = hmm_path
                 all_entry_exit_events[ion_idx] = entry_exit
                 all_quality_flags[ion_idx] = quality_flags

        # Save HMM results
        hmm_dwell_path = _save_hmm_transition_data(run_dir, all_final_transitions, db_conn, module_name)
        hmm_quality_path = _save_hmm_quality_data(run_dir, all_quality_flags, all_full_hmm_paths, time_points, site_names_hmm, db_conn, module_name)

        if not hmm_dwell_path: raise ValueError("Failed to save HMM dwell event data.")
        results['files']['hmm_dwell_events'] = hmm_dwell_path
        if hmm_quality_path: results['files']['hmm_quality_data'] = hmm_quality_path

        # Calculate and store HMM stats
        _calculate_and_store_hmm_stats(all_final_transitions, site_names_hmm, db_conn, module_name)

        # Calculate and store HMM conduction events
        hmm_cond_path = _calculate_and_save_hmm_conduction(run_dir, all_final_transitions, site_names_hmm, db_conn, module_name)
        if hmm_cond_path: results['files']['hmm_conduction_events'] = hmm_cond_path

        # Log aggregated warnings for HMM processing
        warning_aggregator.log_summary()
        
        hmm_success = True

    except Exception as e_hmm:
        error_msg = f"HMM analysis failed: {e_hmm}"
        logger.error(error_msg, exc_info=True)
        # Update status immediately if HMM fails
        update_module_status(db_conn, module_name, 'failed', error_message=error_msg)
        results['error'] = error_msg
        # Keep going only to finalize overall status below

    # --- Finalize ---
    exec_time = time.time() - start_time
    # Success depends on G1 ref, tracking, position saving, site optimization, and HMM success
    # Note: Occupancy failure is currently logged as a warning but doesn't prevent success status if HMM works.
    critical_steps_ok = (g1_ref is not None and
                         ions_z_abs is not None and
                         pos_g1_path is not None and
                         pos_abs_path is not None and
                         optimized_sites_rel is not None and
                         hmm_success) # Occupancy result not strictly required for success status

    final_status = 'success' if critical_steps_ok else 'failed'
    if not critical_steps_ok and not results['error']:
         # Assign a more specific error if HMM failed but other steps were OK
         if not hmm_success and results.get('error') and "HMM analysis failed" in results['error']:
              pass # Keep the HMM specific error
         else:
              results['error'] = "One or more critical ion analysis steps failed (HMM workflow, revised order)."

    # Update status one last time with final determination
    update_module_status(db_conn, module_name, final_status, execution_time=exec_time, error_message=results['error'])
    logger.info(f"--- Ion Analysis Computation (HMM) finished in {exec_time:.2f} seconds (Status: {final_status}) ---")
    results['status'] = final_status

    return results