# filename: pore_analysis/modules/tyrosine_analysis/computation.py
"""
Computation functions for analyzing the Chi1 and Chi2 dihedral angles and
rotameric states of the Selectivity Filter (SF) Tyrosine residue using HMM.
Handles data calculation, HMM processing, saving results, and metric storage.
Retrieves filter residue definitions from the database.
Also saves raw dihedral angle time series.
"""

import os
import logging
import numpy as np
import pandas as pd
import time
import sqlite3
import json
from tqdm import tqdm
import MDAnalysis as mda
from typing import Dict, Optional, Tuple, List, Any
from collections import defaultdict

# Import from core and other modules
try:
    from pore_analysis.core.utils import frames_to_time
    from pore_analysis.core.logging import setup_system_logger
    from pore_analysis.core.config import (
        FRAMES_PER_NS,
        TYR_HMM_STATES, TYR_HMM_STATE_ORDER, TYR_HMM_EMISSION_SIGMA,
        TYR_HMM_SELF_TRANSITION_P, TYR_HMM_EPSILON, TYR_HMM_FLICKER_NS
    )
    from pore_analysis.core.database import (
        register_module, update_module_status, register_product, store_metric,
        get_simulation_metadata, get_config_parameters,
        set_simulation_metadata # <<< ADDED IMPORT
    )
    from pore_analysis.modules.ion_analysis.hmm import (
        build_transition_matrix,
        segment_and_filter,
        identify_continuous_segments
    )
except ImportError as e:
    print(f"Error importing dependency modules in tyrosine_analysis/computation.py: {e}")
    raise

logger = logging.getLogger(__name__)

# --- Helper Functions --- #

def calc_dihedral(p1, p2, p3, p4):
    """Calculates the dihedral angle from four points in degrees."""
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    norm_b2 = np.linalg.norm(b2)
    if norm_b2 < 1e-9: return np.nan
    m1 = np.cross(n1, b2 / norm_b2)
    norm_n1 = np.linalg.norm(n1)
    norm_n2 = np.linalg.norm(n2)
    if norm_n1 < 1e-9 or norm_n2 < 1e-9: return np.nan
    x = np.clip(np.dot(n1 / norm_n1, n2 / norm_n2), -1.0, 1.0)
    y = np.dot(m1, n2 / norm_n2)
    angle_rad = np.arctan2(y, x)
    angle_deg = np.degrees(angle_rad)
    return angle_deg

def viterbi_2d_angles(obs: np.ndarray, state_centers: np.ndarray, log_pi: np.ndarray, logA: np.ndarray, emission_sigma: float) -> tuple[np.ndarray, float]:
    """
    Viterbi decoding adapted for 2D angular observations (Chi1, Chi2).
    Assumes independent Gaussian emissions for Chi1 and Chi2 with the same sigma.
    Handles circularity implicitly via Gaussian probability density over -180 to 180.
    """
    T = obs.shape[0]
    nstates = state_centers.shape[0]
    if T == 0: return np.array([], dtype=int), 0.0

    logB = np.zeros((T, nstates))
    sigma_sq = emission_sigma ** 2
    log_norm_factor = -np.log(2 * np.pi * sigma_sq)

    for t in range(T):
        delta_chi1 = obs[t, 0] - state_centers[:, 0]
        delta_chi2 = obs[t, 1] - state_centers[:, 1]
        delta_chi1 = (delta_chi1 + 180) % 360 - 180
        delta_chi2 = (delta_chi2 + 180) % 360 - 180
        sq_dist = delta_chi1**2 + delta_chi2**2
        logB[t, :] = log_norm_factor - 0.5 * sq_dist / sigma_sq

    V = np.full((T, nstates), -np.inf)
    ptr = np.zeros((T, nstates), dtype=int)
    V[0] = log_pi + logB[0]

    for t in range(1, T):
        for j in range(nstates):
            scores = V[t - 1] + logA[:, j]
            ptr[t, j] = np.argmax(scores)
            V[t, j] = scores[ptr[t, j]] + logB[t, j]

    path = np.zeros(T, dtype=int)
    path[-1] = np.argmax(V[-1])
    final_log_likelihood = V[-1, path[-1]]

    for t in range(T - 2, -1, -1):
        path[t] = ptr[t + 1, path[t + 1]]

    return path, final_log_likelihood

def _save_raw_dihedral_data(output_dir, time_points, chi1_values, chi2_values, run_dir, db_conn, module_name):
    """Saves the raw calculated tyrosine dihedral angle time series data."""
    logger.info("Saving SF Tyrosine raw dihedral angle time series data...")
    data_to_save = {'Time (ns)': time_points}
    all_chains = list(chi1_values.keys()) # Assumes chi1_values has keys for all analyzed chains

    max_len = len(time_points)
    if max_len == 0:
        logger.warning("No time points available, cannot save raw dihedral data.")
        return None

    for chain in all_chains:
        chi1_data = chi1_values.get(chain, [])
        chi2_data = chi2_values.get(chain, [])
        # Use pre-allocated arrays directly, avoiding padding logic if possible
        # Assuming chi1_values[chain] and chi2_values[chain] are already numpy arrays of length max_len
        if len(chi1_data) == max_len:
            data_to_save[f'{chain}_Chi1'] = chi1_data
        else:
            logger.error(f"Length mismatch for raw Chi1 data chain {chain} (expected {max_len}, got {len(chi1_data)})")
            return None # Fail if lengths don't match pre-allocation

        if len(chi2_data) == max_len:
            data_to_save[f'{chain}_Chi2'] = chi2_data
        else:
             logger.error(f"Length mismatch for raw Chi2 data chain {chain} (expected {max_len}, got {len(chi2_data)})")
             return None # Fail if lengths don't match pre-allocation

    try:
        df = pd.DataFrame(data_to_save)
    except ValueError as e:
        logger.error(f"Error creating DataFrame for raw dihedral data: {e}. Check column lengths.")
        return None

    # Use the passed output_dir correctly
    csv_path = os.path.join(output_dir, "sf_tyrosine_raw_dihedrals.csv")
    rel_path = os.path.relpath(csv_path, run_dir)
    try:
        df.to_csv(csv_path, index=False, float_format='%.3f', na_rep='NaN')
        logger.info(f"Saved SF Tyrosine raw dihedral time series data to {csv_path}")
        register_product(db_conn, module_name, "csv", "data", rel_path,
                         subcategory="raw_dihedrals",
                         description="Time series of raw SF Tyr Chi1/Chi2 angles per chain.")
        return rel_path
    except Exception as e:
        logger.error(f"Failed to save/register SF Tyrosine raw dihedral CSV: {e}")
        return None

def _save_tyr_hmm_dwells(output_dir: str, run_dir: str, all_dwell_events: Dict, db_conn: sqlite3.Connection, module_name: str) -> Optional[str]:
    """Saves HMM-derived tyrosine dwell events to CSV."""
    os.makedirs(output_dir, exist_ok=True) # Ensure output_dir exists
    compiled_events = []
    for chain, events in all_dwell_events.items():
        for event in events:
            compiled_events.append({
                'Chain': chain,
                'Start Frame': event['start_frame'], 'End Frame': event['end_frame'],
                'Start Time (ns)': event['start_time'], 'End Time (ns)': event['end_time'],
                'Rotamer State': event['site_label'],
                'Duration (ns)': event['end_time'] - event['start_time']
            })
    if not compiled_events: logger.info("No Tyr HMM dwell events to save."); return None
    try:
        df = pd.DataFrame(compiled_events).sort_values(by=['Chain', 'Start Frame'])
        csv_path = os.path.join(output_dir, 'sf_tyrosine_hmm_dwell_events.csv')
        df.to_csv(csv_path, index=False, float_format='%.4f')
        rel_path = os.path.relpath(csv_path, run_dir)
        register_product(db_conn, module_name, "csv", "data", rel_path,
                         subcategory="hmm_dwell_events",
                         description="HMM-derived SF Tyr rotamer dwell events.")
        logger.info(f"Saved Tyr HMM dwell events to {csv_path}")
        return rel_path
    except Exception as e: logger.error(f"Failed to save Tyr HMM dwell events: {e}"); return None

def _save_tyr_hmm_paths(output_dir: str, run_dir: str, time_points: np.ndarray, all_hmm_paths: Dict, state_names: List, db_conn: sqlite3.Connection, module_name: str) -> Optional[str]:
    """Saves the raw HMM state path per frame for each chain."""
    chains = sorted(all_hmm_paths.keys())
    if not chains: logger.info("No Tyr HMM paths to save."); return None
    n_frames = len(time_points)
    data = {'Time (ns)': time_points}
    for chain in chains:
        path_indices = all_hmm_paths[chain]
        if len(path_indices) == n_frames:
             data[f'{chain}_StateIdx'] = path_indices
             # Check bounds before indexing state_names
             data[f'{chain}_State'] = [state_names[idx] if 0 <= idx < len(state_names) else 'Outside' for idx in path_indices]
        else:
             logger.warning(f"Length mismatch for HMM path of chain {chain}. Padding with 'Error'.")
             data[f'{chain}_StateIdx'] = np.full(n_frames, -2)
             data[f'{chain}_State'] = ['Error'] * n_frames
    try:
        df = pd.DataFrame(data)
        csv_path = os.path.join(output_dir, 'sf_tyrosine_hmm_state_path.csv')
        df.to_csv(csv_path, index=False, float_format='%.4f')
        rel_path = os.path.relpath(csv_path, run_dir)
        register_product(db_conn, module_name, "csv", "data", rel_path,
                         subcategory="hmm_state_path",
                         description="Raw HMM state assignment per frame for SF Tyr.")
        logger.info(f"Saved Tyr HMM state path data to {csv_path}")
        return rel_path
    except Exception as e: logger.error(f"Failed to save Tyr HMM path data: {e}"); return None

# --- <<< MODIFIED: Use set_simulation_metadata for dominant state >>> ---
def _calculate_and_store_tyr_hmm_stats(all_dwell_events, state_names, db_conn, module_name, hmm_params):
    """Calculates statistics from HMM dwell events and stores metrics."""
    logger.info("Calculating SF Tyr HMM statistics...")
    total_transitions = 0
    dwell_times = defaultdict(list)
    total_time_in_state = defaultdict(float)
    chain_transitions = defaultdict(int)

    for chain, events in all_dwell_events.items():
        sorted_events = sorted(events, key=lambda x: x['start_time'])
        for i, event in enumerate(sorted_events):
            duration = event['end_time'] - event['start_time']
            state = event['site_label']
            if state != 'Outside':
                 dwell_times[state].append(duration)
                 total_time_in_state[state] += duration
            if i > 0:
                prev_state = sorted_events[i-1]['site_label']
                if state != prev_state and state != 'Outside' and prev_state != 'Outside':
                     total_transitions += 1
                     chain_transitions[chain] += 1

    total_analyzed_time = sum(total_time_in_state.values())
    dominant_state = max(total_time_in_state, key=total_time_in_state.get) if total_time_in_state else 'N/A'
    state_populations = {state: (time / total_analyzed_time * 100) if total_analyzed_time > 0 else 0
                         for state, time in total_time_in_state.items()}

    logger.info(f"Tyr HMM Stats: Dominant={dominant_state}, Total Transitions={total_transitions}")

    # --- FIX: Store Dominant State in Metadata ---
    if dominant_state != 'N/A':
        set_simulation_metadata(db_conn, 'Tyr_HMM_DominantState', dominant_state)
    else:
        set_simulation_metadata(db_conn, 'Tyr_HMM_DominantState', 'N/A')
    # --- END FIX ---

    # Store NUMERICAL metrics as before
    store_metric(db_conn, module_name, 'Tyr_HMM_TotalTransitions', total_transitions, 'count', 'Total transitions between rotamer states (HMM)')

    for state in state_names:
        if state == 'Outside': continue
        mean_dwell = float(np.mean(dwell_times[state])) if dwell_times[state] else 0.0
        median_dwell = float(np.median(dwell_times[state])) if dwell_times[state] else 0.0
        std_dwell = float(np.std(dwell_times[state])) if len(dwell_times[state]) > 1 else 0.0
        population = state_populations.get(state, 0.0)
        store_metric(db_conn, module_name, f'Tyr_HMM_MeanDwell_{state}', mean_dwell, 'ns', f'Mean dwell time in state {state} (HMM)')
        store_metric(db_conn, module_name, f'Tyr_HMM_MedianDwell_{state}', median_dwell, 'ns', f'Median dwell time in state {state} (HMM)')
        store_metric(db_conn, module_name, f'Tyr_HMM_StdDwell_{state}', std_dwell, 'ns', f'Std Dev dwell time in state {state} (HMM)')
        store_metric(db_conn, module_name, f'Tyr_HMM_Population_{state}', population, '%', f'Population of state {state} (HMM)')

    store_metric(db_conn, module_name, 'Config_TyrHMM_EmissionSigma', hmm_params['emission_sigma'], 'degrees', 'Tyrosine HMM emission sigma used')
    store_metric(db_conn, module_name, 'Config_TyrHMM_SelfTransitionP', hmm_params['self_transition_p'], 'probability', 'Tyrosine HMM self-transition probability used')
    store_metric(db_conn, module_name, 'Config_TyrHMM_Epsilon', hmm_params['epsilon'], 'probability', 'Tyrosine HMM epsilon transition probability used')
    store_metric(db_conn, module_name, 'Config_TyrHMM_FlickerNs', hmm_params['flicker_ns'], 'ns', 'Tyrosine HMM flicker filter duration used')
# --- <<< END MODIFICATION for dominant state storage >>> ---

# --- Main Computation Function --- #
def run_tyrosine_analysis(
    run_dir: str,
    universe=None,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    db_conn: sqlite3.Connection = None,
    psf_file: Optional[str] = None,
    dcd_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Performs SF Tyrosine rotamer analysis using HMM, saves data, calculates/stores metrics.
    Retrieves filter residue definitions from the database.

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
    module_name = "tyrosine_analysis"
    start_time = time.time()
    register_module(db_conn, module_name, status='running')
    logger_local = setup_system_logger(run_dir)
    if logger_local is None: logger_local = logging.getLogger(__name__)

    results: Dict[str, Any] = {'status': 'failed', 'error': None}
    # Define output_dir correctly once
    output_dir = os.path.join(run_dir, module_name)
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Load Universe if not provided
        if universe is not None:
            u = universe
            logger_local.info("Using provided Universe object.")
        else:
            if psf_file is None or dcd_file is None:
                raise ValueError("If universe is not provided, both psf_file and dcd_file must be specified.")
            logger_local.info(f"Loading topology: {psf_file}")
            logger_local.info(f"Loading trajectory: {dcd_file}")
            u = mda.Universe(psf_file, dcd_file)
            logger_local.info("Universe loaded from files.")
        
        # Validate frame range
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
        store_metric(db_conn, module_name, "start_frame", start_frame, "frame", "Starting frame index for tyrosine analysis")
        store_metric(db_conn, module_name, "end_frame", actual_end, "frame", "Ending frame index for tyrosine analysis")
        store_metric(db_conn, module_name, "frames_analyzed", actual_end - start_frame, "frame", "Number of frames analyzed")
        
        if n_frames_total < 2: raise ValueError("Trajectory has < 2 frames.")

        # Retrieve Filter Residues from DB
        logger_local.info("Retrieving filter residue definitions from database...")
        filter_res_json = get_simulation_metadata(db_conn, 'filter_residues_dict')
        if filter_res_json:
             try:
                 filter_residues = json.loads(filter_res_json)
                 if not isinstance(filter_residues, dict) or not filter_residues:
                      raise TypeError("Parsed filter_residues_dict is not a valid dictionary.")
                 logger_local.info(f"Successfully retrieved filter residues for chains: {list(filter_residues.keys())}")
             except (json.JSONDecodeError, TypeError) as e:
                  raise ValueError(f"Failed to parse filter_residues_dict from database metadata: {e}")
        else:
            raise ValueError("filter_residues_dict not found in simulation_metadata table. Run ion_analysis first.")

        # Identify SF Tyrosine Resids
        sf_tyrosine_resids = {}
        chains_to_analyze = list(filter_residues.keys())
        for chain in chains_to_analyze[:]:
            resids = filter_residues[chain]
            # Assuming TVGYG, Tyrosine is at index 3
            if len(resids) >= 4: sf_tyrosine_resids[chain] = resids[3]
            else: logger_local.warning(f"Chain {chain}: Filter too short ({len(resids)} res). Skipping Tyr analysis."); chains_to_analyze.remove(chain)
        if not sf_tyrosine_resids: raise ValueError("Could not identify SF Tyrosine in any chain with sufficient filter length.")

        # Calculate Dihedrals
        time_points_list = []
        # Pre-allocate numpy arrays for the specified frame range
        frames_to_analyze = actual_end - start_frame
        chi1_values = {chain: np.full(frames_to_analyze, np.nan) for chain in chains_to_analyze}
        chi2_values = {chain: np.full(frames_to_analyze, np.nan) for chain in chains_to_analyze}

        logger_local.info("Calculating SF Tyrosine dihedrals...")
        chi1_selections = {chain: u.select_atoms(f"segid {chain} and resid {sf_tyrosine_resids[chain]} and name N CA CB CG") for chain in chains_to_analyze}
        chi2_selections = {chain: u.select_atoms(f"segid {chain} and resid {sf_tyrosine_resids[chain]} and name CA CB CG CD1") for chain in chains_to_analyze}

        # Iterate over specified frame range
        for ts in tqdm(u.trajectory[start_frame:actual_end], 
                       desc="SF Tyr Dihedrals", 
                       unit="frame", 
                       disable=not logger_local.isEnabledFor(logging.INFO)):
            frame_idx = ts.frame
            # Calculate local index for storing in arrays
            local_idx = frame_idx - start_frame
            time_points_list.append(frames_to_time([frame_idx])[0])

            for chain in chains_to_analyze:
                try:
                    chi1_sel = chi1_selections[chain]
                    chi2_sel = chi2_selections[chain]
                    if len(chi1_sel) == 4 and len(chi2_sel) == 4:
                        # Store values at the local index position
                        chi1_values[chain][local_idx] = calc_dihedral(*(chi1_sel[i].position for i in range(4)))
                        chi2_values[chain][local_idx] = calc_dihedral(*(chi2_sel[i].position for i in range(4)))
                    elif local_idx == 0:
                         logger_local.warning(f"Incorrect atom count for dihedrals in chain {chain}")
                except Exception as e_calc: pass # Keep iterating other chains/frames

        time_points = np.array(time_points_list)

        # --- HMM Analysis ---
        logger_local.info("Starting HMM analysis for SF Tyrosine...")
        db_params = get_config_parameters(db_conn)
        def get_param(name, default, target_type):
            p_info = db_params.get(name)
            if p_info and p_info.get('value') is not None:
                try: return target_type(p_info['value'])
                except: return default
            return default

        tyr_hmm_sigma = get_param('TYR_HMM_EMISSION_SIGMA', TYR_HMM_EMISSION_SIGMA, float)
        tyr_hmm_self_p = get_param('TYR_HMM_SELF_TRANSITION_P', TYR_HMM_SELF_TRANSITION_P, float)
        tyr_hmm_epsilon = get_param('TYR_HMM_EPSILON', TYR_HMM_EPSILON, float)
        tyr_hmm_flicker = get_param('TYR_HMM_FLICKER_NS', TYR_HMM_FLICKER_NS, float)
        if tyr_hmm_sigma <= 0:
            logger.warning(f"Invalid TYR_HMM_EMISSION_SIGMA ({tyr_hmm_sigma}), using default: {TYR_HMM_EMISSION_SIGMA}")
            tyr_hmm_sigma = TYR_HMM_EMISSION_SIGMA

        hmm_params_used = {
            'emission_sigma': tyr_hmm_sigma,
            'self_transition_p': tyr_hmm_self_p,
            'epsilon': tyr_hmm_epsilon,
            'flicker_ns': tyr_hmm_flicker
        }
        logger_local.info(f"Using HMM parameters: {hmm_params_used}")

        state_names = TYR_HMM_STATE_ORDER
        state_centers = np.array([TYR_HMM_STATES[name] for name in state_names])
        n_states = len(state_names)
        log_pi = np.full(n_states, -np.log(n_states))

        A = np.full((n_states, n_states), tyr_hmm_epsilon)
        np.fill_diagonal(A, tyr_hmm_self_p)
        A = A / A.sum(axis=1, keepdims=True)
        logA = np.log(A)

        all_final_dwells = {}
        all_hmm_paths = {}

        for chain in chains_to_analyze:
            logger_local.info(f"Running HMM for chain {chain}...")
            obs_chi1 = chi1_values[chain]
            obs_chi2 = chi2_values[chain]

            valid_chain_mask = np.isfinite(obs_chi1) & np.isfinite(obs_chi2)
            chain_obs_2d = np.column_stack((obs_chi1[valid_chain_mask], obs_chi2[valid_chain_mask]))

            if len(chain_obs_2d) > 0:
                path_indices, _ = viterbi_2d_angles(chain_obs_2d, state_centers, log_pi, logA, tyr_hmm_sigma)
                full_chain_path = np.full(frames_to_analyze, -1, dtype=int) # -1 for 'Outside'/NaN
                valid_indices_for_chain = np.where(valid_chain_mask)[0]

                if len(path_indices) == len(valid_indices_for_chain):
                     full_chain_path[valid_indices_for_chain] = path_indices
                else:
                     logger.error(f"Path length mismatch for chain {chain} after Viterbi.")
                     full_chain_path.fill(-2) # Use -2 for error state

                all_hmm_paths[chain] = full_chain_path

                chain_dwells_dict = segment_and_filter(full_chain_path, state_names, time_points, tyr_hmm_flicker, f"Tyr_{chain}")
                chain_dwells = []
                for seg_dict in chain_dwells_dict:
                     # Local frame indices within our analysis window
                     start_f, end_f = seg_dict['start'], seg_dict['end']
                     if 0 <= start_f < frames_to_analyze and 0 <= end_f < frames_to_analyze:
                         # Translate to original frame indices for reporting
                         orig_start_f = start_f + start_frame
                         orig_end_f = end_f + start_frame
                         chain_dwells.append({
                              'start_frame': orig_start_f, 'end_frame': orig_end_f,
                              'start_time': time_points[start_f], 'end_time': time_points[end_f],
                              'site_label': seg_dict['label']
                         })
                all_final_dwells[chain] = chain_dwells
            else:
                 logger_local.warning(f"No valid dihedral data points for chain {chain}, skipping HMM.")
                 all_hmm_paths[chain] = np.full(frames_to_analyze, -1, dtype=int)
                 all_final_dwells[chain] = []

        # --- Save HMM Dwell Events ---
        # Pass correct output_dir (defined once above) and run_dir
        dwell_path = _save_tyr_hmm_dwells(output_dir, run_dir, all_final_dwells, db_conn, module_name)
        # --- Save HMM State Paths ---
        path_save = _save_tyr_hmm_paths(output_dir, run_dir, time_points, all_hmm_paths, state_names, db_conn, module_name)
        # --- Save original raw dihedral data ---
        raw_dihedral_path = _save_raw_dihedral_data(output_dir, time_points, chi1_values, chi2_values, run_dir, db_conn, module_name)

        # Calculate and Store HMM Stats
        _calculate_and_store_tyr_hmm_stats(all_final_dwells, state_names, db_conn, module_name, hmm_params_used)

        # Check if essential files were saved
        if not dwell_path or not path_save or not raw_dihedral_path:
             results['error'] = "Failed to save one or more essential Tyr HMM/raw data files."
             results['status'] = 'failed'
             logger.error(results['error'])
        else:
             results['status'] = 'success'

    except Exception as e_main:
        error_msg = f"Error during Tyrosine HMM analysis: {e_main}"
        logger_local.error(error_msg, exc_info=True)
        results['error'] = error_msg
        results['status'] = 'failed' # Ensure status is failed

    # Finalize
    exec_time = time.time() - start_time
    update_module_status(db_conn, module_name, results['status'], execution_time=exec_time, error_message=results['error'])
    logger_local.info(f"--- Tyrosine Analysis HMM computation finished in {exec_time:.2f} seconds (Status: {results['status']}) ---")

    return results