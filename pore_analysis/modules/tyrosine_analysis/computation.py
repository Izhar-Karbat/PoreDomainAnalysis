# filename: pore_analysis/modules/tyrosine_analysis/computation.py
"""
Computation functions for analyzing the Chi1 and Chi2 dihedral angles and
rotameric states of the Selectivity Filter (SF) Tyrosine residue using HMM.
Handles data calculation, HMM processing, saving results, and metric storage.
Retrieves filter residue definitions from the database.
Also performs Tyr-Thr inter-subunit hydrogen bond analysis, with options for
auto-detecting thresholds similar to DW-Gate analysis and using a residue offset
for Thr selection.
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
from scipy.stats import gaussian_kde # For KDE threshold detection
from scipy.signal import find_peaks # For KDE threshold detection

# Import from core and other modules
try:
    from pore_analysis.core.utils import frames_to_time, OneLetter, clean_json_data
    from pore_analysis.core.logging import setup_system_logger
    from pore_analysis.core.config import (
        FRAMES_PER_NS,
        TYR_HMM_STATES, TYR_HMM_STATE_ORDER, TYR_HMM_EMISSION_SIGMA,
        TYR_HMM_SELF_TRANSITION_P, TYR_HMM_EPSILON, TYR_HMM_FLICKER_NS,
        # Import needed from config for H-bond analysis
        TYR_THR_DEFAULT_FORMED_REF_DIST,
        TYR_THR_DEFAULT_BROKEN_REF_DIST,
        TYR_THR_RESIDUE_OFFSET,
        DW_GATE_AUTO_DETECT_REFS,
        DW_GATE_TOLERANCE_FRAMES
    )
    from pore_analysis.core.database import (
        register_module, update_module_status, register_product, store_metric,
        get_simulation_metadata, get_config_parameters,
        set_simulation_metadata
    )
    from pore_analysis.modules.ion_analysis.hmm import (
        build_transition_matrix, # Note: Simple matrix used for Tyr rotamers currently
        segment_and_filter,
        identify_continuous_segments,
        warning_aggregator # Use the shared warning aggregator
    )
    # Import Tyr-Thr H-bond analysis functions from dedicated module
    from pore_analysis.modules.tyrosine_analysis.tyr_thr_hbond import (
        calc_tyr_thr_hbond_distance,
        determine_tyr_thr_ref_distances,
        determine_tyr_thr_hbond_states,
        save_tyr_thr_hbond_data,
        calculate_and_store_tyr_thr_hbond_metrics, # Legacy, might be removed if new one is sole source
        calculate_and_store_tyr_thr_hbond_stats_from_events
    )
    # Import build_events_from_states for Tyr-Thr H-bond event building
    from pore_analysis.modules.dw_gate_analysis.event_building import build_events_from_states
except ImportError as e:
    print(f"Error importing dependency modules in tyrosine_analysis/computation.py: {e}")
    raise

logger = logging.getLogger(__name__)

# --- Helper Functions (calc_dihedral, viterbi_2d_angles, _save_raw_dihedral_data, etc. remain unchanged) ---

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
        if len(chi1_data) == max_len: data_to_save[f'{chain}_Chi1'] = chi1_data
        else: logger.error(f"Length mismatch for raw Chi1 data chain {chain}"); return None
        if len(chi2_data) == max_len: data_to_save[f'{chain}_Chi2'] = chi2_data
        else: logger.error(f"Length mismatch for raw Chi2 data chain {chain}"); return None

    try: df = pd.DataFrame(data_to_save)
    except ValueError as e: logger.error(f"Error creating DataFrame for raw dihedral data: {e}. Check lengths."); return None

    csv_path = os.path.join(output_dir, "sf_tyrosine_raw_dihedrals.csv")
    rel_path = os.path.relpath(csv_path, run_dir)
    try:
        df.to_csv(csv_path, index=False, float_format='%.3f', na_rep='NaN')
        logger.info(f"Saved SF Tyrosine raw dihedral time series data to {csv_path}")
        register_product(db_conn, module_name, "csv", "data", rel_path, subcategory="raw_dihedrals", description="Time series of raw SF Tyr Chi1/Chi2 angles per chain.")
        return rel_path
    except Exception as e: logger.error(f"Failed to save/register SF Tyrosine raw dihedral CSV: {e}"); return None

def _save_tyr_hmm_dwells(output_dir: str, run_dir: str, all_dwell_events: Dict, db_conn: sqlite3.Connection, module_name: str) -> Optional[str]:
    """Saves HMM-derived tyrosine dwell events to CSV."""
    os.makedirs(output_dir, exist_ok=True)
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
        register_product(db_conn, module_name, "csv", "data", rel_path, subcategory="hmm_dwell_events", description="HMM-derived SF Tyr rotamer dwell events.")
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
            data[f'{chain}_State'] = [state_names[idx] if 0 <= idx < len(state_names) else 'Outside' for idx in path_indices]
        else:
            logger.warning(f"Length mismatch for HMM path of chain {chain}. Padding with 'Error'.")
            data[f'{chain}_StateIdx'] = np.full(n_frames, -2); data[f'{chain}_State'] = ['Error'] * n_frames
    try:
        df = pd.DataFrame(data)
        csv_path = os.path.join(output_dir, 'sf_tyrosine_hmm_state_path.csv')
        df.to_csv(csv_path, index=False, float_format='%.4f')
        rel_path = os.path.relpath(csv_path, run_dir)
        register_product(db_conn, module_name, "csv", "data", rel_path, subcategory="hmm_state_path", description="Raw HMM state assignment per frame for SF Tyr.")
        logger.info(f"Saved Tyr HMM state path data to {csv_path}")
        return rel_path
    except Exception as e: logger.error(f"Failed to save Tyr HMM path data: {e}"); return None

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

    # Store Dominant State in Metadata
    set_simulation_metadata(db_conn, 'Tyr_HMM_DominantState', dominant_state if dominant_state != 'N/A' else 'N/A')
    # Store NUMERICAL metrics
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


# --- Tyr-Thr Hydrogen Bond Functions --- #
# These would typically be in tyr_thr_hbond.py but are included here as per the user's pasted file.
# I will assume the external `pore_analysis.modules.tyrosine_analysis.tyr_thr_hbond` functions are the correct ones.


# --- Main Computation Function (Updated) --- #
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
    Performs SF Tyrosine rotamer analysis using HMM and Tyr-Thr H-bond analysis.
    Optionally auto-detects H-bond thresholds. Uses residue offset for Thr selection.
    Saves data, calculates/stores metrics.
    """
    module_name = "tyrosine_analysis"
    start_time_module = time.time() # Renamed to avoid conflict if 'start_time' is used later
    register_module(db_conn, module_name, status='running')
    logger_local = setup_system_logger(run_dir)
    if logger_local is None: logger_local = logging.getLogger(__name__)

    results: Dict[str, Any] = {'status': 'failed', 'error': None}
    output_dir = os.path.join(run_dir, module_name)
    os.makedirs(output_dir, exist_ok=True)

    try:
        # --- Fetch Config (includes TYR_THR_RESIDUE_OFFSET, DW_GATE_AUTO_DETECT_REFS) ---
        db_params = get_config_parameters(db_conn)
        def get_param(name, default, target_type):
            p_info = db_params.get(name)
            if p_info and p_info.get('value') is not None:
                try:
                    if target_type == bool: return str(p_info['value']).lower() == 'true'
                    return target_type(p_info['value'])
                except: return default
            return default

        tyr_thr_residue_offset = get_param('TYR_THR_RESIDUE_OFFSET', -6, int)
        auto_detect_hbond_refs = get_param('DW_GATE_AUTO_DETECT_REFS', True, bool)
        default_formed_ref = get_param('TYR_THR_DEFAULT_FORMED_REF_DIST', 3.5, float)
        default_broken_ref = get_param('TYR_THR_DEFAULT_BROKEN_REF_DIST', 4.5, float)
        tolerance_frames_hbond = get_param('DW_GATE_TOLERANCE_FRAMES', 5, int) # Use common tolerance for H-bond too
        
        tyr_hmm_sigma = get_param('TYR_HMM_EMISSION_SIGMA', TYR_HMM_EMISSION_SIGMA, float)
        tyr_hmm_self_p = get_param('TYR_HMM_SELF_TRANSITION_P', TYR_HMM_SELF_TRANSITION_P, float)
        tyr_hmm_epsilon = get_param('TYR_HMM_EPSILON', TYR_HMM_EPSILON, float)
        tyr_hmm_flicker = get_param('TYR_HMM_FLICKER_NS', TYR_HMM_FLICKER_NS, float)
        if tyr_hmm_sigma <= 0: logger_local.warning(f"Invalid Tyr HMM sigma {tyr_hmm_sigma}, using default."); tyr_hmm_sigma = TYR_HMM_EMISSION_SIGMA

        hmm_params_used = {'emission_sigma': tyr_hmm_sigma, 'self_transition_p': tyr_hmm_self_p, 'epsilon': tyr_hmm_epsilon, 'flicker_ns': tyr_hmm_flicker}
        
        # Determine dt_ns for H-bond event building (needed if FRAMES_PER_NS is not from config)
        frames_per_ns_config = get_param('FRAMES_PER_NS', FRAMES_PER_NS, float)
        if frames_per_ns_config <= 0:
            logger_local.error(f"FRAMES_PER_NS from config is invalid ({frames_per_ns_config}). Aborting.")
            raise ValueError("Invalid FRAMES_PER_NS from configuration.")
        dt_ns = 1.0 / frames_per_ns_config


        # Load Universe & Validate Frames
        if universe is not None: 
            u = universe
            logger_local.info("Using provided Universe object.")
        else:
            if psf_file is None or dcd_file is None: raise ValueError("PSF/DCD must be specified if universe not provided.")
            logger_local.info(f"Loading topology: {psf_file}, trajectory: {dcd_file}")
            u = mda.Universe(psf_file, dcd_file)
            logger_local.info("Universe loaded from files.")
        
        n_frames_total = len(u.trajectory)
        if start_frame < 0 or start_frame >= n_frames_total: raise ValueError(f"Invalid start_frame: {start_frame}")
        actual_end = end_frame if end_frame is not None else n_frames_total
        if actual_end <= start_frame or actual_end > n_frames_total: raise ValueError(f"Invalid end_frame: {actual_end}")
        frames_to_analyze = actual_end - start_frame
        logger_local.info(f"Using frame range: {start_frame} to {actual_end} ({frames_to_analyze} frames)")
        store_metric(db_conn, module_name, "start_frame", start_frame, "frame")
        store_metric(db_conn, module_name, "end_frame", actual_end, "frame")
        store_metric(db_conn, module_name, "frames_analyzed", frames_to_analyze, "frame")
        if n_frames_total < 2: raise ValueError("Trajectory has < 2 frames.")

        # Retrieve Filter Residues
        logger_local.info("Retrieving filter residue definitions from database...")
        filter_res_json = get_simulation_metadata(db_conn, 'filter_residues_dict')
        if filter_res_json: 
            filter_residues = json.loads(filter_res_json)
            logger_local.info(f"Filter residues: {filter_residues}")
        else: 
            raise ValueError("filter_residues_dict not found in metadata.")

        # Identify SF Tyrosine Resids
        sf_tyrosine_resids = {}
        chains_to_analyze = list(filter_residues.keys())
        for chain_idx_loop, chain_segid_loop in enumerate(chains_to_analyze[:]): # Iterate over a copy
            resids = filter_residues[chain_segid_loop]
            if len(resids) == 5: 
                sf_tyrosine_resids[chain_segid_loop] = resids[3] # Tyr is index 3
            else: 
                logger_local.warning(f"Chain {chain_segid_loop}: Filter length != 5. Skipping Tyr analysis for this chain.")
                chains_to_analyze.pop(chain_idx_loop) # Remove from list of chains to process
        if not sf_tyrosine_resids: raise ValueError("Could not identify SF Tyrosine in any chain.")

        # Calculate Dihedrals
        time_points_list = []
        chi1_values = {c: np.full(frames_to_analyze, np.nan) for c in chains_to_analyze}
        chi2_values = {c: np.full(frames_to_analyze, np.nan) for c in chains_to_analyze}
        logger_local.info("Calculating SF Tyrosine dihedrals...")
        chi1_selections = {c: u.select_atoms(f"segid {c} and resid {sf_tyrosine_resids[c]} and name N CA CB CG") for c in chains_to_analyze}
        chi2_selections = {c: u.select_atoms(f"segid {c} and resid {sf_tyrosine_resids[c]} and name CA CB CG CD1") for c in chains_to_analyze}
        
        for ts_idx, ts in enumerate(tqdm(u.trajectory[start_frame:actual_end], desc="SF Tyr Dihedrals", unit="frame", disable=not logger_local.isEnabledFor(logging.INFO))):
            local_idx = ts_idx # Use enumerate for local index
            time_points_list.append(frames_to_time([ts.frame])[0]) # Use global frame for time calc
            for chain_segid_loop in chains_to_analyze:
                try:
                    chi1_sel = chi1_selections[chain_segid_loop]
                    chi2_sel = chi2_selections[chain_segid_loop]
                    if len(chi1_sel) == 4 and len(chi2_sel) == 4:
                        chi1_values[chain_segid_loop][local_idx] = calc_dihedral(*(chi1_sel[i].position for i in range(4)))
                        chi2_values[chain_segid_loop][local_idx] = calc_dihedral(*(chi2_sel[i].position for i in range(4)))
                    elif local_idx == 0: 
                        logger_local.warning(f"Atom count error for dihedrals in chain {chain_segid_loop} at first analyzed frame.")
                except Exception: 
                    pass # Keep NaN if error occurs
        time_points = np.array(time_points_list)

        # HMM Analysis for Rotamers
        logger_local.info("Starting HMM analysis for SF Tyrosine rotamers...")
        state_names_rotamer = TYR_HMM_STATE_ORDER
        state_centers_rotamer = np.array([TYR_HMM_STATES[name] for name in state_names_rotamer])
        n_states_rotamer = len(state_names_rotamer)
        log_pi_rotamer = np.full(n_states_rotamer, -np.log(n_states_rotamer))
        A_rotamer, logA_rotamer, _ = build_transition_matrix(n_states_rotamer, tyr_hmm_self_p, tyr_hmm_epsilon) # Adjacency not directly used by viterbi_2d
        
        all_final_dwells = {}
        all_hmm_paths = {}
        for chain_segid_loop in chains_to_analyze:
            logger_local.info(f"Running rotamer HMM for chain {chain_segid_loop}...")
            obs_chi1 = chi1_values[chain_segid_loop]
            obs_chi2 = chi2_values[chain_segid_loop]
            valid_mask = np.isfinite(obs_chi1) & np.isfinite(obs_chi2)
            chain_obs_2d = np.column_stack((obs_chi1[valid_mask], obs_chi2[valid_mask]))
            
            if len(chain_obs_2d) > 0:
                path_indices, _ = viterbi_2d_angles(chain_obs_2d, state_centers_rotamer, log_pi_rotamer, logA_rotamer, tyr_hmm_sigma)
                full_chain_path = np.full(frames_to_analyze, -1, dtype=int) # -1 for frames with no valid data
                valid_indices_original = np.where(valid_mask)[0] # Indices in the original obs_chi1/2 arrays
                
                if len(path_indices) == len(valid_indices_original):
                    full_chain_path[valid_indices_original] = path_indices
                else: 
                    logger_local.error(f"Rotamer HMM path length mismatch for chain {chain_segid_loop}. Filling with error code.")
                    full_chain_path.fill(-2) # -2 for error
                
                all_hmm_paths[chain_segid_loop] = full_chain_path
                chain_dwells_dict = segment_and_filter(full_chain_path, state_names_rotamer, time_points, tyr_hmm_flicker, f"Tyr_{chain_segid_loop}")
                chain_dwells = []
                for seg_dict in chain_dwells_dict:
                    start_f_local, end_f_local = seg_dict['start'], seg_dict['end']
                    if 0 <= start_f_local < frames_to_analyze and 0 <= end_f_local < frames_to_analyze:
                        # Convert local frame indices back to global for storage if needed, or just use local for consistency
                        orig_start_f = start_f_local + start_frame
                        orig_end_f = end_f_local + start_frame
                        chain_dwells.append({
                            'start_frame': orig_start_f, 
                            'end_frame': orig_end_f, 
                            'start_time': time_points[start_f_local], 
                            'end_time': time_points[end_f_local], 
                            'site_label': seg_dict['label']
                        })
                all_final_dwells[chain_segid_loop] = chain_dwells
            else: 
                logger_local.warning(f"No valid dihedral data for rotamer HMM chain {chain_segid_loop}.")
                all_hmm_paths[chain_segid_loop] = np.full(frames_to_analyze, -1, dtype=int)
                all_final_dwells[chain_segid_loop] = []

        dwell_path = _save_tyr_hmm_dwells(output_dir, run_dir, all_final_dwells, db_conn, module_name)
        path_save = _save_tyr_hmm_paths(output_dir, run_dir, time_points, all_hmm_paths, state_names_rotamer, db_conn, module_name)
        raw_dihedral_path = _save_raw_dihedral_data(output_dir, time_points, chi1_values, chi2_values, run_dir, db_conn, module_name)
        _calculate_and_store_tyr_hmm_stats(all_final_dwells, state_names_rotamer, db_conn, module_name, hmm_params_used)

        # --- Tyr-Thr Hydrogen Bond Analysis ---
        logger_local.info("Starting Tyr-Thr hydrogen bond analysis...")
        hbond_success = False
        final_formed_ref_hbond = default_formed_ref
        final_broken_ref_hbond = default_broken_ref
        
        try:
            logger_local.info("STEP 1: Calculating Tyr-Thr distances...")
            tyr_thr_distances = calc_tyr_thr_hbond_distance(
                u, filter_residues, tyr_thr_residue_offset, start_frame, actual_end
            )
            logger_local.info(f"STEP 1: Completed. Found data for {len(tyr_thr_distances)} pairs.")

            if not tyr_thr_distances:
                logger_local.warning("No Tyr-Thr pairs or distance calculation failed. Skipping H-bond stats.")
            else:
                if auto_detect_hbond_refs:
                    logger_local.info("STEP 2: Auto-detecting H-bond thresholds...")
                    detected_formed, detected_broken, _ = determine_tyr_thr_ref_distances(
                        tyr_thr_distances, default_formed_ref, default_broken_ref,
                        run_dir, db_conn, module_name
                    )
                    if detected_formed is not None and detected_broken is not None:
                        final_formed_ref_hbond, final_broken_ref_hbond = detected_formed, detected_broken
                    else:
                        logger_local.warning("H-bond auto-detection failed, using defaults.")
                
                logger_local.info(f"STEP 3: Using H-bond Thresholds: Formed<={final_formed_ref_hbond:.2f}, Broken>={final_broken_ref_hbond:.2f}")
                store_metric(db_conn, module_name, "TyrThr_RefDist_Formed_Used", final_formed_ref_hbond, "Å")
                store_metric(db_conn, module_name, "TyrThr_RefDist_Broken_Used", final_broken_ref_hbond, "Å")
                store_metric(db_conn, module_name, "Config_TYR_THR_RESIDUE_OFFSET", tyr_thr_residue_offset, "residues")

                logger_local.info("STEP 4: Determining H-bond states...")
                tyr_thr_states_data = determine_tyr_thr_hbond_states(
                    tyr_thr_distances, time_points, final_formed_ref_hbond, final_broken_ref_hbond, tolerance_frames_hbond
                )
                
                logger_local.info("STEP 5: Saving H-bond distance and state data...")
                dist_path, states_path, legacy_events_path = save_tyr_thr_hbond_data(
                    run_dir, output_dir, time_points, tyr_thr_distances, tyr_thr_states_data, db_conn, module_name
                )

                logger_local.info("STEP 6: Preparing debounced states for event building and stats...")
                debounced_states_df_per_pair = {}
                if states_path:
                    states_abs_path = os.path.join(run_dir, states_path)
                    if os.path.exists(states_abs_path):
                        try:
                            df_all_states = pd.read_csv(states_abs_path)
                            for col in df_all_states.columns:
                                if col.endswith('_debounced'): # As saved by save_tyr_thr_hbond_data
                                    pair_id = col.replace('_debounced', '')
                                    if 'Time (ns)' in df_all_states.columns and col in df_all_states.columns:
                                        debounced_states_df_per_pair[pair_id] = df_all_states[['Time (ns)', col]].rename(columns={col: 'State'})
                                    else:
                                        logger_local.warning(f"Missing 'Time (ns)' or '{col}' in {states_abs_path} for pair {pair_id}")
                            logger_local.info(f"Loaded debounced states for {len(debounced_states_df_per_pair)} pairs.")
                        except Exception as e_load_s:
                            logger_local.error(f"Failed to load states CSV {states_abs_path}: {e_load_s}")
                
                logger_local.info("STEP 7: Building H-bond events...")
                # Prepare df_hbond_long_debounced for build_events_from_states
                all_long_hbond_states_list = []
                if debounced_states_df_per_pair: # Check if we have any data
                    global_frame_indices_hbond = np.arange(start_frame, actual_end) # Global frames for this analysis slice
                    df_time_frame_map_hbond = pd.DataFrame({'Time (ns)': time_points, 'Frame': global_frame_indices_hbond})

                    for pair_id, df_pair_state in debounced_states_df_per_pair.items():
                        temp_df = df_pair_state.copy()
                        temp_df['chain'] = pair_id
                        temp_df['state'] = temp_df['State'].apply(lambda x: 'H-bond-formed' if x == 1 else ('H-bond-broken' if x == 0 else 'H-bond-uncertain'))
                        
                        # Merge with df_time_frame_map_hbond to add 'Frame' column
                        merged_df = pd.merge_asof(
                            temp_df.sort_values('Time (ns)'), 
                            df_time_frame_map_hbond.sort_values('Time (ns)'), 
                            on='Time (ns)', direction='nearest', tolerance=dt_ns/2
                        )
                        if 'Frame_y' in merged_df.columns: merged_df.rename(columns={'Frame_y': 'Frame'}, inplace=True)
                        elif 'Frame_x' in merged_df.columns and 'Frame' not in merged_df.columns: merged_df.rename(columns={'Frame_x': 'Frame'}, inplace=True)
                        if 'Frame' not in merged_df.columns: merged_df['Frame'] = merged_df.index + start_frame # Fallback

                        cols_for_builder = ['Frame', 'Time (ns)', 'chain', 'state']
                        if all(c in merged_df for c in cols_for_builder):
                            all_long_hbond_states_list.append(merged_df[cols_for_builder])
                        else:
                            logger_local.warning(f"Missing columns for event building for pair {pair_id}")
                
                hbond_events_df = None
                if all_long_hbond_states_list:
                    df_hbond_long_debounced = pd.concat(all_long_hbond_states_list).sort_values(['chain', 'Frame']).reset_index(drop=True)
                    if not df_hbond_long_debounced.empty:
                        hbond_events_df = build_events_from_states(df_hbond_long_debounced, dt_ns, state_col='state')
                        if hbond_events_df is not None and not hbond_events_df.empty:
                            hbond_proc_events_path = os.path.join(output_dir, "tyr_thr_hbond_processed_events.csv")
                            hbond_events_df.to_csv(hbond_proc_events_path, index=False, float_format='%.4f', na_rep='NaN')
                            register_product(db_conn, module_name, "csv", "data", os.path.relpath(hbond_proc_events_path, run_dir),
                                             subcategory="tyr_thr_hbond_processed_events",
                                             description="Processed Tyr-Thr H-bond events with durations.")
                            logger_local.info(f"Saved processed H-bond events to {hbond_proc_events_path}")
                        else:
                            logger_local.info("No processed H-bond events generated by build_events_from_states.")
                    else:
                        logger_local.info("Concatenated long H-bond states DataFrame is empty.")
                else:
                    logger_local.info("No data to build long H-bond states DataFrame for event building.")


                logger_local.info("STEP 8: Calculating and storing H-bond statistics...")
                total_analysis_duration_ns = time_points[-1] - time_points[0] if len(time_points) > 0 else 0.0
                if total_analysis_duration_ns <= 0:
                    logger_local.error("Invalid total analysis duration for H-bond stats. Skipping.")
                else:
                    calculate_and_store_tyr_thr_hbond_stats_from_events(
                        hbond_events_df if hbond_events_df is not None else pd.DataFrame(), # Pass empty DF if None
                        debounced_states_df_per_pair,
                        total_analysis_duration_ns,
                        db_conn,
                        module_name
                    )
                hbond_success = True # Mark H-bond part as successful if it reached here without critical errors

        except Exception as e_hbond:
            logger_local.error(f"Error during Tyr-Thr H-bond analysis: {e_hbond}", exc_info=True)
            # hbond_success remains False

        # Final status check
        if not dwell_path or not path_save or not raw_dihedral_path:
            results['error'] = "Failed to save essential SF Tyr HMM data."
            results['status'] = 'failed'
        elif not hbond_success: # HMM part succeeded but H-bond part had issues
            results['error'] = results.get('error', "") + " Tyr-Thr H-bond analysis encountered issues."
            results['status'] = 'success' # Module success, but H-bond part might be incomplete
            logger_local.warning("Tyrosine HMM analysis succeeded, but Tyr-Thr H-bond part had issues.")
        else:
            results['status'] = 'success'

    except Exception as e_main:
        error_msg = f"Error during Tyrosine analysis: {e_main}"
        logger_local.error(error_msg, exc_info=True)
        results['error'] = error_msg
        results['status'] = 'failed'

    # Finalize
    exec_time_module = time.time() - start_time_module # Use correct start time variable
    update_module_status(db_conn, module_name, results['status'], execution_time=exec_time_module, error_message=results['error'])
    logger_local.info(f"--- Tyrosine Analysis computation finished in {exec_time_module:.2f} seconds (Status: {results['status']}) ---")

    return results
