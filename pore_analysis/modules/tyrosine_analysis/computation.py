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
        # Import H-bond parameters AND DW Gate params reused here
        DW_GATE_TOLERANCE_FRAMES,
        DW_GATE_AUTO_DETECT_REFS, # <<< IMPORTED for Tyr-Thr threshold detection control
        TYR_THR_DEFAULT_FORMED_REF_DIST,
        TYR_THR_DEFAULT_BROKEN_REF_DIST,
        TYR_THR_RESIDUE_OFFSET # <<< IMPORTED for Thr residue selection
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
    # Import signal processing functions from DW gate analysis (for H-bond state debouncing)
    from pore_analysis.modules.dw_gate_analysis import signal_processing, event_building
    # Import KMeans check from DW gate signal processing
    from pore_analysis.modules.dw_gate_analysis.signal_processing import SKLEARN_AVAILABLE, KMeans
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

def calc_tyr_thr_hbond_distance(
    u: mda.Universe,
    filter_residues: Dict[str, List[int]],
    residue_offset: int, # <<< ADDED: Offset parameter
    start_frame: int = 0,
    end_frame: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Calculate the Tyr-Thr inter-subunit hydrogen bond distance.
    Selects Tyrosine based on filter index 3.
    Selects Threonine based on Tyrosine residue ID + residue_offset.
    Uses flexible atom name selection for Threonine ('OG1 OG').

    Args:
        u: MDAnalysis Universe
        filter_residues: Dictionary mapping chain IDs to filter residue IDs
        residue_offset: Offset from Tyr resid to find Thr resid (e.g., -6)
        start_frame: Starting frame index
        end_frame: Ending frame index (exclusive)

    Returns:
        Dictionary mapping pair IDs (e.g., 'PROA_PROB') to distance arrays
    """
    logger.info(f"Calculating Tyr-Thr H-bonds using Tyr[idx=3] and Thr[offset={residue_offset}]...")

    chain_ids = list(filter_residues.keys())
    if len(chain_ids) < 2:
        logger.warning(f"Insufficient chains ({len(chain_ids)}) for Tyr-Thr H-bond analysis")
        return {}

    if end_frame is None: end_frame = len(u.trajectory)
    frames_to_analyze = end_frame - start_frame

    chain_pairs = []
    for i, chain in enumerate(sorted(chain_ids)):
        next_chain = sorted(chain_ids)[(i + 1) % len(chain_ids)]
        chain_pairs.append((chain, next_chain))
    logger.info(f"Analyzing Tyr-Thr H-bonds between chain pairs: {chain_pairs}")

    tyr_thr_distances = {f"{s}_{t}": np.full(frames_to_analyze, np.nan) for s, t in chain_pairs}
    selections = {}

    for src_chain, tgt_chain in chain_pairs:
        pair_id = f"{src_chain}_{tgt_chain}"
        try:
            # Get Tyr resid from filter list (index 3)
            if len(filter_residues[src_chain]) != 5: raise ValueError(f"Src chain {src_chain} filter length != 5")
            tyr_resid = filter_residues[src_chain][3]

            # Calculate target Thr resid using offset
            target_thr_resid = tyr_resid + residue_offset
            logger.debug(f"Pair {pair_id}: Using Tyr={tyr_resid} (from filter[3]) and Target Thr={target_thr_resid} (offset={residue_offset}) on chain {tgt_chain}")

            # Select Tyr OH atom (always 'OH')
            tyr_oh_sel = u.select_atoms(f"segid {src_chain} and resid {tyr_resid} and name OH")
            # Select Thr OG/OG1 atom using the *calculated* target resid on the adjacent chain
            thr_og_sel = u.select_atoms(f"segid {tgt_chain} and resid {target_thr_resid} and name OG1 OG")

            # Check selection counts
            if len(tyr_oh_sel) != 1: logger.warning(f"Could not find exactly one Tyr OH atom for {pair_id}"); continue
            if len(thr_og_sel) != 1: logger.warning(f"Could not find exactly one Thr OG/OG1 atom for {pair_id} (Target Thr resid {target_thr_resid} on {tgt_chain}). Check offset/PSF name."); continue

            selections[pair_id] = (tyr_oh_sel, thr_og_sel)
            logger.debug(f"Selected atoms for {pair_id}: Tyr{tyr_resid} OH - Thr{target_thr_resid} {thr_og_sel.names[0]}")
        except (IndexError, KeyError, ValueError) as e:
            logger.warning(f"Could not set up Tyr-Thr selection for {pair_id}: {e}")
        except Exception as e_sel:
            logger.error(f"Unexpected error setting up selection for {pair_id}: {e_sel}", exc_info=True)

    if not selections:
        logger.error("No valid Tyr-Thr H-bond atom selections found. Cannot continue.")
        return {}

    for ts in tqdm(u.trajectory[start_frame:end_frame], desc="Tyr-Thr H-bond Distances", unit="frame", disable=not logger.isEnabledFor(logging.INFO)):
        local_idx = ts.frame - start_frame
        for pair_id, (tyr_sel, thr_sel) in selections.items():
            try:
                tyr_pos = tyr_sel.positions[0]; thr_pos = thr_sel.positions[0]
                dist = np.linalg.norm(tyr_pos - thr_pos)
                tyr_thr_distances[pair_id][local_idx] = dist
            except Exception as e: logger.debug(f"Error calculating distance for {pair_id} frame {ts.frame}: {e}")

    logger.info(f"Completed H-bond distance calculation for {len(selections)} pairs")
    return tyr_thr_distances


# <<< NEW FUNCTION: Determine Tyr-Thr Thresholds (Adapted from DW Gate) >>>
def _determine_tyr_thr_ref_distances(
    tyr_thr_distances: Dict[str, np.ndarray],
    default_formed_ref: float,
    default_broken_ref: float,
    run_dir: str,
    db_conn: sqlite3.Connection,
    module_name: str
) -> Tuple[float, float, Optional[Dict[str, Any]]]:
    """
    Determines formed/broken reference distances for Tyr-Thr H-bonds using KDE/KMeans.
    Falls back to defaults if needed. Saves KDE plot data to JSON.

    Args:
        tyr_thr_distances: Dictionary mapping pair IDs to distance arrays.
        default_formed_ref: Default reference distance for the formed state.
        default_broken_ref: Default reference distance for the broken state.
        run_dir: Path to the run directory.
        db_conn: Database connection.
        module_name: Name of the calling module.

    Returns:
        Tuple containing:
        - final_formed_ref: The determined or default formed reference distance.
        - final_broken_ref: The determined or default broken reference distance.
        - kde_plot_data: Dictionary with KDE data for potential plotting, or None if failed.
    """
    if not SKLEARN_AVAILABLE:
        logger.warning("Scikit-learn not found. Cannot perform KMeans clustering for Tyr-Thr reference distances. Using defaults.")
        return default_formed_ref, default_broken_ref, None

    all_distances = np.concatenate([d[np.isfinite(d)] for d in tyr_thr_distances.values() if d is not None])
    if len(all_distances) < 50: # Need sufficient data
        logger.warning(f"Insufficient valid distance points ({len(all_distances)}) for Tyr-Thr KDE/KMeans. Using defaults.")
        return default_formed_ref, default_broken_ref, None

    logger.info("Attempting to determine Tyr-Thr reference distances from data using KDE and KMeans...")
    kde_plot_data = {'all_distances': all_distances.tolist(), 'combined_kde': {}}

    # --- Combined KDE Analysis ---
    try:
        x_all, y_all, peaks_all, heights_all = signal_processing.find_kde_peaks(
            all_distances, # Pass combined distances
            bw_method='silverman', # Or other method
            peak_height_fraction=0.05, # Adjust thresholds as needed for H-bonds
            peak_distance=20,
            peak_prominence=0.03
        )
        kde_plot_data['combined_kde'] = {
            'x': x_all.tolist() if x_all is not None else None,
            'y': y_all.tolist() if y_all is not None else None,
            'peaks': peaks_all.tolist() if peaks_all is not None else [],
            'heights': heights_all.tolist() if heights_all is not None else []
        }
    except Exception as e_kde:
         logger.warning(f"Combined KDE failed for Tyr-Thr distances: {e_kde}. Using defaults.")
         return default_formed_ref, default_broken_ref, kde_plot_data # Return defaults but keep data

    # --- Determine Refs via KMeans on KDE Peaks ---
    final_formed_ref = default_formed_ref
    final_broken_ref = default_broken_ref
    kmeans_centers = None

    if peaks_all is not None and len(peaks_all) >= 2:
        try:
            kp = np.array(peaks_all).reshape(-1, 1)
            kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(kp)
            centers = sorted(kmeans.cluster_centers_.flatten())
            final_formed_ref, final_broken_ref = centers[0], centers[1]
            kmeans_centers = centers
            logger.info(f"KDE/KMeans derived Tyr-Thr references: Formed={final_formed_ref:.2f} Å, Broken={final_broken_ref:.2f} Å")
        except ValueError as ve:
            logger.warning(f"KMeans clustering failed for Tyr-Thr ({ve}). Using defaults.")
        except Exception as e:
            logger.warning(f"KMeans clustering failed unexpectedly for Tyr-Thr: {e}. Using defaults.")
    elif peaks_all is not None and len(peaks_all) == 1:
        logger.warning(f"Only one KDE peak found for Tyr-Thr ({peaks_all[0]:.2f} Å). Using defaults.")
    else: # No peaks found
        logger.warning("No KDE peaks found for Tyr-Thr distances. Using defaults.")

    # Save KDE plot data to JSON
    kde_plot_data['kmeans_centers'] = kmeans_centers
    kde_plot_data['final_formed_ref'] = final_formed_ref # Renamed key
    kde_plot_data['final_broken_ref'] = final_broken_ref # Renamed key
    output_dir = os.path.join(run_dir, module_name) # Save within module dir
    kde_data_json_path = os.path.join(output_dir, "tyr_thr_kde_plot_data.json")
    rel_kde_data_path = os.path.relpath(kde_data_json_path, run_dir)
    try:
        cleaned_kde_data = clean_json_data(kde_plot_data)
        with open(kde_data_json_path, 'w') as f_json: json.dump(cleaned_kde_data, f_json, indent=2)
        logger.info(f"Saved Tyr-Thr KDE plot data to {kde_data_json_path}")
        register_product(db_conn, module_name, "json", "data", rel_kde_data_path,
                         subcategory="tyr_thr_kde_plot_data",
                         description="Data for plotting Tyr-Thr H-bond KDE distribution.")
    except Exception as e_save_kde: logger.error(f"Failed to save Tyr-Thr KDE plot data JSON: {e_save_kde}")

    return final_formed_ref, final_broken_ref, kde_plot_data


# --- determine_tyr_thr_hbond_states, _save_tyr_thr_hbond_data, _calculate_and_store_tyr_thr_hbond_metrics remain unchanged ---
# They already accept the thresholds as arguments.

def determine_tyr_thr_hbond_states(
    distances: Dict[str, np.ndarray],
    time_points: np.ndarray,
    closed_ref_dist: float, # Now takes the determined/default value
    open_ref_dist: float,   # Now takes the determined/default value
    tolerance_frames: int = DW_GATE_TOLERANCE_FRAMES
) -> Dict[str, Dict]:
    """Determine the states of the Tyr-Thr hydrogen bonds (formed/broken)."""
    logger.info(f"Determining Tyr-Thr H-bond states using thresholds: Formed<={closed_ref_dist:.2f}, Broken>={open_ref_dist:.2f}")
    state_results = {}
    for pair_id, dist_array in distances.items():
        raw_states = np.full_like(dist_array, -1, dtype=int)
        raw_states[dist_array <= closed_ref_dist] = 0
        raw_states[dist_array >= open_ref_dist] = 1
        debounced_states = signal_processing.debounce_binary_signal(raw_states, tolerance_frames, fill_gaps=True)
        events = []
        if len(debounced_states) > 0:
            events = event_building.extract_state_events(debounced_states, time_points, open_label="H-bond-broken", closed_label="H-bond-formed")
        state_results[pair_id] = {"raw_states": raw_states, "debounced_states": debounced_states, "events": events}
    logger.info(f"Completed state determination for {len(distances)} H-bond pairs")
    return state_results

def _save_tyr_thr_hbond_data(
    run_dir: str, output_dir: str, time_points: np.ndarray,
    distances: Dict[str, np.ndarray], states: Dict[str, Dict],
    db_conn: sqlite3.Connection, module_name: str
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Save Tyr-Thr hydrogen bond data to files and register."""
    if not distances or not states: logger.warning("No Tyr-Thr H-bond data to save."); return None, None, None
    os.makedirs(output_dir, exist_ok=True); saved_paths = {'distances': None, 'states': None, 'events': None}
    try: # Distances
        dist_df = pd.DataFrame({'Time (ns)': time_points}); [dist_df.update({pair_id: dist_array}) for pair_id, dist_array in distances.items()]
        dist_file = os.path.join(output_dir, "tyr_thr_hbond_distances.csv"); dist_df.to_csv(dist_file, index=False, float_format='%.3f')
        dist_path = os.path.relpath(dist_file, run_dir); register_product(db_conn, module_name, "csv", "data", dist_path, subcategory="tyr_thr_hbond_distances", description="Time series of Tyr445-Thr439 hydrogen bond distances"); logger.info(f"Saved Tyr-Thr H-bond distances to {dist_file}"); saved_paths['distances'] = dist_path
    except Exception as e: logger.error(f"Failed to save H-bond distances: {e}")
    try: # States
        states_df = pd.DataFrame({'Time (ns)': time_points}); [states_df.update({f"{pair_id}_raw": state_data["raw_states"], f"{pair_id}_debounced": state_data["debounced_states"]}) for pair_id, state_data in states.items()]
        states_file = os.path.join(output_dir, "tyr_thr_hbond_states.csv"); states_df.to_csv(states_file, index=False)
        states_path = os.path.relpath(states_file, run_dir); register_product(db_conn, module_name, "csv", "data", states_path, subcategory="tyr_thr_hbond_states", description="Time series of Tyr445-Thr439 hydrogen bond states"); logger.info(f"Saved Tyr-Thr H-bond states to {states_file}"); saved_paths['states'] = states_path
    except Exception as e: logger.error(f"Failed to save H-bond states: {e}")
    try: # Events
        all_events = [{"Pair": pair_id, "Start Time (ns)": event["start_time"], "End Time (ns)": event["end_time"], "State": event["state"], "Duration (ns)": event["end_time"] - event["start_time"]} for pair_id, state_data in states.items() for event in state_data["events"]]
        if all_events:
            events_df = pd.DataFrame(all_events); events_file = os.path.join(output_dir, "tyr_thr_hbond_events.csv"); events_df.to_csv(events_file, index=False, float_format='%.4f')
            events_path = os.path.relpath(events_file, run_dir); register_product(db_conn, module_name, "csv", "data", events_path, subcategory="tyr_thr_hbond_events", description="Tyr445-Thr439 hydrogen bond state events"); logger.info(f"Saved Tyr-Thr H-bond events to {events_file}"); saved_paths['events'] = events_path
    except Exception as e: logger.error(f"Failed to save H-bond events: {e}")
    return saved_paths['distances'], saved_paths['states'], saved_paths['events']

def _calculate_and_store_tyr_thr_hbond_metrics(
    states: Dict[str, Dict], db_conn: sqlite3.Connection, module_name: str
) -> None:
    """Calculate and store metrics for Tyr-Thr hydrogen bonds."""
    if not states: logger.warning("No Tyr-Thr H-bond state data for metrics."); return
    logger.info("Calculating and storing Tyr-Thr H-bond metrics...")
    for pair_id, state_data in states.items():
        if "debounced_states" not in state_data or len(state_data["debounced_states"]) == 0: logger.warning(f"No valid state data for {pair_id}, skipping metrics"); continue
        formed_events = [e for e in state_data["events"] if e["state"] == "H-bond-formed"]; broken_events = [e for e in state_data["events"] if e["state"] == "H-bond-broken"]
        formed_durations = [e["end_time"] - e["start_time"] for e in formed_events]; broken_durations = [e["end_time"] - e["start_time"] for e in broken_events]
        total_frames = len(state_data["debounced_states"]); formed_count = np.sum(state_data["debounced_states"] == 0); broken_count = np.sum(state_data["debounced_states"] == 1)
        formed_fraction = (formed_count / total_frames * 100) if total_frames > 0 else np.nan; broken_fraction = (broken_count / total_frames * 100) if total_frames > 0 else np.nan
        mean_formed = np.mean(formed_durations) if formed_durations else np.nan; median_formed = np.median(formed_durations) if formed_durations else np.nan
        mean_broken = np.mean(broken_durations) if broken_durations else np.nan; median_broken = np.median(broken_durations) if broken_durations else np.nan
        if pd.notna(len(formed_events)): store_metric(db_conn, module_name, f"TyrThr_{pair_id}_formed_Count", len(formed_events), "count", "Count of H-bond formed events")
        if pd.notna(len(broken_events)): store_metric(db_conn, module_name, f"TyrThr_{pair_id}_broken_Count", len(broken_events), "count", "Count of H-bond broken events")
        if pd.notna(mean_formed): store_metric(db_conn, module_name, f"TyrThr_{pair_id}_formed_Mean_ns", mean_formed, "ns", "Mean duration of H-bond formed state")
        if pd.notna(median_formed): store_metric(db_conn, module_name, f"TyrThr_{pair_id}_formed_Median_ns", median_formed, "ns", "Median duration of H-bond formed state")
        if pd.notna(mean_broken): store_metric(db_conn, module_name, f"TyrThr_{pair_id}_broken_Mean_ns", mean_broken, "ns", "Mean duration of H-bond broken state")
        if pd.notna(median_broken): store_metric(db_conn, module_name, f"TyrThr_{pair_id}_broken_Median_ns", median_broken, "ns", "Median duration of H-bond broken state")
        if pd.notna(formed_fraction): store_metric(db_conn, module_name, f"TyrThr_{pair_id}_Formed_Fraction", formed_fraction, "%", "Percentage of time H-bond was formed")
        if pd.notna(broken_fraction): store_metric(db_conn, module_name, f"TyrThr_{pair_id}_Broken_Fraction", broken_fraction, "%", "Percentage of time H-bond was broken")
    logger.info("Completed H-bond metric calculation and storage")


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
    start_time = time.time()
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

        tyr_thr_residue_offset = get_param('TYR_THR_RESIDUE_OFFSET', -6, int) # Get offset
        auto_detect_hbond_refs = get_param('DW_GATE_AUTO_DETECT_REFS', True, bool) # Get auto-detect flag
        default_formed_ref = get_param('TYR_THR_DEFAULT_FORMED_REF_DIST', 3.5, float)
        default_broken_ref = get_param('TYR_THR_DEFAULT_BROKEN_REF_DIST', 4.5, float)
        tolerance_frames = get_param('DW_GATE_TOLERANCE_FRAMES', 5, int)
        # HMM Params (remain unchanged)
        tyr_hmm_sigma = get_param('TYR_HMM_EMISSION_SIGMA', TYR_HMM_EMISSION_SIGMA, float)
        tyr_hmm_self_p = get_param('TYR_HMM_SELF_TRANSITION_P', TYR_HMM_SELF_TRANSITION_P, float)
        tyr_hmm_epsilon = get_param('TYR_HMM_EPSILON', TYR_HMM_EPSILON, float)
        tyr_hmm_flicker = get_param('TYR_HMM_FLICKER_NS', TYR_HMM_FLICKER_NS, float)
        if tyr_hmm_sigma <= 0: logger.warning(f"Invalid sigma {tyr_hmm_sigma}, using default."); tyr_hmm_sigma = TYR_HMM_EMISSION_SIGMA

        hmm_params_used = {'emission_sigma': tyr_hmm_sigma, 'self_transition_p': tyr_hmm_self_p, 'epsilon': tyr_hmm_epsilon, 'flicker_ns': tyr_hmm_flicker}

        # Load Universe & Validate Frames (Code unchanged)
        # Load Universe if not provided
        if universe is not None: u = universe; logger_local.info("Using provided Universe object.")
        else:
            if psf_file is None or dcd_file is None: raise ValueError("PSF/DCD must be specified if universe not provided.")
            logger_local.info(f"Loading topology: {psf_file}, trajectory: {dcd_file}")
            u = mda.Universe(psf_file, dcd_file)
            logger_local.info("Universe loaded from files.")
        # Validate frame range
        n_frames_total = len(u.trajectory)
        if start_frame < 0 or start_frame >= n_frames_total: raise ValueError(f"Invalid start_frame: {start_frame}")
        actual_end = end_frame if end_frame is not None else n_frames_total
        if actual_end <= start_frame or actual_end > n_frames_total: raise ValueError(f"Invalid end_frame: {actual_end}")
        frames_to_analyze = actual_end - start_frame
        logger_local.info(f"Using frame range: {start_frame} to {actual_end} ({frames_to_analyze} frames)")
        store_metric(db_conn, module_name, "start_frame", start_frame, "frame"); store_metric(db_conn, module_name, "end_frame", actual_end, "frame"); store_metric(db_conn, module_name, "frames_analyzed", frames_to_analyze, "frame")
        if n_frames_total < 2: raise ValueError("Trajectory has < 2 frames.")

        # Retrieve Filter Residues (Code unchanged)
        logger_local.info("Retrieving filter residue definitions from database...")
        filter_res_json = get_simulation_metadata(db_conn, 'filter_residues_dict')
        if filter_res_json: filter_residues = json.loads(filter_res_json); logger_local.info(f"Filter residues: {filter_residues}")
        else: raise ValueError("filter_residues_dict not found in metadata.")

        # Identify SF Tyrosine Resids (Code unchanged)
        sf_tyrosine_resids = {}
        chains_to_analyze = list(filter_residues.keys())
        for chain in chains_to_analyze[:]:
            resids = filter_residues[chain]
            if len(resids) == 5: sf_tyrosine_resids[chain] = resids[3] # Tyr is index 3
            else: logger_local.warning(f"Chain {chain}: Filter length != 5. Skipping Tyr analysis."); chains_to_analyze.remove(chain)
        if not sf_tyrosine_resids: raise ValueError("Could not identify SF Tyrosine in any chain.")

        # Calculate Dihedrals (Code unchanged)
        time_points_list = []; chi1_values = {c: np.full(frames_to_analyze, np.nan) for c in chains_to_analyze}; chi2_values = {c: np.full(frames_to_analyze, np.nan) for c in chains_to_analyze}
        logger_local.info("Calculating SF Tyrosine dihedrals..."); chi1_selections = {c: u.select_atoms(f"segid {c} and resid {sf_tyrosine_resids[c]} and name N CA CB CG") for c in chains_to_analyze}; chi2_selections = {c: u.select_atoms(f"segid {c} and resid {sf_tyrosine_resids[c]} and name CA CB CG CD1") for c in chains_to_analyze}
        for ts in tqdm(u.trajectory[start_frame:actual_end], desc="SF Tyr Dihedrals", unit="frame", disable=not logger_local.isEnabledFor(logging.INFO)):
            local_idx = ts.frame - start_frame; time_points_list.append(frames_to_time([ts.frame])[0])
            for chain in chains_to_analyze:
                try:
                    chi1_sel = chi1_selections[chain]; chi2_sel = chi2_selections[chain]
                    if len(chi1_sel) == 4 and len(chi2_sel) == 4:
                        chi1_values[chain][local_idx] = calc_dihedral(*(chi1_sel[i].position for i in range(4))); chi2_values[chain][local_idx] = calc_dihedral(*(chi2_sel[i].position for i in range(4)))
                    elif local_idx == 0: logger_local.warning(f"Atom count error chain {chain}")
                except Exception: pass # Keep NaN
        time_points = np.array(time_points_list)

        # HMM Analysis for Rotamers (Code unchanged)
        logger_local.info("Starting HMM analysis for SF Tyrosine..."); state_names = TYR_HMM_STATE_ORDER; state_centers = np.array([TYR_HMM_STATES[name] for name in state_names]); n_states = len(state_names); log_pi = np.full(n_states, -np.log(n_states))
        A = np.full((n_states, n_states), tyr_hmm_epsilon); np.fill_diagonal(A, tyr_hmm_self_p); A = A / A.sum(axis=1, keepdims=True); logA = np.log(A); adjacency = {i: list(range(n_states)) for i in range(n_states)}
        all_final_dwells = {}; all_hmm_paths = {}
        for chain in chains_to_analyze:
            logger_local.info(f"Running HMM for chain {chain}..."); obs_chi1 = chi1_values[chain]; obs_chi2 = chi2_values[chain]
            valid_mask = np.isfinite(obs_chi1) & np.isfinite(obs_chi2); chain_obs_2d = np.column_stack((obs_chi1[valid_mask], obs_chi2[valid_mask]))
            if len(chain_obs_2d) > 0:
                path_indices, _ = viterbi_2d_angles(chain_obs_2d, state_centers, log_pi, logA, tyr_hmm_sigma)
                full_chain_path = np.full(frames_to_analyze, -1, dtype=int); valid_indices = np.where(valid_mask)[0]
                if len(path_indices) == len(valid_indices): full_chain_path[valid_indices] = path_indices
                else: logger.error(f"Path length mismatch chain {chain}"); full_chain_path.fill(-2)
                all_hmm_paths[chain] = full_chain_path
                chain_dwells_dict = segment_and_filter(full_chain_path, state_names, time_points, tyr_hmm_flicker, f"Tyr_{chain}")
                chain_dwells = []
                for seg_dict in chain_dwells_dict:
                    start_f, end_f = seg_dict['start'], seg_dict['end']
                    if 0 <= start_f < frames_to_analyze and 0 <= end_f < frames_to_analyze:
                        orig_start_f = start_f + start_frame; orig_end_f = end_f + start_frame
                        chain_dwells.append({'start_frame': orig_start_f, 'end_frame': orig_end_f, 'start_time': time_points[start_f], 'end_time': time_points[end_f], 'site_label': seg_dict['label']})
                all_final_dwells[chain] = chain_dwells
            else: logger_local.warning(f"No valid data chain {chain}"); all_hmm_paths[chain] = np.full(frames_to_analyze, -1, dtype=int); all_final_dwells[chain] = []

        # Save HMM & Raw Dihedral Data (Code unchanged)
        dwell_path = _save_tyr_hmm_dwells(output_dir, run_dir, all_final_dwells, db_conn, module_name)
        path_save = _save_tyr_hmm_paths(output_dir, run_dir, time_points, all_hmm_paths, state_names, db_conn, module_name)
        raw_dihedral_path = _save_raw_dihedral_data(output_dir, time_points, chi1_values, chi2_values, run_dir, db_conn, module_name)
        _calculate_and_store_tyr_hmm_stats(all_final_dwells, state_names, db_conn, module_name, hmm_params_used)


        # --- Tyr-Thr Hydrogen Bond Analysis (UPDATED PART) ---
        logger_local.info("Starting Tyr-Thr hydrogen bond analysis...")
        hbond_success = False
        final_formed_ref = default_formed_ref # Initialize with defaults
        final_broken_ref = default_broken_ref
        kde_plot_data_dict = None # Initialize

        try:
            # --- Calculate Distances using Offset ---
            tyr_thr_distances = calc_tyr_thr_hbond_distance(
                u, filter_residues, tyr_thr_residue_offset, # Use offset
                start_frame, actual_end
            )

            if not tyr_thr_distances:
                logger_local.warning("No valid Tyr-Thr pairs or distance calculation failed.")
            else:
                # --- Determine Thresholds (Auto or Default) ---
                if auto_detect_hbond_refs:
                    logger_local.info("Attempting auto-detection of Tyr-Thr H-bond thresholds...")
                    detected_formed, detected_broken, kde_plot_data_dict = _determine_tyr_thr_ref_distances(
                        tyr_thr_distances, default_formed_ref, default_broken_ref,
                        run_dir, db_conn, module_name
                    )
                    # Use detected only if successful, otherwise keep defaults
                    if detected_formed is not None and detected_broken is not None:
                         final_formed_ref = detected_formed
                         final_broken_ref = detected_broken
                    else:
                         logger_local.warning("Auto-detection failed, using default Tyr-Thr thresholds.")
                else:
                    logger_local.info("Using default Tyr-Thr H-bond thresholds.")
                    # final_formed_ref, final_broken_ref already set to defaults

                logger_local.info(f"Using Tyr-Thr Thresholds: Formed <= {final_formed_ref:.2f} Å, Broken >= {final_broken_ref:.2f} Å")
                # Store the thresholds actually used as metrics
                store_metric(db_conn, module_name, "TyrThr_RefDist_Formed_Used", final_formed_ref, "Å", "Final Formed Reference Distance Used")
                store_metric(db_conn, module_name, "TyrThr_RefDist_Broken_Used", final_broken_ref, "Å", "Final Broken Reference Distance Used")
                store_metric(db_conn, module_name, "Config_TYR_THR_RESIDUE_OFFSET", tyr_thr_residue_offset, "residues", "Tyr->Thr offset used")


                # --- Determine States using final thresholds ---
                tyr_thr_states = determine_tyr_thr_hbond_states(
                    tyr_thr_distances, time_points,
                    final_formed_ref, final_broken_ref, # Pass final thresholds
                    tolerance_frames # Reuse tolerance
                )

                # Save H-bond data to files
                dist_path, states_path, events_path = _save_tyr_thr_hbond_data(
                    run_dir, output_dir, time_points, tyr_thr_distances, tyr_thr_states,
                    db_conn, module_name)

                # Calculate and store H-bond metrics
                _calculate_and_store_tyr_thr_hbond_metrics(
                    tyr_thr_states, db_conn, module_name)

                hbond_success = True # Mark H-bond part as successful
                logger_local.info("Completed Tyr-Thr hydrogen bond analysis")

        except Exception as e_hbond:
            logger_local.error(f"Error during Tyr-Thr H-bond analysis: {e_hbond}", exc_info=True)
            logger_local.warning("Continuing analysis despite H-bond analysis failure")
            # hbond_success remains False

        # Check if essential HMM files were saved
        if not dwell_path or not path_save or not raw_dihedral_path:
             results['error'] = "Failed to save one or more essential Tyr HMM/raw data files."
             results['status'] = 'failed'
             logger_local.error(results['error'])
        else:
             # Base success on HMM part, H-bond part is optional for overall module success status
             results['status'] = 'success'

    except Exception as e_main:
        error_msg = f"Error during Tyrosine analysis: {e_main}"
        logger_local.error(error_msg, exc_info=True)
        results['error'] = error_msg
        results['status'] = 'failed' # Ensure status is failed

    # Finalize
    exec_time = time.time() - start_time
    update_module_status(db_conn, module_name, results['status'], execution_time=exec_time, error_message=results['error'])
    logger_local.info(f"--- Tyrosine Analysis computation finished in {exec_time:.2f} seconds (Status: {results['status']}) ---")

    return results
