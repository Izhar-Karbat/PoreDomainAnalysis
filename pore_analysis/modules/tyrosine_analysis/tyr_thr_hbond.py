"""
Functions for analyzing Tyr-Thr hydrogen bonds in the tyrosine module.
This module handles distance calculations, threshold determination, state classification,
event building, and statistics calculation for Tyr-Thr hydrogen bonds.
"""

import os
import logging
import numpy as np
import pandas as pd
import sqlite3
import json
from typing import Dict, Optional, Tuple, List, Any
import MDAnalysis as mda
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks

# Import from core modules
from pore_analysis.core.utils import clean_json_data
from pore_analysis.core.config import (
    FRAMES_PER_NS,
    DW_GATE_TOLERANCE_FRAMES,
    DW_GATE_AUTO_DETECT_REFS,
    TYR_THR_DEFAULT_FORMED_REF_DIST,
    TYR_THR_DEFAULT_BROKEN_REF_DIST,
    TYR_THR_RESIDUE_OFFSET
)
from pore_analysis.core.database import (
    register_product, store_metric
)

# Import event builder from DW-gate module
from pore_analysis.modules.dw_gate_analysis.event_building import build_events_from_states
from pore_analysis.modules.dw_gate_analysis.signal_processing import (
    SKLEARN_AVAILABLE, KMeans, _rle_debounce_single_series
)

logger = logging.getLogger(__name__)

def calc_tyr_thr_hbond_distance(
    u: mda.Universe,
    filter_residues: Dict[str, List[int]],
    residue_offset: int, 
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    step: int = 1
) -> Dict[str, np.ndarray]:
    """
    Calculate the Tyr-Thr inter-subunit hydrogen bond distance.
    Selects Tyrosine from the 'next' chain in a cyclic manner and Threonine from the 'current' chain.
    Tyrosine is selected based on filter_residues (index 3).
    Threonine is selected based on the Tyrosine's residue ID + residue_offset on its respective chain.
    Uses flexible atom name selection for Threonine ('OG1 OG').

    Args:
        u: MDAnalysis Universe.
        filter_residues: Dictionary mapping chain IDs to filter residue IDs.
                         The Tyrosine resid is taken from filter_residues[tyr_chain_id][3].
        residue_offset: Offset from Tyr resid to find Thr resid (e.g., -6).
                        Thr resid = Tyr resid + residue_offset.
        start_frame: Starting frame index for analysis.
        end_frame: Ending frame index (exclusive) for analysis.
        step: Step for trajectory iteration.

    Returns:
        Dictionary mapping pair IDs (e.g., 'TYR_CHAIN_THR_CHAIN') to distance arrays.
        Example: if Tyr from PROB interacts with Thr from PROA, key is 'PROB_PROA'.
    """
    logger.info(f"Calculating Tyr-Thr H-bonds using Tyr[idx=3 from filter] and Thr[Tyr_resid + offset={residue_offset}]...")

    chain_ids = list(filter_residues.keys())
    if len(chain_ids) < 2:
        logger.warning(f"Insufficient chains ({len(chain_ids)}) for Tyr-Thr H-bond analysis. Need at least 2.")
        return {}

    if end_frame is None:
        end_frame = len(u.trajectory)
    
    # Ensure start_frame, end_frame, and step are valid
    if not (0 <= start_frame < end_frame <= len(u.trajectory)):
        logger.error(f"Invalid frame range: start={start_frame}, end={end_frame}, total_frames={len(u.trajectory)}")
        return {}
    if step <= 0:
        logger.error(f"Invalid step value: {step}. Must be > 0.")
        return {}

    frames_to_analyze = len(range(start_frame, end_frame, step))
    if frames_to_analyze == 0:
        logger.warning("No frames to analyze with the given start, end, and step parameters.")
        return {}

    chain_pairs = []
    s_chain_ids = sorted(list(chain_ids)) # Ensure consistent order

    # Corrected pairing: Tyr from 'next_chain_for_tyr', Thr from 'current_chain_for_thr'
    # The interaction is Tyr(chain B) - Thr(chain A), Tyr(chain C) - Thr(chain B), etc.
    for i, current_chain_for_thr in enumerate(s_chain_ids):
        # The 'next' chain in cyclic order will provide the Tyrosine
        next_chain_for_tyr = s_chain_ids[(i + 1) % len(s_chain_ids)]
        
        # Store as (tyrosine_chain_id, threonine_chain_id)
        chain_pairs.append((next_chain_for_tyr, current_chain_for_thr)) 
        
    logger.info(f"Analyzing Tyr-Thr H-bonds between (Tyr_chain, Thr_chain) pairs: {chain_pairs}")

    # Initialize distances dictionary. Keys will be like 'TYRCHAIN_THRCHAIN'
    tyr_thr_distances = {f"{tyr_c}_{thr_c}": np.full(frames_to_analyze, np.nan) for tyr_c, thr_c in chain_pairs}
    selections = {} # To store (tyr_atomgroup, thr_atomgroup)

    # Setup selections for each pair
    # tyr_chain_id is the chain providing Tyrosine
    # thr_chain_id is the chain providing Threonine
    for tyr_chain_id, thr_chain_id in chain_pairs:
        pair_id = f"{tyr_chain_id}_{thr_chain_id}" # e.g., PROB_PROA if Tyr is on PROB, Thr on PROA
        try:
            # Get Tyr resid from filter list (index 3) on the TYROSINE chain
            if tyr_chain_id not in filter_residues or len(filter_residues[tyr_chain_id]) <= 3:
                logger.warning(f"Filter residues for Tyrosine chain {tyr_chain_id} not found or too short.")
                continue
            tyr_resid = filter_residues[tyr_chain_id][3]

            # Calculate target Thr resid using offset from the Tyr resid
            target_thr_resid = tyr_resid + residue_offset
            logger.debug(f"Pair {pair_id}: Using Tyr={tyr_resid} (from filter[3] on chain {tyr_chain_id}) "
                         f"and Target Thr={target_thr_resid} (offset={residue_offset} from Tyr_{tyr_resid}) on chain {thr_chain_id}")

            # Select Tyr OH atom from the TYROSINE chain
            tyr_oh_sel = u.select_atoms(f"segid {tyr_chain_id} and resid {tyr_resid} and name OH")
            
            # Select Thr OG/OG1 atom using the *calculated* target_thr_resid on the THREONINE chain
            thr_og_sel = u.select_atoms(f"segid {thr_chain_id} and resid {target_thr_resid} and name OG1 OG")

            # Check selection counts
            if len(tyr_oh_sel) != 1:
                logger.warning(f"Could not find exactly one Tyr OH atom for {pair_id} (Tyr {tyr_resid} on chain {tyr_chain_id}).")
                continue
            if len(thr_og_sel) != 1:
                logger.warning(f"Could not find exactly one Thr OG/OG1 atom for {pair_id} "
                               f"(Target Thr resid {target_thr_resid} on chain {thr_chain_id}). Check offset/PSF naming.")
                continue

            selections[pair_id] = (tyr_oh_sel, thr_og_sel)
            logger.debug(f"Selected atoms for {pair_id}: Tyr{tyr_resid} (chain {tyr_chain_id}) OH - Thr{target_thr_resid} (chain {thr_chain_id}) {thr_og_sel.names[0]}")
        
        except (IndexError, KeyError, ValueError) as e:
            logger.warning(f"Could not set up Tyr-Thr selection for {pair_id}: {e}")
        except Exception as e_sel:
            logger.error(f"Unexpected error setting up selection for {pair_id}: {e_sel}", exc_info=True)

    if not selections:
        logger.error("No valid Tyr-Thr H-bond atom selections found after setup. Cannot continue distance calculation.")
        return {}

    # Iterate through trajectory and calculate distances
    # Use trajectory slicing with step
    from tqdm import tqdm
    for frame_idx, ts in enumerate(tqdm(u.trajectory[start_frame:end_frame:step], 
                                     desc="Tyr-Thr H-bond Distances", 
                                     unit="frame", 
                                     disable=not logger.isEnabledFor(logging.INFO),
                                     total=frames_to_analyze)):
        # frame_idx is the local index for storing in the np.ndarray
        for pair_id, (tyr_sel, thr_sel) in selections.items():
            try:
                # Positions are updated for each frame by MDAnalysis
                tyr_pos = tyr_sel.positions[0] 
                thr_pos = thr_sel.positions[0]
                dist = np.linalg.norm(tyr_pos - thr_pos)
                tyr_thr_distances[pair_id][frame_idx] = dist
            except Exception as e:
                # Log specific error for this frame and pair, but continue
                logger.debug(f"Error calculating distance for {pair_id} at trajectory frame {ts.frame} (local_idx {frame_idx}): {e}")
                # tyr_thr_distances[pair_id][frame_idx] will remain np.nan

    logger.info(f"Completed H-bond distance calculation for {len(selections)} pairs over {frames_to_analyze} frames.")
    return tyr_thr_distances


def determine_tyr_thr_ref_distances(
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
        # Import signal_processing find_kde_peaks function
        from pore_analysis.modules.dw_gate_analysis.signal_processing import find_kde_peaks
        
        x_all, y_all, peaks_all, heights_all = find_kde_peaks(
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
    kde_plot_data['final_formed_ref'] = final_formed_ref
    kde_plot_data['final_broken_ref'] = final_broken_ref
    output_dir = os.path.join(run_dir, module_name) # Save within module dir
    os.makedirs(output_dir, exist_ok=True)
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


def determine_tyr_thr_hbond_states(
    hbond_distances: Dict[str, np.ndarray],
    time_points: np.ndarray,
    formed_threshold: float,
    broken_threshold: float,
    tolerance_frames: int = 3
) -> Dict[str, Dict[str, Any]]:
    """
    Determines H-bond states (formed/broken) based on distance thresholds and applies debouncing.

    Args:
        hbond_distances: Dictionary mapping pair IDs to distance arrays.
        time_points: Array of time points corresponding to the distances.
        formed_threshold: Distance (Å) at or below which a bond is considered formed.
        broken_threshold: Distance (Å) at or above which a bond is considered broken.
        tolerance_frames: Minimum number of consecutive frames a state must persist.

    Returns:
        Dictionary mapping pair IDs to their state information (raw signal, debounced signal, events).
    """
    logger.info(f"Determining Tyr-Thr H-bond states using thresholds: Formed<={formed_threshold:.2f}, Broken>={broken_threshold:.2f}")
    all_pair_states_data = {}

    if not hbond_distances:
        logger.warning("H-bond distances dictionary is empty. Cannot determine states.")
        return {}

    for pair_id, distances in hbond_distances.items():
        if distances is None or len(distances) == 0:
            logger.warning(f"No distance data for pair {pair_id}. Skipping state determination.")
            continue
        
        # Initialize binary signal: 1 for formed, 0 for broken
        binary_signal = np.zeros_like(distances, dtype=int)

        # Apply thresholds
        binary_signal[distances <= formed_threshold] = 1  # Formed
        binary_signal[distances >= broken_threshold] = 0  # Broken

        # Handle ambiguous region (between formed_threshold and broken_threshold)
        # Maintain previous state if in ambiguous region. Default to 0 (broken) if no previous state.
        for i in range(len(distances)):
            if formed_threshold < distances[i] < broken_threshold:
                binary_signal[i] = binary_signal[i-1] if i > 0 else 0
        
        raw_signal_df = pd.DataFrame({'Time': time_points[:len(binary_signal)], 'State': binary_signal})

        # Apply debouncing to remove short-lived states
        if tolerance_frames > 0 and len(binary_signal) > tolerance_frames:
            logger.debug(f"Applying debouncing for pair {pair_id} with tolerance {tolerance_frames} frames.")
            try:
                # Use _rle_debounce_single_series from signal_processing
                debounced_signal_values = _rle_debounce_single_series(binary_signal.tolist(), tolerance_frames)
                debounced_signal_values = np.array(debounced_signal_values) # Convert back to numpy array
            except Exception as e_debounce:
                logger.error(f"Error during debouncing for pair {pair_id}: {e_debounce}", exc_info=True)
                debounced_signal_values = binary_signal # Fallback to raw signal
        else:
            logger.debug(f"Not applying debouncing for pair {pair_id} (tolerance_frames={tolerance_frames}, signal_len={len(binary_signal)}).")
            debounced_signal_values = binary_signal
            
        debounced_signal_df = pd.DataFrame({'Time': time_points[:len(debounced_signal_values)], 'State': debounced_signal_values})

        # Detect events (transitions between states) from the debounced signal
        events = []
        if len(debounced_signal_values) > 1:
            transitions = np.diff(debounced_signal_values)
            event_indices = np.where(transitions != 0)[0]

            for idx in event_indices:
                event_time = time_points[idx + 1] # Time of state change
                from_state = debounced_signal_values[idx]
                to_state = debounced_signal_values[idx + 1]
                event_type = "Formed" if to_state == 1 else "Broken"
                
                # Determine duration of the previous state
                start_of_prev_state_idx = idx
                while start_of_prev_state_idx > 0 and debounced_signal_values[start_of_prev_state_idx - 1] == from_state:
                    start_of_prev_state_idx -= 1
                
                prev_state_start_time = time_points[start_of_prev_state_idx]
                duration = event_time - prev_state_start_time # Duration of the state ENDING at event_time
                                
                events.append({
                    'EventTime': event_time,
                    'FromState': int(from_state),
                    'ToState': int(to_state),
                    'EventType': event_type,
                    'PreviousStateStartTime': prev_state_start_time,
                    'PreviousStateDuration': duration
                })
        events_df = pd.DataFrame(events)

        all_pair_states_data[pair_id] = {
            'raw_signal': raw_signal_df,
            'debounced_signal': debounced_signal_df,
            'events': events_df
        }
        logger.debug(f"Processed states for pair {pair_id}. Found {len(events_df)} events.")

    # Add more detailed logging about what was found
    event_counts = {}
    for pair_id, pair_data in all_pair_states_data.items():
        if 'events' in pair_data and isinstance(pair_data['events'], pd.DataFrame):
            event_counts[pair_id] = len(pair_data['events'])
        else:
            event_counts[pair_id] = 0
    
    logger.info(f"Completed H-bond state determination for {len(all_pair_states_data)} pairs.")
    logger.info(f"Events found per pair: {event_counts}")
    
    return all_pair_states_data


def save_tyr_thr_hbond_data(
    run_dir: str, output_dir: str, time_points: np.ndarray,
    distances: Dict[str, np.ndarray], states: Dict[str, Dict],
    db_conn: sqlite3.Connection, module_name: str
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Save Tyr-Thr hydrogen bond data to files and register."""
    if not distances or not states:
        logger.warning("No Tyr-Thr H-bond data to save.")
        return None, None, None
    
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = {'distances': None, 'states': None, 'events': None}
    
    # Save distances
    try:
        logger.info("Saving Tyr-Thr H-bond distances data...")
        dist_df = pd.DataFrame({'Time (ns)': time_points})
        
        # Add each pair's distance array to the dataframe
        for pair_id, dist_array in distances.items():
            logger.debug(f"Adding distance data for pair {pair_id} (length={len(dist_array)})")
            # Check distance array vs time_points length
            if len(dist_array) != len(time_points):
                logger.warning(f"Length mismatch for {pair_id}: distance array ({len(dist_array)}) vs time_points ({len(time_points)})")
                # Truncate or pad as needed
                effective_len = min(len(dist_array), len(time_points))
                if len(dist_array) > len(time_points):
                    dist_array = dist_array[:len(time_points)]
                    logger.warning(f"Truncated distance array to {len(time_points)} points")
                else:
                    # Pad with NaN
                    pad_array = np.full(len(time_points) - len(dist_array), np.nan)
                    dist_array = np.concatenate([dist_array, pad_array])
                    logger.warning(f"Padded distance array with NaNs to {len(time_points)} points")
            
            dist_df[pair_id] = dist_array
        
        dist_file = os.path.join(output_dir, "tyr_thr_hbond_distances.csv")
        dist_df.to_csv(dist_file, index=False, float_format='%.3f')
        dist_path = os.path.relpath(dist_file, run_dir)
        
        register_product(
            db_conn, module_name, "csv", "data", dist_path, 
            subcategory="tyr_thr_hbond_distances", 
            description="Time series of Tyr445-Thr439 hydrogen bond distances"
        )
        
        logger.info(f"Saved Tyr-Thr H-bond distances to {dist_file}")
        saved_paths['distances'] = dist_path
    except Exception as e:
        logger.error(f"Failed to save H-bond distances: {e}", exc_info=True)
    
    # Save states
    try:
        logger.info("Saving Tyr-Thr H-bond states data...")
        states_df = pd.DataFrame({'Time (ns)': time_points})
        
        # Add each pair's state data to the dataframe
        for pair_id, state_data in states.items():
            logger.debug(f"Processing state data for pair {pair_id}")
            
            # We need to extract state arrays from the DataFrames in state_data
            # Fix: Check if raw_signal and debounced_signal exist and contain 'State' column
            if 'raw_signal' in state_data and 'State' in state_data['raw_signal']:
                raw_states = state_data['raw_signal']['State'].to_numpy()
                logger.debug(f"Found raw states for {pair_id}, length={len(raw_states)}")
            else:
                logger.warning(f"Missing raw state data for {pair_id}")
                raw_states = np.zeros(len(time_points))
            
            if 'debounced_signal' in state_data and 'State' in state_data['debounced_signal']:
                debounced_states = state_data['debounced_signal']['State'].to_numpy()
                logger.debug(f"Found debounced states for {pair_id}, length={len(debounced_states)}")
            else:
                logger.warning(f"Missing debounced state data for {pair_id}")
                debounced_states = np.zeros(len(time_points))
            
            # Check lengths
            if len(raw_states) != len(time_points):
                logger.warning(f"Length mismatch for {pair_id} raw states: {len(raw_states)} vs {len(time_points)}")
                # Truncate or pad
                if len(raw_states) > len(time_points):
                    raw_states = raw_states[:len(time_points)]
                else:
                    # Pad with zeros
                    pad_array = np.zeros(len(time_points) - len(raw_states))
                    raw_states = np.concatenate([raw_states, pad_array])
            
            if len(debounced_states) != len(time_points):
                logger.warning(f"Length mismatch for {pair_id} debounced states: {len(debounced_states)} vs {len(time_points)}")
                # Truncate or pad
                if len(debounced_states) > len(time_points):
                    debounced_states = debounced_states[:len(time_points)]
                else:
                    # Pad with zeros
                    pad_array = np.zeros(len(time_points) - len(debounced_states))
                    debounced_states = np.concatenate([debounced_states, pad_array])
            
            # Add to dataframe
            states_df[f"{pair_id}_raw"] = raw_states
            states_df[f"{pair_id}_debounced"] = debounced_states
        
        states_file = os.path.join(output_dir, "tyr_thr_hbond_states.csv")
        states_df.to_csv(states_file, index=False)
        states_path = os.path.relpath(states_file, run_dir)
        
        register_product(
            db_conn, module_name, "csv", "data", states_path,
            subcategory="tyr_thr_hbond_states",
            description="Time series of Tyr445-Thr439 hydrogen bond states"
        )
        
        logger.info(f"Saved Tyr-Thr H-bond states to {states_file}")
        saved_paths['states'] = states_path
    except Exception as e:
        logger.error(f"Failed to save H-bond states: {e}", exc_info=True)
    
    # Save events
    try:
        logger.info("Saving Tyr-Thr H-bond events data...")
        all_events = []
        
        # Extract events from each pair
        for pair_id, state_data in states.items():
            if 'events' not in state_data:
                logger.warning(f"No events data for pair {pair_id}")
                continue
                
            # Check if events exists and is a DataFrame
            if not isinstance(state_data['events'], pd.DataFrame):
                logger.warning(f"Events for {pair_id} is not a DataFrame")
                continue
                
            # Extract events from the DataFrame
            for _, event in state_data['events'].iterrows():
                try:
                    # Map event info (adapting to actual column names in the events DataFrame)
                    event_dict = {
                        "Pair": pair_id,
                        "Event Time (ns)": event.get('EventTime', np.nan),
                        "From State": event.get('FromState', -1),
                        "To State": event.get('ToState', -1),
                        "Event Type": event.get('EventType', 'Unknown'),
                        "Previous State Start (ns)": event.get('PreviousStateStartTime', np.nan),
                        "Duration (ns)": event.get('PreviousStateDuration', np.nan)
                    }
                    all_events.append(event_dict)
                except Exception as e_event:
                    logger.warning(f"Error processing event for {pair_id}: {e_event}")
        
        if all_events:
            logger.info(f"Found {len(all_events)} events to save")
            events_df = pd.DataFrame(all_events)
            events_file = os.path.join(output_dir, "tyr_thr_hbond_events.csv")
            events_df.to_csv(events_file, index=False, float_format='%.4f')
            events_path = os.path.relpath(events_file, run_dir)
            
            register_product(
                db_conn, module_name, "csv", "data", events_path,
                subcategory="tyr_thr_hbond_events",
                description="Tyr445-Thr439 hydrogen bond state events"
            )
            
            logger.info(f"Saved Tyr-Thr H-bond events to {events_file}")
            saved_paths['events'] = events_path
        else:
            logger.warning("No events to save")
    except Exception as e:
        logger.error(f"Failed to save H-bond events: {e}", exc_info=True)
    
    return saved_paths['distances'], saved_paths['states'], saved_paths['events']


def calculate_and_store_tyr_thr_hbond_metrics(
    states: Dict[str, Dict], db_conn: sqlite3.Connection, module_name: str
) -> None:
    """Calculate and store metrics for Tyr-Thr hydrogen bonds (legacy method)."""
    if not states:
        logger.warning("No Tyr-Thr H-bond state data for metrics.")
        return
    
    logger.info("Calculating and storing Tyr-Thr H-bond metrics (legacy method)...")
    
    for pair_id, state_data in states.items():
        logger.info(f"Processing metrics for pair {pair_id}...")
        
        # Check if we have debounced state data
        if 'debounced_signal' not in state_data or not isinstance(state_data['debounced_signal'], pd.DataFrame):
            logger.warning(f"No valid debounced_signal DataFrame for {pair_id}, skipping metrics")
            continue
            
        # Check if we have the 'State' column
        if 'State' not in state_data['debounced_signal'].columns:
            logger.warning(f"Missing 'State' column in debounced_signal for {pair_id}, skipping metrics")
            continue
            
        # Extract debounced states as a numpy array
        debounced_states = state_data['debounced_signal']['State'].to_numpy()
        
        if len(debounced_states) == 0:
            logger.warning(f"Empty debounced states for {pair_id}, skipping metrics")
            continue
            
        logger.debug(f"Debounced states for {pair_id}: shape={debounced_states.shape}, unique_values={np.unique(debounced_states)}")
        
        # Check if we have events data
        if 'events' not in state_data or not isinstance(state_data['events'], pd.DataFrame):
            logger.warning(f"No valid events DataFrame for {pair_id}, skipping event-based metrics")
            formed_events = []
            broken_events = []
        else:
            # Extract events based on event type
            if 'EventType' in state_data['events'].columns:
                formed_events = state_data['events'][state_data['events']['EventType'] == 'Formed']
                broken_events = state_data['events'][state_data['events']['EventType'] == 'Broken']
                logger.debug(f"Found {len(formed_events)} formed events and {len(broken_events)} broken events")
            else:
                logger.warning(f"Missing 'EventType' column in events for {pair_id}, events may be malformed")
                formed_events = pd.DataFrame()
                broken_events = pd.DataFrame()
        
        # Calculate durations if we have the right data
        if 'PreviousStateDuration' in formed_events.columns:
            formed_durations = formed_events['PreviousStateDuration'].to_numpy()
        else:
            logger.warning(f"Missing 'PreviousStateDuration' column in formed events")
            formed_durations = []
            
        if 'PreviousStateDuration' in broken_events.columns:
            broken_durations = broken_events['PreviousStateDuration'].to_numpy()
        else:
            logger.warning(f"Missing 'PreviousStateDuration' column in broken events")
            broken_durations = []
        
        # Calculate state statistics
        total_frames = len(debounced_states)
        formed_count = np.sum(debounced_states == 1)  # 1 means formed
        broken_count = np.sum(debounced_states == 0)  # 0 means broken
        
        formed_fraction = (formed_count / total_frames * 100) if total_frames > 0 else np.nan
        broken_fraction = (broken_count / total_frames * 100) if total_frames > 0 else np.nan
        
        logger.debug(f"Statistics for {pair_id}: formed_count={formed_count}, broken_count={broken_count}, total={total_frames}")
        logger.debug(f"Fractions for {pair_id}: formed={formed_fraction:.2f}%, broken={broken_fraction:.2f}%")
        
        # Calculate duration statistics
        mean_formed = np.mean(formed_durations) if formed_durations.size > 0 else np.nan
        median_formed = np.median(formed_durations) if formed_durations.size > 0 else np.nan
        mean_broken = np.mean(broken_durations) if broken_durations.size > 0 else np.nan
        median_broken = np.median(broken_durations) if broken_durations.size > 0 else np.nan
        
        # Store metrics in database
        # Count metrics
        logger.info(f"Storing event count metrics for {pair_id}...")
        store_metric(db_conn, module_name, f"TyrThr_{pair_id}_formed_Count", len(formed_events), "count", "Count of H-bond formed events")
        store_metric(db_conn, module_name, f"TyrThr_{pair_id}_broken_Count", len(broken_events), "count", "Count of H-bond broken events")
        
        # Duration metrics (only if we have valid data)
        logger.info(f"Storing duration metrics for {pair_id}...")
        if not np.isnan(mean_formed):
            store_metric(db_conn, module_name, f"TyrThr_{pair_id}_formed_Mean_ns", mean_formed, "ns", "Mean duration of H-bond formed state")
        if not np.isnan(median_formed):
            store_metric(db_conn, module_name, f"TyrThr_{pair_id}_formed_Median_ns", median_formed, "ns", "Median duration of H-bond formed state")
        if not np.isnan(mean_broken):
            store_metric(db_conn, module_name, f"TyrThr_{pair_id}_broken_Mean_ns", mean_broken, "ns", "Mean duration of H-bond broken state")
        if not np.isnan(median_broken):
            store_metric(db_conn, module_name, f"TyrThr_{pair_id}_broken_Median_ns", median_broken, "ns", "Median duration of H-bond broken state")
        
        # Fraction metrics
        logger.info(f"Storing fraction metrics for {pair_id}...")
        if not np.isnan(formed_fraction):
            store_metric(db_conn, module_name, f"TyrThr_{pair_id}_Formed_Fraction", formed_fraction, "%", "Percentage of time H-bond was formed")
        if not np.isnan(broken_fraction):
            store_metric(db_conn, module_name, f"TyrThr_{pair_id}_Broken_Fraction", broken_fraction, "%", "Percentage of time H-bond was broken")
        
        logger.info(f"Completed metrics for pair {pair_id}")
    
    logger.info("Completed H-bond metric calculation and storage (legacy method)")


def calculate_and_store_tyr_thr_hbond_stats_from_events(
    processed_events_df: pd.DataFrame,
    total_simulation_time_ns: float, 
    db_conn: sqlite3.Connection,
    module_name: str
):
    """
    Calculates and stores statistics from processed Tyr-Thr H-bond events.
    Uses robust event-based approach adapted from DW-gate analysis.
    
    Args:
        processed_events_df: DataFrame with columns 'Chain' (pair_id), 'State', 'Duration (ns)'.
        total_simulation_time_ns: Total duration of the analyzed trajectory segment.
        db_conn: Database connection.
        module_name: Name of the module for storing metrics.
    """
    if processed_events_df is None or processed_events_df.empty:
        logger.warning("No processed Tyr-Thr H-bond events to calculate stats from.")
        return

    logger.info("Calculating statistics from processed Tyr-Thr H-bond events...")

    # Check required columns
    required_columns = ['Chain', 'State', 'Duration (ns)']
    if not all(col in processed_events_df.columns for col in required_columns):
        logger.error(f"Missing required columns in processed events DataFrame. Found: {processed_events_df.columns.tolist()}")
        logger.error("Cannot calculate reliable metrics without required columns. Aborting.")
        return

    for pair_id, group in processed_events_df.groupby('Chain'):
        logger.info(f"Calculating robust statistics for H-bond pair {pair_id}...")

        # Filter events by state
        formed_events = group[group['State'] == 'H-bond-formed']
        broken_events = group[group['State'] == 'H-bond-broken']

        # Calculate total durations for each state
        formed_total_duration_ns = formed_events['Duration (ns)'].sum()
        broken_total_duration_ns = broken_events['Duration (ns)'].sum()

        # Calculate fractions (percentage of time in each state)
        # Ensure total_simulation_time_ns is greater than zero to avoid division by zero
        if total_simulation_time_ns > 1e-9: # Use a small epsilon
            formed_fraction = (formed_total_duration_ns / total_simulation_time_ns) * 100.0
            broken_fraction = (broken_total_duration_ns / total_simulation_time_ns) * 100.0
        elif formed_total_duration_ns > 0: # If total sim time is ~0, but we have formed duration
            formed_fraction = 100.0
            broken_fraction = 0.0
        elif broken_total_duration_ns > 0:
            formed_fraction = 0.0
            broken_fraction = 100.0
        else: # All durations and total sim time are zero or negligible
            formed_fraction = 0.0
            broken_fraction = 0.0

        # Calculate mean/median durations and event counts
        mean_formed_ns = formed_events['Duration (ns)'].mean() if not formed_events.empty else 0.0
        median_formed_ns = formed_events['Duration (ns)'].median() if not formed_events.empty else 0.0
        count_formed = len(formed_events)

        mean_broken_ns = broken_events['Duration (ns)'].mean() if not broken_events.empty else 0.0
        median_broken_ns = broken_events['Duration (ns)'].median() if not broken_events.empty else 0.0
        count_broken = len(broken_events)

        logger.debug(f"Stats for H-bond Pair {pair_id}: Formed%={formed_fraction:.1f}, Broken%={broken_fraction:.1f}, "
                     f"MeanFormedDur={mean_formed_ns:.3f}ns (N={count_formed}), MeanBrokenDur={mean_broken_ns:.3f}ns (N={count_broken})")

        # Store metrics using consistent naming conventions for HTML report compatibility
        store_metric(db_conn, module_name, f"TyrThr_{pair_id}_Formed_Fraction", formed_fraction, "%", f"% Time H-bond {pair_id} was formed")
        store_metric(db_conn, module_name, f"TyrThr_{pair_id}_Broken_Fraction", broken_fraction, "%", f"% Time H-bond {pair_id} was broken")
        store_metric(db_conn, module_name, f"TyrThr_{pair_id}_formed_Mean_ns", mean_formed_ns, "ns", f"Mean duration H-bond {pair_id} formed state (ns)")
        store_metric(db_conn, module_name, f"TyrThr_{pair_id}_broken_Mean_ns", mean_broken_ns, "ns", f"Mean duration H-bond {pair_id} broken state (ns)")
        store_metric(db_conn, module_name, f"TyrThr_{pair_id}_formed_Median_ns", median_formed_ns, "ns", f"Median duration H-bond {pair_id} formed state (ns)")
        store_metric(db_conn, module_name, f"TyrThr_{pair_id}_broken_Median_ns", median_broken_ns, "ns", f"Median duration H-bond {pair_id} broken state (ns)")
        store_metric(db_conn, module_name, f"TyrThr_{pair_id}_formed_Count", count_formed, "count", f"Number of 'H-bond-formed' periods for {pair_id}")
        store_metric(db_conn, module_name, f"TyrThr_{pair_id}_broken_Count", count_broken, "count", f"Number of 'H-bond-broken' periods for {pair_id}")
    
    logger.info("Completed robust H-bond metric calculation from processed events.")