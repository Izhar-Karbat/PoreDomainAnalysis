# filename: pore_analysis/modules/gyration_analysis/computation.py
"""
Computation functions for analyzing the radius of gyration (ρ) of selectivity filter
carbonyls (G1 Glycine, Y Tyrosine).

Calculates gyration radii, detects flipping events based on threshold/tolerance,
saves data, and stores metrics in the database.
"""

import os
import logging
import numpy as np
import pandas as pd
import time
import sqlite3
from collections import defaultdict
from tqdm import tqdm
import MDAnalysis as mda
from typing import Dict, Optional, Tuple, List, Any

# Import from core modules
try:
    from pore_analysis.core.utils import frames_to_time
    # Assume find_filter_residues is now correctly in ion_analysis.structure
    from pore_analysis.modules.ion_analysis.structure import find_filter_residues
    from pore_analysis.core.logging import setup_system_logger
    from pore_analysis.core.config import (
        GYRATION_FLIP_THRESHOLD, GYRATION_FLIP_TOLERANCE_FRAMES, FRAMES_PER_NS
    )
    from pore_analysis.core.database import (
        register_module, update_module_status, register_product, store_metric
    )
    # Import config values needed for storing as metrics
    from pore_analysis.core import config as core_config # Import the module itself
except ImportError as e:
    print(f"Error importing dependency modules in gyration_analysis/computation.py: {e}")
    raise

logger = logging.getLogger(__name__) # Use module-level logger

# --- Helper: Calculate Pore Center (from original gyration_analysis.py) ---
def _calculate_pore_center(universe: mda.Universe, filter_residues: Dict[str, List[int]]) -> Optional[np.ndarray]:
    """
    Calculate the geometric center of the pore for the current frame.
    Uses the alpha carbons of the filter G1 residues. Returns None on error.
    """
    g1_ca_atoms = []
    for segid, resids in filter_residues.items():
        if len(resids) >= 3:  # Need at least up to G1
            g1_resid = resids[2]  # G1 is at index 2 (TVGYG)
            sel_str = f"segid {segid} and resid {g1_resid} and name CA"
            try:
                atom_group = universe.select_atoms(sel_str)
                if len(atom_group) > 0:
                    # Select only the first atom if multiple are found (shouldn't happen for CA)
                    g1_ca_atoms.append(atom_group[0])
                else:
                    logger.warning(f"No G1 CA atom found for segid {segid} resid {g1_resid}. Skipping for pore center.")
            except Exception as e_sel:
                 logger.warning(f"Error selecting G1 CA for segid {segid} resid {g1_resid}: {e_sel}")
        else:
            logger.warning(f"Chain {segid} has fewer than 3 filter residues. Cannot use for pore center.")

    if g1_ca_atoms:
        positions = np.vstack([atom.position for atom in g1_ca_atoms])
        center = np.mean(positions, axis=0)
        return center
    else:
        logger.error("Could not find ANY G1 C-alpha atoms for pore center calculation.")
        return None

# --- Helper: Analyze Carbonyl States (from original gyration_analysis.py, adapted) ---
def _analyze_carbonyl_states(
    gyration_data: Dict[str, Dict[str, List[float]]],
    time_points: np.ndarray,
    threshold: float,
    tolerance_frames: int,
    dt: float,
    run_dir: str, # Added for saving events
    db_conn: sqlite3.Connection, # Added for registration
    module_name: str # Added for registration
    ) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Analyzes carbonyl states, confirms states using tolerance, identifies transitions,
    calculates durations, saves detailed event data, and returns summary stats.
    Returns (results_dict, events_csv_rel_path)
    """
    results = {'g1': {}, 'y': {}}
    all_event_data = [] # To store dicts for CSV saving
    events_csv_rel_path = None

    for residue_type in ['g1', 'y']:
        res_results = {
            'on_flips': {}, 'off_flips': {}, 'flip_durations_ns': {},
            'mean_flip_duration_ns': {}, 'std_flip_duration_ns': {},
            'max_flip_duration_ns': {}, # <<< ADDED: Max duration per chain
            'events': {} # Store events per chain if needed later, maybe remove if only saving CSV
        }
        for chain, radii in gyration_data.get(residue_type, {}).items():
            radii_array = np.array(radii)
            n_frames = len(radii_array)

            if n_frames < tolerance_frames:
                logger.debug(f"Chain {chain} {residue_type.upper()}: Not enough frames ({n_frames}) for tolerance ({tolerance_frames}). Skipping state analysis.")
                res_results['on_flips'][chain] = 0
                res_results['off_flips'][chain] = 0
                res_results['flip_durations_ns'][chain] = []
                res_results['mean_flip_duration_ns'][chain] = np.nan
                res_results['std_flip_duration_ns'][chain] = np.nan
                res_results['max_flip_duration_ns'][chain] = np.nan # <<< ADDED
                continue

            states = np.full(n_frames, -1, dtype=int)
            valid_indices = np.where(np.isfinite(radii_array))[0]
            states[valid_indices] = (radii_array[valid_indices] > threshold).astype(int)

            confirmed_blocks = []
            current_state = -2
            block_start = -1
            consecutive_frames = 0

            for i in range(n_frames):
                frame_state = states[i]
                if frame_state == -1: # Handle NaN/invalid frames
                    if block_start != -1 and consecutive_frames >= tolerance_frames:
                        confirmed_blocks.append((block_start, i - 1, current_state))
                    current_state, block_start, consecutive_frames = -2, -1, 0
                    continue

                if frame_state == current_state:
                    consecutive_frames += 1
                else:
                    if block_start != -1 and consecutive_frames >= tolerance_frames:
                         confirmed_blocks.append((block_start, i - 1, current_state))
                    # Start new block
                    current_state = frame_state
                    block_start = i
                    consecutive_frames = 1 # Start counting from 1 for the current frame

            # Add the last block if it meets tolerance
            if block_start != -1 and consecutive_frames >= tolerance_frames:
                confirmed_blocks.append((block_start, n_frames - 1, current_state))

            on_count, off_count = 0, 0
            durations = []
            chain_events = [] # Simplified internal event tracking
            last_confirmed_state = -2

            for i, block in enumerate(confirmed_blocks):
                start_frame, end_frame, state = block
                # Calculate duration only for FLIPPED state blocks (state == 1)
                if state == 1:
                    duration_frames = (end_frame - start_frame) + 1
                    duration_ns = duration_frames * dt
                    durations.append(duration_ns)

                    # Check for 'On' transition (previous state was Normal=0)
                    if i > 0 and confirmed_blocks[i-1][2] == 0:
                         on_count += 1
                         event_time = time_points[start_frame]
                         # Save 'on' event - duration belongs to the current flipped block
                         all_event_data.append({
                             'residue': residue_type.upper(), 'chain': chain, 'type': 'on',
                             'frame': start_frame, 'time_ns': event_time, 'duration_ns': duration_ns
                         })
                         chain_events.append({'type': 'on', 'frame': start_frame, 'time': event_time})
                    # Handle case where the *first* confirmed block is flipped
                    elif i == 0:
                         logger.debug(f"Chain {chain} {residue_type.upper()} started in or transitioned to flipped state before first confirmed normal block.")
                         # We don't count this as an 'on' *transition* event, but the duration is valid

                # Check for 'Off' transition (current state is Normal=0, previous was Flipped=1)
                elif state == 0 and i > 0 and confirmed_blocks[i-1][2] == 1:
                    off_count += 1
                    event_time = time_points[start_frame]
                    # Duration belongs to the *previous* flipped block
                    prev_duration = durations[-1] if durations else np.nan
                    all_event_data.append({
                        'residue': residue_type.upper(), 'chain': chain, 'type': 'off',
                        'frame': start_frame, 'time_ns': event_time, 'duration_ns': prev_duration
                    })
                    chain_events.append({'type': 'off', 'frame': start_frame, 'time': event_time, 'prev_duration': prev_duration})

                last_confirmed_state = state # Update last state for next iteration

            res_results['on_flips'][chain] = on_count
            res_results['off_flips'][chain] = off_count
            res_results['flip_durations_ns'][chain] = durations
            res_results['mean_flip_duration_ns'][chain] = np.mean(durations) if durations else np.nan
            res_results['std_flip_duration_ns'][chain] = np.std(durations) if len(durations) > 1 else np.nan
            # <<< ADDED: Calculate Max Duration >>>
            res_results['max_flip_duration_ns'][chain] = np.max(durations) if durations else np.nan
            # res_results['events'][chain] = chain_events # Keep if needed internally

        results[residue_type] = res_results

    # Save event data to CSV
    if all_event_data:
        try:
            output_dir = os.path.join(run_dir, "gyration_analysis")
            os.makedirs(output_dir, exist_ok=True)
            events_df = pd.DataFrame(all_event_data)
            # Ensure correct columns exist before saving
            cols_to_save = ['residue', 'chain', 'type', 'frame', 'time_ns', 'duration_ns']
            events_df = events_df[[col for col in cols_to_save if col in events_df.columns]]
            events_csv_path = os.path.join(output_dir, "gyration_flip_events.csv")
            events_df.to_csv(events_csv_path, index=False, float_format='%.4f', na_rep='NaN')
            logger.info(f"Saved gyration flip event data to {events_csv_path}")
            events_csv_rel_path = os.path.relpath(events_csv_path, run_dir)
            register_product(db_conn, module_name, "csv", "data", events_csv_rel_path,
                             subcategory="gyration_flip_events",
                             description="Details of confirmed carbonyl flip events (On/Off).")
        except Exception as e:
            logger.error(f"Failed to save gyration flip events CSV: {e}")
            events_csv_rel_path = None
    else:
         logger.info("No confirmed flip events found to save.")

    return results, events_csv_rel_path

# --- Helper: Aggregate Stats (from original gyration_analysis.py) ---
def _calculate_gyration_statistics(gyration_data, state_analysis_results):
    """Calculate final summary statistics including Max Duration."""
    stats = {}
    # Mean/Std Gyration Radius
    all_g1_radii = np.concatenate([np.array(r) for r in gyration_data.get('g1', {}).values() if r]) # Handle empty lists
    all_y_radii = np.concatenate([np.array(r) for r in gyration_data.get('y', {}).values() if r]) # Handle empty lists
    stats['Gyration_G1_Mean'] = np.nanmean(all_g1_radii) if all_g1_radii.size > 0 else np.nan
    stats['Gyration_G1_Std'] = np.nanstd(all_g1_radii) if all_g1_radii.size > 0 else np.nan
    stats['Gyration_Y_Mean'] = np.nanmean(all_y_radii) if all_y_radii.size > 0 else np.nan
    stats['Gyration_Y_Std'] = np.nanstd(all_y_radii) if all_y_radii.size > 0 else np.nan

    # Aggregate State Analysis Results
    for residue_type in ['g1', 'y']:
        res_results = state_analysis_results.get(residue_type, {})
        stats[f'Gyration_{residue_type.upper()}_OnFlips'] = sum(res_results.get('on_flips', {}).values())
        stats[f'Gyration_{residue_type.upper()}_OffFlips'] = sum(res_results.get('off_flips', {}).values())
        # Collect all durations from all chains for this residue type
        all_durations = [d for chain_durations in res_results.get('flip_durations_ns', {}).values() for d in chain_durations if np.isfinite(d)]
        # Store list of durations if needed for plotting/summary, maybe remove if only mean/std needed
        # stats[f'Gyration_{residue_type.upper()}_Durations_ns'] = all_durations
        stats[f'Gyration_{residue_type.upper()}_MeanDuration_ns'] = np.mean(all_durations) if all_durations else np.nan
        stats[f'Gyration_{residue_type.upper()}_StdDuration_ns'] = np.std(all_durations) if len(all_durations) > 1 else np.nan
        # <<< ADDED: Calculate Overall Max Duration >>>
        stats[f'Gyration_{residue_type.upper()}_MaxDuration_ns'] = np.max(all_durations) if all_durations else np.nan

    return stats

# --- Main Computation Function ---
def run_gyration_analysis(
    run_dir: str,
    psf_file: Optional[str],
    dcd_file: Optional[str],
    db_conn: sqlite3.Connection
    ) -> Dict[str, Any]:
    """
    Performs gyration radius calculation and state analysis for G1/Y carbonyls.
    Saves data to CSV files and stores metrics (including Max Duration) in the database.

    Args:
        run_dir: Path to the specific run directory.
        psf_file: Path to the PSF topology file.
        dcd_file: Path to the DCD trajectory file.
        db_conn: Active database connection.

    Returns:
        Dictionary containing status and error message if applicable.
    """
    module_name = "gyration_analysis"
    start_time = time.time()
    register_module(db_conn, module_name, status='running')
    logger_local = setup_system_logger(run_dir) # Use different name
    if logger_local is None: logger_local = logging.getLogger(__name__) # Fallback

    results: Dict[str, Any] = {'status': 'failed', 'error': None} # Default status
    output_dir = os.path.join(run_dir, module_name) # Save outputs in module-specific folder
    os.makedirs(output_dir, exist_ok=True)

    # Validate input files
    psf_file = psf_file or os.path.join(run_dir, "step5_input.psf")
    dcd_file = dcd_file or os.path.join(run_dir, "MD_Aligned.dcd")
    if not os.path.exists(psf_file) or not os.path.exists(dcd_file):
        results['error'] = f"PSF or DCD file not found: {psf_file}, {dcd_file}"
        logger_local.error(results['error'])
        update_module_status(db_conn, module_name, 'failed', error_message=results['error'])
        return results

    # Load trajectory
    try:
        u = mda.Universe(psf_file, dcd_file)
        n_frames = len(u.trajectory)
        # Use config value for FRAMES_PER_NS fetched from DB or default
        # For computation, it's safer to import directly for now
        if core_config.FRAMES_PER_NS <= 0: raise ValueError("FRAMES_PER_NS must be positive.")
        dt = 1.0 / core_config.FRAMES_PER_NS # Time step in ns
        logger_local.info(f"Loaded trajectory: {n_frames} frames, dt={dt:.4f} ns")
        if n_frames < core_config.GYRATION_FLIP_TOLERANCE_FRAMES:
            logger_local.warning(f"Trajectory has {n_frames} frames, fewer than tolerance {core_config.GYRATION_FLIP_TOLERANCE_FRAMES}. State analysis may be skipped or limited.")
    except Exception as e:
        results['error'] = f"Error loading trajectory: {e}"
        logger_local.error(results['error'], exc_info=True)
        update_module_status(db_conn, module_name, 'failed', error_message=results['error'])
        return results

    try:
        # Get filter residues - reusing function from ion_analysis
        filter_residues = find_filter_residues(u)
        if not filter_residues:
            raise ValueError("Failed to identify filter residues (needed for gyration analysis).")

        # --- Trajectory Iteration for Calculation ---
        frame_indices = []
        gyration_data = {
            'g1': {chain: [] for chain in filter_residues.keys()},
            'y': {chain: [] for chain in filter_residues.keys()}
        }
        g1_oxygen_selections = {}
        y_oxygen_selections = {}

        for chain, info in filter_residues.items():
            if len(info) >= 3: # TVG..
                g1_resid, y_resid = info[2], info[1] # G1=index 2, Y=index 1
                g1_oxygen_selections[chain] = f"segid {chain} and resid {g1_resid} and name O"
                y_oxygen_selections[chain] = f"segid {chain} and resid {y_resid} and name O"
            else: logger_local.warning(f"Chain {chain}: Not enough filter resids ({len(info)})")

        logger_local.info("Calculating G1 and Y carbonyl gyration radii...")
        for ts in tqdm(u.trajectory, desc="Gyration calculation", unit="frame", disable=not logger_local.isEnabledFor(logging.INFO)):
            frame_indices.append(ts.frame)
            pore_center = _calculate_pore_center(u, filter_residues)
            if pore_center is None: # Handle error from helper
                logger_local.error(f"Failed to get pore center at frame {ts.frame}. Appending NaN.")
                for chain in filter_residues.keys():
                    if chain in g1_oxygen_selections: gyration_data['g1'][chain].append(np.nan)
                    if chain in y_oxygen_selections: gyration_data['y'][chain].append(np.nan)
                continue

            for chain in filter_residues.keys():
                for res_key, sel_dict in [('g1', g1_oxygen_selections), ('y', y_oxygen_selections)]:
                    if chain in sel_dict:
                        sel = sel_dict[chain]
                        try:
                            oxygen_atom = u.select_atoms(sel)
                            if len(oxygen_atom) == 1:
                                dist = np.linalg.norm(oxygen_atom.positions[0] - pore_center)
                                gyration_data[res_key][chain].append(dist)
                            else:
                                # logger_local.debug(f"Frame {ts.frame}, Chain {chain}, {res_key.upper()}: Expected 1 oxygen atom for selection '{sel}', found {len(oxygen_atom)}. Appending NaN.")
                                gyration_data[res_key][chain].append(np.nan)
                        except Exception as e_sel:
                            logger_local.warning(f"Error selecting atom '{sel}' at frame {ts.frame}: {e_sel}. Appending NaN.")
                            gyration_data[res_key][chain].append(np.nan)

        time_points = frames_to_time(np.array(frame_indices))

        # --- Save Raw Gyration Data ---
        # G1 Data
        g1_data_dict = {'Time (ns)': time_points}
        for chain, radii in gyration_data['g1'].items(): g1_data_dict[f'{chain}_G1'] = radii
        g1_df = pd.DataFrame(g1_data_dict)
        g1_csv_path = os.path.join(output_dir, "G1_gyration_radii.csv")
        g1_df.to_csv(g1_csv_path, index=False, float_format='%.4f', na_rep='NaN')
        logger_local.info(f"Saved G1 gyration data to {g1_csv_path}")
        register_product(db_conn, module_name, "csv", "data",
                         os.path.relpath(g1_csv_path, run_dir),
                         subcategory="g1_gyration_data", # Match plan
                         description="Time series of G1 carbonyl gyration radius per chain.")

        # Y Data
        y_data_dict = {'Time (ns)': time_points}
        for chain, radii in gyration_data['y'].items(): y_data_dict[f'{chain}_Y'] = radii
        y_df = pd.DataFrame(y_data_dict)
        y_csv_path = os.path.join(output_dir, "Y_gyration_radii.csv")
        y_df.to_csv(y_csv_path, index=False, float_format='%.4f', na_rep='NaN')
        logger_local.info(f"Saved Y gyration data to {y_csv_path}")
        register_product(db_conn, module_name, "csv", "data",
                         os.path.relpath(y_csv_path, run_dir),
                         subcategory="y_gyration_data", # Match plan
                         description="Time series of Y carbonyl gyration radius per chain.")

        # --- Analyze States & Save Events ---
        logger_local.info(f"Analyzing carbonyl states (Threshold={core_config.GYRATION_FLIP_THRESHOLD} Å, Tolerance={core_config.GYRATION_FLIP_TOLERANCE_FRAMES} frames)...")
        state_results, _ = _analyze_carbonyl_states(
            gyration_data, time_points, core_config.GYRATION_FLIP_THRESHOLD,
            core_config.GYRATION_FLIP_TOLERANCE_FRAMES, dt,
            run_dir, db_conn, module_name # Pass DB info
        )

        # --- Calculate and Store Metrics ---
        final_stats = _calculate_gyration_statistics(gyration_data, state_results)
        logger_local.info("Storing gyration metrics...")
        metrics_stored = 0
        for key, value in final_stats.items():
             # Determine units based on metric name
             units = 'Å' if ('Mean' in key or 'Std' in key) and 'Duration' not in key else \
                     'ns' if 'Duration' in key else \
                     'count' if 'Flips' in key else None
             desc = key.replace('_', ' ') # Basic description
             # Add description for Max Duration
             if 'MaxDuration' in key:
                  desc = f"Max duration of confirmed flipped state for {key.split('_')[1]}"

             # Store metric if value is finite
             if value is not None and np.isfinite(value):
                  if store_metric(db_conn, module_name, key, value, units, desc):
                       metrics_stored += 1
             else:
                  # Store non-finite values (like NaN) as NULL in the database
                  logger_local.warning(f"Metric '{key}' is non-finite ({value}), storing as NULL.")
                  if store_metric(db_conn, module_name, key, None, units, desc): # Pass None to store_metric
                      metrics_stored += 1 # Count NULL storage as successful registration

        logger_local.info(f"Stored {metrics_stored} gyration metrics in the database.")

        # --- Store Config Parameters Used ---
        # Moved this to main.py to happen once at the start
        # logger_local.info("Storing Gyration Analysis configuration parameters used...")
        # store_metric(db_conn, module_name, "Config_GYRATION_FLIP_THRESHOLD", core_config.GYRATION_FLIP_THRESHOLD, "Å", "Gyration flip threshold used")
        # store_metric(db_conn, module_name, "Config_GYRATION_FLIP_TOLERANCE_FRAMES", core_config.GYRATION_FLIP_TOLERANCE_FRAMES, "frames", "Gyration flip tolerance frames used")
        # store_metric(db_conn, module_name, "Config_FRAMES_PER_NS", core_config.FRAMES_PER_NS, "frames/ns", "Frames per nanosecond used for time calculations")

        results['status'] = 'success'

    except Exception as e_main:
        results['error'] = f"Error during gyration analysis: {e_main}"
        logger_local.error(results['error'], exc_info=True)
        update_module_status(db_conn, module_name, 'failed', error_message=results['error'])
        return results # Return immediately on critical error

    # --- Finalize ---
    exec_time = time.time() - start_time
    update_module_status(db_conn, module_name, results['status'], execution_time=exec_time)
    logger_local.info(f"--- Gyration Analysis computation finished in {exec_time:.2f} seconds (Status: {results['status']}) ---")

    return results