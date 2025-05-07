# pore_analysis/modules/inner_vestibule_analysis/computation.py
"""
Computation functions for analyzing water occupancy and residence time
in the channel inner vestibule.
"""

import os
import logging
import numpy as np
import pandas as pd
import MDAnalysis as mda
import json
import time
import sqlite3
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm
from typing import Dict, Optional, Tuple, List, Any

# Import from core modules
try:
    from pore_analysis.core.utils import frames_to_time
    from pore_analysis.core.config import EXIT_BUFFER_FRAMES, FRAMES_PER_NS
    from pore_analysis.core.logging import setup_system_logger
    from pore_analysis.core.database import (
        register_module, update_module_status, register_product, store_metric,
        get_product_path, get_simulation_metadata
    )
except ImportError as e:
    print(f"Error importing dependency modules in inner_vestibule_analysis/computation.py: {e}")
    raise

logger = logging.getLogger(__name__)

def _save_vestibule_data_files(
    run_dir: str,
    time_points: np.ndarray,
    water_counts_per_frame: np.ndarray,
    waters_indices_per_frame: List[set],
    all_residence_times_ns: List[float],
    summary_stats: Dict[str, Any], # Pass calculated stats
    db_conn: sqlite3.Connection,
    module_name: str
) -> Tuple[Optional[str], Optional[str]]:
    """
    Internal helper to save vestibule analysis data files (occupancy CSV, residence JSON)
    and register them in the database.
    """
    output_dir = os.path.join(run_dir, "inner_vestibule_analysis")
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = {'occupancy': None, 'residence': None}

    # Save Occupancy Data (Counts & Indices)
    try:
        # Convert indices sets to comma-separated strings for CSV storage
        indices_str_list = [','.join(map(str, sorted(list(indices)))) for indices in waters_indices_per_frame]

        df_occup = pd.DataFrame({
            'Time (ns)': time_points,
            'Water_Count': water_counts_per_frame,
            'Inner_Vestibule_Indices': indices_str_list
        })
        occ_csv_path = os.path.join(output_dir, "inner_vestibule_occupancy.csv")
        df_occup.to_csv(occ_csv_path, index=False, float_format='%.4f', na_rep='NaN')
        logger.info(f"Saved inner vestibule occupancy data (with indices) to {occ_csv_path}")
        rel_path_occ = os.path.relpath(occ_csv_path, run_dir)
        register_product(db_conn, module_name, "csv", "data", rel_path_occ,
                         subcategory="occupancy_per_frame",
                         description="Time series of inner vestibule water counts and water indices.")
        saved_paths['occupancy'] = rel_path_occ
    except Exception as e:
        logger.error(f"Failed to save/register inner vestibule occupancy CSV: {e}")

    # Save Residence Time Data (JSON)
    try:
        # Use already calculated stats
        residence_time_data = {
            'metadata': {
                'mean_residence_time_ns': summary_stats.get('InnerVestibule_AvgResidenceTime_ns', np.nan),
                'median_residence_time_ns': summary_stats.get('InnerVestibule_MedianResidenceTime_ns', np.nan), # Add median if calculated
                'total_exit_events': summary_stats.get('InnerVestibule_TotalExitEvents', 0),
                'simulation_time_ns': time_points[-1] - time_points[0] if len(time_points) > 0 else 0.0,
                'exit_buffer_frames': EXIT_BUFFER_FRAMES,
                'analysis_datetime': datetime.now().isoformat()
            },
            'residence_times_ns': all_residence_times_ns # List of individual residence times
        }

        res_json_path = os.path.join(output_dir, "inner_vestibule_residence_times.json")
        with open(res_json_path, 'w') as f:
            json.dump(residence_time_data, f, indent=2)
        logger.info(f"Saved inner vestibule residence times (n={len(all_residence_times_ns)}) to {res_json_path}")
        rel_path_res = os.path.relpath(res_json_path, run_dir)
        register_product(db_conn, module_name, "json", "data", rel_path_res,
                         subcategory="residence_times",
                         description="Inner vestibule water residence time distribution and metadata.")
        saved_paths['residence'] = rel_path_res
    except Exception as e:
        logger.error(f"Failed to save/register inner vestibule residence times JSON: {e}")

    return saved_paths['occupancy'], saved_paths['residence']

def run_inner_vestibule_analysis(
    run_dir: str,
    universe=None,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    db_conn: sqlite3.Connection = None,
    psf_file: Optional[str] = None,
    dcd_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Computes water occupancy and residence time in the channel inner vestibule.
    Retrieves necessary precursor data (site definitions, G1 ref) from the database.
    Saves results to CSV/JSON and registers products/metrics in the database.

    Args:
        run_dir (str): Path to the specific run directory.
        universe (MDAnalysis.Universe, optional): Pre-loaded MDAnalysis Universe object.
        start_frame (int, optional): Starting frame index for analysis (0-based). Defaults to 0.
        end_frame (int, optional): Ending frame index for analysis (exclusive). If None, goes to the end.
        db_conn (sqlite3.Connection, optional): Active database connection.
        psf_file (str, optional): Path to the PSF topology file. Only used if universe is not provided.
        dcd_file (str, optional): Path to the DCD trajectory file. Only used if universe is not provided.

    Returns:
        dict: A dictionary containing status ('success', 'failed', 'skipped')
              and error message if applicable.
    """
    module_name = "inner_vestibule_analysis"
    start_time = time.time()
    register_module(db_conn, module_name, status='running')
    logger = setup_system_logger(run_dir)
    if logger is None: logger = logging.getLogger(__name__) # Fallback

    results = {'status': 'failed', 'error': None} # Default status

    # --- Retrieve Precursor Data from Database ---
    g1_ref: Optional[float] = None
    filter_sites_rel: Optional[Dict[str, float]] = None
    filter_residues: Optional[Dict[str, List[int]]] = None # Filter resids needed for pore center

    try:
        # Get G1 Reference Z (assuming stored as a metric by ion_analysis)
        # Need to coordinate the exact metric name used in ion_analysis/structure.py
        # Assuming a metric like 'Ion_G1_Ref_Z_Abs' might be stored.
        # TEMPORARY WORKAROUND: Load from site definition file comment if metric not stored yet.
        sites_def_rel = get_product_path(db_conn, 'txt', 'definition', 'binding_sites_definition', 'ion_analysis')
        if sites_def_rel:
            sites_def_abs = os.path.join(run_dir, sites_def_rel)
            if os.path.exists(sites_def_abs):
                with open(sites_def_abs, 'r') as f:
                    filter_sites_rel = {}
                    for line in f:
                        if line.strip().startswith('# Absolute Z-coordinate'):
                            try: g1_ref = float(line.split(':')[-1].strip().split()[0])
                            except: pass
                        elif ':' in line and not line.strip().startswith('#'):
                             parts = line.strip().split(':')
                             try: filter_sites_rel[parts[0].strip()] = float(parts[1].strip())
                             except: pass
                if g1_ref is None: logger.warning("Could not extract G1 ref Z from site definition file comment.")
                if not filter_sites_rel: logger.warning("Could not extract relative site positions from definition file.")
            else: logger.warning(f"Site definition file not found at registered path: {sites_def_abs}")
        else: logger.warning("Path for binding site definition file not found in database.")

        # Retrieve Filter Residues (assuming stored as JSON metadata or a product by ion_analysis)
        # Placeholder: Need a reliable way to get this. Assuming it might be stored as metadata.
        filter_res_json = get_simulation_metadata(db_conn, 'filter_residues_dict')
        if filter_res_json:
             try: filter_residues = json.loads(filter_res_json)
             except json.JSONDecodeError: logger.warning("Failed to decode filter_residues JSON metadata.")
        else:
            # Attempt to load from a potential product file (less ideal)
            filter_res_path_rel = get_product_path(db_conn, 'json', 'definition', 'filter_residues', 'ion_analysis')
            if filter_res_path_rel:
                filter_res_path_abs = os.path.join(run_dir, filter_res_path_rel)
                if os.path.exists(filter_res_path_abs):
                    try:
                         with open(filter_res_path_abs, 'r') as f_res: filter_residues = json.load(f_res)
                    except Exception as e_res_load: logger.warning(f"Failed to load filter residues from {filter_res_path_abs}: {e_res_load}")
                else: logger.warning(f"Filter residues file not found at registered path: {filter_res_path_abs}")

        # --- Input Validation ---
        if filter_sites_rel is None or 'S4' not in filter_sites_rel:
            results['error'] = "Failed to load relative filter site positions (S4 required)."
            logger.error(results['error'])
            update_module_status(db_conn, module_name, 'failed', error_message=results['error'])
            return results
        if g1_ref is None:
            # Try to get from ion_analysis computation directly if possible? Less robust.
            results['error'] = "Failed to determine G1 reference Z position."
            logger.error(results['error'])
            update_module_status(db_conn, module_name, 'failed', error_message=results['error'])
            return results
        if filter_residues is None:
            results['error'] = "Failed to load filter residue definitions (needed for pore center)."
            logger.error(results['error'])
            update_module_status(db_conn, module_name, 'failed', error_message=results['error'])
            return results

        logger.info(f"Retrieved G1 Ref Z: {g1_ref:.3f}, S4 Rel Z: {filter_sites_rel['S4']:.3f}")

    except Exception as e_fetch:
        results['error'] = f"Error fetching precursor data from database: {e_fetch}"
        logger.error(results['error'], exc_info=True)
        update_module_status(db_conn, module_name, 'failed', error_message=results['error'])
        return results

    # --- Universe Handling ---
    try:
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
        store_metric(db_conn, module_name, "start_frame", start_frame, "frame", "Starting frame index for inner vestibule analysis")
        store_metric(db_conn, module_name, "end_frame", end_frame, "frame", "Ending frame index for inner vestibule analysis")
        store_metric(db_conn, module_name, "frames_analyzed", end_frame - start_frame, "frames", "Number of frames analyzed")
            
        logger.info(f"{module_name}: Analyzing frame range {start_frame} to {end_frame} (total: {end_frame - start_frame} frames)")
        
        # The number of frames we'll actually process
        n_frames = end_frame - start_frame
        
    except Exception as e:
        results['error'] = f"Failed to load or validate Universe: {e}"
        logger.error(results['error'], exc_info=True)
        update_module_status(db_conn, module_name, 'failed', error_message=results['error'])
        return results

    # --- Core Logic (Copied & Adapted from Original analyze_inner_vestibule) ---
    try:
        if FRAMES_PER_NS <= 0: raise ValueError(f"FRAMES_PER_NS ({FRAMES_PER_NS}) must be positive.")
        time_per_frame = 1.0 / FRAMES_PER_NS
        frame_indices = np.arange(start_frame, end_frame)
        time_points = frames_to_time(frame_indices)

        Z_S4_abs = filter_sites_rel['S4'] + g1_ref
        Z_upper_boundary = Z_S4_abs - 5.0
        Z_lower_boundary = Z_S4_abs - 15.0
        Radial_cutoff = 6.0
        Radial_cutoff_sq = Radial_cutoff**2
        logger.info(f"Inner vestibule definition: Z=[{Z_lower_boundary:.2f}, {Z_upper_boundary:.2f}) Å, R<{Radial_cutoff:.1f} Å")

        g1_ca_sel_parts = []
        for segid, resids in filter_residues.items():
            if len(resids) >= 3: g1_ca_sel_parts.append(f"(segid {segid} and resid {resids[2]} and name CA)")
        if not g1_ca_sel_parts: raise ValueError("Could not build G1 C-alpha selection string.")
        g1_ca_sel_str = " or ".join(g1_ca_sel_parts)
        g1_ca_atoms = u.select_atoms(g1_ca_sel_str)
        if not g1_ca_atoms: raise ValueError(f"G1 C-alpha selection '{g1_ca_sel_str}' returned no atoms.")

        water_selection_string = "name OW or type OW or ((resname TIP3 or resname WAT or resname HOH) and (name O or name OH2))"
        water_O_atoms = u.select_atoms(water_selection_string)
        if not water_O_atoms:
            logger.warning(f"No water oxygen atoms found using selection: '{water_selection_string}'. Skipping analysis.")
            update_module_status(db_conn, module_name, 'skipped', error_message="No water atoms found")
            results['status'] = 'skipped'
            return results

        logger.info(f"Found {len(water_O_atoms)} water oxygen atoms.")

        # --- Tracking Variables ---
        water_counts_per_frame = np.full(n_frames, -1, dtype=int) # Init with -1 for error indication
        waters_indices_per_frame = [set() for _ in range(n_frames)]
        water_residence_times = defaultdict(list)
        water_entry_frame = {}
        waters_currently_outside = set(water_O_atoms.indices)
        waters_outside_streak = defaultdict(int)
        waters_pending_exit_confirmation = {}

        # --- Trajectory Iteration ---
        logger.info("Iterating trajectory for vestibule analysis...")
        for ts in tqdm(u.trajectory[start_frame:end_frame], 
                      desc=f"Vestibule Water ({os.path.basename(run_dir)})", 
                      unit="frame", 
                      disable=not logger.isEnabledFor(logging.INFO)):
            frame_idx = ts.frame
            # Use a local index for storing in our arrays
            local_idx = frame_idx - start_frame
            try:
                pore_center_xy = g1_ca_atoms.center_of_geometry()[:2]
                water_pos = water_O_atoms.positions
                z_mask = (water_pos[:, 2] >= Z_lower_boundary) & (water_pos[:, 2] < Z_upper_boundary)
                dx = water_pos[:, 0] - pore_center_xy[0]
                dy = water_pos[:, 1] - pore_center_xy[1]
                r_mask = (dx*dx + dy*dy) < Radial_cutoff_sq
                inside_mask = z_mask & r_mask
                current_waters_indices_inside = set(water_O_atoms[inside_mask].indices)

                water_counts_per_frame[local_idx] = len(current_waters_indices_inside)
                waters_indices_per_frame[local_idx] = current_waters_indices_inside
                outside_this_frame = set(water_O_atoms.indices) - current_waters_indices_inside

                # --- State Update Logic (Identical to original) ---
                # 1. Handle waters that were outside last frame and still outside
                for water_idx in waters_currently_outside.intersection(outside_this_frame):
                    waters_outside_streak[water_idx] += 1
                    if water_idx in waters_pending_exit_confirmation and waters_outside_streak[water_idx] > EXIT_BUFFER_FRAMES:
                        if water_idx in water_entry_frame:
                            entry_frame = water_entry_frame[water_idx]
                            exit_frame = waters_pending_exit_confirmation[water_idx]
                            if exit_frame >= entry_frame: # Ensure exit is not before entry
                                residence_time_frames = exit_frame - entry_frame + 1 # Inclusive? Check logic. Let's keep original logic: exit_frame - entry_frame
                                residence_time_ns = (exit_frame - entry_frame) * time_per_frame
                                water_residence_times[water_idx].append(residence_time_ns)
                                water_entry_frame.pop(water_idx, None)
                        waters_pending_exit_confirmation.pop(water_idx, None)

                # 2. Handle waters that were inside but are now outside (starting streak)
                # Corrected logic: symmetric_difference gives elements in either set, but not both.
                # We want elements that were *not* outside before, but *are* outside now.
                # This is simply: current_waters_indices_inside - current_waters_indices_inside = waters_currently_outside - outside_this_frame
                exited_now = waters_currently_outside.symmetric_difference(outside_this_frame) & outside_this_frame # Original logic kept
                for water_idx in exited_now:
                    waters_outside_streak[water_idx] = 1
                    waters_pending_exit_confirmation[water_idx] = frame_idx # Start exit confirmation *now*

                # 3. Handle waters that were outside but now inside (entry or re-entry)
                entered_now = waters_currently_outside - outside_this_frame
                for water_idx in entered_now:
                    waters_outside_streak[water_idx] = 0
                    waters_pending_exit_confirmation.pop(water_idx, None)
                    if water_idx not in water_entry_frame:
                        water_entry_frame[water_idx] = frame_idx

                # Update currently outside set for next frame
                waters_currently_outside = outside_this_frame
            except Exception as e_frame:
                 logger.error(f"Error analyzing frame {frame_idx} (local index {local_idx}): {e_frame}", exc_info=True)
                 water_counts_per_frame[local_idx] = -1 # Mark frame as failed

        # --- Flatten Residence Times ---
        all_residence_times_ns = [rt for times in water_residence_times.values() for rt in times if rt > 0] # Ensure only positive times stored

        # --- Calculate Summary Statistics ---
        valid_counts = water_counts_per_frame[water_counts_per_frame >= 0] # Exclude failed frames
        if len(valid_counts) == 0: raise ValueError("No valid frames processed for occupancy.")
        mean_occupancy = np.mean(valid_counts)
        std_occupancy = np.std(valid_counts)
        avg_residence_time = np.mean(all_residence_times_ns) if all_residence_times_ns else 0.0
        median_residence_time = np.median(all_residence_times_ns) if all_residence_times_ns else 0.0 # Added median
        total_confirmed_exits = len(all_residence_times_ns)
        total_valid_time_ns = len(valid_counts) * time_per_frame # Use valid frames for rate
        exchange_rate_per_ns = total_confirmed_exits / total_valid_time_ns if total_valid_time_ns > 0 else 0.0

        summary_stats = {
            'InnerVestibule_MeanOcc': float(mean_occupancy),
            'InnerVestibule_StdOcc': float(std_occupancy),
            'InnerVestibule_AvgResidenceTime_ns': float(avg_residence_time),
            'InnerVestibule_MedianResidenceTime_ns': float(median_residence_time), # Added
            'InnerVestibule_TotalExitEvents': int(total_confirmed_exits),
            'InnerVestibule_ExchangeRatePerNs': float(exchange_rate_per_ns)
        }
        logger.info(f"Calculated inner vestibule summary stats: {summary_stats}")

        # --- Save Data Files ---
        occ_path, res_path = _save_vestibule_data_files(
            run_dir, time_points, water_counts_per_frame, waters_indices_per_frame,
            all_residence_times_ns, summary_stats, db_conn, module_name
            )
        if not occ_path or not res_path:
             # Log warning but maybe don't fail if stats were calculated
             logger.warning("Failed to save one or both vestibule data files.")

        # --- Store Metrics ---
        for key, value in summary_stats.items():
            units = 'ns' if 'Time_ns' in key else ('count' if ('Occ' in key or 'Events' in key) else ('rate (ns^-1)' if 'Rate' in key else None))
            desc = key.replace('InnerVestibule_', '').replace('_', ' ')
            if np.isfinite(value):
                store_metric(db_conn, module_name, key, value, units, desc)
            else:
                 logger.warning(f"Metric '{key}' has non-finite value ({value}), not storing.")

        results['status'] = 'success'

    except Exception as e_main:
        results['error'] = f"Error during inner vestibule analysis: {e_main}"
        logger.error(results['error'], exc_info=True)
        update_module_status(db_conn, module_name, 'failed', error_message=results['error'])
        return results

    # --- Finalize ---
    exec_time = time.time() - start_time
    update_module_status(db_conn, module_name, results['status'], execution_time=exec_time, error_message=results['error'])
    logger.info(f"--- Inner Vestibule Analysis completed in {exec_time:.2f} seconds (Status: {results['status']}) ---")

    return results
