# filename: pore_analysis/modules/ion_analysis/tracking.py
"""
Ion Analysis: Ion position tracking and data saving.
"""
import os
import logging
import numpy as np
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.analysis import distances
from collections import defaultdict
from tqdm import tqdm
import sqlite3
from typing import Dict, Optional, Tuple, List, Set

# Import from core modules
try:
    from pore_analysis.core.utils import frames_to_time
    from pore_analysis.core.database import register_product
    from pore_analysis.core.config import EXIT_BUFFER_FRAMES # Use config
except ImportError as e:
    print(f"Error importing dependency modules in ion_analysis/tracking.py: {e}")
    raise

logger = logging.getLogger(__name__)

def _save_ion_position_csvs(
    run_dir: str,
    time_points: np.ndarray,
    ions_z_tracked_abs: Dict[int, np.ndarray],
    ion_indices: List[int],
    g1_reference: float,
    db_conn: sqlite3.Connection,
    module_name: str
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Internal helper to save K+ ion Z-position data (Abs, G1-centric, Presence) to CSV files
    and register them in the database.
    """
    output_dir = os.path.join(run_dir, "ion_analysis") # Ensure output goes here
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = {}

    if not ion_indices:
        logger.info("No tracked ion indices provided. Skipping ion position CSV saving.")
        return None, None, None

    n_frames = len(time_points)

    # --- G1-Centric Coordinates ---
    try:
        data_g1 = {'Time (ns)': time_points}
        for ion_idx in ion_indices:
            if ion_idx in ions_z_tracked_abs and len(ions_z_tracked_abs[ion_idx]) == n_frames:
                data_g1[f'Ion_{ion_idx}_Z_G1Centric'] = ions_z_tracked_abs[ion_idx] - g1_reference
            else:
                data_g1[f'Ion_{ion_idx}_Z_G1Centric'] = np.full(n_frames, np.nan)
        df_g1 = pd.DataFrame(data_g1)
        csv_path_g1 = os.path.join(output_dir, 'ion_positions_g1_centric.csv')
        df_g1.to_csv(csv_path_g1, index=False, float_format='%.4f', na_rep='NaN')
        logger.info(f"Saved ion G1-centric Z positions to {csv_path_g1}")
        rel_path = os.path.relpath(csv_path_g1, run_dir)
        register_product(db_conn, module_name, "csv", "data", rel_path,
                         subcategory="ion_positions_g1_centric",
                         description="Time series of K+ ion Z positions relative to G1 C-alpha plane.")
        saved_paths['g1_centric'] = rel_path
    except Exception as e:
        logger.error(f"Failed to save/register G1-centric ion position CSV: {e}")
        saved_paths['g1_centric'] = None

    # --- Absolute Coordinates (Optional, but often useful) ---
    try:
        data_abs = {'Time (ns)': time_points}
        for ion_idx in ion_indices:
            if ion_idx in ions_z_tracked_abs and len(ions_z_tracked_abs[ion_idx]) == n_frames:
                 data_abs[f'Ion_{ion_idx}_Z_Abs'] = ions_z_tracked_abs[ion_idx]
            else:
                 data_abs[f'Ion_{ion_idx}_Z_Abs'] = np.full(n_frames, np.nan)
        df_abs = pd.DataFrame(data_abs)
        csv_path_abs = os.path.join(output_dir, 'ion_positions_absolute.csv')
        df_abs.to_csv(csv_path_abs, index=False, float_format='%.4f', na_rep='NaN')
        logger.info(f"Saved ion absolute Z positions to {csv_path_abs}")
        rel_path = os.path.relpath(csv_path_abs, run_dir)
        register_product(db_conn, module_name, "csv", "data", rel_path,
                         subcategory="ion_positions_absolute",
                         description="Time series of K+ ion absolute Z positions.")
        saved_paths['absolute'] = rel_path
    except Exception as e:
        logger.error(f"Failed to save/register absolute ion position CSV: {e}")
        saved_paths['absolute'] = None

    # --- Ion Presence ---
    try:
        data_meta = {'Time (ns)': time_points}
        for ion_idx in ion_indices:
            if ion_idx in ions_z_tracked_abs and len(ions_z_tracked_abs[ion_idx]) == n_frames:
                is_present = ~np.isnan(ions_z_tracked_abs[ion_idx])
                data_meta[f'Ion_{ion_idx}_InFilter'] = is_present
            else:
                 data_meta[f'Ion_{ion_idx}_InFilter'] = np.full(n_frames, False)
        df_meta = pd.DataFrame(data_meta)
        csv_path_meta = os.path.join(output_dir, 'ion_filter_presence.csv')
        df_meta.to_csv(csv_path_meta, index=False)
        logger.info(f"Saved ion filter presence data to {csv_path_meta}")
        rel_path = os.path.relpath(csv_path_meta, run_dir)
        register_product(db_conn, module_name, "csv", "data", rel_path,
                         subcategory="ion_filter_presence",
                         description="Time series indicating K+ ion presence within filter coordination range.")
        saved_paths['presence'] = rel_path
    except Exception as e:
        logger.error(f"Failed to save/register ion presence CSV: {e}")
        saved_paths['presence'] = None

    return saved_paths.get('g1_centric'), saved_paths.get('absolute'), saved_paths.get('presence')


def track_ion_positions(
    universe: mda.Universe,
    filter_residues: Dict[str, List[int]],
    run_dir: str,
    db_conn: sqlite3.Connection,
    module_name: str = "ion_analysis",
    start_frame: int = 0,
    end_frame: Optional[int] = None
) -> Tuple[Optional[Dict[int, np.ndarray]], Optional[np.ndarray], Optional[List[int]]]:
    """
    Track K+ ions near the selectivity filter over the trajectory.
    Only tracks ions when they are coordinated by filter atoms, with an exit buffer.
    Saves position data to CSVs via a helper function.

    Args:
        universe: MDAnalysis Universe object.
        filter_residues: Dictionary mapping chain segids to filter resids.
        run_dir: Path to the run directory for saving output.
        db_conn: Active database connection.
        module_name: Name of the calling module for registration.
        start_frame: Starting frame index for analysis (0-based). Defaults to 0.
        end_frame: Ending frame index for analysis (exclusive). If None, goes to the end.

    Returns:
        Tuple (ions_z_abs, time_points, ion_indices):
            - ions_z_abs: Dict {ion_idx: np.array(abs_z_positions)} with NaN when not in filter, or None on error.
            - time_points: Array of time points in ns, or None on error.
            - ion_indices: List of tracked ion indices, or None on error.
    """
    logger.info("Tracking K+ ion positions near the filter...")
    n_frames_total = len(universe.trajectory)
    if n_frames_total == 0:
        logger.warning("Trajectory has 0 frames.")
        return None, None, None
        
    # Handle frame range
    if end_frame is None:
        end_frame = n_frames_total
        
    # Validate frame range
    if start_frame < 0 or start_frame >= n_frames_total:
        logger.error(f"Invalid start_frame: {start_frame}. Must be between 0 and {n_frames_total-1}")
        return None, None, None
        
    if end_frame <= start_frame or end_frame > n_frames_total:
        logger.error(f"Invalid end_frame: {end_frame}. Must be between {start_frame+1} and {n_frames_total}")
        return None, None, None
        
    # Calculate the actual number of frames to process
    n_frames = end_frame - start_frame
    logger.info(f"Analyzing frame range {start_frame} to {end_frame} (total: {n_frames} frames)")

    if not filter_residues:
        logger.error("Filter residues dictionary is empty.")
        return None, None, None

    # --- Setup Ion Selection ---
    k_selection_strings = ['name K', 'name POT', 'resname K', 'resname POT', 'type K', 'element K']
    potassium_atoms = None
    for sel in k_selection_strings:
        try:
            atoms = universe.select_atoms(sel)
            if len(atoms) > 0:
                potassium_atoms = atoms
                logger.info(f"Found {len(potassium_atoms)} K+ ions using selection: '{sel}'")
                break
        except Exception as e_sel: logger.debug(f"Selection '{sel}' failed: {e_sel}")

    if not potassium_atoms:
        logger.error("No potassium ions found using standard selections.")
        return None, None, None

    # --- Setup Filter Atom Selection (Carbonyl Oxygens) ---
    filter_sel_parts = []
    for segid, resids in filter_residues.items():
        if len(resids) == 5: # Assuming TVGYG
             filter_sel_parts.append(f"(segid {segid} and name O and resid {' '.join(map(str, resids))})")
    if not filter_sel_parts:
        logger.error("Could not build selection string for filter carbonyl oxygens.")
        return None, None, None

    filter_selection_str = " or ".join(filter_sel_parts)
    logger.debug(f"Using filter CARBONYL OXYGEN selection: {filter_selection_str}")
    filter_atoms_group = universe.select_atoms(filter_selection_str)
    if not filter_atoms_group:
        logger.error("Could not select filter carbonyl oxygens!")
        return None, None, None

    # --- Tracking Initialization ---
    frame_indices = []
    ions_z_positions_abs = {ion.index: np.full(n_frames, np.nan) for ion in potassium_atoms}
    ions_currently_inside: Set[int] = set()
    ions_outside_streak: Dict[int, int] = defaultdict(int)
    ions_pending_exit_confirmation: Dict[int, int] = {}
    tracked_ion_indices: Set[int] = set()
    cutoff = 3.5  # Distance cutoff for coordination
    exit_buffer_frames = EXIT_BUFFER_FRAMES # Use value from config

    logger.info(f"Tracking K+ ions within {cutoff} Å of filter atoms, exit buffer: {exit_buffer_frames} frames.")

    # --- Trajectory Iteration ---
    try:
        for i, ts in enumerate(tqdm(universe.trajectory[start_frame:end_frame], 
                       desc=f"Tracking K+ ({os.path.basename(run_dir)})", 
                       unit="frame", 
                       disable=not logger.isEnabledFor(logging.INFO))):
            # Store the global frame index for reference
            global_frame_idx = ts.frame
            # Use a local frame index (0-based for the slice) for array access
            local_frame_idx = i
            frame_indices.append(global_frame_idx)

            filter_atoms_pos = filter_atoms_group.positions
            potassium_pos = potassium_atoms.positions
            box = universe.dimensions

            dist_matrix = distances.distance_array(potassium_pos, filter_atoms_pos, box=box)
            min_dists = np.min(dist_matrix, axis=1)
            ions_near_this_frame = set(potassium_atoms[min_dists <= cutoff].indices)

            if i == 0:  # First frame in the slice
                exited_now, stayed_outside = set(), set(k.index for k in potassium_atoms) - ions_near_this_frame
                entered_now = ions_near_this_frame
            else:
                exited_now = ions_currently_inside - ions_near_this_frame
                entered_now = ions_near_this_frame - ions_currently_inside
                stayed_outside = set(k.index for k in potassium_atoms) - ions_near_this_frame - ions_currently_inside

            # State update logic - use local_frame_idx for array access
            for idx in entered_now:
                tracked_ion_indices.add(idx)
                ions_currently_inside.add(idx)
                ion_pos_idx = np.where(potassium_atoms.indices == idx)[0][0]
                ions_z_positions_abs[idx][local_frame_idx] = potassium_pos[ion_pos_idx, 2]
                ions_outside_streak.pop(idx, None)
                ions_pending_exit_confirmation.pop(idx, None)

            for idx in ions_currently_inside - exited_now:
                ion_pos_idx = np.where(potassium_atoms.indices == idx)[0][0]
                ions_z_positions_abs[idx][local_frame_idx] = potassium_pos[ion_pos_idx, 2]

            for idx in exited_now:
                if idx not in ions_pending_exit_confirmation:
                    ions_pending_exit_confirmation[idx] = global_frame_idx - 1  # Store global frame for reference
                ions_outside_streak[idx] += 1
                ions_currently_inside.discard(idx) # Use discard for sets

            for idx in stayed_outside:
                 if idx in ions_outside_streak: ions_outside_streak[idx] += 1
                 else: ions_outside_streak[idx] = 1 # Start streak if not present

                 if idx in ions_pending_exit_confirmation and ions_outside_streak[idx] > exit_buffer_frames:
                    ions_pending_exit_confirmation.pop(idx, None)
                    # Keep streak counter

            # Check ions returning during exit buffer
            for idx in list(ions_pending_exit_confirmation.keys()):
                if idx in ions_near_this_frame: # It came back
                    ions_pending_exit_confirmation.pop(idx, None)
                    ions_outside_streak.pop(idx, None)
                    ions_currently_inside.add(idx)
                    # Position already added if it's in entered_now or stayed_inside logic handled above
                    if idx not in ions_z_positions_abs or np.isnan(ions_z_positions_abs[idx][local_frame_idx]):
                        ion_pos_idx = np.where(potassium_atoms.indices == idx)[0][0]
                        ions_z_positions_abs[idx][local_frame_idx] = potassium_pos[ion_pos_idx, 2]

    except Exception as e:
        logger.error(f"Error during ion tracking loop: {e}", exc_info=True)
        return None, None, None

    # --- Post-Processing ---
    # Convert frames list to original frame indices from the trajectory
    time_points = frames_to_time(np.array(frame_indices))
    final_tracked_indices = sorted(list(tracked_ion_indices))
    logger.info(f"Identified {len(final_tracked_indices)} unique K+ ions passing through the filter in the analyzed range.")

    # Filter the main dictionary to keep only tracked ions
    ions_z_tracked_abs = {idx: ions_z_positions_abs[idx] for idx in final_tracked_indices}

    # Save position data CSVs using helper
    # We need g1_ref here, but tracking doesn't calculate it. Assume it's calculated
    # elsewhere (e.g., structure.py) and passed to the orchestrator.
    # This function should likely receive g1_ref as an argument.
    # For now, we just return the raw data. The orchestrator will call the save helper.
    # _save_ion_position_csvs(run_dir, time_points, ions_z_tracked_abs, final_tracked_indices, g1_ref_needs_to_be_passed, db_conn, module_name)

    return ions_z_tracked_abs, time_points, final_tracked_indices
