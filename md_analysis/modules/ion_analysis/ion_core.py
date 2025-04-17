"""
Core module for ion tracking in potassium channels.

Provides the main entry point (track_potassium_ions) and coordinates
all other ion analysis functionality.
"""

import os
import logging
import numpy as np
from tqdm import tqdm
import MDAnalysis as mda
from MDAnalysis.analysis import distances
from collections import defaultdict

# Internal imports
from .filter_structure import find_filter_residues, calculate_tvgyg_sites
from .binding_sites import visualize_binding_sites_g1_centric
from .ion_position import save_ion_position_data, plot_ion_positions

# External imports
try:
    from md_analysis.core.utils import frames_to_time
    from md_analysis.core.logging import setup_system_logger
except ImportError as e:
    print(f"Error importing dependency modules in ion_core.py: {e}")
    raise

# Module logger
module_logger = logging.getLogger(__name__)

def track_potassium_ions(run_dir, psf_file=None, dcd_file=None, exit_buffer_frames=5):
    """
    Track K+ ions near the selectivity filter over the trajectory.
    Only tracks ions when they are coordinated by binding sites, with a buffer period
    before considering them "out" of the filter region.

    Args:
        run_dir (str): Directory containing trajectory files and for saving results.
        psf_file (str, optional): Path to PSF file. Defaults to 'step5_input.psf'.
        dcd_file (str, optional): Path to DCD file. Defaults to 'MD_Aligned.dcd'.
        exit_buffer_frames (int, optional): Number of consecutive frames an ion must 
                                           be outside filter before confirming exit.

    Returns:
        tuple: (ions_z_abs, time_points, ion_indices, g1_ref, sites, filter_residues)
               - ions_z_abs (dict): {ion_idx: np.array(abs_z_positions)} with NaN for frames when ion is not in filter
               - time_points (np.ndarray): Time points in ns.
               - ion_indices (list): Indices of tracked ions.
               - g1_ref (float | None): Absolute Z of G1 C-alpha plane.
               - sites (dict | None): Site positions relative to G1 C-alpha=0.
               - filter_residues (dict): Identified filter residues mapping.
               Returns ({}, np.array([0]), [], None, None, None) on critical error.
    """
    # Setup logger for this run
    logger = setup_system_logger(run_dir)
    if logger is None:
        logger = logging.getLogger()  # Fallback

    run_name = os.path.basename(run_dir)
    logger.info(f"Starting K+ ion tracking for {run_name}")

    # --- File Handling & Universe Loading ---
    if psf_file is None:
        psf_file = os.path.join(run_dir, "step5_input.psf")
    if dcd_file is None:
        dcd_file = os.path.join(run_dir, "MD_Aligned.dcd")
    if not os.path.exists(psf_file) or not os.path.exists(dcd_file):
        logger.error(f"PSF or DCD file missing for ion tracking in {run_dir}")
        return {}, np.array([0]), [], None, None, None

    try:
        u = mda.Universe(psf_file, dcd_file)
        n_frames = len(u.trajectory)
        if n_frames == 0:
            logger.warning("Trajectory has 0 frames. Cannot track ions.")
            return {}, np.array([]), [], None, None, None
        logger.info(f"Loaded universe for ion tracking ({n_frames} frames).")
    except Exception as e:
        logger.error(f"Failed to load Universe for ion tracking: {e}", exc_info=True)
        return {}, np.array([0]), [], None, None, None

    # --- Identify Filter and Sites ---
    filter_residues = find_filter_residues(u, logger)
    if not filter_residues:
        logger.error("Failed to identify filter residues. Aborting ion tracking.")
        return {}, frames_to_time(np.arange(n_frames)), [], None, None, None

    filter_sites, g1_reference = calculate_tvgyg_sites(u, filter_residues, logger)
    if not filter_sites or g1_reference is None:
        logger.error("Failed to calculate binding sites or G1 reference. Aborting ion tracking.")
        return {}, frames_to_time(np.arange(n_frames)), [], None, None, filter_residues

    # Create binding site visualization (using G1-centric coordinates)
    _ = visualize_binding_sites_g1_centric(filter_sites, g1_reference, run_dir, logger)

    # --- Setup Ion Selection & Tracking ---
    k_selection_strings = ['name K', 'name POT', 'resname K', 'resname POT', 'type K', 'element K']
    potassium_atoms = None
    for sel in k_selection_strings:
        try:
            atoms = u.select_atoms(sel)
            if len(atoms) > 0:
                potassium_atoms = atoms
                logger.info(f"Found {len(potassium_atoms)} K+ ions using selection: '{sel}'")
                break
        except Exception as e_sel:
            logger.debug(f"Selection '{sel}' failed: {e_sel}")

    if not potassium_atoms:
        logger.error("No potassium ions found using standard selections.")
        return {}, frames_to_time(np.arange(n_frames)), [], g1_reference, filter_sites, filter_residues

    # Create selection for FILTER CARBONYL OXYGEN atoms
    filter_sel_parts = []
    for segid, resids in filter_residues.items():
        if len(resids) == 5:  # Ensure we have TVGYG
            # Select only 'O' atoms for the specific filter resids in this chain
            filter_sel_parts.append(f"(segid {segid} and name O and resid {' '.join(map(str, resids))})")

    if not filter_sel_parts:
        logger.error("Could not build selection string for filter carbonyl oxygens.")
        time_points_err = frames_to_time(np.arange(u.trajectory.n_frames)) if u else np.array([0])
        return {}, time_points_err, [], g1_reference, filter_sites, filter_residues

    filter_selection_str = " or ".join(filter_sel_parts)
    logger.info(f"Using filter CARBONYL OXYGEN selection: {filter_selection_str}")

    filter_atoms_group = u.select_atoms(filter_selection_str)  # Select the group once
    if not filter_atoms_group:
        logger.error("Could not select filter carbonyl oxygens! Aborting ion tracking.")
        time_points_err = frames_to_time(np.arange(u.trajectory.n_frames)) if u else np.array([0])
        return {}, time_points_err, [], g1_reference, filter_sites, filter_residues

    # Variables for tracking
    frame_indices = []
    # Initialize numpy arrays filled with NaN for all ions (we'll only populate when ions are in filter)
    ions_z_positions_abs = {ion.index: np.full(n_frames, np.nan) for ion in potassium_atoms}
    
    # Track state for each ion
    ions_currently_inside = set()  # Set of ion indices currently in filter
    ions_outside_streak = defaultdict(int)  # Track consecutive frames outside filter
    ions_pending_exit_confirmation = {}  # Last frame ion was seen in filter before potential exit
    tracked_ion_indices = set()  # Set of all ions that ever entered the filter
    
    cutoff = 3.5  # Distance cutoff for coordination
    logger.info(f"Tracking K+ ions within {cutoff} Å of filter atoms, applying {exit_buffer_frames} frame exit buffer.")

    # --- Trajectory Iteration ---
    for ts in tqdm(u.trajectory, desc=f"Tracking K+ ({run_name})", unit="frame"):
        frame_idx = ts.frame
        frame_indices.append(frame_idx)
        
        # Get current positions
        filter_atoms_pos = filter_atoms_group.positions
        potassium_pos = potassium_atoms.positions
        box = u.dimensions  # Use box dimensions for distance calculation

        # Calculate distances between all K+ ions and all filter atoms
        dist_matrix = distances.distance_array(potassium_pos, filter_atoms_pos, box=box)

        # Find minimum distance for each K+ ion to any filter atom
        min_dists = np.min(dist_matrix, axis=1)

        # Identify ions currently near the filter
        ions_near_this_frame = set(potassium_atoms[min_dists <= cutoff].indices)
        
        # --- State Transition Logic (with Exit Buffer) ---
        # Determine ions that just entered, just exited, or stayed outside
        if frame_idx > 0:
            # Ions that were inside previous frame but exited this frame
            exited_now = ions_currently_inside - ions_near_this_frame
            # Ions that entered the filter this frame
            entered_now = ions_near_this_frame - ions_currently_inside
            # Ions that were outside and remained outside
            stayed_outside = set(potassium_atoms.indices) - ions_near_this_frame - ions_currently_inside
        else:
            # First frame
            exited_now = set()
            entered_now = ions_near_this_frame
            stayed_outside = set(potassium_atoms.indices) - ions_near_this_frame
        
        # 1. Handle ions entering now
        for idx in entered_now:
            tracked_ion_indices.add(idx)  # Add to the set of all tracked ions
            ions_currently_inside.add(idx)
            # Record Z position for this ion
            ion_pos_idx = np.where(potassium_atoms.indices == idx)[0][0]
            ions_z_positions_abs[idx][frame_idx] = potassium_pos[ion_pos_idx, 2]
            # Clear any outside streak or pending exit
            if idx in ions_outside_streak:
                del ions_outside_streak[idx]
            if idx in ions_pending_exit_confirmation:
                del ions_pending_exit_confirmation[idx]
        
        # 2. Handle ions still inside filter
        for idx in ions_currently_inside - exited_now:
            # Record Z position
            ion_pos_idx = np.where(potassium_atoms.indices == idx)[0][0]
            ions_z_positions_abs[idx][frame_idx] = potassium_pos[ion_pos_idx, 2]
        
        # 3. Handle ions exiting now
        for idx in exited_now:
            if idx not in ions_pending_exit_confirmation:
                ions_pending_exit_confirmation[idx] = frame_idx - 1  # Record last frame it was in
            ions_outside_streak[idx] = 1  # Start outside streak
            ions_currently_inside.remove(idx)  # Remove from inside set
        
        # 4. Handle ions staying outside
        for idx in stayed_outside:
            if idx in ions_outside_streak:
                ions_outside_streak[idx] += 1
                # Check if exit is confirmed (outside for > buffer frames)
                if idx in ions_pending_exit_confirmation and ions_outside_streak[idx] > exit_buffer_frames:
                    # Exit confirmed, clean up tracking state
                    del ions_pending_exit_confirmation[idx]
                    # Note: Keep outside streak for tracking purposes
            elif idx in ions_pending_exit_confirmation:
                # This shouldn't happen normally, but handle edge case
                ions_outside_streak[idx] = 1
        
        # 5. Handle ions in exit confirmation window (they stay in 'currently_inside')
        for idx in list(ions_pending_exit_confirmation.keys()):
            if idx not in exited_now and idx not in stayed_outside:
                # Ion returned to filter during confirmation window
                del ions_pending_exit_confirmation[idx]
                if idx in ions_outside_streak:
                    del ions_outside_streak[idx]
                # Make sure it's tracked as inside
                ions_currently_inside.add(idx)
                # Update position
                ion_pos_idx = np.where(potassium_atoms.indices == idx)[0][0]
                ions_z_positions_abs[idx][frame_idx] = potassium_pos[ion_pos_idx, 2]

    # --- Post-Processing & Clean-up ---
    time_points = frames_to_time(frame_indices)
    final_tracked_indices = sorted(list(tracked_ion_indices))
    logger.info(f"Identified {len(final_tracked_indices)} unique K+ ions passing through the filter.")

    # Filter the main dictionary to keep only tracked ions
    ions_z_tracked_abs = {idx: ions_z_positions_abs[idx] for idx in final_tracked_indices}

    # Save position data (Absolute and G1-centric)
    save_ion_position_data(run_dir, time_points, ions_z_tracked_abs, final_tracked_indices, g1_reference)

    # Plot positions (using G1-centric coordinates)
    plot_ion_positions(run_dir, time_points, ions_z_tracked_abs, final_tracked_indices, filter_sites, g1_reference, logger=logger)

    return ions_z_tracked_abs, time_points, final_tracked_indices, g1_reference, filter_sites, filter_residues
