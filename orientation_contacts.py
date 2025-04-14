# orientation_contacts.py
"""
Functions for analyzing toxin orientation relative to the channel,
toxin rotation over time, and contacts (atom and residue) at the
toxin-channel interface.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import MDAnalysis as mda
from MDAnalysis.analysis import distances
import math

# Import from other modules
try:
    from utils import OneLetter, frames_to_time
    # from config import FRAMES_PER_NS # Not directly needed if using frames_to_time
except ImportError as e:
    print(f"Error importing dependency modules in orientation_contacts.py: {e}")
    raise

# Configure matplotlib for non-interactive backend
plt.switch_backend('agg')
# Set a consistent plot style (optional)
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.2)

# Get a logger for this module
logger = logging.getLogger(__name__)

# --- Helper Functions ---

def rotation_matrix_to_euler(R):
    """
    Convert a 3x3 rotation matrix to Euler angles (XYZ sequence, degrees).

    Handles singularity cases.

    Parameters:
    -----------
    R : numpy.ndarray (3x3)
        Rotation matrix.

    Returns:
    --------
    tuple
        (angle_x, angle_y, angle_z) in degrees.
    """
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        # Gimbal lock: set z arbitrarily to 0
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0.0
        logger.debug("Euler angle singularity detected.")

    return np.degrees(x), np.degrees(y), np.degrees(z)

def calculate_orientation_vectors(u, toxin_sel, channel_sel, prev_toxin_axis=None, prev_channel_axis=None):
    """
    Calculate principal axis vectors for toxin and channel for the current frame.
    Ensures consistent vector direction relative to the previous frame.

    Args:
        u (MDAnalysis.Universe): Universe object at the current timestep.
        toxin_sel (str): Selection string for the toxin.
        channel_sel (str): Selection string for the channel.
        prev_toxin_axis (np.ndarray, optional): Toxin axis from the previous frame.
        prev_channel_axis (np.ndarray, optional): Channel axis from the previous frame.

    Returns:
        tuple: (toxin_axis, channel_axis, angle, current_toxin_axis, current_channel_axis)
               - toxin_axis: Principal axis vector for the toxin (direction-checked).
               - channel_axis: Principal axis vector for the channel (direction-checked).
               - angle: Angle (degrees) between the two principal axes (0-90).
               - current_toxin_axis: Raw toxin axis (before direction check, for state update).
               - current_channel_axis: Raw channel axis (before direction check, for state update).
               Returns (None, None, np.nan, None, None) on error.
    """
    try:
        toxin = u.select_atoms(toxin_sel)
        channel = u.select_atoms(channel_sel)
        if not toxin or not channel:
            logger.warning("Toxin or Channel selection empty in calculate_orientation_vectors.")
            return None, None, np.nan, None, None

        # Calculate principal axes using SVD on centered coordinates
        toxin_coords = toxin.positions - toxin.center_of_mass()
        u_t, _, _ = np.linalg.svd(toxin_coords.T, full_matrices=False)
        current_toxin_axis = u_t[:, 0] # Principal axis (first left singular vector)

        channel_coords = channel.positions - channel.center_of_mass()
        u_c, _, _ = np.linalg.svd(channel_coords.T, full_matrices=False)
        current_channel_axis = u_c[:, 0] # Principal axis

        # --- Ensure consistent direction of vectors between frames ---
        checked_toxin_axis = np.copy(current_toxin_axis)
        if prev_toxin_axis is not None:
            if np.dot(checked_toxin_axis, prev_toxin_axis) < 0:
                checked_toxin_axis *= -1 # Flip direction

        checked_channel_axis = np.copy(current_channel_axis)
        if prev_channel_axis is not None:
            if np.dot(checked_channel_axis, prev_channel_axis) < 0:
                checked_channel_axis *= -1 # Flip direction

        # Calculate angle between the direction-checked principal axes
        # Use absolute dot product for angle between lines (0-90 degrees)
        dot_product = np.clip(np.abs(np.dot(checked_toxin_axis, checked_channel_axis)), 0.0, 1.0)
        angle = np.degrees(np.arccos(dot_product))

        # Return direction-checked axes for angle calc, and raw axes for state update
        return checked_toxin_axis, checked_channel_axis, angle, current_toxin_axis, current_channel_axis

    except Exception as e:
        logger.warning(f"Error calculating orientation vectors: {e}", exc_info=True)
        return None, None, np.nan, None, None


def calculate_rotation_matrix(reference_coords, current_coords):
    """
    Calculate the optimal rotation matrix to align current_coords onto reference_coords
    using Kabsch algorithm (SVD).

    Assumes coordinates are already centered.

    Parameters:
    -----------
    reference_coords : numpy.ndarray (N, 3)
        Centered reference coordinates.
    current_coords : numpy.ndarray (N, 3)
        Centered current coordinates.

    Returns:
    --------
    tuple:
        - R (numpy.ndarray): 3x3 rotation matrix.
        - euler_angles (tuple): (angle_x, angle_y, angle_z) in degrees.
        Returns (None, (nan, nan, nan)) on error.
    """
    if reference_coords.shape != current_coords.shape:
         logger.error("Coordinate shapes mismatch in calculate_rotation_matrix.")
         return None, (np.nan, np.nan, np.nan)
    try:
        # Calculate covariance matrix H = P^T * Q
        H = np.dot(reference_coords.T, current_coords)

        # SVD decomposition: H = U * S * Vt
        U, S, Vt = np.linalg.svd(H)

        # Calculate rotation matrix R = V * U^T
        # Check for reflection correction
        d = np.linalg.det(np.dot(Vt.T, U.T))
        if d < 0:
            # logger.debug("Reflection detected, correcting rotation matrix.")
            Vt[2, :] *= -1 # Modify V matrix before calculation

        R = np.dot(Vt.T, U.T)

        # Extract Euler angles from rotation matrix
        euler_angles = rotation_matrix_to_euler(R)
        return R, euler_angles

    except Exception as e:
        logger.error(f"Error calculating rotation matrix: {e}", exc_info=True)
        return None, (np.nan, np.nan, np.nan)


def analyze_interface_contacts(u, toxin_sel, channel_sel, cutoff=3.5):
    """
    Calculates atom contacts and residue contacts for a single frame.

    Parameters:
    -----------
    u : MDAnalysis.Universe
        Universe object (at a specific timestep).
    toxin_sel : str
        Selection string for the toxin.
    channel_sel : str
        Selection string for the channel.
    cutoff : float, optional
        Distance cutoff for defining a contact (in Angstroms). Default is 3.5.

    Returns:
    --------
    tuple:
        - total_atom_contacts (int): Total number of atom pairs within cutoff.
        - residue_contacts_frame (set): A set of tuples `(toxin_res_idx, channel_res_idx)`
                                          representing residues in contact in this frame.
                                          Indices refer to the order in toxin_residues and channel_residues.
        - toxin_residues (MDAnalysis.ResidueGroup): Residue group for the toxin selection.
        - channel_residues (MDAnalysis.ResidueGroup): Residue group for the channel selection.
        Returns (0, set(), None, None) on error or empty selections.
    """
    try:
        toxin = u.select_atoms(toxin_sel)
        channel = u.select_atoms(channel_sel)

        # Get residues early, even if atoms might be empty later
        toxin_residues = toxin.residues
        channel_residues = channel.residues

        if len(toxin) == 0 or len(channel) == 0:
            logger.warning("Toxin or Channel selection empty in analyze_interface_contacts.")
            return 0, set(), toxin_residues, channel_residues

        if len(toxin_residues) == 0 or len(channel_residues) == 0:
            logger.warning("Toxin or Channel selection yielded no residues.")
            return 0, set(), toxin_residues, channel_residues

        # Use distance_array with box to handle PBC correctly
        box = u.dimensions if hasattr(u, 'dimensions') and u.dimensions is not None else None
        dist_array = distances.distance_array(
            toxin.positions,
            channel.positions,
            box=box
        )

        # Atom-level contact map (boolean)
        atom_contact_map = dist_array < cutoff
        total_atom_contacts = int(np.sum(atom_contact_map)) # Ensure int

        # --- Calculate Residue-Level Contacts ---
        residue_contacts_frame = set()

        # Efficiently find which residue pairs have contacts
        if total_atom_contacts > 0:
            # Map global atom index to local *residue* index (relative to the selection's residues)
            # We need this mapping ONCE before the loop for efficiency if possible,
            # but since it's per frame, it's okay here.
            toxin_atom_to_res_idx = {atom.ix: i for i, res in enumerate(toxin_residues) for atom in res.atoms}
            channel_atom_to_res_idx = {atom.ix: i for i, res in enumerate(channel_residues) for atom in res.atoms}

            # Find indices (relative to selections) where atom contacts occur
            contacting_atom_idx_pairs = np.argwhere(atom_contact_map) # Returns pairs [toxin_idx, channel_idx]

            # Map contacting atom indices back to local residue indices and add to set
            for t_atom_local_idx, c_atom_local_idx in contacting_atom_idx_pairs:
                t_atom_global_ix = toxin.indices[t_atom_local_idx] # Get global index
                c_atom_global_ix = channel.indices[c_atom_local_idx]

                # Ensure atom indices are in our mapping (they should be)
                if t_atom_global_ix in toxin_atom_to_res_idx and c_atom_global_ix in channel_atom_to_res_idx:
                    t_res_idx = toxin_atom_to_res_idx[t_atom_global_ix]
                    c_res_idx = channel_atom_to_res_idx[c_atom_global_ix]
                    residue_contacts_frame.add((t_res_idx, c_res_idx))
                else:
                     # This case indicates a potential logic error in mapping
                     logger.warning(f"Atom index mismatch: Toxin {t_atom_global_ix} or Channel {c_atom_global_ix} not found in residue maps.")

        return total_atom_contacts, residue_contacts_frame, toxin_residues, channel_residues

    except Exception as e:
         logger.error(f"Error analyzing interface contacts: {e}", exc_info=True)
         # Attempt to return residue groups even on error, they might be valid
         try:
             toxin_res = u.select_atoms(toxin_sel).residues
             channel_res = u.select_atoms(channel_sel).residues
             return 0, set(), toxin_res, channel_res
         except:
             return 0, set(), None, None # Fallback


def get_residue_info(universe, toxin_sel, channel_sel):
    """
    Extract residue information mapping existing SegID_ResID labels to
    specific descriptive labels (e.g., "GLY77(A)", "LYS12").
    """
    # (Implementation copied from the original script's get_residue_info)
    # ... (ensure logging uses logger = logging.getLogger(__name__)) ...
    func_logger = logging.getLogger(__name__) # Use logger defined at module level
    func_logger.setLevel(logging.DEBUG) # Ensure debug logs are shown

    segid_to_subunit = {
        'PROA': 'A', 'PROB': 'B', 'PROC': 'C', 'PROD': 'D',
        'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D'
    }
    all_segids = np.unique(universe.atoms.segids)
    func_logger.debug(f"ALL SEGIDS IN UNIVERSE: {all_segids}")

    channel_label_map = {}
    toxin_label_map = {}

    try:
        # Process Channel Residues
        channel_atoms = universe.select_atoms(channel_sel)
        if not channel_atoms:
            func_logger.warning(f"Channel selection ('{channel_sel}') yielded no atoms.")
        else:
            unique_channel_residues = {}
            for res in channel_atoms.residues:
                if res.segid not in unique_channel_residues:
                    unique_channel_residues[res.segid] = set()
                unique_channel_residues[res.segid].add(res) # Store the residue object itself

            func_logger.debug(f"CHANNEL SEGIDS FOUND: {list(unique_channel_residues.keys())}")

            for segid, res_set in unique_channel_residues.items():
                subunit_char = segid_to_subunit.get(segid, segid[0]) # Default to first char if not mapped
                func_logger.debug(f"Processing channel segid {segid} -> subunit '{subunit_char}'")

                # Sort residues by resid to ensure consistent label order if needed later
                sorted_res_list = sorted(list(res_set), key=lambda r: r.resid)

                for res in sorted_res_list:
                    key = f"{subunit_char}_{res.resid}"
                    value = f"{res.resname}{res.resid}({subunit_char})"
                    channel_label_map[key] = value
                    if len(channel_label_map) <= 5 or len(channel_label_map) % 100 == 0:
                         func_logger.debug(f"  CHANNEL MAP: {key} -> {value}")

            func_logger.info(f"Generated channel label map with {len(channel_label_map)} entries")

        # Process Toxin Residues
        toxin_atoms = universe.select_atoms(toxin_sel)
        if not toxin_atoms:
            func_logger.warning(f"Toxin selection ('{toxin_sel}') yielded no atoms.")
        else:
            toxin_segids = np.unique(toxin_atoms.segids)
            toxin_segid = toxin_segids[0] if len(toxin_segids) > 0 else "E" # Default 'E'
            toxin_seg_char = toxin_segid[0] if toxin_segid else "E" # Use first char
            func_logger.debug(f"Processing toxin segid {toxin_segid} -> char '{toxin_seg_char}'")

            # Sort residues by resid
            sorted_toxin_res = sorted(list(toxin_atoms.residues), key=lambda r: r.resid)

            for res in sorted_toxin_res:
                key = f"{toxin_seg_char}_{res.resid}"
                value = f"{res.resname}{res.resid}"
                toxin_label_map[key] = value
                if len(toxin_label_map) <= 5 or len(toxin_label_map) % 20 == 0:
                     func_logger.debug(f"  TOXIN MAP: {key} -> {value}")
            func_logger.info(f"Generated toxin label map with {len(toxin_label_map)} entries")

    except Exception as e:
        func_logger.error(f"Error extracting residue information: {e}", exc_info=True)
        return {}, {} # Return empty maps on error

    return channel_label_map, toxin_label_map


# --- Main Analysis Function for this Module ---

def analyze_toxin_orientation(traj_file, psf_file, output_dir, stride=None):
    """
    Analyze toxin orientation relative to channel, toxin rotation relative to
    start frame, and average residue contact frequencies over a trajectory.

    Saves orientation data (including total atom contacts) and residue contact
    frequency map (using descriptive labels) to CSV files. Generates contact
    map visualizations with improved labels.

    Args:
        traj_file (str): Path to the DCD trajectory file.
        psf_file (str): Path to the PSF topology file.
        output_dir (str): Directory to save CSV files and plots.
        stride (int, optional): Frame stride for analysis. If None, analyzes
                                max ~500 frames for contact map.

    Returns:
        tuple:
            - orientation_angles (list): List of angles between principal axes.
            - rotation_euler_angles (list): List of (x, y, z) Euler angles for toxin rotation.
            - total_contact_counts_list (list): List of total atom contacts per analyzed frame.
            - residue_contact_df_display (pd.DataFrame | None): DataFrame of average residue
              contact frequencies with display labels, or None if analysis failed.
            Returns ([], [], [], None) on critical error.
    """
    logger.info(f"Starting toxin orientation and contact analysis for: {traj_file}")
    os.makedirs(output_dir, exist_ok=True) # Ensure output dir exists

    try:
        u = mda.Universe(psf_file, traj_file)
        n_frames = len(u.trajectory)
        if n_frames < 2:
            logger.warning("Trajectory has < 2 frames. Orientation/contact analysis requires at least 2 frames.")
            return [], [], [], None
    except Exception as e:
        logger.error(f"Failed to load Universe: {e}", exc_info=True)
        return [], [], [], None

    # --- Selections ---
    toxin_sel = 'segid PROE'
    channel_sel = 'segid PROA or segid PROB or segid PROC or segid PROD'
    # Fallback logic (simplified from original, could be enhanced)
    if len(u.select_atoms(toxin_sel)) == 0:
        for alt_sel in ['segid E', 'segid PEP', 'segid TOX']:
            if len(u.select_atoms(alt_sel)) > 0: toxin_sel = alt_sel; break
    if len(u.select_atoms(channel_sel)) == 0:
        if len(u.select_atoms('segid A or segid B or segid C or segid D')) > 0:
            channel_sel = 'segid A or segid B or segid C or segid D'

    toxin_atoms = u.select_atoms(toxin_sel)
    channel_atoms = u.select_atoms(channel_sel)

    if not toxin_atoms or not channel_atoms:
        logger.error(f"Could not select toxin ('{toxin_sel}') or channel ('{channel_sel}') atoms. Aborting orientation analysis.")
        return [], [], [], None
    logger.info(f"Using Toxin: '{toxin_sel}', Channel: '{channel_sel}'")

    # --- Initialization ---
    orientation_angles = []
    rotation_euler_angles = [] # Store tuples of (x,y,z) angles
    total_contact_counts_list = []

    # Residue contact analysis setup
    toxin_residues = toxin_atoms.residues
    channel_residues = channel_atoms.residues
    n_toxin_res = len(toxin_residues)
    n_channel_res = len(channel_residues)
    residue_contact_accumulator = None
    if n_toxin_res > 0 and n_channel_res > 0:
        residue_contact_accumulator = np.zeros((n_toxin_res, n_channel_res), dtype=int)
        logger.info(f"Initializing residue contact map ({n_toxin_res} toxin x {n_channel_res} channel residues).")
    else:
        logger.warning("Toxin or Channel selection has zero residues. Skipping residue contact analysis.")

    # Get reference structure (first frame) for rotation calculation
    try:
        u.trajectory[0]
        # Select within the first frame context
        ref_toxin_atoms = u.select_atoms(toxin_sel)
        if not ref_toxin_atoms: raise ValueError("Failed to select toxin in first frame for reference.")
        reference_coords_centered = ref_toxin_atoms.positions - ref_toxin_atoms.center_of_mass()
    except Exception as e:
        logger.error(f"Error getting reference coordinates: {e}", exc_info=True)
        return [], [], [], None

    # --- Trajectory Processing ---
    # Determine stride
    if stride is None:
        stride = max(1, n_frames // 500) # Analyze max ~500 frames if not specified
    logger.info(f"Using stride: {stride} (Analyzing ~{n_frames // stride} frames)")

    processed_frames_indices = []
    analyzed_frame_count = 0

    # Initialize state for orientation vector direction check (NO GLOBALS)
    prev_toxin_axis_state = None
    prev_channel_axis_state = None

    # Loop through trajectory with progress tracking
    frame_iter = u.trajectory[::stride]
    for i, ts in enumerate(frame_iter):
        frame_idx = ts.frame # Actual frame index
        processed_frames_indices.append(frame_idx)
        analyzed_frame_count += 1

        # Periodically log progress
        if analyzed_frame_count % 100 == 0:
             logger.debug(f"Processing frame {frame_idx} ({analyzed_frame_count} analyzed)...")

        # 1. Calculate Orientation Angle
        try:
            # Pass previous state, get back results and new state
            _, _, angle, current_toxin_axis, current_channel_axis = calculate_orientation_vectors(
                u, toxin_sel, channel_sel, prev_toxin_axis_state, prev_channel_axis_state
            )
            orientation_angles.append(angle)
            # Update state for the next iteration
            prev_toxin_axis_state = current_toxin_axis
            prev_channel_axis_state = current_channel_axis
        except Exception as e:
            logger.warning(f"Error calculating orientation vectors at frame {frame_idx}: {e}", exc_info=False)
            orientation_angles.append(np.nan)
            prev_toxin_axis_state = None # Reset state on error
            prev_channel_axis_state = None

        # 2. Calculate Rotation Relative to Reference
        try:
            toxin_current = u.select_atoms(toxin_sel)
            # Ensure same number of atoms as reference for rotation calculation
            if len(toxin_current) != len(ref_toxin_atoms):
                logger.warning(f"Atom count mismatch for rotation at frame {frame_idx}. Skipping rotation calc.")
                rotation_euler_angles.append((np.nan, np.nan, np.nan))
            else:
                current_coords_centered = toxin_current.positions - toxin_current.center_of_mass()
                _, euler = calculate_rotation_matrix(reference_coords_centered, current_coords_centered)
                rotation_euler_angles.append(euler)
        except Exception as e:
            logger.warning(f"Error calculating rotation matrix at frame {frame_idx}: {e}", exc_info=False)
            rotation_euler_angles.append((np.nan, np.nan, np.nan))

        # 3. Calculate Contacts (Atom and Residue)
        try:
            # analyze_interface_contacts returns: total_atoms, res_set, toxin_res_group, channel_res_group
            # We only need the first two outputs here, groups are already defined outside loop
            total_atom_contacts, residue_contacts_frame, _, _ = analyze_interface_contacts(
                u, toxin_sel, channel_sel
            )
            total_contact_counts_list.append(total_atom_contacts)

            # Accumulate residue contacts
            if residue_contact_accumulator is not None:
                for t_res_idx, c_res_idx in residue_contacts_frame:
                    # Check bounds strictly
                    if 0 <= t_res_idx < n_toxin_res and 0 <= c_res_idx < n_channel_res:
                        residue_contact_accumulator[t_res_idx, c_res_idx] += 1
                    else: # Should not happen if mapping is correct
                        logger.warning(f"Residue index out of bounds at frame {frame_idx}: T{t_res_idx} (max {n_toxin_res-1}), C{c_res_idx} (max {n_channel_res-1})")

        except Exception as e:
            logger.warning(f"Error analyzing contacts at frame {frame_idx}: {e}", exc_info=False)
            total_contact_counts_list.append(np.nan)

    # --- Post-Processing & Saving ---
    if analyzed_frame_count == 0:
        logger.error("No frames were analyzed in orientation/contact analysis.")
        return [], [], [], None

    time_points_analysis = frames_to_time(processed_frames_indices) # Use utility function

    # Save Orientation Data CSV
    orientation_df = pd.DataFrame({
        'Frame': processed_frames_indices,
        'Time (ns)': time_points_analysis,
        'Orientation_Angle': orientation_angles,
        'Rotation_X': [r[0] for r in rotation_euler_angles],
        'Rotation_Y': [r[1] for r in rotation_euler_angles],
        'Rotation_Z': [r[2] for r in rotation_euler_angles],
        'Total_Atom_Contacts': total_contact_counts_list
    })
    orientation_csv_path = os.path.join(output_dir, "Toxin_Orientation.csv")
    try:
        orientation_df.to_csv(orientation_csv_path, index=False, float_format='%.3f', na_rep='NaN')
        logger.info(f"Saved orientation and total contacts data to {orientation_csv_path}")
    except Exception as e:
        logger.error(f"Failed to save {orientation_csv_path}: {e}")

    # --- Save Residue Contact Frequency Data & Plots ---
    residue_contact_df_display = None # Initialize return variable
    if residue_contact_accumulator is not None and analyzed_frame_count > 0:
        avg_residue_contact_map = residue_contact_accumulator / analyzed_frame_count

        # --- Generate consistent labels for internal use (matching get_residue_info keys) ---
        # Define the mapping here, consistent with get_residue_info
        segid_to_subunit = {
            'PROA': 'A', 'PROB': 'B', 'PROC': 'C', 'PROD': 'D',
            'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D'
            # Add other mappings if necessary based on your PSF
        }
        logger.debug(f"Using segid->subunit mapping: {segid_to_subunit}")

        # Generate Toxin Labels (e.g., "E_77") - Internal Keys
        toxin_res_labels_internal = []
        unique_toxin_segids = np.unique(toxin_residues.segids) if len(toxin_residues.segids) > 0 else ["E"] # Default E
        toxin_segid_base = unique_toxin_segids[0] if unique_toxin_segids else "E"
        toxin_seg_char = toxin_segid_base[0] if toxin_segid_base else "E" # First char of first segid
        logger.debug(f"Using '{toxin_seg_char}' as base for toxin internal labels.")
        for i, res in enumerate(toxin_residues):
             key = f"{toxin_seg_char}_{res.resid}"
             if key in toxin_res_labels_internal: # Check for duplicates based on resid+segchar
                 logger.warning(f"Duplicate internal toxin label generated: {key} for residue index {i}. Check selection/PSF.")
             toxin_res_labels_internal.append(key)

        # Generate Channel Labels (e.g., "A_68", "B_68") - Internal Keys
        channel_res_labels_internal = []
        for i, res in enumerate(channel_residues):
             subunit_char = segid_to_subunit.get(res.segid, res.segid[0] if res.segid else 'X')
             key = f"{subunit_char}_{res.resid}"
             # Note: Duplicates like A_68, B_68 are expected here and handled by order.
             channel_res_labels_internal.append(key)

        # Basic sanity check: Ensure label count matches matrix dimensions
        if len(toxin_res_labels_internal) != avg_residue_contact_map.shape[0]: # CORRECTED VARIABLE NAME
             logger.error(f"FATAL: Toxin label count ({len(toxin_res_labels_internal)}) mismatch with contact map rows ({avg_residue_contact_map.shape[0]})") # CORRECTED VARIABLE NAME
             return orientation_angles, rotation_euler_angles, total_contact_counts_list, None # Return early
        if len(channel_res_labels_internal) != avg_residue_contact_map.shape[1]: # CORRECTED VARIABLE NAME
             logger.error(f"FATAL: Channel label count ({len(channel_res_labels_internal)}) mismatch with contact map columns ({avg_residue_contact_map.shape[1]})") # CORRECTED VARIABLE NAME
             return orientation_angles, rotation_euler_angles, total_contact_counts_list, None # Return early

        # --- Get the Display Labels from get_residue_info ---
        # Make sure the Universe `u` is at a state where selections work (e.g., first frame)
        try:
            u.trajectory[0] # Go to first frame to ensure selections are valid for get_residue_info
            channel_label_map, toxin_label_map = get_residue_info(u, toxin_sel, channel_sel)
        except Exception as e_labelmap:
            logger.error(f"Error calling get_residue_info: {e_labelmap}. Proceeding with basic labels.", exc_info=True)
            channel_label_map, toxin_label_map = {}, {} # Ensure maps are empty dicts

        if not channel_label_map or not toxin_label_map:
             logger.warning("Failed to generate residue label maps via get_residue_info. Plot labels and CSV will use basic internal keys.")
             # Create fallback maps using internal labels
             toxin_label_map = {lbl: lbl for lbl in toxin_res_labels_internal}
             channel_label_map = {lbl: lbl for lbl in channel_res_labels_internal}


        # --- Save CSV with Human-Readable Display Labels ---
        try:
            # Map internal indices/labels to display labels for the DataFrame
            # Use the maps obtained from get_residue_info or the fallback
            display_toxin_labels = [toxin_label_map.get(k, k) for k in toxin_res_labels_internal]
            display_channel_labels = [channel_label_map.get(k, k) for k in channel_res_labels_internal]

            residue_contact_df_display = pd.DataFrame(avg_residue_contact_map,
                                                    index=display_toxin_labels,
                                                    columns=display_channel_labels)
            # Round for readability
            residue_contact_df_display = residue_contact_df_display.round(4)

            # Save the DataFrame with display labels
            contacts_csv_path = os.path.join(output_dir, "Residue_Contact_Frequency.csv")
            residue_contact_df_display.to_csv(contacts_csv_path, index=True, index_label="Toxin_Residue", float_format='%.4f')
            logger.info(f"Saved average residue contact frequency map to {contacts_csv_path}")

        except Exception as e_csv:
            logger.error(f"Error saving residue contact frequency CSV: {e_csv}", exc_info=True)
            residue_contact_df_display = None # Ensure it's None if save failed

        # --- Generate Plots ---
        try:
            # Generate the full heatmap
            create_contact_map_visualization(avg_residue_contact_map,
                                         toxin_res_labels_internal,
                                         channel_res_labels_internal,
                                         output_dir,
                                         toxin_label_map, channel_label_map) # Pass the maps

            # Generate the focused heatmap
            create_enhanced_focused_heatmap(avg_residue_contact_map,
                                          toxin_res_labels_internal,
                                          channel_res_labels_internal,
                                          toxin_label_map, channel_label_map, # Pass the maps
                                          output_dir)

            # Generate time-series plots for orientation, rotation, and contacts
            plot_orientation_data(time_points_analysis,
                                  orientation_angles,
                                  rotation_euler_angles,
                                  total_contact_counts_list,
                                  output_dir)
            logger.info(f"Generated and saved contact maps and orientation plots in {output_dir}")
        except Exception as e_plot:
            logger.error(f"Error generating contact maps or orientation plots: {e_plot}", exc_info=True)

    else:
        logger.info("Residue contact analysis was skipped or produced no results.")

    # After the main loop:
    rotation_stats = {}
    if rotation_euler_angles: # Check if list is not empty
        # Separate components, handling potential NaNs
        rot_x = np.array([r[0] for r in rotation_euler_angles])
        rot_y = np.array([r[1] for r in rotation_euler_angles])
        rot_z = np.array([r[2] for r in rotation_euler_angles])

        rotation_stats['Orient_RotX_Mean'] = np.nanmean(rot_x)
        rotation_stats['Orient_RotX_Std'] = np.nanstd(rot_x)
        rotation_stats['Orient_RotY_Mean'] = np.nanmean(rot_y)
        rotation_stats['Orient_RotY_Std'] = np.nanstd(rot_y)
        rotation_stats['Orient_RotZ_Mean'] = np.nanmean(rot_z)
        rotation_stats['Orient_RotZ_Std'] = np.nanstd(rot_z)
    else: # Handle empty list case
        keys = ['Orient_RotX_Mean', 'Orient_RotX_Std', 'Orient_RotY_Mean', 'Orient_RotY_Std', 'Orient_RotZ_Mean', 'Orient_RotZ_Std']
        for k in keys: rotation_stats[k] = np.nan

    logger.info("Finished toxin orientation and contact analysis.")
    return orientation_angles, rotation_euler_angles, total_contact_counts_list, residue_contact_df_display, rotation_stats


# --- Plotting Functions for this Module ---

def plot_orientation_data(time_points, orientation_angles, rotation_euler_angles, contact_counts, output_dir):
    """Create plots for toxin orientation angle, rotation, and total contacts."""
    # Ensure input lists/arrays are not empty
    if len(time_points) == 0:
        logger.warning("No time points available for orientation plotting.")
        return

    # --- Orientation angle plot ---
    try:
        fig, ax = plt.subplots(figsize=(10, 5)) # Slightly smaller height
        sns.lineplot(x=time_points, y=orientation_angles, ax=ax, linewidth=1.5)
        ax.set_xlabel('Time (ns)', fontsize=12)
        ax.set_ylabel('Toxin-Channel Angle (°)', fontsize=12)
        ax.set_title('Toxin Orientation Angle vs. Channel Axis', fontsize=14)
        ax.tick_params(axis='both', labelsize=10)
        # Optional: Set Y limits (e.g., 0-90 degrees)
        ax.set_ylim(0, 90)
        ax.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, "Toxin_Orientation_Angle.png"), dpi=150)
        plt.close(fig)
    except Exception as e:
         logger.error(f"Failed to create orientation angle plot: {e}", exc_info=True)

    # --- Rotation components plot ---
    try:
        fig, ax = plt.subplots(figsize=(10, 5))
        if rotation_euler_angles and len(rotation_euler_angles) == len(time_points):
            x_rot = [r[0] for r in rotation_euler_angles]
            y_rot = [r[1] for r in rotation_euler_angles]
            z_rot = [r[2] for r in rotation_euler_angles]
            sns.lineplot(x=time_points, y=x_rot, label='X rotation', ax=ax, linewidth=1.0)
            sns.lineplot(x=time_points, y=y_rot, label='Y rotation', ax=ax, linewidth=1.0)
            sns.lineplot(x=time_points, y=z_rot, label='Z rotation', ax=ax, linewidth=1.0)
            ax.legend(fontsize='small')
        else:
             ax.text(0.5, 0.5, 'Rotation data unavailable or length mismatch.', ha='center', va='center')

        ax.set_xlabel('Time (ns)', fontsize=12)
        ax.set_ylabel('Rotation (°)', fontsize=12)
        ax.set_title('Toxin Rotation Relative to Start Frame', fontsize=14)
        ax.tick_params(axis='both', labelsize=10)
        ax.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, "Toxin_Rotation_Components.png"), dpi=150)
        plt.close(fig)
    except Exception as e:
         logger.error(f"Failed to create rotation components plot: {e}", exc_info=True)

    # --- Contact count plot ---
    try:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(x=time_points, y=contact_counts, ax=ax, linewidth=1.5)
        ax.set_xlabel('Time (ns)', fontsize=12)
        ax.set_ylabel('Number of Atom Contacts', fontsize=12)
        ax.set_title('Toxin-Channel Interface Atom Contacts (< 3.5 Å)', fontsize=14)
        ax.tick_params(axis='both', labelsize=10)
        ax.grid(True, linestyle=':', alpha=0.6)
        # Set lower y-limit to 0
        finite_contacts = np.array(contact_counts)[np.isfinite(contact_counts)]
        y_max = np.nanmax(finite_contacts) * 1.1 if len(finite_contacts)>0 else 10 # Default max if no data
        ax.set_ylim(bottom=0, top=y_max)
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, "Toxin_Channel_Contacts.png"), dpi=150)
        plt.close(fig)
    except Exception as e:
         logger.error(f"Failed to create contact count plot: {e}", exc_info=True)

def create_contact_map_visualization(avg_contact_map, toxin_res_labels_internal, channel_res_labels_internal, output_dir,
                                     toxin_label_map, channel_label_map):
    """
    Create heatmap visualization for average toxin-channel residue contacts.
    Uses label maps to create descriptive tick labels from internal keys.

    Args:
        avg_contact_map (np.ndarray): Matrix of average contact frequencies.
        toxin_res_labels_internal (list): List of internal keys for toxin residues (e.g., E_77).
        channel_res_labels_internal (list): List of internal keys for channel residues (e.g., A_68).
        output_dir (str): Directory to save the plot.
        toxin_label_map (dict): Map from internal toxin key to display label (e.g., E_77 -> GLY77).
        channel_label_map (dict): Map from internal channel key to display label (e.g., A_68 -> ASP68(A)).
    """
    func_logger = logging.getLogger(__name__)
    if avg_contact_map is None or avg_contact_map.size == 0:
        func_logger.warning("Contact map data is empty. Skipping full map visualization.")
        return
    if not toxin_label_map or not channel_label_map:
         func_logger.warning("Label maps are empty/missing. Full map visualization will use basic internal labels.")
         # Fallback: Use internal labels as display labels if maps are missing/empty
         display_toxin_labels = list(toxin_res_labels_internal) # Ensure list copy
         display_channel_labels = list(channel_res_labels_internal)
    else:
        # --- Generate Display Labels using the maps ---
        # Use the map, fall back to the original internal label if a key is somehow missing
        try:
            display_toxin_labels = [toxin_label_map.get(lbl, str(lbl)) for lbl in toxin_res_labels_internal] # Ensure fallback is string
            display_channel_labels = [channel_label_map.get(lbl, str(lbl)) for lbl in channel_res_labels_internal]
            func_logger.debug(f"Generated {len(display_toxin_labels)} toxin and {len(display_channel_labels)} channel display labels for full map.")
        except Exception as e_map:
             func_logger.error(f"Error applying label maps: {e_map}. Falling back to internal labels.", exc_info=True)
             display_toxin_labels = list(toxin_res_labels_internal)
             display_channel_labels = list(channel_res_labels_internal)

    # --- Dynamic Figure Sizing ---
    # Adjust multiplier and base size as needed for your typical data
    height = max(6, len(toxin_res_labels_internal) * 0.15 + 1) # Base size + scaling
    width = max(8, len(channel_res_labels_internal) * 0.10 + 1)
    # Clamp max size to prevent excessively large images
    max_dim = 40 # Inches
    height = min(height, max_dim)
    width = min(width, max_dim)
    func_logger.debug(f"Full contact map figure size: {width:.1f} x {height:.1f} inches")

    fig, ax = plt.subplots(figsize=(width, height))
    cmap = "viridis" # Or "magma", "plasma", "hot", etc.

    # Create the heatmap - still uses the raw data (avg_contact_map)
    # Set xticklabels/yticklabels explicitly below, so turn off defaults here.
    try:
        sns.heatmap(avg_contact_map, cmap=cmap, ax=ax, linewidths=0.1, linecolor='lightgrey',
                    cbar_kws={'label': 'Average Contact Frequency', 'shrink': 0.7}, # Adjust colorbar size
                    xticklabels=False, yticklabels=False) # Turn off default labels
    except Exception as e_heat:
         func_logger.error(f"Error during sns.heatmap call: {e_heat}", exc_info=True)
         plt.close(fig)
         return # Cannot proceed if heatmap fails

    ax.set_ylabel('Toxin Residues', fontsize=11)
    ax.set_xlabel('Channel Residues', fontsize=11)
    ax.set_title('Average Toxin-Channel Residue Contacts (Full Map)', fontsize=13)

    # --- Set Ticks using Display Labels ---
    # Adjust divisor to control label density (higher number = fewer labels)
    xtick_divisor = 45 if width > 20 else 30 # Fewer labels on very wide plots
    ytick_divisor = 60 if height > 25 else 40
    xtick_step = max(1, len(channel_res_labels_internal) // xtick_divisor)
    ytick_step = max(1, len(toxin_res_labels_internal) // ytick_divisor)
    func_logger.debug(f"Full map tick steps: x={xtick_step}, y={ytick_step}")


    # Get the indices for ticks based on original labels list length
    xtick_indices = np.arange(len(channel_res_labels_internal))[::xtick_step]
    ytick_indices = np.arange(len(toxin_res_labels_internal))[::ytick_step]

    # Apply the indices to the DISPLAY labels list
    # Ensure indices are within bounds of display labels (should be if generated correctly)
    if len(xtick_indices) > 0 and xtick_indices[-1] < len(display_channel_labels) and \
       len(ytick_indices) > 0 and ytick_indices[-1] < len(display_toxin_labels):
        ax.set_xticks(xtick_indices + 0.5) # Position ticks in the middle of cells
        ax.set_xticklabels([display_channel_labels[i] for i in xtick_indices], rotation=90, fontsize=7) # Use generated display labels
        ax.set_yticks(ytick_indices + 0.5)
        ax.set_yticklabels([display_toxin_labels[i] for i in ytick_indices], rotation=0, fontsize=7) # Use generated display labels
    else:
        func_logger.warning("Could not set tick labels due to index mismatch or empty ticks.")


    plt.tight_layout(pad=1.2) # Adjust padding if labels overlap title
    plot_path = os.path.join(output_dir, "Toxin_Channel_Residue_Contact_Map_Full.png")
    try:
        fig.savefig(plot_path, dpi=150, bbox_inches='tight') # Use bbox_inches for potentially long labels
        func_logger.info(f"Saved full contact map visualization to {plot_path}")
    except Exception as e:
        func_logger.error(f"Failed to save full contact map plot: {e}", exc_info=True)
    finally:
        # Ensure the figure is closed even if saving fails
        if 'fig' in locals() and plt.fignum_exists(fig.number):
             plt.close(fig)
             

def create_enhanced_focused_heatmap(avg_contact_map, toxin_res_labels_internal, channel_res_labels_internal,
                                     toxin_label_map, channel_label_map, output_dir,
                                     top_n_toxin=10, top_n_channel=15, # Adjusted defaults
                                     title="Focused Toxin-Channel Contact Map"):
    """
    Create and save an enhanced heatmap focusing only on top interacting residues,
    using specific label formats provided by maps.

    Args:
        avg_contact_map (np.ndarray): Full matrix of average contact frequencies.
        toxin_res_labels_internal (list): List of internal keys for toxin residues (e.g., E_77).
        channel_res_labels_internal (list): List of internal keys for channel residues (e.g., A_68).
        toxin_label_map (dict): Map from internal toxin key to display label (e.g., E_77 -> GLY77).
        channel_label_map (dict): Map from internal channel key to display label (e.g., A_68 -> ASP68(A)).
        output_dir (str): Directory to save the plot.
        top_n_toxin (int): Number of top interacting toxin residues to show.
        top_n_channel (int): Number of top interacting channel residues to show.
        title (str): Title for the plot.

    Returns:
        pd.DataFrame or None: DataFrame of the focused contact map data with display labels,
                              or None on error.
    """
    # --- Input validation ---
    if avg_contact_map is None or avg_contact_map.size == 0:
        logger.error("Contact map data is empty. Cannot create focused heatmap.") # Use logger
        return None
    if not toxin_res_labels_internal or not channel_res_labels_internal:
        logger.error("Internal residue labels are missing. Cannot create focused heatmap.") # Use logger
        return None
    # Check shape match between map and internal labels passed
    if avg_contact_map.shape[0] != len(toxin_res_labels_internal):
        logger.error(f"Shape mismatch: Map rows ({avg_contact_map.shape[0]}) != Toxin Labels ({len(toxin_res_labels_internal)}). Cannot create focused heatmap.") # Use logger
        return None
    if avg_contact_map.shape[1] != len(channel_res_labels_internal):
        logger.error(f"Shape mismatch: Map columns ({avg_contact_map.shape[1]}) != Channel Labels ({len(channel_res_labels_internal)}). Cannot create focused heatmap.") # Use logger
        return None
    if not toxin_label_map or not channel_label_map:
         logger.warning("Label maps are empty/missing. Focused map plot labels might be basic.") # Use logger
         # Create fallback maps using internal labels if they are missing
         toxin_label_map = {lbl: lbl for lbl in toxin_res_labels_internal}
         channel_label_map = {lbl: lbl for lbl in channel_res_labels_internal}


    try:
        logger.debug("Generating focused contact map...") # Use logger
        # --- Calculate interaction sums using the full map and internal labels ---
        # Need temporary DataFrame with internal keys for easier summation by label
        full_map_df_internal = pd.DataFrame(avg_contact_map, index=toxin_res_labels_internal, columns=channel_res_labels_internal)
        toxin_sums = full_map_df_internal.sum(axis=1) # Sum across columns for each toxin row
        channel_sums = full_map_df_internal.sum(axis=0) # Sum across rows for each channel column

        # --- Select top residues by interaction sum ---
        sorted_toxin = sorted([item for item in toxin_sums.items() if item[1] > 1e-6], key=lambda x: x[1], reverse=True)
        sorted_channel = sorted([item for item in channel_sums.items() if item[1] > 1e-6], key=lambda x: x[1], reverse=True)

        actual_top_n_toxin = min(top_n_toxin, len(sorted_toxin))
        actual_top_n_channel = min(top_n_channel, len(sorted_channel))

        if actual_top_n_toxin == 0 or actual_top_n_channel == 0:
            logger.warning("No significantly interacting residues found for focused heatmap.") # Use logger
            return None

        top_toxin_labels_internal = [item[0] for item in sorted_toxin[:actual_top_n_toxin]]
        top_channel_labels_internal = [item[0] for item in sorted_channel[:actual_top_n_channel]]
        logger.debug(f"Top Toxin Internal Keys: {top_toxin_labels_internal}") # Use logger
        logger.debug(f"Top Channel Internal Keys: {top_channel_labels_internal}") # Use logger


        # --- Create the focused matrix ---
        try:
             toxin_idx_map = {label: i for i, label in enumerate(toxin_res_labels_internal)}
             channel_idx_map = {label: i for i, label in enumerate(channel_res_labels_internal)}
             top_toxin_indices = [toxin_idx_map[lbl] for lbl in top_toxin_labels_internal]
             top_channel_indices = [channel_idx_map[lbl] for lbl in top_channel_labels_internal]
        except ValueError as e_idx:
            logger.error(f"Error finding index for top residue label: {e_idx}. Inconsistency between labels and map.", exc_info=True) # Use logger
            return None

        focused_matrix = avg_contact_map[np.ix_(top_toxin_indices, top_channel_indices)]

        # --- Create descriptive labels for the plot using the label maps ---
        display_toxin_labels = [toxin_label_map.get(label, str(label)) for label in top_toxin_labels_internal]
        display_channel_labels = [channel_label_map.get(label, str(label)) for label in top_channel_labels_internal]
        logger.debug(f"Display Toxin Labels: {display_toxin_labels}") # Use logger
        logger.debug(f"Display Channel Labels: {display_channel_labels}") # Use logger


        # --- Create and save the heatmap ---
        height = max(6, actual_top_n_toxin * 0.4 + 1)
        width = max(7, actual_top_n_channel * 0.4 + 1)
        max_dim = 30
        height = min(height, max_dim)
        width = min(width, max_dim)
        logger.debug(f"Focused map figure size: {width:.1f} x {height:.1f} inches") # Use logger

        fig, ax = plt.subplots(figsize=(width, height))

        sns.heatmap(focused_matrix, cmap="viridis", ax=ax,
                    linewidths=0.5, linecolor='lightgrey',
                    cbar_kws={'label': 'Average Contact Frequency', 'shrink': 0.8},
                    annot=True, fmt=".2f", annot_kws={"size": 8},
                    xticklabels=display_channel_labels,
                    yticklabels=display_toxin_labels)

        ax.set_title(f'{title}\n(Top {actual_top_n_toxin} Toxin × Top {actual_top_n_channel} Channel Residues by Summed Contact Freq.)',
                     fontsize=12)
        plt.xlabel('Channel Residues', fontsize=10)
        plt.ylabel('Toxin Residues', fontsize=10)

        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(rotation=0, fontsize=8)

        plt.tight_layout(pad=1.5)

        plot_filename = "Toxin_Channel_Residue_Contact_Map_Focused.png"
        plot_path = os.path.join(output_dir, plot_filename)
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        logger.info(f"Saved focused contact map ({actual_top_n_toxin}x{actual_top_n_channel}) visualization to {plot_path}") # Use logger

        # --- Return a DataFrame representation of the focused map ---
        return pd.DataFrame(focused_matrix, index=display_toxin_labels, columns=display_channel_labels)

    except Exception as e:
        logger.error(f"Error generating focused contact map: {e}", exc_info=True) # Use logger
        if 'fig' in locals() and plt.fignum_exists(fig.number): plt.close(fig)
        return None         
