# pore_analysis/modules/orientation_contacts/computation.py
"""
Computation functions for analyzing toxin orientation, rotation, and contacts
relative to the channel. Separated from visualization.
"""

import os
import logging
import numpy as np
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.analysis import distances
import math
import time

# Import from other modules
try:
    from pore_analysis.core.utils import OneLetter, frames_to_time
    from pore_analysis.core.logging import setup_system_logger
    from pore_analysis.core.database import (
        connect_db, register_module, register_product, store_metric,
        update_module_status, get_simulation_metadata
    )
except ImportError as e:
    print(f"Error importing dependency modules in orientation_contacts/computation.py: {e}")
    raise

logger = logging.getLogger(__name__)

# --- Helper Functions (Merged from orientation.py & orientation_contacts.py) ---

def rotation_matrix_to_euler(R):
    """
    Convert a 3x3 rotation matrix to Euler angles (XYZ sequence, degrees).
    Handles singularity cases. [Derived from: orientation_contacts.py]

    Parameters:
    -----------
    R : numpy.ndarray (3x3)
        Rotation matrix.

    Returns:
    --------
    tuple
        (angle_x, angle_y, angle_z) in degrees. Returns (nan, nan, nan) on error.
    """
    try:
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
    except Exception as e:
        logger.error(f"Error converting rotation matrix to Euler angles: {e}")
        return np.nan, np.nan, np.nan


def calculate_orientation_vectors(u, toxin_sel, channel_sel, prev_toxin_axis=None, prev_channel_axis=None):
    """
    Calculate principal axis vectors for toxin and channel for the current frame.
    Ensures consistent vector direction relative to the previous frame.
    [Derived from: orientation_contacts.py]

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
        dot_product = np.clip(np.abs(np.dot(checked_toxin_axis, checked_channel_axis)), 0.0, 1.0)
        angle = np.degrees(np.arccos(dot_product))

        return checked_toxin_axis, checked_channel_axis, angle, current_toxin_axis, current_channel_axis

    except Exception as e:
        logger.warning(f"Error calculating orientation vectors: {e}", exc_info=False) # Less verbose logging
        return None, None, np.nan, None, None

def calculate_rotation_matrix(reference_coords, current_coords):
    """
    Calculate the optimal rotation matrix to align current_coords onto reference_coords
    using Kabsch algorithm (SVD). Assumes coordinates are already centered.
    [Derived from: orientation_contacts.py]

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
        - Returns None on error.
    """
    if reference_coords.shape != current_coords.shape:
         logger.error("Coordinate shapes mismatch in calculate_rotation_matrix.")
         return None
    try:
        H = np.dot(reference_coords.T, current_coords)
        U, _, Vt = np.linalg.svd(H)
        d = np.linalg.det(np.dot(Vt.T, U.T))
        if d < 0:
            Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)
        return R
    except Exception as e:
        logger.error(f"Error calculating rotation matrix: {e}", exc_info=True)
        return None

def analyze_interface_contacts(u, toxin_sel, channel_sel, cutoff=3.5):
    """
    Calculates atom contacts and residue contacts for a single frame.
    [Derived from: orientation_contacts.py]

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
        - toxin_residues (MDAnalysis.ResidueGroup): Residue group for the toxin selection.
        - channel_residues (MDAnalysis.ResidueGroup): Residue group for the channel selection.
        Returns (0, set(), None, None) on error or empty selections.
    """
    try:
        toxin = u.select_atoms(toxin_sel)
        channel = u.select_atoms(channel_sel)
        toxin_residues = toxin.residues
        channel_residues = channel.residues

        if len(toxin) == 0 or len(channel) == 0 or len(toxin_residues) == 0 or len(channel_residues) == 0:
            return 0, set(), toxin_residues, channel_residues

        box = u.dimensions if hasattr(u, 'dimensions') and u.dimensions is not None else None
        dist_array = distances.distance_array(toxin.positions, channel.positions, box=box)

        atom_contact_map = dist_array < cutoff
        total_atom_contacts = int(np.sum(atom_contact_map))

        residue_contacts_frame = set()
        if total_atom_contacts > 0:
            toxin_atom_to_res_idx = {atom.ix: i for i, res in enumerate(toxin_residues) for atom in res.atoms}
            channel_atom_to_res_idx = {atom.ix: i for i, res in enumerate(channel_residues) for atom in res.atoms}
            contacting_atom_idx_pairs = np.argwhere(atom_contact_map)

            for t_atom_local_idx, c_atom_local_idx in contacting_atom_idx_pairs:
                t_atom_global_ix = toxin.indices[t_atom_local_idx]
                c_atom_global_ix = channel.indices[c_atom_local_idx]
                if t_atom_global_ix in toxin_atom_to_res_idx and c_atom_global_ix in channel_atom_to_res_idx:
                    t_res_idx = toxin_atom_to_res_idx[t_atom_global_ix]
                    c_res_idx = channel_atom_to_res_idx[c_atom_global_ix]
                    residue_contacts_frame.add((t_res_idx, c_res_idx))

        return total_atom_contacts, residue_contacts_frame, toxin_residues, channel_residues

    except Exception as e:
         logger.error(f"Error analyzing interface contacts: {e}", exc_info=False)
         try:
             toxin_res = u.select_atoms(toxin_sel).residues if u else None
             channel_res = u.select_atoms(channel_sel).residues if u else None
             return 0, set(), toxin_res, channel_res
         except: return 0, set(), None, None


# --- Main Analysis Function ---

def run_orientation_analysis(run_dir, psf_file, dcd_file, stride=None, db_conn=None):
    """
    Performs orientation, rotation, and contact analysis for toxin-channel systems.
    Saves time-series data and residue contact frequencies to CSV files.
    Registers module status, products, and metrics in the database.

    Args:
        run_dir (str): Path to the specific run directory.
        psf_file (str): Path to the PSF topology file.
        dcd_file (str): Path to the DCD trajectory file.
        stride (int, optional): Frame stride for analysis. Defaults to analyzing ~500 frames.
        db_conn (sqlite3.Connection, optional): Database connection. If None, connects automatically.

    Returns:
        dict: Contains status ('success', 'skipped', 'failed') and error message if applicable.
    """
    module_name = "orientation_analysis"
    start_time = time.time()

    # Set up logging & DB connection
    logger = setup_system_logger(run_dir)
    if logger is None: logger = logging.getLogger() # Fallback
    if db_conn is None: db_conn = connect_db(run_dir)
    if db_conn is None:
        logger.error(f"{module_name}: Failed to connect to database for {run_dir}")
        return {'status': 'failed', 'error': 'Database connection failed'}

    # Define output directory within the run_dir
    output_dir = os.path.join(run_dir, "orientation_contacts")
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"{module_name}: Outputs will be saved to: {output_dir}")

    # --- Check if Control System ---
    is_control = get_simulation_metadata(db_conn, 'is_control_system')
    if is_control == 'True':
        logger.info(f"{module_name}: Skipping analysis - system identified as control (no toxin).")
        register_module(db_conn, module_name, status='skipped') # Use register_module for skipped status
        update_module_status(db_conn, module_name, 'skipped') # Ensure terminal status update
        return {'status': 'skipped', 'reason': 'Control system'}

    # Register module start
    register_module(db_conn, module_name, status='running')

    # --- Load Universe ---
    try:
        logger.info(f"{module_name}: Loading topology: {psf_file}")
        logger.info(f"{module_name}: Loading trajectory: {dcd_file}")
        u = mda.Universe(psf_file, dcd_file)
        n_frames = len(u.trajectory)
        logger.info(f"{module_name}: Universe loaded with {n_frames} frames.")
        if n_frames < 2:
             raise ValueError("Trajectory has < 2 frames. Orientation/contact analysis requires at least 2 frames.")
    except Exception as e:
        error_msg = f"Failed to load Universe: {e}"
        logger.error(f"{module_name}: {error_msg}", exc_info=True)
        update_module_status(db_conn, module_name, 'failed', error_message=error_msg)
        return {'status': 'failed', 'error': error_msg}

    # --- Define Selections ---
    # Simplified fallback logic, adjust as needed for your specific PSFs
    toxin_sel = 'segid PROE'
    if not u.select_atoms(toxin_sel): toxin_sel = 'segid E'
    if not u.select_atoms(toxin_sel): toxin_sel = 'segid PEP' # Add other common segids if needed
    if not u.select_atoms(toxin_sel):
        error_msg = "Could not find toxin atoms using common selections (PROE, E, PEP)."
        logger.error(f"{module_name}: {error_msg}")
        update_module_status(db_conn, module_name, 'failed', error_message=error_msg)
        return {'status': 'failed', 'error': error_msg}

    channel_sel = 'segid PROA or segid PROB or segid PROC or segid PROD'
    if not u.select_atoms(channel_sel): channel_sel = 'segid A or segid B or segid C or segid D'
    if not u.select_atoms(channel_sel):
        error_msg = "Could not find channel atoms using common selections (PROA-D, A-D)."
        logger.error(f"{module_name}: {error_msg}")
        update_module_status(db_conn, module_name, 'failed', error_message=error_msg)
        return {'status': 'failed', 'error': error_msg}

    logger.info(f"{module_name}: Using Toxin selection: '{toxin_sel}'")
    logger.info(f"{module_name}: Using Channel selection: '{channel_sel}'")

    # --- Initialization ---
    orientation_angles = []
    rotation_euler_angles = []
    total_contact_counts_list = []
    processed_frames_indices = []
    analyzed_frame_count = 0
    residue_contact_accumulator = None
    n_toxin_res, n_channel_res = 0, 0

    # --- Trajectory Iteration Setup ---
    if stride is None: stride = max(1, n_frames // 500)
    logger.info(f"{module_name}: Using stride: {stride} (~{n_frames // stride} frames)")
    frame_iter = u.trajectory[::stride]

    # Get reference structure (first frame)
    try:
        ts_ref = u.trajectory[0]
        ref_toxin_atoms = u.select_atoms(toxin_sel)
        if not ref_toxin_atoms: raise ValueError("Failed to select toxin in first frame for reference.")
        reference_coords_centered = ref_toxin_atoms.positions - ref_toxin_atoms.center_of_mass()
        toxin_residues = u.select_atoms(toxin_sel).residues # Get groups once
        channel_residues = u.select_atoms(channel_sel).residues
        n_toxin_res = len(toxin_residues)
        n_channel_res = len(channel_residues)
        if n_toxin_res > 0 and n_channel_res > 0:
            residue_contact_accumulator = np.zeros((n_toxin_res, n_channel_res), dtype=int)
        else:
            logger.warning(f"{module_name}: Toxin ({n_toxin_res}) or Channel ({n_channel_res}) selection has zero residues. Skipping residue contact accumulation.")
    except Exception as e:
        error_msg = f"Error getting reference data or residue groups: {e}"
        logger.error(f"{module_name}: {error_msg}", exc_info=True)
        update_module_status(db_conn, module_name, 'failed', error_message=error_msg)
        return {'status': 'failed', 'error': error_msg}

    # --- Trajectory Analysis Loop ---
    prev_toxin_axis_state, prev_channel_axis_state = None, None
    try:
        for ts in frame_iter:
            frame_idx = ts.frame
            processed_frames_indices.append(frame_idx)
            analyzed_frame_count += 1

            # 1. Orientation Angle
            _, _, angle, current_toxin_axis, current_channel_axis = calculate_orientation_vectors(
                u, toxin_sel, channel_sel, prev_toxin_axis_state, prev_channel_axis_state)
            orientation_angles.append(angle)
            prev_toxin_axis_state = current_toxin_axis
            prev_channel_axis_state = current_channel_axis

            # 2. Rotation Relative to Reference
            toxin_current = u.select_atoms(toxin_sel)
            if len(toxin_current) == len(ref_toxin_atoms):
                current_coords_centered = toxin_current.positions - toxin_current.center_of_mass()
                R = calculate_rotation_matrix(reference_coords_centered, current_coords_centered)
                if R is not None:
                    euler = rotation_matrix_to_euler(R)
                    rotation_euler_angles.append(euler)
                else: rotation_euler_angles.append((np.nan, np.nan, np.nan)) # Handle rotation calc failure
            else: rotation_euler_angles.append((np.nan, np.nan, np.nan)) # Handle atom count mismatch

            # 3. Contacts (Atom and Residue)
            total_atom_contacts, residue_contacts_frame, _, _ = analyze_interface_contacts(
                u, toxin_sel, channel_sel)
            total_contact_counts_list.append(total_atom_contacts if np.isfinite(total_atom_contacts) else np.nan) # Ensure NaN handling

            if residue_contact_accumulator is not None:
                for t_res_idx, c_res_idx in residue_contacts_frame:
                    if 0 <= t_res_idx < n_toxin_res and 0 <= c_res_idx < n_channel_res:
                        residue_contact_accumulator[t_res_idx, c_res_idx] += 1

            # Progress Log
            if analyzed_frame_count % 100 == 0: logger.debug(f"{module_name}: Processed frame {frame_idx}...")

    except Exception as e:
        error_msg = f"Error during trajectory iteration: {e}"
        logger.error(f"{module_name}: {error_msg}", exc_info=True)
        update_module_status(db_conn, module_name, 'failed', error_message=error_msg)
        return {'status': 'failed', 'error': error_msg}

    # --- Post-Processing & Saving ---
    if analyzed_frame_count == 0:
        error_msg = "No frames were analyzed."
        logger.error(f"{module_name}: {error_msg}")
        update_module_status(db_conn, module_name, 'failed', error_message=error_msg)
        return {'status': 'failed', 'error': error_msg}

    time_points_analysis = frames_to_time(np.array(processed_frames_indices))

    # --- Save Orientation Time Series Data ---
    try:
        orientation_df = pd.DataFrame({
            'Frame': processed_frames_indices,
            'Time (ns)': time_points_analysis,
            'Orientation_Angle': orientation_angles,
            'Rotation_X': [r[0] for r in rotation_euler_angles],
            'Rotation_Y': [r[1] for r in rotation_euler_angles],
            'Rotation_Z': [r[2] for r in rotation_euler_angles],
            'Total_Atom_Contacts': total_contact_counts_list
        })
        ts_csv_filename = "orientation_data.csv"
        ts_csv_path = os.path.join(output_dir, ts_csv_filename)
        orientation_df.to_csv(ts_csv_path, index=False, float_format='%.3f', na_rep='NaN')
        logger.info(f"{module_name}: Saved orientation time series data to {ts_csv_path}")
        # Register product
        register_product(db_conn, module_name, "csv", "data",
                         os.path.relpath(ts_csv_path, run_dir), # Relative path
                         subcategory='orientation_timeseries',
                         description="Time series of orientation angle, rotation, and atom contacts")
    except Exception as e:
        logger.error(f"{module_name}: Failed to save orientation time series CSV: {e}", exc_info=True)
        # Continue analysis if possible, but log the error

    # --- Save Residue Contact Frequency Data ---
    if residue_contact_accumulator is not None:
        try:
            avg_residue_contact_map = residue_contact_accumulator / analyzed_frame_count
            # Get residue labels (simplified, consider enhancing get_residue_info if needed)
            toxin_res_labels = [f"{res.resname}{res.resid}" for res in toxin_residues]
            channel_res_labels = [f"{res.resname}{res.resid}({res.segid[-1]})" for res in channel_residues] # Use last char of segid

            residue_freq_df = pd.DataFrame(avg_residue_contact_map, index=toxin_res_labels, columns=channel_res_labels)
            freq_csv_filename = "residue_contact_frequency.csv"
            freq_csv_path = os.path.join(output_dir, freq_csv_filename)
            residue_freq_df.round(4).to_csv(freq_csv_path, index=True, index_label="Toxin_Residue", float_format='%.4f')
            logger.info(f"{module_name}: Saved residue contact frequency data to {freq_csv_path}")
            # Register product
            register_product(db_conn, module_name, "csv", "data",
                            os.path.relpath(freq_csv_path, run_dir), # Relative path
                            subcategory='residue_frequency',
                            description="Average residue contact frequency map")
        except Exception as e:
            logger.error(f"{module_name}: Failed to save residue contact frequency CSV: {e}", exc_info=True)

    # --- Calculate and Store Metrics ---
    try:
        # Use nan-aware functions
        metrics = {
            'Orient_Angle_Mean': np.nanmean(orientation_angles),
            'Orient_Angle_Std': np.nanstd(orientation_angles),
            'Orient_Contacts_Mean': np.nanmean(total_contact_counts_list),
            'Orient_Contacts_Std': np.nanstd(total_contact_counts_list),
            'Orient_RotX_Mean': np.nanmean([r[0] for r in rotation_euler_angles]),
            'Orient_RotX_Std': np.nanstd([r[0] for r in rotation_euler_angles]),
            'Orient_RotY_Mean': np.nanmean([r[1] for r in rotation_euler_angles]),
            'Orient_RotY_Std': np.nanstd([r[1] for r in rotation_euler_angles]),
            'Orient_RotZ_Mean': np.nanmean([r[2] for r in rotation_euler_angles]),
            'Orient_RotZ_Std': np.nanstd([r[2] for r in rotation_euler_angles]),
        }
        units = {'Angle': '°', 'Contacts': 'count', 'Rot': '°'}
        descriptions = {
            'Angle_Mean': 'Mean Toxin-Channel Orientation Angle', 'Angle_Std': 'Std Dev Toxin-Channel Orientation Angle',
            'Contacts_Mean': 'Mean Atom Contacts (<3.5A)', 'Contacts_Std': 'Std Dev Atom Contacts (<3.5A)',
            'RotX_Mean': 'Mean Toxin Rotation (X)', 'RotX_Std': 'Std Dev Toxin Rotation (X)',
            'RotY_Mean': 'Mean Toxin Rotation (Y)', 'RotY_Std': 'Std Dev Toxin Rotation (Y)',
            'RotZ_Mean': 'Mean Toxin Rotation (Z)', 'RotZ_Std': 'Std Dev Toxin Rotation (Z)',
        }
        for key, val in metrics.items():
             metric_base = key.split('_')[1] # Angle, Contacts, RotX, etc.
             metric_stat = key.split('_')[2] # Mean, Std
             unit_key = metric_base.split('Rot')[0] if 'Rot' in metric_base else metric_base # Angle, Contacts, Rot
             unit = units.get(unit_key, '')
             desc_key = f"{metric_base}_{metric_stat}"
             desc = descriptions.get(desc_key, key)
             # Store only if value is finite
             if np.isfinite(val):
                 store_metric(db_conn, module_name, key, float(val), unit, desc)
             else:
                 logger.warning(f"{module_name}: Metric '{key}' is NaN or Inf, not storing.")

    except Exception as e:
        logger.error(f"{module_name}: Failed to calculate or store metrics: {e}", exc_info=True)
        # Don't fail the whole module just for metrics

    # --- Finalize ---
    exec_time = time.time() - start_time
    update_module_status(db_conn, module_name, 'success', execution_time=exec_time)
    logger.info(f"{module_name}: Analysis completed successfully in {exec_time:.2f} seconds.")
    return {'status': 'success'}