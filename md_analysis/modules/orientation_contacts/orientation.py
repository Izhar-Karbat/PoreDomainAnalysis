"""
Functions for analyzing toxin orientation relative to the channel
and toxin rotation over time.
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
    from md_analysis.core.utils import OneLetter, frames_to_time
    from md_analysis.core.logging import setup_system_logger
except ImportError as e:
    print(f"Error importing dependency modules in orientation.py: {e}")
    raise

# Configure matplotlib for non-interactive backend
plt.switch_backend('agg')
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.2)
logger = logging.getLogger(__name__)

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
        (alpha, beta, gamma) Euler angles in degrees.
    """
    # Check for singularity
    if abs(R[2, 0]) > 0.9999:
        # Gimbal lock case
        alpha = math.atan2(R[0, 1], R[0, 2])
        beta = math.copysign(math.pi/2, R[2, 0])
        gamma = 0.0
    else:
        # Normal case
        alpha = math.atan2(-R[1, 0], R[0, 0])
        beta = math.asin(R[2, 0])
        gamma = math.atan2(-R[2, 1], R[2, 2])

    # Convert to degrees
    alpha = math.degrees(alpha)
    beta = math.degrees(beta)
    gamma = math.degrees(gamma)

    return alpha, beta, gamma

def calculate_orientation_vectors(u, toxin_sel, channel_sel, prev_toxin_axis=None, prev_channel_axis=None):
    """
    Calculate orientation vectors for toxin and channel.

    Parameters:
    -----------
    u : MDAnalysis.Universe
        The universe containing the trajectory.
    toxin_sel : str
        Selection string for toxin atoms.
    channel_sel : str
        Selection string for channel atoms.
    prev_toxin_axis : numpy.ndarray, optional
        Previous toxin axis for continuity.
    prev_channel_axis : numpy.ndarray, optional
        Previous channel axis for continuity.

    Returns:
    --------
    tuple
        (toxin_axis, channel_axis) normalized vectors.
    """
    # Get toxin and channel atoms
    toxin_atoms = u.select_atoms(toxin_sel)
    channel_atoms = u.select_atoms(channel_sel)

    if len(toxin_atoms) == 0 or len(channel_atoms) == 0:
        logger.error("No atoms found for orientation calculation")
        return None, None

    # Calculate centers of mass
    toxin_com = toxin_atoms.center_of_mass()
    channel_com = channel_atoms.center_of_mass()

    # Calculate principal axes
    toxin_axis = toxin_atoms.principal_axes()[0]
    channel_axis = channel_atoms.principal_axes()[0]

    # Ensure continuity with previous frame
    if prev_toxin_axis is not None and np.dot(toxin_axis, prev_toxin_axis) < 0:
        toxin_axis = -toxin_axis
    if prev_channel_axis is not None and np.dot(channel_axis, prev_channel_axis) < 0:
        channel_axis = -channel_axis

    return toxin_axis, channel_axis

def calculate_rotation_matrix(reference_coords, current_coords):
    """
    Calculate rotation matrix between two sets of coordinates.

    Parameters:
    -----------
    reference_coords : numpy.ndarray
        Reference coordinates.
    current_coords : numpy.ndarray
        Current coordinates.

    Returns:
    --------
    numpy.ndarray
        3x3 rotation matrix.
    """
    # Center coordinates
    ref_centered = reference_coords - np.mean(reference_coords, axis=0)
    curr_centered = current_coords - np.mean(current_coords, axis=0)

    # Calculate covariance matrix
    H = np.dot(ref_centered.T, curr_centered)

    # Singular value decomposition
    U, S, Vt = np.linalg.svd(H)

    # Calculate rotation matrix
    R = np.dot(Vt.T, U.T)

    # Handle reflection case
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)

    return R

def analyze_toxin_orientation(traj_file, psf_file, run_dir, stride=None):
    """
    Analyze toxin orientation and rotation over time.

    Parameters:
    -----------
    traj_file : str
        Path to trajectory file.
    psf_file : str
        Path to PSF topology file.
    run_dir : str
        Path to run directory.
    stride : int, optional
        Stride for trajectory analysis.

    Returns:
    --------
    tuple
        (orientation_angles, rotation_euler_angles, contact_counts, mean_orientation_angle, rotation_stats_dict)
    """
    # Set up logging
    logger = setup_system_logger(run_dir)
    if logger is None:
        logger = logging.getLogger()
        logger.error(f"Failed to setup system logger for {run_dir}. Using root logger.")

    # Create output directory
    output_dir = os.path.join(run_dir, "orientation_contacts")
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Load universe
        logger.info(f"Loading topology: {psf_file}")
        logger.info(f"Loading trajectory: {traj_file}")
        u = mda.Universe(psf_file, traj_file)
        n_frames = len(u.trajectory)
        logger.info(f"Successfully loaded universe with {n_frames} frames")

        if n_frames == 0:
            logger.warning("Trajectory contains 0 frames.")
            # Return 5 values with None
            return None, None, None, None, {
                'Orient_RotX_Mean': None, 'Orient_RotX_Std': None,
                'Orient_RotY_Mean': None, 'Orient_RotY_Std': None,
                'Orient_RotZ_Mean': None, 'Orient_RotZ_Std': None
            }

        # Define selections
        toxin_sel = 'segid PROE or segid E or segid PEP or segid TOX'
        channel_sel = 'segid PROA or segid PROB or segid PROC or segid PROD or segid A or segid B or segid C or segid D'

        # Initialize arrays
        orientation_angles = []
        rotation_euler_angles = []
        frame_indices = []

        # Get reference frame
        u.trajectory[0]
        ref_toxin_atoms = u.select_atoms(toxin_sel)
        ref_coords = ref_toxin_atoms.positions

        # Analyze trajectory
        prev_toxin_axis = None
        prev_channel_axis = None

        for ts in u.trajectory[::stride] if stride else u.trajectory:
            frame_indices.append(ts.frame)

            # Calculate orientation vectors
            toxin_axis, channel_axis = calculate_orientation_vectors(
                u, toxin_sel, channel_sel, prev_toxin_axis, prev_channel_axis
            )

            if toxin_axis is not None and channel_axis is not None:
                # Calculate angle between vectors
                angle = np.degrees(np.arccos(np.clip(np.dot(toxin_axis, channel_axis), -1.0, 1.0)))
                orientation_angles.append(angle)

                # Calculate rotation matrix
                current_coords = u.select_atoms(toxin_sel).positions
                R = calculate_rotation_matrix(ref_coords, current_coords)
                euler_angles = rotation_matrix_to_euler(R)
                rotation_euler_angles.append(euler_angles)

                # Update previous axes
                prev_toxin_axis = toxin_axis
                prev_channel_axis = channel_axis
            else:
                orientation_angles.append(np.nan)
                rotation_euler_angles.append((np.nan, np.nan, np.nan))

        # Convert to numpy arrays
        orientation_angles = np.array(orientation_angles)
        rotation_euler_angles = np.array(rotation_euler_angles)
        frame_indices = np.array(frame_indices)
        time_points = frames_to_time(frame_indices)

        # Calculate orientation statistics
        mean_orientation_angle = np.nanmean(orientation_angles)
        std_orientation_angle = np.nanstd(orientation_angles)

        # Calculate rotation statistics
        rotation_stats = {
            'Orient_RotX_Mean': np.nanmean(rotation_euler_angles[:, 0]),
            'Orient_RotX_Std': np.nanstd(rotation_euler_angles[:, 0]),
            'Orient_RotY_Mean': np.nanmean(rotation_euler_angles[:, 1]),
            'Orient_RotY_Std': np.nanstd(rotation_euler_angles[:, 1]),
            'Orient_RotZ_Mean': np.nanmean(rotation_euler_angles[:, 2]),
            'Orient_RotZ_Std': np.nanstd(rotation_euler_angles[:, 2])
        }

        # Save results
        df = pd.DataFrame({
            'Frame': frame_indices,
            'Time (ns)': time_points,
            'Orientation_Angle': orientation_angles,
            'Rotation_X': rotation_euler_angles[:, 0],
            'Rotation_Y': rotation_euler_angles[:, 1],
            'Rotation_Z': rotation_euler_angles[:, 2]
        })
        
        output_file = os.path.join(output_dir, "Orientation_Data.csv")
        df.to_csv(output_file, index=False, float_format='%.4f')
        logger.info(f"Saved orientation data to {output_file}")

        return orientation_angles, rotation_euler_angles, None, mean_orientation_angle, rotation_stats

    except Exception as e:
        logger.error(f"Error in toxin orientation analysis: {e}", exc_info=True)
        # Return 5 values with None
        return None, None, None, None, {
            'Orient_RotX_Mean': None, 'Orient_RotX_Std': None,
            'Orient_RotY_Mean': None, 'Orient_RotY_Std': None,
            'Orient_RotZ_Mean': None, 'Orient_RotZ_Std': None
        }
