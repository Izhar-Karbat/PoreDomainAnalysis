# gyration_analysis.py
"""
Module for analyzing the radius of gyration (ρ) of selectivity filter glycines
in K+ channel simulations.

This module calculates how far carbonyl oxygens in the first glycine residue (G1 in GYG motif)
are from the pore center, detecting potential flipping events that can affect ion permeation.

Functions:
- analyze_carbonyl_gyration: Main entry point for radius of gyration analysis
- calculate_pore_center: Determines the geometric center of the pore for each frame
- calculate_gyration_radii: Calculates distances of carbonyl oxygens from pore center
- detect_carbonyl_flips_gyration: Identifies significant changes in gyration radius
- plot_gyration_data: Creates time series plots for gyration radii and flipping events
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import MDAnalysis as mda
from MDAnalysis.analysis import distances
import warnings

# Import from other modules
try:
    from utils import frames_to_time
    from ion_analysis import find_filter_residues
    from logger_setup import setup_system_logger
except ImportError as e:
    print(f"Error importing dependency modules in gyration_analysis.py: {e}")
    raise

# Configure matplotlib for non-interactive backend
plt.switch_backend('agg')
# Set a consistent plot style
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.2)

# Suppress MDAnalysis warnings that might flood output
warnings.filterwarnings('ignore', category=UserWarning, module='MDAnalysis')

# Get a logger for this module
module_logger = logging.getLogger(__name__)

def analyze_carbonyl_gyration(run_dir, psf_file=None, dcd_file=None, system_type="unknown"):
    """
    Analyze the radius of gyration (ρ) of carbonyl oxygens in the first selectivity filter glycine (G1).

    This function tracks how far the carbonyl oxygens of the G1 glycine residue (GYG motif)
    are from the geometric center of the pore. Changes in this distance can indicate
    carbonyl flipping events that affect ion permeation.

    Parameters:
    -----------
    run_dir : str
        Path to the run directory (for saving results)
    psf_file : str, optional
        Path to the PSF topology file. Defaults to 'step5_input.psf' in run_dir.
    dcd_file : str, optional
        Path to the DCD trajectory file. Defaults to 'MD_Aligned.dcd' in run_dir.
    system_type : str, optional
        'toxin' or 'control' to categorize the system

    Returns:
    --------
    dict: Dictionary containing gyration analysis statistics for G1
        - mean_gyration_g1: Mean ρ for G1 residues
        - std_gyration_g1: Standard deviation of ρ for G1 residues
        - flips_detected: Number of significant flips detected in G1
        - max_gyration_change: Maximum change in G1 gyration radius
    """
    # Set up logging
    logger = setup_system_logger(run_dir)
    if logger is None:
        logger = module_logger  # Fallback to module logger

    logger.info("Starting carbonyl gyration (ρ) analysis for G1...")

    # Validate input files
    if psf_file is None:
        psf_file = os.path.join(run_dir, "step5_input.psf")
    if dcd_file is None:
        dcd_file = os.path.join(run_dir, "MD_Aligned.dcd")

    if not os.path.exists(psf_file) or not os.path.exists(dcd_file):
        logger.error(f"PSF or DCD file not found: {psf_file}, {dcd_file}")
        return {
            'mean_gyration_g1': np.nan,
            'std_gyration_g1': np.nan,
            'flips_detected': 0,
            'max_gyration_change': np.nan
        }

    # Create output directory
    output_dir = os.path.join(run_dir, "gyration_analysis")
    os.makedirs(output_dir, exist_ok=True)

    # Load trajectory
    try:
        u = mda.Universe(psf_file, dcd_file)
        n_frames = len(u.trajectory)
        logger.info(f"Loaded trajectory with {n_frames} frames")
    except Exception as e:
        logger.error(f"Error loading trajectory: {e}")
        return {
            'mean_gyration_g1': np.nan,
            'std_gyration_g1': np.nan,
            'flips_detected': 0,
            'max_gyration_change': np.nan
        }

    # Get filter residues
    filter_residues = find_filter_residues(u, logger)
    if not filter_residues:
        logger.error("Failed to identify filter residues")
        return {
            'mean_gyration_g1': np.nan,
            'std_gyration_g1': np.nan,
            'flips_detected': 0,
            'max_gyration_change': np.nan
        }

    # Setup gyration analysis
    frame_indices = []
    pore_centers = []  # Store pore center for each frame
    gyration_data = {
        'g1': {chain: [] for chain in filter_residues.keys()}
        # G2 data removed
    }

    # Build selections for G1 carbonyl oxygens
    g1_oxygen_selections = {}

    for chain, info in filter_residues.items():
        if len(info) >= 3:  # Ensure we have at least up to G1
            # In TVGYG, G1 is at index 2
            g1_resid = info[2]

            # Store selection strings for oxygens
            g1_sel = f"segid {chain} and resid {g1_resid} and name O"
            g1_oxygen_selections[chain] = g1_sel

            logger.debug(f"Chain {chain}: G1 oxygen selection: {g1_sel}")
            # G2 selection removed
        else:
            logger.warning(f"Chain {chain} has fewer than 3 filter residues. Skipping.")

    # Iterate through trajectory
    logger.info("Calculating G1 carbonyl gyration radii...")
    for ts in tqdm(u.trajectory, desc="Gyration analysis", unit="frame"):
        frame_idx = ts.frame
        frame_indices.append(frame_idx)

        # Calculate pore center for this frame (geometric center of filter CA atoms)
        pore_center = calculate_pore_center(u, filter_residues)
        pore_centers.append(pore_center)

        # Calculate gyration radii for G1 oxygens in each chain
        for chain in filter_residues.keys():
            if chain in g1_oxygen_selections:
                # Get G1 carbonyl oxygen position
                g1_oxygen = u.select_atoms(g1_oxygen_selections[chain])
                if len(g1_oxygen) == 1:
                    # Calculate distance from pore center
                    g1_dist = np.linalg.norm(g1_oxygen.positions[0] - pore_center)
                    gyration_data['g1'][chain].append(g1_dist)
                else:
                    gyration_data['g1'][chain].append(np.nan)

            # G2 calculation removed

    # Convert to time points
    time_points = frames_to_time(frame_indices)

    # Detect potential carbonyl flips based on gyration radius changes
    flips_g1 = detect_carbonyl_flips_gyration(gyration_data['g1'], time_points) # Pass only G1 data

    # Save gyration data to CSV
    save_gyration_data(gyration_data, time_points, output_dir)

    # Create plots
    plot_gyration_data(gyration_data, time_points, flips_g1, system_type, output_dir) # Remove flips_g2

    # Calculate statistics
    stats = calculate_gyration_statistics(gyration_data, flips_g1) # Remove flips_g2

    logger.info(f"G1 Gyration analysis completed: {stats['flips_detected']} flips detected")

    return stats

def calculate_pore_center(universe, filter_residues):
    """
    Calculate the geometric center of the pore for the current frame.
    Uses the alpha carbons of the filter G1 residues to define the pore.

    Parameters:
    -----------
    universe : MDAnalysis.Universe
        The universe object at the current frame
    filter_residues : dict
        Dictionary of filter residues by chain

    Returns:
    --------
    numpy.ndarray: The [x, y, z] coordinates of the pore center
    """
    # Get CA atoms of G1 (first glycine in GYG motif)
    g1_ca_atoms = []

    for chain, info in filter_residues.items():
        if len(info) >= 3:  # Need at least up to G1
            g1_resid = info[2]  # G1 is at index 2 (TVGYG)
            sel_str = f"segid {chain} and resid {g1_resid} and name CA"
            atom = universe.select_atoms(sel_str)
            if atom:
                g1_ca_atoms.append(atom)

    # If we found atoms, calculate their geometric center
    if g1_ca_atoms:
        positions = np.vstack([atom.positions for atom in g1_ca_atoms])
        center = np.mean(positions, axis=0)
        return center
    else:
        # Fallback: use the origin (should never happen if filter_residues is valid)
        module_logger.warning("Could not find G1 C-alpha atoms for pore center calculation. Using origin [0,0,0].")
        return np.array([0.0, 0.0, 0.0])

def detect_carbonyl_flips_gyration(g1_gyration_data, time_points, threshold=2.0):
    """
    Detect potential carbonyl flips based on significant changes in G1 gyration radius.

    Parameters:
    -----------
    g1_gyration_data : dict
        Dictionary containing G1 gyration radii by chain
    time_points : numpy.ndarray
        Array of time points
    threshold : float, optional
        Threshold change in Å to consider a flip. Default is 2.0 Å.

    Returns:
    --------
    dict: Mapping chain to list of flip events for G1
          Each flip event is a dict with keys: frame, time, radius_before, radius_after, change
    """
    flips_g1 = {}

    # Process G1
    for chain, radii in g1_gyration_data.items():
        flips_g1[chain] = []
        # Convert to numpy array for easier processing
        radii_array = np.array(radii)

        # Check if array is empty or all NaN
        if radii_array.size == 0 or np.all(np.isnan(radii_array)):
            continue

        # Calculate differences between consecutive non-NaN frames
        finite_indices = np.where(np.isfinite(radii_array))[0]
        if len(finite_indices) < 2: # Need at least two points to calculate diff
             continue

        finite_radii = radii_array[finite_indices]
        diffs = np.diff(finite_radii)
        diff_indices = finite_indices[:-1] # Indices corresponding to the start of the diff interval

        # Find indices where difference exceeds threshold
        flip_diff_indices = np.where(np.abs(diffs) > threshold)[0]
        
        # Map back to original frame indices
        for idx in flip_diff_indices:
             original_frame_idx_before = diff_indices[idx]
             original_frame_idx_after = finite_indices[idx + 1] # Get the index of the next finite point
             
             flip_info = {
                 'frame': original_frame_idx_after, # Frame where the 'after' radius is recorded
                 'time': time_points[original_frame_idx_after],
                 'radius_before': radii_array[original_frame_idx_before],
                 'radius_after': radii_array[original_frame_idx_after],
                 'change': diffs[idx]
             }
             flips_g1[chain].append(flip_info)

    # G2 processing removed

    return flips_g1

def save_gyration_data(gyration_data, time_points, output_dir):
    """
    Save G1 gyration data to CSV files.

    Parameters:
    -----------
    gyration_data : dict
        Dictionary containing gyration radii for G1 by chain
    time_points : numpy.ndarray
        Array of time points
    output_dir : str
        Directory to save the CSV files
    """
    # Create a DataFrame for G1
    g1_data = {'Time (ns)': time_points}
    for chain, radii in gyration_data['g1'].items():
        g1_data[f'{chain}_G1'] = radii

    g1_df = pd.DataFrame(g1_data)
    g1_csv_path = os.path.join(output_dir, "G1_gyration_radii.csv")
    try:
        g1_df.to_csv(g1_csv_path, index=False, float_format='%.4f', na_rep='NaN')
        module_logger.info(f"Saved G1 gyration data to {g1_csv_path}")
    except Exception as e:
        module_logger.error(f"Failed to save G1 gyration CSV: {e}")

    # G2 and combined saving removed

def plot_gyration_data(gyration_data, time_points, flips_g1, system_type, output_dir):
    """
    Create plots for G1 gyration radii data and flipping events.

    Parameters:
    -----------
    gyration_data : dict
        Dictionary containing G1 gyration radii by chain
    time_points : numpy.ndarray
        Array of time points
    flips_g1 : dict
        Dictionary of G1 flip events by chain
    system_type : str
        'toxin' or 'control' to categorize the system
    output_dir : str
        Directory to save the plots
    """
    # Plot settings
    plt.rcParams['figure.figsize'] = (12, 6) # Adjust size for single plot

    # 1. Plot G1 gyration radii
    fig_g1, ax_g1 = plt.subplots()

    for chain, radii in gyration_data['g1'].items():
        line, = ax_g1.plot(time_points, radii, label=f'{chain} G1')

        # Mark flip events
        if chain in flips_g1:
            for flip in flips_g1[chain]:
                ax_g1.axvline(x=flip['time'], color=line.get_color(), linestyle='--', alpha=0.5)
                # Plot a marker at the 'after' position
                ax_g1.plot(flip['time'], flip['radius_after'], 'o', color=line.get_color(), markersize=4)

    ax_g1.set_xlabel('Time (ns)')
    ax_g1.set_ylabel('G1 Gyration Radius (Å)')
    ax_g1.set_title(f'{system_type.capitalize()} System - G1 Carbonyl Gyration Radius')
    ax_g1.legend()
    ax_g1.grid(True, alpha=0.3)

    # Add horizontal line at average radius
    all_g1_radii = np.concatenate([np.array(radii) for radii in gyration_data['g1'].values()])
    if all_g1_radii.size > 0 and np.any(np.isfinite(all_g1_radii)):
        avg_g1_radius = np.nanmean(all_g1_radii)
        ax_g1.axhline(y=avg_g1_radius, color='black', linestyle=':', alpha=0.7, label=f'Avg G1 ρ: {avg_g1_radius:.2f} Å')
        ax_g1.legend() # Update legend to include average line

    plt.tight_layout()
    try:
        plt.savefig(os.path.join(output_dir, "G1_gyration_radii.png"), dpi=200)
    except Exception as e:
        module_logger.error(f"Failed to save G1 gyration plot: {e}")
    plt.close()

    # G2 and combined plots removed

def calculate_gyration_statistics(gyration_data, flips_g1):
    """
    Calculate summary statistics for G1 gyration analysis.

    Parameters:
    -----------
    gyration_data : dict
        Dictionary containing G1 gyration radii by chain
    flips_g1 : dict
        Dictionary of G1 flip events by chain

    Returns:
    --------
    dict: Dictionary containing statistics for G1
        - mean_gyration_g1: Mean ρ for G1 residues
        - std_gyration_g1: Standard deviation of ρ for G1 residues
        - flips_detected: Total number of G1 flips detected across all chains
        - max_gyration_change: Maximum absolute change during a G1 flip event
    """
    all_g1_radii = np.concatenate([np.array(radii) for radii in gyration_data['g1'].values()])
    
    mean_g1 = np.nanmean(all_g1_radii) if all_g1_radii.size > 0 else np.nan
    std_g1 = np.nanstd(all_g1_radii) if all_g1_radii.size > 0 else np.nan

    total_flips_g1 = sum(len(flips) for flips in flips_g1.values())
    max_change_g1 = 0.0
    for chain_flips in flips_g1.values():
        for flip in chain_flips:
            max_change_g1 = max(max_change_g1, abs(flip['change']))
            
    # If no flips detected, max_change should be NaN or 0? Let's use NaN
    if total_flips_g1 == 0:
         max_change_g1 = np.nan

    stats = {
        'mean_gyration_g1': mean_g1,
        'std_gyration_g1': std_g1,
        'flips_detected': total_flips_g1, # Only count G1 flips
        'max_gyration_change': max_change_g1 # Only max G1 change
    }

    # G2 stats removed

    return stats