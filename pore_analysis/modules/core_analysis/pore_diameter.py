"""
Pore diameter analysis functions for molecular dynamics trajectories.
This module provides functions for calculating and analyzing pore diameter
from trajectory data.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import MDAnalysis as mda
from MDAnalysis.analysis import distances

# Import from other modules
try:
    from pore_analysis.core.utils import frames_to_time
    from pore_analysis.core.logging import setup_system_logger
except ImportError as e:
    print(f"Error importing dependency modules in pore_diameter.py: {e}")
    raise

# Configure matplotlib for non-interactive backend
plt.switch_backend('agg')
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.2)
logger = logging.getLogger(__name__)

def calculate_pore_diameter(run_dir, psf_file=None, dcd_file=None):
    """
    Calculate pore diameter from a molecular dynamics trajectory.
    
    Args:
        run_dir (str): Path to the run directory.
        psf_file (str, optional): Path to the PSF topology file.
        dcd_file (str, optional): Path to the DCD trajectory file.
        
    Returns:
        tuple: Contains:
            - pore_diameters (np.ndarray): Pore diameters over time.
            - time_points (np.ndarray): Time points corresponding to frames.
    """
    # Set up logging
    logger = setup_system_logger(run_dir)
    if logger is None:
        logger = logging.getLogger()
        logger.error(f"Failed to setup system logger for {run_dir}. Using root logger.")

    # Create output directory
    output_dir = os.path.join(run_dir, "core_analysis")
    os.makedirs(output_dir, exist_ok=True)

    # Default file paths
    if psf_file is None:
        psf_file = os.path.join(run_dir, "step5_input.psf")
    if dcd_file is None:
        dcd_file = os.path.join(run_dir, "MD_Aligned.dcd")

    # Check if files exist
    if not os.path.exists(psf_file):
        logger.error(f"PSF file not found: {psf_file}")
        return np.array([]), np.array([])
    if not os.path.exists(dcd_file):
        logger.error(f"DCD file not found: {dcd_file}")
        return np.array([]), np.array([])

    try:
        # Load universe
        logger.info(f"Loading topology: {psf_file}")
        logger.info(f"Loading trajectory: {dcd_file}")
        u = mda.Universe(psf_file, dcd_file)
        n_frames = len(u.trajectory)
        logger.info(f"Successfully loaded universe with {n_frames} frames")

        if n_frames == 0:
            logger.warning("Trajectory contains 0 frames.")
            return np.array([]), np.array([])

        # Select atoms for pore diameter calculation
        # This is a simplified example - adjust selection as needed
        pore_atoms = u.select_atoms('protein and name CA')
        if len(pore_atoms) == 0:
            logger.error("No atoms found for pore diameter calculation")
            return np.array([]), np.array([])

        # Calculate pore diameter over time
        pore_diameters = []
        frame_indices = []

        for ts in u.trajectory:
            frame_indices.append(ts.frame)
            
            # Calculate pore diameter for this frame
            # This is a simplified example - adjust calculation as needed
            try:
                # Get coordinates of pore atoms
                coords = pore_atoms.positions
                
                # Calculate distances between opposing chains
                # This is a simplified example - adjust as needed
                dist_matrix = distances.distance_array(coords, coords)
                
                # Find maximum distance as pore diameter
                # This is a simplified example - adjust as needed
                pore_diameter = np.max(dist_matrix)
                pore_diameters.append(pore_diameter)
                
            except Exception as e:
                logger.error(f"Error calculating pore diameter for frame {ts.frame}: {e}")
                pore_diameters.append(np.nan)

        # Convert to numpy arrays
        pore_diameters = np.array(pore_diameters)
        frame_indices = np.array(frame_indices)
        time_points = frames_to_time(frame_indices)

        # Save results
        df = pd.DataFrame({
            'Frame': frame_indices,
            'Time (ns)': time_points,
            'Pore_Diameter': pore_diameters
        })
        
        output_file = os.path.join(output_dir, "Pore_Diameter.csv")
        df.to_csv(output_file, index=False, float_format='%.4f')
        logger.info(f"Saved pore diameter data to {output_file}")

        return pore_diameters, time_points

    except Exception as e:
        logger.error(f"Error in pore diameter calculation: {e}", exc_info=True)
        return np.array([]), np.array([])
