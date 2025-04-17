"""
Ion coordination analysis module.

Functions for analyzing ion coordination with filter residues
and binding site occupancy.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import MDAnalysis as mda

# Import from other modules
try:
    from md_analysis.core.utils import frames_to_time
    from md_analysis.core.logging import setup_system_logger
except ImportError as e:
    print(f"Error importing dependency modules in coordination.py: {e}")
    raise

# Configure matplotlib for non-interactive backend
plt.switch_backend('agg')
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.2)
logger = logging.getLogger(__name__)

def analyze_ion_coordination(run_dir, time_points, ions_z_positions_abs, ion_indices, filter_sites, g1_reference):
    """
    Analyze ion coordination with filter residues and binding site occupancy.

    Parameters:
    -----------
    run_dir : str
        Path to the run directory.
    time_points : np.ndarray
        Time points in ns.
    ions_z_positions_abs : dict
        Dictionary mapping ion indices to their Z positions.
    ion_indices : list
        List of ion indices to analyze.
    filter_sites : dict
        Dictionary of filter site positions.
    g1_reference : float
        Reference Z position of G1.

    Returns:
    --------
    dict
        Dictionary containing coordination statistics.
    """
    # Set up logging
    logger = setup_system_logger(run_dir)
    if logger is None:
        logger = logging.getLogger()
        logger.error(f"Failed to setup system logger for {run_dir}. Using root logger.")

    # Create output directory
    output_dir = os.path.join(run_dir, "ion_analysis")
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Load universe
        psf_file = os.path.join(run_dir, "step5_input.psf")
        dcd_file = os.path.join(run_dir, "MD_Aligned.dcd")

        logger.info(f"Loading topology: {psf_file}")
        logger.info(f"Loading trajectory: {dcd_file}")
        u = mda.Universe(psf_file, dcd_file)
        n_frames = len(u.trajectory)
        logger.info(f"Successfully loaded universe with {n_frames} frames")

        if n_frames == 0:
            logger.warning("Trajectory contains 0 frames.")
            return {}

        # Define selections
        filter_sel = 'segid PROA or segid PROB or segid PROC or segid PROD or segid A or segid B or segid C or segid D'
        filter_atoms = u.select_atoms(filter_sel)
        if len(filter_atoms) == 0:
            logger.error("No filter atoms found")
            return {}

        # Initialize arrays
        coordination_stats = {
            'site_occupancy': {},
            'ion_transit_times': {},
            'binding_site_residence': {}
        }

        # Analyze each ion
        for ion_idx in ion_indices:
            if ion_idx not in ions_z_positions_abs:
                logger.warning(f"Ion {ion_idx} not found in positions dictionary")
                continue

            z_positions = ions_z_positions_abs[ion_idx]
            if len(z_positions) != len(time_points):
                logger.error(f"Time points and positions mismatch for ion {ion_idx}")
                continue

            # Calculate site occupancy
            site_occupancy = {}
            for site_name, site_pos in filter_sites.items():
                # Count frames where ion is within 3.5 Å of site
                site_occupancy[site_name] = np.sum(
                    np.abs(z_positions - (site_pos + g1_reference)) < 3.5
                ) / len(time_points)

            coordination_stats['site_occupancy'][ion_idx] = site_occupancy

            # Calculate transit times
            in_filter = np.abs(z_positions - g1_reference) < 10.0  # 10 Å cutoff for filter region
            if np.any(in_filter):
                transit_times = []
                current_transit = 0
                for i in range(len(time_points)):
                    if in_filter[i]:
                        current_transit += 1
                    elif current_transit > 0:
                        transit_times.append(current_transit)
                        current_transit = 0
                if current_transit > 0:
                    transit_times.append(current_transit)
                coordination_stats['ion_transit_times'][ion_idx] = transit_times

        # Save results
        df = pd.DataFrame(coordination_stats['site_occupancy']).T
        output_file = os.path.join(output_dir, "Ion_Coordination_Stats.csv")
        df.to_csv(output_file, float_format='%.4f')
        logger.info(f"Saved ion coordination statistics to {output_file}")

        return coordination_stats

    except Exception as e:
        logger.error(f"Error in ion coordination analysis: {e}", exc_info=True)
        return {} 