# inner_vestibule_analysis.py
"""
Functions for analyzing water occupancy and residence time in the channel inner vestibule
below the selectivity filter (S4 site).
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import MDAnalysis as mda
import json
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm
from MDAnalysis.analysis import contacts, distances

# Import from other modules
try:
    from md_analysis.core.utils import frames_to_time
    from md_analysis.core.config import EXIT_BUFFER_FRAMES, FRAMES_PER_NS # Need EXIT_BUFFER_FRAMES
    from md_analysis.core.logging import setup_system_logger
except ImportError as e:
    print(f"Error importing dependency modules in inner_vestibule_analysis.py: {e}")
    raise

# Configure matplotlib for non-interactive backend
plt.switch_backend('agg')
# Set a consistent plot style (optional)
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.2)

# Get a logger for this module (used if specific logger setup fails)
module_logger = logging.getLogger(__name__)

def analyze_inner_vestibule(run_dir, psf_file, dcd_file, filter_sites, g1_reference, filter_residues):
    """
    Analyzes water occupancy and exchange dynamics in the channel inner vestibule
    defined relative to the S4 binding site. Includes an exit buffer requirement
    and outputs water indices per frame. Requires filter_residues to locate pore center.

    Inner Vestibule Definition:
        - Z_upper = Z_S4_absolute - 5.0 Å
        - Z_lower = Z_S4_absolute - 15.0 Å
        - Radius < 6.0 Å from pore center (G1 C-alpha COG XY)

    Args:
        run_dir (str): Directory containing trajectory and for saving results.
        psf_file (str): Path to the PSF topology file.
        dcd_file (str): Path to the DCD trajectory file.
        filter_sites (dict): Dictionary mapping site names ('S0'-'S4', 'Cavity')
                             to Z-coordinates relative to G1 C-alpha plane. Required key: 'S4'.
        g1_reference (float): The absolute Z-coordinate of the G1 C-alpha plane.
        filter_residues (dict): Dictionary mapping chain segids to lists of filter
                                residue IDs (resids). Used to find G1 C-alpha atoms.

    Returns:
        dict: A dictionary containing summary statistics (keys prefixed with
              'InnerVestibule_'), or an empty dictionary {} if critical errors occur.
    """
    # --- Setup ---
    logger = setup_system_logger(run_dir)
    if logger is None: logger = module_logger # Fallback

    logger.info(f"Starting inner vestibule analysis for {os.path.basename(run_dir)}")

    # --- Define Output Directory ---
    output_dir = os.path.join(run_dir, "inner_vestibule_analysis")
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Inner vestibule analysis outputs will be saved to: {output_dir}")

    # --- Input Validation ---
    if not isinstance(filter_sites, dict) or 'S4' not in filter_sites:
        logger.error("Invalid or incomplete 'filter_sites' dictionary provided. 'S4' key is required.")
        return {}
    if not isinstance(g1_reference, (float, int, np.number)):
         logger.error(f"Invalid g1_reference provided (type: {type(g1_reference)}). Must be a number.")
         return {}
    if not isinstance(filter_residues, dict) or not filter_residues:
         logger.error("Invalid or empty 'filter_residues' dictionary provided. Cannot determine pore center.")
         return {}

    # --- Load Universe ---
    try:
        u = mda.Universe(psf_file, dcd_file)
        n_frames = len(u.trajectory)
        if n_frames < 2:
            logger.warning("Trajectory has < 2 frames. Cannot perform time-dependent inner vestibule analysis.")
            return {}
        logger.info(f"Loaded Universe for inner vestibule analysis ({n_frames} frames).")
    except FileNotFoundError as e:
        logger.error(f"Input file not found for inner vestibule analysis: {e}")
        return {}
    except Exception as e:
        logger.error(f"Failed to load Universe for inner vestibule analysis: {e}", exc_info=True)
        return {}

    # --- Time Step Calculation ---
    # Consistent with the rest of the script via FRAMES_PER_NS
    if FRAMES_PER_NS <= 0:
        logger.error(f"FRAMES_PER_NS ({FRAMES_PER_NS}) must be positive.")
        return {}
    time_per_frame = 1.0 / FRAMES_PER_NS
    time_points = frames_to_time(np.arange(n_frames)) # Uses FRAMES_PER_NS via utils
    logger.info(f"Time step calculated: {time_per_frame:.4f} ns/frame.")

    # --- Define Inner Vestibule Geometry ---
    try:
        Z_S4_abs = filter_sites['S4'] + g1_reference
        Z_upper_boundary = Z_S4_abs - 5.0
        Z_lower_boundary = Z_S4_abs - 15.0
        Radial_cutoff = 6.0
        Radial_cutoff_sq = Radial_cutoff**2
        logger.info(f"Inner vestibule definition: Z = [{Z_lower_boundary:.2f}, {Z_upper_boundary:.2f}) Å (Absolute), Radius < {Radial_cutoff:.1f} Å")
    except Exception as e:
        logger.error(f"Error defining inner vestibule geometry: {e}", exc_info=True)
        return {}

    # --- Build G1 C-alpha selection string for pore center ---
    # Assumes filter_residues contains the 5 filter resids for each chain, G1 is index 2
    g1_ca_sel_parts = []
    for segid, resids in filter_residues.items():
        if len(resids) >= 3: # Need at least T, V, G1
            g1_resid = resids[2] # G1 is the third residue in the list
            g1_ca_sel_parts.append(f"(segid {segid} and resid {g1_resid} and name CA)")
        else:
             logger.warning(f"Chain {segid} has fewer than 3 filter residues. Cannot use for pore center.")

    if not g1_ca_sel_parts:
        logger.error("Could not build G1 C-alpha selection string from filter_residues. Aborting.")
        return {}
    g1_ca_sel_str = " or ".join(g1_ca_sel_parts)
    logger.debug(f"Using G1 C-alpha selection for pore center: {g1_ca_sel_str}")

    # Check selection on first frame
    try:
        u.trajectory[0]
        g1_ca_atoms_check = u.select_atoms(g1_ca_sel_str)
        if not g1_ca_atoms_check:
            logger.error(f"G1 C-alpha selection '{g1_ca_sel_str}' returned no atoms in first frame.")
            return {}
    except Exception as e:
         logger.error(f"Error checking G1 C-alpha selection: {e}", exc_info=True)
         return {}


    # --- Select Water Oxygens ---
    water_selection_string = "name OW or type OW or ((resname TIP3 or resname WAT or resname HOH) and (name O or name OH2))" # Added type OW
    try:
        water_O_atoms = u.select_atoms(water_selection_string)
        if not water_O_atoms:
            logger.warning(f"No water oxygen atoms found using selection: '{water_selection_string}'. Returning zero stats.")
            # Return default stats indicating no water found
            return { 'InnerVestibule_MeanOcc': 0.0, 'InnerVestibule_StdOcc': 0.0,
                     'InnerVestibule_AvgResidenceTime_ns': 0.0, 'InnerVestibule_TotalExitEvents': 0,
                     'InnerVestibule_ExchangeRatePerNs': 0.0 }
        logger.info(f"Found {len(water_O_atoms)} water oxygen atoms.")
    except Exception as e:
        logger.error(f"Error selecting water atoms: {e}", exc_info=True)
        return {}

    # --- Initialize Tracking Variables ---
    water_counts_per_frame = np.zeros(n_frames, dtype=int)
    waters_indices_per_frame = [set() for _ in range(n_frames)] # Store indices per frame
    water_residence_times = defaultdict(list) # Stores lists of *confirmed* residence durations (ns) per water index
    water_entry_frame = {} # Stores {water_idx: frame_idx} when water last entered
    # For exit buffer
    waters_currently_outside = set(water_O_atoms.indices) # Initially, all waters are outside
    waters_outside_streak = defaultdict(int) # {water_idx: consecutive_frames_outside}
    waters_pending_exit_confirmation = {} # {water_idx: first_frame_it_exited_in_streak}

    logger.info(f"Using exit buffer: Water must be outside > {EXIT_BUFFER_FRAMES} frames to confirm exit.")

    # --- Iterate Through Trajectory ---
    logger.info("Starting trajectory iteration for inner vestibule analysis...")
    g1_ca_atoms = u.select_atoms(g1_ca_sel_str) # Select the group once

    for ts in tqdm(u.trajectory, desc=f"Inner Vestibule ({os.path.basename(run_dir)})", unit="frame"):
        frame_idx = ts.frame
        try:
            # Calculate pore center XY for the current frame
            if not g1_ca_atoms: # Should not happen if initial check passed, but safety first
                 logger.warning(f"G1 CA atoms lost at frame {frame_idx}? Skipping.")
                 water_counts_per_frame[frame_idx] = -1 # Indicate error/skip
                 continue
            pore_center_xy = g1_ca_atoms.center_of_geometry()[:2] # XY coordinates

            # Get water positions and apply filters
            water_pos = water_O_atoms.positions
            z_mask = (water_pos[:, 2] >= Z_lower_boundary) & (water_pos[:, 2] < Z_upper_boundary)
            dx = water_pos[:, 0] - pore_center_xy[0]
            dy = water_pos[:, 1] - pore_center_xy[1]
            r_mask = (dx*dx + dy*dy) < Radial_cutoff_sq
            inside_mask = z_mask & r_mask

            # Get *global atom indices* of water oxygens currently inside
            current_waters_indices_inside = set(water_O_atoms[inside_mask].indices)

            # Store count and indices for this frame
            water_counts_per_frame[frame_idx] = len(current_waters_indices_inside)
            waters_indices_per_frame[frame_idx] = current_waters_indices_inside

            # Update outside streaks for all waters not inside
            outside_this_frame = set(water_O_atoms.indices) - current_waters_indices_inside

            # 1. Handle waters that were outside last frame and still outside
            for water_idx in waters_currently_outside.intersection(outside_this_frame):
                waters_outside_streak[water_idx] += 1
                
                # If water was pending exit confirmation and now exceeded buffer,
                # confirm it and register its residence time
                if water_idx in waters_pending_exit_confirmation and \
                   waters_outside_streak[water_idx] > EXIT_BUFFER_FRAMES:
                    # Get when it originally entered
                    if water_idx in water_entry_frame:
                        entry_frame = water_entry_frame[water_idx]
                        # Use first outside frame (when streak started) as exit frame
                        exit_frame = waters_pending_exit_confirmation[water_idx]
                        if exit_frame > entry_frame:
                            residence_time_frames = exit_frame - entry_frame
                            residence_time_ns = residence_time_frames * time_per_frame
                            # Add residence time to collection
                            water_residence_times[water_idx].append(residence_time_ns)
                            # Remove from entry tracking (must re-enter)
                            water_entry_frame.pop(water_idx, None)
                    
                    # Remove from pending list
                    waters_pending_exit_confirmation.pop(water_idx, None)

            # 2. Handle waters that were inside but are now outside (starting streak)
            for water_idx in (waters_currently_outside.symmetric_difference(outside_this_frame) & outside_this_frame):
                # This is a water that just left - start streak
                waters_outside_streak[water_idx] = 1
                # Track first frame out for later residence time calculation
                waters_pending_exit_confirmation[water_idx] = frame_idx

            # 3. Handle waters that were outside but now inside (entry or re-entry)
            for water_idx in waters_currently_outside - outside_this_frame:
                # Reset streak counter
                waters_outside_streak[water_idx] = 0
                # Remove pending exit (if any)
                waters_pending_exit_confirmation.pop(water_idx, None)
                # Register entry if not already registered
                if water_idx not in water_entry_frame:
                    water_entry_frame[water_idx] = frame_idx

            # Update currently outside set for next frame
            waters_currently_outside = outside_this_frame

        except Exception as e:
            logger.error(f"Error analyzing frame {frame_idx}: {e}", exc_info=True)
            water_counts_per_frame[frame_idx] = -1

    # --- Flatten All Residence Times ---
    # Combine all confirmed exits from all waters
    all_residence_times_ns = []
    for water_idx, times in water_residence_times.items():
        all_residence_times_ns.extend(times)
    
    # --- Calculate Summary Statistics ---
    try:
        # Water count stats
        mean_occupancy = np.mean(water_counts_per_frame)
        std_occupancy = np.std(water_counts_per_frame)
        
        # Residence time stats
        avg_residence_time = np.mean(all_residence_times_ns) if all_residence_times_ns else 0.0
        
        # Exchange stats
        total_confirmed_exits = len(all_residence_times_ns)
        total_time_ns = n_frames * time_per_frame
        exchange_rate_per_ns = total_confirmed_exits / total_time_ns if total_time_ns > 0 else 0.0

        summary_stats = {
            'InnerVestibule_MeanOcc': float(mean_occupancy),
            'InnerVestibule_StdOcc': float(std_occupancy),
            'InnerVestibule_AvgResidenceTime_ns': float(avg_residence_time),
            'InnerVestibule_TotalExitEvents': int(total_confirmed_exits),
            'InnerVestibule_ExchangeRatePerNs': float(exchange_rate_per_ns)
        }
        logger.info(f"Calculated inner vestibule summary stats: {summary_stats}")
    except Exception as e:
        logger.error(f"Error calculating summary statistics: {e}", exc_info=True)
        summary_stats = {}

    # --- Generate Plots ---
    logger.info("Generating inner vestibule plots...")
    try:
        # Convert indices sets to a string for CSV storage
        indices_str_list = [','.join(map(str, indices)) for indices in waters_indices_per_frame]
        # Save water count data
        df_occup = pd.DataFrame({
            'Time (ns)': time_points,
            'Water_Count': water_counts_per_frame,
            'Inner_Vestibule_Indices': indices_str_list
        })
        occ_csv_path = os.path.join(output_dir, "Inner_Vestibule_Occupancy.csv")
        df_occup.to_csv(occ_csv_path, index=False, float_format='%.4f')
        logger.info(f"Saved inner vestibule occupancy data (with indices) to {occ_csv_path}")

        # Save residence time data
        residence_time_data = {
            'metadata': {
                'mean_residence_time_ns': float(avg_residence_time) if 'avg_residence_time' in locals() else 0.0,
                'simulation_time_ns': float(total_time_ns) if 'total_time_ns' in locals() else 0.0,
                'exit_buffer_frames': EXIT_BUFFER_FRAMES,
                'analysis_datetime': datetime.now().isoformat()
            },
            'residence_times_ns': all_residence_times_ns
        }
        
        res_json_path = os.path.join(output_dir, "Inner_Vestibule_ResidenceTimes.json")
        with open(res_json_path, 'w') as f:
            json.dump(residence_time_data, f, indent=2)
        logger.info(f"Saved inner vestibule residence times (n={len(all_residence_times_ns)}) to {res_json_path}")

        # Plot 1: Water count over time
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Water count
        ax1.plot(time_points, water_counts_per_frame, 'b-', linewidth=1.0)
        ax1.plot(time_points, water_counts_per_frame, 'ko', markersize=2, alpha=0.5)
        
        # Add mean line
        ax1.axhline(y=mean_occupancy, color='r', linestyle='--', alpha=0.7)
        ax1.text(time_points[-1]*0.95, mean_occupancy*1.05, f"Mean: {mean_occupancy:.2f}", 
                color='r', ha='right', fontsize=10)
        
        ax1.set_xlabel('Time (ns)', fontsize=12)
        ax1.set_ylabel('Water Count', fontsize=12)
        ax1.set_title(f'Inner Vestibule Water Occupancy ({os.path.basename(run_dir)})', fontsize=14)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Residence time histogram
        if all_residence_times_ns:
            bins = np.linspace(0, np.max(all_residence_times_ns)*1.1, max(10, min(30, len(all_residence_times_ns)//5)))
            ax2.hist(all_residence_times_ns, bins=bins, alpha=0.7, color='skyblue', edgecolor='navy')
            ax2.axvline(x=avg_residence_time, color='r', linestyle='--', alpha=0.7)
            ax2.text(avg_residence_time*0.95, ax2.get_ylim()[1]*0.9, f"Mean: {avg_residence_time:.2f} ns", 
                    color='r', ha='right', fontsize=10)
            
            ax2.set_xlabel('Residence Time (ns)', fontsize=12)
            ax2.set_ylabel('Frequency', fontsize=12)
            ax2.set_title(f'Inner Vestibule Residence Time ({os.path.basename(run_dir)})', fontsize=14)
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No confirmed exits recorded\nNo residence times to plot', 
                    ha='center', va='center', fontsize=12)
            ax2.set_title('Residence Time Histogram (No Data)', fontsize=14)

        plt.tight_layout()
        plot1_path = os.path.join(output_dir, "Inner_Vestibule_Count_Plot.png")
        plt.savefig(plot1_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        # Plot 3: Residence time distribution
        if len(all_residence_times_ns) > 5:
            plt.figure(figsize=(10, 6))
            sns.kdeplot(all_residence_times_ns, fill=True, cut=0, color='skyblue')
            plt.axvline(x=avg_residence_time, color='r', linestyle='--', alpha=0.7)
            plt.text(avg_residence_time*1.05, plt.gca().get_ylim()[1]*0.9, f"Mean: {avg_residence_time:.2f} ns", 
                    color='r', fontsize=10)
            
            plt.xlabel('Residence Time (ns)', fontsize=12)
            plt.ylabel('Density', fontsize=12)
            plt.title(f'Inner Vestibule Residence Time Distribution ({os.path.basename(run_dir)})', fontsize=14)
            plt.grid(True, alpha=0.3)
            
            plot2_path = os.path.join(output_dir, "Inner_Vestibule_Residence_Hist.png")
            plt.savefig(plot2_path, dpi=150, bbox_inches='tight')
            plt.close()
    except Exception as e:
        logger.error(f"Failed to generate inner vestibule plots: {e}", exc_info=True)

    # --- Return Results ---
    logger.info(f"Inner vestibule analysis complete for {os.path.basename(run_dir)}.")
    if summary_stats:
        return summary_stats
    else:
        return {} 