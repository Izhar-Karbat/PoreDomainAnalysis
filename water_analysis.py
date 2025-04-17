# water_analysis.py
"""
Functions for analyzing water occupancy and residence time in the channel cavity
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
    from utils import frames_to_time
    from config import EXIT_BUFFER_FRAMES, FRAMES_PER_NS # Need EXIT_BUFFER_FRAMES
    from logger_setup import setup_system_logger
except ImportError as e:
    print(f"Error importing dependency modules in water_analysis.py: {e}")
    raise

# Configure matplotlib for non-interactive backend
plt.switch_backend('agg')
# Set a consistent plot style (optional)
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.2)

# Get a logger for this module (used if specific logger setup fails)
module_logger = logging.getLogger(__name__)

def analyze_cavity_water(run_dir, psf_file, dcd_file, filter_sites, g1_reference, filter_residues):
    """
    Analyzes water occupancy and exchange dynamics in the channel cavity
    defined relative to the S4 binding site. Includes an exit buffer requirement
    and outputs water indices per frame. Requires filter_residues to locate pore center.

    Cavity Definition:
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
              'CavityWater_'), or an empty dictionary {} if critical errors occur.
    """
    # --- Setup ---
    logger = setup_system_logger(run_dir)
    if logger is None: logger = module_logger # Fallback

    logger.info(f"Starting cavity water analysis for {os.path.basename(run_dir)}")

    # --- Define Output Directory ---
    output_dir = os.path.join(run_dir, "water_analysis")
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Water analysis outputs will be saved to: {output_dir}")

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
            logger.warning("Trajectory has < 2 frames. Cannot perform time-dependent water analysis.")
            return {}
        logger.info(f"Loaded Universe for water analysis ({n_frames} frames).")
    except FileNotFoundError as e:
        logger.error(f"Input file not found for water analysis: {e}")
        return {}
    except Exception as e:
        logger.error(f"Failed to load Universe for water analysis: {e}", exc_info=True)
        return {}

    # --- Time Step Calculation ---
    # Consistent with the rest of the script via FRAMES_PER_NS
    if FRAMES_PER_NS <= 0:
        logger.error(f"FRAMES_PER_NS ({FRAMES_PER_NS}) must be positive.")
        return {}
    time_per_frame = 1.0 / FRAMES_PER_NS
    time_points = frames_to_time(np.arange(n_frames)) # Uses FRAMES_PER_NS via utils
    logger.info(f"Time step calculated: {time_per_frame:.4f} ns/frame.")

    # --- Define Cavity Geometry ---
    try:
        Z_S4_abs = filter_sites['S4'] + g1_reference
        Z_upper_boundary = Z_S4_abs - 5.0
        Z_lower_boundary = Z_S4_abs - 15.0
        Radial_cutoff = 6.0
        Radial_cutoff_sq = Radial_cutoff**2
        logger.info(f"Cavity definition: Z = [{Z_lower_boundary:.2f}, {Z_upper_boundary:.2f}) Å (Absolute), Radius < {Radial_cutoff:.1f} Å")
    except Exception as e:
        logger.error(f"Error defining cavity geometry: {e}", exc_info=True)
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
            return { 'CavityWater_MeanOcc': 0.0, 'CavityWater_StdOcc': 0.0,
                     'CavityWater_AvgResidenceTime_ns': 0.0, 'CavityWater_TotalExitEvents': 0,
                     'CavityWater_ExchangeRatePerNs': 0.0 }
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
    logger.info("Starting trajectory iteration for cavity water analysis...")
    g1_ca_atoms = u.select_atoms(g1_ca_sel_str) # Select the group once

    for ts in tqdm(u.trajectory, desc=f"Cavity Water ({os.path.basename(run_dir)})", unit="frame"):
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

            # --- State Transition Logic (with Exit Buffer) ---
            # Determine waters that just entered, just exited, or stayed outside
            waters_exited_this_frame = waters_currently_outside.intersection(current_waters_indices_inside) # Mistake: Should be waters that *were* inside prev frame but not now
            waters_entered_this_frame = current_waters_indices_inside - waters_currently_outside # Waters now inside that were outside

            # Corrected Logic: Need state from *previous* frame
            if frame_idx > 0:
                 prev_waters_inside = waters_indices_per_frame[frame_idx - 1]
                 # Waters that were inside previous frame but not now
                 exited_now = prev_waters_inside - current_waters_indices_inside
                 # Waters that are inside now but were not previously
                 entered_now = current_waters_indices_inside - prev_waters_inside
                 # All waters currently outside
                 current_waters_outside = set(water_O_atoms.indices) - current_waters_indices_inside
                 # Waters that were outside last frame and are still outside
                 stayed_outside = waters_currently_outside.intersection(current_waters_outside)
            else: # First frame
                 exited_now = set()
                 entered_now = current_waters_indices_inside # All inside waters entered on frame 0
                 stayed_outside = set(water_O_atoms.indices) - current_waters_indices_inside

            # 1. Handle Waters Exiting Now
            for idx in exited_now:
                if idx not in waters_pending_exit_confirmation: # Start tracking exit only if not already pending
                    waters_pending_exit_confirmation[idx] = frame_idx # Record frame it potentially exited
                waters_outside_streak[idx] = 1 # Reset/start streak

            # 2. Handle Waters Staying Outside
            for idx in stayed_outside:
                if idx in waters_outside_streak:
                    waters_outside_streak[idx] += 1
                else: # Should have been in exited_now or already outside
                    waters_outside_streak[idx] = 1

                # Check for CONFIRMED exit
                if idx in waters_pending_exit_confirmation and waters_outside_streak[idx] > EXIT_BUFFER_FRAMES:
                    if idx in water_entry_frame:
                        first_exit_frame = waters_pending_exit_confirmation[idx]
                        entry_frame = water_entry_frame[idx]
                        duration_frames = first_exit_frame - entry_frame
                        if duration_frames > 0:
                            duration_ns = duration_frames * time_per_frame
                            water_residence_times[idx].append(duration_ns)
                        # Clean up state for this completed residence period
                        del water_entry_frame[idx]
                    # Remove from pending confirmation regardless
                    del waters_pending_exit_confirmation[idx]
                    # Note: Don't remove from waters_outside_streak here, it might continue to stay out

            # 3. Handle Waters Entering Now
            for idx in entered_now:
                water_entry_frame[idx] = frame_idx # Record entry time
                # If it was pending exit or had an outside streak, clear that state
                if idx in waters_pending_exit_confirmation: del waters_pending_exit_confirmation[idx]
                if idx in waters_outside_streak: del waters_outside_streak[idx]

            # Update the set of waters currently outside for the next iteration
            waters_currently_outside = set(water_O_atoms.indices) - current_waters_indices_inside

        except Exception as frame_err:
            logger.error(f"Error processing frame {frame_idx} for water analysis: {frame_err}", exc_info=True)
            water_counts_per_frame[frame_idx] = -1 # Mark as invalid

    # --- Post-Processing & Saving ---
    logger.info("Trajectory iteration finished. Post-processing water data...")
    valid_frames_mask = water_counts_per_frame >= 0 # Exclude frames marked with -1
    n_valid_frames = np.sum(valid_frames_mask)

    if n_valid_frames < 2: # Need at least 2 valid frames for stats
        logger.error("Insufficient valid frames processed for water analysis.")
        return {}

    # Filter data for valid frames only
    valid_time_points = time_points[valid_frames_mask]
    valid_water_counts = water_counts_per_frame[valid_frames_mask]
    valid_indices_sets = [waters_indices_per_frame[i] for i, valid in enumerate(valid_frames_mask) if valid]

    # Prepare indices strings for CSV
    indices_str_list = []
    for idx_set in valid_indices_sets:
        if idx_set is None or not idx_set:
            indices_str_list.append("")
        else:
            indices_str_list.append(' '.join(map(str, sorted(list(idx_set)))))

    # Save detailed occupancy data with indices
    try:
        df_occupancy = pd.DataFrame({
            'Frame': np.arange(n_frames)[valid_frames_mask],
            'Time (ns)': valid_time_points,
            'N_Waters_Cavity': valid_water_counts,
            'Cavity_Water_Indices': indices_str_list
        })
        occ_csv_path = os.path.join(output_dir, "Cavity_Water_Occupancy.csv")
        df_occupancy.to_csv(occ_csv_path, index=False, float_format='%.4f')
        logger.info(f"Saved cavity water occupancy data (with indices) to {occ_csv_path}")
    except Exception as e:
        logger.error(f"Failed to save occupancy CSV: {e}", exc_info=True)

    # Save residence times
    all_residence_times_ns = [t for times in water_residence_times.values() for t in times if t > 0]
    logger.info(f"Total number of confirmed water exit events: {len(all_residence_times_ns)}")
    res_data_to_save = {
        'residence_times_ns': all_residence_times_ns,
        'exit_buffer_frames': EXIT_BUFFER_FRAMES
    }
    res_json_path = os.path.join(output_dir, "Cavity_Water_ResidenceTimes.json")
    try:
        with open(res_json_path, 'w') as f_res:
            json.dump(res_data_to_save, f_res, indent=4)
        logger.info(f"Saved cavity water residence times (n={len(all_residence_times_ns)}) to {res_json_path}")
    except Exception as e:
        logger.error(f"Failed to save residence time JSON: {e}", exc_info=True)

    # --- Calculate Summary Statistics ---
    mean_occupancy = np.mean(valid_water_counts)
    std_occupancy = np.std(valid_water_counts)
    avg_residence_time = np.mean(all_residence_times_ns) if all_residence_times_ns else 0.0
    total_confirmed_exits = len(all_residence_times_ns)
    total_valid_time_ns = valid_time_points[-1] - valid_time_points[0]
    exchange_rate_per_ns = total_confirmed_exits / total_valid_time_ns if total_valid_time_ns > 1e-9 else 0.0

    summary_stats = {
        'CavityWater_MeanOcc': float(mean_occupancy),
        'CavityWater_StdOcc': float(std_occupancy),
        'CavityWater_AvgResidenceTime_ns': float(avg_residence_time),
        'CavityWater_TotalExitEvents': int(total_confirmed_exits),
        'CavityWater_ExchangeRatePerNs': float(exchange_rate_per_ns)
    }
    logger.info(f"Calculated cavity water summary stats: {summary_stats}")

    # --- Generate Plots and Save to Subdirectory ---
    logger.info("Generating cavity water plots...")
    try:
        # 1. Water count vs time
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(valid_time_points, valid_water_counts, label='Water Count', color='dodgerblue', linewidth=1.0)
        ax1.set_xlabel('Time (ns)', fontsize=12)
        ax1.set_ylabel('Number of Water Molecules', fontsize=12)
        ax1.set_title(f'Water Occupancy in Cavity ({os.path.basename(run_dir)})', fontsize=14)
        ax1.grid(True, linestyle=':', alpha=0.6)
        y_max_occ = np.max(valid_water_counts) * 1.1 if len(valid_water_counts)>0 else 10
        ax1.set_ylim(bottom=0, top=max(1, y_max_occ)) # Ensure ylim >= 1
        ax1.legend()
        plt.tight_layout()
        plot1_path = os.path.join(output_dir, "Cavity_Water_Count_Plot.png")
        fig1.savefig(plot1_path, dpi=150, bbox_inches='tight')
        plt.close(fig1)
        logger.debug(f"Saved water count plot to {plot1_path}")

        # 2. Residence time histogram
        if all_residence_times_ns:
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            sns.histplot(all_residence_times_ns, kde=True, ax=ax2, bins=30, color='mediumseagreen', stat="count") # Use count stat
            ax2.set_xlabel('Residence Time (ns)', fontsize=12)
            ax2.set_ylabel('Number of Exit Events', fontsize=12) # Changed ylabel
            ax2.set_title(f'Cavity Water Residence Time ({os.path.basename(run_dir)})', fontsize=14)
            ax2.axvline(avg_residence_time, color='red', linestyle='--', label=f'Mean: {avg_residence_time:.3f} ns')
            ax2.legend()
            ax2.grid(True, linestyle=':', alpha=0.6)
            plt.tight_layout()
            plot2_path = os.path.join(output_dir, "Cavity_Water_Residence_Hist.png")
            fig2.savefig(plot2_path, dpi=150, bbox_inches='tight')
            plt.close(fig2)
            logger.debug(f"Saved residence time histogram to {plot2_path}")
        else:
            logger.info("No confirmed residence times recorded, skipping histogram plot.")

    except Exception as e:
        logger.error(f"Failed to generate cavity water plots: {e}", exc_info=True)
        # Ensure figures are closed if error occurs mid-plotting
        if 'fig1' in locals() and plt.fignum_exists(fig1.number): plt.close(fig1)
        if 'fig2' in locals() and plt.fignum_exists(fig2.number): plt.close(fig2)

    logger.info(f"Cavity water analysis complete for {os.path.basename(run_dir)}.")
    return summary_stats
