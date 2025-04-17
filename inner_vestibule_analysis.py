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
    from utils import frames_to_time
    from config import EXIT_BUFFER_FRAMES, FRAMES_PER_NS # Need EXIT_BUFFER_FRAMES
    from logger_setup import setup_system_logger
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
        filter_sites (dict): Dictionary mapping site names ('S0'-'S4')
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

    logger.info(f"Starting inner vestibule water analysis for {os.path.basename(run_dir)}")

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
        logger.info(f"Inner Vestibule definition: Z = [{Z_lower_boundary:.2f}, {Z_upper_boundary:.2f}) Å (Absolute), Radius < {Radial_cutoff:.1f} Å")
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
    logger.info("Starting trajectory iteration for inner vestibule water analysis...")
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
                    # Reset streak counter only after confirmation
                    del waters_outside_streak[idx]

            # 3. Handle Waters Entering Now
            for idx in entered_now:
                if idx not in water_entry_frame: # Only record entry if not already inside
                    water_entry_frame[idx] = frame_idx
                # Reset/remove any pending exit confirmation and streak info
                if idx in waters_pending_exit_confirmation:
                    del waters_pending_exit_confirmation[idx]
                if idx in waters_outside_streak:
                    del waters_outside_streak[idx]

            # Update the set of waters considered outside for the next iteration
            waters_currently_outside = set(water_O_atoms.indices) - current_waters_indices_inside

        except Exception as frame_err:
            logger.error(f"Error processing frame {frame_idx} in inner vestibule analysis: {frame_err}", exc_info=True)
            water_counts_per_frame[frame_idx] = -1 # Mark frame as erroneous
            # Attempt to recover state (might be risky)
            try:
                 waters_currently_outside = set(water_O_atoms.indices) - waters_indices_per_frame[frame_idx-1] if frame_idx > 0 else set(water_O_atoms.indices)
            except:
                 waters_currently_outside = set(water_O_atoms.indices)

    # --- Handle Waters Still Inside at End ---
    # Updated log message
    logger.info("Processing waters potentially still inside inner vestibule at trajectory end...")
    final_frame_idx = n_frames - 1
    for idx, entry_frame in list(water_entry_frame.items()):
        # If a water entered and never had a *confirmed* exit, calculate its duration until the end
        # Check if it was inside the very last processed frame
        if idx in waters_indices_per_frame[final_frame_idx]:
             duration_frames = final_frame_idx - entry_frame + 1 # Include final frame
             if duration_frames > 0:
                 duration_ns = duration_frames * time_per_frame
                 # Append this potentially incomplete residence time
                 water_residence_times[idx].append(duration_ns)
                 logger.debug(f"Water {idx} still inside at end. Recorded final residence time: {duration_ns:.3f} ns.")
        # else: If it wasn't inside the last frame, its exit confirmation might be pending or occurred just before end.
        # The 'stayed_outside' loop check handles confirmation up to the end.

    # --- Calculate Statistics ---
    logger.info("Calculating final inner vestibule water statistics...")

    # Filter out error frames (-1 counts) if any
    valid_frame_indices = np.where(water_counts_per_frame >= 0)[0]
    if len(valid_frame_indices) < n_frames:
        logger.warning(f"Excluded {n_frames - len(valid_frame_indices)} frames due to processing errors.")
    if len(valid_frame_indices) == 0:
        logger.error("No valid frames processed. Cannot calculate statistics.")
        return {}

    valid_counts = water_counts_per_frame[valid_frame_indices]
    mean_occupancy = np.mean(valid_counts) if len(valid_counts) > 0 else 0.0
    std_occupancy = np.std(valid_counts) if len(valid_counts) > 0 else 0.0

    # Residence time calculations
    all_residence_times_ns = []
    total_exit_events = 0
    for idx, times in water_residence_times.items():
        all_residence_times_ns.extend(times)
        total_exit_events += len(times) # Each recorded time represents a confirmed exit event

    avg_residence_time = np.mean(all_residence_times_ns) if all_residence_times_ns else 0.0
    median_residence_time = np.median(all_residence_times_ns) if all_residence_times_ns else 0.0

    # Exchange Rate: Number of *exits* per unit time
    total_simulation_time_ns = (len(valid_frame_indices) * time_per_frame) if len(valid_frame_indices) > 0 else 0.0
    exchange_rate_per_ns = (total_exit_events / total_simulation_time_ns) if total_simulation_time_ns > 0 else 0.0

    # --- Save Results ---
    # 1. Water Count Time Series
    count_df = pd.DataFrame({
        'Time (ns)': time_points[valid_frame_indices],
        'Inner_Vestibule_Count': valid_counts
    })
    count_csv_path = os.path.join(output_dir, "Inner_Vestibule_Count.csv")
    try:
        count_df.to_csv(count_csv_path, index=False, float_format='%.4f')
        logger.info(f"Saved inner vestibule water count data to {count_csv_path}")
    except Exception as e:
        logger.error(f"Failed to save inner vestibule count CSV: {e}")

    # 2. Water Indices Per Frame
    indices_json_path = os.path.join(output_dir, "Inner_Vestibule_Indices_Per_Frame.json")
    try:
        # Convert sets to lists for JSON serialization
        serializable_indices = {frame_idx: list(waters_indices_per_frame[frame_idx]) for frame_idx in valid_frame_indices}
        with open(indices_json_path, 'w') as f:
            json.dump(serializable_indices, f, indent=2)
        logger.info(f"Saved inner vestibule water indices per frame to {indices_json_path}")
    except Exception as e:
        logger.error(f"Failed to save inner vestibule indices JSON: {e}")

    # 3. Residence Times (save all individual times)
    residence_df = pd.DataFrame(all_residence_times_ns, columns=['ResidenceTime_ns'])
    residence_csv_path = os.path.join(output_dir, "Inner_Vestibule_Residence_Times.csv")
    try:
        residence_df.to_csv(residence_csv_path, index=False, float_format='%.4f')
        logger.info(f"Saved individual residence times to {residence_csv_path}")
    except Exception as e:
        logger.error(f"Failed to save inner vestibule residence times CSV: {e}")

    # --- Plotting ---
    # 1. Water Count Plot
    try:
        plt.figure(figsize=(10, 5))
        plt.plot(count_df['Time (ns)'], count_df['Inner_Vestibule_Count'], lw=1)
        plt.xlabel("Time (ns)")
        plt.ylabel("Number of Waters")
        plt.title(f"Inner Vestibule Water Count ({os.path.basename(run_dir)})")
        plt.grid(True, alpha=0.5)
        plt.tight_layout()
        plot_path_count = os.path.join(output_dir, "Inner_Vestibule_Count_Plot.png")
        plt.savefig(plot_path_count, dpi=200)
        plt.close()
        logger.info(f"Saved inner vestibule water count plot to {plot_path_count}")
    except Exception as e:
        logger.error(f"Failed to create inner vestibule water count plot: {e}")

    # 2. Residence Time Histogram
    try:
        plt.figure(figsize=(8, 5))
        if all_residence_times_ns:
            sns.histplot(all_residence_times_ns, kde=True, bins=30)
            plt.xlabel("Residence Time (ns)")
            plt.ylabel("Frequency")
            plt.title(f"Inner Vestibule Water Residence Times ({os.path.basename(run_dir)})")
            # Add stats to title or text box
            plt.text(0.95, 0.95, f"Mean: {avg_residence_time:.3f} ns\nMedian: {median_residence_time:.3f} ns\nEvents: {total_exit_events}",
                     transform=plt.gca().transAxes, ha='right', va='top', fontsize=10,
                     bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
        else:
            plt.text(0.5, 0.5, "No residence events recorded", ha='center', va='center', fontsize=12)
            plt.title(f"Inner Vestibule Water Residence Times ({os.path.basename(run_dir)}) - No Events")
        plt.grid(True, alpha=0.5)
        plt.tight_layout()
        plot_path_res = os.path.join(output_dir, "Inner_Vestibule_Residence_Hist.png")
        plt.savefig(plot_path_res, dpi=200)
        plt.close()
        logger.info(f"Saved inner vestibule residence time histogram to {plot_path_res}")
    except Exception as e:
        logger.error(f"Failed to create inner vestibule residence time histogram: {e}")

    # --- Prepare Summary Dictionary ---
    summary_stats = {
        'InnerVestibule_MeanOcc': mean_occupancy,
        'InnerVestibule_StdOcc': std_occupancy,
        'InnerVestibule_AvgResidenceTime_ns': avg_residence_time,
        'InnerVestibule_TotalExitEvents': total_exit_events,
        'InnerVestibule_ExchangeRatePerNs': exchange_rate_per_ns
    }

    logger.info(f"Inner vestibule analysis completed. Mean Occupancy: {mean_occupancy:.2f}, Avg Residence Time: {avg_residence_time:.3f} ns")
    return summary_stats
