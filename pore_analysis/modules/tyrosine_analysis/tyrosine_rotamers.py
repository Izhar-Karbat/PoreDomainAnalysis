# tyrosine_rotamers.py
"""
Analyzes the Chi1 and Chi2 dihedral angles and rotameric states of the
Selectivity Filter (SF) Tyrosine residue in potassium channel simulations.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import MDAnalysis as mda
from MDAnalysis.analysis.dihedrals import Dihedral

# Import from other modules
try:
    from pore_analysis.core.utils import frames_to_time
    from pore_analysis.modules.ion_analysis import find_filter_residues
    from pore_analysis.core.logging import setup_system_logger
    from pore_analysis.core.config import TYROSINE_ROTAMER_TOLERANCE_FRAMES, FRAMES_PER_NS
except ImportError as e:
    print(f"Error importing dependency modules in tyrosine_rotamers.py: {e}")
    raise

# Configure matplotlib for non-interactive backend
plt.switch_backend('agg')
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.2)

# Get a logger for this module
module_logger = logging.getLogger(__name__)

# --- Helper Functions --- #

def calc_dihedral(p1, p2, p3, p4):
    """Calculates the dihedral angle from four points in degrees."""
    # Using the formula from MDAnalysis source / Praxeolitic formula
    # Vectors between points
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3

    # Normal vectors to the planes defined by (p1,p2,p3) and (p2,p3,p4)
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    # Vector orthogonal to n1 along b2
    m1 = np.cross(n1, b2 / np.linalg.norm(b2))

    # Magnitudes for normalization
    norm_n1 = np.linalg.norm(n1)
    norm_n2 = np.linalg.norm(n2)

    if norm_n1 == 0 or norm_n2 == 0:
        return np.nan # Undefined if points are collinear

    # Calculate angle components
    x = np.dot(n1, n2)
    y = np.dot(m1, n2)

    # Angle in radians
    angle_rad = np.arctan2(y, x)

    # Convert to degrees
    angle_deg = np.degrees(angle_rad)

    return angle_deg

def classify_rotamer(angle):
    """Classifies dihedral angle into t, p, m states."""
    if np.isnan(angle):
        return 'nan' # Handle NaN input
    # Normalize angle to [-180, 180]
    angle = (angle + 180) % 360 - 180
    if -120 <= angle < 0:
        return 'm' # minus (gauche-)
    elif 0 <= angle < 120:
        return 'p' # plus (gauche+)
    else: # Covers [120, 180] and [-180, -120)
        return 't' # trans

# --- Main Analysis Function --- #

def analyze_sf_tyrosine_rotamers(run_dir, psf_file=None, dcd_file=None, filter_residues=None):
    """
    Track chi1 and chi2 dihedral angles of the SF tyrosine across the trajectory.

    Args:
        run_dir (str): Directory for saving results.
        psf_file (str, optional): Path to topology file. Defaults to step5_input.psf.
        dcd_file (str, optional): Path to trajectory file. Defaults to MD_Aligned.dcd.
        filter_residues (dict, optional): Dict mapping chain segids to lists of filter resids.
                                       If None, will attempt to detect automatically.

    Returns:
        dict: Dictionary containing statistics about SF tyrosine rotamer states (placeholder for now).
    """
    logger = setup_system_logger(run_dir)
    if logger is None:
        logger = module_logger

    logger.info("Starting SF Tyrosine Rotamer Analysis...")

    # Validate input files
    if psf_file is None:
        psf_file = os.path.join(run_dir, "step5_input.psf")
    if dcd_file is None:
        dcd_file = os.path.join(run_dir, "MD_Aligned.dcd")

    # Define default empty stats dictionary for early exit
    default_stats = {
        'Tyr_CommonRotamerState': 'N/A', 'Tyr_RotamerTransitions': 0
        # Add more detailed stats later
    }

    if not os.path.exists(psf_file) or not os.path.exists(dcd_file):
        logger.error(f"PSF or DCD file not found: {psf_file}, {dcd_file}")
        return default_stats

    # Create output directory
    output_dir = os.path.join(run_dir, "tyrosine_analysis")
    os.makedirs(output_dir, exist_ok=True)

    # Load trajectory
    try:
        u = mda.Universe(psf_file, dcd_file)
        n_frames = len(u.trajectory)
        logger.info(f"Loaded trajectory with {n_frames} frames")
    except Exception as e:
        logger.error(f"Error loading trajectory: {e}")
        return default_stats

    # Identify the SF tyrosine in each chain
    if filter_residues is None:
        logger.debug("Filter residues not provided, detecting automatically...")
        filter_residues = find_filter_residues(u, logger)

    if not filter_residues:
        logger.error("Failed to identify filter residues. Cannot proceed with SF Tyrosine analysis.")
        return default_stats

    # Assuming TVGYG motif, Y is typically at index 3 (0-based)
    sf_tyrosine_resids = {}
    chains_to_analyze = list(filter_residues.keys())
    for chain in chains_to_analyze[:]: # Iterate over a copy for safe removal
        resids = filter_residues[chain]
        if len(resids) >= 4:
            # Assuming index 3 based on TVGYG. Make this configurable?
            sf_tyr_resid = resids[3]
            sf_tyrosine_resids[chain] = sf_tyr_resid
            logger.debug(f"Chain {chain}: Identified SF Tyrosine at resid {sf_tyr_resid}")
        else:
            logger.warning(f"Chain {chain}: Filter too short ({len(resids)} residues), cannot identify SF Tyrosine. Skipping chain.")
            chains_to_analyze.remove(chain)

    if not sf_tyrosine_resids:
         logger.error("Could not identify SF Tyrosine in any chain.")
         return default_stats

    # Setup data collection
    time_points = []
    chi1_values = {chain: [] for chain in chains_to_analyze}
    chi2_values = {chain: [] for chain in chains_to_analyze}
    rotamer_states = {chain: [] for chain in chains_to_analyze}

    # Define atom names for dihedrals
    chi1_atom_names = ["N", "CA", "CB", "CG"]
    chi2_atom_names = ["CA", "CB", "CG", "CD1"]

    logger.info("Calculating SF Tyrosine Chi1 and Chi2 dihedrals...")
    # Iterate through trajectory
    for ts in tqdm(u.trajectory, desc="SF Tyr Dihedrals", unit="frame"):
        time_points.append(frames_to_time([ts.frame])[0]) # Assume frames_to_time handles conversion

        for chain in chains_to_analyze:
            resid = sf_tyrosine_resids[chain]

            # Select atoms using MDAnalysis selections
            try:
                # Chi1: N-CA-CB-CG
                chi1_sel = u.select_atoms(f"segid {chain} and resid {resid} and name {' '.join(chi1_atom_names)}")
                # Chi2: CA-CB-CG-CD1
                chi2_sel = u.select_atoms(f"segid {chain} and resid {resid} and name {' '.join(chi2_atom_names)}")

                # Ensure correct order and number of atoms
                if len(chi1_sel) == 4 and len(chi2_sel) == 4:
                    # Reorder based on expected atom names for safety
                    chi1_atoms = [chi1_sel.select_atoms(f"name {n}")[0] for n in chi1_atom_names]
                    chi2_atoms = [chi2_sel.select_atoms(f"name {n}")[0] for n in chi2_atom_names]

                    # Calculate dihedrals using helper
                    chi1 = calc_dihedral(chi1_atoms[0].position, chi1_atoms[1].position,
                                        chi1_atoms[2].position, chi1_atoms[3].position)
                    chi2 = calc_dihedral(chi2_atoms[0].position, chi2_atoms[1].position,
                                        chi2_atoms[2].position, chi2_atoms[3].position)

                    chi1_values[chain].append(chi1)
                    chi2_values[chain].append(chi2)

                    # Classify rotamer state (t: trans, m: minus, p: plus)
                    chi1_state = classify_rotamer(chi1)
                    chi2_state = classify_rotamer(chi2)
                    rotamer_states[chain].append(f"{chi1_state}{chi2_state}")
                else:
                    logger.debug(f"Frame {ts.frame}, Chain {chain}: Incorrect number of atoms found for dihedral calc (chi1: {len(chi1_sel)}, chi2: {len(chi2_sel)}). Appending NaN.")
                    chi1_values[chain].append(np.nan)
                    chi2_values[chain].append(np.nan)
                    rotamer_states[chain].append("nan")
            except Exception as e_calc:
                 logger.warning(f"Frame {ts.frame}, Chain {chain}: Error calculating dihedrals: {e_calc}. Appending NaN.")
                 chi1_values[chain].append(np.nan)
                 chi2_values[chain].append(np.nan)
                 rotamer_states[chain].append("error")

    # --- Post-Processing (Placeholder) ---
    logger.info("Dihedral calculation complete. Proceeding to save/plot/analyze...")

    # TODO: Implement saving to CSV
    save_rotamer_data(output_dir, time_points, chi1_values, chi2_values, rotamer_states)

    # TODO: Implement plotting
    create_rotamer_plots(output_dir, time_points, chi1_values, chi2_values, rotamer_states)

    # Calculate statistics and get event details
    stats, non_dominant_events = analyze_rotamer_statistics(time_points, chi1_values, chi2_values, rotamer_states)

    # Save the detailed non-dominant event data
    save_rotamer_event_data(output_dir, non_dominant_events)

    logger.info("SF Tyrosine Rotamer Analysis complete.")
    return stats

def save_rotamer_data(output_dir, time_points, chi1_values, chi2_values, rotamer_states):
    """Saves the calculated tyrosine rotamer data to a CSV file."""
    logger = module_logger # Use module logger
    logger.info("Saving SF Tyrosine rotamer data...")

    # Combine data into a dictionary suitable for DataFrame creation
    data_to_save = {'Time (ns)': time_points}
    all_chains = list(chi1_values.keys()) # Assuming keys are consistent

    for chain in all_chains:
        data_to_save[f'{chain}_Chi1'] = chi1_values.get(chain, [])
        data_to_save[f'{chain}_Chi2'] = chi2_values.get(chain, [])
        data_to_save[f'{chain}_Rotamer'] = rotamer_states.get(chain, [])

    # Create DataFrame
    try:
        df = pd.DataFrame(data_to_save)
        # Ensure all columns have the same length (padding with NaN if necessary?)
        # DataFrame constructor should handle this if lists are equal length, otherwise check
        max_len = len(time_points)
        for col in df.columns:
            if len(df[col]) < max_len:
                 logger.warning(f"Padding column {col} in rotamer data due to length mismatch.")
                 padded = np.full(max_len, np.nan)
                 padded[:len(df[col])] = df[col]
                 df[col] = padded

    except ValueError as e:
         logger.error(f"Error creating DataFrame for rotamer data (likely uneven column lengths): {e}")
         return

    # Define output path
    csv_path = os.path.join(output_dir, "sf_tyrosine_rotamers.csv")

    # Save to CSV
    try:
        df.to_csv(csv_path, index=False, float_format='%.3f', na_rep='NaN')
        logger.info(f"Saved SF Tyrosine rotamer data to {csv_path}")
    except Exception as e:
        logger.error(f"Failed to save SF Tyrosine rotamer CSV: {e}")

def create_rotamer_plots(output_dir, time_points, chi1_values, chi2_values, rotamer_states):
    """Creates plots for SF Tyrosine rotamer analysis."""
    logger = module_logger
    logger.info("Creating SF Tyrosine rotamer plots...")
    chains = list(chi1_values.keys())
    if not chains:
        logger.warning("No data available to create rotamer plots.")
        return

    # --- 1. Time Series Plot --- #
    n_chains = len(chains)
    # Create subplots, 2 rows (Chi1, Chi2), N columns (chains), or adjust layout
    # For simplicity, let's do separate plots per chain first, or one plot with many lines
    fig_ts, ax_ts = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for chain in chains:
        ax_ts[0].plot(time_points, chi1_values.get(chain, []), label=f'{chain} Chi1', alpha=0.8)
        ax_ts[1].plot(time_points, chi2_values.get(chain, []), label=f'{chain} Chi2', alpha=0.8)

    ax_ts[0].set_ylabel('Chi1 Angle (°)')
    ax_ts[0].set_title('SF Tyrosine Chi1 Dihedral Angle Time Series')
    ax_ts[0].legend(loc='upper right')
    ax_ts[0].grid(True, alpha=0.5)
    ax_ts[0].set_ylim(-180, 180)

    ax_ts[1].set_ylabel('Chi2 Angle (°)')
    ax_ts[1].set_title('SF Tyrosine Chi2 Dihedral Angle Time Series')
    ax_ts[1].set_xlabel('Time (ns)')
    ax_ts[1].legend(loc='upper right')
    ax_ts[1].grid(True, alpha=0.5)
    ax_ts[1].set_ylim(-180, 180)

    plt.tight_layout()
    plot_path_ts = os.path.join(output_dir, "SF_Tyrosine_Dihedrals.png")
    try:
        plt.savefig(plot_path_ts, dpi=200)
        logger.info(f"Saved dihedral time series plot to {plot_path_ts}")
    except Exception as e:
        logger.error(f"Failed to save dihedral time series plot: {e}")
    plt.close(fig_ts)

    # --- 2. Chi1 vs Chi2 Scatter Plot --- #
    fig_scatter, ax_scatter = plt.subplots(figsize=(8, 8))
    # Combine data for scatter plot
    all_chi1 = np.concatenate([chi1_values.get(c, []) for c in chains])
    all_chi2 = np.concatenate([chi2_values.get(c, []) for c in chains])
    # Filter out NaNs for scatter
    valid_idx = np.isfinite(all_chi1) & np.isfinite(all_chi2)

    if np.any(valid_idx):
        # Could color by chain, but might be too messy. Use simple scatter for now.
        ax_scatter.scatter(all_chi1[valid_idx], all_chi2[valid_idx], alpha=0.1, s=5)
        ax_scatter.set_xlabel('Chi1 Angle (°)')
        ax_scatter.set_ylabel('Chi2 Angle (°)')
        ax_scatter.set_title('SF Tyrosine Chi1 vs Chi2 Angles')
        ax_scatter.set_xlim(-180, 180)
        ax_scatter.set_ylim(-180, 180)
        ax_scatter.grid(True, alpha=0.5)
        ax_scatter.set_xticks(np.arange(-180, 181, 60))
        ax_scatter.set_yticks(np.arange(-180, 181, 60))
        ax_scatter.axhline(0, color='grey', lw=0.5, linestyle='--')
        ax_scatter.axvline(0, color='grey', lw=0.5, linestyle='--')

        plt.tight_layout()
        plot_path_scatter = os.path.join(output_dir, "SF_Tyrosine_Rotamer_Scatter.png")
        try:
            plt.savefig(plot_path_scatter, dpi=200)
            logger.info(f"Saved rotamer scatter plot to {plot_path_scatter}")
        except Exception as e:
            logger.error(f"Failed to save rotamer scatter plot: {e}")
        plt.close(fig_scatter)
    else:
        logger.warning("No valid Chi1/Chi2 data points for scatter plot.")
        plt.close(fig_scatter)

    # --- 3. Rotamer State Population --- #
    all_states = np.concatenate([rotamer_states.get(c, []) for c in chains])
    # Exclude nan/error states for population count
    valid_states = all_states[(all_states != 'nan') & (all_states != 'error')]

    if valid_states.size > 0:
        state_counts = pd.Series(valid_states).value_counts(normalize=True) * 100
        possible_states = [c1+c2 for c1 in ['m', 'p', 't'] for c2 in ['m', 'p', 't']] # mm, mp, mt, pm, pp, pt, tm, tp, tt
        # Ensure all possible states are present, default to 0 if missing
        state_counts = state_counts.reindex(possible_states, fill_value=0)

        fig_pop, ax_pop = plt.subplots(figsize=(8, 5))
        state_counts.plot(kind='bar', ax=ax_pop, color=sns.color_palette("viridis", n_colors=len(state_counts)))
        ax_pop.set_xlabel('Rotamer State (Chi1, Chi2)')
        ax_pop.set_ylabel('Population (%)')
        ax_pop.set_title('SF Tyrosine Rotamer State Population')
        ax_pop.tick_params(axis='x', rotation=45)
        ax_pop.grid(True, axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plot_path_pop = os.path.join(output_dir, "SF_Tyrosine_Rotamer_Population.png")
        try:
            plt.savefig(plot_path_pop, dpi=200)
            logger.info(f"Saved rotamer population plot to {plot_path_pop}")
        except Exception as e:
            logger.error(f"Failed to save rotamer population plot: {e}")
        plt.close(fig_pop)
    else:
        logger.warning("No valid rotamer states found for population plot.")

# --- New function to save event data --- #
def save_rotamer_event_data(output_dir, events_list):
    """Saves the detailed non-dominant state events to a CSV file."""
    logger = module_logger
    if not events_list:
        logger.info("No non-dominant rotamer state events detected to save.")
        return

    logger.info("Saving detailed SF Tyrosine non-dominant state events...")
    df_events = pd.DataFrame(events_list)
    # Reorder columns for clarity
    df_events = df_events[['Chain', 'Start Frame', 'End Frame', 'Duration (frames)', 'Duration (ns)']]

    csv_path = os.path.join(output_dir, "sf_tyrosine_non_dominant_events.csv")
    try:
        df_events.to_csv(csv_path, index=False, float_format='%.3f')
        logger.info(f"Saved non-dominant state events to {csv_path}")
    except Exception as e:
        logger.error(f"Failed to save non-dominant state events CSV: {e}")

def analyze_rotamer_statistics(time_points, chi1_values, chi2_values, rotamer_states):
    """
    Calculates summary statistics for SF Tyrosine rotamers, applying tolerance
    to transitions and calculating non-dominant state durations.

    Returns:
        tuple: (dict: stats, list: non_dominant_events)
               - stats: Dictionary containing summary statistics.
               - non_dominant_events: List of dictionaries, each describing a non-dominant state event.
    """
    logger = module_logger
    logger.info("Analyzing SF Tyrosine rotamer statistics...")
    stats = {}
    non_dominant_events = [] # Initialize list to store event details
    chains = list(rotamer_states.keys())

    # Default return if no chains
    if not chains:
        stats = {
            'Tyr_DominantRotamerOverall': 'N/A', 'Tyr_RotamerTransitions': 0,
            'Tyr_NonDominant_MeanDuration_ns': np.nan, 'Tyr_NonDominant_StdDuration_ns': np.nan,
            'Tyr_NonDominant_EventCount': 0, 'Tyr_RotamerToleranceFrames': TYROSINE_ROTAMER_TOLERANCE_FRAMES
        }
        return stats, non_dominant_events # Return empty list with default stats

    all_valid_states_list = []
    total_confirmed_transitions = 0
    dominant_states_per_chain = []

    # --- First pass: Calculate overall dominant state and confirmed transitions --- ##
    for chain in chains:
        states = np.array(rotamer_states.get(chain, []))
        valid_mask = (states != 'nan') & (states != 'error')
        valid_chain_states = states[valid_mask]

        if valid_chain_states.size > 1:
            all_valid_states_list.extend(valid_chain_states)
            counts = pd.Series(valid_chain_states).value_counts()
            dominant_states_per_chain.append(counts.idxmax() if not counts.empty else 'N/A')

            confirmed_transitions_chain = 0
            current_state = valid_chain_states[0]
            n_valid_frames = len(valid_chain_states)
            i = 1
            while i < n_valid_frames:
                if valid_chain_states[i] != current_state:
                    new_state = valid_chain_states[i]
                    persists = True
                    if i + TYROSINE_ROTAMER_TOLERANCE_FRAMES > n_valid_frames:
                        persists = False
                    else:
                        for j in range(1, TYROSINE_ROTAMER_TOLERANCE_FRAMES):
                            if valid_chain_states[i + j] != new_state:
                                persists = False
                                break
                    if persists:
                        confirmed_transitions_chain += 1
                        current_state = new_state
                        i += (TYROSINE_ROTAMER_TOLERANCE_FRAMES - 1)
                i += 1
            total_confirmed_transitions += confirmed_transitions_chain
        elif valid_chain_states.size == 1:
            all_valid_states_list.extend(valid_chain_states)
            dominant_states_per_chain.append(valid_chain_states[0])
        else:
            dominant_states_per_chain.append('N/A')

    # --- Calculate Overall Dominant State --- ##
    if all_valid_states_list:
        overall_counts = pd.Series(all_valid_states_list).value_counts()
        dominant_state_overall = overall_counts.idxmax() if not overall_counts.empty else 'N/A'
        stats['Tyr_DominantRotamerOverall'] = dominant_state_overall
        stats['Tyr_DominantRotamerPerChain'] = ", ".join(dominant_states_per_chain)
    else:
        dominant_state_overall = 'N/A'
        stats['Tyr_DominantRotamerOverall'] = 'N/A'
        stats['Tyr_DominantRotamerPerChain'] = 'N/A'

    stats['Tyr_RotamerTransitions'] = total_confirmed_transitions
    stats['Tyr_RotamerToleranceFrames'] = TYROSINE_ROTAMER_TOLERANCE_FRAMES

    # --- Second pass: Calculate Non-Dominant State Durations & Collect Event Details --- ##
    non_dominant_durations_frames = []
    time_per_frame_ns = 1.0 / FRAMES_PER_NS # Calculate conversion factor once

    if dominant_state_overall != 'N/A': # Only proceed if a dominant state exists
        for chain in chains:
            states = np.array(rotamer_states.get(chain, []))
            valid_mask = (states != 'nan') & (states != 'error')
            valid_chain_states = states[valid_mask]
            # Get original frame indices corresponding to valid states
            original_indices = np.where(valid_mask)[0]
            n_valid = len(valid_chain_states)

            if n_valid > 0:
                in_non_dominant_block = False
                block_start_valid_index = -1
                for i in range(n_valid):
                    is_non_dominant = (valid_chain_states[i] != dominant_state_overall)

                    if is_non_dominant and not in_non_dominant_block:
                        # Start of a new non-dominant block
                        in_non_dominant_block = True
                        block_start_valid_index = i
                    elif (not is_non_dominant or i == n_valid - 1) and in_non_dominant_block:
                        # End of a non-dominant block (either state changed back or end of trajectory)
                        # Determine end index carefully based on whether the last frame was non-dominant
                        block_end_valid_index = i if is_non_dominant else i - 1

                        duration_frames = (block_end_valid_index - block_start_valid_index) + 1
                        non_dominant_durations_frames.append(duration_frames)

                        # Get original start/end frame numbers
                        start_frame = original_indices[block_start_valid_index]
                        end_frame = original_indices[block_end_valid_index]
                        duration_ns = duration_frames * time_per_frame_ns

                        # Store event details
                        non_dominant_events.append({
                            'Chain': chain,
                            'Start Frame': start_frame,
                            'End Frame': end_frame,
                            'Duration (frames)': duration_frames,
                            'Duration (ns)': duration_ns
                        })

                        # Reset block tracking
                        in_non_dominant_block = False
                        block_start_valid_index = -1

    # --- Calculate Duration Statistics --- ##
    if non_dominant_events: # Check if the list has events
        # Durations in ns are already calculated and stored in the events list
        non_dominant_durations_ns = [event['Duration (ns)'] for event in non_dominant_events]
        stats['Tyr_NonDominant_MeanDuration_ns'] = np.mean(non_dominant_durations_ns)
        stats['Tyr_NonDominant_StdDuration_ns'] = np.std(non_dominant_durations_ns)
        stats['Tyr_NonDominant_EventCount'] = len(non_dominant_events)
    else:
        # No non-dominant events found
        stats['Tyr_NonDominant_MeanDuration_ns'] = 0.0
        stats['Tyr_NonDominant_StdDuration_ns'] = 0.0
        stats['Tyr_NonDominant_EventCount'] = 0

    logger.info(f"Rotamer Statistics: Dominant={stats['Tyr_DominantRotamerOverall']}, "
                f"Confirmed Transitions={total_confirmed_transitions}, "
                f"Non-Dominant Events={stats['Tyr_NonDominant_EventCount']}, "
                f"Mean Non-Dom Duration={stats['Tyr_NonDominant_MeanDuration_ns']:.3f} ns (Tolerance: {TYROSINE_ROTAMER_TOLERANCE_FRAMES} frames)")

    return stats, non_dominant_events # Return both stats and the list of events 