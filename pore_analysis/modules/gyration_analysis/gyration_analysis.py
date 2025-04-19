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
    from pore_analysis.core.utils import frames_to_time
    from pore_analysis.modules.ion_analysis import find_filter_residues
    from pore_analysis.core.logging import setup_system_logger
    from pore_analysis.core.config import GYRATION_FLIP_THRESHOLD, GYRATION_FLIP_TOLERANCE_FRAMES, FRAMES_PER_NS
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
    Analyze the radius of gyration (ρ) and flipping states of carbonyl oxygens
    in the selectivity filter glycine (G1) and tyrosine (Y) residues (GYG motif).

    Identifies 'on'/'off' transitions based on a radius threshold and calculates
    the duration spent in the flipped state.

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
    dict: Dictionary containing detailed gyration and state analysis statistics.
          Includes mean/std gyration radius for G1/Y, and counts for 'on'/'off'
          flips, mean/std duration, and list of individual durations for both.
        Example structure:
        {
            'mean_gyration_g1': ..., 'std_gyration_g1': ..., 'mean_gyration_y': ..., 'std_gyration_y': ...,
            'g1_on_flips': ..., 'g1_off_flips': ..., 'g1_mean_flip_duration_ns': ..., 'g1_std_flip_duration_ns': ...,
            'g1_flip_durations_ns': [...],
            'y_on_flips': ..., 'y_off_flips': ..., 'y_mean_flip_duration_ns': ..., 'y_std_flip_duration_ns': ...,
            'y_flip_durations_ns': [...]
        }
    """
    # Set up logging
    logger = setup_system_logger(run_dir)
    if logger is None:
        logger = module_logger  # Fallback to module logger

    logger.info("Starting carbonyl gyration (ρ) and state analysis for G1 and Y...")

    # Validate input files
    if psf_file is None:
        psf_file = os.path.join(run_dir, "step5_input.psf")
    if dcd_file is None:
        dcd_file = os.path.join(run_dir, "MD_Aligned.dcd")

    if not os.path.exists(psf_file) or not os.path.exists(dcd_file):
        logger.error(f"PSF or DCD file not found: {psf_file}, {dcd_file}")
        return {
            'mean_gyration_g1': np.nan, 'std_gyration_g1': np.nan,
            'mean_gyration_y': np.nan, 'std_gyration_y': np.nan,
            'g1_on_flips': 0, 'g1_off_flips': 0, 'g1_mean_flip_duration_ns': np.nan, 'g1_std_flip_duration_ns': np.nan,
            'g1_flip_durations_ns': [],
            'y_on_flips': 0, 'y_off_flips': 0, 'y_mean_flip_duration_ns': np.nan, 'y_std_flip_duration_ns': np.nan,
            'y_flip_durations_ns': []
        }

    # Create output directory
    output_dir = os.path.join(run_dir, "gyration_analysis")
    os.makedirs(output_dir, exist_ok=True)

    # Load trajectory
    try:
        u = mda.Universe(psf_file, dcd_file)
        n_frames = len(u.trajectory)
        dt = u.trajectory.dt / 1000.0 # Time step in ns (assuming ps originally)
        logger.info(f"Loaded trajectory with {n_frames} frames, dt={dt:.4f} ns")
    except Exception as e:
        logger.error(f"Error loading trajectory: {e}")
        return {
            'mean_gyration_g1': np.nan, 'std_gyration_g1': np.nan,
            'mean_gyration_y': np.nan, 'std_gyration_y': np.nan,
            'g1_on_flips': 0, 'g1_off_flips': 0, 'g1_mean_flip_duration_ns': np.nan, 'g1_std_flip_duration_ns': np.nan,
            'g1_flip_durations_ns': [],
            'y_on_flips': 0, 'y_off_flips': 0, 'y_mean_flip_duration_ns': np.nan, 'y_std_flip_duration_ns': np.nan,
            'y_flip_durations_ns': []
        }

    # Get filter residues
    filter_residues = find_filter_residues(u, logger)
    if not filter_residues:
        logger.error("Failed to identify filter residues")
        return {
            'mean_gyration_g1': np.nan, 'std_gyration_g1': np.nan,
            'mean_gyration_y': np.nan, 'std_gyration_y': np.nan,
            'g1_on_flips': 0, 'g1_off_flips': 0, 'g1_mean_flip_duration_ns': np.nan, 'g1_std_flip_duration_ns': np.nan,
            'g1_flip_durations_ns': [],
            'y_on_flips': 0, 'y_off_flips': 0, 'y_mean_flip_duration_ns': np.nan, 'y_std_flip_duration_ns': np.nan,
            'y_flip_durations_ns': []
        }

    # Setup gyration analysis
    frame_indices = []
    pore_centers = []  # Store pore center for each frame
    gyration_data = {
        'g1': {chain: [] for chain in filter_residues.keys()},
        'y': {chain: [] for chain in filter_residues.keys()} # Add storage for Y
    }

    # Build selections for G1 and Y carbonyl oxygens
    g1_oxygen_selections = {}
    y_oxygen_selections = {} # Add selections for Y

    for chain, info in filter_residues.items():
        if len(info) >= 3:  # Ensure we have at least up to G1 (TVG)
            # G1 is index 2, Y is index 1 in TVGYG
            g1_resid = info[2]
            y_resid = info[1] # Assuming Y is at index 1

            # Store selection strings for oxygens
            g1_sel = f"segid {chain} and resid {g1_resid} and name O"
            y_sel = f"segid {chain} and resid {y_resid} and name O" # Selection for Y carbonyl
            g1_oxygen_selections[chain] = g1_sel
            y_oxygen_selections[chain] = y_sel

            logger.debug(f"Chain {chain}: G1 oxygen selection: {g1_sel}")
            logger.debug(f"Chain {chain}: Y oxygen selection: {y_sel}")
        else:
            logger.warning(f"Chain {chain} has fewer than 3 filter residues. Skipping gyration.")

    # Iterate through trajectory
    logger.info("Calculating G1 and Y carbonyl gyration radii...")
    for ts in tqdm(u.trajectory, desc="Gyration analysis", unit="frame"):
        frame_idx = ts.frame
        frame_indices.append(frame_idx)

        # Calculate pore center for this frame (geometric center of filter G1 CA atoms)
        pore_center = calculate_pore_center(u, filter_residues)
        pore_centers.append(pore_center)

        # Calculate gyration radii for G1 and Y oxygens in each chain
        for chain in filter_residues.keys():
            # G1 calculation
            if chain in g1_oxygen_selections:
                g1_oxygen = u.select_atoms(g1_oxygen_selections[chain])
                if len(g1_oxygen) == 1:
                    g1_dist = np.linalg.norm(g1_oxygen.positions[0] - pore_center)
                    gyration_data['g1'][chain].append(g1_dist)
                else:
                    gyration_data['g1'][chain].append(np.nan)

            # Y calculation (added)
            if chain in y_oxygen_selections:
                y_oxygen = u.select_atoms(y_oxygen_selections[chain])
                if len(y_oxygen) == 1:
                    y_dist = np.linalg.norm(y_oxygen.positions[0] - pore_center)
                    gyration_data['y'][chain].append(y_dist)
                else:
                    gyration_data['y'][chain].append(np.nan)

    # Convert to time points
    time_points = frames_to_time(frame_indices)

    # Analyze carbonyl states
    logger.info(f"Analyzing G1 and Y carbonyl states using threshold {GYRATION_FLIP_THRESHOLD} Å and tolerance {GYRATION_FLIP_TOLERANCE_FRAMES} frames...")
    state_analysis_results = analyze_carbonyl_states(
        gyration_data, time_points,
        GYRATION_FLIP_THRESHOLD,
        GYRATION_FLIP_TOLERANCE_FRAMES,
        FRAMES_PER_NS
    )

    # Save gyration data to CSV (updated function call)
    save_gyration_data(gyration_data, time_points, output_dir)

    # Create plots (updated function call)
    plot_gyration_data(gyration_data, time_points, state_analysis_results, system_type, output_dir)
    plot_flip_durations(state_analysis_results, output_dir)

    # Calculate statistics (updated function call)
    final_stats = calculate_gyration_statistics(gyration_data, state_analysis_results)

    g1_flips = final_stats.get('g1_on_flips', 0)
    y_flips = final_stats.get('y_on_flips', 0)
    logger.info(f"Gyration/State analysis complete. G1 On Flips: {g1_flips}, Y On Flips: {y_flips}")

    return final_stats

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

def analyze_carbonyl_states(gyration_data, time_points, threshold, tolerance_frames, frames_per_ns):
    """
    Analyzes carbonyl states (normal/flipped) based on gyration radius threshold,
    confirms states using a frame tolerance, identifies confirmed 'on'/'off' transitions,
    and calculates confirmed flip durations for G1 and Y.

    Parameters:
    -----------
    gyration_data : dict
        Dictionary containing G1 and Y gyration radii by chain {'g1': {chain: [...]}, 'y': {chain: [...]}}
    time_points : numpy.ndarray
        Array of time points corresponding to the gyration data.
    threshold : float
        Gyration radius threshold (Å) to define the 'flipped' state (> threshold).
    tolerance_frames : int
        Minimum number of consecutive frames a state must persist to be confirmed.
    frames_per_ns : int
        Number of frames per nanosecond, used to calculate dt.

    Returns:
    --------
    dict: A dictionary containing detailed state analysis results:
        {
            'g1': {
                'on_flips': {chain: count},
                'off_flips': {chain: count},
                'flip_durations_ns': {chain: [list_of_durations]},
                'mean_flip_duration_ns': {chain: mean_duration},
                'std_flip_duration_ns': {chain: std_duration},
                'events': {chain: [list_of_event_dicts]} # 'on'/'off' with time
            },
            'y': {
                'on_flips': {chain: count},
                'off_flips': {chain: count},
                'flip_durations_ns': {chain: [list_of_durations]},
                'mean_flip_duration_ns': {chain: mean_duration},
                'std_flip_duration_ns': {chain: std_duration},
                'events': {chain: [list_of_event_dicts]} # 'on'/'off' with time
            }
        }
    """
    results = {'g1': {}, 'y': {}}
    logger = module_logger # Use module logger

    if frames_per_ns <= 0:
        logger.error("Invalid FRAMES_PER_NS configuration. Cannot calculate durations.")
        return results
    dt = 1.0 / frames_per_ns # Time step in ns

    for residue_type in ['g1', 'y']:
        res_results = {
            'on_flips': {}, 'off_flips': {}, 'flip_durations_ns': {},
            'mean_flip_duration_ns': {}, 'std_flip_duration_ns': {},
            'events': {}
        }
        for chain, radii in gyration_data.get(residue_type, {}).items():
            radii_array = np.array(radii)
            n_frames = len(radii_array)

            if n_frames < tolerance_frames:
                logger.debug(f"Chain {chain} {residue_type.upper()}: Not enough frames ({n_frames}) for tolerance ({tolerance_frames}). Skipping state analysis.")
                # Initialize empty results for this chain
                res_results['on_flips'][chain] = 0
                res_results['off_flips'][chain] = 0
                res_results['flip_durations_ns'][chain] = []
                res_results['mean_flip_duration_ns'][chain] = np.nan
                res_results['std_flip_duration_ns'][chain] = np.nan
                res_results['events'][chain] = []
                continue

            # 1. Determine state for each frame (0: normal, 1: flipped, -1: NaN)
            states = np.full(n_frames, -1, dtype=int)
            valid_indices = np.where(np.isfinite(radii_array))[0]
            states[valid_indices] = (radii_array[valid_indices] > threshold).astype(int)

            # 2. Identify contiguous blocks and filter by tolerance
            confirmed_blocks = [] # Stores tuples: (start_frame, end_frame, state)
            current_state = -2 # Init state different from 0, 1, -1
            block_start = -1
            consecutive_frames = 0

            for i in range(n_frames):
                frame_state = states[i]
                if frame_state == -1: # Handle NaN
                    if block_start != -1 and consecutive_frames >= tolerance_frames:
                        # End the previous confirmed block if NaN encountered
                        confirmed_blocks.append((block_start, i - 1, current_state))
                    # Reset for NaN
                    current_state = -2
                    block_start = -1
                    consecutive_frames = 0
                    continue

                if frame_state == current_state:
                    consecutive_frames += 1
                else:
                    # State changed, check previous block
                    if block_start != -1 and consecutive_frames >= tolerance_frames:
                         confirmed_blocks.append((block_start, i - 1, current_state))
                    # Start new block
                    current_state = frame_state
                    block_start = i
                    consecutive_frames = 1

            # Check the last block after loop finishes
            if block_start != -1 and consecutive_frames >= tolerance_frames:
                confirmed_blocks.append((block_start, n_frames - 1, current_state))

            # 3. Identify confirmed transitions and calculate durations
            on_count = 0
            off_count = 0
            durations = []
            events = []
            last_confirmed_state = -2

            for i, block in enumerate(confirmed_blocks):
                start_frame, end_frame, state = block
                block_time = time_points[start_frame]
                block_radius = radii_array[start_frame] # Radius at the start of the block

                if state == 1: # Confirmed Flipped Block
                    duration_frames = (end_frame - start_frame) + 1
                    duration_ns = duration_frames * dt
                    durations.append(duration_ns)
                    # Record duration associated with the *end* of the flipped state (next block start)
                    # We'll add it to the 'off' event later if applicable

                    if last_confirmed_state == 0: # Previous was confirmed normal
                        on_count += 1
                        events.append({'type': 'on', 'frame': start_frame, 'time': block_time, 'radius': block_radius})

                elif state == 0: # Confirmed Normal Block
                    if last_confirmed_state == 1: # Previous was confirmed flipped
                        off_count += 1
                        # Get duration of the preceding flipped block
                        preceding_duration = durations[-1] if durations else np.nan # Get last calculated duration
                        events.append({'type': 'off', 'frame': start_frame, 'time': block_time, 'radius': block_radius, 'duration_ns': preceding_duration})

                last_confirmed_state = state # Update last confirmed state

            # Store results for the chain
            res_results['on_flips'][chain] = on_count
            res_results['off_flips'][chain] = off_count
            res_results['flip_durations_ns'][chain] = durations
            res_results['mean_flip_duration_ns'][chain] = np.mean(durations) if durations else np.nan
            res_results['std_flip_duration_ns'][chain] = np.std(durations) if len(durations) > 1 else np.nan
            res_results['events'][chain] = events

        results[residue_type] = res_results

    return results

def save_gyration_data(gyration_data, time_points, output_dir):
    """
    Save G1 and Y gyration data to CSV files.

    Parameters:
    -----------
    gyration_data : dict
        Dictionary containing gyration radii for G1 and Y by chain
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

    # Create a DataFrame for Y (added)
    y_data = {'Time (ns)': time_points}
    for chain, radii in gyration_data['y'].items():
        y_data[f'{chain}_Y'] = radii

    y_df = pd.DataFrame(y_data)
    y_csv_path = os.path.join(output_dir, "Y_gyration_radii.csv")
    try:
        y_df.to_csv(y_csv_path, index=False, float_format='%.4f', na_rep='NaN')
        module_logger.info(f"Saved Y gyration data to {y_csv_path}")
    except Exception as e:
        module_logger.error(f"Failed to save Y gyration CSV: {e}")

def plot_gyration_data(gyration_data, time_points, state_analysis_results, system_type, output_dir):
    """
    Create plots for G1 and Y gyration radii data, marking 'on'/'off' events.

    Parameters:
    -----------
    gyration_data : dict
        Dictionary containing G1 and Y gyration radii by chain
    time_points : numpy.ndarray
        Array of time points
    state_analysis_results : dict
        Dictionary containing state analysis results for G1 and Y
    system_type : str
        'toxin' or 'control' to categorize the system
    output_dir : str
        Directory to save the plots
    """
    logger = module_logger
    logger.info("Plotting gyration radii time series with state events...")

    # Plot settings
    plt.rcParams['figure.figsize'] = (12, 6)

    for res_type, res_key in [('G1', 'g1'), ('Y', 'y')]:
        fig, ax = plt.subplots()
        has_events_to_mark = False

        for chain, radii in gyration_data.get(res_key, {}).items():
            line, = ax.plot(time_points, radii, label=f'{chain} {res_type}')
            color = line.get_color()

            # Mark 'on' and 'off' events
            events = state_analysis_results.get(res_key, {}).get('events', {}).get(chain, [])
            if events:
                 has_events_to_mark = True
            for event in events:
                if event['type'] == 'on':
                    # Dashed line for 'on' flip start
                    ax.axvline(x=event['time'], color=color, linestyle='--', alpha=0.6, linewidth=1.0)
                    # Marker at the event point
                    ax.plot(event['time'], event['radius'], '^', color=color, markersize=5, label='_nolegend_') # Up triangle for ON
                elif event['type'] == 'off':
                    # Dotted line for 'off' flip end
                    ax.axvline(x=event['time'], color=color, linestyle=':', alpha=0.6, linewidth=1.0)
                    # Marker at the event point
                    ax.plot(event['time'], event['radius'], 'v', color=color, markersize=5, label='_nolegend_') # Down triangle for OFF

        ax.set_xlabel('Time (ns)')
        ax.set_ylabel(f'{res_type} Gyration Radius (Å)')
        ax.set_title(f'{system_type.capitalize()} System - {res_type} Carbonyl Gyration Radius & State Transitions')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add threshold line
        ax.axhline(y=GYRATION_FLIP_THRESHOLD, color='red', linestyle='--', alpha=0.7, linewidth=1, label=f'Flip Threshold ({GYRATION_FLIP_THRESHOLD} Å)')

        # Add average line
        all_radii = np.concatenate([np.array(r) for r in gyration_data.get(res_key, {}).values()])
        if all_radii.size > 0 and np.any(np.isfinite(all_radii)):
            avg_radius = np.nanmean(all_radii)
            ax.axhline(y=avg_radius, color='black', linestyle=':', alpha=0.7, label=f'Avg {res_type} ρ: {avg_radius:.2f} Å')

        ax.legend() # Update legend
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"{res_type}_gyration_radii.png")
        try:
            plt.savefig(plot_path, dpi=200)
            logger.info(f"Saved {res_type} gyration plot to {plot_path}")
        except Exception as e:
            logger.error(f"Failed to save {res_type} gyration plot: {e}")
        plt.close(fig)

def calculate_gyration_statistics(gyration_data, state_analysis_results):
    """
    Calculate final summary statistics including gyration radius and state analysis.

    Parameters:
    -----------
    gyration_data : dict
        Dictionary containing G1 and Y gyration radii by chain.
    state_analysis_results : dict
        Dictionary containing detailed state analysis results from analyze_carbonyl_states.

    Returns:
    --------
    dict: Aggregated statistics ready for the main summary file.
    """
    stats = {}

    # --- Calculate Mean/Std Gyration Radius --- (As before)
    all_g1_radii = np.concatenate([np.array(radii) for radii in gyration_data.get('g1', {}).values()])
    stats['mean_gyration_g1'] = np.nanmean(all_g1_radii) if all_g1_radii.size > 0 else np.nan
    stats['std_gyration_g1'] = np.nanstd(all_g1_radii) if all_g1_radii.size > 0 else np.nan

    all_y_radii = np.concatenate([np.array(radii) for radii in gyration_data.get('y', {}).values()])
    stats['mean_gyration_y'] = np.nanmean(all_y_radii) if all_y_radii.size > 0 else np.nan
    stats['std_gyration_y'] = np.nanstd(all_y_radii) if all_y_radii.size > 0 else np.nan

    # --- Aggregate State Analysis Results --- (New)
    for residue_type in ['g1', 'y']:
        res_results = state_analysis_results.get(residue_type, {})

        # Sum flips across chains
        stats[f'{residue_type}_on_flips'] = sum(res_results.get('on_flips', {}).values())
        stats[f'{residue_type}_off_flips'] = sum(res_results.get('off_flips', {}).values())

        # Concatenate durations from all chains
        all_durations = []
        for chain_durations in res_results.get('flip_durations_ns', {}).values():
            all_durations.extend(chain_durations)
        stats[f'{residue_type}_flip_durations_ns'] = all_durations

        # Calculate overall mean/std duration
        stats[f'{residue_type}_mean_flip_duration_ns'] = np.mean(all_durations) if all_durations else np.nan
        stats[f'{residue_type}_std_flip_duration_ns'] = np.std(all_durations) if len(all_durations) > 1 else np.nan

    # Add a simple total flip count for backward compatibility/simplicity if needed?
    # stats['flips_detected'] = stats.get('g1_on_flips', 0)

    return stats

def plot_flip_durations(state_analysis_results, output_dir):
    """Plots flip duration distributions for G1 and Y."""
    logger = module_logger # Use module logger
    logger.info("Plotting flip duration distributions...")

    all_durations = {'G1': [], 'Y': []}
    all_chains = {'G1': [], 'Y': []}

    for res_type, res_key in [('G1', 'g1'), ('Y', 'y')]:
        if res_key in state_analysis_results:
            for chain, durations in state_analysis_results[res_key].get('flip_durations_ns', {}).items():
                all_durations[res_type].extend(durations)
                all_chains[res_type].extend([f"{chain}" for _ in durations]) # Track chain for potential coloring

    # Combine into a DataFrame for plotting
    plot_data = []
    for res_type in ['G1', 'Y']:
        for duration, chain in zip(all_durations[res_type], all_chains[res_type]):
             plot_data.append({'Residue': res_type, 'Duration (ns)': duration, 'Chain': chain})

    if not plot_data:
        logger.warning("No flip duration data to plot.")
        return

    df = pd.DataFrame(plot_data)

    # Define custom palettes
    # Pastel: Red (index 3), Blue (index 0)
    # Bright: Red (index 3), Blue (index 0)
    try:
        pastel_palette_map = {'G1': sns.color_palette("pastel")[3], 'Y': sns.color_palette("pastel")[0]}
        bright_palette_map = {'G1': sns.color_palette("bright")[3], 'Y': sns.color_palette("bright")[0]}
        marker_size = 6 # Increased marker size
        edge_color = 'black'
        line_width = 0.5 # Edge width
    except IndexError:
         logger.warning("Seaborn palette index out of range, using default colors.")
         pastel_palette_map = None
         bright_palette_map = None
         marker_size = 5 # Default size
         edge_color = None
         line_width = 0


    plt.figure(figsize=(8, 6))
    # Violin Plot
    sns.violinplot(x='Residue', y='Duration (ns)', data=df,
                   inner=None, palette=pastel_palette_map, cut=0)
    # Swarm Plot (Scatter overlay)
    sns.swarmplot(x='Residue', y='Duration (ns)', data=df,
                  size=marker_size, palette=bright_palette_map,
                  edgecolor=edge_color, linewidth=line_width)

    plt.title('Distribution of Carbonyl Flip Durations')
    plt.ylabel('Duration in Flipped State (ns)')
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "Flip_Duration_Distribution.png")
    try:
        plt.savefig(plot_path, dpi=200)
        logger.info(f"Saved flip duration distribution plot to {plot_path}")
    except Exception as e:
        logger.error(f"Failed to save flip duration plot: {e}")
    plt.close()