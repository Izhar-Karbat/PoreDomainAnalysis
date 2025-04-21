"""
DW-Gate State Analysis: Calculate distances, determine state, and derive kinetics.
"""

import os
import logging
from collections import defaultdict
from typing import Dict, Any

import MDAnalysis as mda
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns # Import seaborn

try:
    # Relative imports within the pore_analysis package
    from ..ion_analysis.filter_structure import find_filter_residues # For identifying chains
    from .residue_identification import find_gate_residues_for_chain, GateResidues
    from ...core.utils import frames_to_time # Navigate up two levels to core
    from ...core.config import DW_GATE_TOLERANCE_FRAMES # Import tolerance constant
except ImportError as e:
    print(f"Error importing dependencies in dw_gate_state.py: {e}")
    # Re-raise to ensure failure if imports are broken
    raise

logger = logging.getLogger(__name__)

# Define constants
DISTANCE_THRESHOLD = 3.5 # Angstrom
CLOSED_STATE = "closed"
OPEN_STATE = "open"

# --- Plotting Helpers (Minimalist, adapt if plot_utils exist elsewhere) ---
def setup_plot(figsize=(8, 3)):
    """Basic plot setup."""
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax

def save_plot(fig, path, dpi=200):
    """Save plot with error handling."""
    try:
        # Ensure the plot directory exists if it's not the current one
        # plot_dir = os.path.dirname(path)
        # if plot_dir:
        #      os.makedirs(plot_dir, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved plot: {path}")
    except Exception as e:
        logger.error(f"Failed to save plot {path}: {e}")
    finally:
        plt.close(fig)

# --- New Improved Duration Plot Function (User Provided) ---
def plot_improved_duration_distribution(open_durations_by_chain, closed_durations_by_chain, valid_chain_ids, output_dir):
    """Create an improved duration distribution visualization with separate subplots and CDFs."""

    # Create a figure with 2 rows (open/closed) and 2 columns (histogram/CDF)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

    # Get all durations for proper bin scaling
    all_open_durations = [dur for subdurs in open_durations_by_chain.values() for dur in subdurs]
    all_closed_durations = [dur for subdurs in closed_durations_by_chain.values() for dur in subdurs]

    # Handle cases with no data for one state type gracefully
    if not all_open_durations and not all_closed_durations:
        logger.warning("Insufficient data for duration distribution plot - no open or closed events found.")
        plt.close(fig) # Close the empty figure
        return

    # Determine overall min/max for consistent binning, handle empty lists
    all_durations_flat = all_open_durations + all_closed_durations
    min_dur = max(0.01, min(all_durations_flat) if all_durations_flat else 0.01)
    max_dur = max(1.0, max(all_durations_flat) if all_durations_flat else 1.0)
    bins = np.logspace(np.log10(min_dur), np.log10(max_dur), num=25)

    # Color map for chains - Use seaborn pastel
    sns_pastel = sns.color_palette("pastel", n_colors=max(4, len(valid_chain_ids))) # Get at least 4 pastel colors
    chain_color_map = {chain_id[-1]: color for chain_id, color in zip(valid_chain_ids, sns_pastel)}
    chain_chars_plot = sorted([c[-1] for c in valid_chain_ids]) # Define the list of chain chars to iterate over

    # Ensure consistent x-axis ranges
    x_min = min_dur * 0.9
    x_max = max_dur * 1.1

    # Create legend handles & labels for the consolidated legend
    legend_handles, legend_labels = [], []

    # --- Plot Open States --- #
    ax_open_hist = axes[0, 0]
    ax_open_cdf = axes[0, 1]
    plotted_open = False

    for i, chain_char in enumerate(chain_chars_plot):
        chain_color = chain_color_map.get(chain_char, sns_pastel[i % len(sns_pastel)])
        chain_open_durs = open_durations_by_chain.get(chain_char, [])
        if chain_open_durs:
            plotted_open = True
            # Histogram
            counts, _, patches = ax_open_hist.hist(
                chain_open_durs,
                bins=bins,
                alpha=0.7,
                color=chain_color,
                label=f'Chain {chain_char}',
                density=True
            )
            # Store for legend
            if chain_char not in [label.split()[-1] for label in legend_labels]:
                legend_handles.append(patches[0])
                legend_labels.append(f'Chain {chain_char}')

            # Add mean/median markers with annotations
            mean_val = np.mean(chain_open_durs)
            median_val = np.median(chain_open_durs)
            
            # Add mean line with annotation
            ax_open_hist.axvline(mean_val, color=chain_color, linestyle='-', alpha=0.8)
            ax_open_hist.text(mean_val*1.1, 0.9*ax_open_hist.get_ylim()[1], 
                              f'Mean: {mean_val:.2f} ns', color=chain_color, 
                              fontsize=8, ha='left', rotation=45, va='top')
            
            # Add median line with different style
            ax_open_hist.axvline(median_val, color=chain_color, linestyle=':', alpha=0.8)

            # CDF
            data_sorted = np.sort(chain_open_durs)
            y = np.arange(1, len(data_sorted) + 1) / len(data_sorted)
            ax_open_cdf.step(data_sorted, y, color=chain_color)
            
            # Add median marker on CDF (where y=0.5)
            ax_open_cdf.plot(median_val, 0.5, 'o', color=chain_color, markersize=5, alpha=0.7)

    ax_open_hist.set_title('Open State Duration Distribution')
    ax_open_hist.set_xlabel('Duration (ns)')
    ax_open_hist.set_ylabel('Probability Density')
    ax_open_hist.set_xscale('log')
    # Lighter grid for less distraction
    ax_open_hist.grid(True, which='major', linestyle='--', alpha=0.3)
    ax_open_hist.grid(False, which='minor')
    ax_open_hist.set_xlim(x_min, x_max)

    ax_open_cdf.set_title('Open State Duration CDF')
    ax_open_cdf.set_xlabel('Duration (ns)')
    ax_open_cdf.set_ylabel('Cumulative Probability')
    ax_open_cdf.set_xscale('log')
    ax_open_cdf.grid(True, which='major', linestyle='--', alpha=0.3)
    ax_open_cdf.grid(False, which='minor')
    ax_open_cdf.set_xlim(x_min, x_max)

    if not plotted_open:
        ax_open_hist.text(0.5, 0.5, 'No Open Events', horizontalalignment='center', 
                          verticalalignment='center', transform=ax_open_hist.transAxes)
        ax_open_cdf.text(0.5, 0.5, 'No Open Events', horizontalalignment='center', 
                         verticalalignment='center', transform=ax_open_cdf.transAxes)

    # --- Plot Closed States --- #
    ax_closed_hist = axes[1, 0]
    ax_closed_cdf = axes[1, 1]
    plotted_closed = False

    for i, chain_char in enumerate(chain_chars_plot):
        chain_color = chain_color_map.get(chain_char, sns_pastel[i % len(sns_pastel)])
        chain_closed_durs = closed_durations_by_chain.get(chain_char, [])
        if chain_closed_durs:
            plotted_closed = True
            # Histogram
            counts, _, patches = ax_closed_hist.hist(
                chain_closed_durs,
                bins=bins,
                alpha=0.7,
                color=chain_color,
                density=True
            )

            # Add mean/median markers with annotations
            mean_val = np.mean(chain_closed_durs)
            median_val = np.median(chain_closed_durs)
            
            # Add mean line with annotation
            ax_closed_hist.axvline(mean_val, color=chain_color, linestyle='-', alpha=0.8)
            ax_closed_hist.text(mean_val*1.1, 0.9*ax_closed_hist.get_ylim()[1], 
                                f'Mean: {mean_val:.2f} ns', color=chain_color, 
                                fontsize=8, ha='left', rotation=45, va='top')
            
            # Add median line with different style
            ax_closed_hist.axvline(median_val, color=chain_color, linestyle=':', alpha=0.8)

            # CDF
            data_sorted = np.sort(chain_closed_durs)
            y = np.arange(1, len(data_sorted) + 1) / len(data_sorted)
            ax_closed_cdf.step(data_sorted, y, color=chain_color)
            
            # Add median marker on CDF (where y=0.5)
            ax_closed_cdf.plot(median_val, 0.5, 'o', color=chain_color, markersize=5, alpha=0.7)

    ax_closed_hist.set_title('Closed State Duration Distribution')
    ax_closed_hist.set_xlabel('Duration (ns)')
    ax_closed_hist.set_ylabel('Probability Density')
    ax_closed_hist.set_xscale('log')
    ax_closed_hist.grid(True, which='major', linestyle='--', alpha=0.3)
    ax_closed_hist.grid(False, which='minor')
    ax_closed_hist.set_xlim(x_min, x_max)

    ax_closed_cdf.set_title('Closed State Duration CDF')
    ax_closed_cdf.set_xlabel('Duration (ns)')
    ax_closed_cdf.set_ylabel('Cumulative Probability')
    ax_closed_cdf.set_xscale('log')
    ax_closed_cdf.grid(True, which='major', linestyle='--', alpha=0.3)
    ax_closed_cdf.grid(False, which='minor')
    ax_closed_cdf.set_xlim(x_min, x_max)

    if not plotted_closed:
        ax_closed_hist.text(0.5, 0.5, 'No Closed Events', horizontalalignment='center', 
                            verticalalignment='center', transform=ax_closed_hist.transAxes)
        ax_closed_cdf.text(0.5, 0.5, 'No Closed Events', horizontalalignment='center', 
                           verticalalignment='center', transform=ax_closed_cdf.transAxes)

    # Add consolidated legend with chain information
    if legend_handles:
        fig.legend(
            handles=legend_handles, 
            labels=legend_labels, 
            loc='center right', 
            fontsize='medium', 
            title="Chains",
            bbox_to_anchor=(1.0, 0.5)
        )
        
    # Add explanatory note for line styles
    line_styles_ax = fig.add_axes([0.85, 0.3, 0.1, 0.1])  # [left, bottom, width, height]
    line_styles_ax.axis('off')
    line_styles_ax.plot([0.1, 0.4], [0.7, 0.7], 'k-', label='Mean')
    line_styles_ax.plot([0.1, 0.4], [0.3, 0.3], 'k:', label='Median')
    line_styles_ax.legend(loc='center', fontsize='small', title="Statistics")

    # Add layout improvements
    fig.suptitle('DW Gate Event Duration Analysis', fontsize=16, y=0.98)

    # Save the figure
    save_plot(fig, os.path.join(output_dir, "dw_gate_duration_analysis.png"), dpi=300)

# --- New Plot Function: Idealized vs Actual Distance ---
def plot_dw_distance_vs_state(df_timeseries, valid_chain_ids, DISTANCE_THRESHOLD, chain_color_map, sns_pastel, output_dir):
    """Generates a stacked plot showing raw distance vs. idealized state for each chain."""
    logger.info("Generating DW Gate distance vs. state plot...")
    n_chains = len(valid_chain_ids)
    if n_chains == 0 or df_timeseries.empty:
        logger.warning("Skipping distance vs state plot: No valid chains or timeseries data.")
        return

    try:
        fig, axes = plt.subplots(n_chains, 1, figsize=(10, 2 * n_chains), sharex=True, constrained_layout=True)
        if n_chains == 1:
            axes = [axes]

        chain_chars_plot = sorted([c[-1] for c in valid_chain_ids])
        min_time = df_timeseries["time_ns"].min()
        max_time = df_timeseries["time_ns"].max()

        # Define Y-positions for idealized states (adjust offset if needed)
        state_y_pos = {CLOSED_STATE: DISTANCE_THRESHOLD, OPEN_STATE: DISTANCE_THRESHOLD + 0.5}
        state_colors = {CLOSED_STATE: 'darkred', OPEN_STATE: 'darkgreen'}

        for i, chain_char in enumerate(chain_chars_plot):
            ax = axes[i]
            df_ch = df_timeseries[df_timeseries["chain"] == chain_char].copy()

            if not df_ch.empty:
                # Plot raw distance (actual)
                ax.plot(df_ch["time_ns"], df_ch["distance"],
                        label="Actual Distance", lw=0.8, alpha=0.6,
                        color=chain_color_map.get(chain_char, sns_pastel[i % len(sns_pastel)]))

                # Plot idealized state
                # Group by consecutive states
                df_ch['state_block'] = (df_ch['state'] != df_ch['state'].shift()).cumsum()
                for _, block in df_ch.groupby('state_block'):
                    start_time = block['time_ns'].iloc[0]
                    end_time = block['time_ns'].iloc[-1]
                    state = block['state'].iloc[0]
                    y_val = state_y_pos[state]
                    color = state_colors[state]
                    # Plot a thicker horizontal line segment for the state duration
                    ax.plot([start_time, end_time], [y_val, y_val], lw=3, color=color, solid_capstyle='butt')

                # Add threshold line
                ax.axhline(DISTANCE_THRESHOLD, color='grey', linestyle=':', lw=1.0, label=f'Threshold ({DISTANCE_THRESHOLD}Å)')

                # Axis labels and limits
                ax.set_ylabel(f"Chain {chain_char} Dist (Å)")
                ax.set_ylim(bottom=0)
                # Determine reasonable upper ylim
                max_dist = df_ch["distance"].max()
                ax.set_ylim(top=max(max_dist * 1.1, DISTANCE_THRESHOLD + 2.0))
                ax.grid(True, axis='y', linestyle=':', alpha=0.5)
            else:
                ax.text(0.5, 0.5, f'No data for Chain {chain_char}', ha='center', va='center', transform=ax.transAxes)

            # Remove x-tick labels for all but the bottom plot
            if i < n_chains - 1:
                ax.tick_params(labelbottom=False)

        # Common X label
        axes[-1].set_xlabel("Time (ns)")
        fig.suptitle("DW Gate Distance (Actual vs. Idealized State)", y=1.02)

        # Create a custom legend for the figure
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='gray', lw=0.8, alpha=0.6, label='Actual Distance'),
            Line2D([0], [0], color=state_colors[CLOSED_STATE], lw=3, label='Idealized Closed'),
            Line2D([0], [0], color=state_colors[OPEN_STATE], lw=3, label='Idealized Open'),
            Line2D([0], [0], color='grey', linestyle=':', lw=1.0, label=f'Threshold ({DISTANCE_THRESHOLD}Å)')
        ]
        fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.1, 0.95), fontsize='small')

        # Save plot
        plot_filename = "dw_distance_state_overlay.png"
        save_plot(fig, os.path.join(output_dir, plot_filename))

    except Exception as e:
        logger.error(f"Failed to generate distance vs state plot: {e}", exc_info=True)

# --- Main Analysis Function ---
def analyse_dw_gates(
    run_dir: str,
    psf_file: str,
    dcd_file: str,
    time_points: np.ndarray,
    stride: int = 1,
) -> Dict[str, Any]:
    """
    Analyzes the DW-gate hydrogen bond distance and state over time for each chain.

    Calculates:
    - Per-frame distance between Asp/Glu (OD1/OD2) and Trp (NE1).
    - Per-frame state ("open" or "closed") based on a distance threshold.
    - Per-chain closed fraction and mean distance.
    - Per-chain mean closed state dwell time.
    - Global closed fraction.

    Saves timeseries data to a subdirectory and plots directly to the run directory.

    Args:
        run_dir (str): Path to the simulation run directory.
        psf_file (str): Path to the topology (PSF) file.
        dcd_file (str): Path to the trajectory (DCD) file.
        time_points (np.ndarray): Array of time points (in ns) for the full trajectory.
        stride (int): Analysis stride. Process every nth frame.

    Returns:
        Dict[str, Any]: Dictionary containing calculated statistics, including
                        'DWhbond_closed_global', 'DWhbond_closed_per_subunit',
                        and potentially others like 'Mean_Closed_Dwell_ms_A'.
                        Returns an empty dictionary on critical failure.
    """
    logger.info(f"--- Starting DW-Gate Analysis (Stride: {stride}) ---")
    # Define output directory for CSV (plots go directly to run_dir)
    output_dir = os.path.join(run_dir, "dw_gate_analysis")
    os.makedirs(output_dir, exist_ok=True)

    # --- Load Universe --- 
    try:
        logger.debug(f"Loading Topology: {psf_file}, Trajectory: {dcd_file}")
        u = mda.Universe(psf_file, dcd_file)
        n_frames_total = len(u.trajectory)
        if n_frames_total == 0:
            raise ValueError("Trajectory contains 0 frames.")
        # Estimate dt from full time_points array if possible
        if len(time_points) == n_frames_total and n_frames_total > 1:
             dt_ns = time_points[1] - time_points[0]
             logger.info(f"Universe loaded: {n_frames_total} frames. Estimated dt={dt_ns:.6f} ns from time_points.")
        else:
            dt_ps = u.trajectory.dt
            dt_ns = dt_ps / 1000.0
            logger.warning(f"Mismatch between time_points length ({len(time_points)}) and trajectory frames ({n_frames_total}). Using trajectory dt={dt_ps} ps ({dt_ns:.6f} ns). Time axis might be inaccurate.")
            # Recalculate time_points if mismatched (use original length for safety)
            time_points = np.arange(n_frames_total) * dt_ns

        n_frames_analyzed = len(range(0, n_frames_total, stride))
        logger.info(f"Analyzing {n_frames_analyzed} frames (stride={stride}).")

    except FileNotFoundError as e:
         logger.error(f"Input file not found for DW Gate analysis: {e}")
         return {'Error': 'Input file not found'}
    except Exception as e:
        logger.error(f"Failed to load Universe for DW Gate analysis: {e}", exc_info=True)
        return {'Error': f'Universe load failed: {e}'}

    # --- Identify Filter & Gate Residues --- 
    try:
        filter_res_map = find_filter_residues(u, logger)
        if not filter_res_map:
            raise ValueError("find_filter_residues returned empty or None.")
    except Exception as e:
        logger.error(f"Failed to find filter residues for DW Gate analysis: {e}", exc_info=True)
        return {'Error': f'Filter residue ID failed: {e}'}

    valid_gates: Dict[str, GateResidues] = {}
    gate_atom_indices: Dict[str, Dict[str, int]] = defaultdict(dict)
    chain_failures = 0
    chain_ids = sorted(filter_res_map.keys())

    for segid in chain_ids:
        try:
            gate = find_gate_residues_for_chain(u, segid, filter_res_map)
            valid_gates[segid] = gate

            asp_atoms = gate.asp_glu_res.atoms.select_atoms("name OD1 OD2")
            trp_atoms = gate.trp_res.atoms.select_atoms("name NE1")

            if len(asp_atoms.select_atoms("name OD1")) == 1:
                gate_atom_indices[segid]['asp_od1'] = asp_atoms.select_atoms("name OD1").indices[0]
            else: raise ValueError("Could not find exactly one OD1 atom")
            if len(asp_atoms.select_atoms("name OD2")) == 1:
                gate_atom_indices[segid]['asp_od2'] = asp_atoms.select_atoms("name OD2").indices[0]
            else: raise ValueError("Could not find exactly one OD2 atom")
            if len(trp_atoms) == 1:
                gate_atom_indices[segid]['trp_ne1'] = trp_atoms.indices[0]
            else: raise ValueError("Could not find exactly one NE1 atom")

            logger.debug(f"Stored DW gate atom indices for {segid}")

        except ValueError as e:
            logger.warning(f"Skipping chain {segid} due to DW gate residue identification error: {e}")
            chain_failures += 1

    if chain_failures == len(chain_ids):
        logger.error("Failed to identify valid DW gate residues for ANY chain. Aborting DW Gate analysis.")
        return {'Error': 'DW Gate residue ID failed for all chains'}
    if chain_failures > 0:
        logger.warning(f"DW Gate analysis proceeding with {len(valid_gates)} chains.")

    valid_chain_ids = sorted(valid_gates.keys())

    # --- Define consistent color mapping for plots ---
    sns_pastel = sns.color_palette("pastel", n_colors=max(4, len(valid_chain_ids)))
    chain_color_map = {chain_id[-1]: color for chain_id, color in zip(valid_chain_ids, sns_pastel)}

    # --- Process Trajectory --- 
    timeseries_data = []
    frame_indices_analyzed = np.arange(0, n_frames_total, stride)
    time_points_analyzed = time_points[frame_indices_analyzed]

    logger.info("Calculating DW distances and states...")
    for i, ts in enumerate(tqdm(u.trajectory[::stride], total=n_frames_analyzed, desc="DW Gate Analysis")):
        frame = frame_indices_analyzed[i]
        time_ns = time_points_analyzed[i]
        positions = ts.positions

        for chain_id in valid_chain_ids:
            try:
                idx = gate_atom_indices[chain_id]
                pos_od1 = positions[idx['asp_od1']]
                pos_od2 = positions[idx['asp_od2']]
                pos_ne1 = positions[idx['trp_ne1']]

                dist1 = np.linalg.norm(pos_od1 - pos_ne1)
                dist2 = np.linalg.norm(pos_od2 - pos_ne1)
                dw_distance = min(dist1, dist2)

                state = CLOSED_STATE if dw_distance <= DISTANCE_THRESHOLD else OPEN_STATE

                timeseries_data.append({
                    "frame": frame,
                    "time_ns": time_ns,
                    "chain": chain_id[-1], # Use A, B, C, D for simplicity
                    "distance": dw_distance,
                    "state": state
                })
            except KeyError as e:
                 logger.warning(f"Frame {frame}: Missing atom index for chain {chain_id}, skipping distance calc. Error: {e}")
                 continue
            except Exception as e:
                 logger.warning(f"Frame {frame}: Error calculating distance for chain {chain_id}: {e}", exc_info=False)
                 continue

    if not timeseries_data:
        logger.error("No DW Gate timeseries data was generated. Aborting statistics calculation.")
        return {'Error': 'No timeseries data generated'}

    # --- Save Timeseries Data (to subdirectory) --- 
    df_timeseries = pd.DataFrame(timeseries_data)
    csv_path = os.path.join(output_dir, "dw_gate_timeseries.csv")
    try:
        df_timeseries.to_csv(csv_path, index=False, float_format="%.3f")
        logger.info(f"Saved DW Gate timeseries data to {csv_path}")
    except Exception as e:
        logger.error(f"Failed to save DW Gate timeseries CSV: {e}")
        # Continue with analysis even if CSV save fails?

    # --- Calculate Statistics --- 
    raw_stats: Dict[str, Any] = {}
    per_subunit_fractions = {}
    all_closed_fractions = []
    open_durations_by_chain = defaultdict(list) # New dict for per-chain data
    closed_durations_by_chain = defaultdict(list) # New dict for per-chain data
    event_summary_data = [] # List to hold data for the event summary CSV
    all_events_data = [] # New list to hold data for ALL individual events

    logger.info("Calculating DW Gate summary statistics and individual events...")
    for chain_id_full in valid_chain_ids:
        chain_char = chain_id_full[-1]
        df_chain = df_timeseries[df_timeseries["chain"] == chain_char]

        if df_chain.empty:
            logger.warning(f"No data found for chain {chain_char} after processing.")
            raw_stats[f'Mean_DW_Distance_{chain_char}'] = np.nan
            per_subunit_fractions[chain_char] = np.nan
            raw_stats[f'DW_OpenEvents_{chain_char}'] = np.nan
            raw_stats[f'DW_ClosedEvents_{chain_char}'] = np.nan
            raw_stats[f'DW_MeanOpenTime_ns_{chain_char}'] = np.nan
            raw_stats[f'DW_StdOpenTime_ns_{chain_char}'] = np.nan
            raw_stats[f'DW_MeanClosedTime_ns_{chain_char}'] = np.nan
            raw_stats[f'DW_StdClosedTime_ns_{chain_char}'] = np.nan
            continue

        raw_stats[f'Mean_DW_Distance_{chain_char}'] = float(df_chain["distance"].mean())

        closed_count = (df_chain["state"] == CLOSED_STATE).sum()
        total_count = len(df_chain)
        closed_fraction = closed_count / total_count if total_count > 0 else 0.0
        per_subunit_fractions[chain_char] = float(closed_fraction)
        all_closed_fractions.append(closed_fraction)

        # --- Event Duration Analysis (Using Tolerance) ---
        logger.debug(f"Analyzing event durations for chain {chain_char} with tolerance {DW_GATE_TOLERANCE_FRAMES} frames...")
        open_durations_frames = []
        closed_durations_frames = []
        current_state = None
        current_duration = 0
        current_start_frame = None
        current_start_time_ns = None
        prev_frame = None
        prev_time_ns = None

        time_per_frame_ns = stride * dt_ns # Calculate once per chain

        for row in df_chain.itertuples(): # Iterate over rows to get frame/time info
            frame = row.frame
            time_ns = row.time_ns
            state = row.state

            if state == current_state:
                current_duration += 1
            else:
                # End of a block, check if it met tolerance and record if so
                if current_state is not None and current_duration >= DW_GATE_TOLERANCE_FRAMES:
                    # Calculate end frame/time (use previous row's data)
                    end_frame = prev_frame
                    end_time_ns = prev_time_ns
                    duration_frames = current_duration
                    duration_ns = duration_frames * time_per_frame_ns

                    # Append to the list for detailed event CSV
                    all_events_data.append({
                        'chain': chain_char,
                        'state': current_state,
                        'start_frame': current_start_frame,
                        'end_frame': end_frame,
                        'start_time_ns': current_start_time_ns,
                        'end_time_ns': end_time_ns,
                        'duration_frames': duration_frames,
                        'duration_ns': duration_ns
                    })

                    # Append duration to lists for summary stats/plotting
                    if current_state == OPEN_STATE:
                        open_durations_frames.append(duration_frames)
                    elif current_state == CLOSED_STATE:
                        closed_durations_frames.append(duration_frames)

                # Start of a new block
                current_state = state
                current_duration = 1
                current_start_frame = frame
                current_start_time_ns = time_ns

            # Store current frame/time for the next iteration's 'end' calculation
            prev_frame = frame
            prev_time_ns = time_ns

        # Check the last block at the end of the series for this chain
        if current_state is not None and current_duration >= DW_GATE_TOLERANCE_FRAMES:
            end_frame = prev_frame # The last frame processed
            end_time_ns = prev_time_ns # The last time point processed
            duration_frames = current_duration
            duration_ns = duration_frames * time_per_frame_ns

            all_events_data.append({
                'chain': chain_char,
                'state': current_state,
                'start_frame': current_start_frame,
                'end_frame': end_frame,
                'start_time_ns': current_start_time_ns,
                'end_time_ns': end_time_ns,
                'duration_frames': duration_frames,
                'duration_ns': duration_ns
            })

            if current_state == OPEN_STATE:
                open_durations_frames.append(duration_frames)
            elif current_state == CLOSED_STATE:
                closed_durations_frames.append(duration_frames)

        # --- Calculate summary statistics from the collected valid durations ---
        n_open_events = len(open_durations_frames)
        n_closed_events = len(closed_durations_frames)

        # time_per_frame_ns = stride * dt_ns # Moved calculation up
        open_durations_ns = None # Initialize as None
        if open_durations_frames: # Check the list before converting
            open_durations_ns_array = np.array(open_durations_frames) * time_per_frame_ns
            mean_open_ns = float(np.mean(open_durations_ns_array))
            std_open_ns = float(np.std(open_durations_ns_array))
            open_durations_ns = open_durations_ns_array # Assign numpy array for extension
        else:
            mean_open_ns = 0.0
            std_open_ns = 0.0
            # open_durations_ns remains None

        closed_durations_ns = None # Initialize as None
        if closed_durations_frames: # Check the list before converting
            closed_durations_ns_array = np.array(closed_durations_frames) * time_per_frame_ns
            mean_closed_ns = float(np.mean(closed_durations_ns_array))
            std_closed_ns = float(np.std(closed_durations_ns_array))
            closed_durations_ns = closed_durations_ns_array # Assign numpy array for extension
        else:
            mean_closed_ns = 0.0
            std_closed_ns = 0.0
            # closed_durations_ns remains None

        # Append chain durations to overall lists for histogram
        # Check if the numpy arrays were created before extending
        if open_durations_ns is not None:
             open_durations_by_chain[chain_char].extend(open_durations_ns)
        if closed_durations_ns is not None:
             closed_durations_by_chain[chain_char].extend(closed_durations_ns)

        # Store results in raw_stats (to be added to final_stats later)
        # Keep these for compatibility or direct access if needed, but primary detailed
        # event data will go to the CSV.
        raw_stats[f'DW_OpenEvents_{chain_char}'] = n_open_events
        raw_stats[f'DW_ClosedEvents_{chain_char}'] = n_closed_events
        raw_stats[f'DW_MeanOpenTime_ns_{chain_char}'] = mean_open_ns
        raw_stats[f'DW_StdOpenTime_ns_{chain_char}'] = std_open_ns
        raw_stats[f'DW_MeanClosedTime_ns_{chain_char}'] = mean_closed_ns
        raw_stats[f'DW_StdClosedTime_ns_{chain_char}'] = std_closed_ns

        # Append structured data for the event summary CSV
        event_summary_data.append({
            'chain': chain_char,
            'state': OPEN_STATE,
            'event_count': n_open_events,
            'mean_duration_ns': mean_open_ns,
            'std_duration_ns': std_open_ns
        })
        event_summary_data.append({
            'chain': chain_char,
            'state': CLOSED_STATE,
            'event_count': n_closed_events,
            'mean_duration_ns': mean_closed_ns,
            'std_duration_ns': std_closed_ns
        })

        logger.debug(f"Chain {chain_char}: Open Events={n_open_events}, Mean(ns)={mean_open_ns:.3f} +/- {std_open_ns:.3f}")
        logger.debug(f"Chain {chain_char}: Closed Events={n_closed_events}, Mean(ns)={mean_closed_ns:.3f} +/- {std_closed_ns:.3f}")

    # --- Save Event Summary Data --- 
    if event_summary_data:
        df_event_summary = pd.DataFrame(event_summary_data)
        event_csv_path = os.path.join(output_dir, "dw_gate_event_summary.csv")
        try:
            df_event_summary.to_csv(event_csv_path, index=False, float_format="%.3f")
            logger.info(f"Saved DW Gate event summary data to {event_csv_path}")
        except Exception as e:
            logger.error(f"Failed to save DW Gate event summary CSV: {e}")
    else:
        logger.warning("No event summary data generated to save.")

    # --- Save All Individual Events Data --- 
    if all_events_data:
        df_all_events = pd.DataFrame(all_events_data)
        # Sort by start time for better readability
        df_all_events = df_all_events.sort_values(by='start_time_ns').reset_index(drop=True)
        all_events_csv_path = os.path.join(output_dir, "dw_gate_events.csv")
        try:
            df_all_events.to_csv(all_events_csv_path, index=False, float_format="%.3f")
            logger.info(f"Saved all individual DW Gate events data to {all_events_csv_path}")
        except Exception as e:
            logger.error(f"Failed to save all DW Gate events CSV: {e}")
    else:
        logger.warning("No individual event data generated to save.")

    # --- Combine Final Stats (for return value) --- 
    final_stats = {
        'DWhbond_closed_per_subunit': per_subunit_fractions,
        'DWhbond_closed_global': float(np.mean(all_closed_fractions)) if all_closed_fractions else np.nan
    }
    # Add the new event stats to the final dictionary
    final_stats.update(raw_stats)
    # Add tolerance value used
    final_stats['DW_GATE_TOLERANCE_FRAMES'] = DW_GATE_TOLERANCE_FRAMES

    # --- Generate Plots (Save directly to run_dir) --- 
    logger.info("Generating DW Gate plots...")
    # 1. Distance Timeseries (Stacked Layout)
    try:
        n_chains = len(valid_chain_ids)
        if n_chains > 0:
            # Create stacked subplots, sharing the x-axis
            fig1, axes1 = plt.subplots(n_chains, 1, figsize=(10, 2 * n_chains), sharex=True, constrained_layout=True)
            # If only one chain, axes1 might not be an array, handle this
            if n_chains == 1:
                axes1 = [axes1]

            chain_chars_plot = sorted([c[-1] for c in valid_chain_ids])

            for i, chain_char in enumerate(chain_chars_plot):
                ax = axes1[i]
                df_ch = df_timeseries[df_timeseries["chain"] == chain_char]
                if not df_ch.empty:
                     # Plot the distance data for the current chain
                     ax.plot(df_ch["time_ns"], df_ch["distance"], label=f"Chain {chain_char}", lw=1, color=chain_color_map.get(chain_char, sns_pastel[i % len(sns_pastel)]))
                     # Add the threshold line to each subplot
                     ax.axhline(DISTANCE_THRESHOLD, color='grey', linestyle='--', lw=1.5)
                     # Set y-axis label specific to the chain
                     ax.set_ylabel(f"Chain {chain_char} Dist (Å)")
                     # Set y-limits (optional: customize based on data range?)
                     ax.set_ylim(bottom=0)
                     ax.grid(True, axis='y', linestyle=':', alpha=0.7)
                else:
                    ax.text(0.5, 0.5, f'No data for Chain {chain_char}', 
                            horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

                # Remove x-tick labels for all but the bottom plot
                if i < n_chains - 1:
                    ax.tick_params(labelbottom=False)

            # Set common x-axis label on the bottom plot
            axes1[-1].set_xlabel("Time (ns)")
            # Set a main title for the figure
            fig1.suptitle("DW Gate Distance Over Time (Stacked)", y=1.0) # Adjust y if needed
            # Add threshold label once, maybe to the last axis or in the title?
            # Let's add it near the line on the last plot for clarity
            axes1[-1].text(time_points_analyzed.max()*0.9, DISTANCE_THRESHOLD + 0.2, f'{DISTANCE_THRESHOLD}Å Threshold', 
                           color='grey', va='bottom', ha='right', fontsize='small')
            # No need for a legend now as chains are identified by axes
            # fig1.legend(fontsize='small')
            save_plot(fig1, os.path.join(output_dir, "dw_distance_timeseries.png"))
        else:
             logger.warning("Skipping distance timeseries plot: No valid chains found.")
    except Exception as e:
         logger.error(f"Failed to generate stacked distance timeseries plot: {e}", exc_info=True)

    # 1b. Idealized vs Actual Distance Plot (NEW)
    plot_dw_distance_vs_state(
        df_timeseries,
        valid_chain_ids,
        DISTANCE_THRESHOLD,
        chain_color_map,
        sns_pastel,
        output_dir
    )

    # 2. State Heatmap
    try:
        df_pivot = df_timeseries.pivot(index="time_ns", columns="chain", values="state")
        state_map = {CLOSED_STATE: 1, OPEN_STATE: 0}
        df_numeric = df_pivot.replace(state_map).fillna(-1)
        if not df_numeric.empty:
            fig2, ax2 = setup_plot(figsize=(10, 2.5))
            cmap = mcolors.ListedColormap(['lightgrey', 'lightblue', 'darkblue'])
            bounds = [-1.5, -0.5, 0.5, 1.5]
            norm = mcolors.BoundaryNorm(bounds, cmap.N)
            im = ax2.imshow(df_numeric.T, aspect='auto', cmap=cmap, norm=norm,
                            interpolation='nearest', extent=[time_points_analyzed.min(), time_points_analyzed.max(), -0.5, len(valid_chain_ids)-0.5])
            ax2.set_yticks(np.arange(len(valid_chain_ids)))
            ax2.set_yticklabels(sorted([c[-1] for c in valid_chain_ids]))
            ax2.set_xlabel("Time (ns)")
            ax2.set_ylabel("Chain")
            ax2.set_title("DW Gate State (Closed=Blue, Open=LightBlue)")
            cbar = fig2.colorbar(im, ax=ax2, ticks=[-1, 0, 1], orientation='vertical', fraction=0.05, pad=0.04)
            cbar.ax.set_yticklabels(['Missing', 'Open', 'Closed'], fontsize='small')
            save_plot(fig2, os.path.join(output_dir, "dw_gate_state_heatmap.png"))
        else:
            logger.warning("Skipping DW Gate heatmap plot: Pivoted DataFrame is empty.")
    except Exception as e:
        logger.error(f"Failed to generate state heatmap plot: {e}", exc_info=True)

    # 3. Closed Fraction Bar Chart
    try:
        chain_chars_plot = sorted([c[-1] for c in valid_chain_ids])
        fractions = [per_subunit_fractions.get(ch, np.nan) for ch in chain_chars_plot]
        if any(not np.isnan(f) for f in fractions):
             fig3, ax3 = setup_plot(figsize=(5, 4))
             ax3.bar(chain_chars_plot, fractions, color='skyblue')
             global_avg = final_stats.get('DWhbond_closed_global', np.nan)
             if not np.isnan(global_avg):
                  ax3.axhline(global_avg, color='red', linestyle='--', lw=1, label=f"Global Avg ({global_avg:.2f})")
             ax3.set_xlabel("Chain")
             ax3.set_ylabel("Fraction of Time Closed")
             ax3.set_title("DW Gate Closed Fraction")
             ax3.set_ylim(0, 1)
             ax3.legend()
             save_plot(fig3, os.path.join(output_dir, "dw_gate_closed_fraction_bar.png"))
        else:
             logger.warning("Skipping closed fraction bar plot: No valid fraction data.")
    except Exception as e:
        logger.error(f"Failed to generate closed fraction bar plot: {e}", exc_info=True)

    # 4. Improved Event Duration Plot (NEW)
    try:
        plot_improved_duration_distribution(
            open_durations_by_chain,
            closed_durations_by_chain,
            valid_chain_ids,
            output_dir
        )
    except Exception as e:
        logger.error(f"Failed to generate improved event duration plot: {e}", exc_info=True)

    logger.info(f"--- DW-Gate Analysis Finished ---")
    return final_stats 