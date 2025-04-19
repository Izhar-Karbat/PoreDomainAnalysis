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

    logger.info("Calculating DW Gate summary statistics...")
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
        states = df_chain["state"].to_numpy()
        open_durations_frames = []
        closed_durations_frames = []
        current_state = None
        current_duration = 0

        for state in states:
            if state == current_state:
                current_duration += 1
            else:
                # End of a block, check if it met tolerance
                if current_state is not None and current_duration >= DW_GATE_TOLERANCE_FRAMES:
                    if current_state == OPEN_STATE:
                        open_durations_frames.append(current_duration)
                    elif current_state == CLOSED_STATE:
                        closed_durations_frames.append(current_duration)

                # Start of a new block
                current_state = state
                current_duration = 1

        # Check the last block at the end of the series
        if current_state is not None and current_duration >= DW_GATE_TOLERANCE_FRAMES:
            if current_state == OPEN_STATE:
                open_durations_frames.append(current_duration)
            elif current_state == CLOSED_STATE:
                closed_durations_frames.append(current_duration)

        # Calculate statistics from durations
        n_open_events = len(open_durations_frames)
        n_closed_events = len(closed_durations_frames)

        time_per_frame_ns = stride * dt_ns
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
        raw_stats[f'DW_OpenEvents_{chain_char}'] = n_open_events
        raw_stats[f'DW_ClosedEvents_{chain_char}'] = n_closed_events
        raw_stats[f'DW_MeanOpenTime_ns_{chain_char}'] = mean_open_ns
        raw_stats[f'DW_StdOpenTime_ns_{chain_char}'] = std_open_ns
        raw_stats[f'DW_MeanClosedTime_ns_{chain_char}'] = mean_closed_ns
        raw_stats[f'DW_StdClosedTime_ns_{chain_char}'] = std_closed_ns

        logger.debug(f"Chain {chain_char}: Open Events={n_open_events}, Mean(ns)={mean_open_ns:.3f} +/- {std_open_ns:.3f}")
        logger.debug(f"Chain {chain_char}: Closed Events={n_closed_events}, Mean(ns)={mean_closed_ns:.3f} +/- {std_closed_ns:.3f}")

    # --- Combine Final Stats --- 
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
    # 1. Distance Timeseries
    try:
        fig1, ax1 = setup_plot(figsize=(10, 4))
        for chain_char in sorted([c[-1] for c in valid_chain_ids]):
            df_ch = df_timeseries[df_timeseries["chain"] == chain_char]
            if not df_ch.empty:
                 ax1.plot(df_ch["time_ns"], df_ch["distance"], label=f"Chain {chain_char}", lw=1)
        ax1.axhline(DISTANCE_THRESHOLD, color='grey', linestyle='--', lw=1.5, label=f'{DISTANCE_THRESHOLD}Å Threshold')
        ax1.set_xlabel("Time (ns)")
        ax1.set_ylabel("DW Distance (Å)")
        ax1.set_title("DW Gate Distance Over Time")
        ax1.legend(fontsize='small')
        ax1.grid(True, axis='y', linestyle=':', alpha=0.7)
        ax1.set_ylim(bottom=0)
        save_plot(fig1, os.path.join(output_dir, "dw_distance_timeseries.png"))
    except Exception as e:
         logger.error(f"Failed to generate distance timeseries plot: {e}", exc_info=True)

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

    # 4. Event Duration Histogram (Per Chain - NEW)
    try:
        # Check if there is any duration data across all chains
        all_durations_flat = [dur for subdurs in open_durations_by_chain.values() for dur in subdurs] + \
                           [dur for subdurs in closed_durations_by_chain.values() for dur in subdurs]

        if all_durations_flat:
            fig4, ax4 = setup_plot(figsize=(10, 6)) # Slightly larger plot
            min_dur = max(0.01, min(all_durations_flat)) # Avoid log(0)
            max_dur = max(1, max(all_durations_flat))
            bins = np.logspace(np.log10(min_dur), np.log10(max_dur), num=25) # Log bins based on overall range

            # Define consistent colors for chains (use seaborn or default cycle)
            colors = plt.cm.get_cmap('tab10') # Use a colormap

            chain_chars_plot = sorted([c[-1] for c in valid_chain_ids])
            for i, chain_char in enumerate(chain_chars_plot):
                chain_color = colors(i / len(chain_chars_plot)) # Assign color per chain

                open_durs = open_durations_by_chain.get(chain_char, [])
                closed_durs = closed_durations_by_chain.get(chain_char, [])

                # Plot histograms only if data exists for that chain/state
                if open_durs:
                    ax4.hist(open_durs, bins=bins, alpha=0.6, label=f'Chain {chain_char} Open', color=chain_color, density=True, histtype='step', linewidth=1.5)
                if closed_durs:
                    ax4.hist(closed_durs, bins=bins, alpha=0.8, label=f'Chain {chain_char} Closed', color=chain_color, density=True, histtype='step', linewidth=1.5, linestyle='--')

            ax4.set_xscale('log')
            ax4.set_xlabel("Event Duration (ns)")
            ax4.set_ylabel("Probability Density")
            ax4.set_title(f"DW Gate Event Duration Distribution (Tolerance: {DW_GATE_TOLERANCE_FRAMES} frames)")
            ax4.legend(fontsize='small', ncol=2) # Adjust legend
            ax4.grid(True, which="both", ls="--", alpha=0.6)

            save_plot(fig4, os.path.join(output_dir, "dw_gate_duration_histogram.png"))
        else:
            logger.warning("Skipping event duration histogram: No confirmed open or closed events found across all chains.")
    except Exception as e:
        logger.error(f"Failed to generate event duration histogram plot: {e}", exc_info=True)

    logger.info(f"--- DW-Gate Analysis Finished ---")
    return final_stats 