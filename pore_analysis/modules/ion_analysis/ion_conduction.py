"""
Module for analyzing K+ ion conduction and site‐to‐site transitions.
"""

import os
import logging
from typing import Dict, Any, List, Optional
from collections import defaultdict
import colorsys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import to_rgb, rgb_to_hsv, hsv_to_rgb # Import color conversion tools
from scipy.stats import gaussian_kde
import seaborn as sns

from pore_analysis.core.config import (
    FRAMES_PER_NS,
    ION_TRANSITION_TOLERANCE_FRAMES,
    ION_TRANSITION_TOLERANCE_MODE,
    ION_USE_SITE_SPECIFIC_THRESHOLDS,
    ION_SITE_OCCUPANCY_THRESHOLD_A,
)

logger = logging.getLogger(__name__)


def save_conduction_data(
    run_dir: str,
    conduction_events: List[Dict[str, Any]],
    transition_events: List[Dict[str, Any]],
    stats: Dict[str, Any]
) -> None:
    """
    Save detailed conduction and transition events to CSV files.

    Side Effects:
        Creates (or reuses) directory <run_dir>/ion_analysis/,
        writes:
          - ion_conduction_events.csv
          - ion_transition_events.csv
    """
    output_dir = os.path.join(run_dir, "ion_analysis")
    os.makedirs(output_dir, exist_ok=True)

    # Conduction events
    if conduction_events:
        logger.info("Saving ion conduction events...")
        dfc = pd.DataFrame(conduction_events)
        cols = ['ion_idx','direction','entry_frame','exit_frame','entry_time','exit_time','transit_time','sites_visited']
        dfc = dfc[[c for c in cols if c in dfc.columns]]
        path = os.path.join(output_dir, "ion_conduction_events.csv")
        try:
            dfc.to_csv(path, index=False, float_format="%.3f")
            logger.info(f"Conduction events written to {path}")
        except Exception as e:
            logger.error(f"Failed to write conduction CSV: {e}")
    else:
        logger.info("No conduction events to save.")

    # Transition events
    if transition_events:
        logger.info("Saving site-transition events...")
        dft = pd.DataFrame(transition_events)
        cols = ['frame','time','ion_idx','from_site','to_site','direction','z_position']
        dft = dft[[c for c in cols if c in dft.columns]]
        path = os.path.join(output_dir, "ion_transition_events.csv")
        try:
            dft.to_csv(path, index=False, float_format="%.3f")
            logger.info(f"Transition events written to {path}")
        except Exception as e:
            logger.error(f"Failed to write transition CSV: {e}")
    else:
        logger.info("No transition events to save.")


def analyze_ion_conduction(
    run_dir: str,
    time_points: np.ndarray,
    ions_z_positions: Dict[int, np.ndarray],
    ion_indices: List[int],
    filter_sites: Dict[str, float],
    g1_reference: float
) -> Dict[str, Any]:
    """
    Analyze ion conduction (full Cavity↔S0 passes) and adjacent site transitions.

    Args:
        run_dir: Path to the simulation run directory.
        time_points: 1D array of times (ns).
        ions_z_positions: Mapping ion_idx -> array of Z coordinates.
        ion_indices: List of ion indices to track.
        filter_sites: Map site_label -> Z_rel_to_G1 (ns).
        g1_reference: Absolute Z position of G1 reference (Å).

    Returns:
        stats: Dictionary containing:
          - Ion_ConductionEvents_Total, _Outward, _Inward
          - Ion_Conduction_MeanTransitTime_ns, _MedianTransitTime_ns, _StdTransitTime_ns
          - Ion_TransitionEvents_Total, _Upward, _Downward
          - Ion_Transition_<Site1>_<Site2> counts for each adjacent pair
          - Ion_Transition_ToleranceMode, Ion_Transition_ToleranceFrames
    """
    logger.info("Starting Ion Conduction & Transition Analysis…")

    # -- input checks --
    if time_points.size == 0 or not ions_z_positions or not ion_indices or len(filter_sites) < 2:
        logger.error("Insufficient inputs; aborting conduction analysis.")
        return {}

    # build ordered list of sites (Cavity lowest Z → S0 highest Z)
    site_order = [s for s in ['Cavity','S4','S3','S2','S1','S0'] if s in filter_sites]
    abs_pos = {s: filter_sites[s] + g1_reference for s in site_order}

    # compute per-site occupancy thresholds
    thresholds = {}
    if ION_USE_SITE_SPECIFIC_THRESHOLDS:
        for i, site in enumerate(site_order):
            neighbors = []
            if i > 0:
                neighbors.append(abs(abs_pos[site] - abs_pos[site_order[i-1]]))
            if i < len(site_order)-1:
                neighbors.append(abs(abs_pos[site_order[i+1]] - abs_pos[site]))
            thresholds[site] = (min(neighbors)/2.0) if neighbors else ION_SITE_OCCUPANCY_THRESHOLD_A
    else:
        for site in site_order:
            thresholds[site] = ION_SITE_OCCUPANCY_THRESHOLD_A

    # initialize per‑ion state
    states = {}
    for idx in ion_indices:
        states[idx] = {
            'previous_site':   None,
            'current_site':    None,
            'in_transit':      False,
            'transit_direction': None,
            'transit_entry_frame': None,
            'transit_entry_time':  None,
            'transit_sites_visited': set(),
            'conductions':     [],
            'transitions':     [],
            'last_bound_site': None,
            'last_exit_frame': None,
        }

    n_frames = len(time_points)
    # iterate frames
    for fi in range(n_frames):
        t = time_points[fi]
        for ion_idx in ion_indices:
            zarr = ions_z_positions.get(ion_idx)
            state = states[ion_idx]

            # handle missing or NaN
            if zarr is None or fi >= zarr.size or np.isnan(zarr[fi]):
                state['previous_site'] = state['current_site']
                state['current_site']  = None
                state['in_transit']    = False
                state['transit_sites_visited'].clear()
                state['last_bound_site'] = None
                state['last_exit_frame'] = None
                continue

            z = zarr[fi]
            state['previous_site'] = state['current_site']

            # assign to nearest site or intermediate
            best_site, best_dist = None, np.inf
            for site, pos in abs_pos.items():
                d = abs(z - pos)
                if d < thresholds[site] and d < best_dist:
                    best_dist, best_site = d, site
            if best_site is None:
                lo, hi = abs_pos[site_order[0]], abs_pos[site_order[-1]]
                if lo > hi:
                    lo, hi = hi, lo
                if lo <= z <= hi:
                    best_site = 'intermediate'
            state['current_site'] = best_site

            # track last real site for smoothing
            if best_site and best_site != 'intermediate':
                state['last_bound_site'] = best_site
                state['last_exit_frame'] = None
            elif best_site == 'intermediate':
                prev = state['previous_site']
                if prev and prev != 'intermediate':
                    state['last_bound_site'] = prev
                    state['last_exit_frame'] = fi

            # --- detect adjacent site transitions ---
            prev, curr = state['previous_site'], state['current_site']
            if prev in site_order and curr in site_order and prev != curr:
                i_p, i_c = site_order.index(prev), site_order.index(curr)
                if abs(i_c - i_p) == 1:
                    # tolerance windows
                    mode = ION_TRANSITION_TOLERANCE_MODE
                    tf = ION_TRANSITION_TOLERANCE_FRAMES
                    # backward check
                    ok = True
                    if mode != 'forward':
                        if fi < tf:
                            ok = False
                        else:
                            hits = 0
                            for k in range(1, tf):
                                zp = ions_z_positions[ion_idx][fi-k]
                                # re-assign site
                                s_ = None
                                md = np.inf
                                for st, pz in abs_pos.items():
                                    d_ = abs(zp - pz)
                                    if d_ < thresholds[st] and d_ < md:
                                        md, s_ = d_, st
                                if mode == 'strict' and s_ != prev:
                                    ok = False
                                    break
                                if mode == 'majority' and s_ == prev:
                                    hits += 1
                            if mode == 'majority' and hits < ((tf-1)//2 + 1):
                                ok = False
                    # forward check
                    if ok:
                        hits, total = 0, tf
                        for j in range(tf):
                            idx2 = fi + j
                            if idx2 < n_frames:
                                zp = ions_z_positions[ion_idx][idx2]
                                s_ = None
                                md = np.inf
                                for st, pz in abs_pos.items():
                                    d_ = abs(zp - pz)
                                    if d_ < thresholds[st] and d_ < md:
                                        md, s_ = d_, st
                                if mode in ('strict','forward') and s_ != curr:
                                    ok = False
                                    break
                                if mode == 'majority' and s_ == curr:
                                    hits += 1
                        if mode == 'majority' and hits < (total//2 + 1):
                            ok = False
                    if ok:
                        direction = 'upward' if i_c > i_p else 'downward'
                        evt = {
                            'ion_idx': ion_idx,
                            'frame': fi,
                            'time': t,
                            'from_site': prev,
                            'to_site': curr,
                            'direction': direction,
                            'z_position': z
                        }
                        state['transitions'].append(evt)

            # --- conduction event detection ---
            # start transit
            if prev is None and curr in (site_order[0], site_order[-1]):
                state['in_transit'] = True
                state['transit_direction'] = 'outward' if curr == site_order[0] else 'inward'
                state['transit_entry_frame'] = fi
                state['transit_entry_time'] = t
                state['transit_sites_visited'] = {curr}
            # exit channel region
            elif prev and curr is None:
                state['in_transit'] = False
                state['transit_sites_visited'].clear()
            # update transit
            elif state['in_transit']:
                if curr and curr != 'intermediate':
                    state['transit_sites_visited'].add(curr)
                # check for completion
                visited_sf = any(s in site_order[1:-1] for s in state['transit_sites_visited'])
                if state['transit_direction']=='outward' and curr==site_order[-1] and visited_sf:
                    transit_time = t - state['transit_entry_time']
                    evt = {
                        'ion_idx': ion_idx,
                        'direction': 'outward',
                        'entry_frame': state['transit_entry_frame'],
                        'exit_frame': fi,
                        'entry_time': state['transit_entry_time'],
                        'exit_time': t,
                        'transit_time': transit_time,
                        'sites_visited': sorted(state['transit_sites_visited'], key=site_order.index)
                    }
                    state['conductions'].append(evt)
                    state['in_transit'] = False
                    state['transit_sites_visited'].clear()
                elif state['transit_direction']=='inward' and curr==site_order[0] and visited_sf:
                    transit_time = t - state['transit_entry_time']
                    evt = {
                        'ion_idx': ion_idx,
                        'direction': 'inward',
                        'entry_frame': state['transit_entry_frame'],
                        'exit_frame': fi,
                        'entry_time': state['transit_entry_time'],
                        'exit_time': t,
                        'transit_time': transit_time,
                        'sites_visited': sorted(state['transit_sites_visited'],
                                                key=site_order.index, reverse=True)
                    }
                    state['conductions'].append(evt)
                    state['in_transit'] = False
                    state['transit_sites_visited'].clear()

    # compile across all ions
    all_trans  = [e for s in states.values() for e in s['transitions']]
    all_cond   = [e for s in states.values() for e in s['conductions']]

    # compute summary stats
    stats: Dict[str, Any] = {}
    # conduction counts
    stats['Ion_ConductionEvents_Total'] = len(all_cond)
    stats['Ion_ConductionEvents_Outward'] = sum(1 for e in all_cond if e['direction']=='outward')
    stats['Ion_ConductionEvents_Inward']  = sum(1 for e in all_cond if e['direction']=='inward')
    times = [e['transit_time'] for e in all_cond]
    stats['Ion_Conduction_MeanTransitTime_ns']   = float(np.mean(times))   if times else np.nan
    stats['Ion_Conduction_MedianTransitTime_ns'] = float(np.median(times)) if times else np.nan
    stats['Ion_Conduction_StdTransitTime_ns']    = float(np.std(times))    if times else np.nan

    # transition counts
    stats['Ion_TransitionEvents_Total']   = len(all_trans)
    stats['Ion_TransitionEvents_Upward']  = sum(1 for e in all_trans if e['direction']=='upward')
    stats['Ion_TransitionEvents_Downward']= sum(1 for e in all_trans if e['direction']=='downward')

    # per‐pair adjacent counts
    for s1, s2 in zip(site_order[:-1], site_order[1:]):
        key = f"Ion_Transition_{s1}_{s2}"
        stats[key] = sum(
            1 for e in all_trans
            if (e['from_site']==s1 and e['to_site']==s2)
            or (e['from_site']==s2 and e['to_site']==s1)
        )

    # record tolerance settings
    stats['Ion_Transition_ToleranceMode']   = ION_TRANSITION_TOLERANCE_MODE
    stats['Ion_Transition_ToleranceFrames'] = ION_TRANSITION_TOLERANCE_FRAMES

    # dump CSVs
    save_conduction_data(run_dir, all_cond, all_trans, stats)
    logger.info("Ion conduction & transition analysis complete.")
    return stats 

def make_pastel(color, factor=0.5, light_boost=0.15):
    """Create a pastel version of a color."""
    try:
        rgb = to_rgb(color)
        h, s, v = rgb_to_hsv(rgb)
        # Decrease saturation, increase value (lightness)
        pastel_s = max(0, s * (1.0 - factor))
        pastel_v = min(1, v + light_boost)
        return hsv_to_rgb((h, pastel_s, pastel_v))
    except Exception as e:
        logger.warning(f"Could not create pastel color for {color}: {e}")
        return color # Return original on error

def plot_idealized_transitions(
    run_dir: str,
    time_points: np.ndarray,
    filter_sites: Dict[str, float],
    g1_reference: float,
    ions_z_positions: Dict[int, np.ndarray], # Added
    ion_indices: List[int] # Added
) -> None:
    """
    Plots idealized ion trajectories based on detected site transitions,
    overlaid with the actual Z-position trajectories.

    Reads ion_transition_events.csv for idealized states.
    Uses ions_z_positions for actual trajectories.

    Args:
        run_dir: Path to the simulation run directory.
        time_points: Full time axis for the plot.
        filter_sites: Map site_label -> Z_rel_to_G1.
        g1_reference: Absolute Z position of G1 reference (Å).
        ions_z_positions: Mapping ion_idx -> array of Z coordinates.
        ion_indices: List of ion indices present in ions_z_positions.
    """
    logger.info("Generating overlayed idealized and actual ion transition plot...")
    output_dir = os.path.join(run_dir, "ion_analysis")
    transitions_csv = os.path.join(output_dir, "ion_transition_events.csv")
    plot_path = os.path.join(output_dir, "K_Ion_Overlayed_Transitions.png") # New filename

    if not os.path.exists(transitions_csv):
        logger.warning(f"Transition events CSV not found: {transitions_csv}. Skipping overlay plot.")
        return

    try:
        df_trans = pd.read_csv(transitions_csv)
        if df_trans.empty:
            logger.info("No transitions found in CSV. Skipping overlay plot.")
            return
    except Exception as e:
        logger.error(f"Error reading transitions CSV {transitions_csv}: {e}")
        return

    df_trans = df_trans.sort_values(by=['ion_idx', 'frame'])
    abs_site_positions = {site: z + g1_reference for site, z in filter_sites.items()}
    site_order = [s for s in ['Cavity', 'S4', 'S3', 'S2', 'S1', 'S0'] if s in filter_sites]

    # --- Plotting Setup - Single Plot --- 
    fig, ax = plt.subplots(figsize=(15, 8)) # Single axes, matched size
    sns.set_style("whitegrid")

    # Use unique ions present in *both* transitions and actual positions
    plot_ion_ids = sorted(list(set(df_trans['ion_idx'].unique()) & set(ion_indices)))
    if not plot_ion_ids:
        logger.warning("No common ion IDs found between transition data and actual positions. Skipping plot.")
        plt.close(fig)
        return

    colors = sns.color_palette("husl", n_colors=len(plot_ion_ids))
    ion_color_map = {ion_id: color for ion_id, color in zip(plot_ion_ids, colors)}
    pastel_color_map = {ion_id: make_pastel(color) for ion_id, color in ion_color_map.items()}

    min_time, max_time = time_points[0], time_points[-1]
    idealized_linewidth = 4
    actual_linewidth = 1
    actual_alpha = 0.5

    # Plot Actual and Idealized Trajectories
    for ion_id in plot_ion_ids:
        color = ion_color_map[ion_id]
        pastel_color = pastel_color_map[ion_id]

        # Plot Actual Trajectory (background)
        if ion_id in ions_z_positions:
            z_actual = ions_z_positions[ion_id]
            # Ensure actual time matches z_actual length, trim if needed
            time_actual = time_points[:len(z_actual)]
            if len(time_actual) == len(z_actual):
                ax.plot(time_actual, z_actual, color=pastel_color, linewidth=actual_linewidth,
                        alpha=actual_alpha, zorder=1, label=f'Ion {ion_id} (Actual)' if False else "_nolegend_") # Legend optional
            else:
                 logger.warning(f"Skipping actual plot for ion {ion_id}: Time points ({len(time_actual)}) vs Z positions ({len(z_actual)}) mismatch.")
        else:
            logger.warning(f"Actual position data missing for ion {ion_id}.")

        # Plot Idealized Trajectory (foreground)
        ion_trans = df_trans[df_trans['ion_idx'] == ion_id]
        if not ion_trans.empty:
            last_time = min_time
            last_site = None
            first_segment = True

            for _, row in ion_trans.iterrows():
                current_time = row['time']
                from_site = row['from_site']
                to_site = row['to_site']

                if last_site and last_site in abs_site_positions:
                    site_z = abs_site_positions[last_site]
                    ax.plot([last_time, current_time], [site_z, site_z],
                             color=color, linewidth=idealized_linewidth,
                             linestyle='-', zorder=2, # Solid line, foreground
                             label=f'Ion {ion_id} (Idealized)' if first_segment else "_nolegend_")
                    first_segment = False

                # Removed vertical transition lines for clarity
                # if to_site in abs_site_positions:
                #     to_site_z = abs_site_positions[to_site]
                #     ax.plot([current_time, current_time], [site_z, to_site_z],
                #              color=color, linewidth=idealized_linewidth - 1,
                #              linestyle=":", alpha=0.7, zorder=2)

                last_time = current_time
                last_site = to_site

            # Final idealized state segment
            if last_site and last_site in abs_site_positions:
                site_z = abs_site_positions[last_site]
                ax.plot([last_time, max_time], [site_z, site_z],
                         color=color, linewidth=idealized_linewidth, linestyle='-', zorder=2,
                         label="_nolegend_")

    # --- Site Position Lines & Labels --- 
    site_colors = sns.color_palette("muted", n_colors=len(site_order))
    z_positions_sorted = sorted(abs_site_positions.items(), key=lambda item: item[1])
    label_x_pos = max_time + (max_time - min_time) * 0.01 # Small offset for labels

    for i, (site, z_pos) in enumerate(z_positions_sorted):
         color = site_colors[i % len(site_colors)]
         ax.axhline(z_pos, color=color, linestyle='--', linewidth=1, alpha=0.8, zorder=0) # Behind everything
         ax.text(label_x_pos, z_pos, site,
                 verticalalignment='center', horizontalalignment='left',
                 color=color, fontsize=9) # Fontsize 9 matches original site labels

    # --- Final Touches --- 
    ax.set_xlabel("Time (ns)", fontsize=14)
    ax.set_ylabel("Z-Position (Å)", fontsize=14) # Generic Y label
    ax.tick_params(axis='both', labelsize=10)
    ax.grid(axis='y', linestyle=':', alpha=0.5, zorder=0)
    ax.legend(loc='upper right', fontsize='x-small') # Legend for idealized lines
    
    # Set Y limits based on site positions
    min_z_plot = min(abs_site_positions.values()) - 5.0 # Add padding
    max_z_plot = max(abs_site_positions.values()) + 5.0
    ax.set_ylim(min_z_plot, max_z_plot)

    # Ensure consistent X limits 
    x_padding = (max_time - min_time) * 0.05
    ax.set_xlim(min_time, max_time + x_padding)

    fig.suptitle("Overlayed Actual vs Idealized K+ Ion Trajectories", fontsize=16, y=0.99)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96]) # Keep layout rect from previous version

    try:
        fig.savefig(plot_path, dpi=200)
        logger.info(f"Saved overlayed transition plot to {plot_path}")
    except Exception as e:
        logger.error(f"Failed to save overlayed transition plot: {e}")
    plt.close(fig) 