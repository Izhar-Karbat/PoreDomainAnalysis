# filename: pore_analysis/modules/ion_analysis/conduction.py
"""
Ion Analysis: Conduction and site transition analysis.
Calculates transition statistics based on the filtered events saved to CSV.
"""

import os
import logging
import numpy as np
import pandas as pd # Ensure pandas is imported
import sqlite3
from typing import Dict, Optional, Tuple, List, Any
from collections import defaultdict # Import defaultdict

# Import from core modules
try:
    from pore_analysis.core.database import register_product, store_metric
    from pore_analysis.core.config import (
        ION_TRANSITION_TOLERANCE_FRAMES,
        ION_TRANSITION_TOLERANCE_MODE,
        ION_USE_SITE_SPECIFIC_THRESHOLDS,
        ION_SITE_OCCUPANCY_THRESHOLD_A,
    )
except ImportError as e:
    print(f"Error importing dependency modules in ion_analysis/conduction.py: {e}")
    raise

logger = logging.getLogger(__name__)

def _save_conduction_event_data(
    run_dir: str,
    conduction_events: List[Dict[str, Any]],
    transition_events: List[Dict[str, Any]], # Accept raw transition events list
    db_conn: sqlite3.Connection,
    module_name: str
) -> Tuple[Optional[str], Optional[str]]:
    """
    Internal helper to save detailed conduction and transition events to CSV files
    and register them in the database.

    Args:
        run_dir: Path to the run directory.
        conduction_events: List of detected full conduction events.
        transition_events: List of detected (raw, filtered) transition events.
        db_conn: Active database connection.
        module_name: Name of the calling module.

    Returns:
        Tuple (conduction_csv_rel_path, transition_csv_rel_path)
    """
    output_dir = os.path.join(run_dir, "ion_analysis")
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = {'conduction': None, 'transition': None} # Initialize paths as None

    # Conduction events
    if conduction_events:
        logger.debug(f"Saving {len(conduction_events)} ion conduction events...")
        try:
            dfc = pd.DataFrame(conduction_events)
            cols = ['ion_idx','direction','entry_frame','exit_frame','entry_time','exit_time','transit_time','sites_visited']
            # Ensure columns exist before indexing
            dfc = dfc[[c for c in cols if c in dfc.columns]]
            path = os.path.join(output_dir, "ion_conduction_events.csv")
            dfc.to_csv(path, index=False, float_format="%.3f")
            logger.info(f"Conduction events written to {path}")
            rel_path = os.path.relpath(path, run_dir)
            register_product(db_conn, module_name, "csv", "data", rel_path,
                             subcategory="ion_conduction_events",
                             description="Details of full K+ ion conduction events.")
            saved_paths['conduction'] = rel_path
        except Exception as e:
            logger.error(f"Failed to save/register conduction CSV: {e}")
            # saved_paths['conduction'] remains None
    else:
        logger.info("No conduction events to save.")
        # saved_paths['conduction'] remains None

    # Transition events
    if transition_events:
        logger.debug(f"Saving {len(transition_events)} site-transition events...")
        try:
            dft = pd.DataFrame(transition_events)
            cols = ['frame','time','ion_idx','from_site','to_site','direction','z_position']
             # Ensure columns exist before indexing
            dft = dft[[c for c in cols if c in dft.columns]]
            path = os.path.join(output_dir, "ion_transition_events.csv")
            dft.to_csv(path, index=False, float_format="%.3f")
            logger.info(f"Transition events written to {path}")
            rel_path = os.path.relpath(path, run_dir)
            register_product(db_conn, module_name, "csv", "data", rel_path,
                             subcategory="ion_transition_events",
                             description="Details of K+ ion site-to-site transition events (filtered by tolerance).")
            saved_paths['transition'] = rel_path
        except Exception as e:
            logger.error(f"Failed to save/register transition CSV: {e}")
            # saved_paths['transition'] remains None
    else:
        logger.info("No transition events to save.")
        # saved_paths['transition'] remains None

    return saved_paths['conduction'], saved_paths['transition']


def _store_conduction_metrics(
    stats: Dict[str, Any],
    site_order: List[str],
    db_conn: sqlite3.Connection,
    module_name: str
):
    """Helper to store conduction/transition stats in the DB."""
    logger.debug(f"Storing conduction/transition metrics: {list(stats.keys())}")
    try:
        # Conduction counts & times
        # Use .get with default 0 or np.nan for robustness
        store_metric(db_conn, module_name, 'Ion_ConductionEvents_Total', stats.get('Ion_ConductionEvents_Total', 0), 'count', 'Total full conduction events')
        store_metric(db_conn, module_name, 'Ion_ConductionEvents_Outward', stats.get('Ion_ConductionEvents_Outward', 0), 'count', 'Outward (Cav->S0) conduction events')
        store_metric(db_conn, module_name, 'Ion_ConductionEvents_Inward', stats.get('Ion_ConductionEvents_Inward', 0), 'count', 'Inward (S0->Cav) conduction events')
        store_metric(db_conn, module_name, 'Ion_Conduction_MeanTransitTime_ns', stats.get('Ion_Conduction_MeanTransitTime_ns', np.nan), 'ns', 'Mean full transit time')
        store_metric(db_conn, module_name, 'Ion_Conduction_MedianTransitTime_ns', stats.get('Ion_Conduction_MedianTransitTime_ns', np.nan), 'ns', 'Median full transit time')
        store_metric(db_conn, module_name, 'Ion_Conduction_StdTransitTime_ns', stats.get('Ion_Conduction_StdTransitTime_ns', np.nan), 'ns', 'Std Dev full transit time')

        # Transition counts (these now come from the CSV read)
        store_metric(db_conn, module_name, 'Ion_TransitionEvents_Total', stats.get('Ion_TransitionEvents_Total', 0), 'count', 'Total adjacent site transition events (from CSV)')
        store_metric(db_conn, module_name, 'Ion_TransitionEvents_Upward', stats.get('Ion_TransitionEvents_Upward', 0), 'count', 'Upward transition events (from CSV)')
        store_metric(db_conn, module_name, 'Ion_TransitionEvents_Downward', stats.get('Ion_TransitionEvents_Downward', 0), 'count', 'Downward transition events (from CSV)')

        # Per-pair transitions (these now come from the CSV read)
        for s1, s2 in zip(site_order[:-1], site_order[1:]):
            key = f"Ion_Transition_{s1}_{s2}"
            desc = f"Transitions between {s1} and {s2} (from CSV)"
            store_metric(db_conn, module_name, key, stats.get(key, 0), 'count', desc)

        # Tolerance settings (these are parameters, not calculated stats)
        store_metric(db_conn, module_name, 'Ion_Transition_ToleranceMode', stats.get('Ion_Transition_ToleranceMode', 'N/A'), '', 'Transition tolerance mode')
        store_metric(db_conn, module_name, 'Ion_Transition_ToleranceFrames', stats.get('Ion_Transition_ToleranceFrames', np.nan), 'frames', 'Transition tolerance window size')

    except Exception as e:
        logger.error(f"Failed to store conduction/transition metrics: {e}", exc_info=True)



def analyze_conduction_events(
    run_dir: str,
    time_points: np.ndarray,
    ions_z_positions_abs: Dict[int, np.ndarray],
    ion_indices: List[int],
    filter_sites: Dict[str, float],
    g1_reference: float,
    db_conn: sqlite3.Connection,
    module_name: str = "ion_analysis"
) -> Tuple[Optional[str], Optional[str]]:
    """
    Analyze ion conduction (full Cavity<->S0 passes) and adjacent site transitions.
    Saves event details to CSVs. Calculates transition summary stats *from the saved
    transition CSV* and stores all summary metrics in the database.
    """
    logger.info("Analyzing Ion Conduction & Site Transitions...")

    # --- Input Checks ---
    if time_points.size == 0 or not ions_z_positions_abs or not ion_indices or len(filter_sites) < 2:
        logger.error("Insufficient inputs; aborting conduction analysis.")
        return None, None

    # --- Setup Site Information ---
    site_order = [s for s in ['Cavity','S4','S3','S2','S1','S0'] if s in filter_sites]
    if len(site_order) < 2:
        logger.error("Need at least two defined sites (Cavity, S0-S4) for conduction analysis.")
        return None, None

    abs_pos = {s: filter_sites[s] + g1_reference for s in site_order}
    thresholds = {}
    if ION_USE_SITE_SPECIFIC_THRESHOLDS:
        for i, site in enumerate(site_order):
            neighbors = []
            if i > 0: neighbors.append(abs(abs_pos[site] - abs_pos[site_order[i-1]]))
            if i < len(site_order)-1: neighbors.append(abs(abs_pos[site_order[i+1]] - abs_pos[site]))
            # Ensure threshold isn't zero if sites overlap (use fallback if needed)
            min_neighbor_dist = min(neighbors) if neighbors else 0
            thresholds[site] = (min_neighbor_dist / 2.0) if min_neighbor_dist > 1e-6 else ION_SITE_OCCUPANCY_THRESHOLD_A
    else:
        for site in site_order: thresholds[site] = ION_SITE_OCCUPANCY_THRESHOLD_A
    logger.debug(f"Using site thresholds: {thresholds}")

    # --- Initialize Per-Ion State Tracking ---
    states = {}
    for idx in ion_indices:
        states[idx] = {
            'previous_site': None, 'current_site': None,
            'in_transit': False, 'transit_direction': None,
            'transit_entry_frame': None, 'transit_entry_time': None,
            'transit_sites_visited': set(),
            'conductions': [], 'transitions': []
        }

    n_frames = len(time_points)
    # --- Iterate Frames to Detect Events ---
    logger.info("Detecting conduction and transition events...")
    for fi in range(n_frames):
        t = time_points[fi]
        for ion_idx in ion_indices:
            # Skip if ion index is not valid or data missing
            if ion_idx not in ions_z_positions_abs: continue
            zarr = ions_z_positions_abs[ion_idx]
            state = states[ion_idx]

            if fi >= len(zarr) or np.isnan(zarr[fi]):
                state['previous_site'] = state['current_site']
                state['current_site']  = None
                if state['in_transit']: state['in_transit'] = False; state['transit_sites_visited'].clear()
                continue

            z = zarr[fi]
            state['previous_site'] = state['current_site']

            best_site, best_dist = None, np.inf
            for site, pos in abs_pos.items():
                d = abs(z - pos)
                if d < thresholds[site] and d < best_dist:
                    best_dist, best_site = d, site

            if best_site is None:
                lo_site, hi_site = site_order[0], site_order[-1]
                lo_z, hi_z = abs_pos[lo_site], abs_pos[hi_site]
                if lo_z > hi_z: lo_z, hi_z = hi_z, lo_z
                if lo_z <= z <= hi_z: best_site = 'intermediate'
            state['current_site'] = best_site

            prev, curr = state['previous_site'], state['current_site']

            # Detect Adjacent Site Transitions (with Tolerance)
            if prev in site_order and curr in site_order and prev != curr:
                try:
                    i_p, i_c = site_order.index(prev), site_order.index(curr)
                    if abs(i_c - i_p) == 1:
                        mode = ION_TRANSITION_TOLERANCE_MODE
                        tf = ION_TRANSITION_TOLERANCE_FRAMES
                        transition_ok = True

                        # Backward check
                        if mode != 'forward' and fi > 0:
                            start_back = max(0, fi - tf)
                            window_back = zarr[start_back:fi]
                            if len(window_back) < 1 : transition_ok = False # Need at least one frame back
                            else:
                                sites_back = []
                                for zk in window_back:
                                    if np.isnan(zk): sites_back.append(None); continue
                                    sk, dk = None, np.inf
                                    for site, pos in abs_pos.items():
                                        d_ = abs(zk - pos);
                                        if d_ < thresholds[site] and d_ < dk: dk, sk = d_, site
                                    sites_back.append(sk)

                                if mode == 'strict' and any(s != prev for s in sites_back if s is not None): transition_ok = False
                                elif mode == 'majority' and sites_back.count(prev) <= (len(sites_back) // 2): transition_ok = False # Strict majority

                        # Forward check
                        if transition_ok:
                            start_fwd = fi
                            end_fwd = min(n_frames, fi + tf)
                            window_fwd = zarr[start_fwd:end_fwd]
                            if len(window_fwd) < tf and mode != 'majority': transition_ok = False # Need full window for strict/forward unless majority
                            elif len(window_fwd) == 0: transition_ok = False # Need at least current frame
                            else:
                                sites_fwd = []
                                for zk in window_fwd:
                                    if np.isnan(zk): sites_fwd.append(None); continue
                                    sk, dk = None, np.inf
                                    for site, pos in abs_pos.items():
                                        d_ = abs(zk - pos);
                                        if d_ < thresholds[site] and d_ < dk: dk, sk = d_, site
                                    sites_fwd.append(sk)

                                if mode in ('strict', 'forward') and any(s != curr for s in sites_fwd if s is not None): transition_ok = False
                                elif mode == 'majority' and sites_fwd.count(curr) <= (len(sites_fwd) // 2): transition_ok = False # Strict majority

                        # Record Event
                        if transition_ok:
                            direction = 'upward' if i_c > i_p else 'downward'
                            evt = {'ion_idx': ion_idx, 'frame': fi, 'time': t,
                                   'from_site': prev, 'to_site': curr, 'direction': direction,
                                   'z_position': z}
                            state['transitions'].append(evt)
                except ValueError: # site not in site_order (shouldn't happen if check passed)
                    logger.warning(f"Site index error for prev='{prev}', curr='{curr}' at frame {fi}")

            # Conduction Event Detection
            outermost_site = site_order[-1]; innermost_site = site_order[0]
            if not state['in_transit'] and curr in (innermost_site, outermost_site):
                state['in_transit'] = True
                state['transit_direction'] = 'outward' if curr == innermost_site else 'inward'
                state['transit_entry_frame'] = fi
                state['transit_entry_time'] = t
                state['transit_sites_visited'] = {curr}
            elif state['in_transit']:
                if curr and curr != 'intermediate': state['transit_sites_visited'].add(curr)
                visited_sf = any(s in site_order[1:-1] for s in state['transit_sites_visited'])
                if state['transit_direction'] == 'outward' and curr == outermost_site and visited_sf:
                    transit_time = t - state['transit_entry_time']
                    evt = {'ion_idx': ion_idx, 'direction': 'outward',
                           'entry_frame': state['transit_entry_frame'], 'exit_frame': fi,
                           'entry_time': state['transit_entry_time'], 'exit_time': t,
                           'transit_time': transit_time,
                           'sites_visited': sorted(list(state['transit_sites_visited']), key=site_order.index)}
                    state['conductions'].append(evt)
                    state['in_transit'] = False; state['transit_sites_visited'].clear()
                elif state['transit_direction'] == 'inward' and curr == innermost_site and visited_sf:
                    transit_time = t - state['transit_entry_time']
                    evt = {'ion_idx': ion_idx, 'direction': 'inward',
                           'entry_frame': state['transit_entry_frame'], 'exit_frame': fi,
                           'entry_time': state['transit_entry_time'], 'exit_time': t,
                           'transit_time': transit_time,
                           'sites_visited': sorted(list(state['transit_sites_visited']), key=site_order.index, reverse=True)}
                    state['conductions'].append(evt)
                    state['in_transit'] = False; state['transit_sites_visited'].clear()

    # --- Compile Raw Events & Save CSVs ---
    all_cond_raw   = [e for s in states.values() for e in s['conductions']]
    all_trans_raw  = [e for s in states.values() for e in s['transitions']]
    cond_csv_rel_path, trans_csv_rel_path = _save_conduction_event_data(
        run_dir, all_cond_raw, all_trans_raw, db_conn, module_name
    )

    # --- Calculate Stats (Conduction from raw, Transition from CSV) ---
    stats: Dict[str, Any] = {}

    # Conduction stats (from raw events)
    stats['Ion_ConductionEvents_Total'] = len(all_cond_raw)
    stats['Ion_ConductionEvents_Outward'] = sum(1 for e in all_cond_raw if e['direction']=='outward')
    stats['Ion_ConductionEvents_Inward']  = sum(1 for e in all_cond_raw if e['direction']=='inward')
    times = [e['transit_time'] for e in all_cond_raw if 'transit_time' in e]
    stats['Ion_Conduction_MeanTransitTime_ns']   = float(np.mean(times))   if times else np.nan
    stats['Ion_Conduction_MedianTransitTime_ns'] = float(np.median(times)) if times else np.nan
    stats['Ion_Conduction_StdTransitTime_ns']    = float(np.std(times))    if times else np.nan

    # MODIFICATION: Use all_trans_raw directly instead of reading from CSV
    # This ensures we use the idealized transitions that have already been filtered
    transition_stats = {}
    if all_trans_raw:
        transition_stats['Ion_TransitionEvents_Total'] = len(all_trans_raw)
        transition_stats['Ion_TransitionEvents_Upward'] = sum(1 for t in all_trans_raw if t['direction'] == 'upward')
        transition_stats['Ion_TransitionEvents_Downward'] = sum(1 for t in all_trans_raw if t['direction'] == 'downward')
        
        # Count transitions between each pair of sites
        for s1, s2 in zip(site_order[:-1], site_order[1:]):
            key = f"Ion_Transition_{s1}_{s2}"
            # Count transitions between s1 and s2 regardless of direction
            count = sum(1 for t in all_trans_raw if 
                       ((t['from_site'] == s1) and (t['to_site'] == s2)) or
                       ((t['from_site'] == s2) and (t['to_site'] == s1)))
            transition_stats[key] = count
    else:
        transition_stats['Ion_TransitionEvents_Total'] = 0
        transition_stats['Ion_TransitionEvents_Upward'] = 0
        transition_stats['Ion_TransitionEvents_Downward'] = 0
        for s1, s2 in zip(site_order[:-1], site_order[1:]):
            transition_stats[f"Ion_Transition_{s1}_{s2}"] = 0

    # Combine stats and add tolerance info
    stats.update(transition_stats)
    stats['Ion_Transition_ToleranceMode']   = ION_TRANSITION_TOLERANCE_MODE
    stats['Ion_Transition_ToleranceFrames'] = ION_TRANSITION_TOLERANCE_FRAMES

    # --- Store Metrics in DB ---
    _store_conduction_metrics(stats, site_order, db_conn, module_name)

    logger.info("Ion conduction & transition analysis complete.")
    # Return the relative paths to the saved CSVs (might be None if saving failed)
    return cond_csv_rel_path, trans_csv_rel_path