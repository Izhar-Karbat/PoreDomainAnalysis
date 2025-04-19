"""
Functions for analyzing ion conduction events and site-to-site transitions.
"""

import os
import logging
import numpy as np
import pandas as pd

try:
    from pore_analysis.core.config import FRAMES_PER_NS, ION_TRANSITION_TOLERANCE_FRAMES
except ImportError:
    # Fallback if config import fails (should not happen in normal execution)
    FRAMES_PER_NS = 10.0
    ION_TRANSITION_TOLERANCE_FRAMES = 10 # Use default if import fails
    print("Warning: Could not import constants from config. Using defaults.")

# Get a logger for this module
module_logger = logging.getLogger(__name__)

def save_conduction_data(run_dir, conduction_events, transition_events, stats):
    """Saves conduction event lists and summary stats to CSV files."""
    output_dir = os.path.join(run_dir, "ion_analysis") # Save within ion_analysis folder
    os.makedirs(output_dir, exist_ok=True)
    logger = module_logger

    # Save Conduction Events
    if conduction_events:
        logger.info("Saving ion conduction events...")
        df_cond = pd.DataFrame(conduction_events)
        # Order columns nicely
        cond_cols = ['ion_idx', 'direction', 'entry_frame', 'exit_frame', 'entry_time', 'exit_time', 'transit_time', 'sites_visited']
        df_cond = df_cond[[col for col in cond_cols if col in df_cond.columns]] # Ensure columns exist
        cond_path = os.path.join(output_dir, "ion_conduction_events.csv")
        try:
            df_cond.to_csv(cond_path, index=False, float_format='%.3f')
            logger.info(f"Saved conduction events to {cond_path}")
        except Exception as e:
            logger.error(f"Failed to save conduction events CSV: {e}")
    else:
        logger.info("No completed ion conduction events detected.")

    # Save Transition Events
    if transition_events:
        logger.info("Saving site-to-site transition events...")
        df_trans = pd.DataFrame(transition_events)
        trans_cols = ['frame', 'time', 'ion_idx', 'from_site', 'to_site', 'direction', 'z_position']
        # Add ion_idx if it wasn't explicitly added during collection (it should be added)
        if 'ion_idx' not in df_trans.columns and conduction_events:
             # Need to map back - complex. Assume ion_idx is added during collection.
             logger.warning("Transition events DataFrame missing 'ion_idx'. Check collection logic.")

        df_trans = df_trans[[col for col in trans_cols if col in df_trans.columns]]
        trans_path = os.path.join(output_dir, "ion_transition_events.csv")
        try:
            df_trans.to_csv(trans_path, index=False, float_format='%.3f')
            logger.info(f"Saved transition events to {trans_path}")
        except Exception as e:
            logger.error(f"Failed to save transition events CSV: {e}")
    else:
        logger.info("No site-to-site transition events detected.")

    # Save Summary Stats (Optional, as they are returned anyway)
    # logger.info("Saving conduction/transition summary stats...")
    # stats_path = os.path.join(output_dir, "ion_conduction_summary.csv")
    # try:
    #     pd.Series(stats).reset_index().rename(columns={'index':'Metric', 0:'Value'}).to_csv(stats_path, index=False)
    #     logger.info(f"Saved conduction summary stats to {stats_path}")
    # except Exception as e:
    #     logger.error(f"Failed to save conduction summary stats CSV: {e}")

def create_conduction_plots(run_dir, conduction_events, transition_events, stats, time_points):
    """Placeholder for creating conduction/transition plots."""
    logger = module_logger
    logger.info("Plotting for ion conduction/transitions is not yet implemented.")
    # Add plotting logic here in the future (e.g., using matplotlib, seaborn, plotly for Sankey)
    pass

def analyze_ion_conduction(run_dir, time_points, ions_z_positions, ion_indices, filter_sites, g1_reference):
    """
    Analyze ion conduction and transition events through the channel.

    Args:
        run_dir (str): Directory to save output.
        time_points (np.ndarray): Array of time points (ns).
        ions_z_positions (dict): Dictionary {ion_idx: np.array(z_positions)}.
        ion_indices (list): List of tracked ion indices.
        filter_sites (dict): Dictionary of filter site labels to G1-centric Z positions (S0-S4, Cavity).
        g1_reference (float): Absolute Z position of the G1 C-alpha reference plane.

    Returns:
        dict: Dictionary with conduction and transition statistics.
    """
    logger = module_logger
    logger.info("Starting Ion Conduction & Transition Analysis...")

    # Input validation
    if not all([time_points is not None, ions_z_positions, ion_indices is not None, filter_sites, g1_reference is not None]):
        logger.error("Missing essential input data for conduction analysis. Aborting.")
        return {}
    if len(time_points) == 0:
        logger.error("Time points array is empty. Aborting conduction analysis.")
        return {}

    # Define site order for direction calculation
    # Ensure Cavity is first (lowest Z) and S0 is last (highest Z)
    site_order = [s for s in ['Cavity', 'S4', 'S3', 'S2', 'S1', 'S0'] if s in filter_sites]
    if len(site_order) < 2:
        logger.error(f"Insufficient filter sites defined ({site_order}) for conduction analysis.")
        return {}

    # Convert relative site positions to absolute positions
    try:
        abs_site_positions = {site: filter_sites[site] + g1_reference for site in site_order}
        logger.debug(f"Absolute site positions: {abs_site_positions}")
    except KeyError as e:
        logger.error(f"Missing site {e} in filter_sites dictionary provided. Aborting.")
        return {}
    except Exception as e:
        logger.error(f"Error calculating absolute site positions: {e}")
        return {}

    # Define site occupancy threshold (adjust if needed)
    SITE_OCCUPANCY_THRESHOLD_A = 2.0
    logger.info(f"Using site occupancy threshold: {SITE_OCCUPANCY_THRESHOLD_A} Å")

    # Initialize tracking structures for each ion
    ion_states = {}
    for ion_idx in ion_indices:
        ion_states[ion_idx] = {
            'current_site': None,
            'previous_site': None,
            'in_transit': False,
            'transit_direction': None, # 'inward', 'outward'
            'transit_entry_frame': None,
            'transit_entry_time': None,
            'transit_sites_visited': set(), # Track sites visited during this specific transit
            'completed_conductions': [], # List to store completed conduction event dicts
            'site_transitions': []       # List to store site-to-site transition dicts
        }

    n_frames = len(time_points)
    logger.info(f"Processing {n_frames} frames for {len(ion_indices)} ions...")

    # --- Process each frame --- #
    for frame_idx in range(n_frames):
        t = time_points[frame_idx]

        for ion_idx in ion_indices:
            ion_z_array = ions_z_positions.get(ion_idx)
            if ion_z_array is None or frame_idx >= len(ion_z_array) or np.isnan(ion_z_array[frame_idx]):
                # Ion doesn't exist, is out of bounds for this frame, or has NaN position
                # If it was previously in a site, mark it as leaving
                if ion_states[ion_idx]['current_site'] is not None:
                    ion_states[ion_idx]['previous_site'] = ion_states[ion_idx]['current_site']
                    ion_states[ion_idx]['current_site'] = None
                    ion_states[ion_idx]['in_transit'] = False # Exited the channel region
                    ion_states[ion_idx]['transit_direction'] = None
                continue # Skip to next ion

            z_pos = ion_z_array[frame_idx]
            state = ion_states[ion_idx] # Get current state dict for this ion
            state['previous_site'] = state['current_site'] # Store last frame's site

            # --- Determine current binding site --- #
            current_site_determined = None
            min_dist = float('inf')
            # Find closest site within threshold
            for site_label, site_z in abs_site_positions.items():
                dist = abs(z_pos - site_z)
                if dist < SITE_OCCUPANCY_THRESHOLD_A and dist < min_dist:
                    min_dist = dist
                    current_site_determined = site_label

            # If not in a specific site, check if between Cavity and S0 (intermediate)
            if current_site_determined is None:
                 # Use actual min/max Z of defined sites as boundaries
                 min_z_boundary = abs_site_positions[site_order[0]]
                 max_z_boundary = abs_site_positions[site_order[-1]]
                 # Ensure min < max
                 if min_z_boundary > max_z_boundary: min_z_boundary, max_z_boundary = max_z_boundary, min_z_boundary

                 if min_z_boundary <= z_pos <= max_z_boundary:
                     current_site_determined = 'intermediate'
                 # else: it's outside the S0-Cavity range, current_site_determined remains None

            state['current_site'] = current_site_determined
            previous_site = state['previous_site']

            # --- Site Transition Detection (with 2-sided tolerance) --- #
            if previous_site != state['current_site'] and previous_site is not None and state['current_site'] is not None:
                # Check if the transition is between two valid, adjacent, non-intermediate sites
                if previous_site in site_order and state['current_site'] in site_order:
                    try:
                        prev_idx = site_order.index(previous_site)
                        curr_idx = site_order.index(state['current_site'])

                        if abs(curr_idx - prev_idx) == 1:
                            # Potential adjacent transition detected at frame_idx.
                            # Now apply the stricter two-sided tolerance check.
                            new_site = state['current_site']
                            is_confirmed_transition = True

                            # 1. Backward Check: Was it stable in previous_site before frame_idx?
                            if frame_idx < ION_TRANSITION_TOLERANCE_FRAMES:
                                is_confirmed_transition = False # Not enough history
                            else:
                                for k in range(1, ION_TRANSITION_TOLERANCE_FRAMES):
                                    past_frame_idx = frame_idx - k
                                    past_z = ions_z_positions.get(ion_idx)[past_frame_idx] if ions_z_positions.get(ion_idx) is not None and past_frame_idx < len(ions_z_positions.get(ion_idx)) else np.nan
                                    past_site = None
                                    if not np.isnan(past_z):
                                        min_past_dist = float('inf')
                                        for site_label_past, site_z_past in abs_site_positions.items():
                                            dist_past = abs(past_z - site_z_past)
                                            if dist_past < SITE_OCCUPANCY_THRESHOLD_A and dist_past < min_past_dist:
                                                min_past_dist = dist_past
                                                past_site = site_label_past
                                    # If any past frame wasn't in the required previous_site, fail the check
                                    if past_site != previous_site:
                                        is_confirmed_transition = False
                                        break

                            # 2. Forward Check: Does it stay stable in new_site from frame_idx onwards?
                            if is_confirmed_transition: # Only check forward if backward passed
                                if frame_idx + ION_TRANSITION_TOLERANCE_FRAMES > n_frames:
                                    is_confirmed_transition = False # Not enough future frames
                                else:
                                    for j in range(ION_TRANSITION_TOLERANCE_FRAMES): # Check from frame_idx itself up to tolerance-1 ahead
                                        future_frame_idx = frame_idx + j
                                        future_z = ions_z_positions.get(ion_idx)[future_frame_idx] if ions_z_positions.get(ion_idx) is not None and future_frame_idx < len(ions_z_positions.get(ion_idx)) else np.nan
                                        future_site = None
                                        if not np.isnan(future_z):
                                            min_future_dist = float('inf')
                                            for site_label_future, site_z_future in abs_site_positions.items():
                                                dist_future = abs(future_z - site_z_future)
                                                if dist_future < SITE_OCCUPANCY_THRESHOLD_A and dist_future < min_future_dist:
                                                    min_future_dist = dist_future
                                                    future_site = site_label_future
                                        # If the site at any point in the window doesn't match the new site, fail the check
                                        if future_site != new_site:
                                            is_confirmed_transition = False
                                            break

                            # Record the transition only if both backward and forward checks passed
                            if is_confirmed_transition:
                                direction = 'upward' if curr_idx > prev_idx else 'downward'
                                transition = {
                                    'ion_idx': ion_idx,
                                    'frame': frame_idx,
                                    'time': t,
                                    'from_site': previous_site,
                                    'to_site': new_site,
                                    'direction': direction,
                                    'z_position': z_pos
                                }
                                state['site_transitions'].append(transition)
                                # Note: We don't skip frames here, as subsequent frames might start
                                # the backward check for a potential transition back.

                    except ValueError:
                         logger.warning(f"Site {previous_site} or {state['current_site']} not found in site_order list during transition check.")

            # --- Conduction Event Logic --- #
            # Entering the channel region?
            if previous_site is None and state['current_site'] is not None and state['current_site'] != 'intermediate':
                if state['current_site'] == site_order[0]: # Entered at Cavity end
                    state['in_transit'] = True
                    state['transit_direction'] = 'outward'
                    state['transit_entry_frame'] = frame_idx
                    state['transit_entry_time'] = t
                    state['transit_sites_visited'] = {state['current_site']} # Start with entry site
                elif state['current_site'] == site_order[-1]: # Entered at S0 end
                    state['in_transit'] = True
                    state['transit_direction'] = 'inward'
                    state['transit_entry_frame'] = frame_idx
                    state['transit_entry_time'] = t
                    state['transit_sites_visited'] = {state['current_site']}

            # Exiting the channel region?
            elif previous_site is not None and state['current_site'] is None:
                state['in_transit'] = False # Exited channel
                state['transit_direction'] = None
                state['transit_sites_visited'] = set() # Reset visited sites on exit

            # Continuing a transit?
            elif state['in_transit']:
                # Add newly visited site (if valid)
                if state['current_site'] is not None and state['current_site'] != 'intermediate':
                    state['transit_sites_visited'].add(state['current_site'])

                # Check for completion
                sf_sites_visited = any(s in ['S1','S2','S3','S4'] for s in state['transit_sites_visited'])

                # Completed outward conduction (Cavity -> S0)?
                if state['transit_direction'] == 'outward' and state['current_site'] == site_order[-1]:
                    if sf_sites_visited: # Must have visited at least one SF site (S1-S4)
                        conduction = {
                            'ion_idx': ion_idx,
                            'direction': 'outward',
                            'entry_frame': state['transit_entry_frame'],
                            'exit_frame': frame_idx,
                            'entry_time': state['transit_entry_time'],
                            'exit_time': t,
                            'transit_time': t - state['transit_entry_time'],
                            'sites_visited': sorted(list(state['transit_sites_visited']), key=site_order.index)
                        }
                        state['completed_conductions'].append(conduction)
                    # Reset transit state regardless of whether it counted
                    state['in_transit'] = False
                    state['transit_direction'] = None
                    state['transit_sites_visited'] = set()

                # Completed inward conduction (S0 -> Cavity)?
                elif state['transit_direction'] == 'inward' and state['current_site'] == site_order[0]:
                    if sf_sites_visited: # Must have visited at least one SF site (S1-S4)
                        conduction = {
                            'ion_idx': ion_idx,
                            'direction': 'inward',
                            'entry_frame': state['transit_entry_frame'],
                            'exit_frame': frame_idx,
                            'entry_time': state['transit_entry_time'],
                            'exit_time': t,
                            'transit_time': t - state['transit_entry_time'],
                            'sites_visited': sorted(list(state['transit_sites_visited']), key=site_order.index, reverse=True)
                        }
                        state['completed_conductions'].append(conduction)
                    # Reset transit state
                    state['in_transit'] = False
                    state['transit_direction'] = None
                    state['transit_sites_visited'] = set()

    # --- Compile statistics --- #
    logger.info("Compiling conduction and transition statistics...")
    all_conduction_events = []
    all_transition_events = []

    for ion_idx, state in ion_states.items():
        all_conduction_events.extend(state['completed_conductions'])
        all_transition_events.extend(state['site_transitions'])

    # Calculate summary statistics
    stats = {}
    stats['Ion_ConductionEvents_Total'] = len(all_conduction_events)
    stats['Ion_ConductionEvents_Outward'] = sum(1 for e in all_conduction_events if e['direction'] == 'outward')
    stats['Ion_ConductionEvents_Inward'] = sum(1 for e in all_conduction_events if e['direction'] == 'inward')
    transit_times = [e['transit_time'] for e in all_conduction_events if e['transit_time'] is not None]
    stats['Ion_Conduction_MeanTransitTime_ns'] = np.mean(transit_times) if transit_times else np.nan
    stats['Ion_Conduction_MedianTransitTime_ns'] = np.median(transit_times) if transit_times else np.nan
    stats['Ion_Conduction_StdTransitTime_ns'] = np.std(transit_times) if transit_times else np.nan

    stats['Ion_TransitionEvents_Total'] = len(all_transition_events)
    stats['Ion_TransitionEvents_Upward'] = sum(1 for e in all_transition_events if e['direction'] == 'upward')
    stats['Ion_TransitionEvents_Downward'] = sum(1 for e in all_transition_events if e['direction'] == 'downward')

    # Site-specific transitions (adjacent only)
    site_pairs = list(zip(site_order[:-1], site_order[1:]))
    for site1, site2 in site_pairs:
        key = f'Ion_Transition_{site1}_{site2}'
        # Count transitions FROM the saved list, which now only contains confirmed transitions
        count = sum(1 for e in all_transition_events if (e['from_site']==site1 and e['to_site']==site2) or (e['from_site']==site2 and e['to_site']==site1))
        stats[key] = count

    stats['Ion_Transition_ToleranceFrames'] = ION_TRANSITION_TOLERANCE_FRAMES # Add tolerance to stats

    logger.info(f"Found {stats['Ion_ConductionEvents_Total']} conduction events ({stats['Ion_ConductionEvents_Outward']} outward, {stats['Ion_ConductionEvents_Inward']} inward).")
    logger.info(f"Found {stats['Ion_TransitionEvents_Total']} confirmed adjacent site transitions (Tolerance: {ION_TRANSITION_TOLERANCE_FRAMES} frames; {stats['Ion_TransitionEvents_Upward']} upward, {stats['Ion_TransitionEvents_Downward']} downward).")

    # Save data to CSV files
    save_conduction_data(run_dir, all_conduction_events, all_transition_events, stats)

    # Create visualizations (Placeholder)
    create_conduction_plots(run_dir, all_conduction_events, all_transition_events, stats, time_points)

    logger.info("Ion Conduction & Transition Analysis complete.")
    return stats 