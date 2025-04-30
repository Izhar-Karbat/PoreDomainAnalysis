# filename: pore_analysis/modules/dw_gate_analysis/event_building.py
"""
Functions for building discrete state events from a time series of states.
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

def _verify_and_fix_coverage_per_chain(
    chain_events_list: List[Dict],
    chain_id: str,
    min_frame_overall: int,
    max_frame_overall: int,
    dt_ns: float,
    df_raw_for_time: Optional[pd.DataFrame] = None # Optional raw df for precise times
) -> List[Dict]:
    """
    Internal helper: Checks for gaps in event coverage for a single chain and fills them.
    Uses uppercase 'Frame' and 'Time (ns)' for lookup if df_raw_for_time is provided.

    Args:
        chain_events_list: List of event dictionaries for a single chain, sorted by start_frame.
        chain_id: Identifier for the chain being processed.
        min_frame_overall: The minimum frame index expected (usually 0).
        max_frame_overall: The maximum frame index expected for the trajectory.
        dt_ns: Time step in nanoseconds.
        df_raw_for_time: Optional DataFrame containing original 'Frame' and 'Time (ns)'
                         columns to get more precise start/end times for filled gaps.
                         If None, times are estimated using dt_ns.

    Returns:
        List of event dictionaries with gaps filled. Returns empty list if input is empty.
    """
    if not chain_events_list:
        # If no events, we cannot reliably determine the state for the full duration.
        # Returning an empty list is safer than assuming a state.
        # The calling function should handle the case of no events for a chain.
        logger.warning(f"Chain {chain_id}: No initial events passed to coverage check. Returning empty list.")
        return []

    fixed_events = []
    # Use the index of df_raw_for_time for quick time lookups if available
    time_lookup = None
    # --- MODIFIED: Check for uppercase 'Frame', 'Time (ns)' ---
    if df_raw_for_time is not None and 'Frame' in df_raw_for_time.columns and 'Time (ns)' in df_raw_for_time.columns:
         try:
             # Use 'Frame' as index
             time_lookup = df_raw_for_time.set_index('Frame')['Time (ns)']
             if time_lookup.index.has_duplicates:
                 logger.warning(f"Duplicate frames found in time lookup data for chain {chain_id}. Using first time value for duplicates.")
                 time_lookup = time_lookup[~time_lookup.index.duplicated(keep='first')]
         except Exception as e_lookup:
             logger.warning(f"Failed to create time lookup index for chain {chain_id}: {e_lookup}")
             time_lookup = None # Fallback to estimation
    # --- END MODIFICATION ---

    def get_time(frame_index):
        """Helper to get time from lookup or estimate."""
        if time_lookup is not None:
            try: return time_lookup.loc[frame_index]
            except KeyError: pass # Fall through to estimation if frame not found
        # Estimate if lookup failed or frame missing
        return frame_index * dt_ns

    last_end_frame = min_frame_overall - 1 # Start checking from the beginning

    # 1. Check for gap before the first event
    first_event = chain_events_list[0]
    if first_event['start_frame'] > min_frame_overall:
        gap_start_frame = min_frame_overall
        gap_end_frame = first_event['start_frame'] - 1
        # Use the state of the *next* event to fill the gap at the beginning
        gap_state = first_event['state']
        gap_count = gap_end_frame - gap_start_frame + 1
        start_ns = get_time(gap_start_frame)
        # For end_ns, use the start time of the *next* event if possible, otherwise estimate
        end_ns = first_event.get('start_ns', get_time(first_event['start_frame']))
        duration_ns = end_ns - start_ns

        logger.warning(f"Chain {chain_id}: Found gap at start (Frames {gap_start_frame}-{gap_end_frame}). Filling with state '{gap_state}'.")
        fixed_events.append({
            'chain': chain_id, 'state': gap_state,
            'start_frame': gap_start_frame, 'end_frame': gap_end_frame,
            'frame_count': gap_count, 'duration_ns': duration_ns,
            'start_ns': start_ns, 'end_ns': end_ns
        })
        last_end_frame = gap_end_frame

    # 2. Process existing events and check for gaps between them
    for i, event in enumerate(chain_events_list):
        # Check for gap *before* this event (if start > last_end + 1)
        if event['start_frame'] > last_end_frame + 1:
            gap_start_frame = last_end_frame + 1
            gap_end_frame = event['start_frame'] - 1
            # Assign state based on the *previous* event's state
            # Ensure fixed_events is not empty before accessing [-1]
            gap_state = fixed_events[-1]['state'] if fixed_events else event['state'] # Fallback needed?
            gap_count = gap_end_frame - gap_start_frame + 1
            start_ns = get_time(gap_start_frame)
            # Use start time of current event as end time of gap
            end_ns = event.get('start_ns', get_time(event['start_frame']))
            duration_ns = end_ns - start_ns

            logger.warning(f"Chain {chain_id}: Found mid gap (Frames {gap_start_frame}-{gap_end_frame}). Filling with state '{gap_state}'.")
            fixed_events.append({
                'chain': chain_id, 'state': gap_state,
                'start_frame': gap_start_frame, 'end_frame': gap_end_frame,
                'frame_count': gap_count, 'duration_ns': duration_ns,
                'start_ns': start_ns, 'end_ns': end_ns
            })

        # Add the current event itself
        fixed_events.append(event)
        last_end_frame = event['end_frame']

    # 3. Check for gap after the last event
    if last_end_frame < max_frame_overall:
        gap_start_frame = last_end_frame + 1
        gap_end_frame = max_frame_overall
        # Ensure fixed_events is not empty before accessing [-1]
        gap_state = fixed_events[-1]['state'] if fixed_events else "Unknown" # Assume state of last event continues
        gap_count = gap_end_frame - gap_start_frame + 1
        start_ns = get_time(gap_start_frame)
        # Estimate end_ns based on last frame
        end_ns = get_time(gap_end_frame + 1) # Estimate time *after* last frame
        duration_ns = end_ns - start_ns

        logger.warning(f"Chain {chain_id}: Found gap at end (Frames {gap_start_frame}-{gap_end_frame}). Filling with state '{gap_state}'.")
        fixed_events.append({
            'chain': chain_id, 'state': gap_state,
            'start_frame': gap_start_frame, 'end_frame': gap_end_frame,
            'frame_count': gap_count, 'duration_ns': duration_ns,
            'start_ns': start_ns, 'end_ns': end_ns
        })

    return fixed_events


def build_events_from_states(
    df_states_long: pd.DataFrame,
    dt_ns: float,
    state_col: str = 'state' # Column containing the debounced states
) -> pd.DataFrame:
    """
    Builds continuous state events from a time series of debounced states.
    Ensures full trajectory coverage by filling gaps.

    Args:
        df_states_long: DataFrame in long format with columns: 'Frame', 'Time (ns)',
                        'chain', and the specified `state_col` containing debounced states.
                        Must be sorted by chain, then Frame.
        dt_ns: Time step between frames in nanoseconds.
        state_col: The name of the column holding the debounced state information.

    Returns:
        DataFrame of events with columns: 'Chain', 'State', 'Start Frame', 'End Frame',
        'Frame Count', 'Start Time (ns)', 'End Time (ns)', 'Duration (ns)'.
        Returns an empty DataFrame if input is invalid or no events are built.
    """
    # --- MODIFIED required_cols to use 'Frame', 'Time (ns)' (uppercase) ---
    required_cols = ['Frame', 'Time (ns)', 'chain', state_col]
    # --- END MODIFICATION ---

    if df_states_long is None or df_states_long.empty or not all(col in df_states_long.columns for col in required_cols):
        logger.error(f"Cannot build events: Input DataFrame is invalid or missing required columns ({required_cols}). Found columns: {list(df_states_long.columns) if df_states_long is not None else 'None'}")
        # Return empty DataFrame with expected FINAL column names
        return pd.DataFrame(columns=[
            'Chain', 'State', 'Start Frame', 'End Frame',
            'Frame Count', 'Start Time (ns)', 'End Time (ns)', 'Duration (ns)'
        ])

    if dt_ns <= 0:
        logger.error(f"Cannot build events: Invalid time step dt_ns ({dt_ns}). Must be positive.")
        # Return empty DataFrame with expected FINAL column names
        return pd.DataFrame(columns=[
            'Chain', 'State', 'Start Frame', 'End Frame',
            'Frame Count', 'Start Time (ns)', 'End Time (ns)', 'Duration (ns)'
        ])

    logger.info(f"Building events from debounced states in column '{state_col}'...")
    all_final_events = []
    # --- MODIFIED: Use uppercase 'Frame' ---
    min_frame_overall = df_states_long['Frame'].min()
    max_frame_overall = df_states_long['Frame'].max()
    # --- END MODIFICATION ---

    # Sort just in case, using uppercase 'Frame'
    df_sorted = df_states_long.sort_values(['chain', 'Frame']).reset_index(drop=True)

    for chain_id, sub_df in df_sorted.groupby('chain'):
        if sub_df.empty:
            logger.warning(f"No data for chain {chain_id} during event building.")
            continue

        sub_df = sub_df.copy() # Avoid SettingWithCopyWarning
        # Detect consecutive blocks of the same state, handle NaNs correctly
        # Create a temporary column where NaNs are replaced by a unique placeholder
        nan_placeholder = "###NAN_STATE###"
        state_filled = sub_df[state_col].fillna(nan_placeholder)
        sub_df['block'] = (state_filled != state_filled.shift()).cumsum()

        chain_initial_events = []
        for _, group in sub_df.groupby('block'):
            if group.empty:
                continue
            # --- MODIFIED: Use uppercase 'Frame' and 'Time (ns)' ---
            start_frame = group['Frame'].min()
            end_frame = group['Frame'].max()
            frame_count = end_frame - start_frame + 1
            state_val = group[state_col].iloc[0]
            start_time = group['Time (ns)'].iloc[0] # Use first time_ns value in the group
            # --- END MODIFICATION ---

            # Estimate duration and end time
            duration_ns = frame_count * dt_ns
            # Use time of the last frame in the group + dt_ns for end time for better accuracy with variable dt
            # Or simply start_time + duration_ns if dt is assumed constant
            end_time = start_time + duration_ns # Simpler approach assuming constant dt

            chain_initial_events.append({
                'chain': chain_id,
                'state': state_val, # This can be NaN
                'start_frame': start_frame,
                'end_frame': end_frame,
                'frame_count': frame_count,
                'duration_ns': duration_ns,
                'start_ns': start_time,
                'end_ns': end_time
            })

        if not chain_initial_events:
            logger.warning(f"No initial events generated for chain {chain_id} from state blocks.")
            continue

        # Verify and fix coverage for this chain
        # Ensure _verify_and_fix_coverage_per_chain also uses 'Frame' and 'Time (ns)'
        fixed_events_for_chain = _verify_and_fix_coverage_per_chain(
            chain_initial_events, chain_id,
            min_frame_overall, max_frame_overall,
            dt_ns,
            # --- MODIFIED: Pass correct columns ---
            df_raw_for_time=sub_df[['Frame', 'Time (ns)']]
            # --- END MODIFICATION ---
        )
        all_final_events.extend(fixed_events_for_chain)

    if not all_final_events:
        logger.warning("Failed to build any events across all chains.") # Changed to warning
        # Return empty DataFrame with expected FINAL column names
        return pd.DataFrame(columns=[
            'Chain', 'State', 'Start Frame', 'End Frame',
            'Frame Count', 'Start Time (ns)', 'End Time (ns)', 'Duration (ns)'
        ])

    # Create final DataFrame
    events_df = pd.DataFrame(all_final_events)

    # --- MODIFIED: Rename columns AFTER DataFrame creation ---
    # Ensure correct column names before returning (using standard convention)
    column_rename_map = {
        'chain': 'Chain',
        'state': 'State',
        'start_frame': 'Start Frame',
        'end_frame': 'End Frame',
        'frame_count': 'Frame Count',
        'start_ns': 'Start Time (ns)',
        'end_ns': 'End Time (ns)',
        'duration_ns': 'Duration (ns)'
    }
    events_df.rename(columns=column_rename_map, inplace=True)
    # --- END MODIFICATION ---

    # Ensure all expected columns exist, add if missing (filled with NA)
    expected_final_cols = list(column_rename_map.values())
    for col in expected_final_cols:
        if col not in events_df.columns:
            events_df[col] = pd.NA


    # Sort and reset index
    events_df = events_df.sort_values(['Chain', 'Start Frame']).reset_index(drop=True)

    # Final verification of coverage (optional, logs warnings)
    # Ensure _verify_final_coverage also uses 'Start Frame' and 'End Frame'
    coverage_ok = _verify_final_coverage(events_df, min_frame_overall, max_frame_overall)
    if not coverage_ok:
         logger.warning("Final event coverage verification failed for one or more chains.")

    logger.info(f"Built {len(events_df)} final events across all chains.")
    if not events_df.empty:
         logger.debug(f"Event DataFrame head:\n{events_df.head().to_string()}")

    # Return only the expected columns in the standard order
    return events_df[expected_final_cols]


def _verify_final_coverage(events_df: pd.DataFrame, min_frame: int, max_frame: int) -> bool:
    """
    Helper function to verify final event coverage after building and gap filling.
    Uses uppercase 'Start Frame' and 'End Frame'.
    """
    if events_df is None or events_df.empty:
        return True # Coverage is trivially ok if no events

    is_continuous = True
    # --- MODIFIED: Use uppercase column names ---
    required_verify_cols = ['Chain', 'Start Frame', 'End Frame']
    if not all(col in events_df.columns for col in required_verify_cols):
         logger.error(f"Cannot verify coverage: Missing columns {required_verify_cols}")
         return False

    for ch, ev_df in events_df.groupby('Chain'):
        if ev_df.empty:
            continue
        e = ev_df.sort_values('Start Frame')

        # Check start
        if e['Start Frame'].iloc[0] != min_frame:
            logger.warning(f"Final Coverage Issue [Chain {ch}]: Events do not start at frame {min_frame} (starts at {e['Start Frame'].iloc[0]}).")
            is_continuous = False

        # Check gaps between events
        for i in range(len(e) - 1):
            if e['End Frame'].iloc[i] + 1 != e['Start Frame'].iloc[i+1]:
                logger.warning(f"Final Coverage Issue [Chain {ch}]: Gap found between frame {e['End Frame'].iloc[i]} and {e['Start Frame'].iloc[i+1]}.")
                is_continuous = False

        # Check end
        if e['End Frame'].iloc[-1] != max_frame:
            logger.warning(f"Final Coverage Issue [Chain {ch}]: Events do not end at frame {max_frame} (ends at {e['End Frame'].iloc[-1]}).")
            is_continuous = False
    # --- END MODIFICATION ---

    return is_continuous