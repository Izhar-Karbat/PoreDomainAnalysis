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

    Args:
        chain_events_list: List of event dictionaries for a single chain, sorted by start_frame.
        chain_id: Identifier for the chain being processed.
        min_frame_overall: The minimum frame index expected (usually 0).
        max_frame_overall: The maximum frame index expected for the trajectory.
        dt_ns: Time step in nanoseconds.
        df_raw_for_time: Optional DataFrame containing original 'frame' and 'time_ns'
                         columns to get more precise start/end times for filled gaps.
                         If None, times are estimated using dt_ns.

    Returns:
        List of event dictionaries with gaps filled.
    """
    if not chain_events_list:
        # If no events, create a single event covering the whole range, using the first frame's state if possible
        # This requires access to the original state data, which isn't ideal here.
        # For now, return empty, or perhaps log an error/warning upstream.
        logger.warning(f"Chain {chain_id}: No initial events found during coverage check. Cannot determine state for full duration.")
        return []

    fixed_events = []
    last_end_frame = min_frame_overall - 1 # Start checking from the beginning

    # 1. Check for gap before the first event
    first_event = chain_events_list[0]
    if first_event['start_frame'] > min_frame_overall:
        gap_start_frame = min_frame_overall
        gap_end_frame = first_event['start_frame'] - 1
        gap_state = first_event['state'] # Assume gap takes state of the *next* event
        gap_count = gap_end_frame - gap_start_frame + 1
        start_ns = gap_start_frame * dt_ns # Estimate time
        if df_raw_for_time is not None:
            time_val = df_raw_for_time.loc[df_raw_for_time['frame'] == gap_start_frame, 'time_ns'].iloc[0]
            if pd.notna(time_val):
                start_ns = time_val

        duration_ns = gap_count * dt_ns
        end_ns = start_ns + duration_ns

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
            gap_state = fixed_events[-1]['state']
            gap_count = gap_end_frame - gap_start_frame + 1
            start_ns = gap_start_frame * dt_ns # Estimate time
            if df_raw_for_time is not None:
                 time_val = df_raw_for_time.loc[df_raw_for_time['frame'] == gap_start_frame, 'time_ns'].iloc[0]
                 if pd.notna(time_val):
                     start_ns = time_val

            duration_ns = gap_count * dt_ns
            end_ns = start_ns + duration_ns

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
        gap_state = fixed_events[-1]['state'] # Assume state of last event continues
        gap_count = gap_end_frame - gap_start_frame + 1
        start_ns = gap_start_frame * dt_ns # Estimate time
        if df_raw_for_time is not None:
             time_val = df_raw_for_time.loc[df_raw_for_time['frame'] == gap_start_frame, 'time_ns'].iloc[0]
             if pd.notna(time_val):
                 start_ns = time_val

        duration_ns = gap_count * dt_ns
        end_ns = start_ns + duration_ns

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
        df_states_long: DataFrame in long format with columns: 'frame', 'time_ns',
                        'chain', and the specified `state_col` containing debounced states.
                        Must be sorted by chain, then frame.
        dt_ns: Time step between frames in nanoseconds.
        state_col: The name of the column holding the debounced state information.

    Returns:
        DataFrame of events with columns: 'chain', 'state', 'start_frame', 'end_frame',
        'frame_count', 'duration_ns', 'start_ns', 'end_ns'.
        Returns an empty DataFrame if input is invalid or no events are built.
    """
    required_cols = ['frame', 'time_ns', 'chain', state_col]
    if df_states_long is None or df_states_long.empty or not all(col in df_states_long.columns for col in required_cols):
        logger.error(f"Cannot build events: Input DataFrame is invalid or missing required columns ({required_cols}).")
        return pd.DataFrame(columns=[
            'chain', 'state', 'start_frame', 'end_frame',
            'frame_count', 'duration_ns', 'start_ns', 'end_ns'
        ])

    logger.info(f"Building events from debounced states in column '{state_col}'...")
    all_final_events = []
    min_frame_overall = df_states_long['frame'].min()
    max_frame_overall = df_states_long['frame'].max()

    # Sort just in case, although caller should ideally provide sorted data
    df_sorted = df_states_long.sort_values(['chain', 'frame']).reset_index(drop=True)

    for chain_id, sub_df in df_sorted.groupby('chain'):
        if sub_df.empty:
            logger.warning(f"No data for chain {chain_id} during event building.")
            continue

        sub_df = sub_df.copy() # Avoid SettingWithCopyWarning
        # Detect consecutive blocks of the same state
        sub_df['block'] = (sub_df[state_col] != sub_df[state_col].shift()).cumsum()

        chain_initial_events = []
        for _, group in sub_df.groupby('block', observed=True):
            if group.empty:
                continue
            start_frame = group['frame'].min()
            end_frame = group['frame'].max()
            frame_count = end_frame - start_frame + 1
            start_time = group['time_ns'].min() # Get actual start time from data
            # Estimate duration and end time
            # A more precise end time would be the start time of the *next* event's first frame,
            # but frame_count * dt_ns is a common and usually sufficient approximation.
            duration_ns = frame_count * dt_ns
            end_time = start_time + duration_ns

            chain_initial_events.append({
                'chain': chain_id,
                'state': group[state_col].iloc[0],
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
        # Pass only relevant columns of original df if needed for time lookup
        df_time_lookup = df_sorted[['frame', 'time_ns']].drop_duplicates(subset='frame').set_index('frame')

        fixed_events_for_chain = _verify_and_fix_coverage_per_chain(
            chain_initial_events, chain_id,
            min_frame_overall, max_frame_overall,
            dt_ns,
            df_raw_for_time=df_sorted[['frame', 'time_ns']] # Pass raw data for potential time lookup
        )
        all_final_events.extend(fixed_events_for_chain)

    if not all_final_events:
        logger.error("Failed to build any events across all chains.")
        return pd.DataFrame(columns=[
            'chain', 'state', 'start_frame', 'end_frame',
            'frame_count', 'duration_ns', 'start_ns', 'end_ns'
        ])

    # Create final DataFrame
    events_df = pd.DataFrame(all_final_events)
    # Sort and reset index
    events_df = events_df.sort_values(['chain', 'start_frame']).reset_index(drop=True)

    # Final verification of coverage (optional, logs warnings)
    _verify_final_coverage(events_df, min_frame_overall, max_frame_overall)

    logger.info(f"Built {len(events_df)} events across all chains.")
    logger.debug(f"Event DataFrame head:\n{events_df.head().to_string()}")

    return events_df

def _verify_final_coverage(events_df: pd.DataFrame, min_frame: int, max_frame: int):
    """Helper function to verify final event coverage after building and gap filling."""
    if events_df is None or events_df.empty:
        return True

    is_continuous = True
    for ch, ev_df in events_df.groupby('chain'):
        if ev_df.empty:
            continue
        e = ev_df.sort_values('start_frame')

        # Check start
        if e['start_frame'].iloc[0] != min_frame:
            logger.warning(f"Final Coverage Issue [Chain {ch}]: Events do not start at frame {min_frame} (starts at {e['start_frame'].iloc[0]}).")
            is_continuous = False

        # Check gaps between events
        for i in range(len(e) - 1):
            if e['end_frame'].iloc[i] + 1 != e['start_frame'].iloc[i+1]:
                logger.warning(f"Final Coverage Issue [Chain {ch}]: Gap found between frame {e['end_frame'].iloc[i]} and {e['start_frame'].iloc[i+1]}.")
                is_continuous = False

        # Check end
        if e['end_frame'].iloc[-1] != max_frame:
            logger.warning(f"Final Coverage Issue [Chain {ch}]: Events do not end at frame {max_frame} (ends at {e['end_frame'].iloc[-1]}).")
            is_continuous = False

    return is_continuous

# Placeholder for event building functions 