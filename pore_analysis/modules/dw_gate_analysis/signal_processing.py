# filename: pore_analysis/modules/dw_gate_analysis/signal_processing.py
"""
Functions for processing the raw DW-gate distance signal, including:
- Kernel Density Estimation (KDE) for identifying potential state distances.
- Determining reference distances for open/closed states (using KDE/KMeans or defaults).
- Run-Length Encoding (RLE) debouncing to filter out short-lived fluctuations.
"""

import logging
import os
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks

# KMeans import needed for determine_reference_distances
try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    # Define a dummy KMeans to avoid errors if sklearn is missing,
    # determine_reference_distances will handle the fallback to defaults.
    class KMeans:
        def __init__(self, *args, **kwargs): pass
        def fit(self, *args, **kwargs): raise ImportError("Scikit-learn not installed.")
        cluster_centers_ = None

CLOSED_STATE = "closed"
OPEN_STATE = "open"

logger = logging.getLogger(__name__)

def find_kde_peaks(
    distances: np.ndarray,
    bw_method: Optional[str] = 'scott',
    peak_height_fraction: float = 0.05,
    peak_distance: int = 30,
    peak_prominence: float = 0.05
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Performs Gaussian KDE on distances and finds peaks.

    Args:
        distances: Array of distance values.
        bw_method: Bandwidth method for KDE (see scipy.stats.gaussian_kde).
        peak_height_fraction: Minimum peak height as a fraction of max KDE value.
        peak_distance: Required minimal horizontal distance (samples) between peaks.
        peak_prominence: Required prominence of peaks.

    Returns:
        Tuple containing:
        - x_kde: X values for the KDE plot.
        - y_kde: Y values (density) for the KDE plot.
        - peak_distances: Distances corresponding to found peaks.
        - peak_heights: KDE density values at the peaks.
        Returns (None, None, None, None) if KDE fails or no peaks are found.
    """
    if distances is None or len(distances) < 10:
        logger.debug("Insufficient data points for KDE.")
        return None, None, None, None

    try:
        # Handle potential LinAlgError during KDE instantiation
        try:
            kde = gaussian_kde(distances, bw_method=bw_method)
        except np.linalg.LinAlgError as lae:
            logger.warning(f"LinAlgError during KDE instantiation (likely constant data): {lae}. Cannot perform KDE.")
            return None, None, None, None

        # Define x range based on data, ensuring it's reasonable
        dist_min = np.min(distances)
        dist_max = np.max(distances)
        # Add padding, ensure min is not negative if data allows
        x_min = max(0, dist_min - 1.0)
        x_max = dist_max + 1.0
        # Handle case where min == max
        if np.isclose(x_min, x_max): x_max = x_min + 1.0
        x_kde = np.linspace(x_min, x_max, 500)
        y_kde = kde(x_kde)

        max_height = y_kde.max()
        if max_height <= 0:
             logger.warning("KDE maximum height is zero or negative, cannot find peaks.")
             return x_kde, y_kde, np.array([]), np.array([])

        peaks_indices, _ = find_peaks(
            y_kde,
            height=peak_height_fraction * max_height,
            distance=peak_distance,
            prominence=peak_prominence * max_height # Prominence relative to max height
        )

        if len(peaks_indices) == 0:
            logger.debug("No peaks found meeting the criteria.")
            return x_kde, y_kde, np.array([]), np.array([])

        peak_distances = x_kde[peaks_indices]
        peak_heights = y_kde[peaks_indices]
        logger.debug(f"Found {len(peak_distances)} KDE peaks at distances: {np.round(peak_distances, 2)}")
        return x_kde, y_kde, peak_distances, peak_heights

    except Exception as e:
        logger.error(f"KDE calculation or peak finding failed: {e}", exc_info=True)
        return None, None, None, None

# --- MODIFIED: Removed plotting, returns calculated refs ---
def determine_reference_distances(
    df_raw: pd.DataFrame,
    default_closed_ref: float,
    default_open_ref: float,
    # Removed plotting arguments: distance_threshold, output_dir, plot_distributions
    kde_bw_method: Optional[str] = 'scott',
    kde_peak_height_fraction: float = 0.05,
    kde_peak_distance: int = 30,
    kde_peak_prominence: float = 0.05,
) -> Tuple[float, float, Dict[str, Any]]: # Return refs and KDE results for potential plotting elsewhere
    """
    Determines closed and open reference distances using KDE on combined and per-chain data,
    followed by KMeans clustering on detected peaks.
    Falls back to defaults if KDE/KMeans fails.
    **This version does NOT generate plots.**

    Args:
        df_raw: DataFrame with raw distances (must contain 'distance' and 'chain' columns).
        default_closed_ref: Default reference distance for the closed state.
        default_open_ref: Default reference distance for the open state.
        kde_bw_method: Bandwidth method for KDE.
        kde_peak_height_fraction: Peak height fraction for KDE peak finding.
        kde_peak_distance: Peak distance for KDE peak finding.
        kde_peak_prominence: Peak prominence for KDE peak finding.

    Returns:
        Tuple containing:
        - final_closed_ref: The determined or default closed reference distance.
        - final_open_ref: The determined or default open reference distance.
        - kde_plot_data: Dictionary containing data needed to optionally generate
                         the KDE plot later (e.g., by computation.py or visualization.py).
                         Includes keys like 'combined_kde', 'per_chain_kde', 'all_distances'.
    """
    if not SKLEARN_AVAILABLE:
        logger.error("Scikit-learn not found. Cannot perform KMeans clustering for reference distances.")
        logger.warning(f"Using default reference distances: Closed={default_closed_ref:.2f}, Open={default_open_ref:.2f}")
        return default_closed_ref, default_open_ref, {} # Return empty plot data

    if df_raw is None or df_raw.empty or 'distance' not in df_raw.columns or 'chain' not in df_raw.columns:
        logger.warning("Raw distance data invalid or missing. Cannot calculate reference distances. Using defaults.")
        return default_closed_ref, default_open_ref, {}

    all_distances = df_raw['distance'].dropna().values
    if len(all_distances) < 10: # Need enough points for meaningful KDE/KMeans
        logger.warning(f"Only {len(all_distances)} valid distance values found. Insufficient for KDE/KMeans. Using defaults.")
        return default_closed_ref, default_open_ref, {'all_distances': all_distances.tolist()} # Return distances for basic histogram


    logger.info("Attempting to determine reference distances from data using KDE and KMeans...")
    kde_plot_data = {'all_distances': all_distances.tolist(), 'per_chain_kde': {}} # Store data for potential plotting

    # --- Per-chain analysis to collect peaks ---
    all_chain_peaks = []
    unique_chains = sorted(df_raw['chain'].unique())
    for chain in unique_chains:
        chain_dist = df_raw.loc[df_raw['chain'] == chain, 'distance'].dropna().values
        x_kde, y_kde, peaks, heights = find_kde_peaks(
            chain_dist, kde_bw_method, kde_peak_height_fraction, kde_peak_distance, kde_peak_prominence
        )
        if peaks is not None and len(peaks) > 0:
            all_chain_peaks.extend(peaks)
        # Store results even if no peaks found, for potential plotting
        kde_plot_data['per_chain_kde'][chain] = {
            'x': x_kde.tolist() if x_kde is not None else None,
            'y': y_kde.tolist() if y_kde is not None else None,
            'peaks': peaks.tolist() if peaks is not None else [],
            'heights': heights.tolist() if heights is not None else []
        }

    # --- Combined analysis ---
    x_all, y_all, peaks_all, heights_all = find_kde_peaks(
         all_distances, kde_bw_method, kde_peak_height_fraction, kde_peak_distance, kde_peak_prominence
    )
    # Store combined results for potential plotting
    kde_plot_data['combined_kde'] = {
        'x': x_all.tolist() if x_all is not None else None,
        'y': y_all.tolist() if y_all is not None else None,
        'peaks': peaks_all.tolist() if peaks_all is not None else [],
        'heights': heights_all.tolist() if heights_all is not None else []
    }

    # --- Determine References via KMeans ---
    final_closed_ref = default_closed_ref
    final_open_ref = default_open_ref
    kmeans_centers = None

    if all_chain_peaks:
        unique_peaks = sorted(list(set(all_chain_peaks)))
        if len(unique_peaks) >= 2:
            try:
                kp = np.array(unique_peaks).reshape(-1, 1)
                # Explicitly set n_init to suppress warning
                kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(kp)
                # Original error was here: sorted() returns list, flatten returns np.array
                kmeans_centers_np = kmeans.cluster_centers_.flatten() # This is a NumPy array
                centers = sorted(kmeans_centers_np) # This is a Python list
                final_closed_ref, final_open_ref = centers[0], centers[1]
                kmeans_centers = centers # Store the *list* for plot data
                logger.info(f"KDE/KMeans derived references: Closed={final_closed_ref:.2f} Å, Open={final_open_ref:.2f} Å")
            except ValueError as ve:
                 logger.warning(f"KMeans clustering failed ({ve}). Using defaults ({default_closed_ref:.2f}, {default_open_ref:.2f}).")
            except Exception as e:
                logger.warning(f"KMeans clustering on peaks failed unexpectedly: {e}. Using defaults ({default_closed_ref:.2f}, {default_open_ref:.2f}).")
        elif len(unique_peaks) == 1:
            logger.warning(f"Only one unique KDE peak found ({unique_peaks[0]:.2f} Å). Insufficient for clustering. Using defaults ({default_closed_ref:.2f}, {default_open_ref:.2f}).")
        else:
            logger.warning(f"No KDE peaks found across any chain. Using defaults ({default_closed_ref:.2f}, {default_open_ref:.2f}).")
    else:
        logger.warning(f"No KDE peaks collected from per-chain analysis. Using defaults ({default_closed_ref:.2f}, {default_open_ref:.2f}).")

    # --- FIX: Assign the list directly, don't call .tolist() ---
    kde_plot_data['kmeans_centers'] = kmeans_centers if kmeans_centers is not None else None
    # --- END FIX ---
    kde_plot_data['final_closed_ref'] = final_closed_ref
    kde_plot_data['final_open_ref'] = final_open_ref

    return final_closed_ref, final_open_ref, kde_plot_data


def _rle_debounce_single_series(states: List[Any], tolerance_frames: int) -> List[Any]:
    """
    Internal RLE debouncing logic for a single list of states (can be string or numeric).
    Merges runs shorter than tolerance into the longer neighboring run. Handles NaNs.

    Args:
        states: List of states.
        tolerance_frames: Minimum run length to keep.

    Returns:
        Debounced list of states.
    """
    if not states:
        return []

    if tolerance_frames <= 1:
        logger.debug("Debouncing tolerance <= 1, skipping RLE.")
        return list(states) # Return a copy

    s = list(states) # Work on a copy
    n = len(s)
    iteration = 0
    max_iterations = n * 2 # Safety break

    while iteration < max_iterations:
        iteration += 1
        runs = []
        if not s: break

        # 1. Identify runs, handling NaN transitions correctly
        current_val = s[0]
        start_idx = 0
        for i in range(1, n):
            # Check for change, handling NaN comparison carefully
            change = False
            if pd.isna(s[i]) and not pd.isna(current_val): change = True
            elif not pd.isna(s[i]) and pd.isna(current_val): change = True
            elif not pd.isna(s[i]) and not pd.isna(current_val) and s[i] != current_val: change = True
            # No change if both are NaN

            if change:
                runs.append({'value': current_val, 'start': start_idx, 'length': i - start_idx})
                current_val = s[i]
                start_idx = i
        runs.append({'value': current_val, 'start': start_idx, 'length': n - start_idx})

        # 2. Find short runs (excluding NaN runs)
        merges_made_in_pass = 0
        indices_to_process = [i for i, run in enumerate(runs) if run['length'] < tolerance_frames and not pd.isna(run['value'])]

        if not indices_to_process:
            logger.debug(f"RLE converged after {iteration-1} passes.")
            break

        # 3. Merge short runs (backward iteration)
        merged_indices = set() # Keep track of indices involved in a merge in this pass
        for idx in sorted(indices_to_process, reverse=True):
            if idx in merged_indices: continue # Skip if already merged

            current_run = runs[idx]
            start = current_run['start']
            length = current_run['length']
            current_val = current_run['value'] # Value of the short run

            # Determine neighbor values and lengths
            prev_run = runs[idx - 1] if idx > 0 else None
            next_run = runs[idx + 1] if idx < len(runs) - 1 else None

            prev_val = prev_run['value'] if prev_run else None
            next_val = next_run['value'] if next_run else None
            prev_len = prev_run['length'] if prev_run else -1
            next_len = next_run['length'] if next_run else -1

            # Decide merge target (priority: non-NaN neighbor)
            merge_target_val = None
            if prev_run and not pd.isna(prev_val) and (next_run is None or pd.isna(next_val) or prev_len >= next_len):
                merge_target_val = prev_val
            elif next_run and not pd.isna(next_val):
                merge_target_val = next_val
            # If both neighbors are NaN, or at ends next to NaN, keep original value (no merge)

            # Apply merge if target is defined and different from current
            if merge_target_val is not None and merge_target_val != current_val:
                for j in range(start, start + length):
                    if 0 <= j < n: s[j] = merge_target_val
                merges_made_in_pass += 1
                merged_indices.add(idx) # Mark as merged
                # Potentially mark neighbours if they get absorbed conceptually,
                # simpler to just recalculate runs in the next pass.

        if merges_made_in_pass == 0 and indices_to_process:
             logger.debug(f"RLE pass {iteration}: No effective merges made despite short runs existing. Converged.")
             break

    if iteration >= max_iterations:
        logger.warning(f"RLE debouncing reached max iterations ({max_iterations}). Results might be unstable or incomplete.")

    return s


def apply_rle_debouncing(
    df_states: pd.DataFrame,
    tolerance_frames: int,
    state_col: str, # Input column with raw states (e.g., 'state_raw')
    debounced_col: str # Output column name for debounced states (e.g., 'state')
) -> pd.DataFrame:
    """
    Applies Run-Length Encoding (RLE) debouncing to state assignments in a DataFrame.
    Operates independently on each 'chain'. Handles NaNs.
    Ensures the output DataFrame contains the debounced column, falling back to raw
    state data within that column if errors occur during processing or assignment.

    Args:
        df_states: DataFrame containing time series data with 'Frame' and 'chain' columns,
                   sorted by chain then frame. Must contain the `state_col`.
        tolerance_frames: Minimum number of consecutive frames for a state to be kept.
        state_col: Name of the column containing the raw state assignments to be debounced.
        debounced_col: Name of the column to store the debounced state assignments.

    Returns:
        DataFrame with an added column `debounced_col` containing the debounced states.
        Returns a copy of the original DataFrame (potentially with a raw-copied debounced column)
        if input is invalid or debouncing fails critically.
    """
    # --- FIX: Use 'Frame' (uppercase) in required_cols check ---
    required_cols = ['Frame', 'chain', state_col]
    # --- END FIX ---

    if df_states is None or df_states.empty or not all(col in df_states.columns for col in required_cols):
        logger.error(f"Cannot apply debouncing: DataFrame invalid or missing required columns ({required_cols}). Found columns: {list(df_states.columns) if df_states is not None else 'None'}")
        # Return a copy but try to add the debounced_col filled with raw state as fallback
        df_out = df_states.copy() if df_states is not None else pd.DataFrame()
        if state_col in df_out.columns:
            df_out[debounced_col] = df_out[state_col]
        else:
            # If even state_col is missing, add an empty column to avoid downstream errors
            df_out[debounced_col] = pd.NA
        return df_out

    if tolerance_frames <= 1:
        logger.info("Debouncing tolerance <= 1, copying raw states to debounced column.")
        df_out = df_states.copy()
        df_out[debounced_col] = df_out[state_col]
        return df_out

    logger.info(f"Applying RLE debouncing (tolerance={tolerance_frames} frames) to column '{state_col}' -> '{debounced_col}'")
    df_out = df_states.copy()
    all_debounced_states = []
    original_indices = []

    # Use 'Frame' (uppercase) for sorting
    df_sorted = df_out.sort_values(['chain', 'Frame'])

    for chain, chain_data in df_sorted.groupby('chain'):
        if chain_data.empty:
            continue

        raw_states_list = chain_data[state_col].tolist() # Work with list, handles NaN correctly
        try:
             debounced_list = _rle_debounce_single_series(raw_states_list, tolerance_frames)
             # Check length consistency after debouncing for this chain
             if len(debounced_list) != len(raw_states_list):
                 logger.error(f"Debouncing length mismatch for chain {chain}! Input: {len(raw_states_list)}, Output: {len(debounced_list)}. Using raw states for this chain.")
                 all_debounced_states.extend(raw_states_list) # Append original if failed
             else:
                 all_debounced_states.extend(debounced_list) # Append debounced result

        except Exception as e_debounce_chain:
            logger.error(f"Error during debouncing for chain {chain}: {e_debounce_chain}. Using raw states for this chain.", exc_info=True)
            all_debounced_states.extend(raw_states_list) # Append original on error

        original_indices.extend(chain_data.index.tolist())

    # --- Robust Assignment Logic ---
    logger.debug(f"Length Check - all_debounced_states: {len(all_debounced_states)}, df_out rows: {len(df_out)}, original_indices: {len(original_indices)}")

    # Attempt to create the series regardless of length check outcome initially
    try:
        # Ensure indices are unique if groupby caused issues (unlikely but possible)
        original_pd_index = pd.Index(original_indices)
        if original_pd_index.has_duplicates:
             logger.warning(f"Duplicate indices found ({original_pd_index.duplicated().sum()} instances) when creating debounced series. Check groupby logic. Using first value for duplicates.")
             # Create a temporary DataFrame to handle duplicates before making the Series
             temp_df_for_series = pd.DataFrame({'state': all_debounced_states}, index=original_indices)
             temp_df_for_series = temp_df_for_series[~temp_df_for_series.index.duplicated(keep='first')]
             debounced_series = temp_df_for_series['state']
        # Check length before creating series if indices were unique
        elif len(all_debounced_states) != len(df_out):
             logger.error(f"Overall length mismatch after processing all chains (States: {len(all_debounced_states)}, DF Rows: {len(df_out)}). Will attempt assignment but results may be incomplete.")
             # Attempt to create series anyway, might raise error if lengths truly mismatch indices
             # Ensure lengths match for Series creation if possible, otherwise it will raise ValueError
             if len(all_debounced_states) == len(original_indices):
                  debounced_series = pd.Series(all_debounced_states, index=original_indices)
             else:
                  # Cannot create series if lengths differ fundamentally from indices
                  raise ValueError(f"Cannot create pd.Series: Length of states ({len(all_debounced_states)}) does not match length of indices ({len(original_indices)}).")
        else:
             # Lengths match and indices are unique
             debounced_series = pd.Series(all_debounced_states, index=original_indices)

        # Assign the series to the output dataframe using .loc for robust index alignment
        # Create the column first if it doesn't exist, fill with NaN
        if debounced_col not in df_out.columns:
             df_out[debounced_col] = pd.NA
        df_out.loc[debounced_series.index, debounced_col] = debounced_series

        # Check if assignment resulted in unexpected NaNs (could indicate index issues)
        # And ensure the column actually exists after assignment attempt
        if debounced_col not in df_out.columns:
             logger.error(f"CRITICAL: Failed to add debounced column '{debounced_col}' to DataFrame even after assignment attempt.")
             df_out[debounced_col] = df_out[state_col] # Fallback to raw state
        elif df_out[debounced_col].isnull().sum() > df_out[state_col].isnull().sum():
             logger.warning(f"Assignment of debounced series introduced NaNs. Original NaNs: {df_out[state_col].isnull().sum()}, New NaNs: {df_out[debounced_col].isnull().sum()}. Check index alignment.")
             # Optional: Consider filling newly introduced NaNs with raw state?
             # fill_mask = df_out[debounced_col].isna() & df_out[state_col].notna()
             # df_out.loc[fill_mask, debounced_col] = df_out.loc[fill_mask, state_col]

        # Log if the length mismatch error occurred but assignment was still attempted
        if len(all_debounced_states) != len(df_out) and debounced_col in df_out.columns:
            logger.warning(f"Length mismatch condition met, but column '{debounced_col}' was assigned. Results might be partial or contain NaNs.")
        elif debounced_col in df_out.columns:
            logger.debug("Debounced column assigned successfully.")

    except Exception as e_assign:
        logger.error(f"Error creating or assigning debounced series: {e_assign}. Falling back to raw state in column '{debounced_col}'.", exc_info=True)
        # Ensure the column exists, filled with raw data as fallback
        if state_col in df_out.columns:
             df_out[debounced_col] = df_out[state_col]
        else:
             df_out[debounced_col] = pd.NA


    # Restore original sorting
    df_out = df_out.sort_index()

    # Count changes comparing strings for NaN robustness
    # Ensure both columns exist before comparing
    if state_col in df_out and debounced_col in df_out:
        try:
             # Handle potential mixed types by converting to string
             changes = (df_out[state_col].astype(str) != df_out[debounced_col].astype(str)).sum()
             logger.info(f"Debouncing complete. {changes} state entries changed in column '{debounced_col}'.")
        except Exception as e_compare:
             logger.warning(f"Could not compare state changes: {e_compare}")
    else:
        logger.error(f"Could not compare changes: Missing '{state_col}' or '{debounced_col}'.")

    return df_out