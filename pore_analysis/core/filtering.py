# pore_analysis/core/filtering.py
"""
Functions for filtering distance data from MD simulations, including
PBC artifact correction (standard and multi-level) and smoothing.
"""

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde
import logging
from typing import Dict, Any, Optional # Added Optional, Dict, Any

# Get a logger for this module
logger = logging.getLogger(__name__)

# --- Core Filtering Functions ---

def pbc_unwrap_distance(distance_array, threshold=None, data_type='com_distance'):
    """
    PBC unwrapping for distance measurements (first pass) and spike correction.

    Corrects large jumps likely caused by periodic boundary conditions.
    Also includes a simple spike correction step.

    Parameters:
    ----------
    distance_array : numpy.ndarray
        Array of distance measurements over time.
    threshold : float, optional
        Explicit threshold for jump detection (Å). If None, uses defaults
        based on data_type.
    data_type : str, optional
        Type of distance ('gg_distance', 'com_distance', or 'generic') to help
        determine the default threshold if none is provided.

    Returns:
    -------
    unwrapped : numpy.ndarray
        Distance array with PBC jumps and spikes corrected.
    info_dict : dict
        Dictionary with debug and summary information about the filtering steps.
    """
    if distance_array is None or len(distance_array) < 2:
        logger.warning("Insufficient data for PBC unwrapping.")
        return np.array(distance_array), {'jump_count': 0, 'spike_count': 0, 'threshold': threshold}

    # Determine threshold if not provided
    if threshold is None:
        if data_type == 'gg_distance':
            threshold = 2.0
            logger.debug(f"Using default PBC threshold for G-G distance: {threshold} Å")
        elif data_type == 'com_distance':
            threshold = 6.0
            logger.debug(f"Using default PBC threshold for COM distance: {threshold} Å")
        else:
            threshold = 3.0
            logger.debug(f"Using generic default PBC threshold: {threshold} Å")
    else:
         logger.debug(f"Using provided PBC threshold: {threshold} Å")


    info_dict = {}
    original_data = np.array(distance_array, dtype=float) # Ensure float array
    unwrapped = np.zeros_like(original_data)
    unwrapped[0] = original_data[0]
    cumulative_adjustment = 0.0
    jump_count = 0
    jump_locations = []

    # --- PBC Jump Correction ---
    for i in range(1, len(original_data)):
        # Check for NaN or infinite values - propagate them
        if not np.isfinite(original_data[i]) or not np.isfinite(original_data[i-1]):
             unwrapped[i] = original_data[i] # Propagate non-finite value
             continue

        jump = original_data[i] - original_data[i - 1]
        if abs(jump) > threshold:
            # Apply adjustment based on the jump direction
            # Correct based on difference magnitude - should handle positive/negative jumps
            adjustment = -np.sign(jump) * abs(jump)
            # Alternative (might be more robust if threshold represents box/2?):
            # adjustment = -np.sign(jump) * box_dimension_if_known
            cumulative_adjustment += adjustment
            jump_count += 1
            jump_locations.append((i, jump))
            # Reduce verbosity - log only if DEBUG is enabled
            if logger.isEnabledFor(logging.DEBUG):
                 logger.debug(f"PBC jump detected at frame {i}: {jump:.2f} Å. Cumulative adjustment: {cumulative_adjustment:.2f} Å.")
        unwrapped[i] = original_data[i] + cumulative_adjustment

    info_dict['pbc_unwrap_threshold'] = threshold
    info_dict['pbc_jump_count'] = jump_count
    info_dict['pbc_jump_locations'] = jump_locations # Store frame index and jump size
    info_dict['original_mean'] = np.nanmean(original_data)
    info_dict['original_std'] = np.nanstd(original_data)
    info_dict['unwrapped_mean_after_pbc'] = np.nanmean(unwrapped)
    info_dict['unwrapped_std_after_pbc'] = np.nanstd(unwrapped)

    # --- Isolated Spike Correction (using median filtering logic) ---
    # Replace spikes with the median of a local window
    spike_corrected = np.copy(unwrapped) # Work on a copy after unwrapping
    spike_count = 0
    spike_locations = []
    window_size = 7 # Must be odd
    half_window = window_size // 2
    local_std_threshold_factor = 4.0 # How many local std devs define a spike

    for i in range(len(spike_corrected)):
        if not np.isfinite(spike_corrected[i]): continue # Skip non-finite points

        start = max(0, i - half_window)
        end = min(len(spike_corrected), i + half_window + 1)
        local_window_indices = list(range(start, i)) + list(range(i + 1, end)) # Exclude the central point

        if len(local_window_indices) < 3: # Need at least 3 points for robust median/std
             continue

        local_values = spike_corrected[local_window_indices]
        # Filter out NaNs from the local window before calculating median/std
        finite_local_values = local_values[np.isfinite(local_values)]

        if len(finite_local_values) < 3: continue # Still not enough valid points

        local_median = np.median(finite_local_values)
        local_std = np.std(finite_local_values)

        # Avoid division by zero or near-zero std
        if local_std < 1e-6:
             local_std = 1e-6

        # Check if the point deviates significantly from the local median
        if abs(spike_corrected[i] - local_median) > local_std_threshold_factor * local_std:
            old_value = spike_corrected[i]
            spike_corrected[i] = local_median # Replace spike with local median
            spike_count += 1
            spike_locations.append((i, old_value, spike_corrected[i])) # Frame, old value, new value
            # Reduce verbosity
            if logger.isEnabledFor(logging.DEBUG):
                 logger.debug(f"Spike corrected at frame {i}: {old_value:.2f} -> {spike_corrected[i]:.2f} (Local Median: {local_median:.2f}, Std: {local_std:.2f})")

    info_dict['spike_correction_window'] = window_size
    info_dict['spike_correction_threshold_factor'] = local_std_threshold_factor
    info_dict['spike_count'] = spike_count
    info_dict['spike_locations'] = spike_locations # Store frame, old value, new value
    info_dict['final_mean'] = np.nanmean(spike_corrected)
    info_dict['final_std'] = np.nanstd(spike_corrected)

    return spike_corrected, info_dict

def moving_average_smooth(data, window_size=5):
    """
    Apply a centered moving window average (uniform filter) to smooth data.
    Handles NaN values by ignoring them in the window average calculation.

    Parameters:
    ----------
    data : numpy.ndarray
        Data array to smooth.
    window_size : int
        Size of the moving window (will be forced to be odd).

    Returns:
    -------
    smoothed : numpy.ndarray
        Smoothed data array (same size as input).
    info_dict : dict
        Dictionary with smoothing parameters and effect.
    """
    if data is None or len(data) == 0:
        return np.array([]), {'window_size': window_size, 'effect_std_reduction_percent': 0}
    if window_size <= 0:
        logger.warning(f"Invalid window_size ({window_size}) for moving average. Returning original data.")
        return np.array(data), {'window_size': 0, 'effect_std_reduction_percent': 0}

    # Ensure window size is odd for a centered window
    if window_size % 2 == 0:
        window_size += 1
        logger.debug(f"Moving average window size adjusted to be odd: {window_size}")

    original_data = np.array(data, dtype=float) # Ensure float array
    info_dict = {'window_size': window_size}
    pre_smoothing_std = np.nanstd(original_data)
    info_dict['pre_smoothing_std'] = pre_smoothing_std

    # Handle NaNs: Use uniform_filter1d which handles boundaries but not NaNs directly.
    # A common approach is to interpolate NaNs, smooth, then maybe re-insert NaNs,
    # or use a NaN-aware rolling mean (e.g., pandas).
    # Let's use pandas for robust NaN handling if available.
    try:
        import pandas as pd
        # Use centered rolling window, ignore NaNs, use min_periods=1 to get output even at edges/NaN gaps
        smoothed = pd.Series(original_data).rolling(window=window_size, center=True, min_periods=1).mean().to_numpy()
        logger.debug("Used pandas rolling mean for smoothing (NaN-aware).")
    except ImportError:
        logger.warning("Pandas not found. Using scipy.ndimage.uniform_filter1d for smoothing (may not handle NaNs optimally).")
        # Fallback: Scipy's uniform_filter1d (less robust with NaNs)
        # It might be better to interpolate NaNs first if using this fallback.
        # For simplicity here, we apply it directly, which might spread NaN influence.
        smoothed = uniform_filter1d(original_data, size=window_size, mode='nearest')

    post_smoothing_std = np.nanstd(smoothed)
    info_dict['post_smoothing_std'] = post_smoothing_std

    # Calculate smoothing effect (reduction in standard deviation)
    if pre_smoothing_std > 1e-9: # Avoid division by zero
         std_reduction_percent = (pre_smoothing_std - post_smoothing_std) / pre_smoothing_std * 100
    else:
         std_reduction_percent = 0.0
    info_dict['effect_std_reduction_percent'] = std_reduction_percent

    logger.debug(f"Smoothing applied. Window: {window_size}. Std reduction: {std_reduction_percent:.1f}%")

    return smoothed, info_dict

# --- MODIFIED: Added box_size to signature ---
def detect_and_correct_multilevel_pbc(data, min_level_size=30, smoothing_window=5, quality_threshold=0.8, box_size=None):
# --- END MODIFICATION ---
    """
    Detects multiple "levels" in trajectory data potentially caused by PBC artifacts
    and attempts to correct them by normalizing to a reference level.
    Includes quality control to prevent introducing artifacts.

    Parameters:
    -----------
    data : numpy.ndarray
        Raw distance data that might show multiple distinct levels.
    min_level_size : int, optional
        Minimum number of data points required to define a significant level.
    smoothing_window : int, optional
        Window size for final moving average smoothing.
    quality_threshold : float, optional
        Threshold (0-1) for quality control checks. Higher values mean stricter
        checks before applying multi-level correction (more likely to fall back
        to standard filtering).
    box_size : float, optional # <-- MODIFIED: Added box_size
        Estimate of the simulation box size (e.g., Z-dimension) in Angstroms.
        Used to help distinguish PBC jumps from biological transitions. If None,
        it might be estimated from data jumps.

    Returns:
    --------
    corrected_data : numpy.ndarray
        Corrected data, either multi-level adjusted or standard filtered.
    info_dict : dict
        Dictionary containing detailed information about the detection,
        correction process, quality checks, and whether multi-level correction
        was applied.
    """
    info_dict = {'method_attempted': 'multi_level'}
    original_data = np.array(data, dtype=float)

    if len(original_data) < min_level_size * 2: # Need enough data for at least two potential levels
         logger.warning("Insufficient data points for multi-level PBC detection. Applying standard filter.")
         info_dict['multi_level_applied'] = False
         info_dict['fallback_reason'] = "Insufficient data points"
         # --- Pass only smoothing_window to standard_filter ---
         filtered_data, standard_info = standard_filter(original_data, smoothing_window=smoothing_window)
         info_dict.update({'standard_filter_fallback': standard_info})
         return filtered_data, info_dict

    # --- Initial Data Characteristics ---
    finite_data = original_data[np.isfinite(original_data)]
    if len(finite_data) < min_level_size * 2:
        logger.warning("Insufficient finite data points for multi-level PBC detection. Applying standard filter.")
        info_dict['multi_level_applied'] = False
        info_dict['fallback_reason'] = "Insufficient finite data points"
        # --- Pass only smoothing_window to standard_filter ---
        filtered_data, standard_info = standard_filter(original_data, smoothing_window=smoothing_window)
        info_dict.update({'standard_filter_fallback': standard_info})
        return filtered_data, info_dict

    original_std = np.std(finite_data)
    original_range = np.ptp(finite_data) # Peak-to-peak range
    info_dict['original_min'] = np.min(finite_data)
    info_dict['original_max'] = np.max(finite_data)
    info_dict['original_range'] = original_range
    info_dict['original_std'] = original_std
    logger.debug(f"Multi-level check: Range={original_range:.2f}, Std={original_std:.2f}")

    # --- Estimate Box Size if not Provided ---
    estimated_box_size = None
    if box_size is None:
        # Estimate from large jumps in the data (potential PBC wrap-around)
        diffs = np.abs(np.diff(finite_data))
        # Increased jump threshold slightly for estimation robustness
        large_jumps = diffs[diffs > 8.0]
        if len(large_jumps) > 5: # Require a few large jumps for a reasonable estimate
             potential_box_size = np.median(large_jumps)
             # Sanity check: typical box sizes are ~40-150Å
             if 40 < potential_box_size < 150:
                 estimated_box_size = potential_box_size
                 info_dict['estimated_box_size'] = estimated_box_size
                 logger.info(f"Estimated box size from data jumps: {estimated_box_size:.2f} Å")
             else:
                  logger.debug(f"Potential box size estimate ({potential_box_size:.2f} Å) outside typical range (40-150 Å).")
        else:
            logger.debug("Not enough large jumps (>8Å) to estimate box size reliably.")
    else:
        info_dict['provided_box_size'] = box_size
        logger.debug(f"Using provided box size: {box_size:.2f} Å")
    # Use the estimated or provided box size
    effective_box_size = box_size if box_size is not None else estimated_box_size

    # --- KDE Level Detection ---
    try:
        # Use Silverman's rule for bandwidth, can be sensitive
        kde = gaussian_kde(finite_data, bw_method='silverman')
        x_grid = np.linspace(np.min(finite_data), np.max(finite_data), 500) # Use fewer points for grid
        density = kde(x_grid)
    except Exception as kde_err:
         logger.warning(f"KDE calculation failed: {kde_err}. Applying standard filter.")
         info_dict['multi_level_applied'] = False
         info_dict['fallback_reason'] = f"KDE calculation error: {kde_err}"
         # --- Pass only smoothing_window to standard_filter ---
         filtered_data, standard_info = standard_filter(original_data, smoothing_window=smoothing_window)
         info_dict.update({'standard_filter_fallback': standard_info})
         return filtered_data, info_dict

    # Enhanced peak detection parameters
    min_peak_height = np.max(density) * 0.05 # Peak must be at least 5% of max density
    min_peak_distance = max(10, int(len(x_grid) / 50)) # Dynamic distance based on grid size
    min_peak_prominence = np.max(density) * 0.02 # Peak must stand out by 2% of max density

    peaks, peak_properties = find_peaks(density,
                                        height=min_peak_height,
                                        distance=min_peak_distance,
                                        prominence=min_peak_prominence)

    level_values = x_grid[peaks]
    peak_densities = density[peaks]
    # Sort detected levels by their position (value)
    sorted_indices = np.argsort(level_values)
    level_values_sorted = level_values[sorted_indices]
    peak_densities_sorted = peak_densities[sorted_indices]

    info_dict['kde_detected_level_count'] = len(peaks)
    info_dict['kde_level_values'] = level_values_sorted.tolist()
    info_dict['kde_peak_densities'] = peak_densities_sorted.tolist()
    info_dict['kde_peak_properties'] = {key: val.tolist() for key, val in peak_properties.items()}

    if len(peaks) <= 1:
        logger.info("KDE detected 0 or 1 level. Applying standard filter.")
        info_dict['multi_level_applied'] = False
        info_dict['fallback_reason'] = "KDE detected <= 1 level"
        # --- Pass only smoothing_window to standard_filter ---
        filtered_data, standard_info = standard_filter(original_data, smoothing_window=smoothing_window)
        info_dict.update({'standard_filter_fallback': standard_info})
        return filtered_data, info_dict

    # --- Assign Data Points to Levels ---
    # Define boundaries midway between sorted level peaks
    level_boundaries = []
    for i in range(len(level_values_sorted) - 1):
        boundary = (level_values_sorted[i] + level_values_sorted[i+1]) / 2
        level_boundaries.append(boundary)
    # Add outer bounds (extend +/- infinity conceptually)
    full_boundaries = [-np.inf] + level_boundaries + [np.inf]

    # Assign each original data point to a level index based on boundaries
    level_assignments = np.zeros(len(original_data), dtype=int)
    for i, point in enumerate(original_data):
         if np.isfinite(point):
             # Find which bin the point falls into using searchsorted
             level_idx = np.searchsorted(level_boundaries, point)
             level_assignments[i] = level_idx
         else:
             level_assignments[i] = -1 # Assign NaN/Inf to level -1

    # Count points in each level
    unique_levels, level_counts = np.unique(level_assignments[level_assignments != -1], return_counts=True)
    level_info = {}
    valid_level_indices = [] # Indices corresponding to level_values_sorted
    for level_idx, count in zip(unique_levels, level_counts):
         level_val = level_values_sorted[level_idx]
         level_info[level_idx] = {'value': level_val, 'count': count}
         if count >= min_level_size:
             valid_level_indices.append(level_idx)

    info_dict['level_assignment_info'] = level_info
    info_dict['valid_level_indices'] = valid_level_indices
    info_dict['valid_level_count'] = len(valid_level_indices)

    if len(valid_level_indices) <= 1:
        logger.info(f"Only {len(valid_level_indices)} level(s) met the minimum size ({min_level_size}). Applying standard filter.")
        info_dict['multi_level_applied'] = False
        info_dict['fallback_reason'] = "Only <= 1 level met min_level_size"
        # --- Pass only smoothing_window to standard_filter ---
        filtered_data, standard_info = standard_filter(original_data, smoothing_window=smoothing_window)
        info_dict.update({'standard_filter_fallback': standard_info})
        return filtered_data, info_dict

    # --- Analyze Transitions between Valid Levels ---
    transitions = [] # Stores (frame_index, from_level_idx, to_level_idx)
    valid_level_set = set(valid_level_indices)

    for i in range(1, len(level_assignments)):
         prev_level = level_assignments[i - 1]
         curr_level = level_assignments[i]
         # Record transition only if both levels are valid and different
         if prev_level != curr_level and prev_level in valid_level_set and curr_level in valid_level_set:
             transitions.append((i, prev_level, curr_level))

    info_dict['transition_count'] = len(transitions)

    # Check if transitions look like PBC jumps vs. biological motion
    is_likely_pbc = False
    if transitions:
        level_diffs = []
        transition_durations = [] # How many frames between transitions
        for i in range(len(transitions)):
             frame_idx, from_level_idx, to_level_idx = transitions[i]
             level_diff = abs(level_values_sorted[from_level_idx] - level_values_sorted[to_level_idx])
             level_diffs.append(level_diff)
             if i > 0:
                 prev_frame_idx = transitions[i-1][0]
                 transition_durations.append(frame_idx - prev_frame_idx)

        median_level_diff = np.median(level_diffs)
        info_dict['median_transition_level_diff'] = median_level_diff

        # Check 1: Are level differences consistent with box size?
        if effective_box_size is not None:
             diff_from_box = abs(median_level_diff - effective_box_size)
             relative_diff = diff_from_box / effective_box_size
             info_dict['median_level_diff_rel_box_diff'] = relative_diff
             if relative_diff < 0.25: # If median jump is within 25% of box size
                 is_likely_pbc = True
                 logger.info(f"Transitions consistent with box size (Median diff: {median_level_diff:.2f} Å, Box: {effective_box_size:.2f} Å). Likely PBC.")
             else:
                  logger.info(f"Transitions not clearly related to box size (Median diff: {median_level_diff:.2f} Å, Box: {effective_box_size:.2f} Å).")

        # Check 2: Are transitions very rapid (few frames)?
        if not is_likely_pbc and transition_durations:
             median_duration = np.median(transition_durations)
             info_dict['median_frames_between_transitions'] = median_duration
             if median_duration < 50:
                 logger.warning(f"Transitions occur frequently (median duration: {median_duration} frames). Could be PBC or unstable system.")
             else:
                 logger.info(f"Transitions occur less frequently (median duration: {median_duration} frames). Less likely to be solely PBC artifacts.")

        # Check 3: Did standard filter already reduce std significantly?
        temp_filtered, temp_info = standard_filter(original_data)
        std_reduction_standard = temp_info.get('overall_effect_std_reduction_percent', 0)
        info_dict['standard_filter_potential_std_reduction'] = std_reduction_standard
        if std_reduction_standard > 50:
             logger.info(f"Standard filter already achieved significant std reduction ({std_reduction_standard:.1f}%). Considering fallback.")
             quality_threshold = max(0.1, quality_threshold * 0.5)
    else: # No transitions between *valid* levels detected
        logger.info("No transitions detected between valid levels. Applying standard filter.")
        info_dict['multi_level_applied'] = False
        info_dict['fallback_reason'] = "No transitions between valid levels"
        # --- Pass only smoothing_window to standard_filter ---
        filtered_data, standard_info = standard_filter(original_data, smoothing_window=smoothing_window)
        info_dict.update({'standard_filter_fallback': standard_info})
        return filtered_data, info_dict

    # --- Apply Multi-Level Correction ---
    reference_level_idx = max(valid_level_indices, key=lambda idx: level_info[idx]['count'])
    reference_value = level_values_sorted[reference_level_idx]
    logger.info(f"Using level {reference_level_idx} (Value: {reference_value:.2f} Å, Count: {level_info[reference_level_idx]['count']}) as reference.")
    info_dict['reference_level_index'] = reference_level_idx
    info_dict['reference_level_value'] = reference_value

    corrected_data = np.copy(original_data)
    shifts_applied = {}

    for level_idx in valid_level_indices:
        if level_idx == reference_level_idx: continue
        level_value = level_values_sorted[level_idx]
        shift = level_value - reference_value
        shifts_applied[level_idx] = -shift
        mask = (level_assignments == level_idx)
        corrected_data[mask] -= shift
        logger.debug(f"Applied shift of {-shift:.2f} Å to level {level_idx} (Value: {level_value:.2f} Å).")

    info_dict['level_shifts_applied'] = shifts_applied

    # --- Final Smoothing ---
    smoothed_data, smooth_info = moving_average_smooth(corrected_data, window_size=smoothing_window)
    info_dict.update({f"final_smooth_{k}": v for k, v in smooth_info.items()})

    # --- Quality Control ---
    final_std = np.nanstd(smoothed_data)
    info_dict['final_std'] = final_std
    std_change_percent = (original_std - final_std) / original_std * 100 if original_std > 1e-6 else 0
    info_dict['final_std_reduction_percent'] = std_change_percent

    quality_passed = True
    quality_issues = []

    if std_change_percent < -15:
         quality_issues.append(f"StdDev increased significantly ({std_change_percent:.1f}%)")
         quality_passed = False

    original_mean = np.nanmean(finite_data)
    final_mean = np.nanmean(smoothed_data[np.isfinite(smoothed_data)])
    mean_shift_percent = abs(final_mean - original_mean) / abs(original_mean) * 100 if abs(original_mean) > 1e-6 else 0
    info_dict['final_mean_shift_percent'] = mean_shift_percent
    if mean_shift_percent > 15:
         quality_issues.append(f"Mean shifted significantly ({mean_shift_percent:.1f}%)")
         quality_passed = False

    max_diff = np.nanmax(np.abs(np.diff(smoothed_data))) if len(smoothed_data) > 1 else 0
    info_dict['max_diff_in_corrected'] = max_diff
    max_reasonable_jump = max(2.0, original_std * 3)
    if max_diff > max_reasonable_jump:
         quality_issues.append(f"Large jump ({max_diff:.2f} Å) detected in corrected data (Threshold: {max_reasonable_jump:.2f} Å)")
         quality_passed = False

    info_dict['quality_control_passed'] = quality_passed
    info_dict['quality_control_issues'] = quality_issues

    # --- Decide whether to use multi-level corrected data or fallback ---
    should_fallback = False
    if not quality_passed:
         if np.random.random() > (1.0 - quality_threshold):
             should_fallback = True
             info_dict['fallback_reason'] = f"Quality issues detected ({', '.join(quality_issues)}) and random check failed quality threshold ({quality_threshold:.2f})."
             logger.warning(f"Multi-level correction failed quality checks: {quality_issues}. Falling back to standard filter.")
         else:
             logger.warning(f"Multi-level correction had quality issues ({quality_issues}), but passed random check (threshold {quality_threshold:.2f}). Using multi-level result cautiously.")
             info_dict['quality_control_override'] = "Passed random check despite issues."
    elif not is_likely_pbc:
         logger.info("Transitions did not strongly resemble PBC jumps. Considering standard filter fallback.")
         if std_reduction_standard > 30:
              should_fallback = True
              info_dict['fallback_reason'] = f"Transitions not clearly PBC and standard filter was effective (Std Reduc: {std_reduction_standard:.1f}%)."
              logger.info("Falling back to standard filter as transitions were not clearly PBC and standard filter was effective.")
         else:
              logger.info("Proceeding with multi-level correction despite transitions not clearly being PBC.")

    if should_fallback:
        info_dict['multi_level_applied'] = False
        # --- Pass only smoothing_window to standard_filter ---
        filtered_data, standard_info = standard_filter(original_data, smoothing_window=smoothing_window)
        info_dict.update({'standard_filter_fallback': standard_info})
        info_dict['final_mean'] = np.nanmean(filtered_data)
        info_dict['final_std'] = np.nanstd(filtered_data)
        if original_std > 1e-6: info_dict['final_std_reduction_percent'] = (original_std - info_dict['final_std']) / original_std * 100
        else: info_dict['final_std_reduction_percent'] = 0.0
        return filtered_data, info_dict
    else:
        info_dict['multi_level_applied'] = True
        logger.info(f"Multi-level correction applied. Final std reduction: {std_change_percent:.1f}%")
        return smoothed_data, info_dict

# --- Combined Filtering Functions ---

def standard_filter(data, data_type='com_distance', smoothing_window=5):
    """
    Applies standard two-pass filtering: PBC unwrapping followed by moving average.

    Parameters:
    ----------
    data : numpy.ndarray
        Input distance data array.
    data_type : str, optional
        Type of data ('gg_distance', 'com_distance', 'generic') passed to
        pbc_unwrap_distance for threshold determination.
    smoothing_window : int, optional
        Window size for the moving average smoothing step.

    Returns:
    -------
    smoothed_data : numpy.ndarray
        Data after unwrapping and smoothing.
    info_dict : dict
        Combined information dictionary from unwrapping and smoothing steps.
    """
    info = {'method_applied': 'standard'}
    original_mean = np.nanmean(data)
    original_std = np.nanstd(data)
    info['original_mean'] = original_mean
    info['original_std'] = original_std

    # 1. PBC Unwrap + Spike Correction
    unwrapped_data, unwrap_info = pbc_unwrap_distance(data, data_type=data_type)
    info.update({f"unwrap_{k}": v for k, v in unwrap_info.items()}) # Prefix keys

    # 2. Moving Average Smoothing
    smoothed_data, smooth_info = moving_average_smooth(unwrapped_data, window_size=smoothing_window)
    info.update({f"smooth_{k}": v for k, v in smooth_info.items()}) # Prefix keys

    # Overall effect
    filtered_mean = np.nanmean(smoothed_data)
    filtered_std = np.nanstd(smoothed_data)
    info['filtered_mean'] = filtered_mean
    info['filtered_std'] = filtered_std
    if original_std > 1e-9:
         info['overall_effect_std_reduction_percent'] = (original_std - filtered_std) / original_std * 100
    else:
         info['overall_effect_std_reduction_percent'] = 0.0

    logger.info(f"Standard filter applied. Overall std reduction: {info['overall_effect_std_reduction_percent']:.1f}%")
    return smoothed_data, info

# --- MODIFIED: auto_select_filter to handle kwargs ---
def auto_select_filter(
    data: np.ndarray,
    data_type: str = 'com_distance',
    std_threshold: float = 1.5,
    range_threshold: float = 5.0,
    **kwargs: Any
) -> tuple[np.ndarray, Dict[str, Any]]:
    """
    Automatically selects and applies the appropriate filter (standard or multi-level)
    based on data characteristics (standard deviation, range).

    Parameters:
    ----------
    data : numpy.ndarray
        Input distance data array.
    data_type : str, optional
        Type of data ('gg_distance', 'com_distance', 'generic'). G-G distance
        currently always uses standard filtering.
    std_threshold : float, optional
        Standard deviation threshold above which multi-level filtering might be triggered.
    range_threshold : float, optional
        Range (max-min) threshold above which multi-level filtering might be triggered.
    **kwargs : dict
        Additional keyword arguments. Only relevant arguments are passed to the
        selected filtering function (e.g., `box_size`, `smoothing_window`).

    Returns:
    -------
    filtered_data : numpy.ndarray
        The filtered data array.
    info_dict : dict
        Information dictionary from the applied filtering function.
    """
    logger.info(f"Auto-selecting filter for data_type: {data_type}")
    if data is None or len(data) < 2:
        logger.warning("Insufficient data for filtering. Returning original data.")
        return np.array(data), {'method_applied': 'none', 'reason': 'Insufficient data'}

    finite_data = data[np.isfinite(data)]
    if len(finite_data) < 2:
         logger.warning("Insufficient finite data points for filtering. Returning original data.")
         return np.array(data), {'method_applied': 'none', 'reason': 'Insufficient finite data'}

    std_dev = np.std(finite_data)
    data_range = np.ptp(finite_data)

    logger.debug(f"Data characteristics: Std Dev={std_dev:.3f}, Range={data_range:.3f}")
    logger.debug(f"Thresholds: Std Dev={std_threshold:.3f}, Range={range_threshold:.3f}")

    # Define arguments accepted by each filter function
    standard_filter_args = {'smoothing_window'}
    multi_level_filter_args = {'min_level_size', 'smoothing_window', 'quality_threshold', 'box_size'}

    # --- Decision Logic ---
    if data_type == 'gg_distance':
        logger.info("Applying standard filtering for G-G distance.")
        # Filter kwargs for standard_filter
        standard_kwargs = {k: v for k, v in kwargs.items() if k in standard_filter_args}
        return standard_filter(data, data_type=data_type, **standard_kwargs)
    else: # For COM distance or generic
        if std_dev > std_threshold or data_range > range_threshold:
            logger.info(f"High std dev ({std_dev:.3f}) or range ({data_range:.3f}) detected. Attempting multi-level filtering.")
            # Filter kwargs for detect_and_correct_multilevel_pbc
            # --- FIX: Correctly handle box_z -> box_size ---
            # The multilevel function expects 'box_size'. If 'box_z' is passed in kwargs,
            # map it to 'box_size'.
            multi_level_kwargs = {}
            if 'box_z' in kwargs and kwargs['box_z'] is not None:
                 multi_level_kwargs['box_size'] = kwargs['box_z']
                 logger.debug("Mapping 'box_z' keyword arg to 'box_size' for multi-level filter.")
            # Include other relevant kwargs if they exist
            for key in multi_level_filter_args:
                 if key in kwargs and key != 'box_size': # Avoid overwriting mapped box_size
                      multi_level_kwargs[key] = kwargs[key]
            # --- END FIX ---
            return detect_and_correct_multilevel_pbc(data, **multi_level_kwargs)
        else:
            logger.info(f"Low std dev ({std_dev:.3f}) and range ({data_range:.3f}). Applying standard filtering.")
            # Filter kwargs for standard_filter
            standard_kwargs = {k: v for k, v in kwargs.items() if k in standard_filter_args}
            return standard_filter(data, data_type=data_type, **standard_kwargs)
# --- END MODIFICATION ---
