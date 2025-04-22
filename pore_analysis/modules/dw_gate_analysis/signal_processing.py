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
# Matplotlib import will be needed for determine_reference_distances if plotting is enabled
# import matplotlib.pyplot as plt
# from matplotlib.lines import Line2D # For custom legends
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
# KMeans import will be needed for determine_reference_distances
# from sklearn.cluster import KMeans

# Assuming utils is a sibling module for constants and plotting
from .utils import save_plot, CLOSED_STATE, OPEN_STATE # save_plot needed later

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
        kde = gaussian_kde(distances, bw_method=bw_method)
        # Define x range based on data, ensuring it's reasonable
        x_min = max(0, np.min(distances) - 1.0) if len(distances) > 0 else 0
        x_max = np.max(distances) + 1.0 if len(distances) > 0 else 5
        x_kde = np.linspace(x_min, x_max, 500)
        y_kde = kde(x_kde)

        max_height = y_kde.max()
        if max_height <= 0:
             logger.warning("KDE maximum height is zero or negative, cannot find peaks.")
             # Return KDE curve even if no peaks found
             return x_kde, y_kde, np.array([]), np.array([])

        peaks_indices, _ = find_peaks(
            y_kde,
            height=peak_height_fraction * max_height,
            distance=peak_distance,
            prominence=peak_prominence
        )

        if len(peaks_indices) == 0:
            logger.debug("No peaks found meeting the criteria.")
             # Return KDE curve even if no peaks found
            return x_kde, y_kde, np.array([]), np.array([])

        peak_distances = x_kde[peaks_indices]
        peak_heights = y_kde[peaks_indices]
        logger.debug(f"Found {len(peak_distances)} KDE peaks at distances: {np.round(peak_distances, 2)}")
        return x_kde, y_kde, peak_distances, peak_heights

    except Exception as e:
        logger.error(f"KDE calculation or peak finding failed: {e}", exc_info=True)
        return None, None, None, None

def determine_reference_distances(
    df_raw: pd.DataFrame,
    default_closed_ref: float,
    default_open_ref: float,
    distance_threshold: float, # For plotting the initial threshold line
    output_dir: str,
    kde_bw_method: Optional[str] = 'scott',
    kde_peak_height_fraction: float = 0.05,
    kde_peak_distance: int = 30,
    kde_peak_prominence: float = 0.05,
    plot_distributions: bool = True
) -> Tuple[float, float, Optional[str]]:
    """
    Determines closed and open reference distances using KDE on combined and per-chain data,
    followed by KMeans clustering on detected peaks.
    Falls back to defaults if KDE/KMeans fails.
    Optionally plots the distance distributions.

    Args:
        df_raw: DataFrame with raw distances (must contain 'distance' and 'chain' columns).
        default_closed_ref: Default reference distance for the closed state.
        default_open_ref: Default reference distance for the open state.
        distance_threshold: Initial distance threshold (used for plotting).
        output_dir: Directory to save the plot.
        kde_bw_method: Bandwidth method for KDE.
        kde_peak_height_fraction: Peak height fraction for KDE peak finding.
        kde_peak_distance: Peak distance for KDE peak finding.
        kde_peak_prominence: Peak prominence for KDE peak finding.
        plot_distributions: Whether to generate and save the distribution plot.

    Returns:
        Tuple containing:
        - final_closed_ref: The determined or default closed reference distance.
        - final_open_ref: The determined or default open reference distance.
        - plot_rel_path: Relative path to the saved plot, or None.
    """
    # Ensure necessary imports for plotting and clustering are available
    if plot_distributions:
        try:
            import matplotlib.pyplot as plt
            from matplotlib.lines import Line2D
        except ImportError:
            logger.warning("Matplotlib not found, cannot generate distribution plot.")
            plot_distributions = False # Disable plotting if import fails

    try:
        from sklearn.cluster import KMeans
    except ImportError:
        logger.error("Scikit-learn not found. Cannot perform KMeans clustering for reference distances.")
        # Force using defaults if KMeans is unavailable
        logger.warning(f"Using default reference distances: Closed={default_closed_ref:.2f}, Open={default_open_ref:.2f}")
        return default_closed_ref, default_open_ref, None

    if df_raw is None or df_raw.empty or 'distance' not in df_raw.columns or 'chain' not in df_raw.columns:
        logger.warning("Raw distance data invalid or missing. Cannot calculate reference distances. Using defaults.")
        return default_closed_ref, default_open_ref, None

    all_distances = df_raw['distance'].dropna().values
    if len(all_distances) == 0:
        logger.warning("No valid distance values found. Using defaults.")
        return default_closed_ref, default_open_ref, None

    logger.info("Attempting to determine reference distances from data using KDE and KMeans...")

    # --- Per-chain analysis to collect peaks --- 
    all_chain_peaks = []
    per_chain_results = {}
    unique_chains = sorted(df_raw['chain'].unique())
    for chain in unique_chains:
        chain_dist = df_raw.loc[df_raw['chain'] == chain, 'distance'].dropna().values
        x_kde, y_kde, peaks, heights = find_kde_peaks(
            chain_dist, kde_bw_method, kde_peak_height_fraction, kde_peak_distance, kde_peak_prominence
        )
        if peaks is not None and len(peaks) > 0:
            all_chain_peaks.extend(peaks)
        # Store results even if no peaks found, for potential plotting
        per_chain_results[chain] = {'x': x_kde, 'y': y_kde, 'peaks': peaks, 'heights': heights}

    # --- Combined analysis --- 
    x_all, y_all, peaks_all, heights_all = find_kde_peaks(
         all_distances, kde_bw_method, kde_peak_height_fraction, kde_peak_distance, kde_peak_prominence
    )
    # Store combined results
    combined_results = {'x': x_all, 'y': y_all, 'peaks': peaks_all, 'heights': heights_all}

    # --- Determine References via KMeans --- 
    final_closed_ref = default_closed_ref
    final_open_ref = default_open_ref
    kmeans_centers = None

    if all_chain_peaks:
        unique_peaks = sorted(list(set(all_chain_peaks)))
        if len(unique_peaks) >= 2:
            try:
                kp = np.array(unique_peaks).reshape(-1, 1)
                kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(kp)
                centers = sorted(kmeans.cluster_centers_.flatten())
                final_closed_ref, final_open_ref = centers[0], centers[1]
                kmeans_centers = centers # For plotting
                logger.info(f"KDE/KMeans derived references: Closed={final_closed_ref:.2f} Å, Open={final_open_ref:.2f} Å")
            except ValueError as ve:
                 # Handle cases where KMeans might fail (e.g., insufficient clusters)
                 logger.warning(f"KMeans clustering failed ({ve}). Using defaults ({default_closed_ref:.2f}, {default_open_ref:.2f}).")
            except Exception as e:
                logger.warning(f"KMeans clustering on peaks failed unexpectedly: {e}. Using defaults ({default_closed_ref:.2f}, {default_open_ref:.2f}).")
        elif len(unique_peaks) == 1:
            logger.warning(f"Only one unique KDE peak found ({unique_peaks[0]:.2f} Å). Insufficient for clustering. Using defaults ({default_closed_ref:.2f}, {default_open_ref:.2f}).")
        else: # len is 0
            logger.warning(f"No KDE peaks found across any chain. Using defaults ({default_closed_ref:.2f}, {default_open_ref:.2f}).")
    else:
        logger.warning(f"No KDE peaks collected from per-chain analysis. Using defaults ({default_closed_ref:.2f}, {default_open_ref:.2f}).")

    # --- Plotting (Optional) --- 
    plot_rel_path = None
    if plot_distributions:
        fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
        # Create the main combined axis spanning the top row
        ax_combined = fig.add_subplot(3, 1, 1) # 3 rows, 1 col, 1st plot
        # Remove the original axes from the grid that ax_combined now occupies
        fig.delaxes(axes[0,0])
        fig.delaxes(axes[0,1])
        # Use the remaining axes for per-chain plots
        chain_axes = axes[1:,:].flatten()

        # Determine common x-range for all subplots
        x_min_all = np.floor(all_distances.min() - 0.5) if len(all_distances) > 0 else 0
        x_max_all = np.ceil(all_distances.max() + 0.5) if len(all_distances) > 0 else 5

        # Plot Combined Histogram and KDE
        combined_handles, combined_labels = [], []
        if combined_results['x'] is not None:
            # Histogram
            hist_vals, hist_bins = np.histogram(all_distances, bins=50, density=True, range=(x_min_all, x_max_all))
            bin_centers = (hist_bins[:-1] + hist_bins[1:]) / 2
            bar_container = ax_combined.bar(bin_centers, hist_vals, width=np.diff(hist_bins)[0]*0.9, alpha=0.5, label='_Combined Histogram')
            combined_handles.append(bar_container[0])
            combined_labels.append('Combined Histogram')

            # Combined KDE
            line_kde, = ax_combined.plot(combined_results['x'], combined_results['y'], label='_Combined KDE', color='k', lw=1.5)
            combined_handles.append(line_kde)
            combined_labels.append('Combined KDE')

            # Combined Peaks
            if combined_results['peaks'] is not None and len(combined_results['peaks']) > 0:
                scatter_peaks = ax_combined.scatter(combined_results['peaks'], combined_results['heights'], color='red', marker='x', s=80, label='_Combined Peaks')
                combined_handles.append(scatter_peaks)
                combined_labels.append('Combined Peaks')

            # KMeans Centers
            if kmeans_centers is not None:
                 # Ensure centers are plotted within y-limits if possible
                 y_max_plot = ax_combined.get_ylim()[1]
                 scatter_kmeans = ax_combined.scatter(kmeans_centers, [y_max_plot * 0.95] * 2, color='blue', marker='D', s=100, label='_KMeans Centers', zorder=5)
                 combined_handles.append(scatter_kmeans)
                 combined_labels.append('KMeans Centers')

            # Reference Lines
            line_thresh = ax_combined.axvline(distance_threshold, color='grey', ls=':', lw=1.5, label=f'_Threshold ({distance_threshold:.1f} Å)')
            combined_handles.append(line_thresh)
            combined_labels.append(f'Threshold ({distance_threshold:.1f} Å)')
            line_closed = ax_combined.axvline(final_closed_ref, color='darkred', ls='-', lw=2, label=f'_Closed Ref ({final_closed_ref:.2f} Å)')
            combined_handles.append(line_closed)
            combined_labels.append(f'Closed Ref ({final_closed_ref:.2f} Å)')
            line_open = ax_combined.axvline(final_open_ref, color='darkgreen', ls='-', lw=2, label=f'_Open Ref ({final_open_ref:.2f} Å)')
            combined_handles.append(line_open)
            combined_labels.append(f'Open Ref ({final_open_ref:.2f} Å)')

            ax_combined.set_title('Combined & Per-Chain DW-Gate Distance Distributions (KDE)')
            ax_combined.set_ylabel('Density')
            ax_combined.grid(False)
            ax_combined.set_xlim(x_min_all, x_max_all)
            # ax_combined.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False) # Hide x-ticks for top plot

        else:
            ax_combined.text(0.5, 0.5, 'Combined KDE Failed', ha='center', va='center', transform=ax_combined.transAxes, color='red')
            ax_combined.set_title('Combined & Per-Chain DW-Gate Distance Distributions (KDE)')
            ax_combined.grid(False)
            ax_combined.set_xlim(x_min_all, x_max_all)
            # ax_combined.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

        # Plot Per-Chain KDEs (max 4 chains)
        per_chain_handles, per_chain_labels = [], []
        chains_plotted = 0
        for i, chain in enumerate(unique_chains):
            if chains_plotted >= len(chain_axes): break
            ax_chain = chain_axes[chains_plotted]
            chain_data = per_chain_results.get(chain, {})
            x_kde, y_kde = chain_data.get('x'), chain_data.get('y')
            peaks, heights = chain_data.get('peaks'), chain_data.get('heights')

            if x_kde is not None and y_kde is not None:
                line_pc, = ax_chain.plot(x_kde, y_kde, label=f'_Chain {chain} KDE')
                if chains_plotted == 0: # Add handle only once
                    per_chain_handles.append(line_pc)
                    per_chain_labels.append('Per-Chain KDE')

                if peaks is not None and len(peaks) > 0:
                    scatter_pc = ax_chain.scatter(peaks, heights, color='red', marker='x', s=50, label='_Peaks')
                    if chains_plotted == 0: # Add handle only once
                         # Need a proxy artist for scatter legend item
                        per_chain_handles.append(Line2D([0], [0], marker='x', color='red', markersize=8, linestyle='None'))
                        per_chain_labels.append('Per-Chain Peaks')
                    # Annotate peaks
                    for pd_val, ph_val in zip(peaks, heights):
                        ax_chain.text(pd_val + 0.02 * (x_max_all - x_min_all), ph_val, f'{pd_val:.1f}Å', fontsize=8, va='bottom', ha='left')

                ax_chain.set_title(f'Chain {chain}', fontsize=10)
                ax_chain.grid(False)
                # ax_chain.set_xlim(x_min_all, x_max_all) # Already shared x-axis
            else:
                # Handle case where chain had insufficient data or KDE failed
                chain_dist_len = len(df_raw.loc[df_raw['chain'] == chain, 'distance'].dropna().values)
                ax_chain.text(0.5, 0.5, f"Chain {chain}\nNo KDE (N={chain_dist_len})", ha='center', va='center', transform=ax_chain.transAxes, fontsize=9)
                ax_chain.set_title(f'Chain {chain}', fontsize=10)
                ax_chain.grid(False)
                # ax_chain.set_xlim(x_min_all, x_max_all)

            # Add labels only to outer axes
            if chains_plotted % 2 == 0: # Left column
                ax_chain.set_ylabel('Density', fontsize=9)
            # Set xlabel only for the last row of chain plots
            if chains_plotted >= len(chain_axes) - 2:
                ax_chain.set_xlabel('Distance (Å)', fontsize=9)

            # Adjust tick label size
            ax_chain.tick_params(axis='both', which='major', labelsize=8)

            chains_plotted += 1

        # Add Combined Legend below all subplots
        all_handles = combined_handles + per_chain_handles
        all_labels = combined_labels + per_chain_labels
        fig.legend(all_handles, all_labels, loc='lower center',
                   bbox_to_anchor=(0.5, 0.01), # Position slightly above bottom edge
                   ncol=min(4, len(all_labels)), fontsize='small') # Adjust columns based on items

        # Improve spacing
        fig.tight_layout(rect=[0, 0.06, 1, 0.96]) # Leave space for legend and title

        # Save Plot
        plot_filename = "dw_gate_distance_distribution.png"
        # Ensure output directory exists (save_plot also does this, but good practice)
        os.makedirs(output_dir, exist_ok=True)
        plot_full_path = os.path.join(output_dir, plot_filename)
        save_plot(fig, plot_full_path) # Use the utility function
        # Return relative path for summary/reporting (assuming output_dir is inside the main run dir)
        dw_gate_subdir = os.path.basename(output_dir) # Get the immediate subdir name (e.g., dw_gate_analysis)
        plot_rel_path = os.path.join(dw_gate_subdir, plot_filename)
        logger.info(f"Saved distance distribution plot: {plot_full_path}")
    else:
        logger.info("Plotting of distance distributions is disabled.")


    return final_closed_ref, final_open_ref, plot_rel_path

def _rle_debounce_single_series(states: List[int], tolerance_frames: int) -> List[int]:
    """
    Internal RLE debouncing logic for a single list of numeric states.
    Merges runs shorter than tolerance into the longer neighboring run.

    Args:
        states: List of numeric states (e.g., 0 for closed, 1 for open).
        tolerance_frames: Minimum run length to keep.

    Returns:
        Debounced list of numeric states.
    """
    if not states:
        return []

    if tolerance_frames <= 1:
        logger.debug("Debouncing tolerance <= 1, skipping RLE.")
        return states

    s = list(states) # Work on a copy
    n = len(s)
    iteration = 0
    max_iterations = n * 2 # Safety break, allow more iterations than length for complex cases

    while iteration < max_iterations:
        iteration += 1
        runs = []
        if not s: break # Should not happen normally

        # 1. Identify runs
        current_val = s[0]
        start_idx = 0
        for i in range(1, n):
            if s[i] != current_val:
                runs.append({'value': current_val, 'start': start_idx, 'length': i - start_idx})
                current_val = s[i]
                start_idx = i
        runs.append({'value': current_val, 'start': start_idx, 'length': n - start_idx})

        # 2. Find short runs
        merges_made_in_pass = 0
        indices_to_merge = [i for i, run in enumerate(runs) if run['length'] < tolerance_frames]

        if not indices_to_merge:
            logger.debug(f"RLE converged after {iteration-1} passes.")
            break # Debouncing complete

        # 3. Merge short runs (backward iteration to avoid index shifts)
        for idx in sorted(indices_to_merge, reverse=True):
            # Re-check run length in case previous merges changed it
            # This requires re-calculating runs or carefully updating lengths, simpler to just apply merge based on original detection
            current_run = runs[idx]
            val_to_merge = current_run['value']
            start = current_run['start']
            length = current_run['length']

            # Determine the value to merge into (neighbor's value)
            new_val = -1 # Placeholder for invalid state or no merge needed
            if idx == 0: # First run is short
                if len(runs) > 1:
                    new_val = runs[1]['value']
                # else: Only one run, and it's short -> keep its value
            elif idx == len(runs) - 1: # Last run is short
                if len(runs) > 1:
                     new_val = runs[idx - 1]['value']
                # else: Only one run, and it's short -> keep its value
            else: # Middle run is short
                # Merge into the longer neighbor
                prev_run = runs[idx - 1]
                next_run = runs[idx + 1]
                if prev_run['length'] >= next_run['length']:
                    new_val = prev_run['value']
                else:
                    new_val = next_run['value']

            # Apply the merge in the state list 's' only if neighbor exists and has a different state
            if new_val != -1 and new_val != val_to_merge:
                for j in range(start, start + length):
                    # Check index bounds just in case
                    if 0 <= j < n:
                        s[j] = new_val
                merges_made_in_pass += 1
            # If new_val is same as original, or no valid neighbor, no change needed for this run

        if merges_made_in_pass == 0 and indices_to_merge:
             # This can happen if short runs are surrounded by runs of the same value
             # or are at the ends with no different neighbor. Stop iterating.
             logger.debug(f"RLE pass {iteration}: No effective merges made despite short runs existing. Converged.")
             break

    if iteration >= max_iterations:
        logger.warning(f"RLE debouncing reached max iterations ({max_iterations}). Results might be unstable or incomplete.")

    return s

def apply_rle_debouncing(
    df_states: pd.DataFrame,
    tolerance_frames: int,
    state_col: str, # Input column with raw states (e.g., 'state_initial')
    debounced_col: str # Output column name for debounced states (e.g., 'state')
) -> pd.DataFrame:
    """
    Applies Run-Length Encoding (RLE) debouncing to state assignments in a DataFrame.
    Operates independently on each 'chain'. Assumes states are comparable (e.g., strings or numbers).

    Args:
        df_states: DataFrame containing time series data with 'frame' and 'chain' columns,
                   sorted by chain then frame. Must contain the `state_col`.
        tolerance_frames: Minimum number of consecutive frames for a state to be kept.
                          Runs shorter than this will be merged into neighbors.
        state_col: Name of the column containing the raw state assignments to be debounced.
        debounced_col: Name of the column to store the debounced state assignments.

    Returns:
        DataFrame with an added column `debounced_col` containing the debounced states.
        Returns a copy of the original DataFrame if input is invalid or debouncing fails.
    """
    required_cols = ['frame', 'chain', state_col]
    if df_states is None or df_states.empty or not all(col in df_states.columns for col in required_cols):
        logger.error(f"Cannot apply debouncing: DataFrame invalid or missing required columns ({required_cols}).")
        return df_states.copy() # Return a copy to avoid modifying original on failure

    if tolerance_frames <= 1:
        logger.info("Debouncing tolerance <= 1, copying raw states to debounced column.")
        df_out = df_states.copy()
        df_out[debounced_col] = df_out[state_col]
        return df_out

    logger.info(f"Applying RLE debouncing (tolerance={tolerance_frames} frames) to column '{state_col}' -> '{debounced_col}'")
    df_out = df_states.copy()

    # Use numeric mapping for RLE processing for efficiency and consistency
    # Infer states present in the input column
    possible_states = df_out[state_col].unique()
    # Ensure consistent mapping order (important if states are numeric but non-sequential)
    state_map = {state: i for i, state in enumerate(sorted(possible_states))}
    inverse_state_map = {i: state for state, i in state_map.items()}
    logger.debug(f"Using state map for RLE: {state_map}")

    all_debounced_numeric = []
    original_indices = [] # Keep track of original index to map back correctly

    # Ensure data is sorted correctly for RLE logic within chains
    df_sorted = df_out.sort_values(['chain', 'frame'])

    for chain, chain_data in df_sorted.groupby('chain'):
        if chain_data.empty:
            logger.warning(f"No data for chain {chain} during debouncing.")
            continue

        # Map to numeric, run debouncing, map back
        raw_numeric_states = chain_data[state_col].map(state_map).tolist()
        debounced_numeric = _rle_debounce_single_series(raw_numeric_states, tolerance_frames)

        if len(debounced_numeric) != len(raw_numeric_states):
            logger.error(f"Debouncing length mismatch for chain {chain}! Input: {len(raw_numeric_states)}, Output: {len(debounced_numeric)}. Skipping debouncing for this chain.")
            # Append original numeric states if debouncing failed for this chain
            all_debounced_numeric.extend(raw_numeric_states)
        else:
            all_debounced_numeric.extend(debounced_numeric)

        original_indices.extend(chain_data.index.tolist())

    if len(all_debounced_numeric) != len(df_out):
        logger.error("Overall length mismatch after processing all chains. Debouncing failed. Returning original states.")
        df_out[debounced_col] = df_out[state_col]
        return df_out

    # Create a Series with the correct index to assign back to the DataFrame
    # Ensure the series is created with the numeric values first
    debounced_series_numeric = pd.Series(all_debounced_numeric, index=original_indices)
    # Map back to original state representation
    df_out[debounced_col] = debounced_series_numeric.map(inverse_state_map)
    # Re-sort df_out back to its original index order if necessary (usually good practice)
    df_out = df_out.sort_index()

    # Verification (optional)
    changes = (df_out[state_col] != df_out[debounced_col]).sum()
    logger.info(f"Debouncing complete. {changes} state entries changed.")
    # logger.debug(f"Debounced state counts:\n{df_out[debounced_col].value_counts(dropna=False)}")

    return df_out

# Placeholder for signal processing functions
# Next: determine_reference_distances function
# Finally: _rle_debounce_single_series and apply_rle_debouncing functions 