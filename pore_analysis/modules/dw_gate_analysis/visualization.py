"""
Functions for visualizing DW-gate analysis results.
"""

import logging
import os
from typing import Optional, Dict

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Assuming utils is a sibling module
from .utils import save_plot, OPEN_STATE, CLOSED_STATE

logger = logging.getLogger(__name__)

def plot_distance_vs_state(
    df_states: pd.DataFrame,
    events_df: pd.DataFrame, # Need events for plotting state bars efficiently
    output_dir: str,
    distance_threshold: float, # From config
    closed_ref_dist: float,
    open_ref_dist: float,
    state_col: str = 'state', # Column in df_states and events_df with debounced state
    dist_col_prefix: str = 'Dist_', # Prefix for distance columns in df_states
    time_col: str = 'time_ns' # Column for time axis in df_states
) -> Optional[str]:
    """
    Plots raw distance vs. debounced state for each chain.
    Uses the events DataFrame to plot the state bars efficiently.

    Args:
        df_states: DataFrame containing time, chain, distance columns (prefixed),
                   and the debounced state column.
        events_df: DataFrame containing event data (chain, state, start_ns, end_ns).
        output_dir: Directory to save the plot.
        distance_threshold: The initial distance threshold for the dashed line.
        closed_ref_dist: Calculated/default closed reference distance.
        open_ref_dist: Calculated/default open reference distance.
        state_col: Name of the column containing debounced states.
        dist_col_prefix: Prefix for distance columns.
        time_col: Name of the time column (e.g., 'time_ns').

    Returns:
        Relative path to the saved plot file, or None if plotting failed.
    """
    # Check df_states for time and existence of distance columns
    distance_cols = [col for col in df_states.columns if col.startswith(dist_col_prefix)] if df_states is not None else []
    required_df_states_cols = [time_col]
    if df_states is None or df_states.empty or not all(c in df_states.columns for c in required_df_states_cols) or not distance_cols:
        logger.warning(f"Skipping distance vs state plot: Input df_states is empty, missing '{time_col}', or missing distance columns with prefix '{dist_col_prefix}'.")
        return None

    # Check events_df for necessary columns to plot state bars
    required_events_cols = ['chain', state_col, 'start_ns', 'end_ns']
    if events_df is None or events_df.empty or not all(c in events_df.columns for c in required_events_cols):
        logger.warning(f"Skipping distance vs state plot: Events data frame invalid or missing required columns ({required_events_cols}) for state bars.")
        return None

    # Identify distance columns dynamically
    chains = sorted([col[len(dist_col_prefix):] for col in distance_cols])
    n_chains = len(chains)
    if n_chains == 0:
        logger.warning("Skipping distance vs state plot: No chains derived from distance columns.")
        return None

    logger.info(f"Plotting distance vs state for chains: {chains}")
    sns.set_theme(style='ticks', context='notebook')

    # Create palettes based on the actual number of chains found
    pastel_palette = sns.color_palette("pastel", n_colors=n_chains)
    bright_palette = sns.color_palette("bright", n_colors=n_chains)
    # Map chain ID (derived from distance col) to color
    chain_color_map_pastel = {ch: col for ch, col in zip(chains, pastel_palette)}
    chain_color_map_bright = {ch: col for ch, col in zip(chains, bright_palette)}

    fig, axes = plt.subplots(n_chains, 1, figsize=(12, 2.5 * n_chains), sharex=True)
    if n_chains == 1:
        axes = [axes] # Ensure axes is always iterable

    # Find global min/max distance across relevant columns for consistent y-scale
    all_distances = df_states[distance_cols].values.flatten()
    all_distances = all_distances[~np.isnan(all_distances)] # Remove NaNs
    if len(all_distances) > 0:
        y_min = np.min(all_distances) * 0.9
        y_max = np.max(all_distances) * 1.1
    else:
        y_min, y_max = 0, distance_threshold * 2 # Default range if no data
    # Ensure range includes reference lines
    y_min = min(y_min, closed_ref_dist * 0.9)
    y_max = max(y_max, open_ref_dist * 1.1)

    # Reference heights for state indicators
    state_y = {CLOSED_STATE: closed_ref_dist, OPEN_STATE: open_ref_dist}

    for i, ch in enumerate(chains):
        ax = axes[i]
        dist_col = f"{dist_col_prefix}{ch}"
        sub = df_states[[time_col, dist_col]].dropna().sort_values(time_col)

        if sub.empty:
            ax.text(0.5, 0.5, f"No data for Chain {ch}", ha='center', va='center', transform=ax.transAxes)
            ax.set_ylabel(f"Chain {ch} (Å)") # Still label the axis
            ax.grid(False)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set_ylim(y_min, y_max)
            continue

        # Plot distance trace with pastel color
        ax.plot(sub[time_col], sub[dist_col],
                color=chain_color_map_pastel.get(ch, 'grey'), alpha=0.8, linewidth=1.0)

        # Plot debounced state bars using event data
        chain_events = events_df[events_df['chain'] == ch]
        if not chain_events.empty:
            for _, event in chain_events.iterrows():
                state = event[state_col]
                # Use get with default for safety, though state should exist
                y = state_y.get(state, (closed_ref_dist + open_ref_dist) / 2) # Default to midpoint if state unknown
                ax.plot([event['start_ns'], event['end_ns']], [y, y], lw=5,
                        color=chain_color_map_bright.get(ch, 'black'), solid_capstyle='butt',
                        label='_nolegend_')
        else:
            logger.warning(f"No events found for chain {ch} to plot state bars.")

        # Add threshold line
        ax.axhline(distance_threshold, ls=':', color='black', alpha=0.6, linewidth=1.0)

        # Set consistent y-scale and label
        ax.set_ylim(y_min, y_max)
        ax.set_ylabel(f"Chain {ch} (Å)")
        ax.grid(False)

        # Remove top/right spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    # Finalize plot
    axes[-1].set_xlabel('Time (ns)')
    fig.suptitle('DW-Gate Distance vs. Debounced State', fontsize=14)

    # Adjust layout - tight_layout often sufficient
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust rect for suptitle

    # Save plot
    plot_filename = "dw_gate_distance_vs_state.png"
    # Assume output_dir is the specific subdir for this analysis (e.g., .../dw_gate_analysis/)
    plot_full_path = os.path.join(output_dir, plot_filename)
    save_plot(fig, plot_full_path) # Use the utility function from utils.py

    # Return relative path assuming output_dir is within the run directory
    plot_rel_path = os.path.join(os.path.basename(output_dir), plot_filename)
    return plot_rel_path

def plot_open_probability(
    probability_df: pd.DataFrame,
    output_dir: str
) -> Optional[str]:
    """Plots open-state probability per chain as a bar chart.

    Args:
        probability_df: DataFrame containing at least 'chain' and 'open' columns
                       (as calculated by statistics.calculate_dw_statistics).
        output_dir: Directory to save the plot.

    Returns:
        Relative path to the saved plot file, or None if plotting failed.
    """
    required_cols = ['chain', OPEN_STATE]
    if probability_df is None or probability_df.empty or not all(c in probability_df.columns for c in required_cols):
        logger.warning(f"Skipping open probability plot: Probability data invalid or missing columns ({required_cols}).")
        return None

    # Ensure correct column name is used for probability
    prob_col = OPEN_STATE # Use the constant
    prob_plot_df = probability_df.sort_values('chain')

    if prob_plot_df.empty:
         logger.warning("Skipping open probability plot: No data after filtering/sorting.")
         return None

    logger.info("Plotting open probability per chain...")
    sns.set_theme(style='ticks', context='notebook')
    n_chains = len(prob_plot_df['chain'].unique())
    plt.figure(figsize=(max(6, n_chains * 1.5), 5)) # Adjust width based on num chains
    ax = sns.barplot(data=prob_plot_df, x='chain', y=prob_col, palette='viridis_r') # Changed palette

    # Add probability values above bars
    for i, p in enumerate(prob_plot_df[prob_col]):
        plt.text(i, p + 0.02, f'{p:.3f}', ha='center', fontsize=10, fontweight='normal') # Less bold text

    plt.ylim(0, 1.05) # Adjusted ylim slightly
    plt.title('DW-Gate Open State Probability per Chain')
    plt.xlabel('Chain')
    plt.ylabel('Open Probability')
    ax.grid(axis='y', linestyle=':', alpha=0.7) # Add subtle horizontal grid
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tight_layout()

    # Save plot
    plot_filename = "dw_gate_open_probability.png"
    plot_full_path = os.path.join(output_dir, plot_filename)
    save_plot(plt.gcf(), plot_full_path)

    # Return relative path
    plot_rel_path = os.path.join(os.path.basename(output_dir), plot_filename)
    return plot_rel_path

def plot_state_heatmap(
    df_states: pd.DataFrame,
    output_dir: str,
    state_col: str = 'state',
    time_col: str = 'time_ns',
    frame_col: str = 'frame' # Needed for fallback pivot
) -> Optional[str]:
    """
    Plots DW-Gate state transitions as a heatmap.

    Pivots the data to have time/frame as index and chains as columns.
    Uses a custom colormap: Closed=DarkBlue, Open=LightBlue, Missing/NaN=Gray.

    Args:
        df_states: DataFrame in long format with time, frame, chain, and state columns.
                   Must be sorted by chain, then frame.
        output_dir: Directory to save the plot.
        state_col: Name of the column containing debounced state information.
        time_col: Name of the time column (e.g., 'time_ns').
        frame_col: Name of the frame column (used as fallback index).

    Returns:
        Relative path to the saved plot file, or None if plotting failed.
    """
    required_cols = [time_col, frame_col, 'chain', state_col]
    if df_states is None or df_states.empty or not all(c in df_states.columns for c in required_cols):
        logger.warning(f"Skipping state heatmap plot: Input df_states missing required columns ({required_cols}).")
        return None

    logger.info("Plotting state heatmap...")
    sns.set_theme(style='white', context='notebook')

    # Prepare data for heatmap: Pivot debounced data using pivot_table for robustness
    df_pivot = None
    pivot_index_col = time_col # Prefer time index
    try:
        # Use pivot_table with aggfunc='first' to handle potential duplicates gracefully
        df_pivot = pd.pivot_table(df_states, index=time_col, columns='chain', values=state_col, aggfunc='first')
        if df_pivot.index.has_duplicates:
             logger.warning(f"Duplicate time entries found in index after pivot_table on '{time_col}'. This might indicate upstream data issues. Proceeding, but heatmap time axis may be imperfect.")
             # Keep using time index despite duplicates if pivot_table succeeded
             pivot_index_col = time_col
        else:
             pivot_index_col = time_col # Confirm time index is used

    except Exception as e:
        logger.warning(f"Pivot_table failed using time column '{time_col}': {e}. Attempting pivot_table using frame index.")
        df_pivot = None # Ensure fallback is triggered
        pivot_index_col = frame_col # Set index to frame for fallback

    # Fallback: Pivot using frame index if time pivot failed
    if df_pivot is None:
        try:
            df_pivot = pd.pivot_table(df_states, index=frame_col, columns='chain', values=state_col, aggfunc='first')
            pivot_index_col = frame_col # Frame index is now used
            if df_pivot.index.has_duplicates:
                 logger.warning(f"Duplicate frame entries found in index after pivot_table on '{frame_col}'. Upstream data issues likely. Proceeding, but heatmap frame axis may be imperfect.")
                 # Proceed even with duplicate frames

        except Exception as e2:
            logger.error(f"Fallback pivot_table using frame index '{frame_col}' also failed: {e2}. Skipping heatmap.")
            return None

    # Check if pivoting was successful
    if df_pivot is None or df_pivot.empty:
        logger.error("Skipping state heatmap: Pivoted data is None or empty after pivot attempts.")
        return None

    # Map states to numeric values (Closed=0, Open=1)
    state_map = {CLOSED_STATE: 0, OPEN_STATE: 1}
    # Map known states, fill NaNs from pivot with -1 (Missing)
    df_numeric = df_pivot.replace(state_map).fillna(-1)
    # Convert to integer type after filling NaNs if possible (makes heatmap cleaner)
    try:
        df_numeric = df_numeric.astype(int)
    except ValueError:
        logger.warning("Could not convert heatmap numeric data to integer type.")


    # Create a custom colormap: Missing=Gray, Closed=DarkBlue, Open=LightBlue
    # Order corresponds to numeric values: -1, 0, 1
    cmap = mcolors.ListedColormap(['#cccccc', '#00008b', '#add8e6']) # Gray, DarkBlue, LightBlue
    bounds = [-1.5, -0.5, 0.5, 1.5] # Boundaries for -1, 0, 1
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Create figure
    n_chains = len(df_numeric.columns)
    fig_height = max(3, 0.4 * n_chains) # Adjust height based on chains
    plt.figure(figsize=(12, fig_height))
    ax = plt.gca()

    # Determine extent [x_min, x_max, y_min, y_max] for imshow
    # Y extent is based on number of chains
    y_extent = [-0.5, n_chains - 0.5]
    # X extent depends on whether we used time or frame index
    x_min = df_pivot.index.min()
    x_max = df_pivot.index.max()
    x_extent = [x_min, x_max]
    xlabel = f'{pivot_index_col.replace("_", " ").capitalize()}' # Format axis label

    # Plot the heatmap using imshow
    # Transpose df_numeric so chains are on y-axis
    im = ax.imshow(df_numeric.T, aspect='auto', cmap=cmap, norm=norm, interpolation='nearest',
                   extent=[x_extent[0], x_extent[1], y_extent[0], y_extent[1]],
                   origin='lower') # Origin lower puts first chain at bottom

    # Set labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Chain')
    ax.set_title('DW Gate State Heatmap')

    # Set y-ticks to chain labels (use sorted order from columns)
    ax.set_yticks(np.arange(n_chains))
    ax.set_yticklabels(df_numeric.columns) # Columns are chain IDs

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, ticks=[-1, 0, 1], orientation='vertical', fraction=0.05, pad=0.04)
    # Match colorbar labels to the numeric mapping
    cbar.ax.set_yticklabels(['Missing', 'Closed', 'Open'], fontsize='small') # Gray(-1), DarkBlue(0), LightBlue(1)

    # Remove gridlines
    ax.grid(False)
    plt.tight_layout()

    # Save plot
    plot_filename = "dw_gate_state_heatmap.png"
    plot_full_path = os.path.join(output_dir, plot_filename)
    save_plot(plt.gcf(), plot_full_path)

    # Return relative path
    plot_rel_path = os.path.join(os.path.basename(output_dir), plot_filename)
    return plot_rel_path

def _plot_duration_panel(ax, data, chains, state, palette_violin, palette_points, chain_map_bright, title):
    """Internal helper to plot duration distribution for one state (Open/Closed)."""
    if data.empty:
        ax.text(0.5, 0.5, f"No {state.lower()} events found", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        ax.set_xlabel('Chain')
        ax.set_ylabel('') # Y-label set globally
        ax.grid(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        return

    # Violin Plot (main distribution)
    try:
        sns.violinplot(ax=ax, data=data, x='chain', y='duration_ns', hue='chain',
                       order=chains, hue_order=chains, # Ensure consistent order
                       palette=palette_violin, inner=None, density_norm='width', linewidth=1.5,
                       saturation=0.7, legend=False) # Disable automatic legend
    except Exception as e_violin:
        logger.error(f"Error during violinplot for {state} durations: {e_violin}", exc_info=True)
        ax.text(0.5, 0.5, f"Violin plot failed for {state}", ha='center', va='center', transform=ax.transAxes, color='red')
        # Allow function to continue if possible, but mark the failure

    # Box plot (for quartiles, median) - narrow and without outliers
    try:
        sns.boxplot(ax=ax, data=data, x='chain', y='duration_ns', hue='chain',
                    order=chains, hue_order=chains,
                    palette=chain_map_bright, # Use bright colors for box
                    width=0.2, boxprops={'zorder': 2, 'alpha': 0.8},
                    whiskerprops={'zorder': 2, 'alpha': 0.8, 'ls': '-'},
                    capprops={'zorder': 2, 'alpha': 0.8},
                    medianprops={'color': 'black', 'linewidth': 2, 'zorder': 3, 'alpha': 0.9},
                    showfliers=False, showcaps=True, legend=False) # Hide outliers, show caps
    except Exception as e_box:
        logger.error(f"Error during boxplot for {state} durations: {e_box}", exc_info=True)
        # Add text to indicate failure if not already present
        if not ax.texts:
             ax.text(0.5, 0.5, f"Box plot failed for {state}", ha='center', va='center', transform=ax.transAxes, color='red')

    # Swarm plot (individual points) - use bright palette, smaller size
    try:
        # Only plot if number of points isn't excessive
        if len(data) < 5000: # Threshold to avoid overcrowding and slow plotting
             sns.swarmplot(ax=ax, data=data, x='chain', y='duration_ns', hue='chain',
                           order=chains, hue_order=chains,
                           palette=palette_points, size=3, alpha=0.6, legend=False, zorder=1)
        else:
            # Optionally plot stripplot as fallback if too many points
            sns.stripplot(ax=ax, data=data, x='chain', y='duration_ns', hue='chain',
                          order=chains, hue_order=chains,
                          palette=palette_points, size=2, alpha=0.4, jitter=0.2, legend=False, zorder=1)
            logger.info(f"Using stripplot instead of swarmplot for {state} durations due to large number of points ({len(data)}).")
    except Exception as e_swarm:
        logger.error(f"Error during swarmplot/stripplot for {state} durations: {e_swarm}", exc_info=True)
        if not ax.texts:
             ax.text(0.5, 0.5, f"Points plot failed for {state}", ha='center', va='center', transform=ax.transAxes, color='red')

    # Add mean points (optional)
    try:
        means = data.groupby('chain')['duration_ns'].mean()
        ax.scatter(x=range(len(chains)), y=[means.get(ch, np.nan) for ch in chains],
                   color='red', marker='o', s=50, zorder=4, label='Mean', edgecolors='white') # Ensure mean is on top
    except Exception as e_mean:
         logger.error(f"Error calculating or plotting mean points for {state} durations: {e_mean}", exc_info=True)

    ax.set_title(title)
    ax.set_xlabel('Chain')
    ax.set_ylabel('') # Y-label set globally
    ax.grid(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


def plot_duration_distributions(
    events_df: pd.DataFrame,
    output_dir: str
) -> Optional[str]:
    """
    Plots distributions of event durations using violin plots overlaid with
    box plots and swarm/strip plots for individual events.

    Args:
        events_df: DataFrame containing event data (chain, state, duration_ns).
        output_dir: Directory to save the plot.

    Returns:
        Relative path to the saved plot file, or None if plotting failed.
    """
    if events_df is None or events_df.empty or not all(c in events_df.columns for c in ['chain', 'state', 'duration_ns']):
        logger.warning("Skipping duration distribution plot: Events data invalid or missing required columns.")
        return None

    logger.info("Plotting event duration distributions...")
    sns.set_theme(style='ticks', context='notebook')

    # Filter events by state
    open_events = events_df.query("state == @OPEN_STATE")
    closed_events = events_df.query("state == @CLOSED_STATE")

    # Get sorted chains
    chains = sorted(events_df.chain.unique())
    n_chains = len(chains)
    if n_chains == 0:
        logger.warning("Skipping duration distribution plot: No chains found in event data.")
        return None

    # Define consistent palettes
    # Using different palettes for different elements can improve clarity
    palette_violin = sns.color_palette("pastel", n_colors=n_chains)
    palette_points = sns.color_palette("muted", n_colors=n_chains)
    palette_bright = sns.color_palette("bright", n_colors=n_chains)
    chain_map_violin = {ch: col for ch, col in zip(chains, palette_violin)}
    chain_map_points = {ch: col for ch, col in zip(chains, palette_points)}
    chain_map_bright = {ch: col for ch, col in zip(chains, palette_bright)}

    # Create figure with two subplots (Open, Closed) side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=False) # Start with unshared Y

    # Plot Open State Durations
    _plot_duration_panel(axes[0], open_events, chains, OPEN_STATE,
                         chain_map_violin, chain_map_points, chain_map_bright, 'Open State Durations')

    # Plot Closed State Durations
    _plot_duration_panel(axes[1], closed_events, chains, CLOSED_STATE,
                         chain_map_violin, chain_map_points, chain_map_bright, 'Closed State Durations')

    # Determine common y-axis limits or keep separate?
    # Keeping separate allows better view of each distribution if scales differ greatly.
    # y_min = min(axes[0].get_ylim()[0], axes[1].get_ylim()[0])
    # y_max = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])
    # axes[0].set_ylim(y_min, y_max)
    # axes[1].set_ylim(y_min, y_max)
    axes[0].set_ylabel('Duration (ns)') # Set Y label only on the left plot

    # Add a single legend below the plot
    legend_elements = [
        Patch(facecolor='grey', edgecolor='black', alpha=0.6, label='Distribution (Violin)'),
        Line2D([0], [0], color='black', lw=2, label='Median (Box)'),
        Line2D([0], [0], marker='o', color='w', label='Mean', markerfacecolor='red', markersize=8, markeredgecolor='black'),
        Line2D([0], [0], marker='.', color='grey', label='Individual Events', markersize=8, linestyle='None')
    ]
    fig.legend(handles=legend_elements, loc='lower center',
               bbox_to_anchor=(0.5, 0.01), ncol=len(legend_elements), frameon=False)

    fig.suptitle("DW-Gate Event Duration Distributions", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.06, 1, 0.95]) # Adjust layout for suptitle and legend

    # Save plot
    plot_filename = "dw_gate_duration_distribution.png"
    plot_full_path = os.path.join(output_dir, plot_filename)
    save_plot(fig, plot_full_path)

    # Return relative path
    plot_rel_path = os.path.join(os.path.basename(output_dir), plot_filename)
    return plot_rel_path

# Placeholder for other plotting functions 