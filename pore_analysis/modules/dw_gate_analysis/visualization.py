# filename: pore_analysis/modules/dw_gate_analysis/visualization.py
"""
Visualization functions for DW-Gate analysis results.
Generates plots based on data retrieved from the database.
Loads statistics DataFrames and KDE data instead of recomputing.
"""

import logging
import os
import time
import sqlite3
import json # <<< ADDED IMPORT >>>
from typing import Optional, Dict, Any, Tuple, List

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Import from core modules
try:
    from pore_analysis.core.plotting_style import STYLE, setup_style # <<< Ensure STYLE is imported >>>
    from pore_analysis.core.database import (
        register_module, update_module_status, get_product_path, register_product,
        get_module_status, get_all_metrics
    )
    from pore_analysis.core.logging import setup_system_logger
    # Import config defaults needed for metric fallbacks and plot annotations
    from pore_analysis.core.config import (
        DW_GATE_DEFAULT_CLOSED_REF_DIST as DEFAULT_CLOSED_REF,
        DW_GATE_DEFAULT_OPEN_REF_DIST as DEFAULT_OPEN_REF,
        DEFAULT_CUTOFF as DEFAULT_DISTANCE_THRESHOLD # Needed for _plot_kde_distribution
    )
    CORE_AVAILABLE = True
except ImportError as e:
    # Fallback definitions...
    logging.basicConfig(level=logging.INFO) # Basic logging config
    logger = logging.getLogger(__name__)
    logger.error(f"Critical Import Error - Core modules missing: {e}. Using dummy functions/defaults.")
    CORE_AVAILABLE = False
    def register_module(*args, **kwargs): pass
    def update_module_status(*args, **kwargs): pass
    def get_product_path(*args, **kwargs): return None
    def register_product(*args, **kwargs): pass
    def get_module_status(*args, **kwargs): return 'unknown'
    def get_all_metrics(*args, **kwargs): return {} # Dummy return
    def setup_system_logger(*args, **kwargs): return logging.getLogger(__name__) # Use default logger
    STYLE = {'bright_colors': {'A': 'blue', 'B': 'red', 'C': 'green', 'D': 'orange'},
             'chain_colors': {'A': 'lightblue', 'B': 'lightcoral', 'C': 'lightgreen', 'D': 'moccasin'},
             'font_sizes': {'annotation': 9, 'axis_label': 12, 'tick_label': 10}, # Add font sizes
             'threshold_style': {'color': 'black', 'ls': ':'}, 'grid': {'alpha': 0.5, 'color': 'grey', 'linestyle': ':'}}
    def setup_style(): pass
    DEFAULT_CLOSED_REF = 2.7
    DEFAULT_OPEN_REF = 4.7
    DEFAULT_DISTANCE_THRESHOLD = 3.5

# Import from local module utils
from .utils import save_plot, OPEN_STATE, CLOSED_STATE

logger = logging.getLogger(__name__)

# Apply standard plotting style
setup_style()
plt.switch_backend('agg') # Ensure non-interactive backend

# --- Plotting Helper Functions ---

# <<< ADDED: _plot_kde_distribution function >>>
def _plot_kde_distribution(
    kde_plot_data: Dict[str, Any], # Loaded from JSON
    distance_threshold: float, # General distance cutoff
    output_dir: str,
    run_dir: str,
    db_conn: sqlite3.Connection,
    module_name: str # Should be the visualization module name
) -> Optional[str]:
    """
    Generates the distance distribution plot (Histogram + KDE + References)
    using data loaded from the kde_plot_data JSON file.
    Registers the plot product under the visualization module.
    Does NOT set a Python title.
    """
    logger_local = logging.getLogger(__name__) # Use local logger
    if not kde_plot_data:
        logger_local.warning("No KDE data provided for distribution plot.")
        return None

    # Extract data safely
    all_distances = np.array(kde_plot_data.get('all_distances', []))
    combined_kde = kde_plot_data.get('combined_kde', {})
    x_kde = np.array(combined_kde.get('x', [])) if combined_kde.get('x') is not None else np.array([])
    y_kde = np.array(combined_kde.get('y', [])) if combined_kde.get('y') is not None else np.array([])
    peaks = np.array(combined_kde.get('peaks', [])) if combined_kde.get('peaks') is not None else np.array([])
    # kmeans_centers might be None or a list
    kmeans_centers = kde_plot_data.get('kmeans_centers')
    final_closed_ref = kde_plot_data.get('final_closed_ref')
    final_open_ref = kde_plot_data.get('final_open_ref')


    if len(all_distances) < 10 or len(x_kde) == 0 or len(y_kde) == 0:
        logger_local.warning("Insufficient data points loaded for KDE distribution plot.")
        # Optionally create a blank plot indicating insufficient data? For now, return None.
        return None

    try:
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot Histogram
        sns.histplot(all_distances, bins=50, stat="density", kde=False, ax=ax,
                     color=STYLE['chain_colors'].get('A', 'grey'), alpha=0.5, label='Raw Distance Distribution')

        # Plot KDE
        ax.plot(x_kde, y_kde, color=STYLE['bright_colors'].get('B', 'red'), linewidth=2, label='Kernel Density Estimate (KDE)')

        # Plot KDE Peaks
        peak_heights = np.interp(peaks, x_kde, y_kde) if len(peaks) > 0 else []
        ax.plot(peaks, peak_heights, "x", color=STYLE['bright_colors'].get('C','green'), markersize=8, label='KDE Peaks')

        # Plot Reference Lines (Final Used)
        if final_closed_ref is not None:
            ax.axvline(final_closed_ref, color='blue', linestyle='--', linewidth=1.5, label=f'Closed Ref: {final_closed_ref:.2f} Å')
        if final_open_ref is not None:
            ax.axvline(final_open_ref, color='orange', linestyle='--', linewidth=1.5, label=f'Open Ref: {final_open_ref:.2f} Å')

        # Plot K-Means Centers if available
        if kmeans_centers is not None and len(kmeans_centers) == 2:
             # Ensure kmeans_centers is numpy array for plotting if needed, though list works here
            ax.plot(kmeans_centers, [ax.get_ylim()[1] * 0.05] * 2, 'P', # Pentagon marker
                    color='purple', markersize=10, label='K-Means Centers', linestyle='None')

        # Plot original distance threshold (e.g., 3.5 A) for context if needed
        # ax.axvline(distance_threshold, color='grey', linestyle=':', linewidth=1.0, alpha=0.7, label=f'Dist Cutoff: {distance_threshold:.1f} Å')

        ax.set_xlabel("DW Gate Distance (Å)")
        ax.set_ylabel("Density")
        # Use font sizes from STYLE if defined, otherwise defaults will apply via setup_style()
        ax.legend(fontsize=STYLE.get('font_sizes', {}).get('annotation', 'small'))
        ax.grid(True, alpha=STYLE['grid']['alpha'], color=STYLE['grid']['color'], linestyle=STYLE['grid']['linestyle'])
        # Ensure xlim handles potential empty all_distances after filtering? Check logic.
        min_dist_val = np.min(all_distances) if len(all_distances) > 0 else 0
        ax.set_xlim(left=max(0, min_dist_val - 1))
        ax.set_ylim(bottom=0)

        plt.tight_layout()
        plot_filename = "dw_gate_distance_distribution.png" # Consistent filename
        plot_filepath = os.path.join(output_dir, plot_filename)
        save_plot(fig, plot_filepath) # Use the utility save function

        # Register Product (using the VISUALIZATION module name)
        rel_path = os.path.relpath(plot_filepath, run_dir)
        register_product(db_conn, module_name, "png", "plot", rel_path,
                         subcategory="distance_distribution", # MATCHES plots_dict.json
                         description="DW-Gate raw distance distribution (KDE) and reference states.")
        return rel_path

    except Exception as e:
        logger_local.error(f"Failed to generate KDE distribution plot: {e}", exc_info=True)
        if 'fig' in locals() and plt.fignum_exists(fig.number):
             plt.close(fig)
        return None
# --- <<< END ADDED FUNCTION >>> ---


def get_chain_color_maps(chains):
    """
    Creates consistent color mappings for chains using seaborn palettes.
    Uses STYLE from core.plotting_style if available.
    """
    n_chains = len(chains)
    # Use defined STYLE palettes if available, otherwise fallback
    pastel_defaults = sns.color_palette("pastel", n_colors=max(n_chains, 4))
    bright_defaults = sns.color_palette("bright", n_colors=max(n_chains, 4))
    pastel_palette = STYLE.get('chain_colors', {})
    bright_palette = STYLE.get('bright_colors', {})

    chain_color_map_pastel = {}
    chain_color_map_bright = {}
    pastel_list = []
    bright_list = []

    for i, ch in enumerate(chains):
        chain_suffix = ch[-1] # Assume last character is A, B, C, D for specific colors
        # Get specific color from STYLE if suffix matches and exists, else use default index
        pastel_c = pastel_palette.get(chain_suffix, pastel_defaults[i % len(pastel_defaults)])
        bright_c = bright_palette.get(chain_suffix, bright_defaults[i % len(bright_defaults)])
        chain_color_map_pastel[ch] = pastel_c
        chain_color_map_bright[ch] = bright_c
        pastel_list.append(pastel_c)
        bright_list.append(bright_c)

    return {
        'pastel': chain_color_map_pastel,
        'bright': chain_color_map_bright,
        'pastel_list': pastel_list,
        'bright_list': bright_list
    }

def _plot_distance_vs_state(
    df_states_wide: pd.DataFrame, # Needs original distances
    events_df: pd.DataFrame, # Need events for plotting final state bars efficiently
    output_dir: str,
    run_dir: str, # Needed for registering relative path
    db_conn: sqlite3.Connection, # Needed for registration
    module_name: str, # Name of the *visualization* module
    # distance_threshold: float, # Threshold for initial state assignment (not plotted directly)
    closed_ref_dist: float, # Final reference used
    open_ref_dist: float,   # Final reference used
    state_col: str = 'State', # Use Standardized column name from events_df
    dist_col_prefix: str = 'Dist_', # Prefix for distance columns in df_states_wide
    time_col: str = 'Time (ns)' # Use Standardized column name
) -> Optional[str]:
    """
    Plots raw distance vs. debounced state for each chain.
    Uses the events DataFrame to plot the state bars efficiently.
    Registers the plot product.
    """
    # Check df_states_wide for time and existence of distance columns
    distance_cols = [col for col in df_states_wide.columns if col.startswith(dist_col_prefix)] if df_states_wide is not None else []
    required_df_states_cols = [time_col] # Check for standardized 'Time (ns)'
    if df_states_wide is None or df_states_wide.empty or not all(c in df_states_wide.columns for c in required_df_states_cols) or not distance_cols:
        logger.warning(f"Skipping distance vs state plot: Input df_states_wide is empty, missing '{time_col}', or missing distance columns with prefix '{dist_col_prefix}'.")
        return None

    # Check events_df for necessary columns to plot state bars
    # Use standardized names: 'Chain', 'State', 'Start Time (ns)', 'End Time (ns)'
    required_events_cols = ['Chain', state_col, 'Start Time (ns)', 'End Time (ns)']
    if events_df is None or events_df.empty or not all(c in events_df.columns for c in required_events_cols):
        logger.warning(f"Skipping distance vs state plot: Events data frame invalid or missing required columns ({required_events_cols}) for state bars.")
        return None

    chains = sorted([col[len(dist_col_prefix):] for col in distance_cols])
    n_chains = len(chains)
    if n_chains == 0:
        logger.warning("Skipping distance vs state plot: No chains derived from distance columns.")
        return None

    logger.info(f"Plotting distance vs state for chains: {chains}")

    color_maps = get_chain_color_maps(chains)
    chain_color_map_pastel = color_maps['pastel']
    chain_color_map_bright = color_maps['bright']

    fig, axes = plt.subplots(n_chains, 1, figsize=(12, 2.5 * n_chains), sharex=True, squeeze=False)
    axes = axes.flatten()

    all_distances = df_states_wide[distance_cols].values.flatten()
    all_distances = all_distances[~np.isnan(all_distances)]

    # Calculate Y limits based on data and reference points
    y_min, y_max = 0, open_ref_dist * 1.2 # Default
    if len(all_distances) > 0:
        y_min = np.min(all_distances) * 0.9
        y_max = np.max(all_distances) * 1.1
    y_min = min(y_min, closed_ref_dist * 0.9) # Ensure refs are included
    y_max = max(y_max, open_ref_dist * 1.1)

    # Use final reference distances for plotting state lines
    state_y = {CLOSED_STATE: closed_ref_dist, OPEN_STATE: open_ref_dist}

    for i, ch in enumerate(chains):
        ax = axes[i]
        dist_col = f"{dist_col_prefix}{ch}"
        # Use standardized 'Time (ns)' column
        sub = df_states_wide[[time_col, dist_col]].dropna().sort_values(time_col)

        if sub.empty:
            ax.text(0.5, 0.5, f"No data for Chain {ch}", ha='center', va='center', transform=ax.transAxes)
        else:
            # Plot raw distance trace
            ax.plot(sub[time_col], sub[dist_col],
                    color=chain_color_map_pastel.get(ch, 'grey'), alpha=0.8, linewidth=1.0)

            # Plot state bars from events_df using standardized column names
            chain_events = events_df[events_df['Chain'] == ch] # Use 'Chain'
            if not chain_events.empty:
                for _, event in chain_events.iterrows():
                    state = event[state_col] # Use 'State'
                    y = state_y.get(state, (closed_ref_dist + open_ref_dist) / 2) # Use refs for y-pos
                    start_t = event['Start Time (ns)']
                    end_t = event['End Time (ns)']
                    ax.plot([start_t, end_t], [y, y], lw=5,
                            color=chain_color_map_bright.get(ch, 'black'), solid_capstyle='butt',
                            label='_nolegend_')

            # Plot reference lines (optional, maybe confusing with state bars)
            # ax.axhline(closed_ref_dist, color='blue', linestyle=':', alpha=0.5)
            # ax.axhline(open_ref_dist, color='red', linestyle=':', alpha=0.5)

        ax.set_ylim(y_min, y_max)
        ax.set_ylabel(f"Chain {ch} (Å)")
        ax.grid(False)
        ax.spines['right'].set_visible(False); ax.spines['top'].set_visible(False)

    axes[-1].set_xlabel('Time (ns)') # Use standardized name
    plt.tight_layout()

    plot_filename = "dw_gate_distance_vs_state.png"
    plot_full_path = os.path.join(output_dir, plot_filename)
    save_plot(fig, plot_full_path)

    rel_path = os.path.relpath(plot_full_path, run_dir)
    # Register the plot product
    register_product(db_conn, module_name, "png", "plot", rel_path,
                     subcategory="distance_vs_state", # Consistent subcategory
                     description="DW-Gate raw distance vs debounced state per chain.")
    return rel_path

def _plot_open_probability(
    probability_df: pd.DataFrame, # Expects DF loaded from CSV
    output_dir: str,
    run_dir: str, # Needed for registering relative path
    db_conn: sqlite3.Connection, # Needed for registration
    module_name: str # Name of the *visualization* module
) -> Optional[str]:
    """Plots open-state probability per chain as a bar chart using loaded data. Registers product."""
    # Expect standardized columns: 'Chain', OPEN_STATE constant ('open')
    required_cols = ['Chain', OPEN_STATE]
    if probability_df is None or probability_df.empty or not all(c in probability_df.columns for c in required_cols):
        logger.warning(f"Skipping open probability plot: Loaded probability data invalid or missing columns ({required_cols}).")
        return None

    prob_col = OPEN_STATE
    # Use standardized 'Chain' column
    prob_plot_df = probability_df.sort_values('Chain')
    if prob_plot_df.empty: logger.warning("Skipping open probability plot: No data after filtering/sorting."); return None

    logger.info("Plotting open probability per chain...")
    chains = prob_plot_df['Chain'].unique().tolist() # Use 'Chain'
    color_maps = get_chain_color_maps(chains)

    plt.figure(figsize=(max(6, len(chains) * 1.5), 5))
    # Use standardized 'Chain' column for x-axis
    ax = sns.barplot(data=prob_plot_df, x='Chain', y=prob_col, palette=color_maps['pastel_list'])
    for i, p in enumerate(prob_plot_df[prob_col]):
        plt.text(i, p + 0.02, f'{p:.3f}', ha='center', fontsize=10, fontweight='normal')

    plt.ylim(0, 1.05)
    plt.xlabel('Chain') # Use standardized name
    plt.ylabel('Open Probability')
    ax.grid(axis='y', **STYLE.get('grid', {'linestyle': ':', 'alpha': 0.7}))
    ax.spines['right'].set_visible(False); ax.spines['top'].set_visible(False)
    plt.tight_layout()

    plot_filename = "dw_gate_open_probability.png"
    plot_full_path = os.path.join(output_dir, plot_filename)
    save_plot(plt.gcf(), plot_full_path)

    rel_path = os.path.relpath(plot_full_path, run_dir)
    # Register the plot product
    register_product(db_conn, module_name, "png", "plot", rel_path,
                     subcategory="open_probability", # Consistent subcategory
                     description="DW-Gate Open state probability per chain.")
    return rel_path

def _plot_state_heatmap(
    df_states_long: pd.DataFrame, # Requires debounced state column ('state') and chain, Frame/Time (ns)
    output_dir: str,
    run_dir: str, # Needed for registering relative path
    db_conn: sqlite3.Connection, # Needed for registration
    module_name: str, # Name of the *visualization* module
    state_col: str = 'state', # Internal state column name in df_states_long
    time_col: str = 'Time (ns)', # Standardized name
    frame_col: str = 'Frame' # Standardized name
) -> Optional[str]:
    """
    Plots DW-Gate state transitions as a heatmap. Registers product.
    Uses standardized input column names.
    """
    required_cols = [time_col, frame_col, 'chain', state_col] # Check internal names used for processing
    if df_states_long is None or df_states_long.empty or not all(c in df_states_long.columns for c in required_cols):
        logger.warning(f"Skipping state heatmap plot: Input df_states_long missing required columns ({required_cols}).")
        return None

    logger.info("Plotting state heatmap...")
    pastel_blue = (0.68, 0.85, 0.9) # Consistent colors
    pastel_red = (0.98, 0.7, 0.7)
    cmap = mcolors.ListedColormap([pastel_blue, pastel_red])
    bounds = [-0.5, 0.5, 1.5]; norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # --- Pivoting Logic (use standardized index names) ---
    df_pivot = None; pivot_index_col = time_col # Default to Time (ns)
    try:
        df_pivot = pd.pivot_table(df_states_long, index=time_col, columns='chain', values=state_col, aggfunc='first')
        if df_pivot.index.has_duplicates: logger.warning(f"Duplicate time entries found in heatmap pivot using '{time_col}'.")
    except Exception: df_pivot = None; pivot_index_col = frame_col # Fallback to Frame
    if df_pivot is None:
        try:
            df_pivot = pd.pivot_table(df_states_long, index=frame_col, columns='chain', values=state_col, aggfunc='first')
            pivot_index_col = frame_col
            if df_pivot.index.has_duplicates: logger.warning(f"Duplicate frame entries found in heatmap pivot using '{frame_col}'.")
        except Exception as e2: logger.error(f"Fallback pivot using frame failed: {e2}. Skipping heatmap."); return None
    if df_pivot is None or df_pivot.empty: logger.error("Pivoted data is empty. Skipping heatmap."); return None
    # --- End Pivoting Logic ---

    state_map = {CLOSED_STATE: 0, OPEN_STATE: 1}
    df_numeric = df_pivot.replace(state_map).fillna(0) # Fill NaN with closed state
    try: df_numeric = df_numeric.astype(int)
    except ValueError: logger.warning("Could not convert heatmap data to int.")

    n_chains = len(df_numeric.columns)
    fig_height = max(3, 0.4 * n_chains)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    y_extent = [-0.5, n_chains - 0.5]
    x_min, x_max = df_pivot.index.min(), df_pivot.index.max()
    x_extent = [x_min, x_max]
    xlabel = f'{pivot_index_col.replace("_", " ").capitalize()}' # Label based on used index

    im = ax.imshow(df_numeric.T, aspect='auto', cmap=cmap, norm=norm, interpolation='nearest', extent=x_extent + y_extent, origin='lower')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Chain') # Standardized name
    ax.set_yticks(np.arange(n_chains)); ax.set_yticklabels(df_numeric.columns)
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1], orientation='vertical', fraction=0.05, pad=0.04)
    cbar.ax.set_yticklabels(['Closed', 'Open'], fontsize='small')
    ax.grid(False)
    plt.tight_layout()

    plot_filename = "dw_gate_state_heatmap.png"
    plot_full_path = os.path.join(output_dir, plot_filename)
    save_plot(fig, plot_full_path)

    rel_path = os.path.relpath(plot_full_path, run_dir)
    # Register the plot product
    register_product(db_conn, module_name, "png", "plot", rel_path,
                     subcategory="state_heatmap", # Consistent subcategory
                     description="Heatmap of DW-Gate debounced states per chain over time.")
    return rel_path

def _plot_duration_distributions(
    events_df: pd.DataFrame, # Expects DF loaded from events CSV
    output_dir: str,
    run_dir: str, # Needed for registering relative path
    db_conn: sqlite3.Connection, # Needed for registration
    module_name: str # Name of the *visualization* module
) -> Optional[str]:
    """
    Plots distributions of event durations using combined plots. Registers product.
    Internal plotting logic moved to helper _plot_duration_panel.
    Uses standardized column names from events_df.
    """
    # Use standardized names for check
    required_cols = ['Chain', 'State', 'Duration (ns)']
    if events_df is None or events_df.empty or not all(c in events_df.columns for c in required_cols):
        logger.warning(f"Skipping duration distribution plot: Events data invalid or missing required columns ({required_cols}).")
        return None

    logger.info("Plotting event duration distributions...")
    # Use standardized names for query/access
    open_events = events_df.query(f"State == '{OPEN_STATE}'")
    closed_events = events_df.query(f"State == '{CLOSED_STATE}'")
    chains = sorted(events_df['Chain'].unique())
    if not chains: logger.warning("Skipping duration distribution plot: No chains found."); return None

    color_maps = get_chain_color_maps(chains)
    chain_map_pastel = color_maps['pastel']
    chain_map_bright = color_maps['bright']

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=False)

    # Pass standardized names to helper if needed, or let helper use them directly
    _plot_duration_panel(axes[0], open_events, chains, OPEN_STATE, chain_map_pastel, chain_map_bright, 'Open State Durations')
    _plot_duration_panel(axes[1], closed_events, chains, CLOSED_STATE, chain_map_pastel, chain_map_bright, 'Closed State Durations')

    axes[0].set_ylabel('Duration (ns)') # Standardized name

    legend_elements = [
        Patch(facecolor='grey', edgecolor='black', alpha=0.6, label='Distribution (Violin)'),
        Line2D([0], [0], color='black', lw=2, label='Median (Box)'),
        Line2D([0], [0], marker='o', color='w', label='Mean', markerfacecolor='red', markersize=8, markeredgecolor='black'),
        Line2D([0], [0], marker='.', color='grey', label='Individual Events', markersize=8, linestyle='None')
    ]
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=len(legend_elements), frameon=False)

    plt.tight_layout(rect=[0, 0.06, 1, 0.98])

    plot_filename = "dw_gate_duration_distribution.png"
    plot_full_path = os.path.join(output_dir, plot_filename)
    save_plot(fig, plot_full_path)

    rel_path = os.path.relpath(plot_full_path, run_dir)
    # Register the plot product
    register_product(db_conn, module_name, "png", "plot", rel_path,
                     subcategory="duration_distributions", # Consistent subcategory
                     description="Distribution of DW-Gate open and closed state durations.")
    return rel_path

def _plot_duration_panel(ax, data, chains, state_name, chain_map_pastel, chain_map_bright, plot_title):
    """
    Internal helper to plot duration distribution for one state (Open/Closed).
    Uses standardized column names 'Chain', 'Duration (ns)'.
    """
    # Use standardized names
    chain_col = 'Chain'
    duration_col = 'Duration (ns)'

    if data.empty:
        ax.text(0.5, 0.5, f"No {state_name.lower()} events found", ha='center', va='center', transform=ax.transAxes)
        ax.set_xlabel(chain_col) # Use standardized name
        ax.set_ylabel('')
        ax.set_title(plot_title) # Add title for clarity in multi-panel fig
        ax.grid(False); ax.spines['right'].set_visible(False); ax.spines['top'].set_visible(False)
        return

    try:
        sns.violinplot(ax=ax, data=data, x=chain_col, y=duration_col, hue=chain_col, # Standardized names
                       order=chains, hue_order=chains, palette=chain_map_pastel, inner=None,
                       density_norm='width', linewidth=1.5, saturation=0.7, legend=False)
    except Exception as e: logger.error(f"Violin plot failed for {state_name}: {e}")

    try:
        sns.boxplot(ax=ax, data=data, x=chain_col, y=duration_col, hue=chain_col, # Standardized names
                    order=chains, hue_order=chains, palette=chain_map_bright,
                    width=0.2, boxprops={'zorder': 2, 'alpha': 0.8},
                    whiskerprops={'zorder': 2, 'alpha': 0.8, 'ls': '-'},
                    capprops={'zorder': 2, 'alpha': 0.8},
                    medianprops={'color': 'black', 'linewidth': 2, 'zorder': 3, 'alpha': 0.9},
                    showfliers=False, showcaps=True, legend=False)
    except Exception as e: logger.error(f"Box plot failed for {state_name}: {e}")

    try:
        plot_func = sns.stripplot if len(data) >= 5000 else sns.swarmplot # Choose based on points
        kwargs = {'size': 2, 'alpha': 0.4, 'jitter': 0.2} if len(data) >= 5000 else {'size': 3, 'alpha': 0.6}
        # Use standardized names
        plot_func(ax=ax, data=data, x=chain_col, y=duration_col, hue=chain_col,
                  order=chains, hue_order=chains, palette=chain_map_bright, legend=False, zorder=1, **kwargs)
    except Exception as e: logger.error(f"Point plot failed for {state_name}: {e}")

    try:
        # Use standardized names
        means = data.groupby(chain_col)[duration_col].mean()
        ax.scatter(x=range(len(chains)), y=[means.get(ch, np.nan) for ch in chains],
                   color='red', marker='o', s=50, zorder=4, label='Mean', edgecolors='white')
    except Exception as e: logger.error(f"Mean plotting failed for {state_name}: {e}")

    y_min, y_max = ax.get_ylim()
    ax.set_ylim(0, y_max * 1.05)
    ax.set_xlabel(chain_col); ax.set_ylabel('') # Use standardized name
    ax.set_title(plot_title) # Add title for clarity in multi-panel fig
    ax.grid(False); ax.spines['right'].set_visible(False); ax.spines['top'].set_visible(False)


# --- Main Visualization Orchestrator ---
def generate_dw_gate_plots(
    run_dir: str,
    db_conn: sqlite3.Connection
) -> Dict[str, Any]:
    """
    Generates all standard plots for the DW-Gate analysis module.
    Retrieves necessary data file paths and metrics from the database.
    Loads statistics DataFrames and KDE data instead of recomputing.

    Args:
        run_dir: Path to the specific run directory.
        db_conn: Active database connection.

    Returns:
        Dictionary containing status and paths to generated plots.
    """
    module_name = "dw_gate_analysis_visualization" # Use the visualization module name
    start_time = time.time()
    register_module(db_conn, module_name, status='running')
    logger_local = setup_system_logger(run_dir)
    if logger_local is None: logger_local = logger # Fallback

    results: Dict[str, Any] = {'status': 'failed', 'plots': {}, 'error': None}
    output_dir = os.path.join(run_dir, "dw_gate_analysis") # Plots saved in module's dir
    os.makedirs(output_dir, exist_ok=True)

    # --- Check computation status ---
    comp_status = get_module_status(db_conn, "dw_gate_analysis")
    if comp_status != 'success':
        results['status'] = 'skipped'
        results['error'] = f"Skipping visualization: Computation status was '{comp_status}'."
        logger_local.warning(results['error'])
        update_module_status(db_conn, module_name, 'skipped', error_message=results['error'])
        return results

    # --- Retrieve Data File Paths ---
    required_files = {
        "dw_events": None,
        "debounced_states": None,
        "raw_dw_distances": None,
        "dw_probabilities": None,
        "kde_plot_data": None # <<< ADDED: Path for KDE data JSON >>>
    }
    found_all_paths = True
    for subcat in required_files:
        rel_path = get_product_path(db_conn, ('json' if subcat=='kde_plot_data' else 'csv'), 'data', subcat, 'dw_gate_analysis')
        if rel_path:
            required_files[subcat] = os.path.join(run_dir, rel_path)
        else:
            results['error'] = f"Required data path ('{subcat}') not found in database."
            logger_local.error(results['error'])
            found_all_paths = False
            break # Stop if any essential path is missing

    if not found_all_paths:
        update_module_status(db_conn, module_name, 'failed', error_message=results['error'])
        return results

    # --- Retrieve metrics from DB ---
    all_metrics = get_all_metrics(db_conn)
    closed_ref_metric = all_metrics.get('DW_RefDist_Closed_Used', {})
    open_ref_metric = all_metrics.get('DW_RefDist_Open_Used', {})
    closed_ref_dist = closed_ref_metric.get('value', DEFAULT_CLOSED_REF)
    open_ref_dist = open_ref_metric.get('value', DEFAULT_OPEN_REF)
    dist_thresh_metric = all_metrics.get('DEFAULT_CUTOFF', {}) # Fetch general cutoff if needed for plot
    distance_threshold = dist_thresh_metric.get('value', DEFAULT_DISTANCE_THRESHOLD)

    # --- Load Data ---
    try:
        events_df = pd.read_csv(required_files["dw_events"])
        df_states_long_debounced = pd.read_csv(required_files["debounced_states"])
        df_distances_wide = pd.read_csv(required_files["raw_dw_distances"])
        probability_df = pd.read_csv(required_files["dw_probabilities"])
        # <<< ADDED: Load KDE data JSON >>>
        with open(required_files["kde_plot_data"], 'r') as f_kde:
            kde_plot_data = json.load(f_kde)
        # <<< END ADDED >>>

    except FileNotFoundError as fnf_err:
        results['error'] = f"Failed to load data file: {fnf_err}. Check computation step outputs."
        logger_local.error(results['error'], exc_info=True)
        update_module_status(db_conn, module_name, 'failed', error_message=results['error'])
        return results
    except Exception as e:
        results['error'] = f"Failed to load data for plotting: {e}"
        logger_local.error(results['error'], exc_info=True)
        update_module_status(db_conn, module_name, 'failed', error_message=results['error'])
        return results

    # --- Generate Plots ---
    plots_generated = 0
    plots_failed = 0

    # <<< ADDED: Generate KDE Distribution Plot >>>
    kde_dist_path = _plot_kde_distribution(
        kde_plot_data=kde_plot_data,
        distance_threshold=distance_threshold, # Pass the general cutoff value
        output_dir=output_dir,
        run_dir=run_dir,
        db_conn=db_conn,
        module_name=module_name # Pass visualization module name
    )
    if kde_dist_path: plots_generated += 1; results['plots']['dw_distance_distribution'] = kde_dist_path # Use template key
    else: plots_failed += 1
    # <<< END ADDED >>>

    # Plot: Distance vs State
    dist_state_path = _plot_distance_vs_state(
        df_states_wide=df_distances_wide, # Raw distances needed
        events_df=events_df, # Use events for final state bars
        output_dir=output_dir, run_dir=run_dir, db_conn=db_conn, module_name=module_name,
        closed_ref_dist=closed_ref_dist, open_ref_dist=open_ref_dist,
    )
    if dist_state_path: plots_generated += 1; results['plots']['distance_vs_state'] = dist_state_path
    else: plots_failed += 1

    # Plot: Open Probability
    prob_path = _plot_open_probability(probability_df, output_dir, run_dir, db_conn, module_name)
    if prob_path: plots_generated += 1; results['plots']['open_probability'] = prob_path
    else: plots_failed += 1

    # Plot: State Heatmap
    heatmap_path = _plot_state_heatmap(
        df_states_long=df_states_long_debounced, # Use long format with debounced states
        output_dir=output_dir, run_dir=run_dir, db_conn=db_conn, module_name=module_name,
        state_col='state' # Pass internal column name expected by function
    )
    if heatmap_path: plots_generated += 1; results['plots']['state_heatmap'] = heatmap_path
    else: plots_failed += 1

    # Plot: Duration Distributions
    duration_path = _plot_duration_distributions(events_df, output_dir, run_dir, db_conn, module_name)
    if duration_path: plots_generated += 1; results['plots']['duration_distributions'] = duration_path
    else: plots_failed += 1

    # --- Finalize ---
    exec_time = time.time() - start_time
    # Check if ALL expected plots were generated
    expected_plots_count = 5 # KDE, DistVsState, Prob, Heatmap, Duration
    final_status = 'success' if plots_failed == 0 and plots_generated >= expected_plots_count else ('failed' if plots_failed > 0 else 'skipped')
    error_msg = f"{plots_failed} plot(s) failed to generate." if plots_failed > 0 else None
    if final_status == 'skipped': error_msg = "No plots were generated successfully."

    update_module_status(db_conn, module_name, final_status, execution_time=exec_time, error_message=error_msg)
    logger_local.info(f"--- DW-Gate Visualization finished in {exec_time:.2f} seconds (Status: {final_status}) ---")
    results['status'] = final_status
    if error_msg: results['error'] = error_msg

    return results