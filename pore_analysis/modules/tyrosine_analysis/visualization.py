# filename: pore_analysis/modules/tyrosine_analysis/visualization.py
"""
Visualization functions for SF Tyrosine rotamer analysis using HMM results.
Generates plots based on data stored in the database.
Includes separate plots for Chi1 and Chi2 dihedrals with HMM state background,
and plots for Tyr-Thr H-bond analysis if data is available. Fetches used
H-bond thresholds from database metrics for accurate plotting.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from typing import Dict, Optional, List, Any, Tuple
import time

# Import from core modules
try:
    from pore_analysis.core.plotting_style import STYLE, setup_style
    from pore_analysis.core.database import (
        register_module, update_module_status, get_product_path, register_product,
        get_module_status, get_metric_value # <<< Import get_metric_value >>>
    )
    from pore_analysis.core.logging import setup_system_logger
    # Import config values needed for plotting/annotations - KEEP DEFAULTS for fallback
    from pore_analysis.core.config import (
        TYR_HMM_STATE_ORDER,
        TYR_THR_DEFAULT_FORMED_REF_DIST,
        TYR_THR_DEFAULT_BROKEN_REF_DIST
    )
except ImportError as e:
    print(f"Error importing dependency modules in tyrosine_analysis/visualization.py: {e}")
    raise

logger = logging.getLogger(__name__)

# Apply standard plotting style
setup_style()
plt.switch_backend('agg') # Ensure non-interactive backend

# Define rotamer state colors (consistent with computation)
ROTAMER_COLORS = {
    'mt': '#a6cee3', 'mm': '#1f78b4', 'mp': '#b2df8a',
    'pt': '#33a02c', 'pm': '#fb9a99', 'pp': '#e31a1c',
    'tt': '#fdbf6f', 'tp': '#ff7f00', 'tm': '#cab2d6',
    'Outside': '#f0f0f0', 'Error': '#d9d9d9', 'Unknown': '#bdbdbd'
}
DEFAULT_ROTAMER_COLOR = '#ffffff'

# --- Plotting Helper Functions (_plot_chi1_dihedrals_stacked, _plot_chi2_dihedrals_stacked, _plot_tyrosine_scatter, _plot_tyrosine_population remain unchanged) ---

def _plot_chi1_dihedrals_stacked(df_raw_dihedrals: pd.DataFrame, df_hmm_states: pd.DataFrame, output_dir: str, run_dir: str, db_conn: sqlite3.Connection, module_name: str) -> Optional[str]:
    if 'Time (ns)' not in df_raw_dihedrals.columns or 'Time (ns)' not in df_hmm_states.columns:
        logger.error("Missing 'Time (ns)' column in input dataframes for Chi1 plot.")
        return None
    # Ensure dataframes are aligned by time (should be if generated correctly)
    if not np.allclose(df_raw_dihedrals['Time (ns)'], df_hmm_states['Time (ns)']):
        logger.error("Time points mismatch between raw dihedral and HMM state dataframes for Chi1 plot.")
        return None
    time_points = df_raw_dihedrals['Time (ns)'].values

    chains = sorted(list(set(c.split('_')[0] for c in df_raw_dihedrals.columns if c.endswith('_Chi1'))))
    if not chains:
        logger.warning("No chain data found in raw dihedral CSV for Chi1 plot.")
        return None
    n_chains = len(chains)

    try:
        fig, axes = plt.subplots(n_chains, 1, figsize=(12, 2.5 * n_chains), sharex=True, squeeze=False)
        axes = axes.flatten() # Now axes is always a 1D array
        chain_color_map = {chain: STYLE['bright_colors'].get(chain[-1], 'grey') for chain in chains}

        for i, chain in enumerate(chains):
            ax = axes[i]
            chi1_col = f'{chain}_Chi1'
            hmm_state_col = f'{chain}_State' # Assumes this column name from HMM save function

            if chi1_col not in df_raw_dihedrals or hmm_state_col not in df_hmm_states:
                 logger.warning(f"Missing Chi1 raw data or HMM state column for chain {chain}. Skipping Chi1 plot row.")
                 ax.text(0.5, 0.5, f"Data missing for {chain}", ha='center', va='center', transform=ax.transAxes)
                 continue

            chi1 = df_raw_dihedrals[chi1_col].values
            hmm_states = df_hmm_states[hmm_state_col].values

            # Background Shading based on HMM states
            current_state = None
            block_start_idx = 0
            for k in range(len(hmm_states)):
                state = hmm_states[k]
                if state != current_state:
                    if current_state is not None and block_start_idx < k : # Ensure block has width
                        color = ROTAMER_COLORS.get(current_state, DEFAULT_ROTAMER_COLOR)
                        try: ax.axvspan(time_points[block_start_idx], time_points[k], facecolor=color, alpha=0.20, zorder=0, edgecolor='none')
                        except IndexError: logger.warning(f"IndexError during axvspan for chain {chain}, state {current_state}. Skipping span.")
                    current_state = state
                    block_start_idx = k
            # Fill the last block
            if current_state is not None and block_start_idx < len(time_points):
                 color = ROTAMER_COLORS.get(current_state, DEFAULT_ROTAMER_COLOR)
                 try: ax.axvspan(time_points[block_start_idx], time_points[-1], facecolor=color, alpha=0.20, zorder=0, edgecolor='none')
                 except IndexError: logger.warning(f"IndexError during final axvspan for chain {chain}, state {current_state}. Skipping span.")

            valid_chi1_mask = np.isfinite(chi1)
            ax.plot(time_points[valid_chi1_mask], chi1[valid_chi1_mask],
                    label='Chi1', color=chain_color_map[chain], linewidth=STYLE['line_width']*0.9, zorder=1)

            ax.set_ylim(-181, 181); ax.set_yticks(np.arange(-180, 181, 60))
            ax.set_ylabel(f"{chain}\nChi1 Angle (°)", fontsize=STYLE['font_sizes']['axis_label']*0.9)
            ax.tick_params(axis='y', labelsize=STYLE['font_sizes']['tick_label'])
            ax.grid(axis='y', linestyle=STYLE['grid']['linestyle'], alpha=STYLE['grid']['alpha'], color=STYLE['grid']['color'], zorder=0)
            ax.spines['right'].set_visible(False); ax.spines['top'].set_visible(False)
            if i < n_chains - 1: plt.setp(ax.get_xticklabels(), visible=False); ax.tick_params(axis='x', bottom=False)

        axes[-1].set_xlabel('Time (ns)', fontsize=STYLE['font_sizes']['axis_label'])
        axes[-1].tick_params(axis='x', labelsize=STYLE['font_sizes']['tick_label'])
        fig.align_ylabels(axes)
        plt.tight_layout(); plt.subplots_adjust(hspace=0.15)

        plot_filename = "SF_Tyrosine_Chi1_Dihedrals_HMM.png"
        plot_filepath = os.path.join(output_dir, plot_filename)
        rel_path = os.path.relpath(plot_filepath, run_dir)
        fig.savefig(plot_filepath, dpi=150)
        plt.close(fig)
        logger.info(f"Saved stacked SF Tyrosine Chi1 (HMM states) plot to {plot_filepath}")
        register_product(db_conn, module_name, "png", "plot", rel_path,
                         subcategory="dihedrals_chi1",
                         description="Stacked time series of SF Tyr Chi1 angles per chain with HMM rotamer state background.")
        return rel_path
    except Exception as e:
        logger.error(f"Failed to generate stacked Chi1 HMM dihedral plot: {e}", exc_info=True)
        if 'fig' in locals() and plt.fignum_exists(fig.number): plt.close(fig)
        return None

def _plot_chi2_dihedrals_stacked(df_raw_dihedrals: pd.DataFrame, df_hmm_states: pd.DataFrame, output_dir: str, run_dir: str, db_conn: sqlite3.Connection, module_name: str) -> Optional[str]:
    if 'Time (ns)' not in df_raw_dihedrals.columns or 'Time (ns)' not in df_hmm_states.columns:
        logger.error("Missing 'Time (ns)' column in input dataframes for Chi2 plot.")
        return None
    if not np.allclose(df_raw_dihedrals['Time (ns)'], df_hmm_states['Time (ns)']):
        logger.error("Time points mismatch between raw dihedral and HMM state dataframes for Chi2 plot.")
        return None
    time_points = df_raw_dihedrals['Time (ns)'].values

    chains = sorted(list(set(c.split('_')[0] for c in df_raw_dihedrals.columns if c.endswith('_Chi2'))))
    if not chains: logger.warning("No chain data found in raw dihedral CSV for Chi2 plot."); return None
    n_chains = len(chains)

    try:
        fig, axes = plt.subplots(n_chains, 1, figsize=(12, 2.5 * n_chains), sharex=True, squeeze=False)
        axes = axes.flatten()
        chain_color_map = {chain: STYLE['bright_colors'].get(chain[-1], 'grey') for chain in chains}

        for i, chain in enumerate(chains):
            ax = axes[i]
            chi2_col = f'{chain}_Chi2'
            hmm_state_col = f'{chain}_State'

            if chi2_col not in df_raw_dihedrals or hmm_state_col not in df_hmm_states:
                 logger.warning(f"Missing Chi2 raw data or HMM state column for chain {chain}. Skipping Chi2 plot row.")
                 ax.text(0.5, 0.5, f"Data missing for {chain}", ha='center', va='center', transform=ax.transAxes)
                 continue

            chi2 = df_raw_dihedrals[chi2_col].values
            hmm_states = df_hmm_states[hmm_state_col].values

            current_state = None; block_start_idx = 0
            for k in range(len(hmm_states)):
                state = hmm_states[k]
                if state != current_state:
                    if current_state is not None and block_start_idx < k :
                        color = ROTAMER_COLORS.get(current_state, DEFAULT_ROTAMER_COLOR)
                        try: ax.axvspan(time_points[block_start_idx], time_points[k], facecolor=color, alpha=0.20, zorder=0, edgecolor='none')
                        except IndexError: logger.warning(f"IndexError during axvspan for chain {chain}, state {current_state} (Chi2). Skipping span.")
                    current_state = state; block_start_idx = k
            if current_state is not None and block_start_idx < len(time_points):
                 color = ROTAMER_COLORS.get(current_state, DEFAULT_ROTAMER_COLOR)
                 try: ax.axvspan(time_points[block_start_idx], time_points[-1], facecolor=color, alpha=0.20, zorder=0, edgecolor='none')
                 except IndexError: logger.warning(f"IndexError during final axvspan for chain {chain}, state {current_state} (Chi2). Skipping span.")

            valid_chi2_mask = np.isfinite(chi2)
            ax.plot(time_points[valid_chi2_mask], chi2[valid_chi2_mask], label='Chi2', color=chain_color_map[chain], linestyle='--', linewidth=STYLE['line_width']*0.7, alpha=0.9, zorder=1)

            ax.set_ylim(-181, 181); ax.set_yticks(np.arange(-180, 181, 60))
            ax.set_ylabel(f"{chain}\nChi2 Angle (°)", fontsize=STYLE['font_sizes']['axis_label']*0.9)
            ax.tick_params(axis='y', labelsize=STYLE['font_sizes']['tick_label'])
            ax.grid(axis='y', linestyle=STYLE['grid']['linestyle'], alpha=STYLE['grid']['alpha'], color=STYLE['grid']['color'], zorder=0)
            ax.spines['right'].set_visible(False); ax.spines['top'].set_visible(False)
            if i < n_chains - 1: plt.setp(ax.get_xticklabels(), visible=False); ax.tick_params(axis='x', bottom=False)

        axes[-1].set_xlabel('Time (ns)', fontsize=STYLE['font_sizes']['axis_label'])
        axes[-1].tick_params(axis='x', labelsize=STYLE['font_sizes']['tick_label'])
        fig.align_ylabels(axes)
        plt.tight_layout(); plt.subplots_adjust(hspace=0.15)

        plot_filename = "SF_Tyrosine_Chi2_Dihedrals_HMM.png"
        plot_filepath = os.path.join(output_dir, plot_filename)
        rel_path = os.path.relpath(plot_filepath, run_dir)
        fig.savefig(plot_filepath, dpi=150)
        plt.close(fig)
        logger.info(f"Saved stacked SF Tyrosine Chi2 (HMM states) plot to {plot_filepath}")
        register_product(db_conn, module_name, "png", "plot", rel_path,
                         subcategory="dihedrals_chi2",
                         description="Stacked time series of SF Tyr Chi2 angles per chain with HMM rotamer state background.")
        return rel_path
    except Exception as e:
        logger.error(f"Failed to generate stacked Chi2 HMM dihedral plot: {e}", exc_info=True)
        if 'fig' in locals() and plt.fignum_exists(fig.number): plt.close(fig)
        return None

def _plot_tyrosine_scatter(df_raw_dihedrals: pd.DataFrame, output_dir: str, run_dir: str, db_conn: sqlite3.Connection, module_name: str) -> Optional[str]:
    chains = sorted(list(set(c.split('_')[0] for c in df_raw_dihedrals.columns if c.endswith('_Chi1'))))
    if not chains: return None
    all_chi1 = np.concatenate([df_raw_dihedrals[f'{c}_Chi1'].values for c in chains if f'{c}_Chi1' in df_raw_dihedrals])
    all_chi2 = np.concatenate([df_raw_dihedrals[f'{c}_Chi2'].values for c in chains if f'{c}_Chi2' in df_raw_dihedrals])
    valid_idx = np.isfinite(all_chi1) & np.isfinite(all_chi2)
    if not np.any(valid_idx): logger.warning("No valid Chi1/Chi2 data points for scatter plot."); return None
    try:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(all_chi1[valid_idx], all_chi2[valid_idx], alpha=0.1, s=5, color=STYLE['bright_colors']['A'])
        ax.set_xlabel('Chi1 Angle (°)')
        ax.set_ylabel('Chi2 Angle (°)')
        ax.set_xlim(-181, 181); ax.set_ylim(-181, 181)
        ax.grid(True, alpha=STYLE['grid']['alpha'], color=STYLE['grid']['color'], linestyle=STYLE['grid']['linestyle'])
        ax.set_xticks(np.arange(-180, 181, 60)); ax.set_yticks(np.arange(-180, 181, 60))
        ax.axhline(0, color='grey', lw=0.5, linestyle=':'); ax.axvline(0, color='grey', lw=0.5, linestyle=':')
        plt.tight_layout()
        plot_path = os.path.join(output_dir, "SF_Tyrosine_Rotamer_Scatter.png")
        rel_path = os.path.relpath(plot_path, run_dir)
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        logger.info(f"Saved rotamer scatter plot to {plot_path}")
        register_product(db_conn, module_name, "png", "plot", rel_path, subcategory="rotamer_scatter", description="SF Tyr Chi1 vs Chi2 angles scatter plot.")
        return rel_path
    except Exception as e:
        logger.error(f"Failed to save rotamer scatter plot: {e}")
        if 'fig' in locals() and plt.fignum_exists(fig.number): plt.close(fig)
        return None

def _plot_tyrosine_population(df_hmm_states: pd.DataFrame, output_dir: str, run_dir: str, db_conn: sqlite3.Connection, module_name: str) -> Optional[str]:
    chains = sorted(list(set(c.split('_')[0] for c in df_hmm_states.columns if c.endswith('_State'))))
    if not chains: return None
    all_hmm_states = []
    for c in chains:
        state_col = f'{c}_State'
        if state_col in df_hmm_states:
            valid_mask = ~df_hmm_states[state_col].isin(['Outside', 'Error', 'Unknown', np.nan])
            all_hmm_states.extend(df_hmm_states.loc[valid_mask, state_col].values)
    if not all_hmm_states: logger.warning("No valid HMM rotamer states found for population plot."); return None
    try:
        state_counts = pd.Series(all_hmm_states).value_counts(normalize=True) * 100
        possible_states = TYR_HMM_STATE_ORDER
        state_counts = state_counts.reindex(possible_states, fill_value=0)
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = [ROTAMER_COLORS.get(state, '#cccccc') for state in state_counts.index]
        state_counts.plot(kind='bar', ax=ax, color=colors)
        ax.set_xlabel('HMM Rotamer State (Chi1, Chi2)')
        ax.set_ylabel('Population (%)')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)
        ax.set_ylim(bottom=0)
        plt.tight_layout()
        plot_path = os.path.join(output_dir, "SF_Tyrosine_Rotamer_Population_HMM.png")
        rel_path = os.path.relpath(plot_path, run_dir)
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        logger.info(f"Saved HMM rotamer population plot to {plot_path}")
        register_product(db_conn, module_name, "png", "plot", rel_path,
                         subcategory="rotamer_population",
                         description="SF Tyr HMM rotamer state population.")
        return rel_path
    except Exception as e:
        logger.error(f"Failed to save HMM rotamer population plot: {e}")
        if 'fig' in locals() and plt.fignum_exists(fig.number): plt.close(fig)
        return None

# --- Tyr-Thr Hydrogen Bond Visualization Functions --- #

def _plot_tyr_thr_hbond_distances(
    df_distances: pd.DataFrame,
    output_dir: str,
    run_dir: str,
    db_conn: sqlite3.Connection,
    module_name: str
) -> Optional[str]:
    """Plot the Tyr-Thr hydrogen bond distances over time using multiple stacked subplots."""
    formed_ref_used = get_metric_value(db_conn, 'TyrThr_RefDist_Formed_Used')
    if formed_ref_used is None:
        logger.warning("Could not retrieve TyrThr_RefDist_Formed_Used metric, using default from config for plot.")
        formed_ref_used = TYR_THR_DEFAULT_FORMED_REF_DIST
    broken_ref_used = get_metric_value(db_conn, 'TyrThr_RefDist_Broken_Used')
    if broken_ref_used is None:
        logger.warning("Could not retrieve TyrThr_RefDist_Broken_Used metric, using default from config for plot.")
        broken_ref_used = TYR_THR_DEFAULT_BROKEN_REF_DIST

    if 'Time (ns)' not in df_distances.columns:
        logger.error("Missing 'Time (ns)' in H-bond distances DataFrame.")
        return None
        
    pairs = [col for col in df_distances.columns if col != 'Time (ns)']
    if not pairs:
        logger.warning("No pair data columns found in H-bond distances DataFrame.")
        return None
        
    n_pairs = len(pairs)
    
    try:
        # Create figure with stacked subplots (one per chain pair)
        fig, axes = plt.subplots(n_pairs, 1, figsize=(12, 2.5 * n_pairs), sharex=True, squeeze=False)
        axes = axes.flatten()  # Ensure axes is a 1D array even if n_pairs is 1
        
        # Get the maximum distance value for consistent y-axis scaling
        # Initialize with reference values to ensure they are visible
        all_dist_values = df_distances[pairs].values.flatten()
        all_dist_values = all_dist_values[~np.isnan(all_dist_values)] # Remove NaNs

        if len(all_dist_values) > 0:
            min_data_val = np.min(all_dist_values)
            max_data_val = np.max(all_dist_values)
            ymin = min(min_data_val * 0.95, formed_ref_used * 0.9, broken_ref_used * 0.9)
            ymax = max(max_data_val * 1.05, formed_ref_used * 1.1, broken_ref_used * 1.1)
        else: # Fallback if no valid distance data
            ymin = min(formed_ref_used, broken_ref_used) * 0.8
            ymax = max(formed_ref_used, broken_ref_used) * 1.2
        
        # Determine colors for each pair (assuming pairs are like 'PROA_PROB')
        # Use the first chain ID (PROA from PROA_PROB) for color mapping from STYLE
        pair_colors = {}
        for i, pair_label in enumerate(pairs):
            first_chain_char = pair_label.split('_')[0][-1] # e.g., A from PROA
            pair_colors[pair_label] = STYLE['bright_colors'].get(first_chain_char, f'C{i}')

        # Plot each pair in its own subplot
        for i, (pair, ax) in enumerate(zip(pairs, axes)):
            time_points = df_distances['Time (ns)'].values
            pair_distances = df_distances[pair].values
            
            valid_mask = ~np.isnan(pair_distances)
            
            # Plot the distance time series
            if np.any(valid_mask):
                ax.plot(time_points[valid_mask], pair_distances[valid_mask], 
                       color=pair_colors[pair], linewidth=STYLE['line_width']*0.9, zorder=2)
            else:
                ax.text(0.5, 0.5, "No data available", transform=ax.transAxes, ha="center", va="center")

            # Add reference lines for formed/broken thresholds
            ax.axhline(y=formed_ref_used, color='green', linestyle='--', alpha=0.7, 
                      label=f'Formed ({formed_ref_used:.2f} Å)' if i == 0 else "_nolegend_", zorder=1)
            ax.axhline(y=broken_ref_used, color='red', linestyle='--', alpha=0.7, 
                      label=f'Broken ({broken_ref_used:.2f} Å)' if i == 0 else "_nolegend_", zorder=1)
            
            # Add shaded region for uncertain state
            ax.axhspan(formed_ref_used, broken_ref_used, color='gray', alpha=0.1, 
                      label='Uncertain region' if i == 0 else "_nolegend_", zorder=0)
            
            # Set y-axis label and configure grid
            ax.set_ylabel(f"{pair.replace('_', ' - ')}\nDistance (Å)", fontsize=STYLE['font_sizes']['axis_label']*0.9)
            ax.set_ylim(ymin, ymax) # Apply consistent y-limits
            ax.grid(True, alpha=STYLE['grid']['alpha'], color=STYLE['grid']['color'], linestyle=STYLE['grid']['linestyle'], zorder=0)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.tick_params(axis='y', labelsize=STYLE['font_sizes']['tick_label'])
            
            # Add legend only to the first subplot
            if i == 0:
                ax.legend(loc='upper right', fontsize=STYLE['font_sizes']['annotation']*0.8)
            
            # Hide x-labels for all but the last subplot
            if i < n_pairs - 1:
                plt.setp(ax.get_xticklabels(), visible=False)
                ax.tick_params(axis='x', bottom=False)
        
        # Set x-axis label on the bottom subplot
        axes[-1].set_xlabel('Time (ns)', fontsize=STYLE['font_sizes']['axis_label'])
        axes[-1].tick_params(axis='x', labelsize=STYLE['font_sizes']['tick_label'])
        
        # Align y-labels and adjust layout
        fig.align_ylabels(axes) # Align y-axis labels for better appearance
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.15) # Adjust vertical spacing if needed
        
        # Save the plot
        plot_filename = "tyr_thr_hbond_distances_stacked.png" # New filename to reflect change
        plot_path = os.path.join(output_dir, plot_filename)
        rel_path = os.path.relpath(plot_path, run_dir)
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        
        # Register the new plot in the database
        # Important: The subcategory should match what's in plots_dict.json for HTML report
        register_product(
            db_conn, 
            module_name, 
            "png", 
            "plot", 
            rel_path, 
            subcategory="tyr_thr_hbond_distances", # Keep same subcategory to replace old plot
            description="Stacked time series of Tyr445-Thr hydrogen bond distances per chain pair."
        )
        
        logger.info(f"Saved stacked Tyr-Thr H-bond distances plot to {plot_path}")
        return rel_path
        
    except Exception as e:
        logger.error(f"Failed to generate stacked Tyr-Thr H-bond distances plot: {e}", exc_info=True)
        if 'fig' in locals() and plt.fignum_exists(fig.number):
            plt.close(fig)
        return None

def _plot_tyr_thr_hbond_states(
    df_states: pd.DataFrame,
    output_dir: str,
    run_dir: str,
    db_conn: sqlite3.Connection,
    module_name: str
) -> Optional[str]:
    """Plot the Tyr-Thr hydrogen bond states as a heatmap."""
    if 'Time (ns)' not in df_states.columns:
        logger.error("Missing 'Time (ns)' column in H-bond states dataframe")
        return None
    state_cols = [col for col in df_states.columns if col.endswith('_debounced')]
    if not state_cols:
        logger.warning("No debounced state columns found in H-bond states dataframe")
        return None
    pairs = [col.split('_debounced')[0] for col in state_cols]
    try:
        fig, axes_array = plt.subplots(len(pairs), 1, figsize=(12, 2 * len(pairs)), sharex=True, squeeze=False)

        # Correctly handle the axes object whether it's a single Axes or an array
        if len(pairs) == 1:
            axes = [axes_array[0, 0]] # Access the single Axes object
        else:
            axes = axes_array.flatten() # Flatten if multiple pairs

        state_colors = {0: 'green', 1: 'red', -1: 'gray'}
        for ax, pair, state_col in zip(axes, pairs, state_cols):
            states = df_states[state_col].values
            time = df_states['Time (ns)'].values
            current_state = None
            start_time = time[0]
            for i, state_val in enumerate(states): # Renamed 'state' to 'state_val' to avoid conflict
                if state_val != current_state:
                    if current_state is not None:
                        color = state_colors.get(current_state, 'gray')
                        end_time = time[i] if i < len(time) else time[-1]
                        ax.axhspan(0, 1, xmin=start_time / time[-1], xmax=end_time / time[-1], facecolor=color, alpha=0.4)
                    current_state = state_val
                    start_time = time[i]
            if current_state is not None:
                color = state_colors.get(current_state, 'gray')
                ax.axhspan(0, 1, xmin=start_time / time[-1], xmax=1.0, facecolor=color, alpha=0.4)
            ax.set_ylabel(pair, fontsize=STYLE['font_sizes']['axis_label'])
            ax.set_yticks([])
            legend_elements = [
                plt.Rectangle((0, 0), 1, 1, facecolor='green', alpha=0.4, label='H-bond formed'),
                plt.Rectangle((0, 0), 1, 1, facecolor='red', alpha=0.4, label='H-bond broken'),
                plt.Rectangle((0, 0), 1, 1, facecolor='gray', alpha=0.4, label='Uncertain')
            ]
            # Fix: use 'annotation' instead of 'legend' which doesn't exist in STYLE['font_sizes']
            ax.legend(handles=legend_elements, loc='upper right', fontsize=STYLE['font_sizes']['annotation'] * 0.8)
            ax.set_ylim(0, 1)
            ax.spines['left'].set_visible(False); ax.spines['right'].set_visible(False); ax.spines['top'].set_visible(False)
        axes[-1].set_xlabel('Time (ns)', fontsize=STYLE['font_sizes']['axis_label'])
        if len(time) > 0 : ax.set_xlim(time[0], time[-1]) # Check if time is not empty
        else: logger.warning("Time data is empty for H-bond states plot, cannot set xlim.")

        plt.tight_layout(); plt.subplots_adjust(top=0.92, hspace=0.2)
        plot_filename = "tyr_thr_hbond_states.png"
        plot_path = os.path.join(output_dir, plot_filename)
        rel_path = os.path.relpath(plot_path, run_dir)
        fig.savefig(plot_path, dpi=150); plt.close(fig)
        register_product(db_conn, module_name, "png", "plot", rel_path, subcategory="tyr_thr_hbond_states", description="Heatmap of Tyr445-Thr hydrogen bond states over time")
        logger.info(f"Saved Tyr-Thr H-bond states plot to {plot_path}"); return rel_path
    except Exception as e:
        logger.error(f"Failed to generate Tyr-Thr H-bond states plot: {e}", exc_info=True)
        if 'fig' in locals() and plt.fignum_exists(fig.number): plt.close(fig)
        return None

def _plot_tyr_thr_hbond_statistics(
    df_events: pd.DataFrame,
    output_dir: str,
    run_dir: str,
    db_conn: sqlite3.Connection,
    module_name: str
) -> Optional[str]:
    if df_events.empty: logger.warning("Empty events dataframe for H-bond statistics plot"); return None
    if not all(col in df_events.columns for col in ['Pair', 'State', 'Duration (ns)']): logger.error("Missing required columns in H-bond events dataframe"); return None
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        sns.boxplot(x='Pair', y='Duration (ns)', hue='State', data=df_events, palette={'H-bond-formed': 'green', 'H-bond-broken': 'red'}, ax=ax1)
        ax1.set_title('H-bond Event Durations by Pair', fontsize=STYLE['font_sizes']['title'])
        ax1.set_xlabel('Chain Pair', fontsize=STYLE['font_sizes']['axis_label'])
        ax1.set_ylabel('Duration (ns)', fontsize=STYLE['font_sizes']['axis_label'])
        ax1.legend(title='State', fontsize=STYLE['font_sizes']['legend']*0.8)
        event_counts = df_events.groupby(['Pair', 'State']).size().reset_index(name='Count')
        sns.barplot(x='Pair', y='Count', hue='State', data=event_counts, palette={'H-bond-formed': 'green', 'H-bond-broken': 'red'}, ax=ax2)
        ax2.set_title('H-bond Event Counts by Pair', fontsize=STYLE['font_sizes']['title'])
        ax2.set_xlabel('Chain Pair', fontsize=STYLE['font_sizes']['axis_label'])
        ax2.set_ylabel('Number of Events', fontsize=STYLE['font_sizes']['axis_label'])
        ax2.legend(title='State', fontsize=STYLE['font_sizes']['legend']*0.8)
        plt.tight_layout(); plt.subplots_adjust(top=0.85)
        plot_filename = "tyr_thr_hbond_statistics.png"
        plot_path = os.path.join(output_dir, plot_filename)
        rel_path = os.path.relpath(plot_path, run_dir)
        fig.savefig(plot_path, dpi=150); plt.close(fig)
        register_product(db_conn, module_name, "png", "plot", rel_path, subcategory="tyr_thr_hbond_statistics", description="Statistics of Tyr445-Thr hydrogen bond events")
        logger.info(f"Saved Tyr-Thr H-bond statistics plot to {plot_path}"); return rel_path
    except Exception as e:
        logger.error(f"Failed to generate Tyr-Thr H-bond statistics plot: {e}", exc_info=True)
        if 'fig' in locals() and plt.fignum_exists(fig.number): plt.close(fig)
        return None

# --- Main Visualization Function --- #
def generate_tyrosine_plots(
    run_dir: str,
    db_conn: sqlite3.Connection
) -> Dict[str, Any]:
    module_name = "tyrosine_analysis_visualization"
    start_time = time.time()
    register_module(db_conn, module_name, status='running')
    logger_local = setup_system_logger(run_dir)
    if logger_local is None: logger_local = logging.getLogger(__name__)

    results: Dict[str, Any] = {'status': 'failed', 'plots': {}, 'error': None}
    output_dir = os.path.join(run_dir, "tyrosine_analysis") # Plots saved in computation module's folder
    os.makedirs(output_dir, exist_ok=True)

    comp_status = get_module_status(db_conn, "tyrosine_analysis")
    if comp_status != 'success':
        results['status'] = 'skipped'; results['error'] = f"Skipping viz: Computation status was '{comp_status}'."
        logger_local.warning(results['error']); update_module_status(db_conn, module_name, 'skipped', error_message=results['error'])
        return results

    raw_dihedral_rel_path = get_product_path(db_conn, 'csv', 'data', 'raw_dihedrals', 'tyrosine_analysis')
    hmm_state_rel_path = get_product_path(db_conn, 'csv', 'data', 'hmm_state_path', 'tyrosine_analysis')
    if not raw_dihedral_rel_path or not hmm_state_rel_path:
        results['error'] = "Essential rotamer data paths not found."; logger_local.error(results['error'])
        update_module_status(db_conn, module_name, 'failed', error_message=results['error']); return results
    raw_dihedral_abs_path = os.path.join(run_dir, raw_dihedral_rel_path); hmm_state_abs_path = os.path.join(run_dir, hmm_state_rel_path)
    if not os.path.exists(raw_dihedral_abs_path): results['error'] = f"File not found: {raw_dihedral_abs_path}"; logger_local.error(results['error']); update_module_status(db_conn, module_name, 'failed', error_message=results['error']); return results
    if not os.path.exists(hmm_state_abs_path): results['error'] = f"File not found: {hmm_state_abs_path}"; logger_local.error(results['error']); update_module_status(db_conn, module_name, 'failed', error_message=results['error']); return results

    hbond_distances_rel_path = get_product_path(db_conn, 'csv', 'data', 'tyr_thr_hbond_distances', 'tyrosine_analysis')
    hbond_states_rel_path = get_product_path(db_conn, 'csv', 'data', 'tyr_thr_hbond_states', 'tyrosine_analysis')
    hbond_events_rel_path = get_product_path(db_conn, 'csv', 'data', 'tyr_thr_hbond_events', 'tyrosine_analysis')
    hbond_distances_abs_path = os.path.join(run_dir, hbond_distances_rel_path) if hbond_distances_rel_path else None
    hbond_states_abs_path = os.path.join(run_dir, hbond_states_rel_path) if hbond_states_rel_path else None
    hbond_events_abs_path = os.path.join(run_dir, hbond_events_rel_path) if hbond_events_rel_path else None

    try:
        df_raw_dihedrals = pd.read_csv(raw_dihedral_abs_path)
        df_hmm_states = pd.read_csv(hmm_state_abs_path)
        if len(df_raw_dihedrals) != len(df_hmm_states): raise ValueError("HMM/raw dihedral file length mismatch.")
    except Exception as e: results['error'] = f"Failed to load HMM data: {e}"; logger_local.error(results['error']); update_module_status(db_conn, module_name, 'failed', error_message=results['error']); return results

    df_hbond_distances, df_hbond_states, df_hbond_events = None, None, None
    if hbond_distances_abs_path and os.path.exists(hbond_distances_abs_path):
        try: df_hbond_distances = pd.read_csv(hbond_distances_abs_path); logger_local.info("Loaded H-bond distances")
        except Exception as e: logger_local.warning(f"Failed to load H-bond distances: {e}")
    else: logger_local.info("H-bond distances data not found or path invalid.")
    if hbond_states_abs_path and os.path.exists(hbond_states_abs_path):
        try: df_hbond_states = pd.read_csv(hbond_states_abs_path); logger_local.info("Loaded H-bond states")
        except Exception as e: logger_local.warning(f"Failed to load H-bond states: {e}")
    if hbond_events_abs_path and os.path.exists(hbond_events_abs_path):
        try: df_hbond_events = pd.read_csv(hbond_events_abs_path); logger_local.info("Loaded H-bond events")
        except Exception as e: logger_local.warning(f"Failed to load H-bond events: {e}")

    plots_generated = 0; plots_failed = 0; expected_rotamer_plots = 4; expected_hbond_plots = 0

    if (path := _plot_chi1_dihedrals_stacked(df_raw_dihedrals, df_hmm_states, output_dir, run_dir, db_conn, module_name)): plots_generated += 1; results['plots']['sf_tyrosine_chi1_dihedrals'] = path
    else: plots_failed += 1
    if (path := _plot_chi2_dihedrals_stacked(df_raw_dihedrals, df_hmm_states, output_dir, run_dir, db_conn, module_name)): plots_generated += 1; results['plots']['sf_tyrosine_chi2_dihedrals'] = path
    else: plots_failed += 1
    if (path := _plot_tyrosine_scatter(df_raw_dihedrals, output_dir, run_dir, db_conn, module_name)): plots_generated += 1; results['plots']['sf_tyrosine_rotamer_scatter'] = path
    else: plots_failed += 1
    if (path := _plot_tyrosine_population(df_hmm_states, output_dir, run_dir, db_conn, module_name)): plots_generated += 1; results['plots']['sf_tyrosine_rotamer_population'] = path
    else: plots_failed += 1

    if df_hbond_distances is not None and not df_hbond_distances.empty:
        expected_hbond_plots += 1; logger_local.info("Generating H-bond distances plot");
        if (path := _plot_tyr_thr_hbond_distances(df_hbond_distances, output_dir, run_dir, db_conn, module_name)): plots_generated += 1; results['plots']['tyr_thr_hbond_distances'] = path
        else: logger_local.warning("Failed H-bond distances plot generation")
    else: logger_local.warning("Skipping H-bond distances plot (no data)")

    if df_hbond_states is not None and not df_hbond_states.empty:
        expected_hbond_plots += 1; logger_local.info("Generating H-bond states plot");
        if (path := _plot_tyr_thr_hbond_states(df_hbond_states, output_dir, run_dir, db_conn, module_name)): plots_generated += 1; results['plots']['tyr_thr_hbond_states'] = path
        else: logger_local.warning("Failed H-bond states plot generation")
    else: logger_local.warning("Skipping H-bond states plot (no data)")

    if df_hbond_events is not None and not df_hbond_events.empty:
        expected_hbond_plots += 1; logger_local.info("Generating H-bond statistics plot");
        if (path := _plot_tyr_thr_hbond_statistics(df_hbond_events, output_dir, run_dir, db_conn, module_name)): plots_generated += 1; results['plots']['tyr_thr_hbond_statistics'] = path
        else: logger_local.warning("Failed H-bond statistics plot generation")
    else: logger_local.warning("Skipping H-bond statistics plot (no data)")

    exec_time = time.time() - start_time
    final_status = 'success' if plots_failed == 0 and plots_generated >= expected_rotamer_plots else 'failed'
    error_msg = f"{plots_failed} essential rotamer plot(s) failed." if plots_failed > 0 else None
    if final_status == 'failed' and plots_generated < expected_rotamer_plots: error_msg = f"Failed essential rotamer plots (Got {plots_generated-expected_hbond_plots}/{expected_rotamer_plots})."
    update_module_status(db_conn, module_name, final_status, execution_time=exec_time, error_message=error_msg); results['status'] = final_status;
    if error_msg: results['error'] = error_msg
    logger_local.info(f"--- Tyrosine Visualization finished in {exec_time:.2f} seconds (Status: {final_status}) ---")
    return results
