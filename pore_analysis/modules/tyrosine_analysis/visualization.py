# filename: pore_analysis/modules/tyrosine_analysis/visualization.py
"""
Visualization functions for SF Tyrosine rotamer analysis using HMM results.
Generates plots based on data stored in the database.
Includes separate plots for Chi1 and Chi2 dihedrals with HMM state background.
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
        register_module, update_module_status, get_product_path, register_product
    )
    from pore_analysis.core.logging import setup_system_logger
    # Import state order if needed for consistent coloring/ordering
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
    'Outside': '#f0f0f0', # Light grey for outside/NaN periods
    'Error': '#d9d9d9',  # Different grey for errors
    'Unknown': '#bdbdbd' # Fallback if state index is weird
}
DEFAULT_ROTAMER_COLOR = '#ffffff' # White/transparent for unassigned background

# --- Plotting Helper Functions --- #

def _plot_chi1_dihedrals_stacked(
    df_raw_dihedrals: pd.DataFrame,
    df_hmm_states: pd.DataFrame,
    output_dir: str,
    run_dir: str,
    db_conn: sqlite3.Connection,
    module_name: str
) -> Optional[str]:
    """
    Generates stacked Chi1 dihedral time series plot per chain
    with background shading based on HMM rotamer state.
    """
    if 'Time (ns)' not in df_raw_dihedrals.columns or 'Time (ns)' not in df_hmm_states.columns:
        logger.error("Missing 'Time (ns)' column in input dataframes for Chi1 plot.")
        return None
    # Ensure dataframes are aligned by time (should be if generated correctly)
    if not np.allclose(df_raw_dihedrals['Time (ns)'], df_hmm_states['Time (ns)']):
        logger.error("Time points mismatch between raw dihedral and HMM state dataframes.")
        # Attempt merge/reindex? For now, fail.
        return None
    time_points = df_raw_dihedrals['Time (ns)'].values

    chains = sorted(list(set(c.split('_')[0] for c in df_raw_dihedrals.columns if c.endswith('_Chi1'))))
    if not chains:
        logger.warning("No chain data found in raw dihedral CSV for Chi1 plot.")
        return None
    n_chains = len(chains)

    try:
        fig, axes = plt.subplots(n_chains, 1, figsize=(12, 2.5 * n_chains), sharex=True, squeeze=False)
        axes = axes.flatten()
        chain_color_map = {chain: STYLE['bright_colors'].get(chain[-1], 'grey') for chain in chains}

        for i, chain in enumerate(chains):
            ax = axes[i]
            chi1_col = f'{chain}_Chi1'
            hmm_state_col = f'{chain}_State' # Assumes this column name from HMM save function

            if chi1_col not in df_raw_dihedrals or hmm_state_col not in df_hmm_states:
                 logger.warning(f"Missing Chi1 raw data or HMM state column for chain {chain}. Skipping plot row.")
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
                        # Use try-except for time_points indexing as lengths might mismatch if NaNs caused issues earlier
                        try:
                             ax.axvspan(time_points[block_start_idx], time_points[k], facecolor=color, alpha=0.20, zorder=0, edgecolor='none')
                        except IndexError:
                             logger.warning(f"IndexError during axvspan for chain {chain}, state {current_state}. Skipping span.")
                    current_state = state
                    block_start_idx = k
            # Fill the last block
            if current_state is not None and block_start_idx < len(time_points):
                 color = ROTAMER_COLORS.get(current_state, DEFAULT_ROTAMER_COLOR)
                 try:
                      ax.axvspan(time_points[block_start_idx], time_points[-1], facecolor=color, alpha=0.20, zorder=0, edgecolor='none')
                 except IndexError:
                     logger.warning(f"IndexError during final axvspan for chain {chain}, state {current_state}. Skipping span.")


            # Plot Chi1 Line (only where HMM state is not 'Outside' or 'Error'?) - Plot all for now
            valid_chi1_mask = np.isfinite(chi1)
            ax.plot(time_points[valid_chi1_mask], chi1[valid_chi1_mask],
                    label='Chi1', color=chain_color_map[chain], linewidth=STYLE['line_width']*0.9, zorder=1)

            # Configure axes
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

        # Save Figure
        plot_filename = "SF_Tyrosine_Chi1_Dihedrals_HMM.png" # Updated filename
        plot_filepath = os.path.join(output_dir, plot_filename)
        rel_path = os.path.relpath(plot_filepath, run_dir)
        fig.savefig(plot_filepath, dpi=150)
        plt.close(fig)
        logger.info(f"Saved stacked SF Tyrosine Chi1 (HMM states) plot to {plot_filepath}")
        register_product(db_conn, module_name, "png", "plot", rel_path,
                         subcategory="dihedrals_chi1", # Keep subcategory consistent
                         description="Stacked time series of SF Tyr Chi1 angles per chain with HMM rotamer state background.")
        return rel_path

    except Exception as e:
        logger.error(f"Failed to generate stacked Chi1 HMM dihedral plot: {e}", exc_info=True)
        if 'fig' in locals() and plt.fignum_exists(fig.number): plt.close(fig)
        return None

def _plot_chi2_dihedrals_stacked(
    df_raw_dihedrals: pd.DataFrame,
    df_hmm_states: pd.DataFrame,
    output_dir: str,
    run_dir: str,
    db_conn: sqlite3.Connection,
    module_name: str
) -> Optional[str]:
    """
    Generates stacked Chi2 dihedral time series plot per chain
    with background shading based on HMM rotamer state.
    """
    if 'Time (ns)' not in df_raw_dihedrals.columns or 'Time (ns)' not in df_hmm_states.columns:
        logger.error("Missing 'Time (ns)' column in input dataframes for Chi2 plot.")
        return None
    if not np.allclose(df_raw_dihedrals['Time (ns)'], df_hmm_states['Time (ns)']):
        logger.error("Time points mismatch between raw dihedral and HMM state dataframes.")
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
                 logger.warning(f"Missing Chi2 raw data or HMM state column for chain {chain}. Skipping plot row.")
                 ax.text(0.5, 0.5, f"Data missing for {chain}", ha='center', va='center', transform=ax.transAxes)
                 continue

            chi2 = df_raw_dihedrals[chi2_col].values
            hmm_states = df_hmm_states[hmm_state_col].values

            # Background Shading based on HMM states
            current_state = None
            block_start_idx = 0
            for k in range(len(hmm_states)):
                state = hmm_states[k]
                if state != current_state:
                    if current_state is not None and block_start_idx < k :
                        color = ROTAMER_COLORS.get(current_state, DEFAULT_ROTAMER_COLOR)
                        try:
                             ax.axvspan(time_points[block_start_idx], time_points[k], facecolor=color, alpha=0.20, zorder=0, edgecolor='none')
                        except IndexError: logger.warning(f"IndexError during axvspan for chain {chain}, state {current_state}. Skipping span.")
                    current_state = state
                    block_start_idx = k
            if current_state is not None and block_start_idx < len(time_points):
                 color = ROTAMER_COLORS.get(current_state, DEFAULT_ROTAMER_COLOR)
                 try:
                      ax.axvspan(time_points[block_start_idx], time_points[-1], facecolor=color, alpha=0.20, zorder=0, edgecolor='none')
                 except IndexError: logger.warning(f"IndexError during final axvspan for chain {chain}, state {current_state}. Skipping span.")

            # Plot Chi2 Line
            valid_chi2_mask = np.isfinite(chi2)
            ax.plot(time_points[valid_chi2_mask], chi2[valid_chi2_mask], label='Chi2', color=chain_color_map[chain], linestyle='--', linewidth=STYLE['line_width']*0.7, alpha=0.9, zorder=1)

            # Configure axes
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

        # Save Figure
        plot_filename = "SF_Tyrosine_Chi2_Dihedrals_HMM.png" # Updated filename
        plot_filepath = os.path.join(output_dir, plot_filename)
        rel_path = os.path.relpath(plot_filepath, run_dir)
        fig.savefig(plot_filepath, dpi=150)
        plt.close(fig)
        logger.info(f"Saved stacked SF Tyrosine Chi2 (HMM states) plot to {plot_filepath}")
        register_product(db_conn, module_name, "png", "plot", rel_path,
                         subcategory="dihedrals_chi2", # Keep subcategory consistent
                         description="Stacked time series of SF Tyr Chi2 angles per chain with HMM rotamer state background.")
        return rel_path

    except Exception as e:
        logger.error(f"Failed to generate stacked Chi2 HMM dihedral plot: {e}", exc_info=True)
        if 'fig' in locals() and plt.fignum_exists(fig.number): plt.close(fig)
        return None


def _plot_tyrosine_scatter(df_raw_dihedrals: pd.DataFrame, output_dir: str, run_dir: str, db_conn: sqlite3.Connection, module_name: str) -> Optional[str]:
    """Creates Chi1 vs Chi2 scatter plot from raw dihedral data."""
    chains = sorted(list(set(c.split('_')[0] for c in df_raw_dihedrals.columns if c.endswith('_Chi1'))))
    if not chains: return None

    # Use raw data directly
    all_chi1 = np.concatenate([df_raw_dihedrals[f'{c}_Chi1'].values for c in chains if f'{c}_Chi1' in df_raw_dihedrals])
    all_chi2 = np.concatenate([df_raw_dihedrals[f'{c}_Chi2'].values for c in chains if f'{c}_Chi2' in df_raw_dihedrals])
    valid_idx = np.isfinite(all_chi1) & np.isfinite(all_chi2)

    if not np.any(valid_idx):
        logger.warning("No valid Chi1/Chi2 data points for scatter plot.")
        return None

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
    """Creates rotamer state population bar plot based on HMM assignments."""
    chains = sorted(list(set(c.split('_')[0] for c in df_hmm_states.columns if c.endswith('_State'))))
    if not chains: return None

    all_hmm_states = []
    for c in chains:
        state_col = f'{c}_State'
        if state_col in df_hmm_states:
            # Filter out non-rotamer states before counting
            valid_mask = ~df_hmm_states[state_col].isin(['Outside', 'Error', 'Unknown', np.nan])
            all_hmm_states.extend(df_hmm_states.loc[valid_mask, state_col].values)

    if not all_hmm_states:
        logger.warning("No valid HMM rotamer states found for population plot.")
        return None

    try:
        state_counts = pd.Series(all_hmm_states).value_counts(normalize=True) * 100
        # Use the defined state order from config for consistency
        possible_states = TYR_HMM_STATE_ORDER
        state_counts = state_counts.reindex(possible_states, fill_value=0) # Use defined order

        fig, ax = plt.subplots(figsize=(8, 5))
        # Use ROTAMER_COLORS if available, otherwise default palette
        colors = [ROTAMER_COLORS.get(state, '#cccccc') for state in state_counts.index]
        state_counts.plot(kind='bar', ax=ax, color=colors)

        ax.set_xlabel('HMM Rotamer State (Chi1, Chi2)')
        ax.set_ylabel('Population (%)')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)
        ax.set_ylim(bottom=0) # Ensure y-axis starts at 0

        plt.tight_layout()
        plot_path = os.path.join(output_dir, "SF_Tyrosine_Rotamer_Population_HMM.png") # Updated filename
        rel_path = os.path.relpath(plot_path, run_dir)
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        logger.info(f"Saved HMM rotamer population plot to {plot_path}")
        register_product(db_conn, module_name, "png", "plot", rel_path,
                         subcategory="rotamer_population", # Keep subcategory consistent
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
    """
    Plot the Tyr-Thr hydrogen bond distances over time.
    
    Args:
        df_distances: DataFrame containing distance time series
        output_dir: Output directory path
        run_dir: Run directory path
        db_conn: Database connection
        module_name: Module name for product registration
    
    Returns:
        Optional[str]: Path to the saved plot or None if failed
    """
    if 'Time (ns)' not in df_distances.columns:
        logger.error("Missing 'Time (ns)' column in H-bond distances dataframe")
        return None
    
    # Get chain pairs from column names, excluding 'Time (ns)'
    pairs = [col for col in df_distances.columns if col != 'Time (ns)']
    if not pairs:
        logger.warning("No pair data found in H-bond distances dataframe")
        return None
    
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot distances for each pair
        for pair in pairs:
            ax.plot(
                df_distances['Time (ns)'], 
                df_distances[pair], 
                label=pair,
                linewidth=STYLE['line_width']*0.8
            )
        
        # Add reference lines for H-bond thresholds
        ax.axhline(
            y=TYR_THR_DEFAULT_FORMED_REF_DIST, 
            color='green', 
            linestyle='--', 
            alpha=0.7,
            label=f'Formed threshold ({TYR_THR_DEFAULT_FORMED_REF_DIST} Å)'
        )
        ax.axhline(
            y=TYR_THR_DEFAULT_BROKEN_REF_DIST, 
            color='red', 
            linestyle='--', 
            alpha=0.7,
            label=f'Broken threshold ({TYR_THR_DEFAULT_BROKEN_REF_DIST} Å)'
        )
        
        # Add shading between the formed and broken thresholds (uncertain region)
        ax.axhspan(
            TYR_THR_DEFAULT_FORMED_REF_DIST,
            TYR_THR_DEFAULT_BROKEN_REF_DIST,
            color='gray',
            alpha=0.1,
            label='Uncertain region'
        )
        
        # Format plot
        ax.set_xlabel('Time (ns)', fontsize=STYLE['font_sizes']['axis_label'])
        ax.set_ylabel('Distance (Å)', fontsize=STYLE['font_sizes']['axis_label'])
        ax.legend(loc='best', fontsize=STYLE['font_sizes']['legend'])
        ax.grid(True, alpha=STYLE['grid']['alpha'], linestyle=STYLE['grid']['linestyle'])
        ax.set_title('Tyr445-Thr439 Hydrogen Bond Distances', fontsize=STYLE['font_sizes']['title'])
        
        # Set y-axis limits (0 to max + small margin)
        ymax = df_distances[pairs].max().max() * 1.1
        ax.set_ylim(0, max(ymax, TYR_THR_DEFAULT_BROKEN_REF_DIST + 1))
        
        # Save figure
        plt.tight_layout()
        plot_filename = "tyr_thr_hbond_distances.png"  # Change filename to match subcategory
        plot_path = os.path.join(output_dir, plot_filename)
        rel_path = os.path.relpath(plot_path, run_dir)
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        
        # Register product
        register_product(
            db_conn, module_name, "png", "plot", rel_path,
            subcategory="tyr_thr_hbond_distances",
            description="Time series of Tyr445-Thr439 hydrogen bond distances"
        )
        
        logger.info(f"Saved Tyr-Thr H-bond distances plot to {plot_path}")
        return rel_path
    
    except Exception as e:
        logger.error(f"Failed to generate Tyr-Thr H-bond distances plot: {e}")
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
    """
    Plot the Tyr-Thr hydrogen bond states as a heatmap.
    
    Args:
        df_states: DataFrame containing state time series
        output_dir: Output directory path
        run_dir: Run directory path
        db_conn: Database connection
        module_name: Module name for product registration
    
    Returns:
        Optional[str]: Path to the saved plot or None if failed
    """
    if 'Time (ns)' not in df_states.columns:
        logger.error("Missing 'Time (ns)' column in H-bond states dataframe")
        return None
    
    # Get debounced state columns
    state_cols = [col for col in df_states.columns if col.endswith('_debounced')]
    if not state_cols:
        logger.warning("No debounced state columns found in H-bond states dataframe")
        return None
    
    # Extract pair IDs from column names
    pairs = [col.split('_debounced')[0] for col in state_cols]
    
    try:
        fig, axes = plt.subplots(len(pairs), 1, figsize=(12, 2*len(pairs)), sharex=True)
        
        # Handle the case where there's only one pair (axes is not an array)
        if len(pairs) == 1:
            axes = [axes]
        
        # Define colors for states
        state_colors = {
            0: 'green',   # H-bond formed
            1: 'red',     # H-bond broken
            -1: 'gray'    # Uncertain
        }
        
        # Plot states for each pair
        for ax, pair, state_col in zip(axes, pairs, state_cols):
            states = df_states[state_col].values
            time = df_states['Time (ns)'].values
            
            # Create colored segments for each state
            current_state = None
            start_idx = 0
            
            for i, state in enumerate(states):
                if state != current_state:
                    # Plot the previous segment if it exists
                    if current_state is not None and i > start_idx:
                        color = state_colors.get(current_state, 'gray')
                        ax.axhspan(0, 1, xmin=time[start_idx], xmax=time[i-1], 
                                  facecolor=color, alpha=0.4)
                    
                    # Start a new segment
                    current_state = state
                    start_idx = i
            
            # Plot the last segment
            if current_state is not None and start_idx < len(time):
                color = state_colors.get(current_state, 'gray')
                ax.axhspan(0, 1, xmin=time[start_idx], xmax=time[-1], 
                          facecolor=color, alpha=0.4)
            
            # Add state labels
            ax.set_ylabel(pair, fontsize=STYLE['font_sizes']['axis_label'])
            ax.set_yticks([])
            
            # Add colored legend patches
            legend_elements = [
                plt.Rectangle((0, 0), 1, 1, facecolor='green', alpha=0.4, label='H-bond formed'),
                plt.Rectangle((0, 0), 1, 1, facecolor='red', alpha=0.4, label='H-bond broken'),
                plt.Rectangle((0, 0), 1, 1, facecolor='gray', alpha=0.4, label='Uncertain')
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=STYLE['font_sizes']['legend']*0.8)
            
            # Formatting
            ax.set_ylim(0, 1)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
        
        # Set shared x-axis label
        axes[-1].set_xlabel('Time (ns)', fontsize=STYLE['font_sizes']['axis_label'])
        
        # Add overall title
        fig.suptitle('Tyr445-Thr439 Hydrogen Bond States', fontsize=STYLE['font_sizes']['title'])
        
        # Adjust layout and save
        plt.tight_layout()
        plt.subplots_adjust(top=0.92, hspace=0.2)
        
        plot_filename = "tyr_thr_hbond_states.png"  # Change filename to match subcategory
        plot_path = os.path.join(output_dir, plot_filename)
        rel_path = os.path.relpath(plot_path, run_dir)
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        
        # Register product
        register_product(
            db_conn, module_name, "png", "plot", rel_path,
            subcategory="tyr_thr_hbond_states",
            description="Heatmap of Tyr445-Thr439 hydrogen bond states over time"
        )
        
        logger.info(f"Saved Tyr-Thr H-bond states plot to {plot_path}")
        return rel_path
    
    except Exception as e:
        logger.error(f"Failed to generate Tyr-Thr H-bond states plot: {e}")
        if 'fig' in locals() and plt.fignum_exists(fig.number):
            plt.close(fig)
        return None

def _plot_tyr_thr_hbond_statistics(
    df_events: pd.DataFrame,
    output_dir: str,
    run_dir: str,
    db_conn: sqlite3.Connection,
    module_name: str
) -> Optional[str]:
    """
    Plot statistics for Tyr-Thr hydrogen bond events.
    
    Args:
        df_events: DataFrame containing event data
        output_dir: Output directory path
        run_dir: Run directory path
        db_conn: Database connection
        module_name: Module name for product registration
    
    Returns:
        Optional[str]: Path to the saved plot or None if failed
    """
    if df_events.empty:
        logger.warning("Empty events dataframe for H-bond statistics plot")
        return None
    
    if not all(col in df_events.columns for col in ['Pair', 'State', 'Duration (ns)']):
        logger.error("Missing required columns in H-bond events dataframe")
        return None
    
    try:
        # Create a figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Event durations by pair and state
        sns.boxplot(
            x='Pair', 
            y='Duration (ns)', 
            hue='State', 
            data=df_events,
            palette={'H-bond-formed': 'green', 'H-bond-broken': 'red'},
            ax=ax1
        )
        
        ax1.set_title('H-bond Event Durations by Pair', fontsize=STYLE['font_sizes']['title'])
        ax1.set_xlabel('Chain Pair', fontsize=STYLE['font_sizes']['axis_label'])
        ax1.set_ylabel('Duration (ns)', fontsize=STYLE['font_sizes']['axis_label'])
        ax1.legend(title='State', fontsize=STYLE['font_sizes']['legend']*0.8)
        
        # Plot 2: Event count by pair and state
        event_counts = df_events.groupby(['Pair', 'State']).size().reset_index(name='Count')
        
        sns.barplot(
            x='Pair', 
            y='Count', 
            hue='State', 
            data=event_counts,
            palette={'H-bond-formed': 'green', 'H-bond-broken': 'red'},
            ax=ax2
        )
        
        ax2.set_title('H-bond Event Counts by Pair', fontsize=STYLE['font_sizes']['title'])
        ax2.set_xlabel('Chain Pair', fontsize=STYLE['font_sizes']['axis_label'])
        ax2.set_ylabel('Number of Events', fontsize=STYLE['font_sizes']['axis_label'])
        ax2.legend(title='State', fontsize=STYLE['font_sizes']['legend']*0.8)
        
        # Adjust layout and save
        plt.suptitle('Tyr445-Thr439 Hydrogen Bond Statistics', fontsize=STYLE['font_sizes']['title']*1.1)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        
        plot_filename = "tyr_thr_hbond_statistics.png"  # Filename already matched subcategory
        plot_path = os.path.join(output_dir, plot_filename)
        rel_path = os.path.relpath(plot_path, run_dir)
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        
        # Register product
        register_product(
            db_conn, module_name, "png", "plot", rel_path,
            subcategory="tyr_thr_hbond_statistics",
            description="Statistics of Tyr445-Thr439 hydrogen bond events"
        )
        
        logger.info(f"Saved Tyr-Thr H-bond statistics plot to {plot_path}")
        return rel_path
    
    except Exception as e:
        logger.error(f"Failed to generate Tyr-Thr H-bond statistics plot: {e}")
        if 'fig' in locals() and plt.fignum_exists(fig.number):
            plt.close(fig)
        return None

# --- Main Visualization Function (Modified) --- #
def generate_tyrosine_plots(
    run_dir: str,
    db_conn: sqlite3.Connection
) -> Dict[str, Any]:
    """
    Generates all standard plots for the SF Tyrosine rotamer analysis module,
    using HMM analysis results.

    Args:
        run_dir: Path to the specific run directory.
        db_conn: Active database connection.

    Returns:
        Dictionary containing status and paths to generated plots.
    """
    module_name = "tyrosine_analysis_visualization"
    start_time = time.time()
    register_module(db_conn, module_name, status='running')
    logger_local = setup_system_logger(run_dir)
    if logger_local is None: logger_local = logging.getLogger(__name__)

    results: Dict[str, Any] = {'status': 'failed', 'plots': {}, 'error': None}
    output_dir = os.path.join(run_dir, "tyrosine_analysis")
    os.makedirs(output_dir, exist_ok=True)

    # Retrieve paths for HMM-generated data files
    raw_dihedral_rel_path = get_product_path(db_conn, 'csv', 'data', 'raw_dihedrals', 'tyrosine_analysis')
    hmm_state_rel_path = get_product_path(db_conn, 'csv', 'data', 'hmm_state_path', 'tyrosine_analysis')

    # Retrieve paths for Tyr-Thr H-bond data files (optional)
    hbond_distances_rel_path = get_product_path(db_conn, 'csv', 'data', 'tyr_thr_hbond_distances', 'tyrosine_analysis')
    hbond_states_rel_path = get_product_path(db_conn, 'csv', 'data', 'tyr_thr_hbond_states', 'tyrosine_analysis')
    hbond_events_rel_path = get_product_path(db_conn, 'csv', 'data', 'tyr_thr_hbond_events', 'tyrosine_analysis')

    # Check if essential files were found for HMM analysis
    if not raw_dihedral_rel_path:
        results['error'] = "Raw dihedral data path ('raw_dihedrals') not found in database."
        logger_local.error(results['error'])
        update_module_status(db_conn, module_name, 'failed', error_message=results['error'])
        return results
    if not hmm_state_rel_path:
        results['error'] = "HMM state path data ('hmm_state_path') not found in database."
        logger_local.error(results['error'])
        update_module_status(db_conn, module_name, 'failed', error_message=results['error'])
        return results

    # Build absolute paths for HMM analysis files
    raw_dihedral_abs_path = os.path.join(run_dir, raw_dihedral_rel_path)
    hmm_state_abs_path = os.path.join(run_dir, hmm_state_rel_path)

    # Build absolute paths for H-bond files if they exist
    hbond_distances_abs_path = os.path.join(run_dir, hbond_distances_rel_path) if hbond_distances_rel_path else None
    hbond_states_abs_path = os.path.join(run_dir, hbond_states_rel_path) if hbond_states_rel_path else None
    hbond_events_abs_path = os.path.join(run_dir, hbond_events_rel_path) if hbond_events_rel_path else None

    # Check if essential HMM files exist
    if not os.path.exists(raw_dihedral_abs_path):
        results['error'] = f"Raw dihedral file not found at: {raw_dihedral_abs_path}"
        logger_local.error(results['error'])
        update_module_status(db_conn, module_name, 'failed', error_message=results['error'])
        return results
    if not os.path.exists(hmm_state_abs_path):
        results['error'] = f"HMM state path file not found at: {hmm_state_abs_path}"
        logger_local.error(results['error'])
        update_module_status(db_conn, module_name, 'failed', error_message=results['error'])
        return results

    # Load HMM analysis data
    try:
        df_raw_dihedrals = pd.read_csv(raw_dihedral_abs_path)
        df_hmm_states = pd.read_csv(hmm_state_abs_path)
        # Basic check for time alignment (can be more robust)
        if len(df_raw_dihedrals) != len(df_hmm_states):
             raise ValueError("Row count mismatch between raw dihedral and HMM state files.")
    except Exception as e:
        results['error'] = f"Failed to load HMM analysis data files: {e}"
        logger_local.error(results['error'], exc_info=True)
        update_module_status(db_conn, module_name, 'failed', error_message=results['error'])
        return results
        
    # Load H-bond data if available (optional, will not fail visualization if not found)
    df_hbond_distances = None
    df_hbond_states = None
    df_hbond_events = None
    
    if hbond_distances_abs_path and os.path.exists(hbond_distances_abs_path):
        try:
            df_hbond_distances = pd.read_csv(hbond_distances_abs_path)
            logger_local.info("Successfully loaded Tyr-Thr H-bond distances data")
        except Exception as e:
            logger_local.warning(f"Failed to load Tyr-Thr H-bond distances data: {e}")
    else:
        logger_local.info("Tyr-Thr H-bond distances data not found, skipping related plots")
        
    if hbond_states_abs_path and os.path.exists(hbond_states_abs_path):
        try:
            df_hbond_states = pd.read_csv(hbond_states_abs_path)
            logger_local.info("Successfully loaded Tyr-Thr H-bond states data")
        except Exception as e:
            logger_local.warning(f"Failed to load Tyr-Thr H-bond states data: {e}")
            
    if hbond_events_abs_path and os.path.exists(hbond_events_abs_path):
        try:
            df_hbond_events = pd.read_csv(hbond_events_abs_path)
            logger_local.info("Successfully loaded Tyr-Thr H-bond events data")
        except Exception as e:
            logger_local.warning(f"Failed to load Tyr-Thr H-bond events data: {e}")

    # Generate Plots
    plots_generated = 0
    plots_failed = 0
    expected_plots = 4  # Base expected plots from HMM analysis

    # 1. Chi1 Stacked Dihedrals Plot (using HMM states)
    chi1_path = _plot_chi1_dihedrals_stacked(df_raw_dihedrals, df_hmm_states, output_dir, run_dir, db_conn, module_name)
    if chi1_path: plots_generated += 1; results['plots']['dihedrals_chi1'] = chi1_path # Use existing key
    else: plots_failed += 1

    # 2. Chi2 Stacked Dihedrals Plot (using HMM states)
    chi2_path = _plot_chi2_dihedrals_stacked(df_raw_dihedrals, df_hmm_states, output_dir, run_dir, db_conn, module_name)
    if chi2_path: plots_generated += 1; results['plots']['dihedrals_chi2'] = chi2_path # Use existing key
    else: plots_failed += 1

    # 3. Scatter Plot (using raw dihedrals)
    scatter_path = _plot_tyrosine_scatter(df_raw_dihedrals, output_dir, run_dir, db_conn, module_name)
    if scatter_path: plots_generated += 1; results['plots']['rotamer_scatter'] = scatter_path # Use existing key
    else: plots_failed += 1

    # 4. Population Plot (using HMM states)
    pop_path = _plot_tyrosine_population(df_hmm_states, output_dir, run_dir, db_conn, module_name)
    if pop_path: plots_generated += 1; results['plots']['rotamer_population'] = pop_path # Use existing key
    else: plots_failed += 1

    # --- Tyr-Thr Hydrogen Bond Plots (if data available) ---
    # These are optional and won't cause the module to fail if they can't be generated
    
    # 5. H-bond Distances Plot
    if df_hbond_distances is not None:
        logger_local.info("Generating Tyr-Thr H-bond distances plot")
        hbond_dist_path = _plot_tyr_thr_hbond_distances(
            df_hbond_distances, output_dir, run_dir, db_conn, module_name)
        if hbond_dist_path:
            plots_generated += 1
            results['plots']['tyr_thr_hbond_distances'] = hbond_dist_path
            logger_local.info("Successfully generated H-bond distances plot")
        else:
            logger_local.warning("Failed to generate H-bond distances plot")
            # Don't increment plots_failed since these are optional
    
    # 6. H-bond States Plot
    if df_hbond_states is not None:
        logger_local.info("Generating Tyr-Thr H-bond states plot")
        hbond_states_path = _plot_tyr_thr_hbond_states(
            df_hbond_states, output_dir, run_dir, db_conn, module_name)
        if hbond_states_path:
            plots_generated += 1
            results['plots']['tyr_thr_hbond_states'] = hbond_states_path
            logger_local.info("Successfully generated H-bond states plot")
        else:
            logger_local.warning("Failed to generate H-bond states plot")
    
    # 7. H-bond Statistics Plot
    if df_hbond_events is not None:
        logger_local.info("Generating Tyr-Thr H-bond statistics plot")
        hbond_stats_path = _plot_tyr_thr_hbond_statistics(
            df_hbond_events, output_dir, run_dir, db_conn, module_name)
        if hbond_stats_path:
            plots_generated += 1
            results['plots']['tyr_thr_hbond_statistics'] = hbond_stats_path
            logger_local.info("Successfully generated H-bond statistics plot")
        else:
            logger_local.warning("Failed to generate H-bond statistics plot")

    # Finalize
    exec_time = time.time() - start_time
    min_plots_needed = expected_plots  # Only require the core plots to succeed
    final_status = 'success' if plots_failed == 0 and plots_generated >= min_plots_needed else ('failed' if plots_failed > 0 else 'skipped')
    error_msg = f"{plots_failed} plot(s) failed to generate." if plots_failed > 0 else None
    if final_status == 'skipped': error_msg = "No valid data or plots generated."

    update_module_status(db_conn, module_name, final_status, execution_time=exec_time, error_message=error_msg)
    logger_local.info(f"--- Tyrosine Visualization (HMM) finished in {exec_time:.2f} seconds (Status: {final_status}) ---")
    results['status'] = final_status
    if error_msg: results['error'] = error_msg

    return results