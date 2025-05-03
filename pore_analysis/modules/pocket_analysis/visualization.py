# pore_analysis/modules/pocket_analysis/visualization.py
"""
Visualization functions for Peripheral Pocket Water Analysis.

Generates static plots (PNG) based on data retrieved from the database,
suitable for inclusion in the HTML report.
"""

import os
import logging
import time
import sqlite3
import json
from typing import Dict, Optional, List, Any, Tuple

import matplotlib.pyplot as plt
import matplotlib as mpl # Import for rcParams if needed, though setup_style handles most
import seaborn as sns
import numpy as np
import pandas as pd

# --- Core Suite Imports ---
try:
    from pore_analysis.core.plotting_style import STYLE, setup_style
    from pore_analysis.core.database import (
        register_module, update_module_status, get_product_path, register_product,
        get_module_status, get_all_metrics # May need metrics for plot annotations
    )
    from pore_analysis.core.logging import setup_system_logger
    # Import config defaults if needed for fallbacks or plot thresholds
    from pore_analysis.core import config as core_config
    CORE_AVAILABLE = True
except ImportError as e:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.error(f"Critical Import Error - Core modules missing: {e}. Using dummy functions/defaults.")
    CORE_AVAILABLE = False
    # Define dummy functions/values
    def register_module(*args, **kwargs): pass
    def update_module_status(*args, **kwargs): pass
    def get_product_path(*args, **kwargs): return None
    def register_product(*args, **kwargs): pass
    def get_module_status(*args, **kwargs): return 'unknown'
    def get_all_metrics(*args, **kwargs): return {}
    def setup_system_logger(*args, **kwargs): return logging.getLogger(__name__)
    STYLE = {'bright_colors': {'A': 'blue', 'B': 'red', 'C': 'green', 'D': 'orange'},
             'chain_colors': {'A': 'lightblue', 'B': 'lightcoral', 'C': 'lightgreen', 'D': 'moccasin'},
             'font_sizes': {'annotation': 9, 'axis_label': 12, 'tick_label': 10},
             'threshold_style': {'color': 'black', 'ls': ':'}, 'grid': {'alpha': 0.5, 'color': 'grey', 'linestyle': ':'}}
    def setup_style(): pass
    class DummyConfig: pass
    core_config = DummyConfig()
    setattr(core_config, 'POCKET_ANALYSIS_RESIDENCE_THRESHOLD', 10) # Example default

logger = logging.getLogger(__name__)

# Apply standard plotting style
setup_style()
plt.switch_backend('agg') # Ensure non-interactive backend

# --- Plotting Helper Functions ---

def _save_static_plot(fig, path, dpi=150):
    """Save matplotlib figure with error handling."""
    try:
        plot_dir = os.path.dirname(path)
        if plot_dir: os.makedirs(plot_dir, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved static plot: {path}")
    except Exception as e:
        logger.error(f"Failed to save plot {path}: {e}", exc_info=True)
    finally:
        if fig is not None and plt.fignum_exists(fig.number):
            plt.close(fig)

def _plot_occupancy_timeseries_static(
    occupancy_df: pd.DataFrame,
    output_dir: str,
    run_dir: str,
    db_conn: sqlite3.Connection,
    module_name: str # Visualization module name
) -> Optional[str]:
    """
    Plots time series of water occupancy for each pocket using Matplotlib/Seaborn.
    Adapted from plot_filtered_pocket_time_series.
    """
    required_cols = ['Time (ns)', 'Pocket 0', 'Pocket 1', 'Pocket 2', 'Pocket 3']
    if occupancy_df is None or occupancy_df.empty or not all(c in occupancy_df.columns for c in required_cols):
        logger.error(f"Occupancy data for timeseries plot invalid or missing columns ({required_cols}).")
        return None

    logger.info("Generating static pocket occupancy time series plot...")
    try:
        fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True, sharey=True) # Share Y axis
        # Use bright colors for the occupancy plot
        bright_colors = [STYLE.get('bright_colors', {}).get(chr(ord('A') + i), f'C{i}') for i in range(4)]
        # Fallback to standard bright colors if not in STYLE
        bright_colors = ['#FF0000', '#0000FF', '#00CC00', '#FF9900'] if not all(bright_colors) else bright_colors
        labels = [f'Pocket {chr(ord("A") + i)}' for i in range(4)]
        pocket_cols = [f'Pocket {i}' for i in range(4)]

        max_y = occupancy_df[pocket_cols].max().max() * 1.05 # Find max across all pockets
        time_ns = occupancy_df['Time (ns)'].values

        for i, ax in enumerate(axes):
            pocket_col = pocket_cols[i]
            data = occupancy_df[pocket_col].values

            ax.plot(time_ns, data, color=bright_colors[i], linewidth=STYLE.get('line_width', 1.5)*0.8) # Slightly thinner line
            ax.set_ylabel('Water Count', fontsize=STYLE['font_sizes']['axis_label']*0.9) # Slightly smaller
            # Add pocket label directly on the plot (like original script title)
            ax.text(0.02, 0.95, labels[i], transform=ax.transAxes, ha='left', va='top',
                    fontsize=STYLE['font_sizes']['axis_label'], fontweight='bold', color=bright_colors[i])
            ax.set_ylim(bottom=0, top=max_y)
            ax.tick_params(axis='both', which='major', labelsize=STYLE['font_sizes']['tick_label'])
            ax.grid(True, alpha=STYLE['grid']['alpha'], color=STYLE['grid']['color'], linestyle=STYLE['grid']['linestyle'])
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            if i < len(axes) - 1:
                 plt.setp(ax.get_xticklabels(), visible=False)
                 ax.tick_params(axis='x', bottom=False)

        axes[-1].set_xlabel('Time (ns)', fontsize=STYLE['font_sizes']['axis_label'])
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.15) # Adjust vertical spacing

        plot_filename = "pocket_occupancy_plot.png" # Consistent name
        plot_filepath = os.path.join(output_dir, plot_filename)
        _save_static_plot(fig, plot_filepath) # Use helper

        # Register Product
        rel_path = os.path.relpath(plot_filepath, run_dir)
        register_product(db_conn, module_name, "png", "plot", rel_path,
                         subcategory="pocket_occupancy_plot", # Matches plots_dict.json
                         description="Time series of water counts per peripheral pocket.")
        return rel_path

    except Exception as e:
        logger.error(f"Failed to generate static occupancy plot: {e}", exc_info=True)
        if 'fig' in locals() and plt.fignum_exists(fig.number): plt.close(fig)
        return None

def _plot_residence_distribution_static(
    processed_residence_times_ns: Dict[int, List[float]], # Expects times in ns
    output_dir: str,
    run_dir: str,
    db_conn: sqlite3.Connection,
    module_name: str # Visualization module name
) -> Optional[str]:
    """
    Generates static plots for residence time distributions using Matplotlib/Seaborn.
    Creates a comprehensive 2x2 subplot figure with the following components:
    1. Stacked bar chart of residence time categories
    2. Bar chart of short-lived water molecules
    3. Bar chart of Kolmogorov-Smirnov test statistics comparing each pocket to Pocket D
    4. Cumulative probability plot of residence times
    """
    if not processed_residence_times_ns or not any(processed_residence_times_ns.values()):
        logger.warning("No processed residence time data available for distribution plot.")
        return None

    logger.info("Generating 2x2 residence time distribution plots...")
    try:
        # Set up 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        # Use bright colors for better distinction
        bright_colors = [STYLE.get('bright_colors', {}).get(chr(ord('A') + i), f'C{i}') for i in range(4)]
        # Fallback to standard bright colors if not in STYLE
        bright_colors = ['#FF0000', '#0000FF', '#00CC00', '#FF9900'] if not all(bright_colors) else bright_colors
        labels = [f'Pocket {chr(ord("A") + i)}' for i in range(4)]
        
        # Use residence threshold from config
        res_thresh_frames = getattr(core_config, 'POCKET_ANALYSIS_RESIDENCE_THRESHOLD', 10)
        frames_per_ns = getattr(core_config, 'FRAMES_PER_NS', 10.0)
        if frames_per_ns <= 0: frames_per_ns = 10.0 # Fallback
        res_thresh_ns = res_thresh_frames / frames_per_ns
        
        short_lived_thresh_ns = getattr(core_config, 'POCKET_ANALYSIS_SHORT_LIVED_THRESH_NS', 5.0)
        long_lived_thresh_ns = getattr(core_config, 'POCKET_ANALYSIS_LONG_LIVED_THRESH_NS', 10.0)
        
        # 1. Stacked bar chart of residence time categories [Top Left]
        ax_stacked = axes[0, 0]
        categories = ['0-2.5 ns', '2.5-5 ns', '5-10 ns', '>10 ns']
        category_data = []
        
        for i in range(4):  # Each pocket
            times_ns = np.array(processed_residence_times_ns.get(i, []))
            if len(times_ns) > 0:
                counts = [
                    np.sum((times_ns >= 0) & (times_ns < 2.5)),
                    np.sum((times_ns >= 2.5) & (times_ns < 5)),
                    np.sum((times_ns >= 5) & (times_ns < 10)),
                    np.sum(times_ns >= 10)
                ]
            else:
                counts = [0, 0, 0, 0]
            category_data.append(counts)
        
        category_data = np.array(category_data)
        # Calculate percentages, handling zero division
        with np.errstate(divide='ignore', invalid='ignore'):
            data_perc = np.where(category_data.sum(axis=1, keepdims=True) != 0, 
                              category_data / category_data.sum(axis=1, keepdims=True) * 100, 
                              0)
        data_perc = np.nan_to_num(data_perc, nan=0.0)
        
        # Plot stacked bars
        bottom = np.zeros(4)
        for i, cat in enumerate(categories):
            ax_stacked.bar(labels, data_perc[:, i], bottom=bottom, color=[bright_colors[0], bright_colors[1], bright_colors[2], bright_colors[3]][i], label=cat)
            bottom += data_perc[:, i]
        
        ax_stacked.set_title('Residence Time Categories', fontsize=STYLE['font_sizes']['axis_label'])
        ax_stacked.set_ylabel('Percentage', fontsize=STYLE['font_sizes']['axis_label'])
        ax_stacked.legend(fontsize=STYLE['font_sizes']['annotation'], title='Residence Time')
        
        # 2. Bar chart of short-lived molecules [Top Right]
        ax_short = axes[0, 1]
        short_lived = [np.sum((np.array(processed_residence_times_ns.get(i, [])) >= 1.0) & 
                              (np.array(processed_residence_times_ns.get(i, [])) <= short_lived_thresh_ns)) 
                      for i in range(4)]
        
        ax_short.bar(labels, short_lived, color=bright_colors)
        ax_short.set_title(f'Short-lived Water Molecules (1-{short_lived_thresh_ns} ns)', 
                         fontsize=STYLE['font_sizes']['axis_label'])
        ax_short.set_ylabel('Number', fontsize=STYLE['font_sizes']['axis_label'])
        
        # 3. KS test statistics [Bottom Left]
        ax_ks = axes[1, 0]
        # Get times for pocket D (index 3)
        pocket_d_times = np.array(processed_residence_times_ns.get(3, []))
        ks_stats = []
        
        # Calculate KS statistics comparing each pocket to Pocket D
        for i in range(3):  # Only pockets A, B, C compared to D
            times_ns = np.array(processed_residence_times_ns.get(i, []))
            if len(pocket_d_times) > 0 and len(times_ns) > 0:
                try:
                    from scipy import stats
                    ks_stat, _ = stats.ks_2samp(pocket_d_times, times_ns)
                    ks_stats.append(ks_stat)
                except Exception as e:
                    logger.warning(f"KS test failed for pocket {i} vs D: {e}")
                    ks_stats.append(0)
            else:
                ks_stats.append(0)
        
        # Plot KS statistics
        if ks_stats:
            ax_ks.bar([f'{chr(ord("A") + i)} vs D' for i in range(3)], ks_stats, color=bright_colors[:3])
            ax_ks.set_title('KS Test: Pocket D vs Others', fontsize=STYLE['font_sizes']['axis_label'])
            ax_ks.set_ylabel('KS Statistic', fontsize=STYLE['font_sizes']['axis_label'])
        else:
            logger.warning("No KS statistics computed. Skipping KS test plot.")
            ax_ks.set_title('KS Test: No data available', fontsize=STYLE['font_sizes']['axis_label'])
        
        # 4. Cumulative residence times plot [Bottom Right]
        ax_cumulative = axes[1, 1]
        for i in range(4):
            times_ns = np.array(processed_residence_times_ns.get(i, []))
            # Filter based on threshold IN NANOSECONDS
            times_ns_filtered = times_ns[times_ns >= res_thresh_ns]
            
            if len(times_ns_filtered) > 0:
                sorted_times = np.sort(times_ns_filtered)
                cumulative_probs = np.arange(1, len(sorted_times) + 1) / len(sorted_times)
                ax_cumulative.plot(sorted_times, cumulative_probs, color=bright_colors[i], linewidth=2, label=labels[i])
        
        ax_cumulative.set_xscale('log')
        ax_cumulative.set_xlabel('Residence Time (ns)', fontsize=STYLE['font_sizes']['axis_label'])
        ax_cumulative.set_ylabel('Cumulative Probability', fontsize=STYLE['font_sizes']['axis_label'])
        ax_cumulative.legend(fontsize=STYLE['font_sizes']['annotation'])
        ax_cumulative.grid(True, which="both", ls=":", alpha=STYLE['grid']['alpha']*0.7)
        ax_cumulative.tick_params(axis='both', which='major', labelsize=STYLE['font_sizes']['tick_label'])
        
        # Set custom ticks
        tick_values = [1, 2, 5, 10, 50, 100, 200]
        min_plot_x = res_thresh_ns * 0.9
        ax_cumulative.set_xlim(left=min_plot_x)
        ax_cumulative.set_xticks(tick_values)
        ax_cumulative.set_xticklabels([str(t) for t in tick_values])
        
        # Final layout adjustments
        plt.tight_layout()
        plot_filename = "pocket_residence_analysis.png"
        plot_filepath = os.path.join(output_dir, plot_filename)
        _save_static_plot(fig, plot_filepath)
        
        # Register Product
        rel_path = os.path.relpath(plot_filepath, run_dir)
        register_product(db_conn, module_name, "png", "plot", rel_path,
                         subcategory="pocket_residence_analysis",
                         description="Comprehensive analysis of water residence times in pockets.")
        
        # Create the individual cumulative plot as well for backward compatibility
        fig_single, ax_single = plt.subplots(figsize=(8, 6))
        for i in range(4):
            times_ns = np.array(processed_residence_times_ns.get(i, []))
            times_ns_filtered = times_ns[times_ns >= res_thresh_ns]
            if len(times_ns_filtered) > 0:
                sorted_times = np.sort(times_ns_filtered)
                cumulative_probs = np.arange(1, len(sorted_times) + 1) / len(sorted_times)
                ax_single.plot(sorted_times, cumulative_probs, color=bright_colors[i], linewidth=2, label=labels[i])
        
        ax_single.set_xscale('log')
        ax_single.set_xlabel('Residence Time (ns)', fontsize=STYLE['font_sizes']['axis_label'])
        ax_single.set_ylabel('Cumulative Probability', fontsize=STYLE['font_sizes']['axis_label'])
        ax_single.legend(fontsize=STYLE['font_sizes']['annotation'])
        ax_single.grid(True, which="both", ls=":", alpha=STYLE['grid']['alpha']*0.7)
        ax_single.tick_params(axis='both', which='major', labelsize=STYLE['font_sizes']['tick_label'])
        ax_single.set_xlim(left=min_plot_x)
        
        plt.tight_layout()
        single_plot_filename = "pocket_residence_distribution.png"
        single_plot_filepath = os.path.join(output_dir, single_plot_filename)
        _save_static_plot(fig_single, single_plot_filepath)
        
        # Register the single plot too
        single_rel_path = os.path.relpath(single_plot_filepath, run_dir)
        register_product(db_conn, module_name, "png", "plot", single_rel_path,
                        subcategory="pocket_residence_histogram",
                        description="Cumulative distribution of water residence times per pocket.")
        
        return rel_path

    except Exception as e:
        logger.error(f"Failed to generate residence time plots: {e}", exc_info=True)
        if 'fig' in locals() and plt.fignum_exists(fig.number): plt.close(fig)
        if 'fig_single' in locals() and plt.fignum_exists(fig_single.number): plt.close(fig_single)
        return None

# --- Main Visualization Orchestrator ---

def generate_pocket_plots(
    run_dir: str,
    db_conn: sqlite3.Connection
) -> Dict[str, Any]:
    """
    Generates all static plots for the pocket_analysis module.
    Retrieves necessary data file paths from the database.
    """
    module_name = "pocket_analysis_visualization" # Specific name for this viz module
    start_time = time.time()
    register_module(db_conn, module_name, status='running')
    logger_local = setup_system_logger(run_dir)
    if logger_local is None: logger_local = logger # Fallback

    results: Dict[str, Any] = {'status': 'failed', 'plots': {}, 'error': None}
    output_dir = os.path.join(run_dir, "pocket_analysis") # Save plots in module's dir
    os.makedirs(output_dir, exist_ok=True)

    # --- Check computation status ---
    comp_status = get_module_status(db_conn, "pocket_analysis")
    if comp_status != 'success':
        results['status'] = 'skipped'
        results['error'] = f"Skipping visualization: Computation status was '{comp_status}'."
        logger_local.warning(results['error'])
        update_module_status(db_conn, module_name, 'skipped', error_message=results['error'])
        return results

    # --- Retrieve Data File Paths ---
    occ_rel_path = get_product_path(db_conn, 'csv', 'data', 'pocket_occupancy_timeseries', 'pocket_analysis')
    res_rel_path = get_product_path(db_conn, 'json', 'data', 'pocket_residence_stats', 'pocket_analysis')

    occupancy_df = None
    processed_residence_times_ns = None
    can_plot = True

    if occ_rel_path:
        occ_abs_path = os.path.join(run_dir, occ_rel_path)
        if os.path.exists(occ_abs_path):
            try: occupancy_df = pd.read_csv(occ_abs_path)
            except Exception as e: logger.error(f"Failed to load occupancy CSV: {e}"); can_plot = False
        else: logger.error(f"Occupancy CSV not found: {occ_abs_path}"); can_plot = False
    else: logger.error("Occupancy CSV path not found in DB."); can_plot = False

    if res_rel_path:
        res_abs_path = os.path.join(run_dir, res_rel_path)
        if os.path.exists(res_abs_path):
            try:
                 with open(res_abs_path, 'r') as f: res_data = json.load(f)
                 # Convert keys back to int if needed (JSON keys are strings)
                 processed_residence_times_ns = {int(k): v for k, v in res_data.items()}
            except Exception as e: logger.error(f"Failed to load residence JSON: {e}"); can_plot = False
        else: logger.error(f"Residence JSON not found: {res_abs_path}"); can_plot = False
    else: logger.error("Residence JSON path not found in DB."); can_plot = False

    if not can_plot:
        results['error'] = "Failed to load necessary data for plotting."
        update_module_status(db_conn, module_name, 'failed', error_message=results['error'])
        return results

    # --- Generate Plots ---
    plots_generated = 0
    plots_failed = 0

    # Plot 1: Occupancy Timeseries
    occ_plot_path = _plot_occupancy_timeseries_static(occupancy_df, output_dir, run_dir, db_conn, module_name)
    if occ_plot_path: plots_generated += 1; results['plots']['pocket_occupancy_plot'] = occ_plot_path
    else: plots_failed += 1

    # Plot 2: Residence Time Distribution
    res_plot_path = _plot_residence_distribution_static(processed_residence_times_ns, output_dir, run_dir, db_conn, module_name)
    if res_plot_path: plots_generated += 1; results['plots']['pocket_residence_histogram'] = res_plot_path
    else: plots_failed += 1

    # --- Finalize ---
    exec_time = time.time() - start_time
    final_status = 'success' if plots_failed == 0 and plots_generated > 0 else ('failed' if plots_failed > 0 else 'skipped')
    error_msg = f"{plots_failed} plot(s) failed to generate." if plots_failed > 0 else None
    if final_status == 'skipped': error_msg = "No data loaded or plots generated."

    update_module_status(db_conn, module_name, final_status, execution_time=exec_time, error_message=error_msg)
    logger_local.info(f"--- Pocket Analysis Visualization finished in {exec_time:.2f} seconds (Status: {final_status}) ---")
    results['status'] = final_status
    if error_msg: results['error'] = error_msg

    return results