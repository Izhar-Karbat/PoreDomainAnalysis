"""
Visualization functions for inner vestibule water analysis.
Generates plots based on data stored in the database.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import time
import sqlite3
from typing import Dict, Optional, List, Any, Tuple

# Import from core modules
try:
    from pore_analysis.core.plotting_style import STYLE, setup_style
    # MODIFIED: Added get_module_status to imports
    from pore_analysis.core.database import (
        register_module, update_module_status, get_product_path, register_product,
        get_module_status
    )
    from pore_analysis.core.logging import setup_system_logger
except ImportError as e:
    print(f"Error importing dependency modules in inner_vestibule_analysis/visualization.py: {e}")
    raise

logger = logging.getLogger(__name__)

# Apply standard plotting style
setup_style()

def _create_vestibule_plots(
    occupancy_file_path: str,
    residence_file_path: str,
    output_dir: str,
    run_dir: str,
    db_conn: sqlite3.Connection,
    module_name: str
) -> Tuple[Optional[str], Optional[str]]:
    """
    Internal helper to generate and save vestibule plots (Count & Residence Time).
    Does NOT add titles to plots.
    """
    paths = {'count_plot': None, 'residence_hist': None}
    plot_mean_occupancy = np.nan
    plot_avg_residence_time = np.nan
    time_points = np.array([])
    water_counts_per_frame = np.array([])
    all_residence_times_ns: List[float] = []

    # Load Occupancy Data
    if not os.path.exists(occupancy_file_path):
        logger.error(f"Occupancy data file not found: {occupancy_file_path}")
        # Cannot create count plot
    else:
        try:
            df_occup = pd.read_csv(occupancy_file_path)
            if 'Time (ns)' in df_occup and 'Water_Count' in df_occup:
                time_points = df_occup['Time (ns)'].values
                water_counts_per_frame = df_occup['Water_Count'].values
                # Calculate mean from loaded data for plot annotation
                valid_counts = water_counts_per_frame[water_counts_per_frame >= 0]
                if len(valid_counts) > 0:
                    plot_mean_occupancy = np.mean(valid_counts)
            else:
                logger.warning("Occupancy CSV missing required columns ('Time (ns)', 'Water_Count').")
        except Exception as e:
            logger.error(f"Failed to load occupancy data from {occupancy_file_path}: {e}")

    # Load Residence Time Data
    if not os.path.exists(residence_file_path):
        logger.error(f"Residence time data file not found: {residence_file_path}")
        # Cannot create residence time plot
    else:
        try:
            with open(residence_file_path, 'r') as f:
                res_data = json.load(f)
                all_residence_times_ns = res_data.get('residence_times_ns', [])
                # Get mean from loaded data for plot annotation
                plot_avg_residence_time = res_data.get('metadata', {}).get('mean_residence_time_ns', np.nan)
                if not isinstance(all_residence_times_ns, list):
                    logger.warning("Residence times data in JSON is not a list.")
                    all_residence_times_ns = []
        except Exception as e:
            logger.error(f"Failed to load residence time data from {residence_file_path}: {e}")

    # --- Plot 1: Water Count Over Time ---
    if len(time_points) > 0 and len(water_counts_per_frame) == len(time_points):
        try:
            fig_c, ax_c = plt.subplots(figsize=(10, 5))
            ax_c.plot(
                time_points,
                water_counts_per_frame,
                color=STYLE['bright_colors']['A'],
                linewidth=STYLE['line_width'] * 0.8,
            )
            # Add mean line if calculated
            if np.isfinite(plot_mean_occupancy):
                ax_c.axhline(y=plot_mean_occupancy, **STYLE['threshold_style'])
                # Add annotation for mean value
                ax_c.text(
                    time_points[-1] * 0.98,
                    plot_mean_occupancy,
                    f" Mean: {plot_mean_occupancy:.2f}",
                    color=STYLE['threshold_style']['color'],
                    ha='right',
                    va='bottom',
                    fontsize=STYLE['font_sizes']['annotation'],
                )

            ax_c.set_xlabel('Time (ns)')
            ax_c.set_ylabel('Water Count')
            ax_c.tick_params(axis='both')
            ax_c.grid(True, alpha=STYLE['grid']['alpha'], color=STYLE['grid']['color'], linestyle=STYLE['grid']['linestyle'])
            # Adjust y-lim slightly, ensuring bottom is not negative if counts are always >= 0
            min_count = (
                np.min(water_counts_per_frame[water_counts_per_frame >= 0])
                if np.any(water_counts_per_frame >= 0)
                else 0
            )
            ax_c.set_ylim(bottom=max(0, min_count - 1))

            plt.tight_layout()
            plot_filename = "inner_vestibule_count_plot.png"
            plot_filepath = os.path.join(output_dir, plot_filename)
            rel_path = os.path.relpath(plot_filepath, run_dir)
            fig_c.savefig(plot_filepath, dpi=150)
            plt.close(fig_c)
            logger.info(f"Saved vestibule water count plot to {plot_filepath}")
            # Register Product
            register_product(
                db_conn,
                module_name,
                "png",
                "plot",
                rel_path,
                subcategory="count_plot",
                description="Time series of water count in the inner vestibule.",
            )
            paths['count_plot'] = rel_path
        except Exception as e:
            logger.error(f"Failed to generate vestibule count plot: {e}", exc_info=True)
            if 'fig_c' in locals() and plt.fignum_exists(fig_c.number):
                plt.close(fig_c)

    # --- Plot 2: Residence Time Histogram ---
    if all_residence_times_ns:
        try:
            fig_r, ax_r = plt.subplots(figsize=(8, 5))
            # Determine appropriate bins
            if len(all_residence_times_ns) < 20:
                bins = max(5, len(all_residence_times_ns) // 2)
            elif len(all_residence_times_ns) < 100:
                bins = 15
            else:
                bins = 25

            # Ensure max > 0 for range and handle empty list case for np.max
            max_res_time = np.max(all_residence_times_ns) if all_residence_times_ns else 0
            if max_res_time > 0:
                bins = np.linspace(0, max_res_time * 1.05, bins + 1)
            else:
                bins = 10  # Default number of bins if max is 0

            sns.histplot(
                all_residence_times_ns,
                bins=bins,
                ax=ax_r,
                kde=False,
                color=STYLE['bright_colors']['B'],
                alpha=0.7,
                stat='count',
            )

            # Add mean line if calculated
            if np.isfinite(plot_avg_residence_time):
                ax_r.axvline(x=plot_avg_residence_time, **STYLE['threshold_style'])

                # --- MODIFIED TEXT PLACEMENT ---
                # Position text slightly to the RIGHT of the line, left-aligned
                x_range = ax_r.get_xlim()[1] - ax_r.get_xlim()[0]
                x_pos_text = plot_avg_residence_time + x_range * 0.01  # Small offset to the right
                ax_r.text(
                    x_pos_text,
                    ax_r.get_ylim()[1] * 0.95,
                    f" Mean: {plot_avg_residence_time:.3f} ns",
                    color=STYLE['threshold_style']['color'],
                    ha='left',  # Changed alignment to left
                    va='top',
                    fontsize=STYLE['font_sizes']['annotation'],
                )
                # --- END OF MODIFICATION ---

            ax_r.set_xlabel('Residence Time (ns)')
            ax_r.set_ylabel('Frequency (Count)')
            ax_r.tick_params(axis='both')
            ax_r.grid(True, alpha=STYLE['grid']['alpha'], color=STYLE['grid']['color'], linestyle=STYLE['grid']['linestyle'])
            if max_res_time > 0:
                ax_r.set_xlim(left=0)

            plt.tight_layout()
            plot_filename = "inner_vestibule_residence_hist.png"
            plot_filepath = os.path.join(output_dir, plot_filename)
            rel_path = os.path.relpath(plot_filepath, run_dir)
            fig_r.savefig(plot_filepath, dpi=150)
            plt.close(fig_r)
            logger.info(f"Saved vestibule residence time histogram to {plot_filepath}")
            # Register Product
            register_product(
                db_conn,
                module_name,
                "png",
                "plot",
                rel_path,
                subcategory="residence_hist",
                description="Histogram of water residence times in the inner vestibule.",
            )
            paths['residence_hist'] = rel_path
        except Exception as e:
            logger.error(f"Failed to generate residence time histogram: {e}", exc_info=True)
            if 'fig_r' in locals() and plt.fignum_exists(fig_r.number):
                plt.close(fig_r)
    else:
        logger.info("No residence time data to plot.")

    return paths['count_plot'], paths['residence_hist']


def generate_inner_vestibule_plots(
    run_dir: str,
    db_conn: sqlite3.Connection,
) -> Dict[str, Any]:
    """
    Generates all standard plots for the inner vestibule analysis module.
    Retrieves necessary data file paths from the database.

    Args:
        run_dir: Path to the specific run directory.
        db_conn: Active database connection.

    Returns:
        Dictionary containing status and paths to generated plots.
    """
    module_name = "inner_vestibule_analysis_visualization"
    start_time = time.time()
    register_module(db_conn, module_name, status='running')
    logger = setup_system_logger(run_dir)
    if logger is None:
        logger = logging.getLogger(__name__)

    results: Dict[str, Any] = {'status': 'failed', 'plots': {}}
    output_dir = os.path.join(run_dir, "inner_vestibule_analysis")
    os.makedirs(output_dir, exist_ok=True)

    # --- Check computation status BEFORE retrieving paths ---
    comp_status = get_module_status(db_conn, "inner_vestibule_analysis")
    if comp_status != 'success':
        results['status'] = 'skipped'
        results['error'] = f"Skipping visualization: Computation status was '{comp_status}'."
        logger.warning(results['error'])
        update_module_status(db_conn, module_name, 'skipped', error_message=results['error'])
        return results

    # --- Retrieve Data File Paths (only if computation succeeded) ---
    occupancy_rel_path = get_product_path(
        db_conn,
        'csv',
        'data',
        'occupancy_per_frame',
        'inner_vestibule_analysis',
    )
    residence_rel_path = get_product_path(
        db_conn,
        'json',
        'data',
        'residence_times',
        'inner_vestibule_analysis',
    )

    if not occupancy_rel_path:
        results['error'] = "Occupancy data file path not found in database."
        logger.error(results['error'])
        update_module_status(db_conn, module_name, 'failed', error_message=results['error'])
        return results
    if not residence_rel_path:
        results['error'] = "Residence time data file path not found in database."
        logger.error(results['error'])
        update_module_status(db_conn, module_name, 'failed', error_message=results['error'])
        return results

    occupancy_abs_path = os.path.join(run_dir, occupancy_rel_path)
    residence_abs_path = os.path.join(run_dir, residence_rel_path)

    # --- Generate Plots ---
    try:
        count_plot_path, residence_hist_path = _create_vestibule_plots(
            occupancy_abs_path,
            residence_abs_path,
            output_dir,
            run_dir,
            db_conn,
            module_name,
        )

        plots_generated = 0
        if count_plot_path:
            results['plots']['count_plot'] = count_plot_path
            plots_generated += 1
        if residence_hist_path:
            results['plots']['residence_hist'] = residence_hist_path
            plots_generated += 1

        if plots_generated > 0:
            results['status'] = 'success'
            if plots_generated < 2:
                results['error'] = "One vestibule plot failed to generate."
                logger.warning(results['error'])
        else:
            results['status'] = 'failed'
            results['error'] = "Both vestibule plots failed to generate."
            logger.warning(results['error'])

    except Exception as e_plot:
        results['error'] = f"Error during vestibule plot generation: {e_plot}"
        logger.error(results['error'], exc_info=True)
        results['status'] = 'failed'

    # --- Finalize ---
    exec_time = time.time() - start_time
    update_module_status(
        db_conn,
        module_name,
        results['status'],
        execution_time=exec_time,
        error_message=results.get('error'),
    )
    logger.info(
        f"--- Inner Vestibule Visualization completed in {exec_time:.2f} seconds (Status: {results['status']}) ---"
    )

    return results
