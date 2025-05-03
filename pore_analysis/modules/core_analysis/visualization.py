# modules/core_analysis/visualization.py
"""
Core analysis visualization functions for plotting G-G and COM distances.
This module focuses purely on generating visualizations from processed data with consistent styling.
Titles are intentionally omitted from plots generated here; they should be added by the calling context (e.g., HTML template).
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
import sqlite3
import time # <-- Import time for execution time
from typing import Optional
from pore_analysis.core.plotting_style import STYLE, setup_style

# Import from other modules
try:
    from pore_analysis.core.logging import setup_system_logger
    from pore_analysis.core.database import (
        connect_db, register_product, get_product_path, get_module_status,
        register_module, update_module_status # <-- Added register/update status
    )
except ImportError as e:
    print(f"Error importing dependency modules in core_analysis/visualization.py: {e}")
    raise

# Configure matplotlib for non-interactive backend
plt.switch_backend('agg')

# Initialize style
setup_style()

logger = logging.getLogger(__name__)

# --- plot_distances function ---
def plot_distances(run_dir, data_file=None, output_prefix=None, is_gg=True, db_conn: Optional[sqlite3.Connection] = None):
    """
    Create plots for raw and filtered distance data with consistent styling.
    For G-G, creates a two-panel plot comparing raw vs filtered for A:C and B:D.
    For COM, creates ONLY the raw vs filtered comparison plot.
    Titles are intentionally omitted. Font sizes are controlled by STYLE dictionary.

    Args:
        run_dir (str): Directory path for the run data.
        data_file (str, optional): Path to the CSV file containing distance data.
                                  If None, the path will be retrieved from the database.
        output_prefix (str, optional): Prefix for output files. If None, 'G_G' or 'COM'
                                      will be used based on is_gg parameter.
        is_gg (bool): Whether the data is G-G distances (True) or COM distances (False).
        db_conn (sqlite3.Connection, optional): Existing database connection. If None, connects automatically.

    Returns:
        dict: A dictionary of plot paths keyed by plot type (relative to run_dir)
    """
    # Determine module name based on is_gg
    module_name = "core_analysis_visualization_g_g" if is_gg else "core_analysis_visualization_com"
    start_time = time.time() # Start timer

    # Set up logging
    logger = setup_system_logger(run_dir)
    if logger is None:
        logger = logging.getLogger(__name__) # Fallback logger
        logger.error(f"Failed to setup system logger for {run_dir}. Using module logger.")

    # --- CHECK for db_conn ---
    if db_conn is None:
        logger.error("Database connection (db_conn) is None. Cannot generate distance plots or register products.")
        return {} # Return empty dict as expected by the test on failure

    # --- Register Module Start ---
    register_module(db_conn, module_name, status='running')
    # Note: commit might happen later or be handled by the caller in the main workflow

    # Check if filtering was successful (prerequisite)
    filter_status = get_module_status(db_conn, "core_analysis_filtering")
    if filter_status != "success":
        error_msg = f"Cannot generate plots - filtering module status is: {filter_status}"
        logger.error(error_msg)
        update_module_status(db_conn, module_name, 'skipped', error_message=error_msg) # Update status to skipped
        return {}

    # Define output directory
    output_dir = os.path.join(run_dir, "core_analysis")
    os.makedirs(output_dir, exist_ok=True)

    # Set default output prefix if not provided
    if output_prefix is None:
        output_prefix = "G_G" if is_gg else "COM"

    # Set up data file path if not provided
    if data_file is None:
        # Get file path from database
        file_category = "g_g_distance_filtered" if is_gg else "com_stability_filtered"
        db_path = get_product_path(db_conn, "csv", "data", file_category) # Use db_conn here
        if db_path:
            data_file = os.path.join(run_dir, db_path)
        else:
            # Fallback paths if not found in database
            if is_gg:
                data_file = os.path.join(output_dir, "G_G_Distance_Filtered.csv")
            else:
                data_file = os.path.join(output_dir, "COM_Stability_Filtered.csv")

            if not os.path.exists(data_file):
                error_msg = f"Distance data file not found: {data_file}"
                logger.error(error_msg)
                update_module_status(db_conn, module_name, 'failed', error_message=error_msg) # Update status to failed
                return {}

    # Load data from CSV
    try:
        df = pd.read_csv(data_file)

        if is_gg:
            # Check if required columns exist for G-G plots
            required_cols = ["Time (ns)", "G_G_Distance_AC_Raw", "G_G_Distance_BD_Raw",
                           "G_G_Distance_AC_Filt", "G_G_Distance_BD_Filt"]
            if not all(col in df.columns for col in required_cols):
                missing_cols = [col for col in required_cols if col not in df.columns]
                error_msg = f"Missing required columns in G-G data file: {missing_cols}"
                logger.error(error_msg)
                update_module_status(db_conn, module_name, 'failed', error_message=error_msg)
                return {}
            data1_raw_col = "G_G_Distance_AC_Raw"
            data2_raw_col = "G_G_Distance_BD_Raw"
            data1_filt_col = "G_G_Distance_AC_Filt"
            data2_filt_col = "G_G_Distance_BD_Filt"
            color_ac_pastel = STYLE['chain_colors'].get('B', '#FFD1DC')
            color_ac_bright = STYLE['bright_colors'].get('B', '#FF6B81')
            color_bd_pastel = STYLE['chain_colors'].get('A', '#AEC6CF')
            color_bd_bright = STYLE['bright_colors'].get('A', '#4682B4')
        else: # COM plots
            required_cols = ["Time (ns)", "COM_Distance_Raw", "COM_Distance_Filt"]
            if not all(col in df.columns for col in required_cols):
                missing_cols = [col for col in required_cols if col not in df.columns]
                error_msg = f"Missing required columns in COM data file: {missing_cols}"
                logger.error(error_msg)
                update_module_status(db_conn, module_name, 'failed', error_message=error_msg)
                return {}
            data1_raw_col = "COM_Distance_Raw"
            data2_raw_col = None
            data1_filt_col = "COM_Distance_Filt"
            data2_filt_col = None
            raw_colors = [STYLE['chain_colors'].get('A', '#AEC6CF')]
            filt_colors = [STYLE['bright_colors'].get('A', '#4682B4')]

    except Exception as e:
        error_msg = f"Failed to load data from {data_file}: {e}"
        logger.error(error_msg, exc_info=True)
        update_module_status(db_conn, module_name, 'failed', error_message=error_msg)
        return {}

    plot_paths = {}
    plot_success = False # Flag to track if plotting succeeded

    try:
        time_points = df["Time (ns)"].values
        raw_data1 = df[data1_raw_col].values
        raw_data2 = df[data2_raw_col].values if data2_raw_col else None
        filtered_data1 = df[data1_filt_col].values
        filtered_data2 = df[data2_filt_col].values if data2_filt_col else None

        if is_gg:
            # --- G-G PLOTTING ---
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
            ax1.plot(time_points, raw_data1, color=color_ac_pastel, alpha=0.7, label='Raw A:C', linewidth=STYLE['line_width'])
            ax1.plot(time_points, filtered_data1, color=color_ac_bright, label='Filtered A:C', linewidth=STYLE['line_width'])
            ax1.set_ylabel('G-G Distance (Å)', fontsize=STYLE['font_sizes']['axis_label'])
            ax1.set_xlabel('Time (ns)', fontsize=STYLE['font_sizes']['axis_label'])
            ax1.grid(True, alpha=STYLE['grid']['alpha'], color=STYLE['grid']['color'])
            ax1.legend(fontsize=STYLE['font_sizes']['annotation'])
            ax1.tick_params(axis='both', labelsize=STYLE['font_sizes']['tick_label'])

            if raw_data2 is not None and filtered_data2 is not None:
                ax2.plot(time_points, raw_data2, color=color_bd_pastel, alpha=0.7, label='Raw B:D', linewidth=STYLE['line_width'])
                ax2.plot(time_points, filtered_data2, color=color_bd_bright, label='Filtered B:D', linewidth=STYLE['line_width'])
                ax2.set_xlabel('Time (ns)', fontsize=STYLE['font_sizes']['axis_label'])
                ax2.grid(True, alpha=STYLE['grid']['alpha'], color=STYLE['grid']['color'])
                ax2.legend(fontsize=STYLE['font_sizes']['annotation'])
                ax2.tick_params(axis='both', labelsize=STYLE['font_sizes']['tick_label'])
            else:
                 ax2.text(0.5, 0.5, 'B:D data not available', ha='center', va='center', transform=ax2.transAxes)

            plt.tight_layout()
            subunit_comp_filename = f"{output_prefix}_Distance_Subunit_Comparison.png"
            subunit_comp_path = os.path.join(output_dir, subunit_comp_filename)
            subunit_comp_rel_path = os.path.relpath(subunit_comp_path, run_dir)

            plt.savefig(subunit_comp_path, dpi=150)
            plt.close(fig)
            logger.info(f"Saved G-G subunit comparison plot to {subunit_comp_path}")
            plot_paths["subunit_comparison"] = subunit_comp_rel_path
            plot_success = True # Mark success

            register_product(
                db_conn, module_name, "png", "plot", subunit_comp_rel_path,
                subcategory="subunit_comparison",
                description="G-G Distance Raw vs Filtered Comparison by Subunit Pair (A:C Left, B:D Right)"
            )

        else:
            # --- COM PLOTTING ---
            if filtered_data1 is not None:
                fig_comp, ax_comp = plt.subplots(figsize=(10, 6))
                ax_comp.plot(time_points, raw_data1, color=raw_colors[0], alpha=0.6, label='Raw Data', linewidth=STYLE['line_width'])
                ax_comp.plot(time_points, filtered_data1, color=filt_colors[0], label='Filtered Data', linewidth=STYLE['line_width'])
                ax_comp.set_ylabel('COM Distance (Å)', fontsize=STYLE['font_sizes']['axis_label'])
                ax_comp.set_xlabel('Time (ns)', fontsize=STYLE['font_sizes']['axis_label'])
                ax_comp.grid(True, alpha=STYLE['grid']['alpha'], color=STYLE['grid']['color'])
                ax_comp.legend(fontsize=STYLE['font_sizes']['annotation'])
                ax_comp.tick_params(axis='both', labelsize=STYLE['font_sizes']['tick_label'])
                plt.tight_layout()

                comparison_filename = f"{output_prefix}_Stability_Comparison.png"
                comparison_path = os.path.join(output_dir, comparison_filename)
                comparison_rel_path = os.path.relpath(comparison_path, run_dir)

                plt.savefig(comparison_path, dpi=150)
                plt.close(fig_comp)
                logger.info(f"Saved COM raw vs filtered comparison plot to {comparison_path}")
                plot_paths["comparison"] = comparison_rel_path
                plot_success = True # Mark success

                register_product(
                    db_conn, module_name, "png", "plot", comparison_rel_path,
                    subcategory="comparison",
                    description="Comparison of raw and filtered COM Distance"
                )
            else:
                logger.warning("Filtered COM data not available, cannot generate comparison plot.")
                plot_success = False # Comparison plot is the only one for COM, so fail if it can't be made

        # --- Update Status ---
        exec_time = time.time() - start_time
        if plot_success:
             update_module_status(db_conn, module_name, 'success', execution_time=exec_time)
        else:
             update_module_status(db_conn, module_name, 'failed', execution_time=exec_time, error_message="Plot generation failed")

        return plot_paths

    except Exception as e:
        error_msg = f"Error generating distance plots: {e}"
        logger.error(error_msg, exc_info=True)
        # Ensure plot figures are closed
        if is_gg and 'fig' in locals() and plt.fignum_exists(fig.number): plt.close(fig)
        if not is_gg and 'fig_comp' in locals() and plt.fignum_exists(fig_comp.number): plt.close(fig_comp)
        exec_time = time.time() - start_time
        update_module_status(db_conn, module_name, 'failed', execution_time=exec_time, error_message=error_msg)
        return {}


# --- plot_kde_analysis function ---
def plot_kde_analysis(run_dir, data_file=None, box_z=None, db_conn: Optional[sqlite3.Connection] = None):
    """
    # ... (docstring remains the same) ...
    Returns:
        str or None: Path (relative to run_dir) to the saved plot file, or None if plotting failed.
    """
    module_name = "core_analysis_visualization_com" # KDE is only for COM
    start_time = time.time() # Start timer

    # Set up logging
    logger = setup_system_logger(run_dir)
    if logger is None:
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to setup system logger for {run_dir}. Using module logger.")

    # --- CHECK for db_conn ---
    if db_conn is None:
        logger.error("Database connection (db_conn) is None. Cannot generate KDE plot or register products.")
        return None

    # --- Register Module Start ---
    register_module(db_conn, module_name, status='running')
    # Note: We will update status to success/failed at the end

    # Define output directory
    output_dir = os.path.join(run_dir, "core_analysis")
    os.makedirs(output_dir, exist_ok=True)

    # Set up data file path if not provided
    if data_file is None:
        db_path = get_product_path(db_conn, "csv", "data", "com_stability_filtered")
        if db_path:
            data_file = os.path.join(run_dir, db_path)
        else:
            data_file = os.path.join(output_dir, "COM_Stability_Filtered.csv")
            if not os.path.exists(data_file):
                error_msg = f"COM distance data file not found: {data_file}"
                logger.error(error_msg)
                update_module_status(db_conn, module_name, 'failed', error_message=error_msg)
                return None

    # Load data from CSV
    try:
        df = pd.read_csv(data_file)
        if "COM_Distance_Raw" not in df.columns:
            error_msg = "Missing required column 'COM_Distance_Raw' in COM data file"
            logger.error(error_msg)
            update_module_status(db_conn, module_name, 'failed', error_message=error_msg)
            return None
        com_distances = df["COM_Distance_Raw"].values
    except Exception as e:
        error_msg = f"Failed to load data from {data_file}: {e}"
        logger.error(error_msg, exc_info=True)
        update_module_status(db_conn, module_name, 'failed', error_message=error_msg)
        return None

    plot_success = False # Flag to track plot generation
    save_rel_path = None # Initialize path

    try:
        fig, ax1 = plt.subplots(1, 1, figsize=(7, 6))
        ylabel = 'COM Distance (Å)'
        xlabel_kde = 'Density'
        finite_data = com_distances[np.isfinite(com_distances)]
        peak_values = []

        plot_filename = "COM_Stability_KDE_Analysis.png"
        save_path = os.path.join(output_dir, plot_filename)
        save_rel_path = os.path.relpath(save_path, run_dir)

        if len(finite_data) < 5:
            ax1.text(0.5, 0.5, 'Insufficient finite data for KDE/Histogram', ha='center', va='center',
                     fontsize=STYLE['font_sizes']['annotation'])
            logger.warning("Insufficient finite data for KDE plot generation.")
            plot_success = False # Consider this a failure to generate meaningful plot
            description = "COM KDE analysis plot (insufficient data)"
        else:
            try:
                kde = gaussian_kde(finite_data)
                data_min, data_max = np.min(finite_data), np.max(finite_data)
                padding = (data_max - data_min) * 0.1
                y_grid = np.linspace(data_min - padding, data_max + padding, 200)
                density = kde(y_grid)
                peaks, _ = find_peaks(density, height=np.max(density)*0.05, distance=max(5, len(y_grid)//50))
                peak_values = y_grid[peaks]

                ax1.hist(finite_data, bins=30, density=True, alpha=0.5, color=STYLE['chain_colors'].get('A', '#AEC6CF'), orientation='horizontal')
                ax1.plot(density, y_grid, color=STYLE['bright_colors'].get('B', '#FF6B81'), linewidth=STYLE['line_width'])
                for peak in peak_values:
                    ax1.axhline(y=peak, color='green', linestyle='--', alpha=0.7, linewidth=1.0)

                ax1.set_xlabel(xlabel_kde, fontsize=STYLE['font_sizes']['axis_label'])
                ax1.set_ylabel(ylabel, fontsize=STYLE['font_sizes']['axis_label'])
                ax1.tick_params(axis='both', labelsize=STYLE['font_sizes']['tick_label'])
                ax1.grid(True, alpha=STYLE['grid']['alpha'], color=STYLE['grid']['color'])

                state_info = ", ".join([f"{x:.2f} Å" for x in peak_values])
                logger.info(f"KDE analysis detected potential COM distance states at: {state_info}")
                plot_success = True
                description = "COM KDE analysis plot (Distribution only)"

            except Exception as e_kde:
                ax1.text(0.5, 0.5, f'KDE Failed:\n{e_kde}', ha='center', va='center', transform=ax1.transAxes, fontsize=STYLE['font_sizes']['annotation'])
                logger.error(f"KDE plot generation failed: {e_kde}", exc_info=True)
                plot_success = False
                description = "COM KDE analysis plot (Generation Failed)"

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved COM KDE analysis plot to {save_path}")

        # --- Update Status and Register Product ---
        exec_time = time.time() - start_time
        final_status = 'success' if plot_success else 'failed'
        update_module_status(db_conn, module_name, final_status, execution_time=exec_time, error_message="KDE plot failed" if not plot_success else None)
        register_product(db_conn, module_name, "png", "plot", save_rel_path, subcategory="kde_analysis", description=description)

        return save_rel_path if plot_success else None # Return path only on success

    except Exception as e:
        error_msg = f"Error generating COM KDE analysis plot: {e}"
        logger.error(error_msg, exc_info=True)
        if 'fig' in locals() and plt.fignum_exists(fig.number): plt.close(fig)
        exec_time = time.time() - start_time
        update_module_status(db_conn, module_name, 'failed', execution_time=exec_time, error_message=error_msg)
        return None
