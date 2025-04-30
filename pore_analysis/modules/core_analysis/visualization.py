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
import sqlite3 # Added for type hinting db_conn
from typing import Optional # Added for type hinting db_conn
from pore_analysis.core.plotting_style import STYLE, setup_style

# Import from other modules
try:
    from pore_analysis.core.logging import setup_system_logger
    from pore_analysis.core.database import (
        connect_db, register_product, get_product_path, get_module_status
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
# MODIFIED: Added db_conn: Optional[sqlite3.Connection] = None argument
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
        dict: A dictionary of plot paths keyed by plot type
    """
    # Set up logging
    logger = setup_system_logger(run_dir)
    if logger is None:
        logger = logging.getLogger()
        logger.error(f"Failed to setup system logger for {run_dir}. Using root logger.")

    # MODIFIED: Use passed db_conn if available, otherwise connect
    local_db_conn = False
    if db_conn is None:
        db_conn = connect_db(run_dir)
        if db_conn is None:
            logger.error(f"Failed to connect to database for {run_dir}")
            return {}
        local_db_conn = True # Flag that we opened it locally

    # Determine module name based on is_gg
    module_name = "core_analysis_visualization_g_g" if is_gg else "core_analysis_visualization_com"

    # Check if filtering was successful (prerequisite)
    filter_status = get_module_status(db_conn, "core_analysis_filtering")
    if filter_status != "success":
        logger.error(f"Cannot generate plots - filtering module status is: {filter_status}")
        if local_db_conn: db_conn.close() # Close connection if opened locally
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
                logger.error(f"Distance data file not found: {data_file}")
                if local_db_conn: db_conn.close() # Close connection if opened locally
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
                logger.error(f"Missing required columns in G-G data file: {missing_cols}")
                if local_db_conn: db_conn.close() # Close connection if opened locally
                return {}
            data1_raw_col = "G_G_Distance_AC_Raw"
            data2_raw_col = "G_G_Distance_BD_Raw"
            data1_filt_col = "G_G_Distance_AC_Filt"
            data2_filt_col = "G_G_Distance_BD_Filt"
            # Colors based on style guide mapping A->Blue, B->Pink/Red
            color_ac_pastel = STYLE['chain_colors']['B'] # Pastel Pink for A:C (like Red)
            color_ac_bright = STYLE['bright_colors']['B'] # Bright Pink for A:C (like Red)
            color_bd_pastel = STYLE['chain_colors']['A'] # Pastel Blue for B:D
            color_bd_bright = STYLE['bright_colors']['A'] # Bright Blue for B:D
        else: # COM plots
            required_cols = ["Time (ns)", "COM_Distance_Raw", "COM_Distance_Filt"]
            if not all(col in df.columns for col in required_cols):
                missing_cols = [col for col in required_cols if col not in df.columns]
                logger.error(f"Missing required columns in COM data file: {missing_cols}")
                if local_db_conn: db_conn.close() # Close connection if opened locally
                return {}
            data1_raw_col = "COM_Distance_Raw"
            data2_raw_col = None # Only one dataset for COM
            data1_filt_col = "COM_Distance_Filt"
            data2_filt_col = None
            # Use default pastel/bright colors for COM
            raw_colors = [STYLE['chain_colors']['A']] # e.g., Pastel Blue
            filt_colors = [STYLE['bright_colors']['A']] # e.g., Bright Blue

    except Exception as e:
        logger.error(f"Failed to load data from {data_file}: {e}", exc_info=True)
        if local_db_conn: db_conn.close() # Close connection if opened locally
        return {}

    plot_paths = {}

    try:
        time_points = df["Time (ns)"].values
        raw_data1 = df[data1_raw_col].values
        raw_data2 = df[data2_raw_col].values if data2_raw_col else None
        filtered_data1 = df[data1_filt_col].values
        filtered_data2 = df[data2_filt_col].values if data2_filt_col else None

        # --- G-G PLOTTING (New two-panel version) ---
        if is_gg:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=True) # Share Y axis

            # Left Panel: A:C Comparison
            ax1.plot(time_points, raw_data1, color=color_ac_pastel, alpha=0.7, label='Raw A:C', linewidth=STYLE['line_width'])
            ax1.plot(time_points, filtered_data1, color=color_ac_bright, label='Filtered A:C', linewidth=STYLE['line_width'])
            ax1.set_ylabel('G-G Distance (Å)', fontsize=STYLE['font_sizes']['axis_label'])
            ax1.set_xlabel('Time (ns)', fontsize=STYLE['font_sizes']['axis_label'])
            ax1.grid(True, alpha=STYLE['grid']['alpha'], color=STYLE['grid']['color'])
            ax1.legend(fontsize=STYLE['font_sizes']['annotation']) # Font size controlled by STYLE
            ax1.tick_params(axis='both', labelsize=STYLE['font_sizes']['tick_label']) # Font size controlled by STYLE

            # Right Panel: B:D Comparison
            if raw_data2 is not None and filtered_data2 is not None:
                ax2.plot(time_points, raw_data2, color=color_bd_pastel, alpha=0.7, label='Raw B:D', linewidth=STYLE['line_width'])
                ax2.plot(time_points, filtered_data2, color=color_bd_bright, label='Filtered B:D', linewidth=STYLE['line_width'])
                ax2.set_xlabel('Time (ns)', fontsize=STYLE['font_sizes']['axis_label'])
                ax2.grid(True, alpha=STYLE['grid']['alpha'], color=STYLE['grid']['color'])
                ax2.legend(fontsize=STYLE['font_sizes']['annotation']) # Font size controlled by STYLE
                ax2.tick_params(axis='both', labelsize=STYLE['font_sizes']['tick_label']) # Font size controlled by STYLE
            else:
                 ax2.text(0.5, 0.5, 'B:D data not available', ha='center', va='center', transform=ax2.transAxes)

            # Adjust layout and save
            plt.tight_layout()
            subunit_comp_path = os.path.join(output_dir, f"{output_prefix}_Distance_Subunit_Comparison.png")
            plt.savefig(subunit_comp_path, dpi=150)
            plt.close(fig)
            logger.info(f"Saved G-G subunit comparison plot to {subunit_comp_path}")
            plot_paths["subunit_comparison"] = subunit_comp_path # Store the path

            # Register the new plot in the database
            register_product(
                db_conn, # Use db_conn here
                module_name,
                "png",
                "plot",
                os.path.relpath(subunit_comp_path, run_dir), # Use relative path
                "subunit_comparison",
                "G-G Distance Raw vs Filtered Comparison by Subunit Pair (A:C Left, B:D Right)"
            )

        # --- COM PLOTTING (Only Raw vs Filtered Comparison) ---
        else: # if not is_gg (COM analysis)
            if filtered_data1 is not None:
                fig_comp, ax_comp = plt.subplots(figsize=(10, 6)) # Single plot figure
                ax_comp.plot(time_points, raw_data1, color=raw_colors[0], alpha=0.6, label='Raw Data', linewidth=STYLE['line_width'])
                ax_comp.plot(time_points, filtered_data1, color=filt_colors[0], label='Filtered Data', linewidth=STYLE['line_width'])
                ax_comp.set_ylabel('COM Distance (Å)', fontsize=STYLE['font_sizes']['axis_label'])
                ax_comp.set_xlabel('Time (ns)', fontsize=STYLE['font_sizes']['axis_label'])
                ax_comp.grid(True, alpha=STYLE['grid']['alpha'], color=STYLE['grid']['color'])
                ax_comp.legend(fontsize=STYLE['font_sizes']['annotation']) # Font size controlled by STYLE
                ax_comp.tick_params(axis='both', labelsize=STYLE['font_sizes']['tick_label']) # Font size controlled by STYLE
                plt.tight_layout()

                comparison_path = os.path.join(output_dir, f"{output_prefix}_Stability_Comparison.png")
                plt.savefig(comparison_path, dpi=150)
                plt.close(fig_comp)
                plot_paths["comparison"] = comparison_path # Store path with key 'comparison'
                logger.info(f"Saved COM raw vs filtered comparison plot to {comparison_path}")

                # Register the comparison plot
                register_product(
                    db_conn, # Use db_conn here
                    module_name,
                    "png",
                    "plot",
                    os.path.relpath(comparison_path, run_dir), # Use relative path
                    "comparison",
                    "Comparison of raw and filtered COM Distance"
                )
            else:
                logger.warning("Filtered COM data not available, cannot generate comparison plot.")

        if local_db_conn: db_conn.close() # Close connection if opened locally
        return plot_paths # Return dict containing paths to generated plots

    except Exception as e:
        logger.error(f"Error generating distance plots: {e}", exc_info=True)
        if local_db_conn: db_conn.close() # Ensure closure on error too
        return plot_paths


# --- plot_kde_analysis function ---
# MODIFIED: Added db_conn: Optional[sqlite3.Connection] = None argument
def plot_kde_analysis(run_dir, data_file=None, box_z=None, db_conn: Optional[sqlite3.Connection] = None):
    """
    Create KDE (Kernel Density Estimation) plot for COM distances to identify
    distinct stability states with consistent styling.
    Generates a single plot showing the histogram (horizontal) and KDE (vertical, red).
    Titles are intentionally omitted. Font sizes are controlled by STYLE dictionary.

    Args:
        run_dir (str): Directory path for the run data.
        data_file (str, optional): Path to the CSV file containing COM distance data.
                                  If None, the path will be retrieved from the database.
        box_z (float, optional): Box Z dimension. Not used in current implementation but kept for API compatibility.
        db_conn (sqlite3.Connection, optional): Existing database connection. If None, connects automatically.

    Returns:
        str or None: Path to the saved plot file, or None if plotting failed.
    """
    # Set up logging
    logger = setup_system_logger(run_dir)
    if logger is None:
        logger = logging.getLogger()
        logger.error(f"Failed to setup system logger for {run_dir}. Using root logger.")

    # MODIFIED: Use passed db_conn if available, otherwise connect
    local_db_conn = False
    if db_conn is None:
        db_conn = connect_db(run_dir)
        if db_conn is None:
            logger.error(f"Failed to connect to database for {run_dir}")
            return None
        local_db_conn = True # Flag that we opened it locally

    # Define output directory
    output_dir = os.path.join(run_dir, "core_analysis")
    os.makedirs(output_dir, exist_ok=True)

    # Set up data file path if not provided
    if data_file is None:
        # Get file path from database
        db_path = get_product_path(db_conn, "csv", "data", "com_stability_filtered") # Use db_conn here
        if db_path:
            data_file = os.path.join(run_dir, db_path)
        else:
            # Fallback path if not found in database
            data_file = os.path.join(output_dir, "COM_Stability_Filtered.csv")

            if not os.path.exists(data_file):
                logger.error(f"COM distance data file not found: {data_file}")
                if local_db_conn: db_conn.close() # Close connection if opened locally
                return None

    # Load data from CSV
    try:
        df = pd.read_csv(data_file)

        # Use Raw COM distance for KDE as filtering might obscure states
        if "COM_Distance_Raw" not in df.columns:
            logger.error("Missing required column 'COM_Distance_Raw' in COM data file")
            if local_db_conn: db_conn.close() # Close connection if opened locally
            return None

        com_distances = df["COM_Distance_Raw"].values

    except Exception as e:
        logger.error(f"Failed to load data from {data_file}: {e}", exc_info=True)
        if local_db_conn: db_conn.close() # Close connection if opened locally
        return None

    try:
        # Create a single plot figure and axis
        fig, ax1 = plt.subplots(1, 1, figsize=(7, 6)) # Single panel, adjust size

        ylabel = 'COM Distance (Å)'
        xlabel_kde = 'Density'

        finite_data = com_distances[np.isfinite(com_distances)]
        peak_values = []  # Initialize peak_values here

        if len(finite_data) < 5:
            # Handle insufficient data - display text message
            ax1.text(0.5, 0.5, 'Insufficient finite data for KDE/Histogram', ha='center', va='center',
                     fontsize=STYLE['font_sizes']['annotation'])

            plt.tight_layout()
            plot_filename = "COM_Stability_KDE_Analysis.png" # Save even if insufficient data
            save_path = os.path.join(output_dir, plot_filename)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig) # Close the figure
            logger.warning("Insufficient finite data for KDE plot generation.")

            # Register the plot in the database
            register_product(
                db_conn, # Use db_conn here
                "core_analysis_visualization_com",
                "png",
                "plot",
                os.path.relpath(save_path, run_dir), # Ensure relative path
                "kde_analysis",
                "COM KDE analysis plot (insufficient data)"
            )
            if local_db_conn: db_conn.close() # Close connection if opened locally
            return save_path


        # --- Plot Histogram and KDE on the single axis (ax1) ---
        try:
            kde = gaussian_kde(finite_data)
            # Generate grid based on data range for Y axis
            data_min, data_max = np.min(finite_data), np.max(finite_data)
            padding = (data_max - data_min) * 0.1 # Add some padding
            y_grid = np.linspace(data_min - padding, data_max + padding, 200) # Grid for Y
            density = kde(y_grid) # Calculate density along Y grid

            # Adjust peak finding parameters if needed
            peaks, _ = find_peaks(density, height=np.max(density)*0.05, distance=max(5, len(y_grid)//50))
            peak_values = y_grid[peaks]

            # Plot histogram horizontally
            ax1.hist(finite_data, bins=30, density=True, alpha=0.5, color=STYLE['chain_colors']['A'],
                     orientation='horizontal') # Horizontal orientation
            # Plot KDE line with density on X, distance on Y, using RED color
            ax1.plot(density, y_grid, color=STYLE['bright_colors']['B'], linewidth=STYLE['line_width']) # density on X, y_grid on Y

            # Add horizontal lines for peaks
            for peak in peak_values:
                ax1.axhline(y=peak, color='green', linestyle='--', alpha=0.7, linewidth=1.0)

            ax1.set_xlabel(xlabel_kde, fontsize=STYLE['font_sizes']['axis_label']) # Density on X
            ax1.set_ylabel(ylabel, fontsize=STYLE['font_sizes']['axis_label']) # Distance on Y
            ax1.tick_params(axis='both', labelsize=STYLE['font_sizes']['tick_label']) # Font size controlled by STYLE
            ax1.grid(True, alpha=STYLE['grid']['alpha'], color=STYLE['grid']['color'])


            # Add info to log
            state_info = ", ".join([f"{x:.2f} Å" for x in peak_values])
            logger.info(f"KDE analysis detected potential COM distance states at: {state_info}")

        except Exception as e:
            ax1.text(0.5, 0.5, f'KDE Failed:\n{e}', ha='center', va='center', transform=ax1.transAxes,
                    fontsize=STYLE['font_sizes']['annotation'])
            logger.error(f"KDE plot generation failed: {e}", exc_info=True)

        plt.tight_layout() # Adjust layout for single plot

        plot_filename = "COM_Stability_KDE_Analysis.png"
        save_path = os.path.join(output_dir, plot_filename)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig) # Close the figure

        logger.info(f"Saved COM KDE analysis plot to {save_path}")

        # Register the plot in the database (same registration as before)
        register_product(
            db_conn, # Use db_conn here
            "core_analysis_visualization_com",
            "png",
            "plot",
            os.path.relpath(save_path, run_dir), # Ensure relative path
            "kde_analysis",
            "COM KDE analysis plot (Distribution only)" # Updated description
        )

        if local_db_conn: db_conn.close() # Close connection if opened locally
        return save_path

    except Exception as e:
        logger.error(f"Error generating COM KDE analysis plot: {e}", exc_info=True)
        if local_db_conn: db_conn.close() # Ensure closure on error too
        return None