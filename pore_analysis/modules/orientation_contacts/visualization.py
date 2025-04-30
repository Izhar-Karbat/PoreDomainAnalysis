# pore_analysis/modules/orientation_contacts/visualization.py
"""
Visualization functions for toxin orientation, rotation, and contacts analysis.
Generates plots based on data stored in the database.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import MDAnalysis as mda # Needed for get_residue_info fallback
import time
import sqlite3
from typing import Optional

# Import from other modules
try:
    # Import the centralized style definitions
    from pore_analysis.core.plotting_style import STYLE, setup_style
    from pore_analysis.core.logging import setup_system_logger
    from pore_analysis.core.database import (
        connect_db, register_module, register_product, get_product_path,
        update_module_status, get_simulation_metadata, get_module_status
    )
    # Import computation helper only if needed for residue labels fallback
    # from .computation import get_residue_info # If get_residue_info isn't moved to core.utils
except ImportError as e:
    print(f"Error importing dependency modules in orientation_contacts/visualization.py: {e}")
    raise

logger = logging.getLogger(__name__)

# Apply the standard plotting style
setup_style()

# --- Plotting Helper Functions (plot_orientation_data, create_contact_map_visualization, create_enhanced_focused_heatmap) ---
# ... (These functions remain unchanged) ...
def plot_orientation_data(time_points, orientation_angles, rotation_euler_angles, contact_counts, output_dir, db_conn, module_name):
    """
    Create plots for toxin orientation angle, rotation, and total contacts.
    Registers plots in the database. Styling applied via setup_style(). No titles set here.
    [Derived from: orientation_contacts.py]
    """
    run_dir = os.path.dirname(output_dir) # Infer run_dir

    # --- Orientation angle plot ---
    try:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(x=time_points, y=orientation_angles, ax=ax, linewidth=STYLE['line_width'])
        ax.set_xlabel('Time (ns)') # Fontsize set by rcParams
        ax.set_ylabel('Toxin-Channel Angle (°)')
        ax.tick_params(axis='both') # Fontsize set by rcParams
        ax.set_ylim(0, 90)
        ax.grid(True, linestyle=STYLE['grid']['linestyle'], alpha=STYLE['grid']['alpha'], color=STYLE['grid']['color'])
        plt.tight_layout()
        filename = "Toxin_Orientation_Angle.png"
        filepath = os.path.join(output_dir, filename)
        fig.savefig(filepath, dpi=150)
        plt.close(fig)
        register_product(db_conn, module_name, "png", "plot",
                         os.path.relpath(filepath, run_dir),
                         subcategory='orientation_angle', # MUST MATCH html.py query
                         description="Time series of Toxin-Channel Orientation Angle")
    except Exception as e:
         logger.error(f"Failed to create orientation angle plot: {e}", exc_info=True)

    # --- Rotation components plot ---
    try:
        fig, ax = plt.subplots(figsize=(10, 5))
        if rotation_euler_angles and len(rotation_euler_angles) == len(time_points):
            x_rot = [r[0] for r in rotation_euler_angles]
            y_rot = [r[1] for r in rotation_euler_angles]
            z_rot = [r[2] for r in rotation_euler_angles]
            # Using specific colors might be good if STYLE allows, otherwise default cycle
            sns.lineplot(x=time_points, y=x_rot, label='X', ax=ax, linewidth=STYLE['line_width']*0.75)
            sns.lineplot(x=time_points, y=y_rot, label='Y', ax=ax, linewidth=STYLE['line_width']*0.75)
            sns.lineplot(x=time_points, y=z_rot, label='Z', ax=ax, linewidth=STYLE['line_width']*0.75)
            ax.legend() # Fontsize set by rcParams
        else:
             ax.text(0.5, 0.5, 'Rotation data unavailable.', ha='center', va='center', transform=ax.transAxes)
        ax.set_xlabel('Time (ns)')
        ax.set_ylabel('Rotation (°)')
        ax.tick_params(axis='both')
        ax.grid(True, linestyle=STYLE['grid']['linestyle'], alpha=STYLE['grid']['alpha'], color=STYLE['grid']['color'])
        plt.tight_layout()
        filename = "Toxin_Rotation_Components.png"
        filepath = os.path.join(output_dir, filename)
        fig.savefig(filepath, dpi=150)
        plt.close(fig)
        register_product(db_conn, module_name, "png", "plot",
                         os.path.relpath(filepath, run_dir),
                         subcategory='rotation_components', # MUST MATCH html.py query
                         description="Time series of Toxin Rotation Components (X, Y, Z)")
    except Exception as e:
         logger.error(f"Failed to create rotation components plot: {e}", exc_info=True)

    # --- Contact count plot ---
    try:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(x=time_points, y=contact_counts, ax=ax, linewidth=STYLE['line_width'])
        ax.set_xlabel('Time (ns)')
        ax.set_ylabel('Atom Contacts (< 3.5 Å)') # Adjusted label
        ax.tick_params(axis='both')
        ax.grid(True, linestyle=STYLE['grid']['linestyle'], alpha=STYLE['grid']['alpha'], color=STYLE['grid']['color'])
        finite_contacts = np.array(contact_counts)[np.isfinite(contact_counts)]
        y_max = np.nanmax(finite_contacts) * 1.1 if len(finite_contacts)>0 else 10
        ax.set_ylim(bottom=0, top=max(1, y_max)) # Ensure top > 0
        plt.tight_layout()
        filename = "Toxin_Channel_Contacts.png"
        filepath = os.path.join(output_dir, filename)
        fig.savefig(filepath, dpi=150)
        plt.close(fig)
        register_product(db_conn, module_name, "png", "plot",
                         os.path.relpath(filepath, run_dir),
                         subcategory='channel_contacts', # MUST MATCH html.py query
                         description="Time series of Total Toxin-Channel Atom Contacts")
    except Exception as e:
         logger.error(f"Failed to create contact count plot: {e}", exc_info=True)


def create_contact_map_visualization(avg_contact_map_df, output_dir, db_conn, module_name):
    """
    Create heatmap visualization for the full average toxin-channel residue contacts map.
    Assumes avg_contact_map_df has display labels as index/columns.
    Registers plot in the database. Styling applied via setup_style(). No titles set here.
    [Derived from: orientation_contacts.py]
    """
    run_dir = os.path.dirname(output_dir) # Infer run_dir
    if avg_contact_map_df is None or avg_contact_map_df.empty:
        logger.warning("Full contact map data is empty. Skipping visualization.")
        return

    display_toxin_labels = avg_contact_map_df.index.tolist()
    display_channel_labels = avg_contact_map_df.columns.tolist()
    avg_contact_map = avg_contact_map_df.to_numpy() # Get numpy array for heatmap

    # --- Dynamic Figure Sizing ---
    height = max(6, len(display_toxin_labels) * 0.15 + 1)
    width = max(8, len(display_channel_labels) * 0.10 + 1)
    max_dim = 40
    height = min(height, max_dim)
    width = min(width, max_dim)
    logger.debug(f"Full contact map figure size: {width:.1f} x {height:.1f} inches")

    try:
        fig, ax = plt.subplots(figsize=(width, height))
        cmap = "viridis"
        sns.heatmap(avg_contact_map, cmap=cmap, ax=ax, linewidths=0.1, linecolor='lightgrey',
                    cbar_kws={'label': 'Average Contact Frequency', 'shrink': 0.7},
                    xticklabels=False, yticklabels=False) # No default labels

        ax.set_ylabel('Toxin Residues') # Fontsize set by rcParams
        ax.set_xlabel('Channel Residues') # Fontsize set by rcParams

        # --- Set Ticks using Display Labels ---
        xtick_divisor = 45 if width > 20 else 30
        ytick_divisor = 60 if height > 25 else 40
        xtick_step = max(1, len(display_channel_labels) // xtick_divisor)
        ytick_step = max(1, len(display_toxin_labels) // ytick_divisor)
        logger.debug(f"Full map tick steps: x={xtick_step}, y={ytick_step}")

        xtick_indices = np.arange(len(display_channel_labels))[::xtick_step]
        ytick_indices = np.arange(len(display_toxin_labels))[::ytick_step]

        if len(xtick_indices) > 0 and len(ytick_indices) > 0 :
            ax.set_xticks(xtick_indices + 0.5)
            ax.set_xticklabels([display_channel_labels[i] for i in xtick_indices], rotation=90, fontsize=STYLE['font_sizes']['tick_label'] * 0.7) # Smaller font for ticks
            ax.set_yticks(ytick_indices + 0.5)
            ax.set_yticklabels([display_toxin_labels[i] for i in ytick_indices], rotation=0, fontsize=STYLE['font_sizes']['tick_label'] * 0.7)
        else: logger.warning("Could not set full map tick labels.")

        plt.tight_layout(pad=1.2)
        filename = "Toxin_Channel_Residue_Contact_Map_Full.png"
        filepath = os.path.join(output_dir, filename)
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        register_product(db_conn, module_name, "png", "plot",
                         os.path.relpath(filepath, run_dir),
                         subcategory='contact_map_full', # MUST MATCH html.py query
                         description="Full Residue Contact Frequency Heatmap")
        logger.info(f"Saved full contact map visualization to {filepath}")
    except Exception as e:
        logger.error(f"Failed to create full contact map plot: {e}", exc_info=True)
        if 'fig' in locals() and plt.fignum_exists(fig.number): plt.close(fig)


def create_enhanced_focused_heatmap(avg_contact_map_df, output_dir, db_conn, module_name,
                                      top_n_toxin=10, top_n_channel=15):
    """
    Create and save an enhanced heatmap focusing only on top interacting residues.
    Assumes avg_contact_map_df has display labels as index/columns.
    Registers plot in the database. Styling applied via setup_style(). No titles set here.
    [Derived from: orientation_contacts.py]
    """
    run_dir = os.path.dirname(output_dir) # Infer run_dir
    if avg_contact_map_df is None or avg_contact_map_df.empty:
        logger.warning("Focused contact map data is empty. Skipping visualization.")
        return

    try:
        # Calculate interaction sums
        toxin_sums = avg_contact_map_df.sum(axis=1)
        channel_sums = avg_contact_map_df.sum(axis=0)

        # Select top residues
        sorted_toxin = sorted([item for item in toxin_sums.items() if item[1] > 1e-6], key=lambda x: x[1], reverse=True)
        sorted_channel = sorted([item for item in channel_sums.items() if item[1] > 1e-6], key=lambda x: x[1], reverse=True)

        actual_top_n_toxin = min(top_n_toxin, len(sorted_toxin))
        actual_top_n_channel = min(top_n_channel, len(sorted_channel))

        if actual_top_n_toxin == 0 or actual_top_n_channel == 0:
            logger.warning("No significantly interacting residues found for focused heatmap.")
            return

        top_toxin_labels = [item[0] for item in sorted_toxin[:actual_top_n_toxin]]
        top_channel_labels = [item[0] for item in sorted_channel[:actual_top_n_channel]]

        # Create the focused dataframe using .loc
        focused_df = avg_contact_map_df.loc[top_toxin_labels, top_channel_labels]
        focused_matrix = focused_df.to_numpy() # Get numpy array for heatmap

        # Create and save the heatmap
        height = max(6, actual_top_n_toxin * 0.4 + 1)
        width = max(7, actual_top_n_channel * 0.4 + 1)
        max_dim = 30
        height = min(height, max_dim)
        width = min(width, max_dim)
        logger.debug(f"Focused map figure size: {width:.1f} x {height:.1f} inches")

        fig, ax = plt.subplots(figsize=(width, height))
        sns.heatmap(focused_matrix, cmap="viridis", ax=ax,
                    linewidths=0.5, linecolor='lightgrey',
                    cbar_kws={'label': 'Average Contact Frequency', 'shrink': 0.8},
                    annot=True, fmt=".2f", annot_kws={"size": 8},
                    xticklabels=focused_df.columns, # Use labels from focused df
                    yticklabels=focused_df.index)  # Use labels from focused df

        # Note: No ax.set_title() here
        plt.xlabel('Channel Residues') # Fontsize set by rcParams
        plt.ylabel('Toxin Residues') # Fontsize set by rcParams
        # Adjust tick label sizes directly if needed, rcParams might be sufficient
        plt.xticks(rotation=45, ha='right', fontsize=STYLE['font_sizes']['tick_label'] * 0.8)
        plt.yticks(rotation=0, fontsize=STYLE['font_sizes']['tick_label'] * 0.8)
        plt.tight_layout(pad=1.5)

        filename = "Toxin_Channel_Residue_Contact_Map_Focused.png"
        filepath = os.path.join(output_dir, filename)
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        register_product(db_conn, module_name, "png", "plot",
                         os.path.relpath(filepath, run_dir),
                         subcategory='contact_map_focused', # MUST MATCH html.py query
                         description=f"Focused ({actual_top_n_toxin}x{actual_top_n_channel}) Residue Contact Frequency Heatmap")
        logger.info(f"Saved focused contact map visualization to {filepath}")

    except Exception as e:
        logger.error(f"Error generating focused contact map plot: {e}", exc_info=True)
        if 'fig' in locals() and plt.fignum_exists(fig.number): plt.close(fig)


# --- Main Visualization Function ---
# MODIFIED: Commented out the call to create_contact_map_visualization
def generate_orientation_plots(run_dir, db_conn: Optional[sqlite3.Connection] = None):
    """
    Generates all plots for the orientation and contacts analysis module.
    Retrieves necessary data paths from the database.

    Args:
        run_dir (str): Path to the specific run directory.
        db_conn (sqlite3.Connection, optional): Database connection. If None, connects automatically.

    Returns:
        dict: Contains status ('success', 'skipped', 'failed') and error message if applicable.
    """
    module_name = "orientation_analysis_visualization"
    start_time = time.time()

    # Set up logging & DB connection
    logger = setup_system_logger(run_dir)
    if logger is None: logger = logging.getLogger()

    local_db_conn = False
    if db_conn is None:
        db_conn = connect_db(run_dir)
        if db_conn is None:
            logger.error(f"{module_name}: Failed to connect to database for {run_dir}")
            return {'status': 'failed', 'error': 'Database connection failed'}
        local_db_conn = True

    # Check if computational step was skipped or failed
    comp_status = get_module_status(db_conn, "orientation_analysis")
    if comp_status != 'success':
        logger.info(f"{module_name}: Skipping visualization as computation status is '{comp_status}'.")
        register_module(db_conn, module_name, status='skipped')
        update_module_status(db_conn, module_name, 'skipped')
        if local_db_conn: db_conn.close()
        return {'status': 'skipped', 'reason': f'Computation status was {comp_status}'}

    # Register module start
    register_module(db_conn, module_name, status='running')

    # Define output directory
    output_dir = os.path.join(run_dir, "orientation_contacts")
    os.makedirs(output_dir, exist_ok=True)

    # --- Get Data Paths from Database ---
    ts_data_path_rel = get_product_path(db_conn, product_type='csv', category='data', subcategory='orientation_timeseries', module_name='orientation_analysis')
    freq_data_path_rel = get_product_path(db_conn, product_type='csv', category='data', subcategory='residue_frequency', module_name='orientation_analysis')

    # --- Load Data ---
    orientation_df = None
    residue_freq_df = None
    plots_generated = False

    if ts_data_path_rel:
        ts_data_path = os.path.join(run_dir, ts_data_path_rel)
        if os.path.exists(ts_data_path):
            try:
                orientation_df = pd.read_csv(ts_data_path)
                logger.info(f"{module_name}: Loaded time series data from {ts_data_path}")
            except Exception as e:
                logger.error(f"{module_name}: Failed to load time series data from {ts_data_path}: {e}")
        else: logger.warning(f"{module_name}: Time series data file not found at registered path: {ts_data_path}")
    else: logger.warning(f"{module_name}: Path for orientation time series data not found in database.")

    if freq_data_path_rel:
        freq_data_path = os.path.join(run_dir, freq_data_path_rel)
        if os.path.exists(freq_data_path):
            try:
                residue_freq_df = pd.read_csv(freq_data_path, index_col=0)
                logger.info(f"{module_name}: Loaded residue frequency data from {freq_data_path}")
            except Exception as e:
                logger.error(f"{module_name}: Failed to load residue frequency data from {freq_data_path}: {e}")
        else: logger.warning(f"{module_name}: Residue frequency data file not found at registered path: {freq_data_path}")
    else: logger.warning(f"{module_name}: Path for residue frequency data not found in database.")


    # --- Generate Plots ---
    if orientation_df is not None and not orientation_df.empty:
        try:
            required_cols = ['Time (ns)', 'Orientation_Angle', 'Rotation_X', 'Rotation_Y', 'Rotation_Z', 'Total_Atom_Contacts']
            if all(col in orientation_df.columns for col in required_cols):
                plot_orientation_data(
                    time_points=orientation_df['Time (ns)'].values,
                    orientation_angles=orientation_df['Orientation_Angle'].values,
                    rotation_euler_angles=list(zip(orientation_df['Rotation_X'], orientation_df['Rotation_Y'], orientation_df['Rotation_Z'])),
                    contact_counts=orientation_df['Total_Atom_Contacts'].values,
                    output_dir=output_dir,
                    db_conn=db_conn,
                    module_name=module_name
                )
                plots_generated = True
            else:
                missing = [col for col in required_cols if col not in orientation_df.columns]
                logger.warning(f"{module_name}: Skipping time series plots - missing columns in orientation_data.csv: {missing}")
        except Exception as e:
            logger.error(f"{module_name}: Error during time series plot generation: {e}", exc_info=True)

    if residue_freq_df is not None and not residue_freq_df.empty:
        try:
            # MODIFICATION START: Skip Full Contact Map generation
            # logger.info(f"{module_name}: Skipping full contact map PNG generation as requested.")
            # create_contact_map_visualization(
            #     avg_contact_map_df=residue_freq_df,
            #     output_dir=output_dir,
            #     db_conn=db_conn,
            #     module_name=module_name
            # )
            # plots_generated = True # Set flag even if skipped
            # MODIFICATION END

            # Keep Focused Map generation
            create_enhanced_focused_heatmap(
                avg_contact_map_df=residue_freq_df,
                output_dir=output_dir,
                db_conn=db_conn,
                module_name=module_name
            )
            plots_generated = True
        except Exception as e:
            logger.error(f"{module_name}: Error during contact map plot generation: {e}", exc_info=True)

    # --- Finalize ---
    if not plots_generated:
        logger.warning(f"{module_name}: No plots were generated due to missing data or errors.")

    exec_time = time.time() - start_time
    update_module_status(db_conn, module_name, 'success', execution_time=exec_time)
    logger.info(f"{module_name}: Visualization completed in {exec_time:.2f} seconds.")

    if local_db_conn: db_conn.close()
    return {'status': 'success'}