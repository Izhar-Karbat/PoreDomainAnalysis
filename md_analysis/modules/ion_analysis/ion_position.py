"""
Ion position tracking module.

Functions for saving and visualizing ion positions
over the course of the trajectory.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Module logger
module_logger = logging.getLogger(__name__)

def save_ion_position_data(run_dir, time_points, ions_z_positions_abs, ion_indices, g1_reference):
    """
    Save K+ ion Z-position data (absolute and G1-centric) to CSV files.
    Properly handles NaN values for frames where ions are not in the filter.

    Args:
        run_dir (str): Directory to save the CSV files.
        time_points (np.ndarray): Array of time points in ns.
        ions_z_positions_abs (dict): {ion_idx: np.array(abs_z_positions)} with NaN when ion not in filter
        ion_indices (list): List of ion indices included in the dict.
        g1_reference (float): Absolute Z-coordinate of the G1 C-alpha plane.
    """
    if not ion_indices:
        module_logger.info("No tracked ion indices provided. Skipping ion position data saving.")
        return

    # --- Absolute Coordinates ---
    data_abs = {'Time (ns)': time_points}
    for ion_idx in ion_indices:
        if ion_idx in ions_z_positions_abs:
             data_abs[f'Ion_{ion_idx}_Z_Abs'] = ions_z_positions_abs[ion_idx]
        else: # Should not happen if ion_indices matches dict keys
             data_abs[f'Ion_{ion_idx}_Z_Abs'] = np.full(len(time_points), np.nan)
    df_abs = pd.DataFrame(data_abs)
    csv_path_abs = os.path.join(run_dir, 'K_Ion_Z_Positions_Absolute.csv')
    try:
        df_abs.to_csv(csv_path_abs, index=False, float_format='%.4f', na_rep='NaN')
        module_logger.info(f"Saved ion absolute Z positions to {csv_path_abs}")
    except Exception as e:
        module_logger.error(f"Failed to save absolute ion position CSV: {e}")

    # --- G1-Centric Coordinates ---
    data_g1 = {'Time (ns)': time_points}
    for ion_idx in ion_indices:
        if ion_idx in ions_z_positions_abs:
            # Calculate G1-centric positions (retain NaN values)
            data_g1[f'Ion_{ion_idx}_Z_G1Centric'] = ions_z_positions_abs[ion_idx] - g1_reference
        else:
            data_g1[f'Ion_{ion_idx}_Z_G1Centric'] = np.full(len(time_points), np.nan)
    df_g1 = pd.DataFrame(data_g1)
    csv_path_g1 = os.path.join(run_dir, 'K_Ion_Z_Positions_G1Centric.csv') # Explicit name
    try:
        df_g1.to_csv(csv_path_g1, index=False, float_format='%.4f', na_rep='NaN')
        module_logger.info(f"Saved ion G1-centric Z positions to {csv_path_g1}")
    except Exception as e:
        module_logger.error(f"Failed to save G1-centric ion position CSV: {e}")

    # --- Add Column for Ion Presence ---
    # Create a metadata version that includes a boolean column indicating when ion is in filter
    data_meta = {'Time (ns)': time_points}
    for ion_idx in ion_indices:
        if ion_idx in ions_z_positions_abs:
            # Add presence column (True when position is not NaN)
            is_present = ~np.isnan(ions_z_positions_abs[ion_idx])
            data_meta[f'Ion_{ion_idx}_InFilter'] = is_present
    df_meta = pd.DataFrame(data_meta)
    csv_path_meta = os.path.join(run_dir, 'K_Ion_Filter_Presence.csv')
    try:
        df_meta.to_csv(csv_path_meta, index=False)
        module_logger.info(f"Saved ion filter presence data to {csv_path_meta}")
    except Exception as e:
        module_logger.error(f"Failed to save ion presence CSV: {e}")

    # Save metadata about the reference frame
    meta_path = os.path.join(run_dir, 'K_Ion_Coordinate_Reference.txt')
    try:
        with open(meta_path, 'w') as f:
            f.write(f"# K+ ion position coordinate reference information\n")
            f.write(f"# Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"G1_C_alpha_reference_position_absolute_Z: {g1_reference:.4f} Å\n")
            f.write(f"Conversion_formula: Z_G1_centric = Z_Absolute - G1_C_alpha_reference_position_absolute_Z\n")
            f.write(f"# Note: NaN values indicate frames where ion is not in the filter region\n")
        module_logger.info(f"Saved coordinate reference info to {meta_path}")
    except Exception as e:
         module_logger.error(f"Failed to save coordinate reference file: {e}")

def plot_ion_positions(run_dir, time_points, ions_z_positions_abs, ion_indices, filter_sites, g1_reference, logger=None):
    """
    Plot K+ ion Z positions (G1-centric) over time and their density distribution.
    Properly handles NaN values for frames where ions are not in the filter.

    Args:
        run_dir (str): Directory to save the plots.
        time_points (np.ndarray): Time points in ns.
        ions_z_positions_abs (dict): {ion_idx: np.array(abs_z_positions)} with NaN when ion not in filter
        ion_indices (list): Indices of ions to plot.
        filter_sites (dict): Site positions relative to G1 C-alpha=0.
        g1_reference (float): Absolute Z of G1 C-alpha plane.
        logger (logging.Logger, optional): Logger instance. Defaults to module logger.
    """
    log = logger if logger else module_logger
    if not ion_indices:
        log.info("No tracked ion indices to plot.")
        return
    if filter_sites is None:
         log.warning("Filter sites data missing, plots will lack site lines.")
         filter_sites = {} # Use empty dict to avoid errors

    log.info("Generating K+ ion position plots...")

    # Convert absolute positions to G1-centric for plotting
    ions_z_g1 = {idx: ions_z_positions_abs[idx] - g1_reference for idx in ion_indices if idx in ions_z_positions_abs}

    # --- Combined Plot (Time Series + Density) ---
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8), sharey=True, # Share Y axis
                                        gridspec_kw={'width_ratios': [3, 1]})

        # Define site colors (consistent with visualization)
        site_colors = { 'S0': '#FF6347', 'S1': '#4169E1', 'S2': '#2E8B57',
                       'S3': '#BA55D3', 'S4': '#CD853F', 'Cavity': '#708090' }

        # Left subplot: Time series
        plot_colors = sns.color_palette("husl", len(ion_indices)) # Use a distinct color palette
        for i, ion_idx in enumerate(ion_indices):
            if ion_idx in ions_z_g1:
                # Plot only non-NaN values (when ion is in filter)
                ion_data = ions_z_g1[ion_idx]
                mask = np.isfinite(ion_data)
                if np.any(mask):  # Only plot if there's data
                    # Use masked time points and data values
                    t_points = time_points[mask]
                    z_values = ion_data[mask]
                    
                    # Connect points only if they're adjacent in time
                    segments = []
                    current_segment = []
                    
                    for j in range(len(t_points)):
                        if j > 0 and t_points[j] - t_points[j-1] > 1.5 * (time_points[1] - time_points[0]):
                            # Gap detected, finish current segment
                            if current_segment:
                                segments.append(current_segment)
                                current_segment = []
                        
                        current_segment.append((t_points[j], z_values[j]))
                    
                    # Add the last segment if it exists
                    if current_segment:
                        segments.append(current_segment)
                    
                    # Plot each segment separately
                    for segment in segments:
                        t_seg, z_seg = zip(*segment)
                        ax1.plot(t_seg, z_seg, color=plot_colors[i], linewidth=1.0, alpha=0.8)
                    
                    # Optionally, add scatter points to show discrete positions
                    # ax1.scatter(t_points, z_values, color=plot_colors[i], s=2, alpha=0.6)
                    
                    # Add label if few ions
                    if len(ion_indices) <= 10:
                        ax1.plot([], [], color=plot_colors[i], label=f'Ion {ion_idx}', linewidth=1.0)

        # Add site lines to left plot
        for site, z_pos_rel in filter_sites.items():
            color = site_colors.get(site, 'grey')
            ax1.axhline(y=z_pos_rel, color=color, linestyle='--', alpha=0.7, linewidth=1.2, zorder=1)
            ax1.text(0, z_pos_rel, site, va='center', ha='right', fontsize=9, fontweight='bold', color=color, zorder=2,
                     bbox=dict(facecolor='white', alpha=0.5, pad=0.1, edgecolor='none')) # Add background to label
        ax1.axhline(y=0, color='black', linestyle=':', alpha=0.9, linewidth=1.2, zorder=1) # G1 Ref line

        ax1.set_xlabel('Time (ns)', fontsize=14)
        ax1.set_ylabel('Z-Position relative to G1 Cα (Å)', fontsize=14)
        ax1.tick_params(axis='both', labelsize=10)
        ax1.grid(axis='y', linestyle=':', alpha=0.5, zorder=0)
        if len(ion_indices) <= 10: ax1.legend(fontsize='x-small')

        # Right subplot: Density
        # Collect all finite values across all ions (only when they're in filter)
        all_z_g1_flat = np.concatenate([arr[np.isfinite(arr)] for idx, arr in ions_z_g1.items() if np.any(np.isfinite(arr))])
        
        if len(all_z_g1_flat) > 10: # Need sufficient points for KDE
             sns.kdeplot(y=all_z_g1_flat, ax=ax2, color='black', fill=True, alpha=0.2, linewidth=1.5)
             ax2.set_xlabel('Density', fontsize=14)
        else:
             ax2.text(0.5, 0.5, 'Insufficient\ndata for\nDensity Plot', ha='center', va='center', transform=ax2.transAxes)
             ax2.set_xlabel('', fontsize=14)

        # Add site lines to right plot (without labels)
        for site, z_pos_rel in filter_sites.items():
             color = site_colors.get(site, 'grey')
             ax2.axhline(y=z_pos_rel, color=color, linestyle='--', alpha=0.7, linewidth=1.2, zorder=1)
        ax2.axhline(y=0, color='black', linestyle=':', alpha=0.9, linewidth=1.2, zorder=1) # G1 Ref line

        ax2.tick_params(axis='x', labelsize=10)
        ax2.grid(axis='y', linestyle=':', alpha=0.5, zorder=0)

        # Set shared Y limits based on site range or data range
        if filter_sites:
            site_values = list(filter_sites.values())
            y_min = min(site_values) - 3.0
            y_max = max(site_values) + 3.0
        elif len(all_z_g1_flat) > 0:
            y_min = np.nanmin(all_z_g1_flat) - 3.0
            y_max = np.nanmax(all_z_g1_flat) + 3.0
        else:
             y_min, y_max = -15, 15 # Fallback limits
        ax1.set_ylim(y_min, y_max)
        ax2.set_ylim(y_min, y_max) # Ensure shared ylim

        plt.suptitle(f'K+ Ion Positions & Density ({os.path.basename(run_dir)})', fontsize=16, y=0.99)
        plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout for suptitle
        combined_plot_path = os.path.join(run_dir, 'K_Ion_Combined_Plot.png')
        fig.savefig(combined_plot_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        log.debug(f"Saved combined ion plot to {combined_plot_path}")

    except Exception as e:
        log.error(f"Failed to generate combined ion plot: {e}", exc_info=True)
        if 'fig' in locals() and plt.fignum_exists(fig.number): plt.close(fig)