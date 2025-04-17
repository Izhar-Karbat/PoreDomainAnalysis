"""
Binding site visualization module.

Functions for visualizing K+ binding sites and their relative positions.
"""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Module logger
module_logger = logging.getLogger(__name__)

def visualize_binding_sites_g1_centric(sites_g1_centric, g1_ca_z_ref, output_dir, logger=None):
    """
    Create a schematic visualization of binding site positions relative to the
    G1 C-alpha reference plane (Z=0). Also saves site positions to a text file.

    Args:
        sites_g1_centric (dict): Dictionary mapping site names to Z-coordinates
                                 (relative to G1 C-alpha=0).
        g1_ca_z_ref (float): Absolute Z-coordinate of the G1 C-alpha reference plane.
        output_dir (str): Directory to save the visualization and data file.
        logger (logging.Logger, optional): Logger instance. Defaults to module logger.

    Returns:
        str | None: Path to the saved visualization PNG file, or None on error.
    """
    log = logger if logger else module_logger
    if not sites_g1_centric or g1_ca_z_ref is None:
        log.error("Missing site data or G1 reference for visualization.")
        return None

    log.info("Creating G1-centric binding site visualization...")

    fig, ax = plt.subplots(figsize=(6, 8)) # Adjusted size

    site_colors = { # Standard colors
        'S0': '#FF6347', 'S1': '#4169E1', 'S2': '#2E8B57',
        'S3': '#BA55D3', 'S4': '#CD853F', 'Cavity': '#708090'
    }
    channel_width = 0.4
    channel_left = 0.1

    # Determine plot Y limits based on site range
    site_values = list(sites_g1_centric.values())
    y_min = min(site_values) - 2.0
    y_max = max(site_values) + 2.0

    # Draw channel walls (schematic)
    ax.plot([channel_left, channel_left], [y_min, y_max], 'k-', linewidth=1.5, alpha=0.7)
    ax.plot([channel_left + channel_width, channel_left + channel_width], [y_min, y_max], 'k-', linewidth=1.5, alpha=0.7)

    # Add shaded rectangle for selectivity filter region (S0 to S4)
    try:
        filter_top = sites_g1_centric['S0']
        filter_bottom = sites_g1_centric['S4']
        rect = plt.Rectangle((channel_left, filter_bottom), channel_width, filter_top - filter_bottom,
                             color='lightgrey', alpha=0.3, zorder=0)
        ax.add_patch(rect)
        ax.text(channel_left + channel_width / 2, (filter_top + filter_bottom) / 2,
                "Filter", ha='center', va='center', fontsize=9, alpha=0.8, zorder=1)
    except KeyError:
        log.warning("Could not define filter rectangle boundaries (missing S0 or S4).")


    # Add reference line at Z=0 (G1 C-alpha position)
    ax.axhline(y=0, color='black', linestyle=':', alpha=0.9, linewidth=1.5, zorder=2)
    ax.text(channel_left + channel_width + 0.05, 0, "G1 Cα (Z=0)",
            va='center', ha='left', fontsize=10, fontweight='bold', color='black', zorder=3)

    # Add binding sites (horizontal lines and labels)
    for site, z_pos_rel in sites_g1_centric.items():
        color = site_colors.get(site, 'grey')
        ax.axhline(y=z_pos_rel, color=color, linestyle='--', alpha=0.8, linewidth=1.5, zorder=2)
        ax.text(channel_left + channel_width + 0.05, z_pos_rel, f"{site}: {z_pos_rel:.2f} Å",
                va='center', ha='left', fontsize=10, fontweight='bold', color=color, zorder=3)

    # Add Extracellular/Intracellular labels
    ax.text(channel_left + channel_width / 2, y_max + 0.5, "Extracellular",
            ha='center', va='bottom', fontsize=12, color='blue', zorder=1)
    ax.text(channel_left + channel_width / 2, y_min - 0.5, "Intracellular",
            ha='center', va='top', fontsize=12, color='red', zorder=1)

    # Set axis properties
    ax.set_title('K+ Binding Sites (G1 Cα = 0)', fontsize=14)
    ax.set_xlim(0, channel_left + channel_width + 0.6) # Adjust xlim to fit labels
    ax.set_ylim(y_min, y_max)
    ax.set_ylabel('Z-Position relative to G1 Cα (Å)', fontsize=12)
    ax.get_xaxis().set_visible(False) # Hide x-axis as it's schematic
    ax.grid(axis='y', linestyle=':', alpha=0.3, zorder=0)

    # Save the figure
    figure_path = os.path.join(output_dir, 'binding_sites_g1_centric_visualization.png')
    try:
        fig.savefig(figure_path, dpi=200, bbox_inches='tight')
        log.info(f"Binding site visualization saved to {figure_path}")
    except Exception as e:
        log.error(f"Failed to save binding site visualization: {e}")
        figure_path = None # Indicate failure
    finally:
        plt.close(fig)

    # Also save numerical data to a file
    data_path = os.path.join(output_dir, 'binding_site_positions_g1_centric.txt')
    try:
        with open(data_path, 'w') as f:
            f.write(f"# K+ channel binding site positions\n")
            f.write(f"# Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Absolute Z-coordinate of G1 C-alpha reference plane: {g1_ca_z_ref:.4f} Å\n\n")
            f.write("# Site positions relative to G1 C-alpha plane (Z=0):\n")
            for site, pos in sites_g1_centric.items():
                f.write(f"{site}: {pos:.4f}\n")
        log.info(f"Binding site position data saved to {data_path}")
    except Exception as e:
         log.error(f"Failed to save binding site position data: {e}")

    return figure_path