"""
Ion occupancy analysis module.

Functions for analyzing ion occupancy in binding sites
and generation of heatmaps and statistics.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Module logger
module_logger = logging.getLogger(__name__)

def create_ion_occupancy_heatmap(run_dir, time_points, ions_z_g1_centric, ion_indices, filter_sites, logger=None):
    """
    Create heatmap and bar chart showing K+ ion occupancy in binding sites.
    Properly handles NaN values for frames where ions are not in the filter.
    Saves plots and occupancy data CSV.

    Args:
        run_dir (str): Directory to save the plot and CSV.
        time_points (np.ndarray): Time points in ns.
        ions_z_g1_centric (dict): {ion_idx: np.array(g1_centric_z_pos)} with NaN when ion not in filter
        ion_indices (list): List of tracked ion indices.
        filter_sites (dict): Site positions relative to G1 C-alpha=0.
        logger (logging.Logger, optional): Logger instance. Defaults to module logger.

    Returns:
        str | None: Path to the saved heatmap PNG, or None on error.
    """
    log = logger if logger else module_logger
    if not filter_sites:
        log.warning("Filter sites data missing, cannot create occupancy heatmap.")
        return None
    if not ion_indices:
        log.info("No ion indices provided for occupancy heatmap.")
        # Create empty CSV for consistency? Or just return None? Let's return None.
        return None

    log.info("Creating K+ ion occupancy plots...")

    # Define binding site boundaries (midway between ordered sites)
    site_names_ordered = ['S0', 'S1', 'S2', 'S3', 'S4', 'Cavity'] # Extracellular to Intracellular
    # Filter sites present in the input dict and sort them by Z-position (descending)
    available_sites = {site: pos for site, pos in filter_sites.items() if site in site_names_ordered}
    if not available_sites:
         log.error("No standard sites (S0-S4, Cavity) found in filter_sites dict.")
         return None

    sorted_sites = sorted(available_sites.items(), key=lambda item: item[1], reverse=True)
    site_names_plot = [item[0] for item in sorted_sites]
    site_pos_plot = [item[1] for item in sorted_sites]

    # Calculate boundaries: midway points + outer bounds
    boundaries = []
    boundaries.append(site_pos_plot[0] + 1.5 if site_pos_plot else np.inf) # Top boundary
    for i in range(len(site_pos_plot) - 1):
        boundaries.append((site_pos_plot[i] + site_pos_plot[i+1]) / 2)
    boundaries.append(site_pos_plot[-1] - 1.5 if site_pos_plot else -np.inf) # Bottom boundary
    n_sites = len(site_names_plot)

    # Create occupancy matrix (frames x sites)
    n_frames = len(time_points)
    occupancy = np.zeros((n_frames, n_sites), dtype=int) # Use int for counts

    for frame_idx in range(n_frames):
        for ion_idx in ion_indices:
            if ion_idx in ions_z_g1_centric:
                z_pos = ions_z_g1_centric[ion_idx][frame_idx]
                if np.isfinite(z_pos):  # Only consider non-NaN positions (ion is in filter)
                    # Assign to site based on boundaries (upper_bound > z >= lower_bound)
                    for site_idx in range(n_sites):
                        upper_bound = boundaries[site_idx]
                        lower_bound = boundaries[site_idx + 1]
                        if lower_bound <= z_pos < upper_bound:
                            occupancy[frame_idx, site_idx] += 1
                            break # Ion assigned to one site

    # --- Create Heatmap ---
    try:
        fig, ax = plt.subplots(figsize=(12, 6)) # Adjusted size
        max_occ = np.max(occupancy) if occupancy.size > 0 else 1
        cmap = plt.cm.get_cmap("viridis", max_occ + 1) # Discrete colormap based on max occupancy

        im = ax.imshow(occupancy.T, aspect='auto', cmap=cmap,
                       interpolation='nearest', origin='upper', # Origin upper to match site order S0->Cavity
                       extent=[time_points[0], time_points[-1], n_sites, 0], # Adjust extent for origin='upper'
                       vmin=-0.5, vmax=max_occ + 0.5) # Center ticks for discrete colors

        # Add colorbar with integer ticks
        cbar = plt.colorbar(im, ax=ax, ticks=np.arange(max_occ + 1))
        cbar.set_label('Number of K+ Ions', fontsize=12)

        ax.set_yticks(np.arange(n_sites) + 0.5) # Center ticks between site boundaries
        ax.set_yticklabels(site_names_plot) # Use the ordered site names

        ax.set_xlabel('Time (ns)', fontsize=14)
        ax.set_ylabel('Binding Site', fontsize=14)
        ax.set_title(f'K+ Ion Occupancy Heatmap ({os.path.basename(run_dir)})', fontsize=16)
        plt.tight_layout()
        heatmap_path = os.path.join(run_dir, 'K_Ion_Occupancy_Heatmap.png')
        fig.savefig(heatmap_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        log.debug(f"Saved occupancy heatmap to {heatmap_path}")
    except Exception as e:
         log.error(f"Failed to generate occupancy heatmap: {e}", exc_info=True)
         heatmap_path = None # Indicate failure
         if 'fig' in locals() and plt.fignum_exists(fig.number): plt.close(fig)

    # --- Create Average Occupancy Bar Chart ---
    try:
        fig, ax = plt.subplots(figsize=(8, 5)) # Adjusted size
        if occupancy.size > 0:
            avg_occupancy = np.mean(occupancy, axis=0)
            sns.barplot(x=site_names_plot, y=avg_occupancy, ax=ax, palette='viridis', order=site_names_plot)

            # Add exact values above bars
            for i, v in enumerate(avg_occupancy):
                ax.text(i, v + 0.01 * np.max(avg_occupancy), f"{v:.2f}", ha='center', va='bottom', fontsize=9)
        else:
             ax.text(0.5, 0.5, 'No occupancy data', ha='center', va='center')

        ax.set_xlabel('Binding Site', fontsize=12)
        ax.set_ylabel('Average K+ Ion Occupancy', fontsize=12)
        ax.set_title(f'Average Site Occupancy ({os.path.basename(run_dir)})', fontsize=14)
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        avg_path = os.path.join(run_dir, 'K_Ion_Average_Occupancy.png')
        fig.savefig(avg_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        log.debug(f"Saved average occupancy bar chart to {avg_path}")
    except Exception as e:
        log.error(f"Failed to generate average occupancy plot: {e}", exc_info=True)
        if 'fig' in locals() and plt.fignum_exists(fig.number): plt.close(fig)

    # --- Save the raw occupancy data per frame ---
    try:
        df = pd.DataFrame(occupancy, columns=site_names_plot)
        df.insert(0, 'Time (ns)', time_points)
        csv_path = os.path.join(run_dir, 'K_Ion_Occupancy_Per_Frame.csv') # More descriptive name
        df.to_csv(csv_path, index=False, float_format='%.4f')
        log.info(f"Saved frame-by-frame ion occupancy to {csv_path}")
    except Exception as e:
        log.error(f"Failed to save occupancy per frame CSV: {e}")

    return heatmap_path

def analyze_ion_coordination(run_dir, time_points, ions_z_positions_abs, ion_indices, filter_sites, g1_reference):
    """
    Analyze K+ ion occupancy statistics for each binding site.
    Uses G1-centric coordinates for calculations. Properly handles NaN values
    for frames where ions are not in the filter. Saves stats to CSV.

    Args:
        run_dir (str): Directory to save the analysis results CSV.
        time_points (np.ndarray): Array of time points in ns.
        ions_z_positions_abs (dict): {ion_idx: np.array(abs_z_positions)} with NaN when ion not in filter
        ion_indices (list): List of tracked ion indices.
        filter_sites (dict): Site positions relative to G1 C-alpha=0.
        g1_reference (float): Absolute Z of G1 C-alpha plane.

    Returns:
        pd.DataFrame | None: DataFrame containing site statistics, or None on error.
    """
    # Use module logger
    coord_log = logging.getLogger(__name__)
    coord_log.info(f"Analyzing ion coordination stats for {os.path.basename(run_dir)}")

    if filter_sites is None:
        coord_log.error("Filter sites data missing, cannot analyze coordination.")
        return None
    if not ion_indices:
        coord_log.warning("No ion indices provided for coordination analysis.")
        # Create and save empty DataFrame for consistency?
        empty_df = pd.DataFrame(columns=['Site', 'Mean Occupancy', 'Max Occupancy', 'Occupancy > 0 (%)', 'Occupancy > 1 (%)'])
        stats_path = os.path.join(run_dir, 'K_Ion_Site_Statistics.csv')
        try: empty_df.to_csv(stats_path, index=False); coord_log.info(f"Saved empty ion stats file to {stats_path}")
        except: pass
        return empty_df

    # --- Convert to G1-Centric ---
    ions_z_g1 = {}
    num_frames = len(time_points)
    for ion_idx in ion_indices:
        if ion_idx in ions_z_positions_abs:
            abs_z = np.array(ions_z_positions_abs[ion_idx])
            if len(abs_z) == num_frames:
                ions_z_g1[ion_idx] = abs_z - g1_reference
            else:
                coord_log.warning(f"Length mismatch for ion {ion_idx} ({len(abs_z)} vs {num_frames} frames). Skipping.")
        else:
            coord_log.warning(f"Absolute position data missing for ion {ion_idx}. Skipping.")

    # --- Define Site Ranges ---
    # Re-calculate boundaries as in heatmap function
    site_names_ordered = ['S0', 'S1', 'S2', 'S3', 'S4', 'Cavity']
    available_sites = {site: pos for site, pos in filter_sites.items() if site in site_names_ordered}
    if not available_sites:
         coord_log.error("No standard sites (S0-S4, Cavity) found in filter_sites dict for coordination.")
         return None
    sorted_sites = sorted(available_sites.items(), key=lambda item: item[1], reverse=True)
    site_names_analysis = [item[0] for item in sorted_sites]
    site_pos_analysis = [item[1] for item in sorted_sites]

    boundaries = []
    boundaries.append(site_pos_analysis[0] + 1.5 if site_pos_analysis else np.inf) # Top boundary
    for i in range(len(site_pos_analysis) - 1):
        boundaries.append((site_pos_analysis[i] + site_pos_analysis[i+1]) / 2)
    boundaries.append(site_pos_analysis[-1] - 1.5 if site_pos_analysis else -np.inf) # Bottom boundary
    n_sites = len(site_names_analysis)

    # --- Calculate Occupancy Per Frame ---
    site_occupancy_counts = {site: np.zeros(num_frames, dtype=int) for site in site_names_analysis}

    for frame_idx in range(num_frames):
        for ion_idx in ion_indices:
            if ion_idx in ions_z_g1: # Check if ion exists and has valid data
                z_pos = ions_z_g1[ion_idx][frame_idx]
                if np.isfinite(z_pos):  # Only consider non-NaN positions (ion is in filter)
                    for site_idx in range(n_sites):
                        upper_bound = boundaries[site_idx]
                        lower_bound = boundaries[site_idx + 1]
                        if lower_bound <= z_pos < upper_bound:
                            site_name = site_names_analysis[site_idx]
                            site_occupancy_counts[site_name][frame_idx] += 1
                            break

    # --- Calculate Statistics ---
    stats_data = []
    for site in site_names_analysis: # Iterate in the defined order
        occupancy_vector = site_occupancy_counts[site]
        valid_occupancy = occupancy_vector[np.isfinite(occupancy_vector)] # Should be all ints here
        n_valid_frames = len(valid_occupancy)

        if n_valid_frames > 0:
            mean_occ = np.mean(valid_occupancy)
            max_occ = np.max(valid_occupancy)
            frames_gt0 = np.sum(valid_occupancy > 0)
            frames_gt1 = np.sum(valid_occupancy > 1)
            pct_occ_gt0 = (frames_gt0 / n_valid_frames) * 100.0
            pct_occ_gt1 = (frames_gt1 / n_valid_frames) * 100.0
        else: # Handle case where there are no valid frames (shouldn't happen if time_points exist)
            mean_occ, max_occ, pct_occ_gt0, pct_occ_gt1 = np.nan, np.nan, np.nan, np.nan

        stats_data.append({
            'Site': site,
            'Mean Occupancy': mean_occ,
            'Max Occupancy': int(max_occ) if np.isfinite(max_occ) else np.nan, # Max occupancy is integer count
            'Occupancy > 0 (%)': pct_occ_gt0,
            'Occupancy > 1 (%)': pct_occ_gt1
        })

    # Save statistics to CSV
    stats_df = pd.DataFrame(stats_data)
    stats_df = stats_df[['Site', 'Mean Occupancy', 'Max Occupancy', 'Occupancy > 0 (%)', 'Occupancy > 1 (%)']] # Ensure order
    stats_path = os.path.join(run_dir, 'K_Ion_Site_Statistics.csv')
    try:
        stats_df.to_csv(stats_path, index=False, float_format='%.4f', na_rep='NaN')
        coord_log.info(f"Saved K+ Ion site statistics to {stats_path}")
    except Exception as csv_err:
        coord_log.error(f"Failed to save K+ Ion site statistics CSV: {csv_err}")
        return None # Indicate failure

    return stats_df