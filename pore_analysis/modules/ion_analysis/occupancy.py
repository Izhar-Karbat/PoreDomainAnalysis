# filename: pore_analysis/modules/ion_analysis/occupancy.py
"""
Ion Analysis: Site occupancy calculation and statistics.
"""

import os
import logging
import numpy as np
import pandas as pd
import sqlite3
from typing import Dict, Optional, Tuple, List

# Import from core modules
try:
    from pore_analysis.core.database import register_product, store_metric
except ImportError as e:
    print(f"Error importing dependency modules in ion_analysis/occupancy.py: {e}")
    raise

logger = logging.getLogger(__name__)


def _save_occupancy_data(
    run_dir: str,
    time_points: np.ndarray,
    occupancy_counts: Dict[str, np.ndarray],
    site_names_ordered: List[str],
    db_conn: sqlite3.Connection,
    module_name: str
) -> Optional[str]:
    """
    Internal helper to save frame-by-frame occupancy counts to CSV
    and register in the database.
    """
    output_dir = os.path.join(run_dir, "ion_analysis")
    os.makedirs(output_dir, exist_ok=True)
    rel_path = None

    try:
        df_data = {'Time (ns)': time_points}
        for site in site_names_ordered:
            if site in occupancy_counts:
                df_data[site] = occupancy_counts[site]
            else: # Should not happen if calculated correctly
                df_data[site] = np.zeros(len(time_points), dtype=int)

        df = pd.DataFrame(df_data)
        csv_path = os.path.join(output_dir, 'ion_occupancy_per_frame.csv')
        df.to_csv(csv_path, index=False, float_format='%d') # Save counts as integers
        logger.info(f"Saved frame-by-frame ion occupancy to {csv_path}")
        rel_path = os.path.relpath(csv_path, run_dir)
        register_product(db_conn, module_name, "csv", "data", rel_path,
                         subcategory="ion_occupancy_per_frame",
                         description="Time series of K+ ion counts per binding site.")
    except Exception as e:
        logger.error(f"Failed to save/register occupancy per frame CSV: {e}")
        rel_path = None

    return rel_path


def _save_occupancy_stats(
    run_dir: str,
    stats_data: List[Dict[str, any]],
    db_conn: sqlite3.Connection,
    module_name: str
) -> Optional[str]:
    """
    Internal helper to save site occupancy statistics to CSV,
    store metrics in DB, and register the CSV.
    """
    output_dir = os.path.join(run_dir, "ion_analysis")
    os.makedirs(output_dir, exist_ok=True)
    rel_path = None

    if not stats_data:
        logger.warning("No occupancy statistics data to save.")
        return None

    stats_df = pd.DataFrame(stats_data)
    # Ensure standard column order
    cols = ['Site', 'Mean Occupancy', 'Max Occupancy', 'Occupancy > 0 (%)', 'Occupancy > 1 (%)']
    stats_df = stats_df[[c for c in cols if c in stats_df.columns]]

    stats_path = os.path.join(output_dir, 'ion_site_statistics.csv')

    try:
        stats_df.to_csv(stats_path, index=False, float_format='%.4f', na_rep='NaN')
        logger.info(f"Saved K+ Ion site statistics to {stats_path}")
        rel_path = os.path.relpath(stats_path, run_dir)
        register_product(db_conn, module_name, "csv", "data", rel_path,
                         subcategory="ion_site_statistics",
                         description="Summary statistics (Mean/Max/Pct Occ) per binding site.")

        # Store individual metrics in DB
        for _, row in stats_df.iterrows():
            site = row['Site']
            store_metric(db_conn, module_name, f'Ion_AvgOcc_{site}', row['Mean Occupancy'], 'count', f'Mean occupancy of site {site}')
            store_metric(db_conn, module_name, f'Ion_MaxOcc_{site}', row['Max Occupancy'], 'count', f'Max occupancy of site {site}')
            store_metric(db_conn, module_name, f'Ion_PctTimeOcc_{site}', row['Occupancy > 0 (%)'], '%', f'% Time site {site} has >0 ions')

    except Exception as e:
        logger.error(f"Failed to save/register/store site statistics: {e}")
        rel_path = None # Failed

    return rel_path


def calculate_occupancy(
    run_dir: str,
    time_points: np.ndarray,
    ions_z_positions_abs: Dict[int, np.ndarray],
    ion_indices: List[int],
    filter_sites: Dict[str, float],
    g1_reference: float,
    db_conn: sqlite3.Connection,
    module_name: str = "ion_analysis"
) -> Tuple[Optional[str], Optional[str]]:
    """
    Calculates frame-by-frame ion occupancy for each binding site and
    aggregate statistics (Mean, Max, % Time occupied > 0).
    Saves results to CSV files and stores metrics in the database.

    Args:
        run_dir: Path to the run directory.
        time_points: Array of time points in ns.
        ions_z_positions_abs: Dict {ion_idx: np.array(abs_z_positions)} with NaN when not in filter.
        ion_indices: List of tracked ion indices.
        filter_sites: Dict {site_name: relative_z_coord}.
        g1_reference: Absolute Z of G1 C-alpha plane.
        db_conn: Active database connection.
        module_name: Name of the calling module for registration.

    Returns:
        Tuple (occupancy_csv_path, stats_csv_path): Relative paths to the saved
                                                    CSV files, or None if saving failed.
    """
    logger.info("Calculating ion site occupancy...")

    if filter_sites is None or g1_reference is None:
        logger.error("Filter sites data or G1 reference missing.")
        return None, None
    if not ion_indices or not ions_z_positions_abs:
        logger.warning("No ion indices or position data provided.")
        return None, None # Cannot calculate without ions

    num_frames = len(time_points)

    # --- Convert to G1-Centric ---
    ions_z_g1 = {}
    for ion_idx in ion_indices:
        if ion_idx in ions_z_positions_abs:
            abs_z = np.array(ions_z_positions_abs[ion_idx])
            if len(abs_z) == num_frames:
                ions_z_g1[ion_idx] = abs_z - g1_reference
            else:
                logger.warning(f"Length mismatch for ion {ion_idx} ({len(abs_z)} vs {num_frames}). Skipping for occupancy.")
        else:
            logger.warning(f"Absolute position data missing for ion {ion_idx}. Skipping for occupancy.")

    if not ions_z_g1: # Check if any ions remain after validation
         logger.warning("No valid ion position data available for occupancy calculation.")
         return None, None

    # --- Define Site Ranges ---
    site_names_ordered = ['S0', 'S1', 'S2', 'S3', 'S4', 'Cavity']
    available_sites = {site: pos for site, pos in filter_sites.items() if site in site_names_ordered}
    if not available_sites:
         logger.error("No standard sites (S0-S4, Cavity) found in filter_sites dict.")
         return None, None

    sorted_sites = sorted(available_sites.items(), key=lambda item: item[1], reverse=True)
    site_names_analysis = [item[0] for item in sorted_sites]
    site_pos_analysis = [item[1] for item in sorted_sites]

    boundaries = []
    boundaries.append(site_pos_analysis[0] + 1.5 if site_pos_analysis else np.inf)
    for i in range(len(site_pos_analysis) - 1):
        boundaries.append((site_pos_analysis[i] + site_pos_analysis[i+1]) / 2)
    boundaries.append(site_pos_analysis[-1] - 1.5 if site_pos_analysis else -np.inf)
    n_sites = len(site_names_analysis)

    # --- Calculate Occupancy Per Frame ---
    site_occupancy_counts = {site: np.zeros(num_frames, dtype=int) for site in site_names_analysis}

    for frame_idx in range(num_frames):
        # Iterate only over ions with valid G1-centric data for this frame
        for ion_idx in ions_z_g1.keys():
            z_pos = ions_z_g1[ion_idx][frame_idx]
            if np.isfinite(z_pos):
                for site_idx in range(n_sites):
                    upper_bound = boundaries[site_idx]
                    lower_bound = boundaries[site_idx + 1]
                    if lower_bound <= z_pos < upper_bound:
                        site_name = site_names_analysis[site_idx]
                        site_occupancy_counts[site_name][frame_idx] += 1
                        break

    # --- Calculate Statistics ---
    stats_data = []
    for site in site_names_analysis:
        occupancy_vector = site_occupancy_counts[site]
        n_valid_frames = len(occupancy_vector) # Should be n_frames

        if n_valid_frames > 0:
            mean_occ = np.mean(occupancy_vector)
            max_occ = np.max(occupancy_vector)
            frames_gt0 = np.sum(occupancy_vector > 0)
            frames_gt1 = np.sum(occupancy_vector > 1) # Optional, for future use maybe
            pct_occ_gt0 = (frames_gt0 / n_valid_frames) * 100.0 if n_valid_frames > 0 else 0.0
            pct_occ_gt1 = (frames_gt1 / n_valid_frames) * 100.0 if n_valid_frames > 0 else 0.0
        else:
            mean_occ, max_occ, pct_occ_gt0, pct_occ_gt1 = np.nan, np.nan, np.nan, np.nan

        stats_data.append({
            'Site': site,
            'Mean Occupancy': mean_occ,
            'Max Occupancy': int(max_occ) if np.isfinite(max_occ) else np.nan,
            'Occupancy > 0 (%)': pct_occ_gt0,
            'Occupancy > 1 (%)': pct_occ_gt1
        })

    # --- Save Data and Stats using Helpers ---
    occ_csv_path = _save_occupancy_data(run_dir, time_points, site_occupancy_counts, site_names_analysis, db_conn, module_name)
    stats_csv_path = _save_occupancy_stats(run_dir, stats_data, db_conn, module_name)

    return occ_csv_path, stats_csv_path
