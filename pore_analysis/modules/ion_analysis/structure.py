"""
Ion Analysis: Filter structure identification and DATA-OPTIMIZED binding site calculation.
Includes separate function for G1 reference calculation.
"""

import os
import logging
import numpy as np
import pandas as pd
import MDAnalysis as mda
import sqlite3
from typing import Dict, Optional, Tuple, List
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# Import from core modules
try:
    from pore_analysis.core.utils import OneLetter
    from pore_analysis.core.database import register_product, get_product_path
    from pore_analysis.core.plotting_style import STYLE, setup_style
    from pore_analysis.core.config import (
        SITE_OPT_SEARCH_RADIUS, SITE_OPT_MIN_PEAK_HEIGHT,
        SITE_OPT_HIST_BINS, SITE_OPT_SMOOTH_WINDOW
    )
except ImportError as e:
    print(f"Error importing dependency modules in ion_analysis/structure.py: {e}")
    raise

logger = logging.getLogger(__name__)
setup_style() # Apply plotting style

# --- Filter Residue Identification (Unchanged) ---
def find_filter_residues(universe: mda.Universe) -> Optional[Dict[str, List[int]]]:
    """
    Find the selectivity filter residues (assumed TVGYG-like).
    Identifies the 5 residues ending in GYG for each channel chain.

    Args:
        universe: The universe containing the system.

    Returns:
        Dictionary mapping chain segids (e.g., 'PROA') to lists of
        filter residue IDs (resids), or None if chains not found or on error.
    """
    # --- Implementation unchanged ---
    logger.info("Searching for selectivity filter residues...")

    chain_segids = []
    potential_segids = ['PROA', 'PROB', 'PROC', 'PROD', 'A', 'B', 'C', 'D']
    found_segids = set(np.unique(universe.atoms.segids))

    for segid in potential_segids:
        if segid in found_segids:
            if len(universe.select_atoms(f'segid {segid}')) > 0:
                 chain_segids.append(segid)
            else:
                 logger.debug(f"Segid {segid} found in universe but selection yielded 0 atoms.")

    grouped_segids = {}
    for segid in chain_segids:
        chain_letter = segid[-1]
        if chain_letter not in grouped_segids:
             grouped_segids[chain_letter] = segid
        elif segid.startswith("PRO") and not grouped_segids[chain_letter].startswith("PRO"):
             grouped_segids[chain_letter] = segid

    final_chain_segids = list(grouped_segids.values())

    if not final_chain_segids:
        logger.error("Could not find any standard channel chain segids (A/B/C/D or PROA/PROB/PROC/PROD).")
        return None
    logger.info(f"Identified channel chains using segids: {final_chain_segids}")

    filter_residues = {}
    possible_filter_len = 5

    for segid in final_chain_segids:
        try:
            chain = universe.select_atoms(f'segid {segid}')
            if not chain: continue

            resnames = [res.resname for res in chain.residues]
            if not resnames:
                logger.warning(f"Chain {segid} contains no residues.")
                continue

            chain_seq = OneLetter("".join(resnames))
            logger.debug(f"Chain {segid} sequence: {chain_seq[:30]}...")

            idx_gyg_end = chain_seq.rfind('GYG')

            if idx_gyg_end != -1:
                filter_start_seq_idx = idx_gyg_end - 2
                filter_end_seq_idx = idx_gyg_end + 3

                if filter_start_seq_idx >= 0 and filter_end_seq_idx <= len(chain.residues):
                     filter_res_group = chain.residues[filter_start_seq_idx:filter_end_seq_idx]
                     filter_resids = [res.resid for res in filter_res_group]
                     filter_seq = OneLetter("".join(res.resname for res in filter_res_group))

                     if filter_seq.endswith("GYG") and len(filter_resids) == possible_filter_len:
                         filter_residues[segid] = filter_resids
                         logger.info(f"Found filter for {segid}: Resids {filter_resids} (Sequence: {filter_seq})")
                     else:
                          logger.warning(f"Potential filter match for {segid} failed validation (Seq: {filter_seq}, Len: {len(filter_resids)}).")
                else:
                     logger.warning(f"GYG motif found too close to the start/end of chain {segid} sequence.")
            else:
                 logger.error(f"Mandatory 'GYG' motif not found in chain {segid}.")
        except Exception as e:
            logger.error(f"Error processing chain {segid} for filter residues: {e}", exc_info=True)

    if not filter_residues:
         logger.error("Failed to identify filter residues for any chain.")
         return None
    if len(filter_residues) < 4: # Allow processing even if not all 4 found, but warn
         logger.warning(f"Identified filter residues for only {len(filter_residues)} out of 4 expected chains.")

    return filter_residues


# --- G1 Reference Calculation (NEW FUNCTION) ---
def calculate_g1_reference(
    universe: mda.Universe,
    filter_residues: Dict[str, List[int]],
    start_frame: int = 0
) -> Optional[float]:
    """
    Calculates the absolute Z-coordinate of the G1 C-alpha reference plane.

    Args:
        universe: Universe object.
        filter_residues: Dictionary mapping chain segids to filter resids.
        start_frame: Which frame to use for reference (defaults to 0).

    Returns:
        Absolute Z-coordinate of the G1 C-alpha plane, or None on error.
    """
    logger.info("Calculating G1 C-alpha reference Z-position...")
    if not filter_residues:
        logger.error("Filter residues dictionary is empty for G1 ref calculation.")
        return None

    try:
        # Use the specified start_frame as the reference frame
        if start_frame < 0 or start_frame >= len(universe.trajectory):
            logger.warning(f"Invalid start_frame {start_frame} for G1 reference. Using frame 0 instead.")
            universe.trajectory[0]
        else:
            universe.trajectory[start_frame]
            logger.info(f"Using frame {start_frame} for G1 reference calculation.")
        residue_by_position = defaultdict(list)
        position_map = {0: 'T', 1: 'V', 2: 'G1', 3: 'Y', 4: 'G2'} # Assuming TVGYG order
        valid_chains = {segid: resids for segid, resids in filter_residues.items() if len(resids) == 5}
        if not valid_chains: raise ValueError("No chains with 5 filter residues found for G1 ref.")

        for segid, residues in valid_chains.items():
            for i, resid in enumerate(residues):
                pos_key = position_map.get(i)
                if pos_key == 'G1':
                    residue_by_position['G1'].append((segid, resid))

        g1_ca_selection_parts = [f"(segid {segid} and resid {resid} and name CA)" for segid, resid in residue_by_position['G1']]
        if not g1_ca_selection_parts: raise ValueError("Could not build G1 CA selection.")

        g1_ca_atoms = universe.select_atoms(" or ".join(g1_ca_selection_parts))
        if len(g1_ca_atoms) == 0: raise ValueError("G1 CA selection returned 0 atoms.")

        g1_ca_z_ref = np.mean(g1_ca_atoms.positions[:, 2])
        logger.info(f"Reference plane: Average G1 C-alpha absolute Z = {g1_ca_z_ref:.3f} Å")
        return g1_ca_z_ref

    except Exception as e:
        logger.error(f"Failed to calculate G1 C-alpha reference Z: {e}", exc_info=True)
        return None


# --- Binding Site Optimization Function (Unchanged from previous proposed version) ---
def detect_binding_sites_guided(
    all_abs_positions: np.ndarray,
    predefined_sites_rel: Dict[str, float],
    g1_ca_z_ref: float,
    run_dir: str, # Added for saving plot
    db_conn: sqlite3.Connection, # Added for registration
    module_name: str # Added for registration
    ) -> Tuple[Optional[Dict[str, float]], Optional[str]]:
    """
    Detect binding sites using histogram analysis guided by predefined sites.
    Calculates OPTIMIZED ABSOLUTE Z positions.

    Args:
        all_abs_positions: Array of ALL raw K+ ion absolute Z-positions when inside the filter.
        predefined_sites_rel: Dictionary of default relative site positions (G1 Cα = 0).
        g1_ca_z_ref: Absolute Z position of the G1 C-alpha reference plane.
        run_dir: Path to the run directory for saving output.
        db_conn: Active database connection.
        module_name: Name of the calling module for registration.

    Returns:
        Tuple (optimized_sites_abs, plot_rel_path):
            - optimized_sites_abs: Dictionary mapping site names to OPTIMIZED ABSOLUTE Z positions, or None on error.
            - plot_rel_path: Relative path to the saved optimization plot, or None.
    """
    # --- Implementation unchanged ---
    if len(all_abs_positions) < 100: # Need sufficient data for meaningful histogram
        logger.warning(f"Insufficient ion position data ({len(all_abs_positions)} points) for site optimization. Using default sites.")
        # Return default absolute positions
        default_sites_abs = {site: pos + g1_ca_z_ref for site, pos in predefined_sites_rel.items()}
        return default_sites_abs, None # No plot generated

    # Convert predefined relative sites to absolute for comparison
    predefined_sites_abs = {site: pos + g1_ca_z_ref for site, pos in predefined_sites_rel.items()}
    site_order = sorted(predefined_sites_abs.keys(), key=lambda s: predefined_sites_abs[s], reverse=True)

    # Use config parameters
    n_bins = SITE_OPT_HIST_BINS
    smoothing_window = SITE_OPT_SMOOTH_WINDOW
    search_radius = SITE_OPT_SEARCH_RADIUS
    min_peak_height_rel = SITE_OPT_MIN_PEAK_HEIGHT # Relative threshold

    logger.info(f"Optimizing binding sites using {len(all_abs_positions)} data points...")
    logger.debug(f"Optimization params: bins={n_bins}, smooth={smoothing_window}, radius={search_radius}, min_peak_h={min_peak_height_rel}")

    try:
        # Create histogram of ABSOLUTE positions
        hist, bin_edges = np.histogram(all_abs_positions, bins=n_bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Apply smoothing
        smoothed_hist = np.zeros_like(hist, dtype=float)
        half_window = smoothing_window // 2
        for i in range(len(hist)):
            start = max(0, i - half_window)
            end = min(len(hist), i + half_window + 1)
            smoothed_hist[i] = np.mean(hist[start:end])

        # Normalize and find peaks
        max_smoothed_hist = np.max(smoothed_hist)
        if max_smoothed_hist <= 0:
             logger.warning("Smoothed histogram is non-positive. Cannot optimize sites. Using default.")
             return predefined_sites_abs, None

        norm_smoothed_hist = smoothed_hist / max_smoothed_hist
        min_peak_height_abs = min_peak_height_rel # Apply relative threshold directly to normalized hist

        peak_indices = []
        for i in range(1, len(norm_smoothed_hist) - 1):
            if (norm_smoothed_hist[i] > norm_smoothed_hist[i-1] and
                norm_smoothed_hist[i] > norm_smoothed_hist[i+1] and
                norm_smoothed_hist[i] >= min_peak_height_abs):
                peak_indices.append(i)

        peak_positions = bin_centers[peak_indices]
        peak_heights_norm = norm_smoothed_hist[peak_indices]

        logger.info(f"Found {len(peak_positions)} initial peaks in ion distribution.")
        if not peak_positions.any():
            logger.warning("No peaks found meeting criteria. Using default site positions.")
            return predefined_sites_abs, None

        # Match peaks to predefined ABSOLUTE site positions
        optimized_sites_abs = {}
        # Keep track of which peaks are used
        available_peak_indices = list(range(len(peak_positions)))

        for site in site_order:
            predefined_pos = predefined_sites_abs[site]
            best_match_idx = -1
            min_dist = search_radius + 1e-6 # Initialize distance slightly larger than radius

            current_peak_list_indices = list(available_peak_indices) # Iterate over indices currently available
            for peak_list_idx in current_peak_list_indices:
                 # actual_peak_idx = peak_indices[peak_list_idx] # Not needed directly
                 peak_pos = peak_positions[peak_list_idx]
                 dist = abs(peak_pos - predefined_pos)

                 if dist <= search_radius and dist < min_dist:
                      min_dist = dist
                      best_match_idx = peak_list_idx # Store the index within the *current* available_peak_indices list

            if best_match_idx != -1:
                 # Found a peak within radius
                 optimized_sites_abs[site] = peak_positions[best_match_idx]
                 peak_height = peak_heights_norm[best_match_idx]
                 logger.info(f"  Site {site}: Adjusted from {predefined_pos:.2f} Å to {optimized_sites_abs[site]:.2f} Å "
                             f"(peak height: {peak_height:.2f}, dist: {min_dist:.2f} Å)")
                 # Remove the matched peak index from further consideration
                 available_peak_indices.remove(best_match_idx)
            else:
                 # No peak found within radius, use default absolute position
                 optimized_sites_abs[site] = predefined_pos
                 logger.info(f"  Site {site}: Kept predefined position ({predefined_pos:.2f} Å) "
                             f"(nearest peak too far or none available).")

        # --- Generate Optimization Plot ---
        output_dir = os.path.join(run_dir, "ion_analysis")
        plot_filename = "binding_site_optimization.png"
        plot_filepath = os.path.join(output_dir, plot_filename)
        rel_plot_path = os.path.relpath(plot_filepath, run_dir)

        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            # Plot histogram and smoothed version
            ax.bar(bin_centers, hist, width=(bin_centers[1]-bin_centers[0]), alpha=0.4, color='gray', label='Raw Counts')
            ax_smooth = ax.twinx() # Use twinx for smoothed density scale
            ax_smooth.plot(bin_centers, smoothed_hist, 'k-', lw=1.5, label='Smoothed Density') # Plot original smoothed counts
            ax_smooth.set_ylabel('Smoothed Counts / Density', fontsize=STYLE['font_sizes']['axis_label'])
            ax_smooth.tick_params(axis='y', labelsize=STYLE['font_sizes']['tick_label'])

            # Mark optimized binding sites (absolute Z)
            for site, pos in optimized_sites_abs.items():
                ax.axvline(pos, color=STYLE['bright_colors']['B'], linestyle='--', alpha=0.8, lw=1.5) # Use bright color B (e.g., red)
                ax.text(pos, plt.ylim()[1] * 0.95, site, ha='center', va='bottom', color=STYLE['bright_colors']['B'], fontweight='bold', fontsize=STYLE['font_sizes']['annotation'])

            # Mark default binding sites for comparison (absolute Z)
            for site, pos in predefined_sites_abs.items():
                ax.axvline(pos, color=STYLE['bright_colors']['A'], linestyle=':', alpha=0.6, lw=1.0) # Use bright color A (e.g., blue)
                # Position default label slightly lower
                ax.text(pos, plt.ylim()[1] * 0.85, f"{site} (def)", ha='center', va='top', color=STYLE['bright_colors']['A'], fontsize=STYLE['font_sizes']['annotation'] * 0.8)

            ax.set_xlabel('Absolute Z-position (Å)', fontsize=STYLE['font_sizes']['axis_label'])
            ax.set_ylabel('Raw Counts', fontsize=STYLE['font_sizes']['axis_label'])
            ax.tick_params(axis='both', labelsize=STYLE['font_sizes']['tick_label'])
            # Add legend combining both axes
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax_smooth.get_legend_handles_labels()
            ax_smooth.legend(lines + lines2, labels + labels2, loc='upper right', fontsize=STYLE['font_sizes']['annotation'])

            ax.grid(True, linestyle=STYLE['grid']['linestyle'], alpha=STYLE['grid']['alpha'])
            plt.tight_layout()
            fig.savefig(plot_filepath, dpi=150)
            plt.close(fig)
            logger.info(f"Saved binding site optimization plot to {plot_filepath}")
            # Register plot product
            register_product(db_conn, module_name, "png", "plot", rel_plot_path,
                             subcategory="site_optimization_plot", # Use specific subcategory
                             description="Ion position histogram and optimized vs default binding sites.")
        except Exception as e:
            logger.error(f"Failed to generate or save binding site optimization plot: {e}", exc_info=True)
            rel_plot_path = None # Indicate plot failure
            if 'fig' in locals() and plt.fignum_exists(fig.number): plt.close(fig)

        return optimized_sites_abs, rel_plot_path

    except Exception as e:
        logger.error(f"Error during binding site optimization: {e}", exc_info=True)
        return predefined_sites_abs, None # Return default absolute positions on error


# --- Main Site Calculation Function (MODIFIED) ---
def calculate_binding_sites(
    g1_ca_z_ref: float, # Changed: Receive G1 ref as input
    run_dir: str,
    db_conn: sqlite3.Connection,
    module_name: str = "ion_analysis"
) -> Optional[Dict[str, float]]:
    """
    Optimizes binding site Z-positions relative to the provided G1 C-alpha plane (Z=0).
    Loads raw ion positions, runs optimization, saves optimized site definitions
    to a file, registers it, and generates/registers the optimization plot.

    Args:
        g1_ca_z_ref: Absolute Z-coordinate of the G1 C-alpha plane.
        run_dir: Path to the run directory for saving output.
        db_conn: Active database connection.
        module_name: Name of the calling module for registration.

    Returns:
        optimized_sites_rel: Dict mapping site names (S0-S4, Cavity) to OPTIMIZED relative Z-coords, or None on error.
    """
    logger.info("Optimizing binding site Z-positions using ion distribution data...")
    output_dir = os.path.join(run_dir, "ion_analysis")
    os.makedirs(output_dir, exist_ok=True)

    # --- 1. Load Raw Absolute Ion Positions ---
    abs_pos_rel_path = get_product_path(db_conn, 'csv', 'data', 'ion_positions_absolute', module_name) # Use module_name passed in
    if not abs_pos_rel_path:
        logger.error("Absolute ion position CSV path not found in database. Cannot perform site optimization.")
        return None

    abs_pos_filepath = os.path.join(run_dir, abs_pos_rel_path)
    if not os.path.exists(abs_pos_filepath):
        logger.error(f"Absolute ion position CSV file not found at: {abs_pos_filepath}")
        return None

    all_abs_positions = []
    try:
        df_abs = pd.read_csv(abs_pos_filepath)
        ion_cols = [col for col in df_abs.columns if col.startswith('Ion_') and col.endswith('_Z_Abs')]
        if not ion_cols: raise ValueError("No ion absolute Z columns found in CSV.")

        for col in ion_cols:
            valid_pos = df_abs[col].dropna().values
            all_abs_positions.extend(valid_pos)
        all_abs_positions = np.array(all_abs_positions)
        logger.info(f"Loaded {len(all_abs_positions)} valid absolute Z positions for optimization.")

    except Exception as e:
        logger.error(f"Failed to load or process absolute ion positions from {abs_pos_filepath}: {e}", exc_info=True)
        return None

    # --- 2. Define Default Relative Sites (needed for optimization guidance) ---
    default_binding_sites_rel = {
        'S0':      6.0, 'S1':      3.3, 'S2':      0.3,
        'S3':     -2.5, 'S4':     -5.5, 'Cavity': -8.5,
    }

    # --- 3. Run Site Optimization ---
    optimized_sites_abs, _ = detect_binding_sites_guided(
        all_abs_positions,
        default_binding_sites_rel,
        g1_ca_z_ref,
        run_dir,
        db_conn,
        module_name
    )

    if optimized_sites_abs is None:
        logger.error("Binding site optimization failed. Cannot proceed.")
        return None # Optimization failed

    # --- 4. Calculate OPTIMIZED RELATIVE Sites ---
    optimized_sites_rel = {site: pos - g1_ca_z_ref for site, pos in optimized_sites_abs.items()}

    logger.info("Final OPTIMIZED binding site Z-positions (relative to G1 Cα = 0):")
    for site, pos in sorted(optimized_sites_rel.items(), key=lambda item: item[1], reverse=True):
        logger.info(f"  {site}: {pos:.3f} Å")

    # --- 5. Save Optimized Relative Site Definitions ---
    data_path = os.path.join(output_dir, 'binding_site_positions_g1_centric.txt') # Same filename
    rel_path = os.path.relpath(data_path, run_dir)
    description = f"OPTIMIZED K+ binding site positions relative to G1 C-alpha Z=0 (Reference Z={g1_ca_z_ref:.4f} Å)."
    try:
        with open(data_path, 'w') as f:
            f.write(f"# OPTIMIZED K+ channel binding site positions\n")
            f.write(f"# Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Absolute Z-coordinate of G1 C-alpha reference plane: {g1_ca_z_ref:.4f} Å\n\n")
            f.write("# Site positions relative to G1 C-alpha plane (Z=0):\n")
            for site, pos in sorted(optimized_sites_rel.items(), key=lambda item: item[1], reverse=True):
                f.write(f"{site}: {pos:.4f}\n")
        logger.info(f"OPTIMIZED binding site position data saved to {data_path}")
        # Register the definition file product (using the same subcategory as before)
        register_product(db_conn, module_name, "txt", "definition",
                         rel_path, subcategory="binding_sites_definition", # Keep subcategory consistent
                         description=description)
    except Exception as e:
         logger.error(f"Failed to save or register OPTIMIZED binding site position data: {e}")
         return None # Fail if we can't save the definitions

    return optimized_sites_rel