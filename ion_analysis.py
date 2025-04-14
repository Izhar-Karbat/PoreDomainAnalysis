# ion_analysis.py
"""
Functions for analyzing K+ ion behavior in the selectivity filter, including:
- Identifying filter residues (TVGYG).
- Calculating binding site positions (S0-S4, Cavity) relative to G1 C-alpha.
- Tracking ion positions near the filter.
- Analyzing site occupancy and coordination statistics.
- Generating relevant plots and data files.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde # Keep for density plot if still needed
from tqdm import tqdm
import json
from datetime import datetime
import MDAnalysis as mda
from MDAnalysis.analysis import distances

# Import from other modules
try:
    from utils import OneLetter, frames_to_time
    from logger_setup import setup_system_logger
except ImportError as e:
    print(f"Error importing dependency modules in ion_analysis.py: {e}")
    raise

# Configure matplotlib for non-interactive backend
plt.switch_backend('agg')
# Set a consistent plot style (optional)
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.2)

# Get a logger for this module (used by functions not setting up their own)
module_logger = logging.getLogger(__name__)

# --- Filter Residue Identification ---

def find_filter_residues(universe, logger=None):
    """
    Find the selectivity filter residues (assumed TVGYG-like).
    Identifies the 5 residues ending in GYG for each channel chain.

    Args:
        universe (MDAnalysis.Universe): The universe containing the system.
        logger (logging.Logger, optional): Logger instance. Defaults to module logger.

    Returns:
        dict | None: Dictionary mapping chain segids (e.g., 'PROA') to lists of
                     filter residue IDs (resids), or None if chains not found.
                     Returns None on significant error.
    """
    log = logger if logger else module_logger
    log.info("Searching for selectivity filter residues...")

    chain_segids = []
    # Prioritize PROA-D, then A-D
    potential_segids = ['PROA', 'PROB', 'PROC', 'PROD', 'A', 'B', 'C', 'D']
    found_segids = set(np.unique(universe.atoms.segids))

    for segid in potential_segids:
        if segid in found_segids:
            # Check if atoms actually exist for this segid
            if len(universe.select_atoms(f'segid {segid}')) > 0:
                 chain_segids.append(segid)
            else:
                 log.debug(f"Segid {segid} found in universe but selection yielded 0 atoms.")

    # Group by chain letter (A, B, C, D)
    grouped_segids = {}
    for segid in chain_segids:
        chain_letter = segid[-1] # Assumes A, B, C, D or PROA, PROB, etc.
        if chain_letter not in grouped_segids:
             grouped_segids[chain_letter] = segid # Prefer PROA over A if both exist
        elif segid.startswith("PRO") and not grouped_segids[chain_letter].startswith("PRO"):
             grouped_segids[chain_letter] = segid # Overwrite A with PROA

    final_chain_segids = list(grouped_segids.values())

    if not final_chain_segids:
        log.error("Could not find any standard channel chain segids (A/B/C/D or PROA/PROB/PROC/PROD).")
        return None
    log.info(f"Identified channel chains using segids: {final_chain_segids}")

    filter_residues = {}
    possible_filter_len = 5 # Assuming TVGYG structure

    for segid in final_chain_segids:
        try:
            chain = universe.select_atoms(f'segid {segid}')
            if not chain: continue # Should not happen based on earlier check

            # Get sequence of unique residue names
            resnames = [res.resname for res in chain.residues]
            if not resnames:
                log.warning(f"Chain {segid} contains no residues.")
                continue

            chain_seq = OneLetter("".join(resnames))
            log.debug(f"Chain {segid} sequence: {chain_seq[:30]}...") # Log only start

            # Find the last occurrence of 'GYG'
            idx_gyg_end = chain_seq.rfind('GYG') # Find from the right

            if idx_gyg_end != -1:
                # GYG found, the filter starts 2 residues before 'G'
                filter_start_seq_idx = idx_gyg_end - 2
                filter_end_seq_idx = idx_gyg_end + 3 # Exclusive end index for slicing

                if filter_start_seq_idx >= 0 and filter_end_seq_idx <= len(chain.residues):
                     # Extract the 5 residues based on sequence index
                     filter_res_group = chain.residues[filter_start_seq_idx:filter_end_seq_idx]
                     filter_resids = [res.resid for res in filter_res_group]

                     # Verification
                     filter_seq = OneLetter("".join(res.resname for res in filter_res_group))
                     if filter_seq.endswith("GYG") and len(filter_resids) == possible_filter_len:
                         filter_residues[segid] = filter_resids
                         log.info(f"Found filter for {segid}: Resids {filter_resids} (Sequence: {filter_seq})")
                     else:
                          log.warning(f"Potential filter match for {segid} at index {idx_gyg_end} failed validation (Seq: {filter_seq}, Len: {len(filter_resids)}).")
                else:
                     log.warning(f"GYG motif found too close to the start/end of chain {segid} sequence to form a {possible_filter_len}-residue filter.")

            else: # GYG not found - very unusual for K+ channels
                 log.error(f"Mandatory 'GYG' motif not found in chain {segid}. Cannot define filter.")
                 # Decide whether to continue or raise error. Let's skip this chain.

        except Exception as e:
            log.error(f"Error processing chain {segid} for filter residues: {e}", exc_info=True)

    if not filter_residues:
         log.error("Failed to identify filter residues for any chain.")
         return None
    if len(filter_residues) != 4:
         log.warning(f"Identified filter residues for only {len(filter_residues)} out of 4 expected chains.")

    return filter_residues


# --- Binding Site Calculation & Visualization ---

def calculate_tvgyg_sites(universe, filter_residues, logger=None):
    """
    Calculate binding site Z-positions based on carbonyl oxygens in the
    selectivity filter (TVGYG), relative to the G1 C-alpha plane (Z=0).

    Standard K+ channel nomenclature (extracellular to intracellular):
    - S0: ~3Å above S1
    - S1: Midway between G1 (lower G) and Y carbonyl oxygens
    - S2: Midway between V and G1 (lower G) carbonyl oxygens
    - S3: Midway between T and V carbonyl oxygens
    - S4: ~1.5Å below T carbonyl oxygen
    - Cavity: ~4.5Å below T carbonyl oxygen

    Args:
        universe (MDAnalysis.Universe): Universe object (frame 0 used for calculation).
        filter_residues (dict): Dictionary mapping chain segids to filter resids.
        logger (logging.Logger, optional): Logger instance. Defaults to module logger.

    Returns:
        tuple: (sites, g1_ca_z_ref)
               - sites (dict | None): Dictionary mapping site names (S0-S4, Cavity)
                                      to Z-coordinates (relative to G1 C-alpha=0),
                                      or None on failure.
               - g1_ca_z_ref (float | None): Absolute Z-coordinate of the G1 C-alpha
                                             reference plane, or None on failure.
    """
    log = logger if logger else module_logger
    log.info("Calculating binding site Z-positions relative to G1 C-alpha...")

    if not filter_residues:
        log.error("Filter residues dictionary is empty. Cannot calculate sites.")
        return None, None

    # Check if all chains have the expected number of residues
    expected_res_count = 5
    valid_chains = {segid: resids for segid, resids in filter_residues.items() if len(resids) == expected_res_count}
    if len(valid_chains) < 4: # Require all 4 chains for reliable averaging
        log.warning(f"Found only {len(valid_chains)} chains with {expected_res_count} filter residues. Site calculation might be less accurate.")
        if len(valid_chains) == 0:
             log.error("No chains with the correct number of filter residues found.")
             return None, None

    # --- Go to first frame ---
    try:
        universe.trajectory[0]
    except IndexError:
        log.error("Trajectory has no frames. Cannot calculate sites.")
        return None, None

    # --- Define residue positions relative to TVGYG (T=0, V=1, G1=2, Y=3, G2=4) ---
    residue_by_position = {
        'T': [], 'V': [], 'G1': [], 'Y': [], 'G2': []
    }
    position_map = {0: 'T', 1: 'V', 2: 'G1', 3: 'Y', 4: 'G2'}

    for segid, residues in valid_chains.items():
        for i, resid in enumerate(residues):
            pos_key = position_map.get(i)
            if pos_key:
                residue_by_position[pos_key].append((segid, resid))

    # --- Calculate G1 C-alpha reference plane (absolute Z) ---
    g1_ca_selection_parts = []
    for segid, resid in residue_by_position['G1']:
        g1_ca_selection_parts.append(f"(segid {segid} and resid {resid} and name CA)")

    if not g1_ca_selection_parts:
        log.error("Could not build selection string for G1 C-alpha atoms.")
        return None, None

    g1_ca_atoms = universe.select_atoms(" or ".join(g1_ca_selection_parts))
    if len(g1_ca_atoms) == 0:
        log.error(f"Selection for G1 C-alpha atoms returned 0 atoms. Selection: {' or '.join(g1_ca_selection_parts)}")
        return None, None

    g1_ca_z_ref = np.mean(g1_ca_atoms.positions[:, 2])
    log.info(f"Reference plane: Average G1 C-alpha absolute Z = {g1_ca_z_ref:.3f} Å")

    # --- Calculate average Z-position of carbonyl oxygens for each filter residue ---
    carbonyl_z_rel = {} # Store relative Z positions
    try:
        for pos_key, residue_list in residue_by_position.items():
            selection_parts = []
            for segid, resid in residue_list:
                # Select backbone carbonyl oxygen 'O'
                selection_parts.append(f"(segid {segid} and resid {resid} and name O)")

            if not selection_parts:
                 log.warning(f"No residues found for position {pos_key}. Cannot calculate carbonyl position.")
                 continue

            carbonyl_atoms = universe.select_atoms(" or ".join(selection_parts))
            if len(carbonyl_atoms) > 0:
                # Calculate average absolute Z
                raw_z_pos = np.mean(carbonyl_atoms.positions[:, 2])
                # Calculate relative position to G1 C-alpha plane
                rel_z_pos = raw_z_pos - g1_ca_z_ref
                carbonyl_z_rel[pos_key] = rel_z_pos
                log.debug(f"Carbonyl Z for {pos_key}: {rel_z_pos:.3f} Å (relative to G1 Cα)")
            else:
                 log.warning(f"No carbonyl oxygens ('O') found for position {pos_key}.")

    except Exception as e:
         log.error(f"Error calculating carbonyl positions: {e}", exc_info=True)
         return None, None

    # --- Calculate Binding Site Positions (relative to G1 C-alpha = 0) ---
    sites = {}
    required_keys = ['T', 'V', 'G1', 'Y'] # G2 not directly needed for S0-S4/Cavity definition
    if not all(key in carbonyl_z_rel for key in required_keys):
         missing = [key for key in required_keys if key not in carbonyl_z_rel]
         log.error(f"Missing carbonyl Z positions for required residues: {missing}. Cannot define all sites.")
         # Optionally return partial sites or None
         return None, g1_ca_z_ref # Return reference even if sites fail

    try:
        sites['S1'] = (carbonyl_z_rel['G1'] + carbonyl_z_rel['Y']) / 2
        sites['S0'] = sites['S1'] + 3.0 # Definition: 3A above S1
        sites['S2'] = (carbonyl_z_rel['V'] + carbonyl_z_rel['G1']) / 2
        sites['S3'] = (carbonyl_z_rel['T'] + carbonyl_z_rel['V']) / 2
        sites['S4'] = carbonyl_z_rel['T'] - 1.5 # Definition: 1.5A below T
        sites['Cavity'] = carbonyl_z_rel['T'] - 4.5 # Definition: 4.5A below T (start of cavity)

        log.info("Binding site Z-positions calculated (relative to G1 Cα = 0):")
        for site, pos in sites.items():
            log.info(f"  {site}: {pos:.3f} Å")

        return sites, g1_ca_z_ref

    except KeyError as e:
        log.error(f"KeyError calculating binding sites, missing carbonyl data for {e}")
        return None, g1_ca_z_ref
    except Exception as e:
         log.error(f"Unexpected error calculating binding sites: {e}", exc_info=True)
         return None, g1_ca_z_ref


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


# --- Ion Tracking ---

def track_potassium_ions(run_dir, psf_file=None, dcd_file=None):
    """
    Track K+ ions near the selectivity filter over the trajectory.

    Identifies filter, calculates sites, finds nearby K+ ions, records their Z positions,
    and saves data/plots.

    Args:
        run_dir (str): Directory containing trajectory files and for saving results.
        psf_file (str, optional): Path to PSF file. Defaults to 'step5_input.psf'.
        dcd_file (str, optional): Path to DCD file. Defaults to 'MD_Aligned.dcd'.

    Returns:
        tuple: (ions_z_abs, time_points, ion_indices, g1_ref, sites, filter_residues)
               - ions_z_abs (dict): {ion_idx: np.array(abs_z_positions)}
               - time_points (np.ndarray): Time points in ns.
               - ion_indices (list): Indices of tracked ions.
               - g1_ref (float | None): Absolute Z of G1 C-alpha plane.
               - sites (dict | None): Site positions relative to G1 C-alpha=0.
               - filter_residues (dict): Identified filter residues mapping.
               Returns ({}, np.array([0]), [], None, None, None) on critical error.
    """
    # Setup logger for this run
    logger = setup_system_logger(run_dir)
    if logger is None:
        logger = logging.getLogger()  # Fallback

    run_name = os.path.basename(run_dir)
    logger.info(f"Starting K+ ion tracking for {run_name}")

    # --- File Handling & Universe Loading ---
    if psf_file is None:
        psf_file = os.path.join(run_dir, "step5_input.psf")
    if dcd_file is None:
        dcd_file = os.path.join(run_dir, "MD_Aligned.dcd")
    if not os.path.exists(psf_file) or not os.path.exists(dcd_file):
        logger.error(f"PSF or DCD file missing for ion tracking in {run_dir}")
        return {}, np.array([0]), [], None, None, None

    try:
        u = mda.Universe(psf_file, dcd_file)
        n_frames = len(u.trajectory)
        if n_frames == 0:
            logger.warning("Trajectory has 0 frames. Cannot track ions.")
            return {}, np.array([]), [], None, None, None
        logger.info(f"Loaded universe for ion tracking ({n_frames} frames).")
    except Exception as e:
        logger.error(f"Failed to load Universe for ion tracking: {e}", exc_info=True)
        return {}, np.array([0]), [], None, None, None

    # --- Identify Filter and Sites ---
    filter_residues = find_filter_residues(u, logger)
    if not filter_residues:
        logger.error("Failed to identify filter residues. Aborting ion tracking.")
        return {}, frames_to_time(np.arange(n_frames)), [], None, None, None

    filter_sites, g1_reference = calculate_tvgyg_sites(u, filter_residues, logger)
    if not filter_sites or g1_reference is None:
        logger.error("Failed to calculate binding sites or G1 reference. Aborting ion tracking.")
        return {}, frames_to_time(np.arange(n_frames)), [], None, None, filter_residues

    # Create binding site visualization (using G1-centric coordinates)
    _ = visualize_binding_sites_g1_centric(filter_sites, g1_reference, run_dir, logger)

    # --- Setup Ion Selection & Tracking ---
    k_selection_strings = ['name K', 'name POT', 'resname K', 'resname POT', 'type K', 'element K']
    potassium_atoms = None
    for sel in k_selection_strings:
        try:
            atoms = u.select_atoms(sel)
            if len(atoms) > 0:
                potassium_atoms = atoms
                logger.info(f"Found {len(potassium_atoms)} K+ ions using selection: '{sel}'")
                break
        except Exception as e_sel:
            logger.debug(f"Selection '{sel}' failed: {e_sel}")

    if not potassium_atoms:
        logger.error("No potassium ions found using standard selections.")
        return {}, frames_to_time(np.arange(n_frames)), [], g1_reference, filter_sites, filter_residues

    # Create selection for FILTER CARBONYL OXYGEN atoms
    filter_sel_parts = []
    for segid, resids in filter_residues.items():
        if len(resids) == 5:  # Ensure we have TVGYG
            # Select only 'O' atoms for the specific filter resids in this chain
            filter_sel_parts.append(f"(segid {segid} and name O and resid {' '.join(map(str, resids))})")

    if not filter_sel_parts:
        logger.error("Could not build selection string for filter carbonyl oxygens.")
        time_points_err = frames_to_time(np.arange(u.trajectory.n_frames)) if u else np.array([0])
        return {}, time_points_err, [], g1_reference, filter_sites, filter_residues

    filter_selection_str = " or ".join(filter_sel_parts)
    logger.info(f"Using filter CARBONYL OXYGEN selection: {filter_selection_str}")

    filter_atoms_group = u.select_atoms(filter_selection_str)  # Select the group once
    if not filter_atoms_group:
        logger.error("Could not select filter carbonyl oxygens! Aborting ion tracking.")
        time_points_err = frames_to_time(np.arange(u.trajectory.n_frames)) if u else np.array([0])
        return {}, time_points_err, [], g1_reference, filter_sites, filter_residues

    # Variables for tracking
    ions_z_positions_abs = {ion.index: [] for ion in potassium_atoms}  # Store absolute Z
    frame_indices = []
    tracked_ion_indices_near_filter = set()  # Ions that came close at least once
    cutoff = 3.5
    logger.info(f"Tracking K+ ions within {cutoff} Å of filter atoms...")

    # --- Trajectory Iteration ---
    for ts in tqdm(u.trajectory, desc=f"Tracking K+ ({run_name})", unit="frame"):
        frame_indices.append(ts.frame)
        filter_atoms_pos = filter_atoms_group.positions  # Get current positions
        potassium_pos = potassium_atoms.positions
        box = u.dimensions  # Use box dimensions for distance calculation

        # Calculate distances between all K+ ions and all filter atoms
        dist_matrix = distances.distance_array(potassium_pos, filter_atoms_pos, box=box)

        # Find minimum distance for each K+ ion to any filter atom
        min_dists = np.min(dist_matrix, axis=1)

        # Identify ions currently near the filter
        ions_near_this_frame = potassium_atoms[min_dists <= cutoff]

        # Update the set of ions that have been near the filter at least once
        tracked_ion_indices_near_filter.update(ions_near_this_frame.indices)

        # Record absolute Z position for ALL potassium ions in this frame
        for i, ion in enumerate(potassium_atoms):
            ions_z_positions_abs[ion.index].append(potassium_pos[i, 2])

    # --- Post-Processing ---
    time_points = frames_to_time(frame_indices)
    final_tracked_indices = sorted(list(tracked_ion_indices_near_filter))
    logger.info(f"Identified {len(final_tracked_indices)} unique K+ ions passing near the filter.")

    # Filter the main dictionary to keep only tracked ions
    ions_z_tracked_abs = {idx: np.array(ions_z_positions_abs[idx]) for idx in final_tracked_indices if idx in ions_z_positions_abs}

    # Save position data (Absolute and G1-centric)
    save_ion_position_data(run_dir, time_points, ions_z_tracked_abs, final_tracked_indices, g1_reference)

    # Plot positions (using G1-centric coordinates)
    plot_ion_positions(run_dir, time_points, ions_z_tracked_abs, final_tracked_indices, filter_sites, g1_reference, logger=logger)

    return ions_z_tracked_abs, time_points, final_tracked_indices, g1_reference, filter_sites, filter_residues


# --- Ion Data Saving & Plotting ---

def save_ion_position_data(run_dir, time_points, ions_z_positions_abs, ion_indices, g1_reference):
    """
    Save K+ ion Z-position data (absolute and G1-centric) to CSV files.

    Args:
        run_dir (str): Directory to save the CSV files.
        time_points (np.ndarray): Array of time points in ns.
        ions_z_positions_abs (dict): {ion_idx: np.array(abs_z_positions)}
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
            # Calculate G1-centric positions
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

    # Save metadata about the reference frame
    meta_path = os.path.join(run_dir, 'K_Ion_Coordinate_Reference.txt')
    try:
        with open(meta_path, 'w') as f:
            f.write(f"# K+ ion position coordinate reference information\n")
            f.write(f"# Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"G1_C_alpha_reference_position_absolute_Z: {g1_reference:.4f} Å\n")
            f.write(f"Conversion_formula: Z_G1_centric = Z_Absolute - G1_C_alpha_reference_position_absolute_Z\n")
        module_logger.info(f"Saved coordinate reference info to {meta_path}")
    except Exception as e:
         module_logger.error(f"Failed to save coordinate reference file: {e}")


def plot_ion_positions(run_dir, time_points, ions_z_positions_abs, ion_indices, filter_sites, g1_reference, logger=None):
    """
    Plot K+ ion Z positions (G1-centric) over time and their density distribution.

    Args:
        run_dir (str): Directory to save the plots.
        time_points (np.ndarray): Time points in ns.
        ions_z_positions_abs (dict): {ion_idx: np.array(abs_z_positions)}
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
                ax1.plot(time_points, ions_z_g1[ion_idx], color=plot_colors[i], linewidth=1.0, alpha=0.8)
                # Optional: Add label if few ions
                # if len(ion_indices) <= 10: ax1.plot(..., label=f'Ion {ion_idx}')

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
        all_z_g1_flat = np.concatenate([arr for idx, arr in ions_z_g1.items() if arr is not None])
        finite_z_g1 = all_z_g1_flat[np.isfinite(all_z_g1_flat)]

        if len(finite_z_g1) > 10: # Need sufficient points for KDE
             sns.kdeplot(y=finite_z_g1, ax=ax2, color='black', fill=True, alpha=0.2, linewidth=1.5)
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
        elif len(finite_z_g1) > 0 :
            y_min = np.nanmin(finite_z_g1) - 3.0
            y_max = np.nanmax(finite_z_g1) + 3.0
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

    # --- Standalone Density Plot (optional, maybe remove if combined is sufficient) ---
    # (Code similar to right panel of combined plot)

    # --- Occupancy Heatmap/Bar Plot (using dedicated function) ---
    try:
        _ = create_ion_occupancy_heatmap(run_dir, time_points, ions_z_g1, ion_indices, filter_sites, logger=log)
    except Exception as e:
        log.error(f"Failed to create ion occupancy heatmap/plots: {e}", exc_info=True)


def create_ion_occupancy_heatmap(run_dir, time_points, ions_z_g1_centric, ion_indices, filter_sites, logger=None):
    """
    Create heatmap and bar chart showing K+ ion occupancy in binding sites.
    Saves plots and occupancy data CSV.

    Args:
        run_dir (str): Directory to save the plot and CSV.
        time_points (np.ndarray): Time points in ns.
        ions_z_g1_centric (dict): {ion_idx: np.array(g1_centric_z_pos)}
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
                if np.isfinite(z_pos):
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


# --- Ion Coordination Statistics ---

def analyze_ion_coordination(run_dir, time_points, ions_z_positions_abs, ion_indices, filter_sites, g1_reference):
    """
    Analyze K+ ion occupancy statistics for each binding site.
    Uses G1-centric coordinates for calculations. Saves stats to CSV.

    Args:
        run_dir (str): Directory to save the analysis results CSV.
        time_points (np.ndarray): Array of time points in ns.
        ions_z_positions_abs (dict): {ion_idx: np.array(abs_z_positions)}
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
