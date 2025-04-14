"""
Core analysis functions for reading trajectories, calculating raw G-G and COM distances,
applying filtering, and saving/plotting basic distance results.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import MDAnalysis as mda
from MDAnalysis.analysis import distances
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks  # Ensure this is imported for peak detection

# Import from other modules
try:
    from utils import OneLetter, frames_to_time
    from filtering import auto_select_filter  # Import the main filtering function
    from logger_setup import setup_system_logger
except ImportError as e:
    # Handle potential import errors if modules are not found
    # This might happen if running a module standalone without proper path setup
    print(f"Error importing dependency modules in core_analysis.py: {e}")
    raise

# Configure matplotlib for non-interactive backend suitable for scripts
plt.switch_backend('agg')
# Set a consistent plot style (optional, could also be set once in main)
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.2)
logger = logging.getLogger(__name__)


# --- Trajectory Reading and Raw Data Extraction ---

def analyze_trajectory(run_dir, psf_file=None, dcd_file=None):
    """
    Analyze a single trajectory file to extract raw G-G and COM distances.
    Opens the trajectory, calculates metrics per frame, and saves raw data to CSV files.
    Detects if this is a control system (no toxin) and records this information.

    Args:
        run_dir (str): Path to the specific run directory (used for output and logging).
        psf_file (str, optional): Path to the PSF topology file. Defaults to 'step5_input.psf' in run_dir.
        dcd_file (str, optional): Path to the DCD trajectory file. Defaults to 'MD_Aligned.dcd' in run_dir.

    Returns:
        tuple: Contains:
            - dist_ac (np.ndarray): Raw A-C G-G distances.
            - dist_bd (np.ndarray): Raw B-D G-G distances.
            - com_distances (np.ndarray | None): Raw Toxin-Channel COM distances, or None if no toxin.
            - time_points (np.ndarray): Time points corresponding to frames (in ns).
            - system_dir (str): Name of the parent directory (system name). Returns 'Unknown' on error.
            - is_control_system (bool): Flag indicating if this is a control system (no toxin present).
              Returns True if no toxin chain was found, False otherwise.
            Returns (np.array([0]), np.array([0]), None, np.array([0]), 'Unknown', True) on critical error.
    """
    # Set up system-specific logger
    logger = setup_system_logger(run_dir)
    if logger is None:
        # Fallback to root logger if specific logger setup failed
        logger = logging.getLogger()
        logger.error(f"Failed to setup system logger for {run_dir}. Using root logger.")

    run_name = os.path.basename(run_dir)
    system_dir = os.path.basename(os.path.dirname(run_dir)) if os.path.dirname(run_dir) else run_name
    logger.info(f"Starting raw trajectory analysis for system: {system_dir} (Folder: {run_dir})")

    # --- File Handling ---
    if psf_file is None:
        psf_file = os.path.join(run_dir, "step5_input.psf")
    if dcd_file is None:
        dcd_file = os.path.join(run_dir, "MD_Aligned.dcd")

    if not os.path.exists(psf_file):
        logger.error(f"PSF file not found: {psf_file}")
        return np.array([0]), np.array([0]), None, np.array([0]), 'Unknown', True  # Assume control on error
    if not os.path.exists(dcd_file):
        logger.error(f"DCD file not found: {dcd_file}")
        return np.array([0]), np.array([0]), None, np.array([0]), 'Unknown', True  # Assume control on error

    # --- Load Universe ---
    try:
        logger.info(f"Loading topology: {psf_file}")
        logger.info(f"Loading trajectory: {dcd_file}")
        u = mda.Universe(psf_file, dcd_file)
        n_frames = len(u.trajectory)
        logger.info(f"Successfully loaded universe with {n_frames} frames")
        if n_frames == 0:
            logger.warning("Trajectory contains 0 frames.")
            # Treat empty trajectory as control? Or error? Let's assume control for now.
            return np.array([]), np.array([]), None, np.array([]), system_dir, True
    except Exception as e:
        logger.error(f"Failed to load MDAnalysis Universe: {e}", exc_info=True)
        return np.array([]), np.array([]), None, np.array([]), 'Unknown', True  # Assume control on error

    # --- Setup Selections ---
    dist_ac = []
    dist_bd = []
    pep_positions = []
    ch_positions = []
    frame_indices = []
    idx_gyg = None  # Initialize

    # G-G Selection (find central Glycine)
    try:
        # Prefer PROA, fallback to A
        chain_A_atoms = u.select_atoms('segid PROA')
        segid_A = 'PROA'
        if not chain_A_atoms:
            chain_A_atoms = u.select_atoms('segid A')
            segid_A = 'A'
        if not chain_A_atoms:
            raise ValueError(f"Could not find chain A using 'segid PROA' or 'segid A'. Available: {np.unique(u.atoms.segids)}")

        chainA_Seq = OneLetter("".join(chain_A_atoms.residues.resnames))
        logger.info(f"Chain {segid_A} sequence for G-G: {chainA_Seq}")

        idx_gyg = chainA_Seq.find('GYG')
        if idx_gyg == -1:
            logger.warning("'GYG' motif not found. Using fallback middle Glycine.")
            gly_indices = [i for i, res in enumerate(chainA_Seq) if res == 'G']
            if not gly_indices:
                raise ValueError("No GYG motif and no Glycine residues found in chain A.")
            idx_gyg = gly_indices[len(gly_indices) // 2]  # Index in sequence
        else:
            idx_gyg = idx_gyg + 1  # Use the middle G index (1-based relative to GYG start)

        # Convert sequence index to resid
        reference_resid = chain_A_atoms.residues[idx_gyg].resid
        logger.info(f"Using CA of residue {reference_resid} (Index {idx_gyg} in sequence '{chainA_Seq}') for G-G distance.")

    except Exception as e:
        logger.error(f"Error finding reference Glycine for G-G distance: {e}", exc_info=True)
        return np.array([]), np.array([]), None, np.array([]), 'Unknown', True  # Assume control on error

    # COM Selection (Toxin vs Channel)
    toxin_segids = ['PROE', 'E', 'PEP', 'TOX']
    channel_segids_pro = ['PROA', 'PROB', 'PROC', 'PROD']
    channel_segids_alt = ['A', 'B', 'C', 'D']

    pep_sel = None
    for seg in toxin_segids:
        # Check if atoms actually exist for this segid in the Universe
        # Using u.select_atoms is more robust than checking segids array
        if len(u.select_atoms(f'segid {seg}')) > 0:
            pep_sel = f'segid {seg}'
            logger.info(f"Found Toxin selection: '{pep_sel}'")
            break

    chan_sel = 'segid ' + ' or segid '.join(channel_segids_pro)
    if len(u.select_atoms(chan_sel)) == 0:
        # Try alternate channel segids if PROA etc. not found
        chan_sel_alt = 'segid ' + ' or segid '.join(channel_segids_alt)
        if len(u.select_atoms(chan_sel_alt)) > 0:
             chan_sel = chan_sel_alt
             logger.info(f"Using Alternate Channel selection: '{chan_sel}'")
        else:
             logger.warning(f"Could not find channel atoms using standard selections: '{chan_sel}' or '{chan_sel_alt}'.")
             # Attempt to proceed with G-G only if possible? For now, treat as error.
             return np.array([]), np.array([]), None, np.array([]), 'Unknown', True # Assume control if channel not found

    else:
        logger.info(f"Using Channel selection: '{chan_sel}'")

    # Determine if COM analysis is possible (if toxin was found)
    com_analyzed = pep_sel is not None
    is_control_system = not com_analyzed  # NEW: Flag to indicate control system (no toxin)

    if is_control_system:
        logger.info("No toxin selection found. This appears to be a CONTROL system without toxin.")
    else:
        logger.info(f"Toxin found with selection '{pep_sel}'. This appears to be a toxin-channel complex system.")

    # --- Trajectory Iteration ---
    logger.info(f"Starting trajectory analysis loop ({n_frames} frames)...")
    try:
        for ts in u.trajectory:
            frame_indices.append(ts.frame)

            # G-G Calculation
            try:
                ca_atoms = {}
                found_all_gg = True
                # Iterate through pairs of opposing chains (A/C, B/D) based on selected channel segids
                opposing_chains = [('A', 'C'), ('B', 'D')]
                current_channel_segids = [s.replace('segid ', '') for s in chan_sel.split(' or ')]

                # Determine the base names (A, B, C, D) from the found channel segids
                chain_map = {}
                for seg in current_channel_segids:
                    if seg.startswith("PRO"): chain_map[seg[-1]] = seg # Map A to PROA etc.
                    else: chain_map[seg] = seg # Map A to A etc.

                for chain1_base, chain2_base in opposing_chains:
                    segid1 = chain_map.get(chain1_base)
                    segid2 = chain_map.get(chain2_base)

                    if not segid1 or not segid2:
                         logger.warning(f"Could not find matching segids for G-G pair {chain1_base}-{chain2_base} from {current_channel_segids}")
                         found_all_gg = False
                         continue

                    # Find CA for chain 1
                    sel_str1 = f'resid {reference_resid} and name CA and segid {segid1}'
                    atoms1 = u.select_atoms(sel_str1)
                    if len(atoms1) == 1: ca_atoms[chain1_base] = atoms1
                    else: logger.warning(f"G-G: Found {len(atoms1)} atoms for {sel_str1} at frame {ts.frame}"); found_all_gg = False; ca_atoms[chain1_base] = None

                    # Find CA for chain 2
                    sel_str2 = f'resid {reference_resid} and name CA and segid {segid2}'
                    atoms2 = u.select_atoms(sel_str2)
                    if len(atoms2) == 1: ca_atoms[chain2_base] = atoms2
                    else: logger.warning(f"G-G: Found {len(atoms2)} atoms for {sel_str2} at frame {ts.frame}"); found_all_gg = False; ca_atoms[chain2_base] = None

                # Calculate distances if all atoms found for both pairs
                if ca_atoms.get('A') and ca_atoms.get('C'):
                    dist_ac.append(float(distances.dist(ca_atoms['A'], ca_atoms['C'])[2]))
                else:
                    dist_ac.append(np.nan)

                if ca_atoms.get('B') and ca_atoms.get('D'):
                    dist_bd.append(float(distances.dist(ca_atoms['B'], ca_atoms['D'])[2]))
                else:
                    dist_bd.append(np.nan)

            except Exception as e_gg:
                logger.warning(f"Error calculating G-G distances at frame {ts.frame}: {e_gg}", exc_info=False)
                dist_ac.append(np.nan)
                dist_bd.append(np.nan)

            # COM Calculation (only if toxin exists - not for control systems)
            if com_analyzed:
                try:
                    pep = u.select_atoms(pep_sel)
                    cha = u.select_atoms(chan_sel)
                    if len(pep) > 0 and len(cha) > 0:
                        pep_pos = pep.center_of_mass()
                        cha_pos = cha.center_of_mass()
                        pep_positions.append(pep_pos.tolist())
                        ch_positions.append(cha_pos.tolist())
                    else:
                        # Log if selections become empty mid-trajectory
                        if len(pep) == 0: logger.warning(f"Toxin selection '{pep_sel}' empty at frame {ts.frame}")
                        if len(cha) == 0: logger.warning(f"Channel selection '{chan_sel}' empty at frame {ts.frame}")
                        pep_positions.append([np.nan, np.nan, np.nan])
                        ch_positions.append([np.nan, np.nan, np.nan])
                except Exception as e_com:
                    logger.warning(f"Error calculating COM at frame {ts.frame}: {e_com}", exc_info=False)
                    pep_positions.append([np.nan, np.nan, np.nan])
                    ch_positions.append([np.nan, np.nan, np.nan])

    except Exception as loop_err:
        logger.error(f"Error during trajectory loop: {loop_err}", exc_info=True)
        # Return results processed so far, if any, otherwise empty/defaults
        time_points_err = frames_to_time(frame_indices) if frame_indices else np.array([])
        dist_ac_err = np.array(dist_ac) if dist_ac else np.array([])
        dist_bd_err = np.array(dist_bd) if dist_bd else np.array([])
        com_dist_err = np.linalg.norm(np.array(pep_positions) - np.array(ch_positions), axis=1) if com_analyzed and pep_positions and ch_positions else None
        return dist_ac_err, dist_bd_err, com_dist_err, time_points_err, system_dir, is_control_system # Pass calculated flag

    logger.info(f"Completed trajectory analysis loop. Processed {len(frame_indices)} frames.")

    # --- Process Results ---
    time_points = frames_to_time(frame_indices)
    dist_ac = np.array(dist_ac)
    dist_bd = np.array(dist_bd)

    com_distances_raw = None
    if com_analyzed:
        # Check if positions were actually collected before calculating norm
        if pep_positions and ch_positions and len(pep_positions) == len(frame_indices):
            pep_positions_np = np.array(pep_positions)
            ch_positions_np = np.array(ch_positions)
            rel_vector = pep_positions_np - ch_positions_np
            com_distances_raw = np.linalg.norm(rel_vector, axis=1)

            com_df = pd.DataFrame({
                'Frame': frame_indices,
                'Time (ns)': time_points,
                'COM_Distance_Raw': com_distances_raw
            })
            com_csv_path = os.path.join(run_dir, "COM_Stability_Raw.csv")
            try:
                com_df.to_csv(com_csv_path, index=False, float_format='%.4f', na_rep='NaN')
                logger.info(f"Saved raw COM data to {com_csv_path}")
            except Exception as e:
                logger.error(f"Failed to save raw COM CSV: {e}")
        else:
             logger.warning("COM analysis was enabled, but no COM positions were collected (check trajectory loop warnings). Setting raw COM distances to None.")


    gg_df = pd.DataFrame({
        'Frame': frame_indices,
        'Time (ns)': time_points,
        'A_C_Distance_Raw': dist_ac,
        'B_D_Distance_Raw': dist_bd
    })
    gg_csv_path = os.path.join(run_dir, "G_G_Distance_Raw.csv")
    try:
        gg_df.to_csv(gg_csv_path, index=False, float_format='%.4f', na_rep='NaN')
        logger.info(f"Saved raw G-G distance data to {gg_csv_path}")
    except Exception as e:
        logger.error(f"Failed to save raw G-G CSV: {e}")

    # Save system type info to a file for easier access by other modules
    system_type_path = os.path.join(run_dir, "system_type.txt")
    try:
        with open(system_type_path, 'w') as f:
            f.write(f"IS_CONTROL_SYSTEM={is_control_system}\\n")
            f.write(f"HAS_TOXIN={com_analyzed}\\n")
            f.write(f"TOXIN_SELECTION={pep_sel if pep_sel else 'None'}\\n")
            f.write(f"CHANNEL_SELECTION={chan_sel}\\n")
        logger.info(f"Saved system type information to {system_type_path}")
    except Exception as e:
        logger.warning(f"Failed to save system type information: {e}")

    # Final return with the determined control system status
    return dist_ac, dist_bd, com_distances_raw, time_points, system_dir, is_control_system


# --- Filtering Application and Plotting ---

def filter_and_save_data(run_dir, dist_ac, dist_bd, com_distances, time_points, box_z=None, is_control_system=False):
    """
    Apply filtering to the raw distance data (G-G and COM) and save filtered data and plots.
    Handles both normal and control systems appropriately.

    Args:
        run_dir (str): Path to the specific run directory for output.
        dist_ac (np.ndarray): Raw A-C G-G distances.
        dist_bd (np.ndarray): Raw B-D G-G distances.
        com_distances (np.ndarray | None): Raw Toxin-Channel COM distances.
        time_points (np.ndarray): Time points corresponding to frames (in ns).
        box_z (float, optional): Estimated or known box Z-dimension, passed to
                                 multi-level filter if used for COM distance.
        is_control_system (bool, optional): Flag indicating if this is a control system (no toxin).
                                            Defaults to False.

    Returns:
        tuple: Contains:
            - filtered_ac (np.ndarray | None): Filtered A-C G-G distances or None.
            - filtered_bd (np.ndarray | None): Filtered B-D G-G distances or None.
            - filtered_com (np.ndarray | None): Filtered COM distances or None.
            - filter_info_gg (dict): Information about G-G filtering.
            - filter_info_com (dict): Information about COM filtering.
            - raw_stats (dict): Statistics calculated on the raw, unfiltered data.
            - percentile_stats (dict): 10th and 90th percentiles for raw and filtered data.
    """
    logger.info("Starting data filtering and saving process.")
    os.makedirs(run_dir, exist_ok=True)

    # --- Calculate Raw Stats ---
    # Before filtering, calculate the standard deviation and mean for the raw input arrays:
    raw_stats = {}
    raw_stats['GG_AC_Std_Raw'] = np.nanstd(dist_ac) if dist_ac is not None else np.nan
    raw_stats['GG_BD_Std_Raw'] = np.nanstd(dist_bd) if dist_bd is not None else np.nan
    raw_stats['COM_Std_Raw'] = np.nanstd(com_distances) if com_distances is not None else np.nan
    raw_stats['GG_AC_Mean_Raw'] = np.nanmean(dist_ac) if dist_ac is not None else np.nan
    raw_stats['GG_BD_Mean_Raw'] = np.nanmean(dist_bd) if dist_bd is not None else np.nan
    raw_stats['COM_Mean_Raw'] = np.nanmean(com_distances) if com_distances is not None else np.nan

    # --- Filter G-G Distances --- (Always perform for both system types)
    filtered_ac, filter_info_ac = auto_select_filter(dist_ac, data_type='gg_distance')
    filtered_bd, filter_info_bd = auto_select_filter(dist_bd, data_type='gg_distance')
    filter_info_gg = {'AC': filter_info_ac, 'BD': filter_info_bd}

    # --- Filter COM Distances --- (Skip for control systems)
    filtered_com = None
    filter_info_com = {}

    if is_control_system:
        logger.info("Control system detected - skipping COM distance filtering.")
        filter_info_com = {
            'Method': 'None',
            'Threshold': None,
            'Window': None,
            'Polyorder': None,
            'Level': None,
            'Threshold_Factor': None,
            'Z_Threshold_Factor': None,
            'is_control_system': True
        }
        # Ensure filtered_com remains None
        filtered_com = None
    elif com_distances is not None and len(com_distances) > 1:
        try:
            kwargs = {'box_size': box_z} if box_z is not None else {}
            filtered_com, filter_info_com = auto_select_filter(
                com_distances,
                data_type='com_distance',
                **kwargs
            )
            filter_info_com['is_control_system'] = False
        except Exception as e:
            logger.error(f"Error filtering COM data: {e}", exc_info=True)
            filter_info_com = {'error': str(e), 'is_control_system': False}
            filtered_com = np.array(com_distances) # Use raw if filter fails

        # Save filtered COM CSV only if filtering was attempted
        com_filtered_df = pd.DataFrame({
            'Time (ns)': time_points,
            'COM_Distance_Raw': com_distances,
            'COM_Distance_Filtered': filtered_com if filtered_com is not None else np.nan
        })
        com_filtered_path = os.path.join(run_dir, "COM_Stability_Filtered.csv")
        try:
            com_filtered_df.to_csv(com_filtered_path, index=False, float_format='%.4f', na_rep='NaN')
            logger.info(f"Saved filtered COM data to {com_filtered_path}")
        except Exception as e:
            logger.error(f"Failed to save filtered COM CSV: {e}")
    elif com_distances is not None: # Case where COM data exists but too short to filter
        logger.info(f"COM distance data too short to filter ({len(com_distances)} points). Skipping filtering.")
        filter_info_com = {'reason': 'Insufficient data length', 'is_control_system': False}
        filtered_com = np.array(com_distances) # Return raw as filtered
    else: # Case where com_distances was None initially (should be caught by is_control_system=True now)
        logger.info("No COM data provided.")
        filter_info_com = {'reason': 'No COM data provided', 'is_control_system': is_control_system}
        filtered_com = None

    # --- Calculate Percentiles ---
    percentile_stats = {}

    def _calculate_percentiles(data, prefix, suffix):
        stats = {}
        keys = [f'{prefix}_Pctl10_{suffix}', f'{prefix}_Pctl90_{suffix}']
        if data is not None and len(data) > 0:
            finite_data = data[np.isfinite(data)]
            if len(finite_data) >= 1: # np.percentile needs at least one element
                try:
                    pctl = np.nanpercentile(finite_data, [10, 90])
                    stats[keys[0]] = float(pctl[0])
                    stats[keys[1]] = float(pctl[1])
                except ValueError: # Catch potential issues with percentile calculation
                    logger.warning(f"Could not calculate percentiles for {prefix}_{suffix} due to ValueError.", exc_info=False)
                    stats[keys[0]] = np.nan
                    stats[keys[1]] = np.nan
            else:
                logger.warning(f"Insufficient finite data points ({len(finite_data)}) to calculate percentiles for {prefix}_{suffix}.")
                stats[keys[0]] = np.nan
                stats[keys[1]] = np.nan
        else:
            stats[keys[0]] = np.nan # Use NaN if input data is None or empty
            stats[keys[1]] = np.nan
        return stats

    # Always calculate G-G percentiles
    percentile_stats.update(_calculate_percentiles(dist_ac, 'GG_AC', 'Raw'))
    percentile_stats.update(_calculate_percentiles(dist_bd, 'GG_BD', 'Raw'))
    percentile_stats.update(_calculate_percentiles(filtered_ac, 'GG_AC', 'Filt'))
    percentile_stats.update(_calculate_percentiles(filtered_bd, 'GG_BD', 'Filt'))

    # Only calculate COM percentiles for non-control systems
    if not is_control_system:
        percentile_stats.update(_calculate_percentiles(com_distances, 'COM', 'Raw'))
        percentile_stats.update(_calculate_percentiles(filtered_com, 'COM', 'Filt'))
    else:
        # For control systems, explicitly set COM percentiles to None (not NaN)
        percentile_stats['COM_Pctl10_Raw'] = None
        percentile_stats['COM_Pctl90_Raw'] = None
        percentile_stats['COM_Pctl10_Filt'] = None
        percentile_stats['COM_Pctl90_Filt'] = None

    # --- Save Filtered G-G Data --- (Always perform)
    if filtered_ac is not None or filtered_bd is not None:
        gg_df_filt = pd.DataFrame({
            'Time (ns)': time_points,
            'A_C_Distance_Raw': dist_ac if dist_ac is not None else np.nan,
            'B_D_Distance_Raw': dist_bd if dist_bd is not None else np.nan,
            'A_C_Distance_Filtered': filtered_ac if filtered_ac is not None else np.nan,
            'B_D_Distance_Filtered': filtered_bd if filtered_bd is not None else np.nan
        })
        gg_filtered_path = os.path.join(run_dir, "G_G_Distance_Filtered.csv")
        try:
            gg_df_filt.to_csv(gg_filtered_path, index=False, float_format='%.4f', na_rep='NaN')
            logger.info(f"Saved filtered G-G data to {gg_filtered_path}")
        except Exception as e:
            logger.error(f"Failed to save filtered G-G CSV: {e}")

    # --- Generate and Save Plots --- #
    run_name = os.path.basename(run_dir) # Get run name for titles
    try:
        logger.info("Generating comparison and final plots...")

        # G-G Plots (Always generate)
        fig_gg_ac = plot_filtering_comparison(time_points, dist_ac, filtered_ac, filter_info_gg.get('AC', {}), f"{run_name} - A:C", data_type='gg')
        fig_gg_bd = plot_filtering_comparison(time_points, dist_bd, filtered_bd, filter_info_gg.get('BD', {}), f"{run_name} - B:D", data_type='gg')
        fig_gg_raw = plot_pore_diameter(time_points, dist_ac, dist_bd, run_name, filtered=False)
        fig_gg_filtered = plot_pore_diameter(time_points, filtered_ac, filtered_bd, run_name, filtered=True)

        fig_gg_ac.savefig(os.path.join(run_dir, "G_G_Distance_AC_Comparison.png"), bbox_inches='tight')
        fig_gg_bd.savefig(os.path.join(run_dir, "G_G_Distance_BD_Comparison.png"), bbox_inches='tight')
        fig_gg_raw.savefig(os.path.join(run_dir, "GG_Distance_Plot_raw.png"), bbox_inches='tight')
        fig_gg_filtered.savefig(os.path.join(run_dir, "GG_Distance_Plot.png"), bbox_inches='tight')

        plt.close(fig_gg_ac); plt.close(fig_gg_bd); plt.close(fig_gg_raw); plt.close(fig_gg_filtered)
        logger.debug("G-G plots saved.")

        # COM Plots (only for non-control systems with COM data)
        if not is_control_system and com_distances is not None:
            if filtered_com is not None:
                fig_com = plot_filtering_comparison(time_points, com_distances, filtered_com, filter_info_com, run_name, data_type='com')
                fig_com.savefig(os.path.join(run_dir, "COM_Stability_Comparison.png"), bbox_inches='tight')
                plt.close(fig_com)

                fig_com_filtered = plot_com_positions(time_points, filtered_com, run_name, filtered=True)
                fig_com_filtered.savefig(os.path.join(run_dir, "COM_Stability_Plot.png"), bbox_inches='tight')
                plt.close(fig_com_filtered)
            else:
                 logger.warning("Filtered COM data is None, skipping comparison and filtered plots.")

            fig_com_raw = plot_com_positions(time_points, com_distances, run_name, filtered=False)
            fig_com_raw.savefig(os.path.join(run_dir, "COM_Stability_Plot_raw.png"), bbox_inches='tight')
            plt.close(fig_com_raw)

            # Generate KDE plot if raw data exists
            fig_com_kde = plot_kde_analysis(time_points, com_distances, run_name, data_type='com')
            fig_com_kde.savefig(os.path.join(run_dir, "COM_Stability_KDE_Analysis.png"), bbox_inches='tight')
            plt.close(fig_com_kde)
            logger.debug("COM plots saved.")
        elif is_control_system:
            logger.info("Skipping COM plot generation (control system without toxin).")
        else: # Case where com_distances was None
            logger.info("Skipping COM plot generation (no raw COM data).")

    except Exception as e:
        logger.error(f"Error generating plots: {e}", exc_info=True)

    # Add is_control_system to the raw_stats dictionary for downstream use
    raw_stats['is_control_system'] = is_control_system

    logger.info(f"Finished filtering, saving, and plotting for {run_dir}.")
    return filtered_ac, filtered_bd, filtered_com, filter_info_gg, filter_info_com, raw_stats, percentile_stats


# --- Plotting Helper Functions ---

def plot_pore_diameter(time_points, dist_ac, dist_bd, title, filtered=False):
    """Plot G–G distance with A:C and B:D distances overlaid."""
    fig, ax = plt.subplots(figsize=(10, 6))
    if dist_ac is not None:
        sns.lineplot(x=time_points, y=dist_ac, label='A:C Distance', ax=ax, legend=False)
    if dist_bd is not None:
        sns.lineplot(x=time_points, y=dist_bd, label='B:D Distance', ax=ax, legend=False)

    ax.set_xlabel('Time (ns)', fontsize=14)
    ax.set_ylabel('G–G Distance (Å)', fontsize=14)
    plot_type = "(Filtered)" if filtered else "(Raw)"
    ax.set_title(f"Pore G-G Distance: {title} {plot_type}", fontsize=16)
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(fontsize='medium')

    all_dists = np.concatenate([d for d in [dist_ac, dist_bd] if d is not None])
    finite_dists = all_dists[np.isfinite(all_dists)]
    if len(finite_dists) > 0:
        y_min = np.nanmin(finite_dists) - 1.0
        y_max = np.nanmax(finite_dists) + 1.0
        ax.set_ylim(max(0, y_min), y_max)
    else:
        ax.set_ylim(0, 15)

    return fig

def plot_com_positions(time_points, com_dist, title, filtered=False):
    """Plot COM distance."""
    fig, ax = plt.subplots(figsize=(10, 6))
    if com_dist is not None:
        sns.lineplot(x=time_points, y=com_dist, ax=ax)
    ax.set_xlabel('Time (ns)', fontsize=14)
    ax.set_ylabel('Toxin-Channel COM Distance (Å)', fontsize=14)
    plot_type = "(Filtered)" if filtered else "(Raw)"
    ax.set_title(f"Toxin-Channel Stability: {title} {plot_type}", fontsize=16)
    ax.tick_params(axis='both', labelsize=12)

    if com_dist is not None:
        finite_dists = com_dist[np.isfinite(com_dist)]
        if len(finite_dists) > 0:
            y_min = np.nanmin(finite_dists) - 2.0
            y_max = np.nanmax(finite_dists) + 2.0
            ax.set_ylim(max(0, y_min), y_max)
        else:
            ax.set_ylim(0, 50)
    else:
        ax.set_ylim(0, 50)

    return fig

def plot_filtering_comparison(time_points, raw_data, filtered_data, filter_info, title, data_type='com'):
    """Plot a comparison of raw and filtered data."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    ylabel = 'Distance (Å)'
    if data_type == 'com':
        ylabel = 'COM Distance (Å)'
    elif data_type == 'gg':
        ylabel = 'G-G Distance (Å)'

    ax1.plot(time_points, raw_data, color='steelblue', linewidth=1.0, label='Raw Data')
    ax1.set_ylabel(ylabel, fontsize=14)
    ax1.set_title(f'{title} - Raw Data', fontsize=16)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    jumps = filter_info.get('pbc_jump_locations', [])
    if not jumps:
        jumps = filter_info.get('unwrap_pbc_jump_locations', [])
    if jumps:
        jump_times = [time_points[idx] for idx, jump in jumps if idx < len(time_points)]
        if jump_times:
            for t in jump_times:
                ax1.axvline(x=t, color='red', linestyle='--', alpha=0.5, linewidth=0.8)
            ax1.plot([], [], color='red', linestyle='--', alpha=0.5, linewidth=0.8, label='PBC Jump Detected')
            ax1.legend()

    ax2.plot(time_points, filtered_data, color='green', linewidth=1.5, label='Filtered Data')
    ax2.set_xlabel('Time (ns)', fontsize=14)
    ax2.set_ylabel(ylabel, fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    filter_method = filter_info.get('method_applied', 'Unknown')
    details = f"Method: {filter_method.capitalize()}"
    if filter_method == 'multi_level':
        if filter_info.get('multi_level_applied', False):
            details += f" | Levels: {filter_info.get('kde_detected_level_count', '?')}"
            details += f" | Transitions: {filter_info.get('transition_count', '?')}"
            details += f" | Std Reduc: {filter_info.get('final_std_reduction_percent', np.nan):.1f}%"
        else:
            details = "Method: Multi-Level (Fallback to Standard)"
            details += f" | Reason: {filter_info.get('fallback_reason', 'Unknown')}"
            std_reduc_fb = filter_info.get('standard_filter_fallback', {}).get('overall_effect_std_reduction_percent', np.nan)
            details += f" | Std Reduc (Standard): {std_reduc_fb:.1f}%"
    elif filter_method == 'standard':
        jumps = filter_info.get('unwrap_pbc_jump_count', '?')
        spikes = filter_info.get('unwrap_spike_count', '?')
        std_reduc = filter_info.get('overall_effect_std_reduction_percent', np.nan)
        details += f" | PBC Jumps: {jumps} | Spikes: {spikes} | Std Reduc: {std_reduc:.1f}%"

    ax2.set_title(f"Filtered Data\n({details})", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    return fig

def plot_kde_analysis(time_points, data, title, data_type='com'):
    """Plot kernel density estimation analysis of raw data."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [1, 2]})

    ylabel = 'Distance (Å)'
    xlabel = 'Distance (Å)'
    if data_type == 'com':
        xlabel = 'COM Distance (Å)'
        ylabel = xlabel
    elif data_type == 'gg':
        xlabel = 'G-G Distance (Å)'
        ylabel = xlabel

    finite_data = data[np.isfinite(data)]
    peak_values = []  # Initialize peak_values here

    if len(finite_data) < 5:
        ax1.text(0.5, 0.5, 'Insufficient finite data for KDE/Histogram', ha='center', va='center')
        ax2.plot(time_points, data, color='steelblue', linewidth=1)
        ax2.set_title('Raw Time Series Data', fontsize=16)
        ax2.set_xlabel('Time (ns)', fontsize=14)
        ax2.set_ylabel(ylabel, fontsize=14)
        plt.tight_layout()
        return fig

    # KDE Plot (Left)
    try:
        kde = gaussian_kde(finite_data)
        x_grid = np.linspace(np.min(finite_data), np.max(finite_data), 200)
        density = kde(x_grid)
        peaks, _ = find_peaks(density, height=np.max(density)*0.05, distance=max(5, len(x_grid)//50))
        peak_values = x_grid[peaks]  # Assign peak_values inside the try block

        ax1.hist(finite_data, bins=30, density=True, alpha=0.5, color='skyblue', orientation='horizontal')
        ax1.plot(density, x_grid, 'r-', linewidth=2)

        for peak in peak_values:
            ax1.axhline(y=peak, color='green', linestyle='--', alpha=0.7, linewidth=1.0)

        ax1.set_ylabel(xlabel, fontsize=14)
        ax1.set_xlabel('Density', fontsize=14)
        ax1.set_title(f'Value Distribution & KDE\n(Detected {len(peaks)} peaks)', fontsize=14)
        ax1.tick_params(axis='both', labelsize=10)
    except Exception as e:
        ax1.text(0.5, 0.5, f'KDE Failed:\n{e}', ha='center', va='center')
        logger.error(f"KDE plot generation failed: {e}", exc_info=True)

    # Time Series Plot with Level Coloring (Right)
    if len(peak_values) > 1:
        assignments = np.zeros_like(data, dtype=int) - 1
        finite_indices = np.where(np.isfinite(data))[0]
        if len(finite_indices) > 0:
            finite_vals = data[finite_indices]
            distances_to_peaks = np.abs(finite_vals[:, np.newaxis] - peak_values)
            assignments[finite_indices] = np.argmin(distances_to_peaks, axis=1)

        n_levels = len(peak_values)
        cmap = plt.cm.get_cmap('viridis', n_levels)
        colors = [cmap(i) for i in range(n_levels)]

        for i in range(n_levels):
            mask = assignments == i
            if np.any(mask):
                ax2.scatter(time_points[mask], data[mask], color=colors[i], s=5, alpha=0.6)
        ax2.set_title('Time Series with Potential Level Assignments', fontsize=16)
    else:
        ax2.plot(time_points, data, color='steelblue', linewidth=1)

    ax2.set_xlabel('Time (ns)', fontsize=14)
    ax2.set_ylabel(ylabel, fontsize=14)
    ax2.tick_params(axis='both', labelsize=10)
    ax2.grid(True, alpha=0.3)

    if len(peak_values) > 0:
        ax1.set_ylim(ax2.get_ylim())

    fig.suptitle(f'{title} - Raw Data Analysis', fontsize=18, y=1.02)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    return fig
