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
            Returns (np.array([0]), np.array([0]), None, np.array([0]), 'Unknown') on critical error.
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
        return np.array([0]), np.array([0]), None, np.array([0]), 'Unknown'
    if not os.path.exists(dcd_file):
        logger.error(f"DCD file not found: {dcd_file}")
        return np.array([0]), np.array([0]), None, np.array([0]), 'Unknown'

    # --- Load Universe ---
    try:
        logger.info(f"Loading topology: {psf_file}")
        logger.info(f"Loading trajectory: {dcd_file}")
        u = mda.Universe(psf_file, dcd_file)
        n_frames = len(u.trajectory)
        logger.info(f"Successfully loaded universe with {n_frames} frames")
        if n_frames == 0:
            logger.warning("Trajectory contains 0 frames.")
            return np.array([]), np.array([]), None, np.array([]), system_dir

    except Exception as e:
        logger.error(f"Failed to load MDAnalysis Universe: {e}", exc_info=True)
        return np.array([0]), np.array([0]), None, np.array([0]), 'Unknown'

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
        return np.array([0]), np.array([0]), None, np.array([0]), 'Unknown'

    # COM Selection (Toxin vs Channel)
    toxin_segids = ['PROE', 'E', 'PEP', 'TOX']
    channel_segids_pro = ['PROA', 'PROB', 'PROC', 'PROD']
    channel_segids_alt = ['A', 'B', 'C', 'D']

    pep_sel = None
    for seg in toxin_segids:
        if len(u.select_atoms(f'segid {seg}')) > 0:
            pep_sel = f'segid {seg}'
            logger.info(f"Using Toxin selection: '{pep_sel}'")
            break

    chan_sel = 'segid ' + ' or segid '.join(channel_segids_pro)
    if len(u.select_atoms(chan_sel)) == 0:
        chan_sel = 'segid ' + ' or segid '.join(channel_segids_alt)
        logger.info(f"Using Alternate Channel selection: '{chan_sel}'")
    else:
        logger.info(f"Using Channel selection: '{chan_sel}'")

    if len(u.select_atoms(chan_sel)) == 0:
        logger.error(f"Could not find channel atoms using standard selections.")
        return np.array([0]), np.array([0]), None, np.array([0]), 'Unknown'

    # Determine if COM analysis is possible (if toxin was found)
    com_analyzed = pep_sel is not None
    if not com_analyzed:
        logger.info("No toxin selection found. COM distance analysis will be skipped.")

    # --- Trajectory Iteration ---
    logger.info(f"Starting trajectory analysis loop ({n_frames} frames)...")
    try:
        for ts in u.trajectory:
            frame_indices.append(ts.frame)

            # G-G Calculation
            try:
                ca_atoms = {}
                found_all_gg = True
                for pro_seg, alt_seg in zip(channel_segids_pro, channel_segids_alt):
                    sel_str = f'resid {reference_resid} and name CA and (segid {pro_seg} or segid {alt_seg})'
                    atoms = u.select_atoms(sel_str)
                    if len(atoms) == 1:
                        ca_atoms[alt_seg] = atoms
                    elif len(atoms) == 0:
                        logger.warning(f"Missing CA atom for G-G distance at frame {ts.frame}: {sel_str}")
                        ca_atoms[alt_seg] = None
                        found_all_gg = False
                    else:
                        logger.warning(f"Multiple CA atoms found for G-G distance at frame {ts.frame}: {sel_str}")
                        ca_atoms[alt_seg] = atoms[0]

                if found_all_gg and ca_atoms['A'] and ca_atoms['C'] and ca_atoms['B'] and ca_atoms['D']:
                    dist_ac.append(float(distances.dist(ca_atoms['A'], ca_atoms['C'])[2]))
                    dist_bd.append(float(distances.dist(ca_atoms['B'], ca_atoms['D'])[2]))
                else:
                    dist_ac.append(np.nan)
                    dist_bd.append(np.nan)

            except Exception as e_gg:
                logger.warning(f"Error calculating G-G distances at frame {ts.frame}: {e_gg}", exc_info=False)
                dist_ac.append(np.nan)
                dist_bd.append(np.nan)

            # COM Calculation (only if toxin exists)
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
                        pep_positions.append([np.nan, np.nan, np.nan])
                        ch_positions.append([np.nan, np.nan, np.nan])
                except Exception as e_com:
                    logger.warning(f"Error calculating COM at frame {ts.frame}: {e_com}", exc_info=False)
                    pep_positions.append([np.nan, np.nan, np.nan])
                    ch_positions.append([np.nan, np.nan, np.nan])

    except Exception as loop_err:
        logger.error(f"Error during trajectory loop: {loop_err}", exc_info=True)
        return np.array([0]), np.array([0]), None, np.array([0]), 'Unknown'

    logger.info(f"Completed trajectory analysis loop. Processed {len(frame_indices)} frames.")

    # --- Process Results ---
    time_points = frames_to_time(frame_indices)
    dist_ac = np.array(dist_ac)
    dist_bd = np.array(dist_bd)

    com_distances_raw = None
    if com_analyzed:
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

    return dist_ac, dist_bd, com_distances_raw, time_points, system_dir


# --- Filtering Application and Plotting ---

def filter_and_save_data(run_dir, dist_ac, dist_bd, com_distances, time_points, box_z=None):
    """
    Apply filtering to the raw distance data (G-G and COM) and save filtered data and plots.

    Args:
        run_dir (str): Path to the specific run directory for output.
        dist_ac (np.ndarray): Raw A-C G-G distances.
        dist_bd (np.ndarray): Raw B-D G-G distances.
        com_distances (np.ndarray | None): Raw Toxin-Channel COM distances.
        time_points (np.ndarray): Time points corresponding to frames (in ns).
        box_z (float, optional): Estimated or known box Z-dimension, passed to
                                 multi-level filter if used for COM distance.

    Returns:
        tuple: Contains:
            - filtered_ac (np.ndarray): Filtered A-C G-G distances.
            - filtered_bd (np.ndarray): Filtered B-D G-G distances.
            - filtered_com (np.ndarray | None): Filtered COM distances, or None.
            - filter_info_gg (dict): Filtering info for G-G distances ({'AC': info, 'BD': info}).
            - filter_info_com (dict): Filtering info for COM distances.
            Returns input data and empty info dicts on error or insufficient data.
    """
    logger_name = f"{os.path.basename(os.path.dirname(run_dir))}_{os.path.basename(run_dir)}"
    logger = logging.getLogger(logger_name)
    run_name = os.path.basename(run_dir)
    logger.info(f"Applying filtering to G-G and COM data for {run_name}")

    filter_info_gg = {'AC': {}, 'BD': {}}
    filter_info_com = {}
    filtered_ac = np.array(dist_ac)
    filtered_bd = np.array(dist_bd)
    filtered_com = np.array(com_distances) if com_distances is not None else None

    # --- Filter G-G Data ---
    if dist_ac is not None and len(dist_ac) > 1:
        try:
            filtered_ac, filter_info_ac = auto_select_filter(dist_ac, data_type='gg_distance')
            filter_info_gg['AC'] = filter_info_ac
        except Exception as e:
            logger.error(f"Error filtering G-G A:C data: {e}", exc_info=True)
            filter_info_gg['AC'] = {'error': str(e)}
    if dist_bd is not None and len(dist_bd) > 1:
        try:
            filtered_bd, filter_info_bd = auto_select_filter(dist_bd, data_type='gg_distance')
            filter_info_gg['BD'] = filter_info_bd
        except Exception as e:
            logger.error(f"Error filtering G-G B:D data: {e}", exc_info=True)
            filter_info_gg['BD'] = {'error': str(e)}

    gg_filtered_df = pd.DataFrame({
        'Time (ns)': time_points,
        'A_C_Distance_Raw': dist_ac if dist_ac is not None else np.nan,
        'B_D_Distance_Raw': dist_bd if dist_bd is not None else np.nan,
        'A_C_Distance_Filtered': filtered_ac if filtered_ac is not None else np.nan,
        'B_D_Distance_Filtered': filtered_bd if filtered_bd is not None else np.nan
    })
    gg_filtered_path = os.path.join(run_dir, "G_G_Distance_Filtered.csv")
    try:
        gg_filtered_df.to_csv(gg_filtered_path, index=False, float_format='%.4f', na_rep='NaN')
        logger.info(f"Saved filtered G-G data to {gg_filtered_path}")
    except Exception as e:
        logger.error(f"Failed to save filtered G-G CSV: {e}")

    # --- Filter COM Data ---
    if com_distances is not None and len(com_distances) > 1:
        try:
            kwargs = {'box_size': box_z} if box_z is not None else {}
            filtered_com, filter_info_com = auto_select_filter(
                com_distances,
                data_type='com_distance',
                **kwargs
            )
        except Exception as e:
            logger.error(f"Error filtering COM data: {e}", exc_info=True)
            filter_info_com = {'error': str(e)}
            filtered_com = np.array(com_distances)
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
    elif com_distances is not None:
        logger.info("COM distance analysis skipped or data insufficient for filtering.")
        filtered_com = None
        filter_info_com = {'reason': 'Insufficient data or skipped'}
    else:
        filtered_com = None
        filter_info_com = {'reason': 'No COM data provided'}

    # --- Generate Plots ---
    try:
        logger.info("Generating comparison and final plots...")
        fig_gg_ac = plot_filtering_comparison(time_points, dist_ac, filtered_ac, filter_info_gg.get('AC', {}), f"{run_name} - A:C", data_type='gg')
        fig_gg_bd = plot_filtering_comparison(time_points, dist_bd, filtered_bd, filter_info_gg.get('BD', {}), f"{run_name} - B:D", data_type='gg')
        fig_gg_raw = plot_pore_diameter(time_points, dist_ac, dist_bd, run_name, filtered=False)
        fig_gg_filtered = plot_pore_diameter(time_points, filtered_ac, filtered_bd, run_name, filtered=True)

        fig_gg_ac.savefig(os.path.join(run_dir, "G_G_Distance_AC_Comparison.png"), bbox_inches='tight')
        fig_gg_bd.savefig(os.path.join(run_dir, "G_G_Distance_BD_Comparison.png"), bbox_inches='tight')
        fig_gg_raw.savefig(os.path.join(run_dir, "GG_Distance_Plot_raw.png"), bbox_inches='tight')
        fig_gg_filtered.savefig(os.path.join(run_dir, "GG_Distance_Plot.png"), bbox_inches='tight')

        plt.close(fig_gg_ac)
        plt.close(fig_gg_bd)
        plt.close(fig_gg_raw)
        plt.close(fig_gg_filtered)
        logger.debug("G-G plots saved.")

        if com_distances is not None and filtered_com is not None:
            fig_com = plot_filtering_comparison(time_points, com_distances, filtered_com, filter_info_com, run_name, data_type='com')
            fig_com_raw = plot_com_positions(time_points, com_distances, run_name, filtered=False)
            fig_com_filtered = plot_com_positions(time_points, filtered_com, run_name, filtered=True)
            fig_com_kde = plot_kde_analysis(time_points, com_distances, run_name, data_type='com')

            fig_com.savefig(os.path.join(run_dir, "COM_Stability_Comparison.png"), bbox_inches='tight')
            fig_com_raw.savefig(os.path.join(run_dir, "COM_Stability_Plot_raw.png"), bbox_inches='tight')
            fig_com_filtered.savefig(os.path.join(run_dir, "COM_Stability_Plot.png"), bbox_inches='tight')
            fig_com_kde.savefig(os.path.join(run_dir, "COM_Stability_KDE_Analysis.png"), bbox_inches='tight')

            plt.close(fig_com)
            plt.close(fig_com_raw)
            plt.close(fig_com_filtered)
            plt.close(fig_com_kde)
            logger.debug("COM plots saved.")
        else:
            logger.info("Skipping COM plot generation.")

    except Exception as e:
        logger.error(f"Error generating plots: {e}", exc_info=True)

    return filtered_ac, filtered_bd, filtered_com, filter_info_gg, filter_info_com


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
                ax2.scatter(time_points[mask], data[mask], color=colors[i], s=5, alpha=0.6, label=f"Level {i+1} (~{peak_values[i]:.2f}Å)")
        ax2.legend(fontsize=9, loc='upper right')
        ax2.set_title('Time Series with Potential Level Assignments', fontsize=16)
    else:
        ax2.plot(time_points, data, color='steelblue', linewidth=1)
        ax2.set_title('Raw Time Series Data', fontsize=16)

    ax2.set_xlabel('Time (ns)', fontsize=14)
    ax2.set_ylabel(ylabel, fontsize=14)
    ax2.tick_params(axis='both', labelsize=10)
    ax2.grid(True, alpha=0.3)

    if len(peak_values) > 0:
        ax1.set_ylim(ax2.get_ylim())

    fig.suptitle(f'{title} - Raw Data Analysis', fontsize=18, y=1.02)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    return fig
