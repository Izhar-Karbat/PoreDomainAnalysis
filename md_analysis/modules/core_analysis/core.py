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
from scipy.signal import find_peaks

# Import from other modules
try:
    from md_analysis.core.utils import OneLetter, frames_to_time
    from md_analysis.core.filtering import auto_select_filter
    from md_analysis.core.logging import setup_system_logger
except ImportError as e:
    print(f"Error importing dependency modules in core_analysis.py: {e}")
    raise

# Configure matplotlib for non-interactive backend
plt.switch_backend('agg')
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.2)
logger = logging.getLogger(__name__)

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
    """
    # Set up system-specific logger
    logger = setup_system_logger(run_dir)
    if logger is None:
        logger = logging.getLogger()
        logger.error(f"Failed to setup system logger for {run_dir}. Using root logger.")

    run_name = os.path.basename(run_dir)
    system_dir = os.path.basename(os.path.dirname(run_dir)) if os.path.dirname(run_dir) else run_name
    logger.info(f"Starting raw trajectory analysis for system: {system_dir} (Folder: {run_dir})")

    # --- Define Output Directory for Core Analysis ---
    output_dir = os.path.join(run_dir, "core_analysis")
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Core analysis outputs will be saved to: {output_dir}")

    # --- File Handling ---
    if psf_file is None:
        psf_file = os.path.join(run_dir, "step5_input.psf")
    if dcd_file is None:
        dcd_file = os.path.join(run_dir, "MD_Aligned.dcd")

    if not os.path.exists(psf_file):
        logger.error(f"PSF file not found: {psf_file}")
        return np.array([0]), np.array([0]), None, np.array([0]), 'Unknown', True
    if not os.path.exists(dcd_file):
        logger.error(f"DCD file not found: {dcd_file}")
        return np.array([0]), np.array([0]), None, np.array([0]), 'Unknown', True

    # --- Load Universe ---
    try:
        logger.info(f"Loading topology: {psf_file}")
        logger.info(f"Loading trajectory: {dcd_file}")
        u = mda.Universe(psf_file, dcd_file)
        n_frames = len(u.trajectory)
        logger.info(f"Successfully loaded universe with {n_frames} frames")
        if n_frames == 0:
            logger.warning("Trajectory contains 0 frames.")
            return np.array([]), np.array([]), None, np.array([]), system_dir, True
    except Exception as e:
        logger.error(f"Failed to load MDAnalysis Universe: {e}", exc_info=True)
        return np.array([]), np.array([]), None, np.array([]), 'Unknown', True

    # --- Setup Selections ---
    dist_ac = []
    dist_bd = []
    pep_positions = []
    ch_positions = []
    frame_indices = []
    idx_gyg = None

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
            idx_gyg = gly_indices[len(gly_indices) // 2]
        else:
            idx_gyg = idx_gyg + 1

        # Convert sequence index to resid
        reference_resid = chain_A_atoms.residues[idx_gyg].resid
        logger.info(f"Using CA of residue {reference_resid} (Index {idx_gyg} in sequence '{chainA_Seq}') for G-G distance.")

    except Exception as e:
        logger.error(f"Error finding reference Glycine for G-G distance: {e}", exc_info=True)
        return np.array([]), np.array([]), None, np.array([]), 'Unknown', True

    # COM Selection (Toxin vs Channel)
    toxin_segids = ['PROE', 'E', 'PEP', 'TOX']
    channel_segids_pro = ['PROA', 'PROB', 'PROC', 'PROD']
    channel_segids_alt = ['A', 'B', 'C', 'D']

    pep_sel = None
    for seg in toxin_segids:
        if len(u.select_atoms(f'segid {seg}')) > 0:
            pep_sel = f'segid {seg}'
            logger.info(f"Found Toxin selection: '{pep_sel}'")
            break

    chan_sel = 'segid ' + ' or segid '.join(channel_segids_pro)
    if len(u.select_atoms(chan_sel)) == 0:
        chan_sel_alt = 'segid ' + ' or segid '.join(channel_segids_alt)
        if len(u.select_atoms(chan_sel_alt)) > 0:
             chan_sel = chan_sel_alt
             logger.info(f"Using Alternate Channel selection: '{chan_sel}'")
        else:
             logger.warning(f"Could not find channel atoms using standard selections: '{chan_sel}' or '{chan_sel_alt}'.")
             return np.array([]), np.array([]), None, np.array([]), 'Unknown', True

    else:
        logger.info(f"Using Channel selection: '{chan_sel}'")

    # Determine if COM analysis is possible (if toxin was found)
    com_analyzed = pep_sel is not None
    is_control_system = not com_analyzed

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
                opposing_chains = [('A', 'C'), ('B', 'D')]
                current_channel_segids = [s.replace('segid ', '') for s in chan_sel.split(' or ')]

                chain_map = {}
                for seg in current_channel_segids:
                    if seg.startswith("PRO"): chain_map[seg[-1]] = seg
                    else: chain_map[seg] = seg

                for chain1, chain2 in opposing_chains:
                    if chain1 in chain_map and chain2 in chain_map:
                        seg1 = chain_map[chain1]
                        seg2 = chain_map[chain2]
                        ca1 = u.select_atoms(f'segid {seg1} and resid {reference_resid} and name CA')
                        ca2 = u.select_atoms(f'segid {seg2} and resid {reference_resid} and name CA')
                        
                        if len(ca1) == 1 and len(ca2) == 1:
                            ca_atoms[chain1] = ca1.positions[0]
                            ca_atoms[chain2] = ca2.positions[0]
                        else:
                            found_all_gg = False
                            break
                    else:
                        found_all_gg = False
                        break

                if found_all_gg:
                    dist_ac.append(np.linalg.norm(ca_atoms['A'] - ca_atoms['C']))
                    dist_bd.append(np.linalg.norm(ca_atoms['B'] - ca_atoms['D']))
                else:
                    dist_ac.append(np.nan)
                    dist_bd.append(np.nan)

            except Exception as e:
                logger.error(f"Error calculating G-G distances for frame {ts.frame}: {e}")
                dist_ac.append(np.nan)
                dist_bd.append(np.nan)

            # COM Calculation (if toxin present)
            if com_analyzed:
                try:
                    pep_atoms = u.select_atoms(pep_sel)
                    chan_atoms = u.select_atoms(chan_sel)
                    
                    if len(pep_atoms) > 0 and len(chan_atoms) > 0:
                        pep_pos = pep_atoms.center_of_mass()
                        chan_pos = chan_atoms.center_of_mass()
                        pep_positions.append(pep_pos)
                        ch_positions.append(chan_pos)
                    else:
                        pep_positions.append(np.array([np.nan, np.nan, np.nan]))
                        ch_positions.append(np.array([np.nan, np.nan, np.nan]))
                except Exception as e:
                    logger.error(f"Error calculating COM positions for frame {ts.frame}: {e}")
                    pep_positions.append(np.array([np.nan, np.nan, np.nan]))
                    ch_positions.append(np.array([np.nan, np.nan, np.nan]))

    except Exception as e:
        logger.error(f"Error during trajectory iteration: {e}", exc_info=True)
        return np.array([]), np.array([]), None, np.array([]), 'Unknown', True

    # Convert lists to numpy arrays
    dist_ac = np.array(dist_ac)
    dist_bd = np.array(dist_bd)
    frame_indices = np.array(frame_indices)
    time_points = frames_to_time(frame_indices)

    # Calculate COM distances if toxin was found
    com_distances = None
    if com_analyzed and len(pep_positions) > 0 and len(ch_positions) > 0:
        pep_positions = np.array(pep_positions)
        ch_positions = np.array(ch_positions)
        com_distances = np.linalg.norm(pep_positions - ch_positions, axis=1)

    # Save raw data to CSV
    try:
        df_raw = pd.DataFrame({
            'Frame': frame_indices,
            'Time (ns)': time_points,
            'GG_Distance_AC': dist_ac,
            'GG_Distance_BD': dist_bd
        })
        if com_distances is not None:
            df_raw['COM_Distance'] = com_distances

        raw_csv_path = os.path.join(output_dir, "Raw_Distances.csv")
        df_raw.to_csv(raw_csv_path, index=False, float_format='%.4f')
        logger.info(f"Saved raw distance data to {raw_csv_path}")
    except Exception as e:
        logger.error(f"Failed to save raw distance data: {e}", exc_info=True)

    return dist_ac, dist_bd, com_distances, time_points, system_dir, is_control_system

def filter_and_save_data(run_dir, dist_ac, dist_bd, com_distances, time_points, box_z=None, is_control_system=False):
    """
    Apply filtering to raw distance data and save filtered results.
    
    Args:
        run_dir (str): Directory path for the run data.
        dist_ac (np.ndarray): Raw A:C distance data.
        dist_bd (np.ndarray): Raw B:D distance data.
        com_distances (np.ndarray, optional): Raw COM distance data. Can be None.
        time_points (np.ndarray): Time points corresponding to the distance data.
        box_z (float, optional): Z dimension of the simulation box.
        is_control_system (bool): Whether this is a control system without toxin.
        
    Returns:
        tuple: (filtered_ac, filtered_bd, filtered_com, filter_info_g_g, filter_info_com,
                raw_dist_stats, percentile_stats)
    """
    # Set up logging
    logger = setup_system_logger(run_dir)
    if logger is None:
        logger = logging.getLogger()

    # Create output directory
    output_dir = os.path.join(run_dir, "core_analysis")
    os.makedirs(output_dir, exist_ok=True)

    # Initialize return values
    filtered_ac = np.array([])
    filtered_bd = np.array([])
    filtered_com = np.array([])
    filter_info_g_g = {}
    filter_info_com = {}
    raw_dist_stats = {}
    percentile_stats = {}

    # G-G Filtering
    if len(dist_ac) > 0 and len(dist_bd) > 0:
        try:
            # Apply filtering to G-G distances
            filtered_ac, filter_info_g_g = auto_select_filter(dist_ac, data_type='gg_distance')
            filtered_bd, _ = auto_select_filter(dist_bd, data_type='gg_distance')
            
            # Calculate statistics
            raw_dist_stats['G_G_AC_Mean'] = np.nanmean(dist_ac)
            raw_dist_stats['G_G_AC_Std'] = np.nanstd(dist_ac)
            raw_dist_stats['G_G_BD_Mean'] = np.nanmean(dist_bd)
            raw_dist_stats['G_G_BD_Std'] = np.nanstd(dist_bd)
            
            percentile_stats['G_G_AC_Pctl10_Raw'] = np.percentile(dist_ac[np.isfinite(dist_ac)], 10) if np.any(np.isfinite(dist_ac)) else np.nan
            percentile_stats['G_G_AC_Pctl90_Raw'] = np.percentile(dist_ac[np.isfinite(dist_ac)], 90) if np.any(np.isfinite(dist_ac)) else np.nan
            percentile_stats['G_G_BD_Pctl10_Raw'] = np.percentile(dist_bd[np.isfinite(dist_bd)], 10) if np.any(np.isfinite(dist_bd)) else np.nan
            percentile_stats['G_G_BD_Pctl90_Raw'] = np.percentile(dist_bd[np.isfinite(dist_bd)], 90) if np.any(np.isfinite(dist_bd)) else np.nan
            percentile_stats['G_G_AC_Pctl10_Filt'] = np.percentile(filtered_ac[np.isfinite(filtered_ac)], 10) if np.any(np.isfinite(filtered_ac)) else np.nan
            percentile_stats['G_G_AC_Pctl90_Filt'] = np.percentile(filtered_ac[np.isfinite(filtered_ac)], 90) if np.any(np.isfinite(filtered_ac)) else np.nan
            percentile_stats['G_G_BD_Pctl10_Filt'] = np.percentile(filtered_bd[np.isfinite(filtered_bd)], 10) if np.any(np.isfinite(filtered_bd)) else np.nan
            percentile_stats['G_G_BD_Pctl90_Filt'] = np.percentile(filtered_bd[np.isfinite(filtered_bd)], 90) if np.any(np.isfinite(filtered_bd)) else np.nan

            # Save filtered G-G data
            df_g_g = pd.DataFrame({
                'Time (ns)': time_points,
                'G_G_Distance_AC_Raw': dist_ac,
                'G_G_Distance_BD_Raw': dist_bd,
                'G_G_Distance_AC_Filt': filtered_ac,
                'G_G_Distance_BD_Filt': filtered_bd
            })
            gg_csv_path = os.path.join(output_dir, "G_G_Distance_Filtered.csv")
            df_g_g.to_csv(gg_csv_path, index=False, float_format='%.4f')
            logger.info(f"Saved filtered G-G distances to {gg_csv_path}")
            
            # Create G-G distance plots
            plot_distances(time_points, dist_ac, dist_bd, filtered_ac, filtered_bd, 
                          output_dir, "G-G Distances", is_gg=True, logger=logger)
            
        except Exception as e:
            logger.error(f"Error in G-G filtering: {e}", exc_info=True)

    # COM Filtering (if not control system)
    if not is_control_system and com_distances is not None and len(com_distances) > 0:
        try:
            # Apply filtering to COM distances
            filtered_com, filter_info_com = auto_select_filter(com_distances, data_type='com_distance', box_z=box_z)
            
            # Calculate statistics
            raw_dist_stats['COM_Mean'] = np.nanmean(com_distances)
            raw_dist_stats['COM_Std'] = np.nanstd(com_distances)
            percentile_stats['COM_Percentiles'] = np.nanpercentile(com_distances, [25, 50, 75])
            
            # Save filtered COM data
            df_com = pd.DataFrame({
                'Time (ns)': time_points,
                'COM_Distance_Raw': com_distances,
                'COM_Distance_Filt': filtered_com
            })
            com_csv_path = os.path.join(output_dir, "COM_Stability_Filtered.csv")
            df_com.to_csv(com_csv_path, index=False, float_format='%.4f')
            logger.info(f"Saved filtered COM distances to {com_csv_path}")
            
            # Create COM distance plots
            plot_distances(time_points, com_distances, None, filtered_com, None, 
                          output_dir, "COM Distances", is_gg=False, logger=logger)
            
            # Create KDE analysis plot for COM distances (pass the parameters in correct order)
            plot_kde_analysis(time_points, com_distances, box_z, output_dir)
            
        except Exception as e:
            logger.error(f"Error in COM filtering: {e}", exc_info=True)

    return (filtered_ac, filtered_bd, filtered_com, 
            filter_info_g_g, filter_info_com,
            raw_dist_stats, percentile_stats)

def plot_distances(time_points, raw_data1, raw_data2=None, filtered_data1=None, filtered_data2=None, 
                  output_dir=None, title_prefix="Distances", is_gg=True, logger=None):
    """
    Create plots for raw and filtered distance data.
    
    Args:
        time_points (np.ndarray): Time points in ns.
        raw_data1 (np.ndarray): Raw distance data for first measurement.
        raw_data2 (np.ndarray, optional): Raw distance data for second measurement.
        filtered_data1 (np.ndarray, optional): Filtered distance data for first measurement.
        filtered_data2 (np.ndarray, optional): Filtered distance data for second measurement.
        output_dir (str, optional): Directory to save plots.
        title_prefix (str): Prefix for plot titles.
        is_gg (bool): Whether the data is G-G distances (True) or COM distances (False).
        logger (logging.Logger, optional): Logger instance.
        
    Returns:
        bool: True if successful, False otherwise.
    """
    if logger is None:
        logger = logging.getLogger()
    
    if output_dir is None:
        logger.warning("No output directory specified for plot_distances.")
        return False
    
    try:
        # 1. Raw Data Plot
        plt.figure(figsize=(10, 6))
        if is_gg:
            # For G-G distances, plot both A:C and B:D
            plt.plot(time_points, raw_data1, color='blue', alpha=0.7, label='A:C Distance')
            if raw_data2 is not None:
                plt.plot(time_points, raw_data2, color='red', alpha=0.7, label='B:D Distance')
            plt.ylabel('G-G Distance (Å)')
        else:
            # For COM distances, just plot the single dataset
            plt.plot(time_points, raw_data1, color='green', alpha=0.7, label='Toxin-Channel Distance')
            plt.ylabel('COM Distance (Å)')
        
        plt.xlabel('Time (ns)')
        plt.title(f'Raw {title_prefix}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        raw_plot_path = os.path.join(output_dir, f"{'G_G' if is_gg else 'COM'}_Distance_Plot_raw.png")
        plt.savefig(raw_plot_path, dpi=150)
        plt.close()
        logger.info(f"Saved raw {title_prefix.lower()} plot to {raw_plot_path}")
        
        # 2. Filtered Data Plot (if filtered data is provided)
        if filtered_data1 is not None:
            plt.figure(figsize=(10, 6))
            if is_gg:
                plt.plot(time_points, filtered_data1, color='blue', label='A:C Distance (Filtered)')
                if filtered_data2 is not None:
                    plt.plot(time_points, filtered_data2, color='red', label='B:D Distance (Filtered)')
                plt.ylabel('G-G Distance (Å)')
            else:
                plt.plot(time_points, filtered_data1, color='green', label='Toxin-Channel Distance (Filtered)')
                plt.ylabel('COM Distance (Å)')
            
            plt.xlabel('Time (ns)')
            plt.title(f'Filtered {title_prefix}')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            
            filtered_plot_path = os.path.join(output_dir, f"{'G_G' if is_gg else 'COM'}_Distance_Plot.png")
            plt.savefig(filtered_plot_path, dpi=150)
            plt.close()
            logger.info(f"Saved filtered {title_prefix.lower()} plot to {filtered_plot_path}")
        
        # 3. Comparison Plots (Raw vs Filtered)
        if filtered_data1 is not None:
            # A:C / First dataset comparison
            plt.figure(figsize=(10, 6))
            plt.plot(time_points, raw_data1, color='lightblue', alpha=0.5, label='Raw Data')
            plt.plot(time_points, filtered_data1, color='blue', label='Filtered Data')
            
            if is_gg:
                plt.ylabel('A:C G-G Distance (Å)')
                plt.title('A:C G-G Distance Filtering Comparison')
                comparison_path = os.path.join(output_dir, "G_G_Distance_AC_Comparison.png")
            else:
                plt.ylabel('COM Distance (Å)')
                plt.title('COM Distance Filtering Comparison')
                comparison_path = os.path.join(output_dir, "COM_Stability_Comparison.png")
            
            plt.xlabel('Time (ns)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(comparison_path, dpi=150)
            plt.close()
            
            # B:D comparison (only for G-G distances)
            if is_gg and filtered_data2 is not None and raw_data2 is not None:
                plt.figure(figsize=(10, 6))
                plt.plot(time_points, raw_data2, color='lightcoral', alpha=0.5, label='Raw Data')
                plt.plot(time_points, filtered_data2, color='red', label='Filtered Data')
                plt.ylabel('B:D G-G Distance (Å)')
                plt.xlabel('Time (ns)')
                plt.title('B:D G-G Distance Filtering Comparison')
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.tight_layout()
                
                comparison_path_bd = os.path.join(output_dir, "G_G_Distance_BD_Comparison.png")
                plt.savefig(comparison_path_bd, dpi=150)
                plt.close()
        
        return True
    
    except Exception as e:
        logger.error(f"Error generating {title_prefix.lower()} plots: {e}", exc_info=True)
        return False

def plot_kde_analysis(time_points, com_distances, box_z, output_dir):
    """
    Create KDE (Kernel Density Estimation) plot for COM distances to identify
    distinct stability states.
    
    Args:
        time_points (np.ndarray): Time points in ns.
        com_distances (np.ndarray): COM distance data.
        box_z (float): Box Z dimension (not used in current implementation but kept for API compatibility).
        output_dir (str): Directory to save plots.
        
    Returns:
        bool: True if successful, False otherwise.
    """
    logger = logging.getLogger()
        
    if output_dir is None:
        logger.warning("No output directory specified for plot_kde_analysis.")
        return False
        
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [1, 2]})

        ylabel = 'Distance (Å)'
        xlabel = 'Distance (Å)'
        if com_distances is not None and len(com_distances) > 0:
            xlabel = 'COM Distance (Å)'
            ylabel = xlabel # Keep ylabel as COM Distance

        finite_data = com_distances[np.isfinite(com_distances)]
        peak_values = []  # Initialize peak_values here

        if len(finite_data) < 5:
            ax1.text(0.5, 0.5, 'Insufficient finite data for KDE/Histogram', ha='center', va='center')
            # Plot raw data vs time if available
            if time_points is not None and len(time_points) == len(com_distances):
                 ax2.plot(time_points, com_distances, color='steelblue', linewidth=1)
                 ax2.set_xlabel('Time (ns)', fontsize=14)
            else:
                 ax2.text(0.5, 0.5, 'Time data unavailable', ha='center', va='center')
            ax2.set_title('Raw Time Series Data', fontsize=16)
            ax2.set_ylabel(ylabel, fontsize=14)
            plt.tight_layout()
            plot_filename = "COM_Stability_KDE_Analysis.png" # Save even if insufficient data
            save_path = os.path.join(output_dir, plot_filename)
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            plt.close()
            logger.warning("Insufficient finite data for KDE plot generation.")
            return False

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
            
            # Add info to log
            state_info = ", ".join([f"{x:.2f} Å" for x in peak_values])
            logger.info(f"KDE analysis detected potential COM distance states at: {state_info}")
            
        except Exception as e:
            ax1.text(0.5, 0.5, f'KDE Failed:\n{e}', ha='center', va='center')
            logger.error(f"KDE plot generation failed: {e}", exc_info=True)

        # Time Series Plot with Level Coloring (Right)
        if len(peak_values) > 1:
            assignments = np.zeros_like(com_distances, dtype=int) - 1
            finite_indices = np.where(np.isfinite(com_distances))[0]
            if len(finite_indices) > 0:
                finite_vals = com_distances[finite_indices]
                distances_to_peaks = np.abs(finite_vals[:, np.newaxis] - peak_values)
                assignments[finite_indices] = np.argmin(distances_to_peaks, axis=1)

            n_levels = len(peak_values)
            cmap = plt.cm.get_cmap('viridis', n_levels)
            colors = [cmap(i) for i in range(n_levels)]

            for i in range(n_levels):
                mask = assignments == i
                if np.any(mask):
                    # Plot time vs distance
                    ax2.scatter(time_points[mask], com_distances[mask], color=colors[i], s=5, alpha=0.6)
            ax2.set_title('Time Series with Potential Level Assignments', fontsize=16)
        else:
            # Plot raw time series if no significant peaks found or insufficient data
            ax2.plot(time_points, com_distances, color='steelblue', linewidth=1)
            ax2.set_title('Raw Time Series Data', fontsize=16)

        ax2.set_xlabel('Time (ns)', fontsize=14)
        ax2.set_ylabel(ylabel, fontsize=14) # Y label is COM Distance
        ax2.tick_params(axis='both', labelsize=10)
        ax2.grid(True, alpha=0.3)

        if len(peak_values) > 0:
            ax1.set_ylim(ax2.get_ylim())

        fig.suptitle(f'COM Stability - Raw Data Analysis', fontsize=18, y=1.02)
        plt.tight_layout(rect=[0, 0.03, 1, 0.98])

        plot_filename = "COM_Stability_KDE_Analysis.png"
        save_path = os.path.join(output_dir, plot_filename)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved COM KDE analysis plot to {save_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error generating COM KDE analysis plot: {e}", exc_info=True)
        return False
