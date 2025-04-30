"""
Core analysis computation functions for reading trajectories, calculating raw G-G and COM distances,
and applying filtering. This module focuses purely on the computational aspects.
"""

import os
import logging
import numpy as np
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.analysis import distances
import sqlite3
from typing import Optional, Dict, Any

# Import from other modules
try:
    from pore_analysis.core.utils import OneLetter, frames_to_time
    from pore_analysis.core.filtering import auto_select_filter
    from pore_analysis.core.logging import setup_system_logger
    from pore_analysis.core.database import (
        connect_db, register_module, register_product, store_metric,
        update_module_status
    )
except ImportError as e:
    print(f"Error importing dependency modules in core_analysis/computation.py: {e}")
    raise

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
        dict: A dictionary with analysis results and status information, including:
             - 'status': Analysis status ('success', 'failed')
             - 'data': A dict with raw data (dist_ac, dist_bd, com_distances, time_points)
             - 'metadata': A dict with system information (system_dir, is_control_system)
             - 'files': A dict mapping file types to their paths
    """
    # Set up system-specific logger
    logger = setup_system_logger(run_dir)
    if logger is None:
        logger = logging.getLogger()
        logger.error(f"Failed to setup system logger for {run_dir}. Using root logger.")

    # Connect to the database
    db_conn = connect_db(run_dir)
    
    # Register this module run
    module_id = register_module(db_conn, "core_analysis", "running")
    
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
        update_module_status(db_conn, "core_analysis", "failed", error_message="PSF file not found")
        return {
            'status': 'failed',
            'error': 'PSF file not found',
            'data': {'dist_ac': np.array([0]), 'dist_bd': np.array([0]), 'com_distances': None, 'time_points': np.array([0])},
            'metadata': {'system_dir': 'Unknown', 'is_control_system': True},
            'files': {}
        }
    
    if not os.path.exists(dcd_file):
        logger.error(f"DCD file not found: {dcd_file}")
        update_module_status(db_conn, "core_analysis", "failed", error_message="DCD file not found")
        return {
            'status': 'failed',
            'error': 'DCD file not found',
            'data': {'dist_ac': np.array([0]), 'dist_bd': np.array([0]), 'com_distances': None, 'time_points': np.array([0])},
            'metadata': {'system_dir': 'Unknown', 'is_control_system': True},
            'files': {}
        }

    # --- Load Universe ---
    try:
        logger.info(f"Loading topology: {psf_file}")
        logger.info(f"Loading trajectory: {dcd_file}")
        u = mda.Universe(psf_file, dcd_file)
        n_frames = len(u.trajectory)
        logger.info(f"Successfully loaded universe with {n_frames} frames")
        if n_frames == 0:
            logger.warning("Trajectory contains 0 frames.")
            update_module_status(db_conn, "core_analysis", "failed", error_message="Trajectory contains 0 frames")
            return {
                'status': 'failed',
                'error': 'Trajectory contains 0 frames',
                'data': {'dist_ac': np.array([]), 'dist_bd': np.array([]), 'com_distances': None, 'time_points': np.array([])},
                'metadata': {'system_dir': system_dir, 'is_control_system': True},
                'files': {}
            }
    except Exception as e:
        logger.error(f"Failed to load MDAnalysis Universe: {e}", exc_info=True)
        update_module_status(db_conn, "core_analysis", "failed", error_message=f"Failed to load Universe: {str(e)}")
        return {
            'status': 'failed',
            'error': f'Failed to load Universe: {str(e)}',
            'data': {'dist_ac': np.array([]), 'dist_bd': np.array([]), 'com_distances': None, 'time_points': np.array([])},
            'metadata': {'system_dir': 'Unknown', 'is_control_system': True},
            'files': {}
        }

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
        update_module_status(db_conn, "core_analysis", "failed", error_message=f"Error finding reference Glycine: {str(e)}")
        return {
            'status': 'failed',
            'error': f'Error finding reference Glycine: {str(e)}',
            'data': {'dist_ac': np.array([]), 'dist_bd': np.array([]), 'com_distances': None, 'time_points': np.array([])},
            'metadata': {'system_dir': 'Unknown', 'is_control_system': True},
            'files': {}
        }

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
             update_module_status(db_conn, "core_analysis", "failed", error_message="Could not find channel atoms")
             return {
                'status': 'failed',
                'error': 'Could not find channel atoms',
                'data': {'dist_ac': np.array([]), 'dist_bd': np.array([]), 'com_distances': None, 'time_points': np.array([])},
                'metadata': {'system_dir': 'Unknown', 'is_control_system': True},
                'files': {}
             }

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
        update_module_status(db_conn, "core_analysis", "failed", error_message=f"Error during trajectory iteration: {str(e)}")
        return {
            'status': 'failed',
            'error': f'Error during trajectory iteration: {str(e)}',
            'data': {'dist_ac': np.array([]), 'dist_bd': np.array([]), 'com_distances': None, 'time_points': np.array([])},
            'metadata': {'system_dir': 'Unknown', 'is_control_system': True},
            'files': {}
        }

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
    files = {}
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
        
        # Register the product in the database
        product_id = register_product(db_conn, "core_analysis", "csv", "data", "raw_distances", 
                                     "core_analysis/Raw_Distances.csv", 
                                     "Raw distance data for G-G and COM")
        files['raw_distances'] = raw_csv_path
        
    except Exception as e:
        logger.error(f"Failed to save raw distance data: {e}", exc_info=True)
        update_module_status(db_conn, "core_analysis", "failed", error_message=f"Failed to save raw distance data: {str(e)}")
        return {
            'status': 'failed',
            'error': f'Failed to save raw distance data: {str(e)}',
            'data': {'dist_ac': dist_ac, 'dist_bd': dist_bd, 'com_distances': com_distances, 'time_points': time_points},
            'metadata': {'system_dir': system_dir, 'is_control_system': is_control_system},
            'files': {}
        }
    
    # Update module status to success
    update_module_status(db_conn, "core_analysis", "success")
    
    # Return the result dictionary
    return {
        'status': 'success',
        'data': {
            'dist_ac': dist_ac,
            'dist_bd': dist_bd,
            'com_distances': com_distances,
            'time_points': time_points
        },
        'metadata': {
            'system_dir': system_dir,
            'is_control_system': is_control_system
        },
        'files': files
    }

def filter_and_save_data(
    run_dir: str,
    dist_ac: np.ndarray,
    dist_bd: np.ndarray,
    com_distances: Optional[np.ndarray],
    time_points: np.ndarray,
    db_conn: Optional[sqlite3.Connection], # <-- Added db_conn parameter
    box_z: Optional[float] = None,
    is_control_system: bool = False
    ) -> Dict[str, Any]: # Added return type hint
    """
    Apply filtering to raw distance data, calculate stats (Mean, Std, Min, Max),
    and save filtered results and metrics using the provided DB connection.

    Args:
        run_dir (str): Directory path for the run data.
        dist_ac (np.ndarray): Raw A:C distance data.
        dist_bd (np.ndarray): Raw B:D distance data.
        com_distances (np.ndarray, optional): Raw COM distance data. Can be None.
        time_points (np.ndarray): Time points corresponding to the distance data.
        db_conn (sqlite3.Connection, optional): Existing database connection. Must be provided.
        box_z (float, optional): Z dimension of the simulation box.
        is_control_system (bool): Whether this is a control system without toxin.

    Returns:
        dict: Analysis results dictionary with status, data, files, and potential errors.
    """
    module_name = "core_analysis_filtering" # Define module name for registration

    # Set up logging using the standard function
    logger = setup_system_logger(run_dir)
    if logger is None:
        logger = logging.getLogger(__name__) # Use module logger as fallback
        logger.error(f"{module_name}: Failed to setup system logger for {run_dir}. Using module logger.")

    # --- Use the passed database connection ---
    if db_conn is None:
        # Changed behavior: This function now REQUIRES a connection to be passed.
        error_msg = f"{module_name}: Database connection was not provided."
        logger.error(error_msg)
        return {
            'status': 'failed', 'error': error_msg, 'data': {}, 'files': {}
        }

    # Register this module run (sub-process of core_analysis)
    try:
        module_id = register_module(db_conn, module_name, "running")
    except Exception as e_reg:
         logger.error(f"{module_name}: Failed to register module start: {e_reg}", exc_info=True)
         # Decide how to handle this - potentially return failure?
         return {
            'status': 'failed', 'error': f"Failed module registration: {e_reg}", 'data': {}, 'files': {}
         }


    # Create output directory
    output_dir = os.path.join(run_dir, "core_analysis")
    os.makedirs(output_dir, exist_ok=True)

    # Initialize result dictionary
    result = {
        'status': 'success', # Assume success unless error occurs
        'data': {
            'filtered_ac': np.array([]),
            'filtered_bd': np.array([]),
            'filtered_com': np.array([]),
            'filter_info_g_g': {},
            'filter_info_com': {},
            'gg_stats': {}, # Changed from raw_dist_stats for clarity
            'com_stats': {},
            'percentile_stats': {}
        },
        'files': {},
        'errors': [] # List to store non-fatal errors
    }

    gg_success = False # Track success of G-G part
    # --- G-G Filtering and Stats Calculation ---
    if dist_ac is not None and dist_bd is not None and len(dist_ac) > 0 and len(dist_bd) > 0:
        try:
            logger.debug(f"{module_name}: Starting G-G filtering...")
            filtered_ac, filter_info_g_g_ac = auto_select_filter(dist_ac, data_type='gg_distance')
            filtered_bd, filter_info_g_g_bd = auto_select_filter(dist_bd, data_type='gg_distance')
            logger.debug(f"{module_name}: Finished G-G filtering.")

            # Calculate statistics on FILTERED data
            gg_stats = {
                'G_G_AC_Mean_Filt': np.nanmean(filtered_ac), 'G_G_AC_Std_Filt': np.nanstd(filtered_ac),
                'G_G_AC_Min_Filt': np.nanmin(filtered_ac) if np.any(np.isfinite(filtered_ac)) else np.nan,
                'G_G_AC_Max_Filt': np.nanmax(filtered_ac) if np.any(np.isfinite(filtered_ac)) else np.nan,
                'G_G_BD_Mean_Filt': np.nanmean(filtered_bd), 'G_G_BD_Std_Filt': np.nanstd(filtered_bd),
                'G_G_BD_Min_Filt': np.nanmin(filtered_bd) if np.any(np.isfinite(filtered_bd)) else np.nan,
                'G_G_BD_Max_Filt': np.nanmax(filtered_bd) if np.any(np.isfinite(filtered_bd)) else np.nan
            }
            percentile_stats = {
                 'G_G_AC_Pctl10_Filt': np.nanpercentile(filtered_ac, 10) if np.any(np.isfinite(filtered_ac)) else np.nan,
                 'G_G_AC_Pctl90_Filt': np.nanpercentile(filtered_ac, 90) if np.any(np.isfinite(filtered_ac)) else np.nan,
                 'G_G_BD_Pctl10_Filt': np.nanpercentile(filtered_bd, 10) if np.any(np.isfinite(filtered_bd)) else np.nan,
                 'G_G_BD_Pctl90_Filt': np.nanpercentile(filtered_bd, 90) if np.any(np.isfinite(filtered_bd)) else np.nan
            }

            # Save filtered G-G data
            df_g_g = pd.DataFrame({
                'Time (ns)': time_points,
                'G_G_Distance_AC_Raw': dist_ac, 'G_G_Distance_BD_Raw': dist_bd,
                'G_G_Distance_AC_Filt': filtered_ac, 'G_G_Distance_BD_Filt': filtered_bd
            })
            gg_csv_filename = "G_G_Distance_Filtered.csv" # Define filename
            gg_csv_path = os.path.join(output_dir, gg_csv_filename)
            df_g_g.to_csv(gg_csv_path, index=False, float_format='%.4f')
            logger.info(f"{module_name}: Saved filtered G-G distances to {gg_csv_path}")
            relative_gg_path = os.path.relpath(gg_csv_path, run_dir)

            # Register the product in the database (using db_conn)
            register_product(db_conn, module_name, "csv", "data",
                             relative_gg_path, # Relative path
                             subcategory="g_g_distance_filtered",
                             description="Filtered G-G distances data")
            result['files']['g_g_distance_filtered'] = relative_gg_path # Store relative path

            # Store key metrics in the database for G-G distances
            store_metric(db_conn, module_name, "G_G_AC_Mean_Filt", gg_stats['G_G_AC_Mean_Filt'], "Å", "Mean filtered A-C G-G distance")
            store_metric(db_conn, module_name, "G_G_BD_Mean_Filt", gg_stats['G_G_BD_Mean_Filt'], "Å", "Mean filtered B-D G-G distance")
            store_metric(db_conn, module_name, "G_G_AC_Std_Filt", gg_stats['G_G_AC_Std_Filt'], "Å", "Std Dev filtered A-C G-G distance")
            store_metric(db_conn, module_name, "G_G_BD_Std_Filt", gg_stats['G_G_BD_Std_Filt'], "Å", "Std Dev filtered B-D G-G distance")
            store_metric(db_conn, module_name, "G_G_AC_Min_Filt", gg_stats['G_G_AC_Min_Filt'], "Å", "Min filtered A-C G-G distance")
            store_metric(db_conn, module_name, "G_G_BD_Min_Filt", gg_stats['G_G_BD_Min_Filt'], "Å", "Min filtered B-D G-G distance")
            store_metric(db_conn, module_name, "G_G_AC_Max_Filt", gg_stats['G_G_AC_Max_Filt'], "Å", "Max filtered A-C G-G distance")
            store_metric(db_conn, module_name, "G_G_BD_Max_Filt", gg_stats['G_G_BD_Max_Filt'], "Å", "Max filtered B-D G-G distance")

            # Update result data
            result['data']['filtered_ac'] = filtered_ac
            result['data']['filtered_bd'] = filtered_bd
            result['data']['filter_info_g_g'] = {'AC': filter_info_g_g_ac, 'BD': filter_info_g_g_bd}
            result['data']['gg_stats'] = gg_stats
            result['data']['percentile_stats'].update(percentile_stats) # Merge percentiles
            gg_success = True

        except Exception as e:
            error_msg = f"Error in G-G filtering: {e}"
            logger.error(f"{module_name}: {error_msg}", exc_info=True)
            result['errors'].append(error_msg)
            result['status'] = 'failed' # Mark overall as failed if G-G fails

    else:
         logger.warning(f"{module_name}: Skipping G-G filtering due to missing input data.")
         result['errors'].append("Skipped G-G filtering (missing input)")


    com_success = False # Track success of COM part
    # --- COM Filtering and Stats Calculation ---
    if not is_control_system and com_distances is not None and len(com_distances) > 0:
        try:
            logger.debug(f"{module_name}: Starting COM filtering...")
            filtered_com, filter_info_com = auto_select_filter(com_distances, data_type='com_distance', box_z=box_z)
            logger.debug(f"{module_name}: Finished COM filtering.")

            # Calculate statistics on FILTERED COM data
            com_stats = {
                'COM_Mean_Filt': np.nanmean(filtered_com), 'COM_Std_Filt': np.nanstd(filtered_com),
                'COM_Min_Filt': np.nanmin(filtered_com) if np.any(np.isfinite(filtered_com)) else np.nan,
                'COM_Max_Filt': np.nanmax(filtered_com) if np.any(np.isfinite(filtered_com)) else np.nan
            }
            com_percentiles = np.nanpercentile(filtered_com, [10, 25, 50, 75, 90]) if np.any(np.isfinite(filtered_com)) else [np.nan]*5
            result['data']['percentile_stats']['COM_Pctl10_Filt'] = com_percentiles[0]
            result['data']['percentile_stats']['COM_Pctl25_Filt'] = com_percentiles[1]
            result['data']['percentile_stats']['COM_Pctl50_Filt'] = com_percentiles[2]
            result['data']['percentile_stats']['COM_Pctl75_Filt'] = com_percentiles[3]
            result['data']['percentile_stats']['COM_Pctl90_Filt'] = com_percentiles[4]


            # Save filtered COM data
            df_com = pd.DataFrame({
                'Time (ns)': time_points,
                'COM_Distance_Raw': com_distances,
                'COM_Distance_Filt': filtered_com
            })
            com_csv_filename = "COM_Stability_Filtered.csv" # Define filename
            com_csv_path = os.path.join(output_dir, com_csv_filename)
            df_com.to_csv(com_csv_path, index=False, float_format='%.4f')
            logger.info(f"{module_name}: Saved filtered COM distances to {com_csv_path}")
            relative_com_path = os.path.relpath(com_csv_path, run_dir)

            # Register the product in the database
            register_product(db_conn, module_name, "csv", "data",
                             relative_com_path, # Relative path
                             subcategory="com_stability_filtered",
                             description="Filtered COM distances data")
            result['files']['com_stability_filtered'] = relative_com_path # Store relative path

            # Store key metrics in the database for COM metrics
            store_metric(db_conn, module_name, "COM_Mean_Filt", com_stats['COM_Mean_Filt'], "Å", "Mean filtered COM distance")
            store_metric(db_conn, module_name, "COM_Std_Filt", com_stats['COM_Std_Filt'], "Å", "Std Dev filtered COM distance")
            store_metric(db_conn, module_name, "COM_Min_Filt", com_stats['COM_Min_Filt'], "Å", "Min filtered COM distance")
            store_metric(db_conn, module_name, "COM_Max_Filt", com_stats['COM_Max_Filt'], "Å", "Max filtered COM distance")

            # Update result data
            result['data']['filtered_com'] = filtered_com
            result['data']['filter_info_com'] = filter_info_com
            result['data']['com_stats'] = com_stats
            com_success = True

        except Exception as e:
            error_msg = f"Error in COM filtering: {e}"
            logger.error(f"{module_name}: {error_msg}", exc_info=True)
            result['errors'].append(error_msg)
            # Don't necessarily mark overall status as failed just for COM
            # result['status'] = 'failed'

    elif is_control_system:
         logger.info(f"{module_name}: Skipping COM filtering (Control System).")
         com_success = True # Skipping is considered handled
    else:
         logger.warning(f"{module_name}: Skipping COM filtering due to missing input data.")
         result['errors'].append("Skipped COM filtering (missing input)")
         # If COM data *should* have been present, this might indicate an earlier problem
         com_success = True # Treat missing data as 'handled' for now


    # --- Final Status Update ---
    # Succeeds if both G-G (if attempted) and COM (if attempted) did not raise critical errors
    final_status = 'success' if gg_success and com_success else 'failed'
    final_error_message = "; ".join(result['errors']) if result['errors'] else None

    try:
         update_module_status(db_conn, module_name, final_status, error_message=final_error_message)
    except Exception as e_update:
         logger.error(f"{module_name}: Failed to update final module status: {e_update}", exc_info=True)
         # Update the result dict even if DB update fails
         result['status'] = 'failed'
         result['errors'].append(f"Failed DB status update: {e_update}")


    # Update result status based on final determination
    result['status'] = final_status
    # Add final error message summary if any errors occurred
    if final_error_message:
         result['error'] = final_error_message


    # Note: We DO NOT close the db_conn here, as it was passed in from main.py
    logger.debug(f"{module_name}: filter_and_save_data finished with status: {result['status']}")
    return result