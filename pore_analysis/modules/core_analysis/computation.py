# pore_analysis/modules/core_analysis/computation.py
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
    # --- FIX: Import FRAMES_PER_NS ---
    from pore_analysis.core.config import FRAMES_PER_NS
    # --- END FIX ---
except ImportError as e:
    print(f"Error importing dependency modules in core_analysis/computation.py: {e}")
    raise

logger = logging.getLogger(__name__)

def analyze_trajectory(run_dir, universe=None, start_frame=0, end_frame=None, psf_file=None, dcd_file=None):
    """
    Analyze a single trajectory file to extract raw G-G and COM distances.
    Uses a provided MDAnalysis Universe object and frame range for analysis.
    Calculates metrics per frame, and saves raw data to CSV files.
    Detects if this is a control system (no toxin) and records this information.

    Args:
        run_dir (str): Path to the specific run directory (used for output and logging).
        universe (MDAnalysis.Universe, optional): Pre-loaded MDAnalysis Universe object.
        start_frame (int, optional): Starting frame index for analysis (0-based). Defaults to 0.
        end_frame (int, optional): Ending frame index for analysis (exclusive). If None, goes to the end.
        psf_file (str, optional): Path to the PSF topology file. Only used if universe is not provided.
        dcd_file (str, optional): Path to the DCD trajectory file. Only used if universe is not provided.

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
        logger = logging.getLogger() # Use root logger as fallback
        # Avoid double logging if root is already configured
        if not logger.hasHandlers():
             logging.basicConfig(level=logging.INFO) # Basic config if none exists
        logger.error(f"Failed to setup system logger for {run_dir}. Using root logger.")

    # Connect to the database
    db_conn = connect_db(run_dir)
    if db_conn is None:
        # Log error even if logger setup failed
        logging.getLogger().error(f"Failed to connect to database for {run_dir}. Cannot proceed.")
        # Return minimal failure dict
        return {
            'status': 'failed', 'error': 'Database connection failed',
            'data': {}, 'metadata': {}, 'files': {}
        }


    # Register this module run
    module_id = register_module(db_conn, "core_analysis", "running")

    run_name = os.path.basename(run_dir)
    # Handle case where run_dir might be root (e.g., '.')
    parent_dir = os.path.dirname(run_dir)
    system_dir = os.path.basename(parent_dir) if parent_dir and os.path.basename(parent_dir) != '.' else run_name
    logger.info(f"Starting raw trajectory analysis for system: {system_dir} (Folder: {run_dir})")

    # --- Define Output Directory for Core Analysis ---
    output_dir = os.path.join(run_dir, "core_analysis")
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Core analysis outputs will be saved to: {output_dir}")

    # --- Universe Handling ---
    u = None
    try:
        if universe is not None:
            # Use the provided universe
            u = universe
            logger.info("Using provided Universe object")
        else:
            # Need to load the universe from files
            if psf_file is None:
                psf_file = os.path.join(run_dir, "step5_input.psf")
            if dcd_file is None:
                dcd_file = os.path.join(run_dir, "MD_Aligned.dcd")

            if not os.path.exists(psf_file):
                logger.error(f"PSF file not found: {psf_file}")
                update_module_status(db_conn, "core_analysis", "failed", error_message="PSF file not found")
                if db_conn: db_conn.close()
                return {
                    'status': 'failed',
                    'error': 'PSF file not found',
                    'data': {'dist_ac': np.array([]), 'dist_bd': np.array([]), 'com_distances': None, 'time_points': np.array([])},
                    'metadata': {'system_dir': system_dir, 'is_control_system': True},
                    'files': {}
                }

            if not os.path.exists(dcd_file):
                logger.error(f"DCD file not found: {dcd_file}")
                update_module_status(db_conn, "core_analysis", "failed", error_message="DCD file not found")
                if db_conn: db_conn.close()
                return {
                    'status': 'failed',
                    'error': 'DCD file not found',
                    'data': {'dist_ac': np.array([]), 'dist_bd': np.array([]), 'com_distances': None, 'time_points': np.array([])},
                    'metadata': {'system_dir': system_dir, 'is_control_system': True},
                    'files': {}
                }
            
            logger.info(f"Loading topology: {psf_file}")
            logger.info(f"Loading trajectory: {dcd_file}")
            u = mda.Universe(psf_file, dcd_file)
            
        # Validate universe
        n_frames_total = len(u.trajectory)
        logger.info(f"Successfully loaded universe with {n_frames_total} frames")
        
        if n_frames_total == 0:
            logger.warning("Trajectory contains 0 frames.")
            update_module_status(db_conn, "core_analysis", "failed", error_message="Trajectory contains 0 frames")
            if db_conn: db_conn.close()
            return {
                'status': 'failed',
                'error': 'Trajectory contains 0 frames',
                'data': {'dist_ac': np.array([]), 'dist_bd': np.array([]), 'com_distances': None, 'time_points': np.array([])},
                'metadata': {'system_dir': system_dir, 'is_control_system': True},
                'files': {}
            }
            
        # Handle frame range
        if end_frame is None:
            end_frame = n_frames_total
            
        # Log the frame range we'll use
        n_frames_in_slice = end_frame - start_frame
        logger.info(f"Analyzing frame range {start_frame} to {end_frame} (total: {n_frames_in_slice} frames)")
        
    except Exception as e:
        logger.error(f"Failed to load or validate Universe: {e}", exc_info=True)
        update_module_status(db_conn, "core_analysis", "failed", error_message=f"Failed to load or validate Universe: {str(e)}")
        if db_conn: db_conn.close()
        return {
            'status': 'failed',
            'error': f'Failed to load or validate Universe: {str(e)}',
            'data': {'dist_ac': np.array([]), 'dist_bd': np.array([]), 'com_distances': None, 'time_points': np.array([])},
            'metadata': {'system_dir': system_dir, 'is_control_system': True},
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
            idx_gyg = idx_gyg + 1 # Assuming GYG, want the middle G

        # Convert sequence index to resid
        reference_resid = chain_A_atoms.residues[idx_gyg].resid
        logger.info(f"Using CA of residue {reference_resid} (Index {idx_gyg} in sequence '{chainA_Seq}') for G-G distance.")

    except Exception as e:
        logger.error(f"Error finding reference Glycine for G-G distance: {e}", exc_info=True)
        update_module_status(db_conn, "core_analysis", "failed", error_message=f"Error finding reference Glycine: {str(e)}")
        if db_conn: db_conn.close()
        return {
            'status': 'failed',
            'error': f'Error finding reference Glycine: {str(e)}',
            'data': {'dist_ac': np.array([]), 'dist_bd': np.array([]), 'com_distances': None, 'time_points': np.array([])},
            'metadata': {'system_dir': system_dir, 'is_control_system': True},
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

    # Determine Channel Selection Strategy (PROA-D or A-D)
    chan_sel = None
    chan_sel_pro = 'segid ' + ' or segid '.join(channel_segids_pro)
    chan_sel_alt = 'segid ' + ' or segid '.join(channel_segids_alt)
    if len(u.select_atoms(chan_sel_pro)) > 0:
        chan_sel = chan_sel_pro
        logger.info(f"Using Channel selection: '{chan_sel}'")
        current_channel_segids = channel_segids_pro
    elif len(u.select_atoms(chan_sel_alt)) > 0:
        chan_sel = chan_sel_alt
        logger.info(f"Using Alternate Channel selection: '{chan_sel}'")
        current_channel_segids = channel_segids_alt
    else:
         logger.error(f"Could not find channel atoms using standard selections: '{chan_sel_pro}' or '{chan_sel_alt}'.")
         update_module_status(db_conn, "core_analysis", "failed", error_message="Could not find channel atoms")
         if db_conn: db_conn.close()
         return {
            'status': 'failed',
            'error': 'Could not find channel atoms',
            'data': {'dist_ac': np.array([]), 'dist_bd': np.array([]), 'com_distances': None, 'time_points': np.array([])},
            'metadata': {'system_dir': system_dir, 'is_control_system': True},
            'files': {}
         }

    # Determine if COM analysis is possible (if toxin was found)
    com_analyzed = pep_sel is not None
    is_control_system = not com_analyzed

    if is_control_system:
        logger.info("No toxin selection found. This appears to be a CONTROL system without toxin.")
    else:
        logger.info(f"Toxin found with selection '{pep_sel}'. This appears to be a toxin-channel complex system.")

    # --- Trajectory Iteration ---
    logger.info(f"Starting trajectory analysis loop ({n_frames_in_slice} frames, range {start_frame}:{end_frame})...")
    try:
        for ts in u.trajectory[start_frame:end_frame:1]:
            frame_indices.append(ts.frame)

            # G-G Calculation
            try:
                ca_atoms = {}
                found_all_gg = True
                opposing_chains = [('A', 'C'), ('B', 'D')]

                # Map A/B/C/D to actual segids used (PROA or A, etc.)
                chain_map = {}
                for seg in current_channel_segids:
                    if seg.startswith("PRO"): chain_map[seg[-1]] = seg # PROA -> A
                    else: chain_map[seg] = seg # A -> A

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
                            # Log which atom was missing if on first frame
                            if ts.frame == 0: logger.warning(f"G-G calc: Missing CA atom for resid {reference_resid} in segid {seg1} ({len(ca1)}) or {seg2} ({len(ca2)}).")
                            found_all_gg = False
                            break
                    else:
                        # Log which chain was missing if on first frame
                        if ts.frame == 0: logger.warning(f"G-G calc: Chain {chain1} or {chain2} not found in chain_map {chain_map}.")
                        found_all_gg = False
                        break

                if found_all_gg:
                    dist_ac.append(np.linalg.norm(ca_atoms['A'] - ca_atoms['C']))
                    dist_bd.append(np.linalg.norm(ca_atoms['B'] - ca_atoms['D']))
                else:
                    dist_ac.append(np.nan)
                    dist_bd.append(np.nan)

            except Exception as e:
                logger.error(f"Error calculating G-G distances for frame {ts.frame}: {e}", exc_info=True)
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
                        if ts.frame == 0: logger.warning(f"COM calc: Toxin ({len(pep_atoms)}) or Channel ({len(chan_atoms)}) selection empty.")
                        pep_positions.append(np.array([np.nan, np.nan, np.nan]))
                        ch_positions.append(np.array([np.nan, np.nan, np.nan]))
                except Exception as e:
                    logger.error(f"Error calculating COM positions for frame {ts.frame}: {e}", exc_info=True)
                    pep_positions.append(np.array([np.nan, np.nan, np.nan]))
                    ch_positions.append(np.array([np.nan, np.nan, np.nan]))

    except Exception as e:
        logger.error(f"Error during trajectory iteration: {e}", exc_info=True)
        update_module_status(db_conn, "core_analysis", "failed", error_message=f"Error during trajectory iteration: {str(e)}")
        if db_conn: db_conn.close()
        return {
            'status': 'failed',
            'error': f'Error during trajectory iteration: {str(e)}',
            'data': {'dist_ac': np.array([]), 'dist_bd': np.array([]), 'com_distances': None, 'time_points': np.array([])},
            'metadata': {'system_dir': system_dir, 'is_control_system': is_control_system}, # Use determined values
            'files': {}
        }

    # Convert lists to numpy arrays
    dist_ac = np.array(dist_ac)
    dist_bd = np.array(dist_bd)
    frame_indices = np.array(frame_indices)
    # --- FIX: Use FRAMES_PER_NS ---
    if FRAMES_PER_NS <= 0:
        logger.error(f"Invalid FRAMES_PER_NS: {FRAMES_PER_NS}. Cannot calculate time points.")
        update_module_status(db_conn, "core_analysis", "failed", error_message="Invalid FRAMES_PER_NS config")
        if db_conn: db_conn.close()
        # Return appropriate failure dictionary
        return {
            'status': 'failed', 'error': 'Invalid FRAMES_PER_NS config',
            'data': {}, 'metadata': {}, 'files': {}
        }
    time_points = frames_to_time(frame_indices)
    # --- END FIX ---


    # Calculate COM distances if toxin was found
    com_distances = None
    if com_analyzed and len(pep_positions) > 0 and len(ch_positions) > 0:
        pep_positions = np.array(pep_positions)
        ch_positions = np.array(ch_positions)
        # Ensure arrays are not empty after potential NaN appends
        if pep_positions.shape == ch_positions.shape and pep_positions.size > 0:
             com_distances = np.linalg.norm(pep_positions - ch_positions, axis=1)
        else:
             logger.warning("Shape mismatch or empty arrays for COM distance calculation, setting COM distances to None.")
             com_distances = np.full(len(time_points), np.nan) # Assign NaN array


    # Save raw data to CSV
    files = {}
    try:
        df_raw = pd.DataFrame({
            'Frame': frame_indices,
            'Time (ns)': time_points,
            'GG_Distance_AC': dist_ac,
            'GG_Distance_BD': dist_bd
        })
        # Use nan-aware check for com_distances before adding column
        if com_distances is not None and np.any(np.isfinite(com_distances)):
            df_raw['COM_Distance'] = com_distances
        else:
             # Add an empty/NaN column if COM wasn't analyzed or had no valid data
             df_raw['COM_Distance'] = np.nan

        raw_csv_path = os.path.join(output_dir, "Raw_Distances.csv")
        df_raw.to_csv(raw_csv_path, index=False, float_format='%.4f', na_rep='NaN') # Use NaN rep
        logger.info(f"Saved raw distance data to {raw_csv_path}")

        # --- FIX: Correct arguments for register_product ---
        # Calculate relative path correctly
        relative_csv_path = os.path.relpath(raw_csv_path, run_dir)
        # Ensure subcategory matches what the test expects
        product_id = register_product(
            conn=db_conn, # Use the correct connection variable
            module_name="core_analysis", # Correct module name
            product_type="csv",
            category="data",
            relative_path=relative_csv_path, # Pass the calculated relative path
            subcategory="raw_distances", # Match the test query
            description="Raw distance data for G-G and COM"
        )
        # --- END FIX ---

        if product_id is None:
             logger.error("Failed to register raw distances product in database.")
             # Decide if this is critical? Maybe just warn.
             # For now, continue but log error.
        else:
             files['raw_distances'] = relative_csv_path # Store relative path if registered

    except Exception as e:
        logger.error(f"Failed to save raw distance data: {e}", exc_info=True)
        update_module_status(db_conn, "core_analysis", "failed", error_message=f"Failed to save raw distance data: {str(e)}")
        if db_conn: db_conn.close()
        # Return failure but include calculated data
        return {
            'status': 'failed',
            'error': f'Failed to save raw distance data: {str(e)}',
            'data': {'dist_ac': dist_ac, 'dist_bd': dist_bd, 'com_distances': com_distances, 'time_points': time_points},
            'metadata': {'system_dir': system_dir, 'is_control_system': is_control_system},
            'files': {}
        }

    # Update module status to success
    update_module_status(db_conn, "core_analysis", "success")
    if db_conn: db_conn.close() # Close connection at the end

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
        'files': files # Contains relative path if registration succeeded
    }


def filter_and_save_data(
    run_dir: str,
    dist_ac: np.ndarray,
    dist_bd: np.ndarray,
    com_distances: Optional[np.ndarray],
    time_points: np.ndarray,
    db_conn: Optional[sqlite3.Connection], # <-- Added db_conn parameter
    box_z: Optional[float] = None,
    is_control_system: bool = False,
    start_frame: int = 0,
    end_frame: Optional[int] = None
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
        start_frame (int, optional): Starting frame index used for analysis (0-based). Defaults to 0.
        end_frame (int, optional): Ending frame index used for analysis (exclusive). 
                                   If None, assumed to be the last frame.

    Returns:
        dict: Analysis results dictionary with status, data, files, and potential errors.
    """
    module_name = "core_analysis_filtering" # Define module name for registration

    # Set up logging using the standard function
    logger = setup_system_logger(run_dir)
    if logger is None:
        logger = logging.getLogger(__name__) # Use module logger as fallback
        # Avoid double logging if root is already configured
        if not logger.hasHandlers():
             logging.basicConfig(level=logging.INFO) # Basic config if none exists
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
        
        # Store frame range information
        store_metric(db_conn, module_name, "start_frame", start_frame, "frame", "Starting frame index for filtering")
        if end_frame is not None:
            store_metric(db_conn, module_name, "end_frame", end_frame, "frame", "Ending frame index for filtering")
            store_metric(db_conn, module_name, "frames_analyzed", end_frame - start_frame, "frames", "Number of frames filtered")
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
    # Check inputs are valid numpy arrays with size > 0
    if isinstance(dist_ac, np.ndarray) and dist_ac.size > 0 and \
       isinstance(dist_bd, np.ndarray) and dist_bd.size > 0:
        try:
            logger.debug(f"{module_name}: Starting G-G filtering...")
            # Pass only relevant args to standard_filter called within auto_select_filter
            filter_kwargs_gg = {} # No extra args needed usually for G-G standard filter
            filtered_ac, filter_info_g_g_ac = auto_select_filter(dist_ac, data_type='gg_distance', **filter_kwargs_gg)
            filtered_bd, filter_info_g_g_bd = auto_select_filter(dist_bd, data_type='gg_distance', **filter_kwargs_gg)
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
                'Frame': np.arange(start_frame, start_frame + len(time_points)),  # Add frame indices
                'G_G_Distance_AC_Raw': dist_ac, 'G_G_Distance_BD_Raw': dist_bd,
                'G_G_Distance_AC_Filt': filtered_ac, 'G_G_Distance_BD_Filt': filtered_bd
            })
            gg_csv_filename = "G_G_Distance_Filtered.csv" # Define filename
            gg_csv_path = os.path.join(output_dir, gg_csv_filename)
            df_g_g.to_csv(gg_csv_path, index=False, float_format='%.4f', na_rep='NaN')
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
         logger.warning(f"{module_name}: Skipping G-G filtering due to missing or empty input data.")
         result['errors'].append("Skipped G-G filtering (missing/empty input)")


    com_success = False # Track success of COM part
    # --- COM Filtering and Stats Calculation ---
    # Check com_distances is a numpy array with size > 0
    if not is_control_system and isinstance(com_distances, np.ndarray) and com_distances.size > 0:
        try:
            logger.debug(f"{module_name}: Starting COM filtering...")
            # Pass only relevant args to auto_select_filter for COM
            filter_kwargs_com = {'box_size': box_z} # Use box_size which is accepted by multilevel
            filtered_com, filter_info_com = auto_select_filter(com_distances, data_type='com_distance', **filter_kwargs_com)
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
                'Frame': np.arange(start_frame, start_frame + len(time_points)),  # Add frame indices
                'COM_Distance_Raw': com_distances,
                'COM_Distance_Filt': filtered_com
            })
            com_csv_filename = "COM_Stability_Filtered.csv" # Define filename
            com_csv_path = os.path.join(output_dir, com_csv_filename)
            df_com.to_csv(com_csv_path, index=False, float_format='%.4f', na_rep='NaN')
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
            # If G-G also failed, status is already failed. If G-G succeeded, mark as failed now.
            result['status'] = 'failed'

    elif is_control_system:
         logger.info(f"{module_name}: Skipping COM filtering (Control System).")
         com_success = True # Skipping is considered handled
         result['data']['filtered_com'] = np.array([]) # Ensure it's an empty array for consistency
    else:
         logger.warning(f"{module_name}: Skipping COM filtering due to missing or empty input data.")
         result['errors'].append("Skipped COM filtering (missing/empty input)")
         result['data']['filtered_com'] = np.array([]) # Ensure it's an empty array
         # If COM data *should* have been present, this might indicate an earlier problem
         com_success = True # Treat missing data as 'handled' for now


    # --- Final Status Update ---
    # Succeeds if G-G (if attempted) and COM (if attempted) were successful
    final_status = 'success' if gg_success and com_success else 'failed'
    final_error_message = "; ".join(result['errors']) if result['errors'] else None

    try:
         update_module_status(db_conn, module_name, final_status, error_message=final_error_message)
    except Exception as e_update:
         logger.error(f"{module_name}: Failed to update final module status: {e_update}", exc_info=True)
         # Update the result dict even if DB update fails
         result['status'] = 'failed'
         result['errors'].append(f"Failed DB status update: {e_update}")
         # Re-assign final error message if it was previously None
         if final_error_message is None: final_error_message = f"Failed DB status update: {e_update}"


    # Update result status based on final determination
    result['status'] = final_status
    # Add final error message summary if any errors occurred
    if final_error_message:
         result['error'] = final_error_message


    # Note: We DO NOT close the db_conn here, as it was passed in from main.py
    logger.debug(f"{module_name}: filter_and_save_data finished with status: {result['status']}")
    return result
