# filename: pore_analysis/modules/dw_gate_analysis/data_collection.py
"""
Functions for loading the MDAnalysis Universe and calculating raw
DW-gate distances over the trajectory.
"""

import logging
from typing import Dict, List, Optional, Tuple

import MDAnalysis as mda
from MDAnalysis.analysis.distances import distance_array
import numpy as np
import pandas as pd
from tqdm import tqdm

# Import GateResidues type hint (assuming it's defined correctly in residue_identification)
# Need to adjust relative import path if structure differs slightly
try:
    from .residue_identification import GateResidues
except ImportError:
    # Fallback if direct import fails (e.g., during testing)
    # Define a simple placeholder if needed, though type hints are documentation primarily
    class GateResidues: pass


logger = logging.getLogger(__name__)

def load_universe(psf_file: str, dcd_file: str) -> Optional[mda.Universe]:
    """Load the MDAnalysis Universe from PSF and DCD files."""
    try:
        u = mda.Universe(psf_file, dcd_file)
        logger.info(f"Successfully loaded Universe: {len(u.atoms)} atoms, {len(u.trajectory)} frames.")
        # Basic check for trajectory length
        if len(u.trajectory) == 0:
            logger.warning("Loaded trajectory has 0 frames.")
            # Depending on requirements, might return None or the Universe
            # Returning the Universe allows computation.py to handle the 0-frame case potentially
        return u
    except Exception as e:
        logger.error(f"Failed to load Universe from {psf_file} and {dcd_file}: {e}", exc_info=True)
        return None

# Note: select_dw_residues function was removed as its logic is handled
#       in residue_identification.py and called by computation.py

def calculate_dw_distances(
    universe: mda.Universe,
    gate_residues: Dict[str, GateResidues],
    start_frame: int = 0,
    end_frame: Optional[int] = None
) -> Optional[pd.DataFrame]:
    """
    Calculates the minimum distance between Asp/Glu carboxylates (OD1/OD2/OE1/OE2)
    and Trp indole nitrogen (NE1) for each identified gate pair across specified frames.

    Args:
        universe: MDAnalysis Universe, trajectory must be loaded.
        gate_residues: Dictionary mapping chain ID to GateResidues objects.
        start_frame: Starting frame index for analysis (0-based).
        end_frame: Ending frame index for analysis (exclusive). If not specified, analyzes to the end.

    Returns:
        Pandas DataFrame containing distance columns for each chain
        (e.g., 'Dist_PROA', 'Dist_PROB'), indexed by frame number (local indices, 0-based), 
        or None if calculation fails. Distances are in Angstroms.
    """
    if not gate_residues:
        logger.error("Cannot calculate distances: No valid gate residues provided.")
        return None

    valid_segids = sorted(list(gate_residues.keys()))
    n_frames_total = len(universe.trajectory)

    if n_frames_total == 0:
        logger.warning("Cannot calculate distances: Trajectory has 0 frames.")
        # Return an empty DataFrame with expected columns?
        cols = [f"Dist_{segid}" for segid in valid_segids]
        return pd.DataFrame(columns=cols)
        
    # Process frame range
    actual_end = n_frames_total if end_frame is None else end_frame
    actual_start = max(0, start_frame)
    actual_end = min(n_frames_total, actual_end)
    
    if actual_start >= actual_end:
        logger.error(f"Invalid frame range: start={actual_start}, end={actual_end}")
        return None
        
    # Number of frames to analyze
    frames_to_analyze = actual_end - actual_start
    logger.info(f"Analyzing frame range {actual_start} to {actual_end} ({frames_to_analyze} frames)")


    # Prepare atom selections
    selections = {}
    for segid, gate in gate_residues.items():
        try:
            # Select carboxyl oxygens (handle ASP/GLU) and indole nitrogen
            acidic_oxygens = gate.asp_glu_res.atoms.select_atoms("name OD1 OD2 OE1 OE2")
            trp_nitrogen = gate.trp_res.atoms.select_atoms("name NE1")

            if len(acidic_oxygens) < 2: # Should have at least OD1/OD2 or OE1/OE2
                 raise ValueError(f"Chain {segid}: Expected at least 2 carboxyl oxygens in {gate.asp_glu_res}, found {len(acidic_oxygens)}")
            if len(trp_nitrogen) != 1:
                raise ValueError(f"Chain {segid}: Expected 1 NE1 atom in {gate.trp_res}, found {len(trp_nitrogen)}")

            selections[segid] = {'acidic': acidic_oxygens, 'trp': trp_nitrogen}
            logger.debug(f"Prepared atom selections for distance calculation on chain {segid}.")
        except Exception as e:
            logger.warning(f"Failed to prepare atom selections for chain {segid}: {e}. Skipping this chain.")
            if segid in valid_segids:
                valid_segids.remove(segid) # Remove from list to process

    if not valid_segids:
        logger.error("No valid chains remaining after preparing atom selections for distance calculation.")
        return None

    logger.info(f"Calculating DW-gate distances for chains: {valid_segids} over {frames_to_analyze} frames...")
    # Initialize dict to hold numpy arrays for performance - now using frames_to_analyze for length
    distance_data_np = {f"Dist_{segid}": np.full(frames_to_analyze, np.nan) for segid in valid_segids}

    # Iterate through trajectory slice
    for i, ts in enumerate(tqdm(universe.trajectory[actual_start:actual_end], 
                               desc="DW-gate Distances", 
                               total=frames_to_analyze, 
                               disable=not logger.isEnabledFor(logging.INFO))):
        # Use local frame index for storing in arrays (0-based for the analyzed window)
        local_idx = i
        for segid in valid_segids:
            try:
                # Get atom groups for the current frame (positions updated automatically by MDA)
                acidic_ag = selections[segid]['acidic']
                trp_ag = selections[segid]['trp']

                # Calculate all pairwise distances between acidic oxygens and the trp nitrogen
                # distance_array handles PBC if box dimensions are present in ts
                dist_matrix = distance_array(acidic_ag.positions, trp_ag.positions, box=ts.dimensions)

                # Find the minimum distance for this frame and chain
                min_dist = np.min(dist_matrix) if dist_matrix.size > 0 else np.nan
                distance_data_np[f"Dist_{segid}"][local_idx] = min_dist

            except Exception as e:
                # Log only once per chain if errors persist?
                if local_idx == 0: # Log first instance
                    logger.warning(f"Error calculating distance for chain {segid} at frame {ts.frame} (local_idx={local_idx}): {e}", exc_info=False)
                # Keep NaN value assigned during initialization

    # Create DataFrame from the numpy arrays - use range starting from 0 for local indexing
    df_raw = pd.DataFrame(distance_data_np, index=pd.RangeIndex(start=0, stop=frames_to_analyze, name='Frame'))

    logger.info(f"Finished calculating DW-gate distances. DataFrame shape: {df_raw.shape}")
    # Log summary stats for debugging
    for col in df_raw.columns:
        logger.debug(f"Distance summary for {col}: Min={df_raw[col].min():.2f}, Max={df_raw[col].max():.2f}, Mean={df_raw[col].mean():.2f} Å (NaNs={df_raw[col].isna().sum()})")

    return df_raw
