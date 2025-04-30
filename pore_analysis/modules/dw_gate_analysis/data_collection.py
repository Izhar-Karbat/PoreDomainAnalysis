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
    gate_residues: Dict[str, GateResidues]
    # dt_ns removed - time axis handled by computation.py
) -> Optional[pd.DataFrame]:
    """
    Calculates the minimum distance between Asp/Glu carboxylates (OD1/OD2/OE1/OE2)
    and Trp indole nitrogen (NE1) for each identified gate pair across all frames.

    Args:
        universe: MDAnalysis Universe, trajectory must be loaded.
        gate_residues: Dictionary mapping chain ID to GateResidues objects.

    Returns:
        Pandas DataFrame containing distance columns for each chain
        (e.g., 'Dist_PROA', 'Dist_PROB'), indexed by frame number, or None if calculation fails.
        Distances are in Angstroms.
    """
    if not gate_residues:
        logger.error("Cannot calculate distances: No valid gate residues provided.")
        return None

    valid_segids = sorted(list(gate_residues.keys()))
    n_frames = len(universe.trajectory)

    if n_frames == 0:
        logger.warning("Cannot calculate distances: Trajectory has 0 frames.")
        # Return an empty DataFrame with expected columns?
        cols = [f"Dist_{segid}" for segid in valid_segids]
        return pd.DataFrame(columns=cols)


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

    logger.info(f"Calculating DW-gate distances for chains: {valid_segids} over {n_frames} frames...")
    # Initialize dict to hold numpy arrays for performance
    distance_data_np = {f"Dist_{segid}": np.full(n_frames, np.nan) for segid in valid_segids}

    # Iterate through trajectory
    for ts in tqdm(universe.trajectory, desc="DW-gate Distances", total=n_frames, disable=not logger.isEnabledFor(logging.INFO)):
        frame_idx = ts.frame
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
                distance_data_np[f"Dist_{segid}"][frame_idx] = min_dist

            except Exception as e:
                # Log only once per chain if errors persist?
                if frame_idx == 0: # Log first instance
                    logger.warning(f"Error calculating distance for chain {segid} at frame {ts.frame}: {e}", exc_info=False)
                # Keep NaN value assigned during initialization

    # Create DataFrame from the numpy arrays
    df_raw = pd.DataFrame(distance_data_np, index=pd.RangeIndex(start=0, stop=n_frames, name='Frame'))

    logger.info(f"Finished calculating DW-gate distances. DataFrame shape: {df_raw.shape}")
    # Log summary stats for debugging
    for col in df_raw.columns:
        logger.debug(f"Distance summary for {col}: Min={df_raw[col].min():.2f}, Max={df_raw[col].max():.2f}, Mean={df_raw[col].mean():.2f} Å (NaNs={df_raw[col].isna().sum()})")

    return df_raw
