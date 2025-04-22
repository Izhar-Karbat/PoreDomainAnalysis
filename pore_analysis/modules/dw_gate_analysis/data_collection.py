"""
Functions for loading the MDAnalysis Universe, selecting DW-gate residues,
and calculating the raw distances between them over the trajectory.
"""

import logging
from typing import Dict, List, Optional, Tuple

import MDAnalysis as mda
from MDAnalysis.analysis.distances import distance_array
from MDAnalysis.core.groups import AtomGroup
import numpy as np
import pandas as pd
from tqdm import tqdm

# Assuming residue_identification and utils are siblings
from .residue_identification import find_gate_residues_for_chain, GateResidues
from ...core.utils import frames_to_time # For time conversion

logger = logging.getLogger(__name__)

def load_universe(psf_file: str, dcd_file: str) -> Optional[mda.Universe]:
    """Load the MDAnalysis Universe from PSF and DCD files."""
    try:
        u = mda.Universe(psf_file, dcd_file)
        logger.info(f"Successfully loaded Universe: {len(u.atoms)} atoms, {len(u.trajectory)} frames.")
        return u
    except Exception as e:
        logger.error(f"Failed to load Universe from {psf_file} and {dcd_file}: {e}", exc_info=True)
        return None

def select_dw_residues(
    universe: mda.Universe,
    chain_ids: List[str],
    filter_res_map: Dict[str, List[int]] # Added filter_res_map
    # Removed redundant parameters:
    # asp_glu_resname: str = "ASP",
    # asp_glu_resid: int = 27,
    # trp_resname: str = "TRP",
    # trp_resid: int = 31
) -> Dict[str, GateResidues]:
    """
    Selects the D/E and W residues involved in the gate for each specified chain.

    Uses `find_gate_residues_for_chain` from `residue_identification.py` which searches
    relative to the selectivity filter (G1/G2) defined in `filter_res_map`.

    Args:
        universe: MDAnalysis Universe.
        chain_ids: List of segment/chain IDs to search within.
        filter_res_map: Dictionary mapping chain ID to list of filter residue IDs.

    Returns:
        Dictionary mapping chain ID to the found GateResidues object.
        Returns an empty dict if no valid residues are found or chain IDs are invalid.
    """
    gate_residues: Dict[str, GateResidues] = {}
    logger.info(f"Attempting to identify DW-gate residues relative to filter for chains: {chain_ids}")

    if not filter_res_map:
        logger.error("Filter residue map is empty. Cannot identify relative DW-gate residues.")
        return {}

    for segid in chain_ids:
        # Check if this segid exists in the filter map before proceeding
        if segid not in filter_res_map:
            logger.warning(f"Skipping DW gate search for segid {segid}: Not found in filter_res_map.")
            continue

        try:
            # Call find_gate_residues_for_chain with the correct 3 arguments
            gate = find_gate_residues_for_chain(
                u=universe, # Use keyword arg for clarity
                segid=segid,
                filter_res_map=filter_res_map
            )
            if gate:
                gate_residues[segid] = gate
                logger.debug(f"Found gate residues for chain {segid}: {gate}")
            # No else needed, find_gate_residues_for_chain raises ValueError on failure
        except ValueError as e:
            # Catch specific errors from find_gate_residues_for_chain
            logger.warning(f"Could not identify DW gate for chain {segid}: {e}")
        except Exception as e:
            # Catch unexpected errors
            logger.error(f"Unexpected error finding gate residues for chain {segid}: {e}", exc_info=True)

    if not gate_residues:
        logger.error("Failed to identify any valid DW-gate residue pairs for the specified chains.")
    else:
        logger.info(f"Successfully identified gate residues for chains: {list(gate_residues.keys())}")
    return gate_residues

def calculate_dw_distances(
    universe: mda.Universe,
    gate_residues: Dict[str, GateResidues],
    dt_ns: float # Time step in ns
) -> Optional[pd.DataFrame]:
    """
    Calculates the minimum distance between Asp/Glu carboxylates (OD1/OD2/OE1/OE2)
    and Trp indole nitrogen (NE1) for each identified gate pair across all frames.

    Args:
        universe: MDAnalysis Universe, trajectory must be loaded.
        gate_residues: Dictionary mapping chain ID to GateResidues objects.
        dt_ns: Time difference between frames in nanoseconds.

    Returns:
        Pandas DataFrame containing 'Time (ns)' and distance columns for each chain
        (e.g., 'Dist_ChainA', 'Dist_ChainB'), or None if calculation fails.
        Distances are in Angstroms.
    """
    if not gate_residues:
        logger.error("Cannot calculate distances: No valid gate residues provided.")
        return None

    valid_segids = sorted(list(gate_residues.keys()))
    n_frames = len(universe.trajectory)
    time_axis = np.arange(n_frames) * dt_ns

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
    distance_data = {f"Dist_{segid}": np.full(n_frames, np.nan) for segid in valid_segids}

    # Iterate through trajectory
    for i, ts in enumerate(tqdm(universe.trajectory, desc="DW-gate Distances", total=n_frames)):
        for segid in valid_segids:
            try:
                # Get atom groups for the current frame
                acidic_ag = selections[segid]['acidic']
                trp_ag = selections[segid]['trp']

                # Calculate all pairwise distances between acidic oxygens and the trp nitrogen
                # distance_array handles PBC if box dimensions are present in ts
                dist_matrix = distance_array(acidic_ag.positions, trp_ag.positions, box=ts.dimensions)

                # Find the minimum distance for this frame and chain
                min_dist = np.min(dist_matrix)
                distance_data[f"Dist_{segid}"][i] = min_dist

            except Exception as e:
                logger.warning(f"Error calculating distance for chain {segid} at frame {ts.frame}: {e}", exc_info=False) # Avoid spamming logs
                # Keep NaN value

    # Create DataFrame
    df_raw = pd.DataFrame(distance_data)
    df_raw.insert(0, 'Time (ns)', time_axis)

    logger.info(f"Finished calculating DW-gate distances. DataFrame shape: {df_raw.shape}")
    # Log summary stats for debugging
    for col in df_raw.columns:
        if col != 'Time (ns)':
            logger.debug(f"Distance summary for {col}: Min={df_raw[col].min():.2f}, Max={df_raw[col].max():.2f}, Mean={df_raw[col].mean():.2f} Å (NaNs={df_raw[col].isna().sum()})")

    return df_raw

# Placeholder - Remove the old placeholder comment
# Placeholder for data collection functions 