"""
Functions for identifying DW-Gate residues (Asp/Glu and Trp)
relative to the selectivity filter motif.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import MDAnalysis as mda

# Attempt to import the filter finding function from the ion_analysis module
try:
    # Assume relative import from within the modules directory works
    from ..ion_analysis.filter_structure import find_filter_residues
except ImportError:
    # Provide a more specific fallback if needed, or handle the error
    logger = logging.getLogger(__name__) # Need logger for fallback msg
    logger.error("Could not perform relative import of find_filter_residues. Ensure package structure is correct.")
    raise # Re-raise the import error to prevent proceeding without it

logger = logging.getLogger(__name__)

@dataclass
class GateResidues:
    """Holds the identified Asp/Glu and Trp residue objects for a single chain."""
    chain_id: str
    asp_glu_res: mda.core.groups.Residue
    trp_res: mda.core.groups.Residue

def find_gate_residues_for_chain(u: mda.Universe, segid: str, filter_res_map: dict) -> Optional[GateResidues]:
    """
    Identifies the Asp/Glu and Trp residues forming the DW-gate for a specific chain.

    Searches relative to the G1 and G2 residues of the selectivity filter (TVGYG).
    - Asp/Glu: First D/E residue with resid between G2+1 and G2+6 (inclusive).
    - Trp: W residue with the highest resid between G1-16 and G1-6 (inclusive).

    Args:
        u (mda.Universe): The MDAnalysis Universe.
        segid (str): The segment ID of the chain to analyze.
        filter_res_map (dict): Dictionary mapping segid to list of filter residue IDs
                               (output from find_filter_residues).

    Returns:
        GateResidues | None: A dataclass containing the chain_id and the identified
                            Asp/Glu and Trp MDAnalysis.Residue objects, or None if
                            identification fails for this chain.

    Raises:
        ValueError: If filter residues are not found for the chain, if D/E or W
                    residues are not found in the specified ranges, or if the
                    identified D/E and W residues do not meet the spacing validation.
    """
    logger.debug(f"Attempting to find DW gate residues for segid {segid}...")

    filter_resids = filter_res_map.get(segid)
    if not filter_resids or len(filter_resids) != 5:
        raise ValueError(f"Valid 5-residue filter not found for segid {segid} in filter_res_map.")

    # Get G1 (index 2) and G2 (index 4) resids
    g1_resid = filter_resids[2]
    g2_resid = filter_resids[4]
    logger.debug(f"Segid {segid}: G1={g1_resid}, G2={g2_resid}")

    # Select the entire chain's residues once
    try:
        chain_residues = u.select_atoms(f"segid {segid}").residues
        if not chain_residues:
             raise ValueError(f"Segid {segid} selected 0 residues.")
    except Exception as e:
        raise ValueError(f"Failed to select residues for segid {segid}: {e}")

    found_asp_glu: Optional[mda.core.groups.Residue] = None
    found_trp: Optional[mda.core.groups.Residue] = None

    # --- Find Asp/Glu (D/E) --- Residue with LOWEST resid in range (g2+1, g2+6]
    min_asp_glu_resid = float('inf')
    asp_glu_search_start = g2_resid + 1
    asp_glu_search_end = g2_resid + 6
    logger.debug(f"Searching for D/E in resid range ({asp_glu_search_start}, {asp_glu_search_end}]")
    for res in chain_residues:
        if asp_glu_search_start <= res.resid <= asp_glu_search_end:
            if res.resname in ("ASP", "GLU"):
                logger.debug(f"  Found potential D/E: {res.resname}{res.resid}")
                if res.resid < min_asp_glu_resid:
                    min_asp_glu_resid = res.resid
                    found_asp_glu = res

    if found_asp_glu is None:
        raise ValueError(f"No ASP or GLU found C-terminal to filter (resids {asp_glu_search_start}-{asp_glu_search_end}) in segid {segid}.")
    logger.debug(f"Selected D/E for segid {segid}: {found_asp_glu.resname}{found_asp_glu.resid}")

    # --- Find Trp (W) --- Residue with LOWEST resid in range [g1-16, g1-6]
    min_trp_resid = float('inf')
    trp_search_start = g1_resid - 16
    trp_search_end = g1_resid - 6
    logger.debug(f"Searching for W in resid range [{trp_search_start}, {trp_search_end}]")
    for res in chain_residues:
        if trp_search_start <= res.resid <= trp_search_end:
            if res.resname == "TRP":
                logger.debug(f"  Found potential W: {res.resname}{res.resid}")
                if res.resid < min_trp_resid:
                    min_trp_resid = res.resid
                    found_trp = res

    if found_trp is None:
        raise ValueError(f"No TRP found N-terminal to filter (resids {trp_search_start}-{trp_search_end}) in segid {segid}.")
    logger.debug(f"Selected W for segid {segid}: {found_trp.resname}{found_trp.resid}")

    # --- Validate Spacing --- (10 <= asp_idx – trp_idx <= 25)
    spacing = found_asp_glu.resid - found_trp.resid
    logger.debug(f"Residue spacing D/E ({found_asp_glu.resid}) - W ({found_trp.resid}) = {spacing}")
    if not (10 <= spacing <= 25):
        raise ValueError(f"Invalid DW gate spacing ({spacing}) for segid {segid}. Expected 10-25. (D/E: {found_asp_glu.resid}, W: {found_trp.resid})")

    logger.info(f"Successfully identified DW gate for segid {segid}: D/E={found_asp_glu.resname}{found_asp_glu.resid}, W={found_trp.resname}{found_trp.resid}")
    return GateResidues(chain_id=segid, asp_glu_res=found_asp_glu, trp_res=found_trp) 