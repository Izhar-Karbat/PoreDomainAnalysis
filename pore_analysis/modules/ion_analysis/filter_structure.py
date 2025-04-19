"""
Filter structure identification module.

Functions for identifying selectivity filter residues
and calculating binding site positions relative to reference planes.
"""

import logging
import numpy as np
import MDAnalysis as mda

# External imports
try:
    from md_analysis.core.utils import OneLetter
except ImportError as e:
    print(f"Error importing dependency modules in filter_structure.py: {e}")
    raise

# Module logger
module_logger = logging.getLogger(__name__)

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
