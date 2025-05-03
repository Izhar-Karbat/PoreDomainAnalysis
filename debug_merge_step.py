# debug_merge_step.py
import os
import sys
import argparse
import logging
import sqlite3
import json
import numpy as np
import MDAnalysis as mda
from typing import Dict, List, Optional, Tuple

# --- Basic Logging Setup ---
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s] - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger("debug_merge")

# --- Minimal Database Functions ---
# (Same as in the previous debug script: connect_db, get_simulation_metadata)
def connect_db(run_dir: str) -> Optional[sqlite3.Connection]:
    db_path = os.path.join(run_dir, "analysis_registry.db")
    if not os.path.exists(db_path):
        logger.error(f"Database file not found: {db_path}")
        return None
    try:
        conn = sqlite3.connect(db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        logger.info(f"Connected to database: {db_path}")
        return conn
    except sqlite3.Error as e:
        logger.error(f"Failed to connect to database at {db_path}: {e}")
        return None

def get_simulation_metadata(conn: sqlite3.Connection, key: str) -> Optional[str]:
    if conn is None: return None
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM simulation_metadata WHERE key = ?", (key,))
        result = cursor.fetchone()
        return result['value'] if result else None
    except sqlite3.Error as e:
        logger.error(f"DB Error getting simulation metadata '{key}': {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error getting simulation metadata '{key}': {e}")
        return None

# --- Function to replicate the selection process UP TO the merge ---
# (This is the corrected version from the previous step, but returns the list)
def select_atomgroups_for_merge(
    universe: mda.Universe,
    filter_res_map: Dict[str, List[int]],
    n_upstream: int,
    n_downstream: int
) -> Optional[List[mda.AtomGroup]]:
    """
    Selects backbone AtomGroups per chain, returns list ready for merging.
    """
    selected_atoms_list = []
    chain_segids = sorted(list(filter_res_map.keys()))
    logger.debug(f"Selecting backbone atoms for chains {chain_segids} using filter map: {filter_res_map}")

    for subunit_id, segid in enumerate(chain_segids):
        logger.debug(f"Processing subunit {segid} (Internal ID: {subunit_id})...")
        filter_resids = filter_res_map.get(segid)
        if not filter_resids or len(filter_resids) != 5:
            logger.warning(f"Invalid or missing filter resids list for {segid}: {filter_resids}. Skipping.")
            continue

        subunit_protein_atoms = universe.select_atoms(f'segid {segid} and protein')
        if len(subunit_protein_atoms) == 0:
            logger.warning(f"No protein atoms found for subunit {segid}, skipping.")
            continue

        subunit_residues = subunit_protein_atoms.residues
        num_subunit_residues = len(subunit_residues)
        logger.debug(f"  Subunit {segid} has {num_subunit_residues} residue objects.")
        if num_subunit_residues == 0: continue

        filter_start_resid = filter_resids[0]
        filter_end_resid = filter_resids[-1]

        start_res_index, end_res_index = -1, -1
        for i, res in enumerate(subunit_residues):
            if res.resid == filter_start_resid and start_res_index == -1: start_res_index = i
            if res.resid == filter_end_resid: end_res_index = i

        if start_res_index == -1 or end_res_index == -1 or start_res_index > end_res_index:
            logger.warning(f"Filter resids {filter_start_resid}/{filter_end_resid} not found correctly in {segid} residue objects (Indices: {start_res_index}/{end_res_index}). Skipping.")
            continue
        logger.debug(f"  Found start_res_index={start_res_index}, end_res_index={end_res_index}")

        select_start_index = max(0, start_res_index - n_upstream)
        select_end_index = min(num_subunit_residues, end_res_index + 1 + n_downstream)
        logger.debug(f"  Calculated slice indices: select_start_index={select_start_index}, select_end_index={select_end_index}")

        if select_start_index >= select_end_index:
            logger.warning(f"Invalid residue slice indices calculated for {segid} (start={select_start_index}, end={select_end_index}). Skipping.")
            continue

        try:
            start_select_resid = subunit_residues[select_start_index].resid
            if select_end_index > 0: end_select_resid = subunit_residues[select_end_index - 1].resid
            else: continue

            logger.debug(f"  Selecting backbone for {segid} between resids {start_select_resid} and {end_select_resid}")
            backbone_selection = subunit_protein_atoms.select_atoms(f'resid {start_select_resid}:{end_select_resid} and backbone')

            if len(backbone_selection) > 0:
                logger.debug(f"    Selected {len(backbone_selection)} backbone atoms for {segid}. Type: {type(backbone_selection)}")
                selected_atoms_list.append(backbone_selection) # Add the AtomGroup
            else:
                logger.warning(f"Backbone selection yielded 0 atoms for {segid} resid {start_select_resid}:{end_select_resid}.")
        except IndexError as e_idx:
             logger.error(f"IndexError accessing subunit_residues list for {segid}: {e_idx}")
             continue

    if not selected_atoms_list:
        logger.error("Backbone selection failed for ALL subunits.")
        return None

    valid_ags = [ag for ag in selected_atoms_list if isinstance(ag, mda.AtomGroup) and len(ag) > 0]
    if not valid_ags:
        logger.error("No valid AtomGroups found after selection loop.")
        return None

    logger.info(f"Returning list of {len(valid_ags)} valid AtomGroups to be merged.")
    return valid_ags


# --- Main Script Logic ---
def main():
    parser = argparse.ArgumentParser(description="Debug mda.Merge step.")
    parser.add_argument("--run_dir", required=True, help="Path to the simulation run directory.")
    parser.add_argument('--psf', help='Override PSF file path.')
    parser.add_argument('--dcd', help='Override DCD file path.')
    parser.add_argument('--n_upstream', type=int, default=10, help='Number of upstream residues.')
    parser.add_argument('--n_downstream', type=int, default=10, help='Number of downstream residues.')
    parser.add_argument('--frame', type=int, default=0, help='Frame index to analyze (default: 0).')
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    logger.info(f"Debugging merge step for run: {run_dir}")

    # --- Database Connection & Filter Map ---
    db_conn = connect_db(run_dir)
    if db_conn is None: sys.exit(1)
    filter_res_map = None
    try:
        filter_res_json = get_simulation_metadata(db_conn, 'filter_residues_dict')
        if filter_res_json: filter_res_map = json.loads(filter_res_json)
        if not isinstance(filter_res_map, dict): raise TypeError("Filter map is not dict")
    except Exception as e: logger.error(f"Error loading filter residue map: {e}"); filter_res_map = None
    if filter_res_map is None: logger.error("Failed to load filter map."); db_conn.close(); sys.exit(1)

    # --- Load Universe ---
    psf_file = args.psf if args.psf else os.path.join(run_dir, "step5_input.psf")
    dcd_file = args.dcd if args.dcd else os.path.join(run_dir, "MD_Aligned.dcd")
    if not os.path.isabs(psf_file): psf_file = os.path.join(run_dir, psf_file)
    if not os.path.isabs(dcd_file): dcd_file = os.path.join(run_dir, dcd_file)
    if not os.path.exists(psf_file) or not os.path.exists(dcd_file): logger.error("PSF or DCD not found."); db_conn.close(); sys.exit(1)
    universe = None
    try:
        logger.info(f"Loading Universe: PSF={psf_file}, DCD={dcd_file}")
        universe = mda.Universe(psf_file, dcd_file)
        if 0 <= args.frame < len(universe.trajectory): universe.trajectory[args.frame]
        else: logger.error(f"Invalid frame {args.frame}"); universe.trajectory[0]
    except Exception as e: logger.error(f"Failed to load Universe: {e}", exc_info=True); db_conn.close(); sys.exit(1)

    # --- Get the list of AtomGroups to merge ---
    logger.info("--- Calling select_atomgroups_for_merge ---")
    atom_groups_to_merge = select_atomgroups_for_merge(
        universe, filter_res_map, args.n_upstream, args.n_downstream
    )

    # --- Perform the merge and check type/len ---
    if atom_groups_to_merge:
        logger.info(f"Got {len(atom_groups_to_merge)} AtomGroups. Checking types and attempting merge...")
        all_ags_valid = True
        for i, ag in enumerate(atom_groups_to_merge):
            if not isinstance(ag, mda.AtomGroup):
                logger.error(f"Item {i} in list is not an AtomGroup! Type: {type(ag)}")
                all_ags_valid = False
            else:
                logger.debug(f"  AG {i}: type={type(ag)}, len={len(ag)}")

        if all_ags_valid:
            merged_result = None
            try:
                # Explicitly get the .atoms attribute which should be an AtomGroup for merging
                atom_list_for_merge = [ag.atoms for ag in atom_groups_to_merge]
                logger.info(f"Performing: mda.Merge(*[{len(atom_list_for_merge)} AtomGroups])")
                merged_result = mda.Merge(*atom_list_for_merge)

                logger.info(f"Merge successful. Type of result: {type(merged_result)}")

                # Now attempt len()
                logger.info("Attempting len(merged_result)...")
                merged_len = len(merged_result)
                logger.info(f"len() successful. Length: {merged_len}")

            except TypeError as te:
                logger.error(f"Caught TypeError during len(merged_result): {te}", exc_info=True)
                logger.error(f"Object type that caused error: {type(merged_result)}")
                logger.error(f"Object representation: {repr(merged_result)}")
            except Exception as e_merge:
                logger.error(f"Caught Exception during mda.Merge or len(): {e_merge}", exc_info=True)
                if merged_result is not None:
                     logger.error(f"Object type after merge (if available): {type(merged_result)}")
        else:
            logger.error("Cannot proceed with merge, invalid items found in the list.")

    elif atom_groups_to_merge is None:
        logger.error("AtomGroup selection failed, merge step skipped.")
    else: # Empty list
        logger.info("AtomGroup selection returned an empty list. Nothing to merge.")

    # --- Cleanup ---
    if db_conn:
        db_conn.close()
        logger.info("Database connection closed.")

if __name__ == "__main__":
    main()
