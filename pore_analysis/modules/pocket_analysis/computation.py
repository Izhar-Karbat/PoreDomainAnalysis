# pore_analysis/modules/pocket_analysis/computation.py
"""
Computation functions for Peripheral Pocket Water Analysis.

Integrates logic for loading data, preparing graph features, running a pre-trained
Equivariant Transformer model, classifying water molecules into pockets,
analyzing occupancy and residence times, calculating imbalance metrics,
saving results, and interacting with the analysis suite database.

Relies on filter residue definitions pre-calculated by the ion_analysis module.
"""

import os
import json
import logging
import time
import pickle
import sqlite3
import sys
from typing import Dict, Optional, Tuple, List, Any
from collections import defaultdict

import numpy as np
import pandas as pd
import MDAnalysis as mda
import torch
import torch.nn as nn
from torch_geometric.data import Data # Assuming Data is used by prepare_data
from scipy.stats import ks_2samp, skew # For metrics
from tqdm import tqdm


# --- Core Suite Imports ---
try:
    # <<< --- MODIFIED IMPORTS --- >>>
    from pore_analysis.core.utils import frames_to_time, OneLetter, clean_json_data # Use OneLetter
    from pore_analysis.core.database import (
        register_module, update_module_status, register_product, store_metric,
        get_simulation_metadata, get_config_parameters
    )
    from pore_analysis.core.logging import setup_system_logger
    # Import necessary config values using defaults from core.config as fallback
    from pore_analysis.core import config as core_config # Import config module
    # <<< --- END MODIFIED IMPORTS --- >>>
    CORE_AVAILABLE = True
except ImportError as e:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.error(f"Critical Import Error - Core modules missing: {e}. Using dummy functions/defaults.")
    CORE_AVAILABLE = False
    # Define dummy DB/core functions if needed for basic loading
    def register_module(*args, **kwargs): pass
    def update_module_status(*args, **kwargs): pass
    def register_product(*args, **kwargs): pass
    def store_metric(*args, **kwargs): pass
    def get_simulation_metadata(*args, **kwargs): return None
    def get_config_parameters(*args, **kwargs): return {}
    def setup_system_logger(*args, **kwargs): return logging.getLogger(__name__)
    def frames_to_time(frames): return np.array(frames) * 0.1
    def OneLetter(x): return x # Dummy
    def clean_json_data(data): return data
    # Add dummies for core_config attributes if needed later
    class DummyConfig: pass
    core_config = DummyConfig()
    # Set default values directly if core_config cannot be imported
    setattr(core_config, 'POCKET_ANALYSIS_FILTER_SEQUENCE', "GYG")
    setattr(core_config, 'POCKET_ANALYSIS_N_UPSTREAM', 10)
    setattr(core_config, 'POCKET_ANALYSIS_N_DOWNSTREAM', 10)
    setattr(core_config, 'POCKET_ANALYSIS_CYLINDER_RADIUS', 15.0)
    setattr(core_config, 'POCKET_ANALYSIS_INITIAL_HEIGHT', 20.0)
    setattr(core_config, 'POCKET_ANALYSIS_MIN_WATERS', 100)
    setattr(core_config, 'POCKET_ANALYSIS_RMSF_WINDOW', 10)
    setattr(core_config, 'POCKET_ANALYSIS_RESIDENCE_THRESHOLD', 10)
    setattr(core_config, 'POCKET_ANALYSIS_TRAJECTORY_TOLERANCE', 5)
    setattr(core_config, 'POCKET_ANALYSIS_MODEL_CONFIG_RELPATH', "ml_model/pocket_model_config.json")
    setattr(core_config, 'POCKET_ANALYSIS_MODEL_WEIGHTS_RELPATH', "ml_model/pocket_model.pth")
    setattr(core_config, 'POCKET_ANALYSIS_SHORT_LIVED_THRESH_NS', 5.0)
    setattr(core_config, 'POCKET_ANALYSIS_LONG_LIVED_THRESH_NS', 10.0)
    setattr(core_config, 'FRAMES_PER_NS', 10.0)


# --- ML/Dependency Imports ---
try:
    from torchmdnet.models.torchmd_et import TorchMD_ET
except ImportError as e_tmd:
    # Don't create a dummy class, as this causes more issues than it solves
    raise ImportError(f"Failed to import TorchMD_ET from torchmdnet: {e_tmd}. The torchmd-net package must be installed for pocket analysis to work.")

logger = logging.getLogger(__name__)

# --- Dependency Integration: Copied/Adapted from Train_ET.py ---

# Define a mapping from atom names to atomic numbers
atomic_numbers = {
    'H': 1, 'C': 6, 'N': 7, 'O': 8, 'P': 15, 'S': 16, 'K': 19, 'Cl': 17,
}

def get_atomic_numbers(atom_group):
    """
    Get atomic numbers for an AtomGroup or similar structure.
    Enhanced to handle various MDAnalysis objects and edge cases.
    
    Args:
        atom_group: An MDAnalysis AtomGroup, Universe atoms, or iterable of atom-like objects
                   that have a 'name' attribute for element lookup.
    
    Returns:
        torch.Tensor: Tensor of atomic numbers (Z values) for each atom
    
    Raises:
        TypeError: If input isn't iterable or doesn't contain valid atom-like objects
        ValueError: If elements can't be determined from atom names
    """
    logger = logging.getLogger(__name__)
    z = []
    
    # --- Handle empty cases ---
    if atom_group is None:
        logger.warning("Received None as atom_group. Returning empty tensor.")
        return torch.tensor([], dtype=torch.long)
    
    # If we have a n_atoms attribute, check for empty
    if hasattr(atom_group, 'n_atoms') and atom_group.n_atoms == 0:
        logger.debug("Empty AtomGroup detected (n_atoms=0). Returning empty tensor.")
        return torch.tensor([], dtype=torch.long)
    
    # --- Handle various MDAnalysis object types ---
    # Case 1: Universe - extract the atoms
    if hasattr(atom_group, 'atoms') and not hasattr(atom_group, 'is_atom'):
        logger.debug("Input appears to be a Universe or contains .atoms - using its atoms attribute.")
        atom_group = atom_group.atoms
    
    # Case 2: ResidueGroup - extract atoms
    if hasattr(atom_group, 'residues') and hasattr(atom_group, 'atoms') and not hasattr(atom_group, 'is_atom'):
        logger.debug("Input appears to be a ResidueGroup - using its atoms attribute.")
        atom_group = atom_group.atoms
    
    # Case 3: Single Atom object - wrap in list to make iterable
    if hasattr(atom_group, 'is_atom') and atom_group.is_atom:
        logger.debug("Input is a single Atom object - wrapping in list.")
        atom_group = [atom_group]
    
    # --- Check if result is iterable ---
    try:
        # For safety, convert to list if it's an iterable but not a list/tuple already
        if not isinstance(atom_group, (list, tuple)) and hasattr(atom_group, '__iter__'):
            atom_iter = list(atom_group)
        else:
            atom_iter = atom_group
        
        # Double check with explicit iteration attempt
        iter(atom_iter)
    except TypeError:
        error_msg = f"Input 'atom_group' of type {type(atom_group)} is not iterable as required."
        logger.error(error_msg)
        raise TypeError(error_msg)
    
    # --- Process each atom ---
    for atom in atom_iter:
        try:
            # Standard case: Atom has a 'name' attribute with element info
            if hasattr(atom, 'name'):
                # Get element from atom name, cleaning non-alpha characters
                element = ''.join(filter(str.isalpha, atom.name))
                # Use first character as element abbreviation (most common case)
                element_upper = element[:1].upper()
                
                # For two-letter elements like Cl, try to detect from name
                if len(element) > 1 and element[:2].lower() in ['cl', 'na', 'mg', 'ca', 'fe', 'cu', 'zn']:
                    potential_element = element[:2]
                    if potential_element.title() in atomic_numbers:
                        element_upper = potential_element.title()
                
                # Lookup the element in our table
                if element_upper in atomic_numbers:
                    z.append(atomic_numbers[element_upper])
                elif hasattr(atom, 'element') and atom.element in atomic_numbers:
                    # Some versions of MDAnalysis have explicit element attribute
                    z.append(atomic_numbers[atom.element])
                else:
                    logger.warning(f"Unknown element derived from atom name: '{atom.name}' -> '{element_upper}'. Using 'C' as default.")
                    # Default to carbon (6) for protein-like contexts
                    z.append(atomic_numbers['C'])
            else:
                # Fallback for objects without name attribute
                logger.warning(f"Object {atom} of type {type(atom)} has no 'name' attribute. Using 'O' as default.")
                # Default to oxygen (8) for water-like contexts
                z.append(atomic_numbers['O'])
        except (AttributeError, IndexError, TypeError) as e:
            # Last resort fallback for completely unexpected object types
            logger.warning(f"Error processing atom {atom}: {e}. Using 'O' as default.")
            z.append(atomic_numbers['O'])
    
    # Return tensor of atomic numbers
    return torch.tensor(z, dtype=torch.long)

# find_selectivity_filter is NOT copied - use ion_analysis version via DB metadata

def select_tetramer_backbone_atoms(
    universe: mda.Universe,
    filter_res_map: Dict[str, List[int]],
    n_upstream: int,
    n_downstream: int
) -> Tuple[Optional[mda.AtomGroup], Optional[List[int]], Optional[List[int]]]:
    """
    Select backbone atoms relative to filter for all available chains.
    Revised to find indices by iterating over the ResidueGroup.
    """
    selected_atoms_list = []
    subunit_ids = []
    atom_indices = []
    chain_segids = sorted(list(filter_res_map.keys()))

    logger.debug(f"Selecting backbone atoms for chains {chain_segids} using filter map: {filter_res_map}")

    for subunit_id, segid in enumerate(chain_segids):
        logger.debug(f"Processing subunit {segid} (Internal ID: {subunit_id})...")
        filter_resids = filter_res_map.get(segid)
        if not filter_resids or len(filter_resids) != 5:
            logger.warning(f"Invalid or missing filter resids list for {segid}: {filter_resids}. Skipping.")
            continue

        # Select protein atoms for the subunit
        subunit_protein_atoms = universe.select_atoms(f'segid {segid} and protein')
        if len(subunit_protein_atoms) == 0:
            logger.warning(f"No protein atoms found for subunit {segid}, skipping.")
            continue

        # Get the corresponding ResidueGroup - THIS IS THE LIST WE NEED TO INDEX
        subunit_residues = subunit_protein_atoms.residues
        num_subunit_residues = len(subunit_residues)
        logger.debug(f"  Subunit {segid} has {num_subunit_residues} residue objects.")
        if num_subunit_residues == 0:
             logger.warning(f"Subunit {segid} protein selection resulted in 0 residue objects. Skipping.")
             continue

        filter_start_resid = filter_resids[0]
        filter_end_resid = filter_resids[-1]
        logger.debug(f"  Target Filter ResIDs for {segid}: Start={filter_start_resid}, End={filter_end_resid}")

        # --- Find indices by iterating through the ResidueGroup ---
        start_res_index = -1
        end_res_index = -1
        # Create a lookup for faster access if needed, but direct iteration is safer
        # subunit_resid_list = [res.resid for res in subunit_residues] # Optional: Get list of IDs first
        for i, res in enumerate(subunit_residues):
            if res.resid == filter_start_resid:
                # Found the start, store its index IF it hasn't been found yet
                if start_res_index == -1:
                    start_res_index = i
            if res.resid == filter_end_resid:
                # Found the end, store its index. Assume it appears after the start.
                end_res_index = i
                # Optimization: If start is already found, we can potentially break
                # after finding the end, assuming sequential numbering within the filter.
                # if start_res_index != -1:
                #    break # Break if assuming sequential filter

        # Check if filter residues were found within this subunit's ResidueGroup
        if start_res_index == -1:
            logger.warning(f"  Start filter resid {filter_start_resid} NOT FOUND by iterating {segid} residue objects.")
            continue
        if end_res_index == -1:
            logger.warning(f"  End filter resid {filter_end_resid} NOT FOUND by iterating {segid} residue objects.")
            continue
        # Ensure start index is not after end index (sanity check)
        if start_res_index > end_res_index:
             logger.warning(f"  Inconsistency found: start_res_index ({start_res_index}) > end_res_index ({end_res_index}) for {segid} residue objects. Skipping.")
             continue

        logger.debug(f"  Found start_res_index={start_res_index}, end_res_index={end_res_index} for {segid} (indices within subunit_residues list of length {num_subunit_residues})")

        # Calculate slice indices based on the found *list indices*
        select_start_index = max(0, start_res_index - n_upstream)
        # Use the correct length: num_subunit_residues
        select_end_index = min(num_subunit_residues, end_res_index + 1 + n_downstream)
        logger.debug(f"  Calculated slice indices: select_start_index={select_start_index}, select_end_index={select_end_index}")

        # Check if the calculated slice indices are valid
        if select_start_index >= select_end_index:
            logger.warning(f"Invalid residue slice indices calculated for {segid} (start={select_start_index}, end={select_end_index}). Skipping.")
            continue

        # Get the residue IDs at the slice boundaries using the correct indices
        try:
            # Access the ResidueGroup using the calculated slice indices
            start_select_resid = subunit_residues[select_start_index].resid
            # End index for slicing MDAnalysis selection string should be inclusive
            if select_end_index > 0:
                # Get the residue object at the last index to be *included* in the slice
                end_select_resid = subunit_residues[select_end_index - 1].resid
            else:
                 logger.warning(f"Calculated end selection index is invalid ({select_end_index}). Cannot determine end residue for {segid}.")
                 continue

            logger.debug(f"  Selecting backbone for {segid} between resids {start_select_resid} and {end_select_resid}")

            # Perform the final atom selection using the slice *residue IDs* on the original subunit atom group
            backbone_selection = subunit_protein_atoms.select_atoms(f'resid {start_select_resid}:{end_select_resid} and backbone')

            if len(backbone_selection) > 0:
                logger.debug(f"    Selected {len(backbone_selection)} backbone atoms for {segid}.")
                selected_atoms_list.append(backbone_selection)
                subunit_ids.extend([subunit_id] * len(backbone_selection))
                atom_indices.extend(backbone_selection.indices)
            else:
                logger.warning(f"Backbone selection yielded 0 atoms for {segid} resid {start_select_resid}:{end_select_resid}.")

        except IndexError as e_idx:
             # This error means select_start_index or select_end_index-1 was out of bounds for subunit_residues list
             logger.error(f"IndexError accessing subunit_residues list for {segid} with calculated indices start={select_start_index}, end-1={select_end_index-1}. Check lengths and indices. Error: {e_idx}")
             continue

    # --- Post-Loop Processing ---
    if not selected_atoms_list:
        logger.error("Backbone selection failed for ALL subunits.")
        raise ValueError("Backbone selection failed for ALL subunits.")

    try:
        valid_ags = [ag for ag in selected_atoms_list if isinstance(ag, mda.AtomGroup) and len(ag) > 0]
        if not valid_ags:
             logger.error("No valid AtomGroups to merge after selection.")
             raise ValueError("No valid backbone AtomGroups found after selection.")

        logger.debug(f"Attempting to merge {len(valid_ags)} AtomGroups")
        for i, ag in enumerate(valid_ags):
            logger.debug(f"  AG {i}: type={type(ag)}, len={len(ag)}")

        # Perform the merge directly on the list of AtomGroups
        logger.info(f"Performing: mda.Merge(*[{len(valid_ags)} AtomGroups])")
        merged_object = mda.Merge(*valid_ags)  # Pass the list of AGs directly

        logger.info(f"Result type after mda.Merge: {type(merged_object)}")

        # --- ADAPTATION: Handle Universe result ---
        selected_atoms_final = None
        if isinstance(merged_object, mda.Universe):
            logger.warning("mda.Merge returned a Universe. Selecting all atoms from it to get an AtomGroup.")
            # Select all atoms from the newly created Universe
            selected_atoms_final = merged_object.atoms
            logger.info(f"Selected all atoms from merged Universe. Final type: {type(selected_atoms_final)}, Length: {len(selected_atoms_final)}")
        elif isinstance(merged_object, mda.AtomGroup):
            logger.info("mda.Merge returned an AtomGroup as expected.")
            selected_atoms_final = merged_object
        else:
            # This case should ideally not happen if Merge returns Universe or AtomGroup
            logger.error(f"mda.Merge returned an unexpected type: {type(merged_object)}")
            raise ValueError(f"mda.Merge returned unexpected type: {type(merged_object)}")
        # --- END ADAPTATION ---

        # Check length of the final AtomGroup
        if selected_atoms_final is None:
             raise ValueError("Failed to obtain a final AtomGroup after merge.")

        logger.info(f"Successfully selected and merged backbone atoms for {len(valid_ags)} subunits. Total atoms: {len(selected_atoms_final)}")
        # Return the final AtomGroup
        return selected_atoms_final, subunit_ids, atom_indices  # subunit_ids/atom_indices were collected in the loop

    except ValueError as ve:  # Catch ValueErrors raised within the try block
        logger.error(f"ValueError during post-loop processing: {ve}")
        raise  # Re-raise the ValueError
    except Exception as e_merge:
        logger.error(f"Error during post-loop processing (merge/selection): {e_merge}", exc_info=True)
        # Raise a ValueError to indicate failure
        raise ValueError(f"Error merging/processing backbone AtomGroups: {e_merge}")


def select_water_molecules_trajectory(universe, frames_mask, cylinder_radius, initial_height, min_waters, filter_res_map):
    """Selects water molecules within a cylinder defined relative to filter Tyr CAs."""
    # Use a robust water selection string
    water_selection_string = "name OH2 or type OW or ((resname TIP3 or resname WAT or resname HOH) and (name O or name OW))"
    water_oxygen = universe.select_atoms(water_selection_string)
    if len(water_oxygen) == 0:
        raise ValueError(f"No water oxygen atoms found using selection: '{water_selection_string}'.")

    n_frames_to_process = len(frames_mask)
    selected_waters_mask = np.zeros((n_frames_to_process, len(water_oxygen)), dtype=bool)

    # --- Get Tyr CA atoms using filter_res_map ---
    tyr_ca_atoms = []
    chain_segids = sorted(list(filter_res_map.keys()))
    for segid in chain_segids:
         filter_resids = filter_res_map[segid]
         if len(filter_resids) == 5: # Expect TVGYG
             tyr_resid = filter_resids[3] # Y is index 3
             try:
                 ag = universe.select_atoms(f"segid {segid} and resid {tyr_resid} and name CA")
                 if len(ag) == 1: tyr_ca_atoms.append(ag)
                 else: logger.warning(f"Found {len(ag)} CA atoms for TYR {tyr_resid} in {segid}, expected 1.")
             except Exception as e_sel:
                 logger.warning(f"Error selecting TYR CA for {segid} resid {tyr_resid}: {e_sel}")
         else:
             logger.warning(f"Skipping chain {segid} for cylinder center: Incorrect filter length ({len(filter_resids)}).")

    if len(tyr_ca_atoms) != 4:
        raise ValueError(f"Expected 4 Tyrosine CAs for cylinder center, found {len(tyr_ca_atoms)}.")
    
    # Create a merged Universe with the Tyr CA atoms
    try:
        # Check if tyr_ca_atoms contains valid AtomGroups
        valid_groups = [ag for ag in tyr_ca_atoms if isinstance(ag, mda.core.groups.AtomGroup) and len(ag) > 0]
        if not valid_groups:
            raise ValueError(f"No valid AtomGroups to merge. Found {len(tyr_ca_atoms)} items with types: {[type(ag) for ag in tyr_ca_atoms]}")
        
        # Merge the valid groups
        logger.debug(f"Merging {len(valid_groups)} valid Tyr CA AtomGroups")
        tyr_ca_universe = mda.Merge(*valid_groups)
        
        # Extract the AtomGroup from the merged Universe
        if isinstance(tyr_ca_universe, mda.Universe):
            tyr_ca_group = tyr_ca_universe.atoms
            if not isinstance(tyr_ca_group, mda.core.groups.AtomGroup):
                raise TypeError(f"Merged universe atoms is not an AtomGroup but {type(tyr_ca_group)}")
        else:
            # In some MDAnalysis versions, Merge might return an AtomGroup directly
            tyr_ca_group = tyr_ca_universe
            
        logger.info(f"Created tyr_ca_group with {len(tyr_ca_group)} atoms for calculating cylinder center.")
    except Exception as e:
        logger.error(f"Error creating merged Tyr CA group: {e}")
        raise ValueError(f"Failed to create a valid tyrosine CA atoms group for cylinder center: {str(e)}")
    # --- End Tyr CA selection ---

    # Iterate through trajectory
    for idx, frame_index in enumerate(tqdm(frames_mask, desc="Selecting Water Molecules", unit="frame")):
        ts = universe.trajectory[frame_index]
        
        # Calculate COM for the frame with proper error handling
        try:
            # First try the standard center_of_mass method
            cylinder_center = tyr_ca_group.center_of_mass()
        except (AttributeError, ValueError, TypeError) as e:
            logger.warning(f"Failed to use center_of_mass method: {e}. Falling back to manual calculation.")
            # Fallback: Manual calculation using positions
            try:
                # Update frame if needed before accessing positions
                if hasattr(tyr_ca_group, 'universe') and hasattr(tyr_ca_group.universe, 'trajectory'):
                    logger.debug(f"Ensuring positions are updated for frame {frame_index}")
                    tyr_ca_group.universe.trajectory[frame_index]
                
                # Manual center of mass
                if hasattr(tyr_ca_group, 'positions') and len(tyr_ca_group.positions) > 0:
                    cylinder_center = np.mean(tyr_ca_group.positions, axis=0)
                else:
                    raise ValueError(f"Cannot access positions for tyr_ca_group")
            except Exception as e2:
                logger.error(f"Both center_of_mass and fallback failed: {e2}")
                # Last resort: Use Z-axis center of frame, XY from water molecules if available
                logger.warning("Using last resort cylinder center calculation")
                if len(water_oxygen) > 0:
                    xy_center = np.mean(water_oxygen.positions[:, :2], axis=0)
                    z_center = universe.dimensions[2] / 2
                    cylinder_center = np.array([xy_center[0], xy_center[1], z_center])
                else:
                    # Default position using box center
                    cylinder_center = universe.dimensions[:3] / 2
        
        water_positions = water_oxygen.positions

        # Calculate distances in XY plane
        xy_distances = np.sqrt(np.sum((water_positions[:, :2] - cylinder_center[:2])**2, axis=1))
        in_cylinder_radius = xy_distances <= cylinder_radius

        # Expand height if needed
        height = initial_height
        while True:
            in_cylinder_height = np.abs(water_positions[:, 2] - cylinder_center[2]) <= height/2
            in_cylinder = in_cylinder_radius & in_cylinder_height
            if np.sum(in_cylinder) >= min_waters: break
            else: height += 0.5
            if height > 50: # Safety break
                 logger.warning(f"Cylinder height exceeded 50A at frame {frame_index} while searching for {min_waters} waters. Using current selection ({np.sum(in_cylinder)} waters).")
                 break

        selected_waters_mask[idx] = in_cylinder

    return selected_waters_mask, water_oxygen

def compute_rmsf_x_dimension(universe, water_oxygen, frames_mask, window_size):
    """Compute RMSF for the x-dimension of water oxygens."""
    n_selected_frames = len(frames_mask)
    n_waters = len(water_oxygen)
    if n_waters == 0: return np.array([]).reshape(n_selected_frames, 0) # Handle no waters case

    water_positions_x = np.full((n_selected_frames, n_waters), np.nan) # Initialize with NaN
    rmsf = np.full((n_selected_frames, n_waters), np.nan)

    # Collect X positions
    for idx, frame_index in enumerate(tqdm(frames_mask, desc="Collecting water X positions", unit="frame", leave=False)):
        ts = universe.trajectory[frame_index]
        water_positions_x[idx] = water_oxygen.positions[:, 0]

    # Compute rolling RMSF using pandas for efficient NaN handling
    df_pos_x = pd.DataFrame(water_positions_x)
    # Calculate rolling mean and variance (std squared) - handle NaNs
    rolling_mean = df_pos_x.rolling(window=window_size, min_periods=1).mean()
    rolling_sq_diff = (df_pos_x - rolling_mean)**2
    rolling_var = rolling_sq_diff.rolling(window=window_size, min_periods=1).mean()
    rmsf = np.sqrt(rolling_var).values

    # Handle potential NaNs resulting from rolling operations if needed (e.g., fill forward/backward or keep NaN)
    # For RMSF, NaN might be appropriate if not enough data in window
    logger.debug(f"RMSF computation complete. Shape: {rmsf.shape}")
    return rmsf


def prepare_data(universe, labels_dict, frames_mask, filter_res_map, # Pass filter map
                 n_upstream, n_downstream, cylinder_radius, initial_height,
                 min_waters, window_size):
    """Prepare graph data for the ET model."""
    logger = logging.getLogger(__name__)
    logger.info("Preparing data for ET model...")

    # Get backbone atoms based on filter map
    selected_backbone, subunit_ids, atom_indices = select_tetramer_backbone_atoms(
        universe, filter_res_map, n_upstream, n_downstream # Pass map here
    )

    # Get water selection mask and water oxygen group
    selected_waters_mask, water_oxygen = select_water_molecules_trajectory(
        universe, frames_mask, cylinder_radius, initial_height, min_waters, filter_res_map # Pass map here too
    )

    # Compute RMSF
    rmsf_values = compute_rmsf_x_dimension(universe, water_oxygen, frames_mask, window_size)

    data_list = []
    # Iterate using the frames_mask for correct indexing
    for idx, frame_index in enumerate(tqdm(frames_mask, desc="Creating Graph Data", unit="frame")):
        ts = universe.trajectory[frame_index] # Access correct frame

        backbone_z = get_atomic_numbers(selected_backbone)
        backbone_pos = torch.tensor(selected_backbone.positions, dtype=torch.float32)

        # Select waters for the current frame based on the precomputed mask
        current_water_mask = selected_waters_mask[idx]
        current_water_oxygen = water_oxygen[current_water_mask]
        if len(current_water_oxygen) == 0: # Skip frame if no waters selected
             logger.debug(f"Skipping frame {frame_index}: No water molecules selected.")
             # Add an empty Data object or handle differently? For now, skip.
             continue

        water_z = get_atomic_numbers(current_water_oxygen)
        water_pos = torch.tensor(current_water_oxygen.positions, dtype=torch.float32)

        z = torch.cat([backbone_z, water_z])
        pos = torch.cat([backbone_pos, water_pos])

        # Map subunit IDs correctly for backbone atoms
        current_subunit_ids = torch.tensor(subunit_ids, dtype=torch.long)
        # Assign -1 to all water molecules
        water_subunit_ids = torch.full((len(water_pos),), -1, dtype=torch.long)
        subunit_ids_frame = torch.cat([current_subunit_ids, water_subunit_ids])

        batch = torch.zeros(len(z), dtype=torch.long)

        # Get atom indices for backbone and selected waters
        current_atom_indices = torch.tensor(atom_indices + list(current_water_oxygen.indices), dtype=torch.long)

        # Get RMSF values for selected waters in this frame
        current_rmsf = torch.tensor(rmsf_values[idx][current_water_mask], dtype=torch.float32)
        # Assign -1 RMSF to backbone atoms
        backbone_rmsf = torch.full((len(backbone_pos),), -1.0, dtype=torch.float32)
        rmsf_frame = torch.cat([backbone_rmsf, current_rmsf])

        # Initialize labels: -1 (unlabeled)
        labels = torch.full((len(z),), -1, dtype=torch.long)
        # Add logic here if you have actual labels for training/evaluation
        # Currently, pocket assignment happens *after* prediction in the original script

        is_water = torch.cat([torch.zeros(len(backbone_pos), dtype=torch.bool),
                              torch.ones(len(water_pos), dtype=torch.bool)])

        data = Data(pos=pos, z=z, batch=batch, subunit_ids=subunit_ids_frame,
                    atom_indices=current_atom_indices, rmsf=rmsf_frame,
                    labels=labels, is_water=is_water)
        data_list.append(data)

    logger.info(f"Prepared {len(data_list)} graph data objects.")
    return data_list


# Custom Model Class Definition (Copied from Train_ET.py)
class CustomTorchMD_ET(TorchMD_ET):
    def __init__(self, num_subunits, subunit_embedding_dim, rmsf_embedding_dim, label_embedding_dim, hidden_channels, num_layers, num_heads, num_rbf, rbf_type, activation, attn_activation, neighbor_embedding, cutoff_lower, cutoff_upper, max_z, max_num_neighbors, *args, **kwargs):
        # Important: Pass only the parameters that TorchMD_ET expects
        super().__init__(
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            num_heads=num_heads,
            num_rbf=num_rbf,
            rbf_type=rbf_type,
            activation=activation,
            attn_activation=attn_activation,
            neighbor_embedding=neighbor_embedding,
            cutoff_lower=cutoff_lower,
            cutoff_upper=cutoff_upper,
            max_z=max_z,
            max_num_neighbors=max_num_neighbors
        )

        # Explicitly store hidden_channels (don't rely on parent class)
        self.hidden_channels = hidden_channels

        # Initialize custom layers
        self.subunit_embedding = nn.Embedding(num_subunits, subunit_embedding_dim)
        self.rmsf_embedding = nn.Sequential(
            nn.Linear(1, rmsf_embedding_dim),
            nn.ReLU(),
            nn.Linear(rmsf_embedding_dim, rmsf_embedding_dim)
        )
        self.label_embedding = nn.Embedding(2, label_embedding_dim)

        self.subunit_embedding_dim = subunit_embedding_dim
        self.rmsf_embedding_dim = rmsf_embedding_dim
        self.label_embedding_dim = label_embedding_dim

        # Calculate feature_dim exactly as in the original script
        feature_dim = max(self.subunit_embedding_dim, self.rmsf_embedding_dim, self.label_embedding_dim)
        
        # Use the stored hidden_channels value
        classifier_input_dim = self.hidden_channels + feature_dim
        
        # Classifier setup - ensure it matches the trained model dimensions
        # Based on the error message, the trained model has:
        # classifier.0: Linear(in=160, out=64)
        # classifier.3: Linear(in=64, out=1)
        half_hidden = self.hidden_channels // 2
        
        # Allow flexible classifier dimensions for compatibility with different model versions
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, half_hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(half_hidden, 1)
        )

    # Modify forward pass to NOT use label_embedding during inference
    def forward(self, z, pos, batch, subunit_ids, atom_indices, is_water, rmsf, labels=None): # Made labels optional
        # Note: TorchMD_ET's forward returns: x, vec, z, pos, batch
        x, _, _, _, _ = super().forward(z, pos, batch) # Call parent forward pass

        # IMPORTANT: Must match the feature_dim used in __init__ to create classifier
        # The model was trained with label_embedding_dim included in the feature_dim calculation
        feature_dim = max(self.subunit_embedding_dim, self.rmsf_embedding_dim, self.label_embedding_dim)
        features = torch.zeros(z.shape[0], feature_dim, device=z.device)

        non_water_mask = ~is_water
        water_mask = is_water

        # Subunit embeddings for non-water atoms
        if torch.any(non_water_mask):
            valid_subunit_ids = subunit_ids[non_water_mask]
            # Clamp indices to be safe, though they should be correct
            valid_subunit_ids = torch.clamp(valid_subunit_ids, 0, self.subunit_embedding.num_embeddings - 1)
            features[non_water_mask, :self.subunit_embedding_dim] = self.subunit_embedding(valid_subunit_ids)

        # RMSF embeddings for water atoms
        if torch.any(water_mask):
            rmsf_water = rmsf[water_mask].unsqueeze(1)
            # Handle potential NaNs or Infs in RMSF before embedding
            rmsf_water = torch.nan_to_num(rmsf_water, nan=0.0, posinf=10.0, neginf=-10.0) # Replace with 0 or clamp large values
            rmsf_embeddings = self.rmsf_embedding(rmsf_water)
            features[water_mask, :self.rmsf_embedding_dim] = rmsf_embeddings

        # Concatenate ET output with features
        x = torch.cat([x, features], dim=-1)

        # Apply classifier ONLY to water molecules
        logits = torch.zeros(z.shape[0], 1, device=z.device) # Initialize logits for all atoms
        if torch.any(water_mask):
             logits[water_mask] = self.classifier(x[water_mask])

        # Return logits, is_water mask, and original labels (if passed)
        return logits, is_water, labels


# --- End of Copied/Adapted Code from Train_ET.py ---


# --- Core Logic Functions (Adapted from Standalone Script) ---

def load_model_config(config_path):
    """Loads ML model configuration JSON."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Model config file not found: {config_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Error decoding JSON from model config file: {config_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading model config {config_path}: {e}") from e

def load_analysis_model(model_path, model_config, device):
    """
    Load the trained CustomTorchMD_ET model.

    Args:
        model_path (str): Path to the saved model file.
        model_config (dict): Configuration dictionary containing model parameters.
        device (torch.device): Device to load the model onto.

    Returns:
        CustomTorchMD_ET: Loaded model.
    """
    # Add safe globals for numpy components to avoid security restrictions in PyTorch 2.6+
    import torch.serialization
    torch.serialization.add_safe_globals(['numpy._core.multiarray.scalar'])
    
    # Load the checkpoint - allow weights_only=False since this is our own trusted checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Determine num_subunits from the config or use a default value
    num_subunits = model_config.get('num_subunits', 4)  # Default to 4 if not specified

    # Initialize the model with config values
    model = CustomTorchMD_ET(
        num_subunits=num_subunits,
        subunit_embedding_dim=model_config['subunit_embedding_dim'],
        rmsf_embedding_dim=model_config['rmsf_embedding_dim'],
        label_embedding_dim=model_config['label_embedding_dim'],
        hidden_channels=model_config['hidden_channels'],
        num_layers=model_config['num_layers'],
        num_heads=model_config['num_heads'],
        num_rbf=model_config['num_rbf'],
        rbf_type=model_config['rbf_type'],
        activation=model_config['activation'],
        attn_activation=model_config['attn_activation'],
        neighbor_embedding=model_config['neighbor_embedding'],
        cutoff_lower=model_config['cutoff_lower'],
        cutoff_upper=model_config['cutoff_upper'],
        max_z=model_config['max_z'],
        max_num_neighbors=model_config['max_num_neighbors'],
        dtype=torch.float32
    ).to(device)

    # Load the state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    # Set the model to evaluation mode
    model.eval()

    logger.info(f"Model loaded successfully. Number of subunits: {num_subunits}")
    return model

def predict_and_assign_pockets(frame_data, model, device, filter_residues_map, universe):
    """Predicts water states and assigns to nearest subunit's filter for one frame."""
    pocket_assignments = {0: [], 1: [], 2: [], 3: []} # Pockets A, B, C, D
    positive_water_indices = [] # Indices of waters predicted positive

    # Ensure filter_residues_map is valid
    if not filter_residues_map or len(filter_residues_map) != 4:
         # Log error but maybe continue without assignments? Or raise error?
         logging.getLogger(__name__).error("Invalid filter_residues_map for pocket assignment.")
         return pocket_assignments, positive_water_indices

    try:
        frame_data = frame_data.to(device)
        with torch.no_grad(): # Ensure no gradients are calculated
            logits, is_water, _ = model(
                frame_data.z, frame_data.pos, frame_data.batch,
                frame_data.subunit_ids, frame_data.atom_indices,
                frame_data.is_water, frame_data.rmsf, labels=None # Pass None for labels
            )

        water_mask = is_water.bool()
        if not torch.any(water_mask): # No water atoms in this graph data?
             return pocket_assignments, positive_water_indices

        water_logits = logits[water_mask].squeeze(-1)
        probabilities = torch.sigmoid(water_logits)
        predictions = probabilities > 0.5 # Standard threshold

        # Get indices and positions of waters predicted as positive (in pocket)
        positive_water_indices = frame_data.atom_indices[water_mask][predictions].cpu().numpy()
        positive_water_pos = frame_data.pos[water_mask][predictions].cpu() # Move to CPU for distance calc

        # Get filter CA positions (approximate filter center per subunit)
        # Assumes filter_residues_map keys are sorted/consistent PROA, PROB, PROC, PROD or A, B, C, D
        subunit_centers = []
        for subunit_id, segid in enumerate(sorted(filter_residues_map.keys())):
            resids = filter_residues_map[segid]
            if len(resids) == 5: # TVGYG
                gyg_resids = resids[2:5] # G, Y, G
                # Select CA atoms of these residues
                ca_sel_str = f"segid {segid} and resid {' '.join(map(str, gyg_resids))} and name CA"
                ca_atoms = frame_data.pos[torch.isin(frame_data.atom_indices, torch.tensor(universe.select_atoms(ca_sel_str).indices, device=device))]
                # Select backbone atoms of these residues (alternative center?)
                # bb_sel_str = f"segid {segid} and resid {' '.join(map(str, gyg_resids))} and backbone"
                # bb_atoms = frame_data.pos[torch.isin(frame_data.atom_indices, torch.tensor(universe.select_atoms(bb_sel_str).indices, device=device))]

                if len(ca_atoms) > 0:
                     subunit_centers.append(torch.mean(ca_atoms, dim=0).cpu()) # Use CA atoms center
                else:
                     # Fallback: use stored subunit ID (less accurate geom center)
                     # Find center of all atoms belonging to this subunit in the graph
                     subunit_mask = (frame_data.subunit_ids == subunit_id) & (~frame_data.is_water)
                     if torch.any(subunit_mask):
                         subunit_centers.append(torch.mean(frame_data.pos[subunit_mask], dim=0).cpu())
                     else:
                         subunit_centers.append(None) # Cannot determine center
                         logging.getLogger(__name__).warning(f"Could not determine center for subunit {segid}")
            else:
                 subunit_centers.append(None) # Incorrect filter length
                 logging.getLogger(__name__).warning(f"Incorrect filter length for subunit {segid}")

        # Assign positive waters to nearest valid subunit center
        for water_idx, water_pos_tensor in zip(positive_water_indices, positive_water_pos):
            min_dist = float('inf')
            nearest_pocket = -1
            for pocket_idx, center_tensor in enumerate(subunit_centers):
                if center_tensor is not None:
                     dist = torch.norm(water_pos_tensor - center_tensor)
                     if dist < min_dist:
                         min_dist = dist
                         nearest_pocket = pocket_idx # Pocket index 0, 1, 2, 3

            if nearest_pocket != -1:
                pocket_assignments[nearest_pocket].append(int(water_idx)) # Store original water index

    except Exception as e:
         logging.getLogger(__name__).error(f"Error during prediction/assignment: {e}", exc_info=True)
         # Return empty assignments on error
         pocket_assignments = {0: [], 1: [], 2: [], 3: []}
         positive_water_indices = []

    return pocket_assignments, positive_water_indices

def compute_occupancy_and_residence(data_list, model, device, filter_res_map, residence_threshold, tolerance_window, universe):
    """Computes occupancy and residence times using the ML model."""
    logger = logging.getLogger(__name__)
    logger.info("Computing pocket occupancy and residence times using ML model...")
    num_frames = len(data_list)
    # Dictionary to store pocket assignment for each water molecule over time
    # Key: water_idx, Value: numpy array of length num_frames, -1 if not assigned, 0-3 if assigned
    water_trajectories = defaultdict(lambda: np.full(num_frames, -1, dtype=int))
    # Store indices predicted positive in each frame (for debugging/verification)
    positive_waters_per_frame = []
    
    # Create the pockets_dict that maps: frame_idx -> pocket_idx -> [water_idx1, water_idx2, ...]
    # This dictionary is used for visualization and detailed analysis
    pockets_dict = {frame_idx: {p: [] for p in range(4)} for frame_idx in range(num_frames)}

    all_water_indices_seen = set() # Track all water indices present in the data

    # 1. Predict and store assignments for all frames
    logger.info("Phase 1: Predicting pocket assignments per frame...")
    for frame_idx, frame_data in enumerate(tqdm(data_list, desc="Predicting pockets", unit="frame")):
        # Update set of all water indices seen in the graphs
        all_water_indices_seen.update(frame_data.atom_indices[frame_data.is_water.bool()].cpu().numpy())

        assignments, positive_indices = predict_and_assign_pockets(frame_data, model, device, filter_res_map, universe)
        positive_waters_per_frame.append(set(positive_indices))
        for pocket_idx, water_indices in assignments.items():
            # Update the pockets_dict for this frame
            pockets_dict[frame_idx][pocket_idx] = list(water_indices)
            # Update the water trajectories
            for water_idx in water_indices:
                water_trajectories[water_idx][frame_idx] = pocket_idx

    logger.info(f"Phase 1 complete. Processed {num_frames} frames. Seen {len(all_water_indices_seen)} unique water indices.")

    # 2. Process trajectories: Smooth gaps and calculate residence times
    logger.info("Phase 2: Smoothing trajectories and calculating residence times...")
    residence_times_frames = defaultdict(lambda: defaultdict(list)) # {water_idx: {pocket_idx: [duration1, duration2,...]}}
    final_occupancy = np.zeros((num_frames, 4), dtype=int) # Frame x Pocket occupancy count

    # Use only water indices that were actually seen in the data
    # Convert to list for predictable iteration order if needed, though defaultdict handles keys dynamically
    relevant_water_indices = sorted(list(all_water_indices_seen))
    logger.debug(f"Processing trajectories for {len(relevant_water_indices)} relevant water molecules.")


    for water_idx in tqdm(relevant_water_indices, desc="Processing trajectories", unit="molecule"):
        trajectory = water_trajectories[water_idx] # Get the trajectory (-1 or 0-3)

        # --- Enhanced Smoothing: Fill gaps and handle noise with three strategies ---
        smoothed_trajectory = np.copy(trajectory)
        if tolerance_window > 0:
            # Pass 1: Fill single-frame gaps surrounded by same pocket ID
            for i in range(1, num_frames - 1):
                if smoothed_trajectory[i] == -1 and smoothed_trajectory[i-1] == smoothed_trajectory[i+1] and smoothed_trajectory[i-1] != -1:
                    smoothed_trajectory[i] = smoothed_trajectory[i-1]
            
            # Pass 2: Fill wider gaps with consensus from surrounding window
            for i in range(num_frames):
                if smoothed_trajectory[i] == -1:  # Found a gap point
                    # Define window boundaries
                    start_back = max(0, i - tolerance_window)
                    end_fwd = min(num_frames, i + tolerance_window + 1)
                    
                    # Get the window content (before and after current position)
                    window_before = smoothed_trajectory[start_back:i]
                    window_after = smoothed_trajectory[i+1:end_fwd]
                    
                    # Skip if one of the windows is empty
                    if len(window_before) == 0 or len(window_after) == 0:
                        continue
                    
                    # Filter out -1 values and count occurrences of each pocket
                    valid_before = window_before[window_before != -1]
                    valid_after = window_after[window_after != -1]
                    
                    # Skip if either window has no valid pockets
                    if len(valid_before) == 0 or len(valid_after) == 0:
                        continue
                    
                    # Find the most common pocket ID before and after (excluding -1)
                    pockets_before, counts_before = np.unique(valid_before, return_counts=True)
                    pockets_after, counts_after = np.unique(valid_after, return_counts=True)
                    
                    # If the most frequent pocket is the same before and after, fill the gap
                    most_common_before = pockets_before[np.argmax(counts_before)]
                    most_common_after = pockets_after[np.argmax(counts_after)]
                    
                    if most_common_before == most_common_after:
                        smoothed_trajectory[i] = most_common_before
            
            # Pass 3: Remove isolated pocket assignments (noise filtering)
            for i in range(1, num_frames - 1):
                if (smoothed_trajectory[i] != -1 and 
                    smoothed_trajectory[i-1] == -1 and 
                    smoothed_trajectory[i+1] == -1):
                    # This is an isolated pocket assignment - remove it
                    # But only if it's a single frame and the surrounding ±2 frames are also not the same pocket
                    isolated = True
                    if i > 1 and smoothed_trajectory[i-2] == smoothed_trajectory[i]:
                        isolated = False
                    if i < num_frames - 2 and smoothed_trajectory[i+2] == smoothed_trajectory[i]:
                        isolated = False
                    
                    if isolated:
                        smoothed_trajectory[i] = -1  # Remove isolated assignment

        # --- Calculate Residence Times from SMOOTHED trajectory ---
        current_pocket = -1
        start_frame = -1
        for frame_idx in range(num_frames):
            pocket = smoothed_trajectory[frame_idx]
            if pocket != current_pocket:
                 # End of a potential residence period
                 if current_pocket != -1 and start_frame != -1: # Was in a pocket
                      duration = frame_idx - start_frame
                      if duration >= residence_threshold:
                           residence_times_frames[water_idx][current_pocket].append(duration)
                 # Start of a new period (or transition from -1)
                 if pocket != -1: # Started a new pocket residence
                      start_frame = frame_idx
                 else: # Moved to outside (-1)
                      start_frame = -1
                 current_pocket = pocket
        # Check for period extending to the end
        if current_pocket != -1 and start_frame != -1:
             duration = num_frames - start_frame
             if duration >= residence_threshold:
                  residence_times_frames[water_idx][current_pocket].append(duration)

        # --- Recalculate Occupancy based on SMOOTHED trajectory ---
        # Only count frames where water is assigned to a pocket AFTER smoothing & residence check
        # (Original script recalculated based on *filtered_trajectories*, which implicitly depended on residence_threshold)
        # Let's calculate based on smoothed path, consistent with residence time calc
        for frame_idx in range(num_frames):
            pocket = smoothed_trajectory[frame_idx]
            if pocket != -1: # Water is assigned to a pocket in the smoothed path
                 # Further filter: Only count if this frame belongs to a residence period >= threshold?
                 # This requires finding the residence block this frame belongs to.
                 # Simpler: Just use the smoothed trajectory for occupancy count.
                 final_occupancy[frame_idx, pocket] += 1

    # --- Format Output ---
    occupancy_df = pd.DataFrame(final_occupancy, columns=[f'Pocket {i}' for i in range(4)])
    # Add Time (ns) column assuming core_config.FRAMES_PER_NS is available
    time_ns = np.arange(num_frames) / core_config.FRAMES_PER_NS
    occupancy_df.insert(0, 'Time (ns)', time_ns)
    occupancy_df.insert(0, 'Frame', np.arange(num_frames)) # Also add Frame number

    # Convert residence times from frames to ns
    residence_times_ns = defaultdict(lambda: defaultdict(list))
    for water_idx, pocket_data in residence_times_frames.items():
        for pocket_idx, durations_frames in pocket_data.items():
             residence_times_ns[water_idx][pocket_idx] = (np.array(durations_frames) / core_config.FRAMES_PER_NS).tolist()

    # Process for summary/plotting (flatten per pocket)
    processed_residence_times_ns = defaultdict(list)
    for water_data in residence_times_ns.values():
        for pocket, times in water_data.items():
             processed_residence_times_ns[pocket].extend(times)

    logger.info("Phase 2 complete.")
    logger.info(f"Generated pockets_dict with {sum(len(pocket_list) for frame_pockets in pockets_dict.values() for pocket_list in frame_pockets.values())} total water assignments")
    return occupancy_df, processed_residence_times_ns, residence_times_ns, pockets_dict # Return ns times and pockets_dict

def process_residence_times_for_metrics(residence_times_ns: Dict[int, Dict[int, List[float]]]) -> Dict[int, List[float]]:
     """Flattens residence times from per-water/per-pocket to per-pocket."""
     processed_times = defaultdict(list)
     for water_data in residence_times_ns.values():
         for pocket, times in water_data.items():
              processed_times[pocket].extend(times)
     return processed_times

def calculate_imbalance_metrics(processed_residence_times_ns: Dict[int, List[float]], occupancy_df: pd.DataFrame):
    """Calculate imbalance metrics."""
    metrics = {}
    pockets = sorted(processed_residence_times_ns.keys()) # Usually 0, 1, 2, 3

    # 1. CV of MEAN Residence Times
    mean_rts = [np.mean(processed_residence_times_ns[pocket]) for pocket in pockets if processed_residence_times_ns[pocket] and len(processed_residence_times_ns[pocket]) > 0]
    metrics['CV_of_Mean_Residence_Times'] = np.std(mean_rts) / np.mean(mean_rts) if mean_rts and len(mean_rts) > 0 else np.nan

    # 2. Gini Coefficient of TOTAL Residence Time (approximates occupancy duration)
    total_time_per_pocket = [np.sum(processed_residence_times_ns.get(pocket, [])) for pocket in pockets]
    if sum(total_time_per_pocket) > 0 and len(total_time_per_pocket) > 1:
         total_time_arr = np.array(total_time_per_pocket)
         mad = np.abs(np.subtract.outer(total_time_arr, total_time_arr)).mean()
         rmad = mad / np.mean(total_time_arr)
         metrics['Gini_Coefficient_TotalTime'] = 0.5 * rmad
    else:
         metrics['Gini_Coefficient_TotalTime'] = np.nan

    # 3. Entropy of TOTAL Residence Time distribution
    total_time_probs = np.array(total_time_per_pocket) / sum(total_time_per_pocket) if sum(total_time_per_pocket) > 0 else []
    if len(total_time_probs) > 0:
         metrics['Entropy_TotalTime'] = -np.sum(total_time_probs * np.log(total_time_probs + 1e-9)) # Add epsilon for log(0)
    else:
         metrics['Entropy_TotalTime'] = np.nan

    # 4. Max Pairwise KS Statistic (on residence time distributions)
    max_ks = 0.0
    ks_pairs = {} # Store pairwise KS stats
    for i in range(len(pockets)):
        for j in range(i + 1, len(pockets)):
            p1, p2 = pockets[i], pockets[j]
            times1 = np.array(processed_residence_times_ns[p1])
            times2 = np.array(processed_residence_times_ns[p2])
            if len(times1) > 1 and len(times2) > 1: # KS test needs samples
                ks_stat, _ = ks_2samp(times1, times2)
                max_ks = max(max_ks, ks_stat)
                # Define consistent key, e.g., PocketWater_KS_A_B
                key_name = f"PocketWater_KS_{chr(ord('A')+p1)}_{chr(ord('A')+p2)}"
                ks_pairs[key_name] = ks_stat
            else: # Store NaN if KS test cannot be run
                 key_name = f"PocketWater_KS_{chr(ord('A')+p1)}_{chr(ord('A')+p2)}"
                 ks_pairs[key_name] = np.nan
    metrics['Max_Pairwise_KS_Statistic'] = max_ks
    metrics.update(ks_pairs) # Add all pairwise stats

    # 5. Normalized Range of MEDIAN Residence Times
    median_rts = [np.median(processed_residence_times_ns[pocket]) for pocket in pockets 
                 if processed_residence_times_ns[pocket] and len(processed_residence_times_ns[pocket]) > 0]
    if len(median_rts) > 1:
        # Create a list of valid time arrays to concatenate
        valid_time_arrays = [np.array(processed_residence_times_ns[p]) for p in pockets 
                            if processed_residence_times_ns[p] and len(processed_residence_times_ns[p]) > 0]
        if valid_time_arrays:
            overall_median = np.median(np.concatenate(valid_time_arrays))
            metrics['Normalized_Range_Median_RTs'] = (max(median_rts) - min(median_rts)) / overall_median if overall_median > 0 else np.nan
        else:
            metrics['Normalized_Range_Median_RTs'] = np.nan
    else:
        metrics['Normalized_Range_Median_RTs'] = np.nan

    # --- Additional Requested Metrics ---

    # 6. Max-to-Min Mean Occupancy Ratio
    # Ensure 'Pocket 0', 'Pocket 1', etc. columns exist
    pocket_cols = [f'Pocket {i}' for i in range(4)]
    if all(col in occupancy_df.columns for col in pocket_cols):
        mean_occs = occupancy_df[pocket_cols].mean()
        min_mean_occ = mean_occs.min()
        max_mean_occ = mean_occs.max()
        metrics['PocketWater_OccupancyRatio'] = max_mean_occ / min_mean_occ if min_mean_occ > 0 else np.inf
    else:
        metrics['PocketWater_OccupancyRatio'] = np.nan

    # 7. Residence Time Distribution Skewness per Pocket
    for pocket in pockets:
        pocket_times = processed_residence_times_ns.get(pocket, [])
        if pocket_times:
            times = np.array(pocket_times)
            if len(times) > 2: # Skew needs > 2 points
                metrics[f"Pocket{chr(ord('A')+pocket)}_RTSkewness"] = skew(times)
            else:
                metrics[f"Pocket{chr(ord('A')+pocket)}_RTSkewness"] = np.nan
        else:
            metrics[f"Pocket{chr(ord('A')+pocket)}_RTSkewness"] = np.nan

    # 8. % Short/Long Lived Waters per Pocket
    short_thresh = core_config.POCKET_ANALYSIS_SHORT_LIVED_THRESH_NS
    long_thresh = core_config.POCKET_ANALYSIS_LONG_LIVED_THRESH_NS
    for pocket in pockets:
        pocket_times = processed_residence_times_ns.get(pocket, [])
        if pocket_times:
            times = np.array(pocket_times)
            total_periods = len(times)
            if total_periods > 0:
                metrics[f"Pocket{chr(ord('A')+pocket)}_ShortLivedPct"] = (np.sum(times < short_thresh) / total_periods) * 100.0
                metrics[f"Pocket{chr(ord('A')+pocket)}_LongLivedPct"] = (np.sum(times > long_thresh) / total_periods) * 100.0
            else:
                metrics[f"Pocket{chr(ord('A')+pocket)}_ShortLivedPct"] = np.nan
                metrics[f"Pocket{chr(ord('A')+pocket)}_LongLivedPct"] = np.nan
        else:
            metrics[f"Pocket{chr(ord('A')+pocket)}_ShortLivedPct"] = np.nan
            metrics[f"Pocket{chr(ord('A')+pocket)}_LongLivedPct"] = np.nan

    return metrics


def summarize_pocket_analysis(occupancy_df, processed_residence_times_ns):
    """Generates a text summary of the analysis."""
    summary = "--- Peripheral Pocket Water Analysis Summary ---\n\n"
    summary += "Water Occupancy Summary (Mean molecules per frame):\n"
    # Ensure pocket columns exist before calculating mean
    pocket_cols = [f'Pocket {i}' for i in range(4)]
    if all(col in occupancy_df.columns for col in pocket_cols):
         mean_occ = occupancy_df[pocket_cols].mean()
         summary += mean_occ.to_string(float_format="%.2f") + "\n"
    else:
         summary += "  Occupancy data columns not found.\n"

    summary += "\nResidence Times Summary (ns):\n"
    for pocket_idx, times in processed_residence_times_ns.items():
        pocket_label = chr(ord('A') + pocket_idx)
        summary += f"\nPocket {pocket_label}:"
        if times:
            times_arr = np.array(times)
            summary += f"\n  Number of periods >= threshold: {len(times_arr)}"
            summary += f"\n  Mean residence time: {np.mean(times_arr):.2f} ns"
            summary += f"\n  Median residence time: {np.median(times_arr):.2f} ns"
            summary += f"\n  Max residence time: {np.max(times_arr):.2f} ns"
        else:
            summary += " No residence periods recorded >= threshold."
    summary += "\n\n--- End of Summary ---"
    return summary

# --- Main Computation Function ---

def run_pocket_analysis(
    run_dir: str,
    psf_file: str,
    dcd_file: str,
    db_conn: sqlite3.Connection
) -> Dict[str, any]:
    """
    Orchestrates the peripheral pocket water analysis computation.
    """
    module_name = "pocket_analysis"
    start_time = time.time()
    register_module(db_conn, module_name, status='running')
    logger_local = setup_system_logger(run_dir)
    if logger_local is None: logger_local = logger # Fallback

    results = {'status': 'failed', 'error': None} # Default status
    output_dir = os.path.join(run_dir, module_name)
    os.makedirs(output_dir, exist_ok=True)

    # --- Load Config and Model ---
    try:
        logger_local.info("Loading configuration and ML model...")
        # Get relative paths from core config
        config_rel_path = getattr(core_config, 'POCKET_ANALYSIS_MODEL_CONFIG_RELPATH', 'ml_model/pocket_model_config.json')
        weights_rel_path = getattr(core_config, 'POCKET_ANALYSIS_MODEL_WEIGHTS_RELPATH', 'ml_model/pocket_model.pth')
        # Construct absolute paths relative to *this* file's location
        module_dir = os.path.dirname(__file__)
        config_abs_path = os.path.abspath(os.path.join(module_dir, config_rel_path))
        weights_abs_path = os.path.abspath(os.path.join(module_dir, weights_rel_path))

        if not os.path.exists(config_abs_path): raise FileNotFoundError(f"Model config not found: {config_abs_path}")
        if not os.path.exists(weights_abs_path): raise FileNotFoundError(f"Model weights not found: {weights_abs_path}")

        model_config = load_model_config(config_abs_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger_local.info(f"Using device: {device}")
        model = load_analysis_model(weights_abs_path, model_config, device)
        logger_local.info("ML Model loaded successfully.")

        # Load analysis parameters from core_config
        filter_seq = getattr(core_config, 'POCKET_ANALYSIS_FILTER_SEQUENCE', "GYG")
        n_upstream = getattr(core_config, 'POCKET_ANALYSIS_N_UPSTREAM', 10)
        n_downstream = getattr(core_config, 'POCKET_ANALYSIS_N_DOWNSTREAM', 10)
        cyl_radius = getattr(core_config, 'POCKET_ANALYSIS_CYLINDER_RADIUS', 15.0)
        init_height = getattr(core_config, 'POCKET_ANALYSIS_INITIAL_HEIGHT', 20.0)
        min_waters = getattr(core_config, 'POCKET_ANALYSIS_MIN_WATERS', 100)
        rmsf_window = getattr(core_config, 'POCKET_ANALYSIS_RMSF_WINDOW', 10)
        res_thresh_frames = getattr(core_config, 'POCKET_ANALYSIS_RESIDENCE_THRESHOLD', 10)
        traj_tolerance = getattr(core_config, 'POCKET_ANALYSIS_TRAJECTORY_TOLERANCE', 5)

    except Exception as e:
        results['error'] = f"Failed to load ML model or config: {e}"
        logger_local.error(results['error'], exc_info=True)
        update_module_status(db_conn, module_name, 'failed', error_message=results['error'])
        return results

    # --- Main Analysis Workflow ---
    try:
        # 1. Load Universe
        logger_local.info("Loading Universe...")
        universe = mda.Universe(psf_file, dcd_file)
        n_frames_total = len(universe.trajectory)
        if n_frames_total < 2: raise ValueError("Trajectory has < 2 frames.")
        # Define frames_mask (currently processes all frames, add slicing later if needed)
        frames_mask = list(range(n_frames_total)) # Process all frames
        logger_local.info(f"Universe loaded ({n_frames_total} frames). Processing all frames.")

        # 2. Get Filter Residues from DB Metadata
        logger_local.info("Retrieving filter residue map from database...")
        filter_res_json = get_simulation_metadata(db_conn, 'filter_residues_dict')
        if not filter_res_json: raise ValueError("filter_residues_dict not found in DB metadata. Run ion_analysis first.")
        try:
            filter_res_map = json.loads(filter_res_json)
            if not isinstance(filter_res_map, dict) or len(filter_res_map) == 0:
                 raise TypeError("Parsed filter_residues is not a valid dictionary or is empty.")
        except (json.JSONDecodeError, TypeError) as e:
            raise ValueError(f"Failed to load/parse filter_residues_dict from DB: {e}")

        # 3. Prepare Graph Data
        logger_local.info("Preparing data for ET model...")
        # Pass parameters from config
        data_list = prepare_data(
            universe, None, frames_mask, filter_res_map, # Pass None for labels_dict
            n_upstream, n_downstream, cyl_radius, init_height,
            min_waters, rmsf_window
        )
        if not data_list: raise ValueError("Data preparation yielded no graph objects.")

        # 4. Compute Occupancy & Residence Times using Model
        occupancy_df, processed_residence_times_ns, detailed_residence_times, pockets_dict = compute_occupancy_and_residence(
            data_list, model, device, filter_res_map, res_thresh_frames, traj_tolerance, universe
        )
        # detailed_residence_times contains per-water, per-pocket residence times
        
        # 5. Log information about the computed pockets
        logger_local.info(f"Created pockets_dict with data for {len(pockets_dict)} frames")
        # Count total water molecules assigned to pockets
        total_water_assignments = sum(len(pocket_waters) 
                                     for frame_data in pockets_dict.values() 
                                     for pocket_idx, pocket_waters in frame_data.items())
        logger_local.info(f"Total water molecule assignments to pockets: {total_water_assignments}")
        # Count water molecules per pocket
        waters_by_pocket = {i: sum(len(frame_data.get(i, [])) for frame_data in pockets_dict.values()) 
                           for i in range(4)}
        for pocket_idx, count in waters_by_pocket.items():
            logger_local.info(f"Pocket {pocket_idx}: {count} water molecule assignments")

        # 6. Calculate Imbalance Metrics
        logger_local.info("Calculating imbalance metrics...")
        imbalance_metrics_dict = calculate_imbalance_metrics(processed_residence_times_ns, occupancy_df)

        # 7. Generate Summary Text
        logger_local.info("Generating summary text...")
        summary_text = summarize_pocket_analysis(occupancy_df, processed_residence_times_ns)

        # --- Save Outputs & Register Products ---
        logger_local.info("Saving analysis outputs...")
        products_saved = True

        # Occupancy CSV
        occ_csv_path = os.path.join(output_dir, "pocket_occupancy.csv")
        try:
            occupancy_df.to_csv(occ_csv_path, index=False, float_format='%.4f', na_rep='NaN')
            register_product(db_conn, module_name, "csv", "data", os.path.relpath(occ_csv_path, run_dir),
                             subcategory="pocket_occupancy_timeseries",
                             description="Time series of water counts per peripheral pocket.")
        except Exception as e: products_saved = False; logger.error(f"Failed to save/register occupancy CSV: {e}")

        # Residence Stats JSON
        res_json_path = os.path.join(output_dir, "pocket_residence_stats.json")
        try:
            # Convert processed_residence_times_ns to a JSON-serializable format
            # Convert NumPy int64/float64 types to regular Python int/float
            # Create a new dict with str keys for JSON serialization
            json_serializable_residence_times = {}
            for pocket, times in processed_residence_times_ns.items():
                # Convert NumPy int64 to regular Python int for keys
                pocket_key = int(pocket) if isinstance(pocket, np.integer) else pocket
                # Convert NumPy floats to regular Python floats for values
                times_list = [float(t) for t in times]
                json_serializable_residence_times[pocket_key] = times_list
            
            # Dump the converted dict to JSON
            with open(res_json_path, 'w') as f:
                json.dump(json_serializable_residence_times, f, indent=2)
            
            register_product(db_conn, module_name, "json", "data", os.path.relpath(res_json_path, run_dir),
                             subcategory="pocket_residence_stats",
                             description="Lists of water residence times (ns) per pocket.")
        except Exception as e: products_saved = False; logger.error(f"Failed to save/register residence JSON: {e}")

        # Pocket Assignments PKL
        assign_pkl_path = os.path.join(output_dir, "pocket_assignments.pkl")
        try:
            # Convert pockets_dict to use standard Python types
            serializable_pockets_dict = {}
            for frame_idx, frame_data in pockets_dict.items():
                frame_key = int(frame_idx) if isinstance(frame_idx, np.integer) else frame_idx
                serializable_pockets_dict[frame_key] = {}
                for pocket_idx, water_indices in frame_data.items():
                    pocket_key = int(pocket_idx) if isinstance(pocket_idx, np.integer) else pocket_idx
                    # Convert water indices to regular Python ints
                    serializable_pockets_dict[frame_key][pocket_key] = [int(idx) if isinstance(idx, np.integer) else idx for idx in water_indices]
            
            with open(assign_pkl_path, 'wb') as f:
                pickle.dump(serializable_pockets_dict, f)
                
            register_product(db_conn, module_name, "pkl", "data", os.path.relpath(assign_pkl_path, run_dir),
                             subcategory="pocket_assignments_per_frame",
                             description="Pocket assignments (0-3) for water indices per frame.")
        except Exception as e: products_saved = False; logger.error(f"Failed to save/register assignments PKL: {e}")

        # Imbalance Metrics CSV
        imb_csv_path = os.path.join(output_dir, "pocket_imbalance_metrics.csv")
        try:
            # Convert NumPy types to native Python types
            clean_metrics = {}
            for key, value in imbalance_metrics_dict.items():
                # Convert NumPy floats/ints to native Python types
                if isinstance(value, (np.integer, np.floating)):
                    clean_metrics[key] = float(value)
                elif np.isnan(value):
                    clean_metrics[key] = float('nan')  # Convert NumPy NaN to Python float NaN
                else:
                    clean_metrics[key] = value
                    
            imb_df = pd.DataFrame(clean_metrics.items(), columns=['Metric', 'Value'])
            imb_df.to_csv(imb_csv_path, index=False, float_format='%.4f', na_rep='NaN')
            register_product(db_conn, module_name, "csv", "data", os.path.relpath(imb_csv_path, run_dir),
                             subcategory="pocket_imbalance_metrics",
                             description="Calculated metrics quantifying imbalance between pockets.")
        except Exception as e: products_saved = False; logger.error(f"Failed to save/register imbalance CSV: {e}")

        # Summary Text TXT
        summary_txt_path = os.path.join(output_dir, "pocket_summary.txt")
        try:
            with open(summary_txt_path, 'w') as f: f.write(summary_text)
            register_product(db_conn, module_name, "txt", "summary", os.path.relpath(summary_txt_path, run_dir),
                             subcategory="pocket_analysis_summary",
                             description="Text summary of pocket water analysis.")
        except Exception as e: products_saved = False; logger.error(f"Failed to save/register summary TXT: {e}")

        if not products_saved:
             # Raise error if saving failed, as subsequent steps rely on these files
             raise IOError("One or more output files failed to save/register.")

        # --- Store Metrics ---
        logger_local.info("Storing key metrics...")
        metrics_stored_count = 0
        # Occupancy Metrics
        if all(col in occupancy_df.columns for col in [f'Pocket {i}' for i in range(4)]):
             mean_occs = occupancy_df[[f'Pocket {i}' for i in range(4)]].mean()
             std_occs = occupancy_df[[f'Pocket {i}' for i in range(4)]].std()
             for i in range(4):
                  pocket_label = chr(ord('A') + i)
                  store_metric(db_conn, module_name, f"Pocket{pocket_label}_MeanOccupancy", mean_occs[f'Pocket {i}'], 'count')
                  store_metric(db_conn, module_name, f"Pocket{pocket_label}_OccupancyStd", std_occs[f'Pocket {i}'], 'count')
                  metrics_stored_count += 2
        # Residence Time Metrics
        for pocket_idx, times in processed_residence_times_ns.items():
             pocket_label = chr(ord('A') + pocket_idx)
             if times:
                 store_metric(db_conn, module_name, f"Pocket{pocket_label}_MeanResidence_ns", np.mean(times), 'ns')
                 store_metric(db_conn, module_name, f"Pocket{pocket_label}_MedianResidence_ns", np.median(times), 'ns')
                 store_metric(db_conn, module_name, f"Pocket{pocket_label}_MaxResidence_ns", np.max(times), 'ns')
                 store_metric(db_conn, module_name, f"Pocket{pocket_label}_ResidencePeriods", len(times), 'count')
                 metrics_stored_count += 4
                 if len(times) > 2:
                      store_metric(db_conn, module_name, f"Pocket{pocket_label}_RTSkewness", skew(times), '')
                      metrics_stored_count += 1
                 # Short/Long lived %
                 short_thresh = getattr(core_config, 'POCKET_ANALYSIS_SHORT_LIVED_THRESH_NS', 5.0)
                 long_thresh = getattr(core_config, 'POCKET_ANALYSIS_LONG_LIVED_THRESH_NS', 10.0)
                 short_pct = (np.sum(np.array(times) < short_thresh) / len(times)) * 100.0
                 long_pct = (np.sum(np.array(times) > long_thresh) / len(times)) * 100.0
                 store_metric(db_conn, module_name, f"Pocket{pocket_label}_ShortLivedPct", short_pct, '%')
                 store_metric(db_conn, module_name, f"Pocket{pocket_label}_LongLivedPct", long_pct, '%')
                 metrics_stored_count += 2

        # Imbalance Metrics
        for metric_name, value in imbalance_metrics_dict.items():
             # Determine units (mostly unitless, KS is a stat)
             units = '' if metric_name != 'Max_Pairwise_KS_Statistic' and not metric_name.startswith('PocketWater_KS_') else 'KS stat'
             if pd.notna(value): # Only store non-NaN values
                 store_metric(db_conn, module_name, metric_name, value, units)
                 metrics_stored_count += 1

        logger_local.info(f"Stored {metrics_stored_count} metrics.")

        results['status'] = 'success'

    except Exception as e:
        results['error'] = f"Error during pocket analysis computation: {e}"
        logger_local.error(results['error'], exc_info=True)
        update_module_status(db_conn, module_name, 'failed', error_message=results['error'])
        return results

    # --- Finalize ---
    exec_time = time.time() - start_time
    update_module_status(db_conn, module_name, results['status'], execution_time=exec_time, error_message=results['error'])
    logger_local.info(f"--- Pocket Analysis computation finished in {exec_time:.2f} seconds (Status: {results['status']}) ---")

    return results