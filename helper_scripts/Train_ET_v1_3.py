# Train_ET_v1_3_patched.py
"""
Train_ET.py: Training an Equivariant Transformer Model for Water Molecule Classification in Ion Channels

(Original Docstring remains the same)
...

Version: 1.3 (Patched - Corrected super().__init__ call in CustomTorchMD_ET)
"""

import os
import json
import random
import logging
import argparse
import uuid
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
# Corrected import if needed (ReduceLROnPlateau used in multi script, StepLR here)
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau # Keep StepLR if used here
from torch_geometric.data import Data, DataLoader # DataLoader potentially used here
import MDAnalysis as mda

from scipy.spatial import KDTree
from scipy.cluster.vq import kmeans, vq
from scipy.spatial.distance import pdist, squareform

from tqdm import tqdm

# Custom imports (make sure these are defined or imported appropriately)
# Assuming TorchMD_ET is correctly installed and importable
try:
    from torchmdnet.models.torchmd_et import TorchMD_ET
except ImportError as e:
     # Handle missing import if necessary
     print(f"Error importing TorchMD_ET: {e}. Ensure torchmd-net is installed.")
     raise

# setup_logging function (can be defined here or imported)
def setup_logging(output_dir):
    # Remove existing handlers to avoid duplicate logs if re-run in same session
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    # Setup logging
    log_file = os.path.join(output_dir, 'run.log')
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(log_file),
                            logging.StreamHandler() # Log to console as well
                        ])
    logging.info(f"Logging initialized. Log file: {log_file}")


# Define a mapping from atom names to atomic numbers
atomic_numbers = {
    'H': 1, 'C': 6, 'N': 7, 'O': 8, 'P': 15, 'S': 16, 'K': 19, 'Cl': 17,
    # Add more elements as needed
}

# At the top of the file, after imports
three_to_one = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLU': 'E', 'GLN': 'Q',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F',
    'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V', 'HSE': 'H'
}

def find_selectivity_filter(subunit, seq):
    # (Function remains the same)
    sequence_str = ''.join(three_to_one.get(res.resname, 'X') for res in subunit.residues)
    return sequence_str.find(seq)

def get_atomic_numbers(atom_group):
    # (Function remains the same)
    z = []
    for atom in atom_group:
        element = atom.name[0]
        if element in atomic_numbers:
            z.append(atomic_numbers[element])
        else:
            # Log error instead of raising immediately if preferred
            logging.error(f"Unknown element encountered: {element} for atom {atom.name} {atom.index}")
            raise ValueError(f"Unknown element: {element}")
    return torch.tensor(z, dtype=torch.long)

def load_dataset(config, dataset_number, start_frame, end_frame, stride):
    # (Function remains the same)
    data_dir = config['data_dir']
    psf_file = os.path.join(data_dir, f'train_{dataset_number}.psf')
    dcd_file = os.path.join(data_dir, f'train_{dataset_number}.dcd')
    labels_file = os.path.join(data_dir, f'labels_train_{dataset_number}.csv')

    print(f"Loading PSF file: {psf_file}")
    print(f"Loading DCD file: {dcd_file}")
    print(f"Loading labels file: {labels_file}")

    for file in [psf_file, dcd_file, labels_file]:
        if not os.path.exists(file):
            raise FileNotFoundError(f"File not found: {file}")

    universe = mda.Universe(psf_file, dcd_file)

    if len(universe.atoms) == 0:
        raise ValueError("The universe contains no atoms. Check your PSF file.")

    n_frames = len(universe.trajectory)
    if n_frames == 0:
        raise ValueError("The trajectory contains no frames. Check your DCD file.")

    print(f"Loaded universe with {len(universe.atoms)} atoms and {n_frames} frames.")

    if end_frame is None or end_frame > n_frames:
        end_frame = n_frames

    start_frame = max(0, min(start_frame, n_frames - 1))
    end_frame = max(start_frame, min(end_frame, n_frames)) # Ensure end >= start

    # Ensure stride is positive
    stride = max(1, stride)

    frames_mask = list(range(start_frame, end_frame, stride))
    # Handle case where range is empty
    if not frames_mask:
         print(f"Warning: Frame mask is empty for start={start_frame}, end={end_frame}, stride={stride}. No frames will be processed.")
         # Decide how to handle this - return empty dict/mask or raise error?
         # Returning empty allows calling code to handle potentially empty datasets
         return universe, {}, []

    print(f"Frames mask created with {len(frames_mask)} frames.")
    print(f"Trajectory info: Start={start_frame}, Stop={end_frame}, Step={stride}")

    print(f"Loading labels from {labels_file}")
    labels_df = pd.read_csv(labels_file)
    print(f"Total rows in labels_df: {len(labels_df)}")

    # Filter the DataFrame based on frame conditions included in the mask
    # More robust filtering: check if frame is actually in the generated frames_mask list
    # Convert frames_mask to a set for faster lookup
    frames_set = set(frames_mask)
    mask = labels_df['Frame'].isin(frames_set)
    filtered_df = labels_df[mask].copy()
    print(f"Rows after filtering based on frames_mask: {len(filtered_df)}")

    # Create the labels_dict using groupby and apply
    print("Creating labels dictionary...")
    labels_dict = {}
    # Group by the original frame number from the filtered df
    for frame_num, group in tqdm(filtered_df.groupby('Frame')):
        # Store labels under the original frame index
        labels_dict[frame_num] = dict(zip(group['Index'], group['Label']))

    print(f"Labels dictionary created with labels for {len(labels_dict)} frames (keys are original frame indices)")

    return universe, labels_dict, frames_mask


def select_water_molecules_trajectory(universe, frames_mask, cylinder_radius, initial_height, min_waters):
    # (Function remains the same)
    water_oxygen = universe.select_atoms("resname TIP3 and name OH2")

    if len(water_oxygen) == 0:
        raise ValueError("No water oxygen atoms found.")

    n_selected_frames = len(frames_mask) # Use length of the mask
    selected_waters = np.zeros((n_selected_frames, len(water_oxygen)), dtype=bool)

    subunits = [universe.select_atoms(f"segid PRO{chain}") for chain in "ABCD"]
    sf_tyrosines = []
    for subunit in subunits:
        sf_position = find_selectivity_filter(subunit, "GYG")
        if sf_position == -1:
            raise ValueError(f"Selectivity filter sequence 'GYG' not found in subunit {subunit.segids[0]}")
        sf_tyrosine = subunit.residues[sf_position + 1]
        if sf_tyrosine.resname != 'TYR':
            raise ValueError(f"Expected Tyrosine in GYG sequence, found {sf_tyrosine.resname} in subunit {subunit.segids[0]}")
        sf_tyrosines.append(sf_tyrosine)

    tyr_ca = universe.atoms.select_atoms(" or ".join(f"(segid {tyr.segid} and resid {tyr.resid} and name CA)" for tyr in sf_tyrosines))

    if len(tyr_ca) != 4:
        raise ValueError(f"Expected 4 tyrosine alpha carbons from GYG sequences, found {len(tyr_ca)}")

    # Iterate using the frame indices from frames_mask
    for idx, frame_index in enumerate(tqdm(frames_mask, desc="Processing frames", unit="frame")):
        ts = universe.trajectory[frame_index] # Access specific frame
        cylinder_center = tyr_ca.center_of_mass()
        water_positions = water_oxygen.positions
        xy_distances = np.sqrt(np.sum((water_positions[:, :2] - cylinder_center[:2])**2, axis=1))
        in_cylinder_radius = xy_distances <= cylinder_radius

        height = initial_height
        while True:
            in_cylinder_height = np.abs(water_positions[:, 2] - cylinder_center[2]) <= height/2
            in_cylinder = in_cylinder_radius & in_cylinder_height
            if np.sum(in_cylinder) >= min_waters or height > 2 * initial_height + 20 : # Add height limit to prevent infinite loop
                 if height > 2 * initial_height + 20:
                     print(f"Warning: Reached height limit ({height} A) for frame {frame_index} while searching for {min_waters} waters. Found {np.sum(in_cylinder)}.")
                 break
            height += 0.5

        selected_waters[idx] = in_cylinder # Store result at index corresponding to frames_mask order

    return selected_waters, water_oxygen


def compute_rmsf_x_dimension(universe, water_oxygen, frames_mask, window_size=10):
    # (Function remains the same)
    n_selected_frames = len(frames_mask)
    n_waters = len(water_oxygen)

    # Initialize arrays for positions and RMSF based on the number of selected frames
    water_positions = np.zeros((n_selected_frames, n_waters))
    rmsf = np.zeros((n_selected_frames, n_waters))

    # Collect positions only for the selected frames
    for idx, frame_index in enumerate(tqdm(frames_mask, desc="Collecting water positions", unit="frame")):
        ts = universe.trajectory[frame_index]
        water_positions[idx] = water_oxygen.positions[:, 0] # Store X-dimension

    # Calculate RMSF using a sliding window over the collected positions
    # Ensure window size is valid
    window_size = min(window_size, n_selected_frames)
    if window_size <= 1:
         print("Warning: RMSF window size <= 1, RMSF will be zero.")
         return rmsf # Return zeros

    for i in range(n_selected_frames):
        # Define window boundaries, clamping at the start/end of the selected frames
        start = max(0, i - window_size // 2)
        end = min(n_selected_frames, i + (window_size + 1) // 2) # +1 for odd sizes
        current_window_size = end - start

        if current_window_size <= 1:
            rmsf[i] = 0.0 # RMSF undefined or zero for window size <= 1
            continue

        window = water_positions[start:end]
        mean_pos = np.mean(window, axis=0)
        squared_diff = np.mean((window - mean_pos) ** 2, axis=0)
        rmsf[i] = np.sqrt(squared_diff) # Store RMSF at the center index 'i'

    return rmsf


def select_tetramer_backbone_atoms(universe, sequence="GYG", n_upstream=10, n_downstream=10):
    # (Function remains the same)
    selected_atoms_list = []
    subunit_ids = []
    atom_indices = []

    for subunit_id, segid in enumerate(['PROA', 'PROB', 'PROC', 'PROD']):
        subunit = universe.select_atoms(f'segid {segid} and protein')

        if len(subunit) == 0:
            print(f"Warning: No atoms found for subunit {segid}")
            continue

        subunit_sequence = ''.join(three_to_one.get(res.resname, 'X') for res in subunit.residues)
        #print(f"Subunit {segid} sequence: {subunit_sequence}") # Can be verbose

        filter_position = find_selectivity_filter(subunit, sequence)

        if filter_position != -1:
            start_index = max(0, filter_position - n_upstream)
            end_index = min(len(subunit.residues), filter_position + len(sequence) + n_downstream)

            start_resid = subunit.residues[start_index].resid
            end_resid = subunit.residues[end_index - 1].resid

            backbone_selection = subunit.select_atoms(f'resid {start_resid}:{end_resid} and backbone')
            selected_atoms_list.append(backbone_selection)
            subunit_ids.extend([subunit_id] * len(backbone_selection))
            atom_indices.extend(backbone_selection.indices)

            #print(f"Selected residues {start_resid} to {end_resid} for subunit {segid}")
        else:
            print(f"Warning: Selectivity filter sequence '{sequence}' not found in subunit {segid}")

    if not selected_atoms_list:
        raise ValueError("Selectivity filter sequence not found in any subunit")

    selected_atoms = mda.core.groups.AtomGroup([], universe)
    for ag in selected_atoms_list:
        selected_atoms += ag

    return selected_atoms, subunit_ids, atom_indices


def prepare_data(universe, labels_dict, frames_mask, sequence, n_upstream, n_downstream, cylinder_radius, initial_height, min_waters, window_size):
    # (Function remains the same)
    if not frames_mask: # Handle empty frames_mask
        print("Warning: prepare_data received an empty frames_mask. Returning empty list.")
        return []

    selected_backbone, subunit_ids, atom_indices = select_tetramer_backbone_atoms(universe, sequence, n_upstream, n_downstream)
    selected_waters_mask, water_oxygen = select_water_molecules_trajectory(universe, frames_mask, cylinder_radius, initial_height, min_waters)
    rmsf_values = compute_rmsf_x_dimension(universe, water_oxygen, frames_mask, window_size)

    data_list = []
    # Iterate using the frame indices from frames_mask
    for idx, frame_index in enumerate(tqdm(frames_mask, desc="Preparing data", unit="frame")):
        ts = universe.trajectory[frame_index] # Set trajectory to the correct frame

        # Get backbone properties for the current frame
        backbone_z = get_atomic_numbers(selected_backbone) # Constant
        backbone_pos = torch.tensor(selected_backbone.positions, dtype=torch.float32) # Positions at current frame

        # Get properties for selected water molecules for the current frame
        current_frame_water_mask = selected_waters_mask[idx]
        # Ensure there are selected waters before proceeding
        if not np.any(current_frame_water_mask):
            # print(f"Warning: No water molecules selected for frame {frame_index}. Skipping frame.")
            continue # Skip this frame if no waters are selected

        selected_water_oxygens_indices = np.where(current_frame_water_mask)[0]
        num_selected_waters = len(selected_water_oxygens_indices)

        water_z = torch.tensor([atomic_numbers['O']] * num_selected_waters, dtype=torch.long)
        # Ensure indexing into positions and rmsf uses the boolean mask or resulting indices correctly
        water_pos = torch.tensor(water_oxygen.positions[current_frame_water_mask], dtype=torch.float32)
        water_rmsf = torch.tensor(rmsf_values[idx][current_frame_water_mask], dtype=torch.float32)
        water_original_indices = water_oxygen.indices[current_frame_water_mask]

        # Combine backbone and water data
        z = torch.cat([backbone_z, water_z])
        pos = torch.cat([backbone_pos, water_pos])

        # Create feature tensors matching combined atoms
        subunit_ids_frame = torch.tensor(subunit_ids + [-1] * num_selected_waters, dtype=torch.long) # -1 for waters
        batch = torch.zeros(len(z), dtype=torch.long) # All atoms belong to the same graph/batch index 0

        atom_indices_frame = torch.tensor(atom_indices + list(water_original_indices), dtype=torch.long)
        rmsf_frame = torch.cat([torch.full((len(backbone_pos),), -1.0, dtype=torch.float32), # Use -1.0 for backbone RMSF placeholder
                                water_rmsf])

        # Initialize labels: -1 for all atoms (including backbone)
        labels = torch.full((len(z),), -1, dtype=torch.long)

        # Assign labels ONLY to water molecules based on the labels_dict (using original frame index)
        # Use the original indices of the selected water molecules
        frame_labels = labels_dict.get(frame_index, {}) # Get labels for the original frame index, empty dict if frame not in dict
        for i, original_water_idx in enumerate(water_original_indices):
            label_value = frame_labels.get(original_water_idx, 0) # Default to 0 if water not in labels dict for this frame
            # Assign label at the correct position in the combined tensor (after backbone atoms)
            labels[len(backbone_pos) + i] = label_value

        # Create is_water boolean mask
        is_water = torch.cat([torch.zeros(len(backbone_pos), dtype=torch.bool),
                              torch.ones(num_selected_waters, dtype=torch.bool)])

        # Create PyG Data object
        data = Data(pos=pos, z=z, batch=batch, subunit_ids=subunit_ids_frame,
                    atom_indices=atom_indices_frame, rmsf=rmsf_frame,
                    labels=labels, is_water=is_water)
        data_list.append(data)

    return data_list


def print_prepared_data_for_frame(prepared_data, frame):
    # (Function remains the same)
    if frame < 0 or frame >= len(prepared_data):
        print(f"Error: Frame index {frame} out of bounds (0 to {len(prepared_data)-1})")
        return

    graph = prepared_data[frame]

    n_atoms = len(graph.pos) if hasattr(graph, 'pos') else 0
    if n_atoms == 0:
        print(f"Frame {frame}: No data available.")
        return

    n_backbone = (~graph.is_water).sum().item() if hasattr(graph, 'is_water') else 0
    n_water = graph.is_water.sum().item() if hasattr(graph, 'is_water') else 0

    print(f"Frame {frame} Summary:")
    print(f"  Total number of atoms: {n_atoms}")
    print(f"  Number of backbone atoms: {n_backbone}")
    print(f"  Number of water molecules: {n_water}")

    print("\nBackbone Atoms (first 5):")
    count = 0
    for i in range(n_atoms):
        if hasattr(graph, 'is_water') and not graph.is_water[i]:
            subunit_id = graph.subunit_ids[i].item() if hasattr(graph, 'subunit_ids') else 'N/A'
            z_val = graph.z[i].item() if hasattr(graph, 'z') else 'N/A'
            print(f"  Atom {i} - Z: {z_val} (Subunit ID: {subunit_id})")
            count += 1
            if count >= 5: break

    print("\nWater Molecules (first 5):")
    water_atom_indexes = []
    count = 0
    for i in range(n_atoms):
         if hasattr(graph, 'is_water') and graph.is_water[i]:
            water_atom_indexes.append(i) # Store index within the Data object
            z_val = graph.z[i].item() if hasattr(graph, 'z') else 'N/A'
            print(f"  Atom {i} - Z: {z_val}")
            count += 1
            if count >= 5: break

    print("\nRMSF Values (first 5 water molecules):")
    count = 0
    for i in range(n_atoms):
         if hasattr(graph, 'is_water') and graph.is_water[i]:
             rmsf = graph.rmsf[i].item() if hasattr(graph, 'rmsf') else float('nan')
             if not torch.isnan(torch.tensor(rmsf)): # Handle potential NaNs
                 print(f"  Water Atom {i}: RMSF = {rmsf:.3f} Å")
             else:
                 print(f"  Water Atom {i}: RMSF = NaN")
             count += 1
             if count >= 5: break

    print("\nLabels (first 5 water molecules):")
    count = 0
    for i in range(n_atoms):
        if hasattr(graph, 'is_water') and graph.is_water[i]:
            label = graph.labels[i].item() if hasattr(graph, 'labels') else -1
            # Only print assigned labels (0 or 1)
            #if label != -1:
            print(f"  Water Atom {i}: Label = {label}") # Print all labels for debug
            count += 1
            if count >= 5: break


    # VMD selection string using ORIGINAL indices if available
    if hasattr(graph, 'atom_indices') and hasattr(graph, 'is_water'):
        original_water_indices = graph.atom_indices[graph.is_water].cpu().numpy()
        vmd_selection_string = f'index {" ".join(map(str, original_water_indices))}'
        print(f"\nVMD Selection String for Water Molecules (Original Indices):\n{vmd_selection_string}")
    else:
        print("\nCannot generate VMD selection string (missing atom_indices or is_water).")


    print("\nPositions (first 5 atoms):")
    for i in range(min(5, n_atoms)):
        if hasattr(graph, 'pos'):
            pos = graph.pos[i]
            print(f"  Atom {i}: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")

    # Print label statistics
    if hasattr(graph, 'labels') and hasattr(graph, 'is_water') and n_water > 0:
        water_labels = graph.labels[graph.is_water]
        n_positive = (water_labels == 1).sum().item()
        n_negative = (water_labels == 0).sum().item()
        n_unlabeled = (water_labels == -1).sum().item() # Should be 0 if default is 0
        print("\nLabel Statistics (water molecules):")
        print(f"  Positive (1): {n_positive}")
        print(f"  Negative (0): {n_negative}")
        print(f"  Unlabeled (-1): {n_unlabeled}")
    else:
         print("\nLabel statistics unavailable.")

    # Print real universe indices of positively labeled water molecules
    if hasattr(graph, 'atom_indices') and hasattr(graph, 'labels') and hasattr(graph, 'is_water'):
         positive_mask = (graph.is_water) & (graph.labels == 1)
         if positive_mask.any():
             positive_water_indices = graph.atom_indices[positive_mask].tolist()
             print("\nReal universe indices of positively labeled water molecules:")
             print("index", " ".join(map(str, positive_water_indices)))
         else:
             print("\nNo positively labeled water molecules in this frame.")

# ============================================
# PATCHED CustomTorchMD_ET CLASS STARTS HERE
# ============================================
class CustomTorchMD_ET(TorchMD_ET):
    def __init__(self, num_subunits, subunit_embedding_dim, rmsf_embedding_dim, label_embedding_dim, hidden_channels, num_layers, num_heads, num_rbf, rbf_type, activation, attn_activation, neighbor_embedding, cutoff_lower, cutoff_upper, max_z, max_num_neighbors, *args, **kwargs):
        # --- Patched super().__init__() call ---
        # Explicitly pass architectural args to the parent TorchMD_ET class
        # Get dtype from kwargs or set a default if not provided
        dtype = kwargs.pop('dtype', torch.float32) # Remove dtype from kwargs if present
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
             max_num_neighbors=max_num_neighbors,
             dtype=dtype, # Pass dtype explicitly
             # Pass any remaining arguments if the parent signature requires them
             *args, **kwargs
             )
        # --- End Patch ---

        # Explicitly store architectural params on self for clarity and potential use
        # Note: The parent class likely stores these too after the corrected super() call.
        # This ensures they are definitely available on the child instance.
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_heads = num_heads # Store if needed

        # Initialize custom layers AFTER parent is initialized correctly
        self.subunit_embedding = nn.Embedding(num_subunits, subunit_embedding_dim)
        self.rmsf_embedding = nn.Sequential(
            nn.Linear(1, rmsf_embedding_dim),
            nn.ReLU(),
            nn.Linear(rmsf_embedding_dim, rmsf_embedding_dim)
        )
        self.label_embedding = nn.Embedding(2, label_embedding_dim) # Assuming 2 classes (0 and 1)

        self.subunit_embedding_dim = subunit_embedding_dim
        self.rmsf_embedding_dim = rmsf_embedding_dim
        self.label_embedding_dim = label_embedding_dim

        # Calculate feature dim based on max of custom embeddings
        feature_dim = max(subunit_embedding_dim, rmsf_embedding_dim, label_embedding_dim)

        # Classifier input depends on parent output (self.hidden_channels should be correct now) + feature_dim
        classifier_input_dim = self.hidden_channels + feature_dim

        # Classifier definition - uses self.hidden_channels which should now be correctly set
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, self.hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.5), # Consider making dropout configurable
            nn.Linear(self.hidden_channels // 2, 1)
        )

    # Forward pass needs to handle the features correctly
    def forward(self, z, pos, batch, subunit_ids, atom_indices, is_water, rmsf, labels=None): # Added default None for labels
        # Call the parent forward method first
        # It returns: x (node features), vec (equiv vector features), z, pos, batch
        x, vec, z_ret, pos_ret, batch_ret = super().forward(z, pos, batch)
        # Important: Use the returned z, pos, batch as they might be modified by parent (e.g., reordered)

        # Create placeholder for additional features
        feature_dim = max(self.subunit_embedding_dim, self.rmsf_embedding_dim, self.label_embedding_dim)
        # Ensure features tensor matches the device of x
        features = torch.zeros(x.shape[0], feature_dim, device=x.device, dtype=x.dtype)

        # Apply subunit embedding to non-water atoms
        non_water_mask = ~is_water
        if non_water_mask.any(): # Check if there are any non-water atoms
             # Ensure subunit IDs are valid indices for the embedding layer
             valid_subunit_ids = subunit_ids[non_water_mask]
             # Clamp or handle potential invalid indices (e.g., -1 if used)
             valid_subunit_ids = valid_subunit_ids[valid_subunit_ids >= 0]
             mask_for_embedding = non_water_mask.clone()
             mask_for_embedding[non_water_mask] = (subunit_ids[non_water_mask] >= 0)

             if valid_subunit_ids.numel() > 0 :
                 # Check max index vs embedding size
                 if valid_subunit_ids.max() >= self.subunit_embedding.num_embeddings:
                      logging.error(f"Subunit ID {valid_subunit_ids.max()} is out of bounds for embedding size {self.subunit_embedding.num_embeddings}")
                      # Handle error: raise, clamp, or skip embedding? For now, raise.
                      raise IndexError("Subunit ID out of bounds for embedding layer.")
                 features[mask_for_embedding, :self.subunit_embedding_dim] = self.subunit_embedding(valid_subunit_ids)


        # Apply RMSF embedding to water atoms
        water_mask = is_water
        if water_mask.any(): # Check if there are any water atoms
            rmsf_water = rmsf[water_mask].unsqueeze(1).to(x.dtype) # Ensure correct dtype
            # Handle potential NaNs/infs in RMSF if necessary before embedding
            if torch.isnan(rmsf_water).any() or torch.isinf(rmsf_water).any():
                 logging.warning("NaN or Inf found in RMSF values for water molecules. Replacing with 0 before embedding.")
                 rmsf_water = torch.nan_to_num(rmsf_water, nan=0.0, posinf=0.0, neginf=0.0)

            rmsf_embeddings = self.rmsf_embedding(rmsf_water)
            features[water_mask, :self.rmsf_embedding_dim] = rmsf_embeddings

            # Apply label embedding during training if labels are provided
            if self.training and labels is not None:
                 water_labels = labels[water_mask]
                 # Use only valid labels (0 or 1) for embedding lookup
                 valid_label_mask_train = (water_labels == 0) | (water_labels == 1)
                 if valid_label_mask_train.any():
                     valid_labels_for_embedding = water_labels[valid_label_mask_train]
                     label_embeddings = self.label_embedding(valid_labels_for_embedding)
                     # Create a mask specific to waters with valid labels for assignment
                     water_valid_label_indices = water_mask.clone()
                     water_valid_label_indices[water_mask] = valid_label_mask_train
                     features[water_valid_label_indices, :self.label_embedding_dim] = label_embeddings

        # Concatenate parent features (x) with custom features
        x = torch.cat([x, features], dim=-1)

        # Apply classifier ONLY to water molecules
        logits = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype) # Initialize all logits to 0
        if water_mask.any(): # Apply classifier if there are water atoms
             logits[water_mask] = self.classifier(x[water_mask])

        # Return logits, is_water mask, and original labels (passed through) for loss calculation
        # Use the input labels (e.g., batch.labels) for consistency in loss function
        input_labels = labels if labels is not None else torch.full_like(z_ret, -1) # Use placeholder if no labels passed
        return logits, is_water, input_labels

    # Loss computation remains the same logic
    def compute_loss(self, logits, is_water, labels):
        water_mask = is_water.bool()
        # Check if there are any water molecules to compute loss for
        if not water_mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True) # Return zero loss if no water

        water_logits = logits[water_mask].squeeze(-1)
        water_labels = labels[water_mask].float() # Ensure labels are float for BCE loss

        # Use only valid labels (0 or 1) for loss calculation
        valid_label_mask = (water_labels == 0) | (water_labels == 1)

        # Check if there are any valid labels to compute loss for
        if not valid_label_mask.any():
             return torch.tensor(0.0, device=logits.device, requires_grad=True) # Return zero loss if no valid labels

        # Calculate loss only on logits/labels where the label is valid
        loss = nn.functional.binary_cross_entropy_with_logits(
            water_logits[valid_label_mask],
            water_labels[valid_label_mask]
        )
        return loss

# ============================================
# PATCHED CustomTorchMD_ET CLASS ENDS HERE
# ============================================


# train_epoch and validate functions (as defined in the multi-dataset script or here)
# Need to be consistent with the model's forward signature and output
# Assuming the versions defined in Step1_multi_v1_3_debug are sufficient
# If this script is run standalone, redefine train_epoch and validate here:

def train_epoch_standalone(model, train_loader, optimizer, device): # Simplified for standalone example
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        # Ensure all required inputs for forward are present in batch
        logits, is_water, labels = model(batch.z, batch.pos, batch.batch, batch.subunit_ids, batch.atom_indices, batch.is_water, batch.rmsf, batch.labels)
        loss = model.compute_loss(logits, is_water, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def validate_standalone(model, val_loader, device): # Simplified for standalone example
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            logits, is_water, labels = model(batch.z, batch.pos, batch.batch, batch.subunit_ids, batch.atom_indices, batch.is_water, batch.rmsf, batch.labels)
            loss = model.compute_loss(logits, is_water, labels)
            total_loss += loss.item()
    return total_loss / len(val_loader)


def main(config):
    """
    Main function for standalone execution of Train_ET_v1_3.py (if needed).
    This typically wouldn't be run directly if using Step1_multi_v1_3.py.
    """
    run_id = str(uuid.uuid4())
    output_dir = os.path.join(config.get('output_dir', './output_standalone'), run_id)
    os.makedirs(output_dir, exist_ok=True)

    setup_logging(output_dir) # Setup logging to the run-specific directory
    logging.info(f"--- Standalone Training Script Started (Patched Version) ---")
    logging.info(f"Run ID: {run_id}")
    logging.info(f"Output directory: {output_dir}")

    # Save the configuration file to the output directory
    config_path = os.path.join(output_dir, 'config.json')
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        logging.info(f"Configuration saved to: {config_path}")
    except IOError as e:
        logging.error(f"Could not write config file: {e}")


    seed = config.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    logging.info(f"Set random seed: {seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # --- Data Loading ---
    logging.info("Loading data for standalone run...")
    try:
        # Load a single dataset specified in config
        dataset_number = config.get('dataset_number', 1) # Default to dataset 1
        start_frame = config.get('start_frame', 0)
        end_frame = config.get('end_frame', None) # Load all frames by default
        stride = config.get('stride', 1)
        universe, labels_dict, frames_mask = load_dataset(config, dataset_number, start_frame, end_frame, stride)

        # Prepare data
        data_list = prepare_data(
            universe,
            labels_dict,
            frames_mask,
            sequence=config['sequence'],
            n_upstream=config['n_upstream'],
            n_downstream=config['n_downstream'],
            cylinder_radius=config['cylinder_radius'],
            initial_height=config['initial_height'],
            min_waters=config['min_waters'],
            window_size=config['window_size']
        )
        if not data_list:
             logging.error("No data prepared. Exiting.")
             sys.exit(1)
    except Exception as e:
         logging.exception("Error during data loading/preparation in standalone run.")
         sys.exit(1)

    # --- Train/Val Split ---
    val_split = config.get('val_split', 0.2) # Default to 20% validation
    train_size = int((1 - val_split) * len(data_list))
    val_size = len(data_list) - train_size
    # Ensure sizes are valid before splitting
    if train_size <= 0 or val_size <= 0:
         logging.error(f"Invalid train/val split sizes ({train_size}/{val_size}) for {len(data_list)} samples.")
         sys.exit(1)
    train_dataset, val_dataset = torch.utils.data.random_split(data_list, [train_size, val_size])
    logging.info(f"Data split: {len(train_dataset)} train, {len(val_dataset)} validation samples.")


    # --- Dataloaders ---
    batch_size = config.get('batch_size', 16)
    num_workers = config.get('num_workers', 0)
    # Use the imported custom_collate function
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate, num_workers=num_workers)


    # --- Model Initialization (Using Patched Class) ---
    logging.info("Initializing model...")
    try:
        # Determine num_subunits dynamically
        if train_dataset:
             max_sid = -1
             # Iterate through subset used for training if possible
             for i in range(len(train_dataset)):
                  data = train_dataset[i] # Access data point from subset
                  if hasattr(data, 'subunit_ids') and data.subunit_ids.numel() > 0:
                       valid_sids = data.subunit_ids[data.subunit_ids >= 0]
                       if valid_sids.numel() > 0: max_sid = max(max_sid, valid_sids.max().item())
             num_subunits = max_sid + 1 if max_sid >=0 else 1
             logging.info(f"Determined num_subunits from training data: {num_subunits}")
        else:
             num_subunits = config.get('num_subunits', 4)
             logging.warning(f"Using fallback num_subunits: {num_subunits}")


        # Prepare kwargs and initialize (will now use the corrected super init)
        model_kwargs = {
            'num_subunits': num_subunits,
            'subunit_embedding_dim': config['subunit_embedding_dim'],
            'rmsf_embedding_dim': config['rmsf_embedding_dim'],
            'label_embedding_dim': config['label_embedding_dim'],
            'hidden_channels': config['hidden_channels'],
            'num_layers': config['num_layers'],
            'num_heads': config['num_heads'],
            'num_rbf': config['num_rbf'],
            'rbf_type': config['rbf_type'],
            'activation': config['activation'],
            'attn_activation': config['attn_activation'],
            'neighbor_embedding': config['neighbor_embedding'],
            'cutoff_lower': config['cutoff_lower'],
            'cutoff_upper': config['cutoff_upper'],
            'max_z': config.get('max_z', 100),
            'max_num_neighbors': config['max_num_neighbors'],
            'dtype': torch.float32
        }
        logging.info(f"Initializing CustomTorchMD_ET with: {model_kwargs}")
        model = CustomTorchMD_ET(**model_kwargs).to(device)
        logging.info("Model initialized successfully (using patched class).")
        # Optional: Add structure and state dict logging here too if desired
        # log_model_state_dict_summary(model, logging, "After Initialization (Standalone)")

    except Exception as e:
         logging.exception("Error during model initialization in standalone run.")
         sys.exit(1)


    # --- Optimizer, Scheduler, Training Loop ---
    optimizer = optim.Adam(model.parameters(), lr=config.get('learning_rate', 0.001), weight_decay=config.get('weight_decay', 0.0))
    # Use StepLR as potentially intended in the original standalone script
    scheduler = StepLR(optimizer, step_size=config.get('step_size', 10), gamma=config.get('gamma', 0.9))
    logging.info(f"Optimizer: Adam (lr={config.get('learning_rate', 0.001)})")
    logging.info(f"Scheduler: StepLR (step={config.get('step_size', 10)}, gamma={config.get('gamma', 0.9)})")


    patience = config.get('patience', 30)
    best_val_loss = float('inf')
    counter = 0
    num_epochs = config.get('num_epochs', 50) # Reasonable default for standalone
    checkpoint_interval = config.get('checkpoint_interval', 10)

    logging.info(f"Starting standalone training loop for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        logging.info(f"--- Epoch {epoch+1}/{num_epochs} ---")
        train_loss = train_epoch_standalone(model, train_loader, optimizer, device)
        logging.info(f"Epoch {epoch+1} Train Loss: {train_loss:.4f}")

        val_loss = validate_standalone(model, val_loader, device)
        logging.info(f"Epoch {epoch+1} Validation Loss: {val_loss:.4f}")

        scheduler.step() # StepLR doesn't need val_loss

        # Save checkpoint
        if checkpoint_interval > 0 and (epoch + 1) % checkpoint_interval == 0:
            checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch{epoch+1}.pth")
            try:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(), # Save scheduler state
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, checkpoint_path)
                logging.info(f"Saved checkpoint to {checkpoint_path}")
            except Exception as e:
                 logging.error(f"Error saving checkpoint epoch {epoch+1}: {e}")

        # Early stopping check
        # Note: Using StepLR, so early stopping based on val_loss is independent of LR changes
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            # Save only state_dict for best model in standalone, or full dict like checkpoint
            best_model_path = os.path.join(output_dir, 'best_model.pth')
            try:
                torch.save(model.state_dict(), best_model_path)
                logging.info(f"Saved new best model (Val Loss: {best_val_loss:.4f}) to {best_model_path}")
            except Exception as e:
                 logging.error(f"Error saving best model epoch {epoch+1}: {e}")

        else:
            counter += 1
            logging.info(f"Validation loss did not improve. Counter: {counter}/{patience}")
            if counter >= patience:
                logging.info(f"Early stopping triggered after epoch {epoch+1}")
                break

    # Save final model
    final_model_path = os.path.join(output_dir, "final_model.pth")
    try:
        torch.save(model.state_dict(), final_model_path)
        logging.info(f"Saved final model state_dict to {final_model_path}")
    except Exception as e:
        logging.error(f"Error saving final model: {e}")

    logging.info("--- Standalone Training Script Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patched: Train ET model for water molecule classification")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    args = parser.parse_args()

    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
        main(config) # Run the standalone main function
    except FileNotFoundError:
         print(f"FATAL: Config file not found: {args.config}")
    except Exception as e:
         # Use basic print if logger failed, otherwise use logger
         logging.exception(f"FATAL Error in script execution: {e}")
