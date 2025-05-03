# tests/test_pocket_analysis.py
"""
Tests for the pocket_analysis module.
Focuses on workflow, database interactions, and file generation,
using mocking for ML model predictions.
"""

import pytest
import os
import sqlite3
import json
import pickle
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock # Using unittest.mock

# Check if torch is available before importing any module that depends on it
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Create needed torch.nn.Module dummy for mocks 
    # to avoid import errors when torch is not available
    class DummyModule:
        def __init__(self, *args, **kwargs):
            pass
        def __call__(self, *args, **kwargs):
            pass
    
    # Create dummy module and classes
    import types
    import sys
    
    # Create a dummy torch module to avoid import errors
    torch_module = types.ModuleType('torch')
    torch_module.nn = types.ModuleType('torch.nn')
    torch_module.nn.Module = DummyModule
    torch_module.device = lambda x: x
    torch_module.cuda = types.ModuleType('torch.cuda')
    torch_module.cuda.is_available = lambda: False
    torch_module.tensor = lambda x, **kwargs: x
    torch_module.zeros = lambda *args, **kwargs: []
    torch_module.ones = lambda *args, **kwargs: []
    torch_module.sigmoid = lambda x: x
    torch_module.no_grad = lambda: MagicMock()
    
    # Set it in sys.modules to make it importable
    sys.modules['torch'] = torch_module
    sys.modules['torch.nn'] = torch_module.nn
    sys.modules['torch.cuda'] = torch_module.cuda

# Module functions to test - import after torch check
from pore_analysis.core.database import (
    init_db, connect_db, get_module_status, get_product_path, get_all_metrics,
    set_simulation_metadata # Needed for fixture
)

# Only import direct functions if torch is available, otherwise we'll skip the tests
if TORCH_AVAILABLE:
    from pore_analysis.modules.pocket_analysis.computation import run_pocket_analysis
    from pore_analysis.modules.pocket_analysis.visualization import generate_pocket_plots
from pore_analysis.core import config as core_config # For config values

# --- Fixture Setup ---

@pytest.fixture
def setup_pocket_test_db(tmp_path):
    """
    Sets up a temporary run directory with an initialized database,
    dummy ML model files, and necessary metadata for pocket analysis tests.
    """
    run_dir = tmp_path / "pocket_test_run"
    run_dir.mkdir()
    db_path = run_dir / "analysis_registry.db"

    # Create dummy ML model dir and files
    ml_dir = run_dir / "pocket_analysis" / "ml_model"
    ml_dir.mkdir(parents=True)
    # Dummy model weights (empty file)
    (ml_dir / "pocket_model.pth").touch()
    # Dummy model config JSON
    dummy_model_config = {
      "num_subunits": 4, "subunit_embedding_dim": 16, "rmsf_embedding_dim": 16,
      "label_embedding_dim": 2, "hidden_channels": 64, "num_layers": 3,
      "num_heads": 4, "num_rbf": 16, "rbf_type": "expnorm", "activation": "silu",
      "attn_activation": "silu", "neighbor_embedding": True, "cutoff_lower": 0.0,
      "cutoff_upper": 5.0, "max_z": 100, "max_num_neighbors": 32,
      "distance_influence": "both", "attn_dropout": 0.1, "dropout": 0.1
      # Add other keys expected by CustomTorchMD_ET if necessary
    }
    with open(ml_dir / "pocket_model_config.json", 'w') as f:
        json.dump(dummy_model_config, f)

    # Initialize DB
    conn = init_db(str(run_dir))
    assert conn is not None

    # Add required metadata (filter_residues_dict from ion_analysis)
    dummy_filter_map = {'PROA': [1, 2, 3, 4, 5], 'PROB': [10, 11, 12, 13, 14],
                        'PROC': [20, 21, 22, 23, 24], 'PROD': [30, 31, 32, 33, 34]}
    set_simulation_metadata(conn, 'filter_residues_dict', json.dumps(dummy_filter_map))
    # Add a dummy FRAMES_PER_NS for time calculations if core_config isn't available
    set_simulation_metadata(conn, 'FRAMES_PER_NS', getattr(core_config, 'FRAMES_PER_NS', 10.0))
    conn.commit()

    # Yield run_dir and conn for use in tests
    yield str(run_dir), conn

    # Teardown: Close connection
    if conn:
        conn.close()

# --- Mock Objects ---

if TORCH_AVAILABLE:
    # Mock the ML model's forward pass or predict_and_assign_pockets
    # This needs to return data structures that mimic the real output
    # For predict_and_assign_pockets, it should return: pocket_assignments, positive_water_indices
    def mock_predict_and_assign_pockets(*args, **kwargs):
        # args[0] is frame_data
        frame_data = args[0]
        num_waters = frame_data.is_water.sum().item()
        # Simple mock: assign first half of waters to pocket 0, second half to pocket 1
        pocket_assignments = {0: [], 1: [], 2: [], 3: []}
        positive_indices = []
        water_indices = frame_data.atom_indices[frame_data.is_water.bool()].cpu().numpy()

        if num_waters > 0:
            positive_indices = water_indices # Pretend all waters are predicted positive
            split_point = num_waters // 2
            pocket_assignments[0] = list(water_indices[:split_point])
            pocket_assignments[1] = list(water_indices[split_point:])

        return pocket_assignments, positive_indices

    # Mock the model class itself if needed, especially its __init__ and forward
    class MockPocketModel(torch.nn.Module):
         def __init__(self, *args, **kwargs):
             super().__init__()
             # Add any necessary attributes if computation.py checks them
             print("Initialized MockPocketModel") # For debugging tests

         def forward(self, z, pos, batch, subunit_ids, atom_indices, is_water, rmsf, labels=None):
             # Return dummy logits matching expected shape: (num_atoms, 1)
             num_atoms = z.shape[0]
             # Mock logits: positive for waters assigned to pockets 0/1 by our mock_predict
             mock_logits = torch.zeros(num_atoms, 1, device=z.device)
             # This logic needs refinement based on how predict_and_assign uses model output
             # Simpler: Assume mock_predict_and_assign bypasses the forward call entirely via patching
             return mock_logits, is_water, labels
else:
    # Create dummy mocks that won't be used since we'll skip the tests
    def mock_predict_and_assign_pockets(*args, **kwargs):
        return {}, []
    
    # Use the DummyModule created earlier as MockPocketModel
    MockPocketModel = DummyModule

# Mock MDAnalysis Universe and Trajectory to avoid reading large files
class MockTrajectory:
    def __init__(self, n_frames):
        self.n_frames = n_frames
        self._pos = np.random.rand(10, 3) * 10 # Dummy positions for 10 atoms

    def __len__(self):
        return self.n_frames

    def __iter__(self):
        for i in range(self.n_frames):
            yield self[i]

    def __getitem__(self, frame):
        # Return a basic timestep object with frame attribute and potentially positions
        ts = MagicMock()
        ts.frame = frame
        # Add dummy dimensions if needed by distance calcs inside analysis
        ts.dimensions = np.array([50.0, 50.0, 50.0, 90.0, 90.0, 90.0], dtype=np.float32)
        return ts

class MockAtomGroup:
    def __init__(self, n_atoms=10, n_residues=3, name="MOCK"):
        self.n_atoms = n_atoms
        self.n_residues = n_residues
        # Ensure indices is a NumPy array or list
        self.indices = np.arange(n_atoms) # Example indices
        self.atoms = [MagicMock(index=i, name=f"{name}{i}", resid=i//(n_atoms//n_residues)+1, resname="MOL", segid="PROA") for i in range(n_atoms)]
        # Need a basic positions attribute
        self.positions = np.random.rand(n_atoms, 3) * 10
        # Add residues attribute
        self.residues = [MagicMock(resid=i+1, resname="MOL") for i in range(n_residues)]

    def __len__(self):
        return self.n_atoms

    def __getitem__(self, idx):
        # Allow slicing/indexing if necessary
        if isinstance(idx, int):
            return self.atoms[idx]
        elif isinstance(idx, slice):
             # Return a new MockAtomGroup representing the slice
             new_group = MockAtomGroup(n_atoms=len(self.atoms[idx]), name=f"{self.atoms[0].name}_slice")
             new_group.atoms = self.atoms[idx]
             new_group.indices = self.indices[idx]
             new_group.positions = self.positions[idx]
             # Simplification: residue info might not be accurate after slicing like this
             new_group.n_residues = self.n_residues # Keep original residue count? Or recalculate?
             new_group.residues = self.residues # Keep original residues?
             return new_group
        # Handle boolean mask if needed by selection logic
        elif isinstance(idx, (np.ndarray, torch.Tensor)) and idx.dtype == bool:
             # Filter based on boolean mask
             selected_atoms = [atom for i, atom in enumerate(self.atoms) if idx[i]]
             selected_indices = [index for i, index in enumerate(self.indices) if idx[i]]
             selected_positions = self.positions[idx]
             new_group = MockAtomGroup(n_atoms=len(selected_atoms), name=f"{self.atoms[0].name}_masked")
             new_group.atoms = selected_atoms
             new_group.indices = np.array(selected_indices)
             new_group.positions = selected_positions
             # Simplification: residue info might not be accurate after masking
             new_group.n_residues = self.n_residues
             new_group.residues = self.residues
             return new_group

        raise TypeError("MockAtomGroup index must be int, slice, or boolean mask")

    def select_atoms(self, selection_string):
        # Basic mock selection - return self or a smaller mock group
        # This needs to be smarter based on expected selections if tests fail
        if "protein" in selection_string or "backbone" in selection_string or "segid" in selection_string:
            # Return a slightly smaller group to simulate selection
             return self.__getitem__(slice(0, max(1, self.n_atoms // 2)))
        elif "name CA" in selection_string: # Mock finding CA atoms
             ca_group = MockAtomGroup(n_atoms=self.n_residues, name="CA") # Assume 1 CA per residue
             ca_group.residues = self.residues
             return ca_group
        elif "name OH2" in selection_string or "type OW" in selection_string: # Mock finding water
            # Return a mock water group - adjust size as needed for tests
            water_group = MockAtomGroup(n_atoms=50, name="OW", n_residues=50) # Assume 1 atom per res for water
            water_group.residues = [MagicMock(resid=i+100, resname="TIP3") for i in range(50)] # Different resids
            return water_group
        else:
            return self # Default: return self

    def center_of_mass(self):
        return np.mean(self.positions, axis=0) if self.n_atoms > 0 else np.array([0.0, 0.0, 0.0])

class MockUniverse:
    def __init__(self, n_frames=10, n_atoms=100):
        self.trajectory = MockTrajectory(n_frames)
        # Create a basic AtomGroup
        self.atoms = MockAtomGroup(n_atoms=n_atoms, n_residues=n_atoms//5) # Example: 5 atoms per residue
        # Add dummy dimensions if needed by distance calcs
        self.dimensions = np.array([50.0, 50.0, 50.0, 90.0, 90.0, 90.0], dtype=np.float32)


    def select_atoms(self, selection_string):
        # Delegate to the main AtomGroup's mock select_atoms
        return self.atoms.select_atoms(selection_string)

# --- Test Functions ---

# Explicitly mark tests as skipped if torch is not available
if TORCH_AVAILABLE:
    @patch('pore_analysis.modules.pocket_analysis.computation.mda.Universe', return_value=MockUniverse(n_frames=20)) # Patch Universe loading
    @patch('pore_analysis.modules.pocket_analysis.computation.load_analysis_model', return_value=MockPocketModel()) # Patch Model loading
    @patch('pore_analysis.modules.pocket_analysis.computation.predict_and_assign_pockets', side_effect=mock_predict_and_assign_pockets) # Patch prediction logic
    def test_run_pocket_analysis_success(mock_predict, mock_load_model, mock_mda_universe, setup_pocket_test_db):
        """Test successful run of pocket_analysis computation."""
        run_dir, db_conn = setup_pocket_test_db
        psf_file = os.path.join(run_dir, "dummy.psf") # Dummy paths, content not read due to patching
        dcd_file = os.path.join(run_dir, "dummy.dcd")
        
        # Create empty files
        with open(os.path.join(run_dir, "dummy.psf"), 'w') as f:
            f.write("dummy psf")
        with open(os.path.join(run_dir, "dummy.dcd"), 'w') as f:
            f.write("dummy dcd")

        results = run_pocket_analysis(run_dir, psf_file, dcd_file, db_conn)

        # Assertions
        assert results['status'] == 'success', f"Expected success, got {results['status']} with error: {results.get('error')}"
        assert get_module_status(db_conn, "pocket_analysis") == "success"

        # Check for file creation and registration
        expected_products = {
            "pocket_occupancy_timeseries": "pocket_analysis/pocket_occupancy.csv",
            "pocket_residence_stats": "pocket_analysis/pocket_residence_stats.json",
            "pocket_assignments_per_frame": "pocket_analysis/pocket_assignments.pkl",
            "pocket_imbalance_metrics": "pocket_analysis/pocket_imbalance_metrics.csv",
            "pocket_analysis_summary": "pocket_analysis/pocket_summary.txt",
        }
        for subcat, expected_rel_path in expected_products.items():
            abs_path = os.path.join(run_dir, expected_rel_path)
            assert os.path.exists(abs_path), f"Output file missing: {abs_path}"
            registered_path = get_product_path(db_conn, None, "data", subcat, "pocket_analysis") # Check data category
            if registered_path is None and subcat == "pocket_analysis_summary": # Check summary category for txt
                 registered_path = get_product_path(db_conn, None, "summary", subcat, "pocket_analysis")

            assert registered_path == expected_rel_path, f"Product '{subcat}' registration mismatch: Expected '{expected_rel_path}', Got '{registered_path}'"

        # Check for metric storage (example metrics)
        metrics = get_all_metrics(db_conn, "pocket_analysis")
        assert "PocketA_MeanOccupancy" in metrics, "PocketA_MeanOccupancy metric missing"
        assert "PocketB_MedianResidence_ns" in metrics, "PocketB_MedianResidence_ns metric missing"
        assert "CV_of_Mean_Residence_Times" in metrics, "CV_of_Mean_Residence_Times metric missing"
        # Check if pairwise KS was stored (might be NaN if mock data was bad, but key should exist)
        assert "PocketWater_KS_A_B" in metrics, "Pairwise KS metric PocketWater_KS_A_B missing"

    @patch('pore_analysis.modules.pocket_analysis.computation.mda.Universe', return_value=MockUniverse(n_frames=20))
    @patch('pore_analysis.modules.pocket_analysis.computation.load_analysis_model', return_value=MockPocketModel())
    @patch('pore_analysis.modules.pocket_analysis.computation.predict_and_assign_pockets', side_effect=mock_predict_and_assign_pockets)
    def test_generate_pocket_plots_success(mock_predict, mock_load_model, mock_mda_universe, setup_pocket_test_db):
        """Test successful run of pocket_analysis visualization."""
        run_dir, db_conn = setup_pocket_test_db
        psf_file = os.path.join(run_dir, "dummy.psf") # Dummy paths
        dcd_file = os.path.join(run_dir, "dummy.dcd")
        
        # Create empty files
        with open(os.path.join(run_dir, "dummy.psf"), 'w') as f:
            f.write("dummy psf")
        with open(os.path.join(run_dir, "dummy.dcd"), 'w') as f:
            f.write("dummy dcd")

        # 1. Run computation (mocked) to generate data files and register products
        comp_results = run_pocket_analysis(run_dir, psf_file, dcd_file, db_conn)
        assert comp_results['status'] == 'success'
        assert get_module_status(db_conn, "pocket_analysis") == "success"

        # 2. Run visualization
        viz_results = generate_pocket_plots(run_dir, db_conn)

        # Assertions
        assert viz_results['status'] == 'success', f"Visualization failed: {viz_results.get('error')}"
        assert get_module_status(db_conn, "pocket_analysis_visualization") == "success"

        # Check plot file creation and registration
        expected_plots = {
            "pocket_occupancy_plot": "pocket_analysis/pocket_occupancy_plot.png",
            "pocket_residence_histogram": "pocket_analysis/pocket_residence_distribution.png",
        }
        for subcat, expected_rel_path in expected_plots.items():
            abs_path = os.path.join(run_dir, expected_rel_path)
            assert os.path.exists(abs_path), f"Plot file missing: {abs_path}"
            registered_path = get_product_path(db_conn, "png", "plot", subcat, "pocket_analysis_visualization")
            assert registered_path == expected_rel_path, f"Plot '{subcat}' registration mismatch: Expected '{expected_rel_path}', Got '{registered_path}'"
else:
    # Create dummy test functions that are skipped when PyTorch is not available
    @pytest.mark.skip(reason="PyTorch is required for pocket analysis tests")
    def test_run_pocket_analysis_success():
        """Skipped test: successful run of pocket_analysis computation."""
        pytest.skip("PyTorch is required for pocket analysis tests")
        
    @pytest.mark.skip(reason="PyTorch is required for pocket analysis tests")  
    def test_generate_pocket_plots_success():
        """Skipped test: successful run of pocket_analysis visualization."""
        pytest.skip("PyTorch is required for pocket analysis tests")

# Add more tests for failure cases:
# - test_run_pocket_analysis_failure_no_filter_map: remove filter_residues_dict setup
# - test_run_pocket_analysis_failure_model_load: mock load_analysis_model to raise Exception
# - test_generate_pocket_plots_skipped: set computation status to 'failed' before calling viz
# - test handling of empty trajectory / insufficient data frames