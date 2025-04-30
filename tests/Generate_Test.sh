#!/bin/bash

# Create the target directory if it doesn't exist
mkdir -p tests
echo "Creating updated test files in ./tests/"

# === test_utils.py ===
# (Largely compatible, minor updates if needed)
cat << 'EOF' > tests/test_utils.py
import numpy as np
import math
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend for tests
import matplotlib.pyplot as plt
import base64
import pytest
import os # Added for setting config env var if needed

# Assuming core utilities are importable
from pore_analysis.core.utils import (
    frames_to_time,
    OneLetter,
    fig_to_base64,
    clean_json_data,
)
# Import config directly to test FRAMES_PER_NS interaction
from pore_analysis.core import config as core_config

# Helper to reset config value if changed by monkeypatch
original_frames_per_ns = core_config.FRAMES_PER_NS

@pytest.fixture(autouse=True)
def reset_config():
    """Reset FRAMES_PER_NS after each test."""
    yield
    core_config.FRAMES_PER_NS = original_frames_per_ns

def test_frames_to_time_basic():
    # Test with the actual config value
    arr = frames_to_time([0, core_config.FRAMES_PER_NS, core_config.FRAMES_PER_NS * 2])
    assert np.allclose(arr, np.array([0.0, 1.0, 2.0]))
    # Test with a specific value if config is 0 or invalid initially
    core_config.FRAMES_PER_NS = 20.0
    arr_20 = frames_to_time([0, 20, 40])
    assert np.allclose(arr_20, np.array([0.0, 1.0, 2.0]))


def test_frames_to_time_invalid(monkeypatch):
    # Set FRAMES_PER_NS directly on the imported config module
    monkeypatch.setattr(core_config, 'FRAMES_PER_NS', 0)
    with pytest.raises(ValueError, match="FRAMES_PER_NS must be positive"):
        frames_to_time([0, 1, 2])

    monkeypatch.setattr(core_config, 'FRAMES_PER_NS', -10)
    with pytest.raises(ValueError, match="FRAMES_PER_NS must be positive"):
        frames_to_time([0, 1, 2])

def test_oneletter_basic():
    assert OneLetter('CYSASPGLY') == 'CDG'
    assert OneLetter('TRPVALGLUALA') == 'WVGA'
    assert OneLetter('HIS') == 'H' # Test single
    assert OneLetter('HSE') == 'H' # Test variant

def test_oneletter_case_insensitive():
    assert OneLetter('cysAspgLy') == 'CDG'

def test_oneletter_error_unknown():
    with pytest.raises(ValueError, match="Unknown amino acid code 'XXX'"):
        OneLetter('GLYXXXALA')

def test_oneletter_error_length():
    with pytest.raises(ValueError, match="Input length.*must be a multiple of three"):
        OneLetter('GLYA')
    with pytest.raises(ValueError, match="Input length.*must be a multiple of three"):
        OneLetter('GL')

def test_fig_to_base64():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    try:
        b64 = fig_to_base64(fig)
        assert isinstance(b64, str)
        assert len(b64) > 100 # Should be a reasonably long string
        # Check if it decodes and looks like PNG header
        data = base64.b64decode(b64)
        assert data.startswith(b'\x89PNG\r\n\x1a\n')
    finally:
        plt.close(fig) # Ensure figure is closed

def test_fig_to_base64_error(monkeypatch):
    # Simulate an error during fig.savefig
    def mock_savefig(*args, **kwargs):
        raise IOError("Simulated save error")

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    monkeypatch.setattr(fig, 'savefig', mock_savefig)
    try:
        b64 = fig_to_base64(fig)
        assert b64 == "", "Should return empty string on error"
    finally:
        plt.close(fig)


def test_clean_json_data_numpy_types():
    data = {
        'int_': np.int64(10),
        'float_': np.float32(3.14),
        'bool_': np.bool_(True),
        'array_': np.array([1, 2, 3]),
        'nan_': np.nan,
        'inf_': np.inf,
        'neg_inf_': -np.inf,
        'nested_array': np.array([np.nan, 4.0])
    }
    cleaned = clean_json_data(data)
    assert cleaned['int_'] == 10 and isinstance(cleaned['int_'], int)
    assert cleaned['float_'] == 3.14 and isinstance(cleaned['float_'], float)
    assert cleaned['bool_'] is True and isinstance(cleaned['bool_'], bool)
    assert cleaned['array_'] == [1, 2, 3] and isinstance(cleaned['array_'], list)
    assert cleaned['nan_'] is None
    assert cleaned['inf_'] is None
    assert cleaned['neg_inf_'] is None
    assert cleaned['nested_array'] == [None, 4.0]

def test_clean_json_data_standard_types():
    data = {
        'a': math.nan,
        'b': [1, 2, math.inf],
        'c': None,
        'd': "string",
        'e': 123,
        'f': 4.56
    }
    cleaned = clean_json_data(data)
    assert cleaned['a'] is None
    assert cleaned['b'] == [1, 2, None]
    assert cleaned['c'] is None
    assert cleaned['d'] == "string"
    assert cleaned['e'] == 123
    assert cleaned['f'] == 4.56

def test_clean_json_data_nested():
    data = {
        'level1': {
            'list1': [np.int16(1), np.nan, {'key': np.array([5.0, np.inf])}]
        },
        'tuple1': (np.float64(9.9), None)
    }
    cleaned = clean_json_data(data)
    assert cleaned == {
        'level1': {
            'list1': [1, None, {'key': [5.0, None]}]
        },
        'tuple1': [9.9, None] # Tuples become lists
    }

def test_clean_json_data_datetime():
    from datetime import datetime
    now = datetime.now()
    data = {'timestamp': now}
    cleaned = clean_json_data(data)
    assert cleaned['timestamp'] == now.isoformat()

EOF
echo "  Updated tests/test_utils.py"

# === test_core_analysis_computation.py ===
# (Covers tests for analyze_trajectory and filter_and_save_data)
# Needs significant rework due to DB integration and changed function signatures
cat << 'EOF' > tests/test_core_analysis_computation.py
import pytest
import os
import numpy as np
import pandas as pd
import sqlite3
from pathlib import Path

# Adapt imports for refactored code
from pore_analysis.core.database import init_db, connect_db, get_module_status, get_product_path, get_all_metrics, get_simulation_metadata
from pore_analysis.modules.core_analysis.computation import analyze_trajectory, filter_and_save_data

# Fixture to create a dummy PSF and DCD file
@pytest.fixture
def dummy_md_files(tmp_path):
    run_dir = tmp_path / "dummy_run"
    run_dir.mkdir()
    psf_content = """PSF

      1 !NTITLE
 REMARKS generated by VMD
 REMARKS topology ../../common/top_all36_prot.rtf
 REMARKS default

       8 !NATOM
       1 ALA  1    ALA  N    NH3    -0.300000       14.0100           0   0.00000      -1.340000E-07
       2 ALA  1    ALA  HT1  HC      0.330000        1.0080           0   0.00000      -1.340000E-07
       3 ALA  1    ALA  HT2  HC      0.330000        1.0080           0   0.00000      -1.340000E-07
       4 ALA  1    ALA  HT3  HC      0.330000        1.0080           0   0.00000      -1.340000E-07
       5 ALA  1    ALA  CA   CT1     0.070000       12.0100           0   0.00000      -1.340000E-07
       6 ALA  1    ALA  HA   HB1     0.090000        1.0080           0   0.00000      -1.340000E-07
       7 ALA  1    ALA  C    C       0.510000       12.0100           0   0.00000      -1.340000E-07
       8 ALA  1    ALA  O    O      -0.510000       16.0000           0   0.00000      -1.340000E-07

       0 !NBOND


       0 !NTHETA


       0 !NPHI


       0 !NIMPHI


       0 !NDON


       0 !NACC


       0 !NNB

       0       0       0       0

       1       1       0 !NGRP
"""
    dcd_header = np.array([84, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 67, 79, 82, 68], dtype=np.int32).tobytes()
    dcd_frame_header = np.array([48], dtype=np.int32).tobytes()
    dcd_coords = np.arange(8*3*2, dtype=np.float32).reshape(2, 8, 3) # 2 frames, 8 atoms
    dcd_frame_footer = np.array([48], dtype=np.int32).tobytes()
    psf_path = run_dir / "step5_input.psf"
    dcd_path = run_dir / "MD_Aligned.dcd"
    with open(psf_path, "w") as f:
        f.write(psf_content)
    with open(dcd_path, "wb") as f:
        f.write(dcd_header)
        f.write(dcd_frame_header)
        f.write(dcd_coords[0].tobytes())
        f.write(dcd_frame_footer)
        f.write(dcd_frame_header)
        f.write(dcd_coords[1].tobytes())
        f.write(dcd_frame_footer)

    return run_dir, psf_path, dcd_path

# === Tests for analyze_trajectory ===

def test_analyze_trajectory_success(dummy_md_files):
    """Test basic successful run of analyze_trajectory (computation only)."""
    run_dir, psf_path, dcd_path = dummy_md_files
    # Initialize DB before running
    conn = init_db(str(run_dir))
    assert conn is not None
    conn.close()

    # Call the function
    results = analyze_trajectory(str(run_dir), psf_file=str(psf_path), dcd_file=str(dcd_path))

    # Assertions
    assert results['status'] == 'success'
    assert 'data' in results
    assert 'dist_ac' in results['data'] # Should exist, even if all NaN due to dummy data
    assert 'dist_bd' in results['data']
    assert 'time_points' in results['data']
    assert 'metadata' in results
    assert results['metadata']['is_control_system'] is True # Dummy data likely won't have toxin segid
    assert 'files' in results
    assert 'raw_distances' in results['files']

    # Check DB status
    conn = connect_db(str(run_dir))
    assert get_module_status(conn, "core_analysis") == "success"
    # Check product registered
    prod_path = get_product_path(conn, "csv", "data", "raw_distances", module_name="core_analysis")
    assert prod_path == "core_analysis/Raw_Distances.csv"
    conn.close()

    # Check file exists
    raw_csv = run_dir / prod_path
    assert raw_csv.exists()

def test_analyze_trajectory_missing_files(tmp_path):
    """Test analyze_trajectory when PSF/DCD are missing."""
    run_dir = tmp_path / "run_missing"
    run_dir.mkdir()
    # No DB needed as it fails before DB ops
    results = analyze_trajectory(str(run_dir))

    assert results['status'] == 'failed'
    assert 'PSF file not found' in results['error']
    # Check the fallback data structure exists but might be empty/default
    assert 'data' in results
    assert 'metadata' in results

    # Test missing DCD
    psf_path = run_dir / "step5_input.psf"
    psf_path.touch() # Create dummy PSF
    results_dcd = analyze_trajectory(str(run_dir))
    assert results_dcd['status'] == 'failed'
    assert 'DCD file not found' in results_dcd['error']

def test_analyze_trajectory_0_frames(tmp_path):
    """Test analyze_trajectory with a 0-frame DCD (requires setup)."""
    run_dir = tmp_path / "run_0frame"
    run_dir.mkdir()
    psf_path = run_dir / "step5_input.psf"
    dcd_path = run_dir / "MD_Aligned.dcd"
    # Create dummy PSF
    psf_path.write_text("PSF\n\n 1 !NATOM\n 1 A 1 A N N 0 14 0\n 0 !NBOND")
    # Create dummy DCD header indicating 0 frames
    dcd_header = np.array([84, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 67, 79, 82, 68], dtype=np.int32).tobytes()
    with open(dcd_path, "wb") as f:
        f.write(dcd_header)

    # Initialize DB
    conn = init_db(str(run_dir))
    conn.close()

    results = analyze_trajectory(str(run_dir))
    assert results['status'] == 'failed'
    assert '0 frames' in results['error']

    conn = connect_db(str(run_dir))
    assert get_module_status(conn, "core_analysis") == "failed"
    conn.close()


# === Tests for filter_and_save_data ===

def test_filter_and_save_data_success(tmp_path):
    """Test successful filtering and saving."""
    run_dir = tmp_path / "run_filter"
    run_dir.mkdir()
    conn = init_db(str(run_dir))

    # Dummy raw data
    time_points = np.linspace(0, 9, 10)
    dist_ac = time_points + np.random.rand(10) * 0.1 # Simple trend + noise
    dist_bd = time_points * 0.8 + np.random.rand(10) * 0.1 + 5.0
    dist_com = time_points * 2 + np.random.rand(10) * 0.2 + 10.0

    # Run filtering
    results = filter_and_save_data(
        run_dir=str(run_dir),
        dist_ac=dist_ac,
        dist_bd=dist_bd,
        com_distances=dist_com,
        time_points=time_points,
        db_conn=conn, # Pass the connection
        box_z=None,
        is_control_system=False
    )

    # Assertions
    assert results['status'] == 'success'
    assert 'data' in results
    assert 'filtered_ac' in results['data'] and len(results['data']['filtered_ac']) == 10
    assert 'filtered_bd' in results['data'] and len(results['data']['filtered_bd']) == 10
    assert 'filtered_com' in results['data'] and len(results['data']['filtered_com']) == 10
    assert 'files' in results
    assert 'g_g_distance_filtered' in results['files']
    assert 'com_stability_filtered' in results['files']
    assert 'errors' in results and not results['errors'] # Should be no errors

    # Check DB
    assert get_module_status(conn, "core_analysis_filtering") == "success"
    gg_path = get_product_path(conn, "csv", "data", "g_g_distance_filtered")
    com_path = get_product_path(conn, "csv", "data", "com_stability_filtered")
    assert gg_path == "core_analysis/G_G_Distance_Filtered.csv"
    assert com_path == "core_analysis/COM_Stability_Filtered.csv"

    # Check metrics were stored
    metrics = get_all_metrics(conn) # Fetch all metrics for simplicity
    assert 'G_G_AC_Mean_Filt' in metrics
    assert 'G_G_BD_Mean_Filt' in metrics
    assert 'COM_Mean_Filt' in metrics
    assert 'G_G_AC_Std_Filt' in metrics
    assert 'COM_Std_Filt' in metrics

    # Check CSV files exist
    gg_csv = run_dir / gg_path
    com_csv = run_dir / com_path
    assert gg_csv.exists()
    assert com_csv.exists()

    conn.close()


def test_filter_and_save_data_control_system(tmp_path):
    """Test filtering for a control system (no COM)."""
    run_dir = tmp_path / "run_filter_control"
    run_dir.mkdir()
    conn = init_db(str(run_dir))

    time_points = np.linspace(0, 9, 10)
    dist_ac = time_points + np.random.rand(10) * 0.1
    dist_bd = time_points * 0.8 + np.random.rand(10) * 0.1 + 5.0

    results = filter_and_save_data(
        run_dir=str(run_dir),
        dist_ac=dist_ac,
        dist_bd=dist_bd,
        com_distances=None, # No COM data
        time_points=time_points,
        db_conn=conn,
        is_control_system=True # Mark as control
    )

    assert results['status'] == 'success'
    assert 'com_stability_filtered' not in results['files'] # COM file shouldn't be created/registered
    assert not results['data']['filtered_com'] # Filtered COM should be empty

    # Check DB
    assert get_module_status(conn, "core_analysis_filtering") == "success"
    com_path = get_product_path(conn, "csv", "data", "com_stability_filtered")
    assert com_path is None # Should not be registered
    metrics = get_all_metrics(conn)
    assert 'COM_Mean_Filt' not in metrics # No COM metrics

    conn.close()


def test_filter_and_save_data_no_db_conn(tmp_path):
    """Test failure when DB connection is not provided."""
    run_dir = tmp_path / "run_filter_nodb"
    run_dir.mkdir()
    # No DB init

    time_points = np.linspace(0, 9, 10)
    dist_ac = time_points + np.random.rand(10) * 0.1
    dist_bd = time_points * 0.8 + np.random.rand(10) * 0.1 + 5.0

    results = filter_and_save_data(
        run_dir=str(run_dir),
        dist_ac=dist_ac,
        dist_bd=dist_bd,
        com_distances=None,
        time_points=time_points,
        db_conn=None, # Pass None for DB connection
        is_control_system=True
    )

    assert results['status'] == 'failed'
    assert 'Database connection was not provided' in results['error']

EOF
echo "  Created tests/test_core_analysis_computation.py (adapted)"

# === test_core_analysis_visualization.py ===
# (New test file for the visualization part)
cat << 'EOF' > tests/test_core_analysis_visualization.py
import pytest
import os
import numpy as np
import pandas as pd
import sqlite3
from pathlib import Path

# Adapt imports for refactored code
from pore_analysis.core.database import init_db, connect_db, register_module, register_product, get_module_status, get_product_path
from pore_analysis.modules.core_analysis.visualization import plot_distances, plot_kde_analysis

# Helper fixture to create dummy filtered CSV files and register them
@pytest.fixture
def setup_core_viz_data(tmp_path):
    run_dir = tmp_path / "viz_run"
    run_dir.mkdir()
    core_dir = run_dir / "core_analysis"
    core_dir.mkdir()

    # --- Create Dummy Data Files ---
    time_points = np.linspace(0, 9, 10)
    # G-G Data
    gg_data = {
        'Time (ns)': time_points,
        'G_G_Distance_AC_Raw': time_points + 1,
        'G_G_Distance_BD_Raw': time_points + 2,
        'G_G_Distance_AC_Filt': time_points + 1.1,
        'G_G_Distance_BD_Filt': time_points + 2.1,
    }
    gg_csv_path = core_dir / "G_G_Distance_Filtered.csv"
    pd.DataFrame(gg_data).to_csv(gg_csv_path, index=False)

    # COM Data (Non-Control)
    com_data = {
        'Time (ns)': time_points,
        'COM_Distance_Raw': time_points * 2 + 10,
        'COM_Distance_Filt': time_points * 2 + 10.1,
    }
    com_csv_path = core_dir / "COM_Stability_Filtered.csv"
    pd.DataFrame(com_data).to_csv(com_csv_path, index=False)

    # --- Initialize and Populate DB ---
    conn = init_db(str(run_dir))
    assert conn is not None

    # Register dummy filtering module as success (needed for viz check)
    register_module(conn, "core_analysis_filtering", status='success')
    module_id = conn.execute("SELECT module_id FROM analysis_modules WHERE module_name = ?", ("core_analysis_filtering",)).fetchone()[0]

    # Register dummy products
    register_product(conn, "core_analysis_filtering", "csv", "data",
                     os.path.relpath(gg_csv_path, run_dir),
                     subcategory="g_g_distance_filtered", description="Dummy G-G")
    register_product(conn, "core_analysis_filtering", "csv", "data",
                     os.path.relpath(com_csv_path, run_dir),
                     subcategory="com_stability_filtered", description="Dummy COM")

    conn.commit()
    conn.close()

    return str(run_dir)

# Test G-G plot generation
def test_plot_gg_distances(setup_core_viz_data):
    run_dir = setup_core_viz_data
    conn = connect_db(run_dir)
    assert conn is not None

    # Register viz module start
    register_module(conn, "core_analysis_visualization_g_g", status='running')
    conn.commit()

    plot_paths_dict = plot_distances(run_dir, is_gg=True, db_conn=conn) # Pass connection

    # Assertions
    assert isinstance(plot_paths_dict, dict)
    assert 'subunit_comparison' in plot_paths_dict
    plot_path = run_dir / plot_paths_dict['subunit_comparison']
    assert plot_path.exists()
    assert plot_path.stat().st_size > 100 # Check file not empty

    # Check DB
    assert get_module_status(conn, "core_analysis_visualization_g_g") == "success"
    prod_path = get_product_path(conn, "png", "plot", "subunit_comparison", module_name="core_analysis_visualization_g_g")
    assert prod_path == "core_analysis/G_G_Distance_Subunit_Comparison.png"

    conn.close()

# Test COM plot generation
def test_plot_com_distances(setup_core_viz_data):
    run_dir = setup_core_viz_data
    conn = connect_db(run_dir)
    assert conn is not None

    register_module(conn, "core_analysis_visualization_com", status='running')
    conn.commit()

    # Note: plot_distances for is_gg=False now ONLY returns the comparison plot path
    plot_paths_dict = plot_distances(run_dir, is_gg=False, db_conn=conn)

    assert isinstance(plot_paths_dict, dict)
    assert 'comparison' in plot_paths_dict
    plot_path = run_dir / plot_paths_dict['comparison']
    assert plot_path.exists()
    assert plot_path.stat().st_size > 100

    # Check DB
    assert get_module_status(conn, "core_analysis_visualization_com") == "success"
    prod_path = get_product_path(conn, "png", "plot", "comparison", module_name="core_analysis_visualization_com")
    assert prod_path == "core_analysis/COM_Stability_Comparison.png"

    conn.close()

# Test KDE plot generation
def test_plot_kde_analysis_viz(setup_core_viz_data):
    run_dir = setup_core_viz_data
    conn = connect_db(run_dir)
    assert conn is not None

    # Viz module status registration happens within plot_kde_analysis if needed

    # plot_kde_analysis implicitly registers its own viz module
    plot_path_str = plot_kde_analysis(run_dir, db_conn=conn)

    assert plot_path_str is not None
    plot_path = Path(plot_path_str)
    assert plot_path.exists()
    assert plot_path.stat().st_size > 100
    assert plot_path.name == "COM_Stability_KDE_Analysis.png"

    # Check DB
    assert get_module_status(conn, "core_analysis_visualization_com") == "success" # Status updated by func
    prod_path = get_product_path(conn, "png", "plot", "kde_analysis", module_name="core_analysis_visualization_com")
    assert prod_path == "core_analysis/COM_Stability_KDE_Analysis.png"

    conn.close()

def test_plot_distances_no_db_conn(setup_core_viz_data):
    """Test that plots fail gracefully if DB connection is None (or fails)."""
    run_dir = setup_core_viz_data
    # Do NOT provide db_conn
    plot_paths_dict_gg = plot_distances(run_dir, is_gg=True, db_conn=None)
    plot_paths_dict_com = plot_distances(run_dir, is_gg=False, db_conn=None)

    assert plot_paths_dict_gg == {} # Expect empty dict on failure
    assert plot_paths_dict_com == {}

def test_plot_kde_no_db_conn(setup_core_viz_data):
    """Test that KDE plot fails gracefully if DB connection is None."""
    run_dir = setup_core_viz_data
    plot_path = plot_kde_analysis(run_dir, db_conn=None)
    assert plot_path is None # Expect None on failure

def test_plot_distances_no_filtering_data(tmp_path):
    """Test plots fail if filtering step didn't run or register products."""
    run_dir = tmp_path / "viz_run_nofilter"
    run_dir.mkdir()
    conn = init_db(str(run_dir))
    # DO NOT register filtering module or products
    conn.commit()
    conn.close()

    conn = connect_db(str(run_dir))
    register_module(conn, "core_analysis_visualization_g_g", status='running')
    conn.commit()
    plot_paths_dict_gg = plot_distances(str(run_dir), is_gg=True, db_conn=conn)
    assert plot_paths_dict_gg == {} # Should fail as product path is missing
    conn.close()

EOF
echo "  Created tests/test_core_analysis_visualization.py (adapted)"

# === test_html_report.py ===
# (Adapted to check DB state and generate report)
cat << 'EOF' > tests/test_html_report.py
import pytest; # pytest.skip("Skipping HTML template tests in CI", allow_module_level=True) # Allow running locally
import numpy as np
import pandas as pd
from pathlib import Path
import base64
import sqlite3

# Adapt imports for refactored code
from pore_analysis.core.database import init_db, connect_db, register_module, register_product, store_metric, set_simulation_metadata
from pore_analysis.html import generate_html_report
from pore_analysis.summary import generate_summary_from_database # Use this to generate summary dict

# Helper function to create a dummy plot file
def create_dummy_png(filepath, size=(100, 50)):
    from PIL import Image
    img = Image.new('RGB', size, color = 'red')
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    img.save(filepath)
    return os.path.relpath(filepath, os.path.dirname(os.path.dirname(filepath)))

@pytest.fixture
def setup_report_db(tmp_path):
    """Sets up a temporary run directory with a populated database for report generation."""
    run_dir = tmp_path / "report_run"
    run_dir.mkdir()
    conn = init_db(str(run_dir))

    # --- Populate DB with minimal data for the overview tab ---
    set_simulation_metadata(conn, "run_name", "report_run")
    set_simulation_metadata(conn, "system_name", "test_system")
    set_simulation_metadata(conn, "analysis_status", "success")
    set_simulation_metadata(conn, "is_control_system", "True") # Test control system case

    # Register core filtering module success
    register_module(conn, "core_analysis_filtering", status='success')
    module_id_filt = conn.execute("SELECT module_id FROM analysis_modules WHERE module_name = ?", ("core_analysis_filtering",)).fetchone()[0]

    # Store some metrics
    store_metric(conn, "core_analysis_filtering", 'G_G_AC_Mean_Filt', 1.0, 'Å', 'Desc AC Mean')
    store_metric(conn, "core_analysis_filtering", 'G_G_BD_Mean_Filt', 2.0, 'Å', 'Desc BD Mean')
    store_metric(conn, "core_analysis_filtering", 'G_G_AC_Std_Filt', 0.1, 'Å', 'Desc AC Std')
    store_metric(conn, "core_analysis_filtering", 'G_G_BD_Std_Filt', 0.2, 'Å', 'Desc BD Std')
    store_metric(conn, "core_analysis_filtering", 'G_G_AC_Min_Filt', 0.5, 'Å', 'Desc AC Min')
    store_metric(conn, "core_analysis_filtering", 'G_G_BD_Min_Filt', 0.6, 'Å', 'Desc BD Min')
    store_metric(conn, "core_analysis_filtering", 'G_G_AC_Max_Filt', 1.5, 'Å', 'Desc AC Max')
    store_metric(conn, "core_analysis_filtering", 'G_G_BD_Max_Filt', 2.6, 'Å', 'Desc BD Max')
    # No COM metrics needed for control system test

    # Create and register a dummy plot (required by plots_dict.json for overview)
    core_viz_dir = run_dir / "core_analysis"
    core_viz_dir.mkdir()
    dummy_gg_plot_path = core_viz_dir / "G_G_Distance_Subunit_Comparison.png"
    # Need Pillow: pip install Pillow
    try:
        import os
        rel_plot_path = create_dummy_png(dummy_gg_plot_path)
        register_module(conn, "core_analysis_visualization_g_g", status='success')
        module_id_viz_gg = conn.execute("SELECT module_id FROM analysis_modules WHERE module_name = ?", ("core_analysis_visualization_g_g",)).fetchone()[0]
        register_product(conn, "core_analysis_visualization_g_g", "png", "plot", rel_plot_path,
                         subcategory="subunit_comparison", description="Dummy G-G Plot")
    except ImportError:
        print("Warning: Pillow not installed, cannot create dummy PNG for test_html_report.")
        # Don't register plot if creation failed

    conn.commit()
    conn.close()
    return str(run_dir)


def test_generate_html_report_basic(setup_report_db):
    """Test basic HTML report generation for a control system."""
    run_dir = setup_report_db

    # Generate HTML report (generate_html_report will connect to the DB)
    # It now optionally takes summary, but we test generation from DB
    report_path = generate_html_report(run_dir, summary=None)

    assert report_path is not None, "Expected a report path"
    assert os.path.exists(report_path)
    assert Path(report_path).name == "data_analysis_report.html"

    html = Path(report_path).read_text(encoding='utf-8')

    # --- Basic Checks ---
    assert "<h1>MD Analysis Report</h1>" in html
    assert "report_run" in html # Check run name is present
    assert "test_system" in html # Check system name (from metadata)
    assert "CONTROL SYSTEM (NO TOXIN)" in html # Check control banner
    assert "Overview &amp; Distances" in html # Check tab name (HTML escaped)

    # Check if G-G stats are rendered in the overview table
    assert "1.000" in html # G_G_AC_Mean_Filt
    assert "2.000" in html # G_G_BD_Mean_Filt
    assert "0.100" in html # G_G_AC_Std_Filt
    assert "0.600" in html # G_G_BD_Min_Filt

    # Check if COM section indicates control system
    assert "COM distance analysis is not applicable." in html

    # Check if dummy plot was embedded (look for base64 image data start)
    # This check is brittle and depends on plots_dict.json setup matching the fixture registration
    # It might fail if Pillow wasn't available in the fixture.
    if Path(run_dir) / "core_analysis" / "G_G_Distance_Subunit_Comparison.png":
        assert "data:image/png;base64," in html
        assert 'alt="G-G Distance Subunit Comparison (Raw vs Filtered)"' in html
    else:
        print("Skipping plot embed check as dummy PNG likely wasn't created.")


# Add more tests:
# - Non-control system (populate COM metrics/plots)
# - Test presence/absence of plots based on module status/product registration
# - Test rendering of metrics from various modules in their respective tabs

EOF
echo "  Created tests/test_html_report.py (adapted)"

# === test_html_report_full.py (Adaptation) ===
# This tested passing a summary dict. We'll adapt it to test report generation
# from a DB populated with more complete data (non-control, more modules).
cat << 'EOF' > tests/test_html_report_full.py
import pytest; # pytest.skip("Skipping HTML template tests in CI", allow_module_level=True)
import numpy as np
import pandas as pd
from pathlib import Path
import base64
import sqlite3
import os
import json

# Adapt imports for refactored code
from pore_analysis.core.database import init_db, connect_db, register_module, register_product, store_metric, set_simulation_metadata
from pore_analysis.html import generate_html_report
from pore_analysis.summary import generate_summary_from_database # Use this to generate summary dict

# Helper function to create a dummy plot file
def create_dummy_png(filepath, size=(100, 50)):
    try:
        from PIL import Image
        img = Image.new('RGB', size, color = 'green')
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        img.save(filepath)
        # Return relative path assuming run_dir is the parent of the module dir
        return os.path.relpath(filepath, os.path.dirname(os.path.dirname(filepath)))
    except ImportError:
        print(f"Warning: Pillow not installed, cannot create dummy PNG: {filepath}")
        return None

@pytest.fixture
def setup_full_report_db(tmp_path):
    """Sets up a database with multiple modules completed for a non-control system."""
    run_dir = tmp_path / "full_report_run"
    run_dir.mkdir()
    conn = init_db(str(run_dir))

    # --- Populate DB ---
    set_simulation_metadata(conn, "run_name", "full_report_run")
    set_simulation_metadata(conn, "system_name", "toxin_complex")
    set_simulation_metadata(conn, "analysis_status", "success")
    set_simulation_metadata(conn, "is_control_system", "False") # Non-control

    # Register modules as successful
    register_module(conn, "core_analysis_filtering", status='success')
    register_module(conn, "orientation_analysis", status='success')
    register_module(conn, "ion_analysis", status='success') # Needed for ion metrics/plots
    register_module(conn, "inner_vestibule_analysis", status='success')
    register_module(conn, "gyration_analysis", status='success')
    register_module(conn, "tyrosine_analysis", status='success')
    register_module(conn, "dw_gate_analysis", status='success')
    # Visualization modules
    register_module(conn, "core_analysis_visualization_g_g", status='success')
    register_module(conn, "core_analysis_visualization_com", status='success')
    register_module(conn, "orientation_analysis_visualization", status='success')
    register_module(conn, "ion_analysis_visualization", status='success')
    register_module(conn, "inner_vestibule_analysis_visualization", status='success')
    register_module(conn, "gyration_analysis_visualization", status='success')
    register_module(conn, "tyrosine_analysis_visualization", status='success')
    register_module(conn, "dw_gate_analysis_visualization", status='success')

    # --- Store Metrics (representative subset) ---
    # Core
    store_metric(conn, "core_analysis_filtering", 'G_G_AC_Mean_Filt', 1.1, 'Å', '')
    store_metric(conn, "core_analysis_filtering", 'G_G_BD_Mean_Filt', 2.1, 'Å', '')
    store_metric(conn, "core_analysis_filtering", 'COM_Mean_Filt', 10.5, 'Å', '')
    store_metric(conn, "core_analysis_filtering", 'COM_Std_Filt', 0.5, 'Å', '')
    # Orientation
    store_metric(conn, "orientation_analysis", 'Orient_Angle_Mean', 45.0, '°', '')
    store_metric(conn, "orientation_analysis", 'Orient_Contacts_Mean', 55.0, 'count', '')
    # Ions
    for site in ["S0","S1","S2","S3","S4","Cavity"]:
        store_metric(conn, "ion_analysis", f"Ion_AvgOcc_{site}", 0.1 + int(site[1] if len(site)>1 else 5) * 0.05, 'count', '')
        store_metric(conn, "ion_analysis", f"Ion_MaxOcc_{site}", 1, 'count', '')
        store_metric(conn, "ion_analysis", f"Ion_PctTimeOcc_{site}", 10.0 + int(site[1] if len(site)>1 else 5) * 5, '%', '')
    store_metric(conn, "ion_analysis", 'Ion_HMM_ConductionEvents_Total', 5, 'count','')
    store_metric(conn, "ion_analysis", 'Ion_HMM_Transition_S1_S0', 2, 'count','')
    # Inner Vestibule
    store_metric(conn, "inner_vestibule_analysis", 'InnerVestibule_MeanOcc', 3.5, 'count', '')
    store_metric(conn, "inner_vestibule_analysis", 'InnerVestibule_AvgResidenceTime_ns', 0.15, 'ns', '')
    # Gyration
    store_metric(conn, "gyration_analysis", 'Gyration_G1_Mean', 3.0, 'Å', '')
    store_metric(conn, "gyration_analysis", 'Gyration_Y_Mean', 3.2, 'Å', '')
    store_metric(conn, "gyration_analysis", 'Gyration_G1_OnFlips', 2, 'count', '')
    store_metric(conn, "gyration_analysis", 'Gyration_Y_OnFlips', 1, 'count', '')
    # Tyrosine
    store_metric(conn, "tyrosine_analysis", 'Tyr_HMM_TotalTransitions', 15, 'count', '')
    store_metric(conn, "tyrosine_analysis", 'Tyr_HMM_Population_pp', 85.0, '%', '')
    store_metric(conn, "tyrosine_analysis", 'Tyr_HMM_MeanDwell_pp', 10.2, 'ns', '')
    set_simulation_metadata(conn, 'Tyr_HMM_DominantState', 'pp') # Use metadata for dominant state
    # DW Gate
    store_metric(conn, "dw_gate_analysis", 'DW_PROA_Closed_Fraction', 75.0, '%', '')
    store_metric(conn, "dw_gate_analysis", 'DW_PROA_open_Mean_ns', 2.5, 'ns', '')

    # --- Create and Register Dummy Plots (matching plots_dict.json entries) ---
    # Load plot definitions to register required plots
    plots_dict_path = Path(__file__).parent.parent / "pore_analysis" / "plots_dict.json"
    plot_keys_needed = set()
    if plots_dict_path.exists():
        try:
            with open(plots_dict_path, 'r') as f_plots:
                plot_defs = json.load(f_plots)
                for pdef in plot_defs:
                    plot_keys_needed.add((pdef['module_name'], pdef['category'], pdef['subcategory'], pdef['template_key']))
        except Exception as e_json:
            print(f"Warning: Could not load plots_dict.json: {e_json}")

    # Mapping template_key to expected filename (based on subcategory where possible)
    key_to_filename = {
        "subunit_comparison": "G_G_Distance_Subunit_Comparison.png",
        "comparison": "COM_Stability_Comparison.png",
        "com_kde": "COM_Stability_KDE_Analysis.png",
        "orientation_angle": "Toxin_Orientation_Angle.png",
        "rotation_components": "Toxin_Rotation_Components.png",
        "channel_contacts": "Toxin_Channel_Contacts.png",
        "contact_map_focused": "Toxin_Channel_Residue_Contact_Map_Focused.png",
        "k_ion_combined_plot": "K_Ion_Combined_Plot.png",
        "k_ion_occupancy_heatmap": "K_Ion_Occupancy_Heatmap.png",
        "k_ion_average_occupancy": "K_Ion_Average_Occupancy.png",
        "binding_sites_g1_centric_visualization": "binding_sites_g1_centric_visualization.png",
        "hmm_transitions_plot": "ion_transitions_hmm.png",
        "site_optimization_plot": "binding_site_optimization.png", # Registered by structure
        "inner_vestibule_count_plot": "inner_vestibule_count_plot.png",
        "inner_vestibule_residence_hist": "inner_vestibule_residence_hist.png",
        "g1_gyration_radii": "G1_gyration_radii_stacked.png",
        "y_gyration_radii": "Y_gyration_radii_stacked.png",
        "flip_duration_distribution": "Flip_Duration_Distribution.png",
        "sf_tyrosine_chi1_dihedrals": "SF_Tyrosine_Chi1_Dihedrals_HMM.png",
        "sf_tyrosine_chi2_dihedrals": "SF_Tyrosine_Chi2_Dihedrals_HMM.png",
        "sf_tyrosine_rotamer_scatter": "SF_Tyrosine_Rotamer_Scatter.png",
        "sf_tyrosine_rotamer_population": "SF_Tyrosine_Rotamer_Population_HMM.png",
        "dw_distance_distribution": "dw_gate_distance_distribution.png",
        "dw_distance_vs_state": "dw_gate_distance_vs_state.png",
        "dw_open_probability": "dw_gate_open_probability.png",
        "dw_state_heatmap": "dw_gate_state_heatmap.png",
        "dw_duration_distributions": "dw_gate_duration_distribution.png"
    }

    # Register necessary plots to satisfy html.py and plots_dict.json
    for module_name, category, subcategory, template_key in plot_keys_needed:
        filename = key_to_filename.get(template_key)
        if filename:
            # Determine module subdirectory path (handle core vs other modules)
            if module_name.startswith("core"): module_dir_name = "core_analysis"
            elif module_name.startswith("orientation"): module_dir_name = "orientation_contacts"
            elif module_name.startswith("ion"): module_dir_name = "ion_analysis"
            elif module_name.startswith("inner_vestibule"): module_dir_name = "inner_vestibule_analysis"
            elif module_name.startswith("gyration"): module_dir_name = "gyration_analysis"
            elif module_name.startswith("tyrosine"): module_dir_name = "tyrosine_analysis"
            elif module_name.startswith("dw_gate"): module_dir_name = "dw_gate_analysis"
            else: module_dir_name = "unknown_module" # Fallback

            # Create dummy file and register it
            module_output_dir = run_dir / module_dir_name
            dummy_plot_path = module_output_dir / filename
            rel_plot_path = create_dummy_png(dummy_plot_path) # Returns None if Pillow missing
            if rel_plot_path:
                 # Ensure module is registered if not already
                cursor = conn.cursor()
                cursor.execute("SELECT module_id FROM analysis_modules WHERE module_name = ?", (module_name,))
                mod_exists = cursor.fetchone()
                if not mod_exists:
                    register_module(conn, module_name, status='success')

                register_product(conn, module_name, "png", category, rel_plot_path,
                                 subcategory=subcategory, description=f"Dummy plot for {template_key}")
            else:
                 print(f"Skipping registration for {template_key} as dummy PNG failed.")


    conn.commit()
    conn.close()
    return str(run_dir)


def test_generate_html_report_full(setup_full_report_db):
    """Test HTML report generation with a more complete dataset."""
    run_dir = setup_full_report_db

    # Generate report
    report_path = generate_html_report(run_dir, summary=None) # Generate from DB

    assert report_path is not None
    assert Path(report_path).exists()

    html = Path(report_path).read_text(encoding='utf-8')

    # Check for presence of various section titles
    assert "Overview &amp; Distances" in html
    assert "Toxin Interface" in html
    assert "Pore Ion Analysis" in html
    assert "Inner Vestibule Water Analysis" in html
    assert "Carbonyl Dynamics" in html
    assert "SF Tyrosine Analysis" in html
    assert "DW Gate Dynamics" in html

    # Check for specific metrics rendered
    assert "10.500" in html # COM Mean Filt
    assert "45.00" in html # Orient Angle Mean
    assert "Ion_AvgOcc_S0" in html # Check raw key presence before rendering check
    # Check rendered occupancy value (adjust precision as needed)
    assert ">0.10<" in html or ">0.1<" in html # Ion_AvgOcc_S0
    assert "3.50" in html # InnerVestibule_MeanOcc
    assert "2" in html # Gyration G1 On Flips count check
    assert ">pp<" in html # Dominant Tyrosine state (check metadata render)
    assert "75.0%" in html # DW Gate Closed Fraction

    # Check if plots were embedded (presence of base64 image data)
    # Requires Pillow in fixture and plots_dict.json to be correct
    if "data:image/png;base64," in html:
        print("Base64 image data found in report.")
    else:
        print("Warning: No base64 image data found in report (check plots_dict.json and fixture PNG creation).")

EOF
echo "  Created tests/test_html_report_full.py (adapted)"

# === test_html_report_error.py (Adaptation) ===
# Test how html.py handles template errors (e.g., missing essential DB data)
# We can simulate this by creating a DB but NOT populating required metrics/plots
cat << 'EOF' > tests/test_html_report_error.py
import pytest; # pytest.skip("Skipping HTML template tests in CI", allow_module_level=True)
from pathlib import Path
import sqlite3

# Adapt imports for refactored code
from pore_analysis.core.database import init_db, connect_db, set_simulation_metadata
from pore_analysis.html import generate_html_report

@pytest.fixture
def setup_incomplete_db(tmp_path):
    """Sets up a database missing required data for the HTML template."""
    run_dir = tmp_path / "error_report_run"
    run_dir.mkdir()
    conn = init_db(str(run_dir))

    # Add only minimal metadata, deliberately missing metrics/plots needed by template
    set_simulation_metadata(conn, "run_name", "error_report_run")
    set_simulation_metadata(conn, "is_control_system", "False")

    conn.commit()
    conn.close()
    return str(run_dir)

def test_generate_html_report_template_error(setup_incomplete_db):
    """Test HTML generation when required template variables are missing from DB/summary."""
    run_dir = setup_incomplete_db

    # generate_html_report should ideally catch template errors (KeyError, etc.)
    # In the refactored code, it fetches data itself. If fetching fails or
    # required data is missing, it might log errors but could potentially still
    # render a partial/broken report or fail during render.
    # We expect it might succeed but produce a report possibly missing sections
    # or showing 'N/A'. Let's check if it runs and produces *a* file.
    # A more robust test would mock db functions to ensure missing keys.

    report_path = generate_html_report(run_dir, summary=None) # Generate from DB

    # Check if *any* report file was created
    assert report_path is not None, "Report generation should run, even if data is missing"
    assert Path(report_path).exists()
    assert Path(report_path).name == "data_analysis_report.html"

    # Optionally, check the log for warnings about missing data (requires log capture)
    # Check report content for expected placeholders or missing sections (can be brittle)
    html = Path(report_path).read_text(encoding='utf-8')
    assert "<h1>MD Analysis Report</h1>" in html
    # Look for 'N/A' which might indicate missing metrics were handled gracefully
    assert "N/A" in html


EOF
echo "  Created tests/test_html_report_error.py (adapted)"

# === test_analyze_trajectory.py (Original - needs deletion/merge) ===
# This file's logic should be merged into test_core_analysis_computation.py
# We'll create an empty file as a placeholder if needed, or just omit it.
# OMITTING - Logic moved to test_core_analysis_computation.py

# === test_analyze_trajectory_missing.py (Original - needs deletion/merge) ===
# Logic moved to test_core_analysis_computation.py
# OMITTING - Logic moved

# === test_gg_filter.py (Original - needs deletion/merge) ===
# Logic moved to test_core_analysis_computation.py
# OMITTING - Logic moved

# === test_kde_analysis.py (Original - needs deletion/merge) ===
# Logic moved to test_core_analysis_visualization.py
# OMITTING - Logic moved

# === test_summary_fields.py (Original - needs rewrite/deletion) ===
# Original logic tested a function that no longer exists.
# We'll create a new test for the current summary generation.
cat << 'EOF' > tests/test_summary_generation.py
import pytest
import os
import numpy as np
import pandas as pd
import sqlite3
import json
from pathlib import Path

# Adapt imports for refactored code
from pore_analysis.core.database import init_db, connect_db, register_module, register_product, store_metric, set_simulation_metadata
from pore_analysis.summary import generate_summary_from_database

@pytest.fixture
def setup_summary_db(tmp_path):
    """Sets up a database with some data for summary generation testing."""
    run_dir = tmp_path / "summary_run"
    run_dir.mkdir()
    conn = init_db(str(run_dir))

    # --- Populate DB ---
    set_simulation_metadata(conn, "run_name", "summary_run")
    set_simulation_metadata(conn, "system_name", "summary_system")
    set_simulation_metadata(conn, "is_control_system", "False")

    # Modules
    register_module(conn, "core_analysis", status='success')
    register_module(conn, "core_analysis_filtering", status='success')
    register_module(conn, "core_analysis_visualization_g_g", status='success')
    register_module(conn, "ion_analysis", status='failed', error_message="Ion error")

    # Products
    register_product(conn, "core_analysis_filtering", "csv", "data", "core_analysis/G_G_Distance_Filtered.csv", "g_g_distance_filtered", "Desc G-G Filt")
    register_product(conn, "core_analysis_visualization_g_g", "png", "plot", "core_analysis/G_G_Distance_Subunit_Comparison.png", "subunit_comparison", "Desc G-G Plot")

    # Metrics
    store_metric(conn, "core_analysis_filtering", 'G_G_AC_Mean_Filt', 1.1, 'Å', '')
    store_metric(conn, "core_analysis_filtering", 'COM_Mean_Filt', 10.5, 'Å', '')
    store_metric(conn, "ion_analysis", 'Ion_AvgOcc_S0', 0.1, 'count', '') # Metric from failed module

    conn.commit()
    conn.close()
    return str(run_dir)

def test_generate_summary_from_database(setup_summary_db):
    """Test generating the summary dictionary directly from the database."""
    run_dir = setup_summary_db
    conn = connect_db(run_dir)
    assert conn is not None

    summary = generate_summary_from_database(run_dir, conn)
    conn.close() # Close connection after use

    # --- Assertions ---
    assert isinstance(summary, dict)
    assert summary['run_dir'] == run_dir
    assert summary['run_name'] == "summary_run"
    assert summary['metadata']['system_name'] == "summary_system"
    assert summary['is_control_system'] is False

    # Check module status
    assert 'module_status' in summary
    assert summary['module_status'].get('core_analysis_filtering') == 'success'
    assert summary['module_status'].get('ion_analysis') == 'failed'

    # Check metrics (ensure structure {metric: {value: V, units: U}})
    assert 'metrics' in summary
    assert 'G_G_AC_Mean_Filt' in summary['metrics']
    assert summary['metrics']['G_G_AC_Mean_Filt']['value'] == 1.1
    assert summary['metrics']['G_G_AC_Mean_Filt']['units'] == 'Å'
    assert 'COM_Mean_Filt' in summary['metrics']
    assert summary['metrics']['COM_Mean_Filt']['value'] == 10.5
    assert 'Ion_AvgOcc_S0' in summary['metrics'] # Check metric from failed module is still present
    assert summary['metrics']['Ion_AvgOcc_S0']['value'] == 0.1

    # Check key plots (based on plots_dict.json structure)
    assert 'key_plots' in summary
    # The exact key depends on plots_dict.json, assuming 'g_g_distances' is the key for the G-G plot
    assert 'g_g_distances' in summary['key_plots']
    assert summary['key_plots']['g_g_distances'] == "core_analysis/G_G_Distance_Subunit_Comparison.png"

    # Ensure analysis_summary.json was NOT created by this function
    summary_file = Path(run_dir) / "analysis_summary.json"
    assert not summary_file.exists()

EOF
echo "  Created tests/test_summary_generation.py (adapted)"

# === test_html_integration.py (Adaptation) ===
# This test uses subprocess, so it needs fewer code changes, mostly CLI args.
# Needs Pillow installed in the test environment to create dummy PNGs for registration.
cat << 'EOF' > tests/test_html_integration.py
import pytest # pytest.skip("Skipping HTML template tests in CI", allow_module_level=True) # Allow running locally
import subprocess
import sys
import os
from pathlib import Path
import shutil # For copying directory tree

# Helper function to create dummy files
def create_dummy_files(run_dir):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    # Dummy PSF
    (run_dir / "step5_input.psf").write_text("PSF\n\n 1 !NATOM\n 1 A 1 A N N 0 14 0\n 0 !NBOND")
    # Dummy DCD (basic header for 1 frame)
    import numpy as np
    dcd_header = np.array([84, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 67, 79, 82, 68], dtype=np.int32).tobytes()
    dcd_frame_header = np.array([12], dtype=np.int32).tobytes() # 1 atom * 3 coords * 4 bytes/float
    dcd_coords = np.arange(1*3, dtype=np.float32).reshape(1, 3)
    dcd_frame_footer = np.array([12], dtype=np.int32).tobytes()
    with open(run_dir / "MD_Aligned.dcd", "wb") as f:
        f.write(dcd_header)
        f.write(dcd_frame_header)
        f.write(dcd_coords.tobytes())
        f.write(dcd_frame_footer)


def test_main_py_integration(tmp_path):
    """Test running main.py end-to-end."""
    # Create dummy run directory with minimal files
    test_run_dir = tmp_path / "integration_run"
    create_dummy_files(test_run_dir)

    # Run the main script (defaults to all modules)
    # Ensure pore_analysis is in PYTHONPATH or installed
    cmd = [
        sys.executable, "-m", "pore_analysis.main",
        "--folder", str(test_run_dir),
        "--no-plots", # Speed up test by skipping plot generation
        "--report" # Still generate report based on computation
    ]
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=False) # check=False to see output on failure

    print("--- STDOUT ---")
    print(result.stdout)
    print("--- STDERR ---")
    print(result.stderr)

    # Basic success check (exit code 0)
    assert result.returncode == 0, f"main.py failed with stderr:\n{result.stderr}"

    # Check database file exists
    db_file = test_run_dir / "analysis_registry.db"
    assert db_file.exists(), "Database file was not created"

    # Check report file exists
    report_file = test_run_dir / "data_analysis_report.html"
    assert report_file.exists(), "HTML report file was not created"

    # Check basic content of the report
    content = report_file.read_text(encoding="utf-8")
    assert "<h1>MD Analysis Report</h1>" in content
    # Check if control system status is mentioned (likely True for dummy data)
    assert "CONTROL SYSTEM (NO TOXIN)" in content or "Toxin-Channel Complex" in content

    # Optionally, check DB status using sqlite3
    import sqlite3
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("SELECT value FROM simulation_metadata WHERE key = 'analysis_status'")
    status_row = cursor.fetchone()
    conn.close()
    assert status_row is not None and status_row[0] == 'success', "Analysis status in DB is not 'success'"


EOF
echo "  Created tests/test_html_integration.py (adapted)"


# === Make script executable ===
chmod +x create_updated_tests.sh

echo "Finished creating updated test files in ./tests/"
echo "Please review the adapted tests, especially those marked with comments."
echo "You may need to add fixtures (e.g., in conftest.py) for database setup/teardown and dummy data."
