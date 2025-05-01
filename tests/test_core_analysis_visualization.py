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
        'COM_Distance_Raw': np.concatenate([np.random.normal(10, 0.5, 5), np.random.normal(15, 0.5, 5)]), # Add some variation for KDE peaks
        'COM_Distance_Filt': np.concatenate([np.random.normal(10, 0.5, 5), np.random.normal(15, 0.5, 5)]) + 0.1,
    }
    com_csv_path = core_dir / "COM_Stability_Filtered.csv"
    pd.DataFrame(com_data).to_csv(com_csv_path, index=False)

    # --- Initialize and Populate DB ---
    conn = init_db(str(run_dir))
    assert conn is not None

    # Register dummy filtering module as success (needed for viz check)
    register_module(conn, "core_analysis_filtering", status='success')
    cur = conn.cursor()
    cur.execute("SELECT module_id FROM analysis_modules WHERE module_name = ?", ("core_analysis_filtering",))
    res = cur.fetchone()
    assert res is not None
    module_id = res['module_id']
    cur.close()

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
    run_dir_str = setup_core_viz_data
    run_dir_path = Path(run_dir_str)
    conn = connect_db(run_dir_str)
    assert conn is not None

    # Module registration now happens inside plot_distances
    # register_module(conn, "core_analysis_visualization_g_g", status='running')
    # conn.commit()

    plot_paths_dict = plot_distances(run_dir_str, is_gg=True, db_conn=conn)

    assert isinstance(plot_paths_dict, dict)
    assert 'subunit_comparison' in plot_paths_dict
    plot_path = run_dir_path / plot_paths_dict['subunit_comparison'] # <-- Path joining corrected
    assert plot_path.exists()
    assert plot_path.stat().st_size > 100

    assert get_module_status(conn, "core_analysis_visualization_g_g") == "success" # <-- Status should now be updated
    prod_path = get_product_path(conn, "png", "plot", "subunit_comparison", module_name="core_analysis_visualization_g_g")
    assert prod_path == "core_analysis/G_G_Distance_Subunit_Comparison.png"

    conn.close()

# Test COM plot generation
def test_plot_com_distances(setup_core_viz_data):
    run_dir_str = setup_core_viz_data
    run_dir_path = Path(run_dir_str)
    conn = connect_db(run_dir_str)
    assert conn is not None

    # Module registration now happens inside plot_distances
    # register_module(conn, "core_analysis_visualization_com", status='running')
    # conn.commit()

    plot_paths_dict = plot_distances(run_dir_str, is_gg=False, db_conn=conn)

    assert isinstance(plot_paths_dict, dict)
    assert 'comparison' in plot_paths_dict
    plot_path = run_dir_path / plot_paths_dict['comparison'] # <-- Path joining corrected
    assert plot_path.exists()
    assert plot_path.stat().st_size > 100

    assert get_module_status(conn, "core_analysis_visualization_com") == "success" # <-- Status should now be updated
    prod_path = get_product_path(conn, "png", "plot", "comparison", module_name="core_analysis_visualization_com")
    assert prod_path == "core_analysis/COM_Stability_Comparison.png"

    conn.close()

# Test KDE plot generation
def test_plot_kde_analysis_viz(setup_core_viz_data):
    run_dir_str = setup_core_viz_data
    run_dir_path = Path(run_dir_str) # Create Path object
    conn = connect_db(run_dir_str)
    assert conn is not None

    # Module registration now happens inside plot_kde_analysis

    plot_path_str = plot_kde_analysis(run_dir_str, db_conn=conn)

    assert plot_path_str is not None
    # --- FIX: Construct full path before checking existence ---
    plot_path = run_dir_path / plot_path_str
    # --- END FIX ---
    assert plot_path.exists() # <-- Should now check the correct absolute path
    assert plot_path.stat().st_size > 100
    assert plot_path.name == "COM_Stability_KDE_Analysis.png"

    assert get_module_status(conn, "core_analysis_visualization_com") == "success" # <-- Status should now be updated
    prod_path = get_product_path(conn, "png", "plot", "kde_analysis", module_name="core_analysis_visualization_com")
    assert prod_path == "core_analysis/COM_Stability_KDE_Analysis.png"

    conn.close()

def test_plot_distances_no_db_conn(setup_core_viz_data):
    """Test that plots fail gracefully if DB connection is None (or fails)."""
    run_dir = setup_core_viz_data
    plot_paths_dict_gg = plot_distances(run_dir, is_gg=True, db_conn=None)
    plot_paths_dict_com = plot_distances(run_dir, is_gg=False, db_conn=None)
    assert plot_paths_dict_gg == {}
    assert plot_paths_dict_com == {}

def test_plot_kde_no_db_conn(setup_core_viz_data):
    """Test that KDE plot fails gracefully if DB connection is None."""
    run_dir = setup_core_viz_data
    plot_path = plot_kde_analysis(run_dir, db_conn=None)
    assert plot_path is None

def test_plot_distances_no_filtering_data(tmp_path):
    """Test plots fail if filtering step didn't run or register products."""
    run_dir = tmp_path / "viz_run_nofilter"
    run_dir.mkdir()
    conn = init_db(str(run_dir))
    if conn is None: pytest.fail("Failed to init DB for no_filtering_data test")
    conn.commit()
    conn.close()

    conn = connect_db(str(run_dir))
    if conn is None: pytest.fail("Failed to connect to DB for no_filtering_data test")
    # Module registration now happens inside plot_distances
    plot_paths_dict_gg = plot_distances(str(run_dir), is_gg=True, db_conn=conn)
    # Expect failure because get_module_status("core_analysis_filtering") will not return 'success'
    assert plot_paths_dict_gg == {}
    # Check that the viz module status was updated to 'skipped' or 'failed'
    status = get_module_status(conn, "core_analysis_visualization_g_g")
    assert status in ['skipped', 'failed']
    conn.close()
