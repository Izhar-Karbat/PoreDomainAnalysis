# tests/test_summary_generation.py

import pytest
import os
import numpy as np
import pandas as pd
import sqlite3
import json
from pathlib import Path

# Adapt imports for refactored code
from pore_analysis.core.database import (
    init_db, connect_db, register_module, register_product, store_metric, set_simulation_metadata
)
from pore_analysis.summary import generate_summary_from_database

# Fixture to set up a temporary database (remains unchanged)
@pytest.fixture
def setup_summary_db(tmp_path):
    """Sets up a database with some data for summary generation testing."""
    run_dir = tmp_path / "summary_run"
    run_dir.mkdir()
    conn = init_db(str(run_dir))

    # --- Populate DB ---
    set_simulation_metadata(conn, "run_name", "summary_run")
    set_simulation_metadata(conn, "system_name", "summary_system") # Added for testing retrieval
    set_simulation_metadata(conn, "is_control_system", "False")

    # Modules
    register_module(conn, "core_analysis", status='success')
    register_module(conn, "core_analysis_filtering", status='success')
    register_module(conn, "core_analysis_visualization_g_g", status='success')
    register_module(conn, "ion_analysis", status='failed') # Example failed module

    # Products
    register_product(
        conn, "core_analysis_filtering", "csv", "data",
        "core_analysis/G_G_Distance_Filtered.csv",
        "g_g_distance_filtered", "Desc G-G Filt"
    )
    # Register the plot using the subcategory that summary.py now finds via plots_dict.json
    register_product(
        conn, "core_analysis_visualization_g_g", "png", "plot",
        "core_analysis/G_G_Distance_Subunit_Comparison.png",
        "subunit_comparison", # Correct subcategory
        "Desc G-G Plot"
    )
    # Add another plot registration example for completeness, if desired
    # register_product(
    #     conn, "core_analysis_visualization_com", "png", "plot",
    #     "core_analysis/COM_Stability_Comparison.png",
    #     "comparison", # Correct subcategory for COM comparison plot
    #     "Desc COM Plot"
    # )


    # Metrics
    store_metric(conn, "core_analysis_filtering", 'G_G_AC_Mean_Filt', 1.1, 'Å', '')
    store_metric(conn, "core_analysis_filtering", 'COM_Mean_Filt', 10.5, 'Å', '')
    store_metric(conn, "ion_analysis", 'Ion_AvgOcc_S0', 0.1, 'count', '') # Metric from failed module

    conn.commit()
    conn.close()
    return str(run_dir)


# Test function with updated assertion
def test_generate_summary_from_database(setup_summary_db):
    """Test generating the summary dictionary directly from the database."""
    run_dir = setup_summary_db
    conn = connect_db(run_dir)
    assert conn is not None

    # Assume load_plot_queries is available and reads plots_dict.json correctly
    # during the execution of generate_summary_from_database
    summary = generate_summary_from_database(run_dir, conn)
    conn.close() # Close connection after use

    # --- Assertions ---
    assert isinstance(summary, dict)
    assert summary['run_dir'] == run_dir
    assert summary['run_name'] == "summary_run" # Check against metadata value

    # Check metadata retrieval
    assert 'metadata' in summary
    assert summary['metadata'].get('system_name') == "summary_system"

    # is_control_system flag should be correctly parsed
    # Use .get for safety, default to False if key missing (adjust default if needed)
    assert summary.get('is_control_system', None) is False # Checks explicit False from "False" string

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

    # Check key_plots structure
    assert 'key_plots' in summary
    assert isinstance(summary['key_plots'], dict)

    # Check expected G-G distance plot using the TEMPLATE_KEY from plots_dict.json
    assert 'subunit_comparison' in summary['key_plots'] # *** MODIFIED ASSERTION KEY ***
    assert summary['key_plots']['subunit_comparison'] == "core_analysis/G_G_Distance_Subunit_Comparison.png" # *** MODIFIED ASSERTION KEY ***

    # Check if another plot was found (e.g., if COM plot was added to fixture)
    # assert 'comparison' in summary['key_plots'] # Example if COM plot was added

    # Ensure analysis_summary.json was NOT created by this function
    summary_file = Path(run_dir) / "analysis_summary.json"
    assert not summary_file.exists()
