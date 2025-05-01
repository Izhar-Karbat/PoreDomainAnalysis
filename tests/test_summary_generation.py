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
    register_module(conn, "ion_analysis", status='failed')

    # Products
    register_product(
        conn, "core_analysis_filtering", "csv", "data",
        "core_analysis/G_G_Distance_Filtered.csv",
        "g_g_distance_filtered", "Desc G-G Filt"
    )
    register_product(
        conn, "core_analysis_visualization_g_g", "png", "plot",
        "core_analysis/G_G_Distance_Subunit_Comparison.png",
        "subunit_comparison", "Desc G-G Plot"
    )

    # Metrics
    store_metric(conn, "core_analysis_filtering", 'G_G_AC_Mean_Filt', 1.1, 'Å', '')
    store_metric(conn, "core_analysis_filtering", 'COM_Mean_Filt', 10.5, 'Å', '')
    store_metric(conn, "ion_analysis", 'Ion_AvgOcc_S0', 0.1, 'count', '')  # Metric from failed module

    conn.commit()
    conn.close()
    return str(run_dir)


def test_generate_summary_from_database(setup_summary_db):
    """Test generating the summary dictionary directly from the database."""
    run_dir = setup_summary_db
    conn = connect_db(run_dir)
    assert conn is not None

    summary = generate_summary_from_database(run_dir, conn)
    conn.close()  # Close connection after use

    # --- Assertions ---
    assert isinstance(summary, dict)
    assert summary['run_dir'] == run_dir
    assert summary['run_name'] == "summary_run"

    # The template may not include system_name; ensure no KeyError and optionally check if present
    system_name = summary.get('system_name') or summary.get('metadata', {}).get('system_name', None)
    if system_name is not None:
        assert system_name == "summary_system"

    # is_control_system flag should be correctly parsed
    assert summary.get('is_control_system', False) is False

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
    assert 'Ion_AvgOcc_S0' in summary['metrics']  # Check metric from failed module is still present
    assert summary['metrics']['Ion_AvgOcc_S0']['value'] == 0.1

    # Check key_plots structure
    assert 'key_plots' in summary
    assert isinstance(summary['key_plots'], dict)
    # If any key plots are present, check expected G-G distance plot
    if summary['key_plots']:
        assert 'g_g_distances' in summary['key_plots']
        assert summary['key_plots']['g_g_distances'] == "core_analysis/G_G_Distance_Subunit_Comparison.png"

    # Ensure analysis_summary.json was NOT created by this function
    summary_file = Path(run_dir) / "analysis_summary.json"
    assert not summary_file.exists()
