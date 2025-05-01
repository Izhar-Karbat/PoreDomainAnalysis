import pytest; # pytest.skip("Skipping HTML template tests in CI", allow_module_level=True)
import numpy as np
import pandas as pd
from pathlib import Path
import base64
import sqlite3
import os
import json

# Adapt imports for refactored code
from pore_analysis.core.database import (
    init_db, connect_db, register_module, register_product,
    store_metric, set_simulation_metadata
)
from pore_analysis.html import generate_html_report
from pore_analysis.summary import generate_summary_from_database  # Use this to generate summary dict

# Helper function to create a dummy plot file
def create_dummy_png(filepath, size=(100, 50)):
    try:
        from PIL import Image
        img = Image.new('RGB', size, color='green')
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
    set_simulation_metadata(conn, "is_control_system", "False")  # Non-control

    # Register modules as successful
    for module in [
        "core_analysis_filtering", "orientation_analysis", "ion_analysis",
        "inner_vestibule_analysis", "gyration_analysis", "tyrosine_analysis",
        "dw_gate_analysis",
        # Visualization modules
        "core_analysis_visualization_g_g", "core_analysis_visualization_com",
        "orientation_analysis_visualization", "ion_analysis_visualization",
        "inner_vestibule_analysis_visualization", "gyration_analysis_visualization",
        "tyrosine_analysis_visualization", "dw_gate_analysis_visualization"
    ]:
        register_module(conn, module, status='success')

    # --- Store Metrics (representative subset) ---
    # Core metrics
    core_metrics = [
        ('G_G_AC_Mean_Filt', 1.1), ('G_G_BD_Mean_Filt', 2.1),
        ('COM_Mean_Filt', 10.5), ('COM_Std_Filt', 0.5)
    ]
    for key, val in core_metrics:
        store_metric(conn, "core_analysis_filtering", key, val, 'Å', '')

    # Orientation metrics
    store_metric(conn, "orientation_analysis", 'Orient_Angle_Mean', 45.0, '°', '')
    store_metric(conn, "orientation_analysis", 'Orient_Contacts_Mean', 55.0, 'count', '')

    # Ion metrics
    for site in ["S0","S1","S2","S3","S4","Cavity"]:
        idx = int(site[1]) if len(site)>1 and site[1].isdigit() else 5
        store_metric(conn, "ion_analysis", f"Ion_AvgOcc_{site}", 0.1 + idx*0.05, 'count', '')
        store_metric(conn, "ion_analysis", f"Ion_MaxOcc_{site}", 1, 'count', '')
        store_metric(conn, "ion_analysis", f"Ion_PctTimeOcc_{site}", 10.0 + idx*5, '%', '')
    store_metric(conn, "ion_analysis", 'Ion_HMM_ConductionEvents_Total', 5, 'count', '')
    store_metric(conn, "ion_analysis", 'Ion_HMM_Transition_S1_S0', 2, 'count', '')

    # Inner Vestibule metrics
    store_metric(conn, "inner_vestibule_analysis", 'InnerVestibule_MeanOcc', 3.5, 'count', '')
    store_metric(conn, "inner_vestibule_analysis", 'InnerVestibule_AvgResidenceTime_ns', 0.15, 'ns', '')

    # Gyration metrics
    store_metric(conn, "gyration_analysis", 'Gyration_G1_Mean', 3.0, 'Å', '')
    store_metric(conn, "gyration_analysis", 'Gyration_Y_Mean', 3.2, 'Å', '')
    store_metric(conn, "gyration_analysis", 'Gyration_G1_OnFlips', 2, 'count', '')
    store_metric(conn, "gyration_analysis", 'Gyration_Y_OnFlips', 1, 'count', '')

    # Tyrosine metrics
    store_metric(conn, "tyrosine_analysis", 'Tyr_HMM_TotalTransitions', 15, 'count', '')
    store_metric(conn, "tyrosine_analysis", 'Tyr_HMM_Population_pp', 85.0, '%', '')
    store_metric(conn, "tyrosine_analysis", 'Tyr_HMM_MeanDwell_pp', 10.2, 'ns', '')
    set_simulation_metadata(conn, 'Tyr_HMM_DominantState', 'pp')

    # DW Gate metrics
    store_metric(conn, "dw_gate_analysis", 'DW_PROA_Closed_Fraction', 75.0, '%', '')
    store_metric(conn, "dw_gate_analysis", 'DW_PROA_open_Mean_ns', 2.5, 'ns', '')

    # Commit and close
    conn.commit()
    conn.close()
    return str(run_dir)


def test_generate_html_report_full(setup_full_report_db):
    """Test HTML report generation with a more complete dataset."""
    run_dir = setup_full_report_db

    report_path = generate_html_report(run_dir, summary=None)

    assert report_path is not None
    assert Path(report_path).exists()

    html = Path(report_path).read_text(encoding='utf-8')

    # Check for presence of various section titles
    assert "Overview & Distances" in html
    assert "Toxin Interface" in html
    assert "Pore Ion Analysis" in html
    assert "Inner Vestibule Water Analysis" in html
    assert "Carbonyl Dynamics" in html
    assert "SF Tyrosine Analysis" in html
    assert "DW Gate Dynamics" in html

    # Check for specific metrics rendered
    assert "10.5" in html  # COM Mean Filt
    assert "45.0" in html  # Orient Angle Mean
    # Numeric occupancy check instead of raw key presence
    assert ">0.1<" in html  # Ion_AvgOcc_S0
    assert "3.5" in html  # InnerVestibule_MeanOcc
    assert "2" in html  # Gyration G1 On Flips count
    assert ">pp<" in html  # Dominant Tyrosine state
    assert "75.0%" in html  # DW Gate Closed Fraction

    # Check for embedded plots (optional)
    if "data:image/png;base64," in html:
        print("Base64 image data found in report.")
    else:
        print("Warning: No base64 image data found in report.")
