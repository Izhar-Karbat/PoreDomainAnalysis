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
    module_id_filt = conn.execute("SELECT module_id FROM analysis_modules WHERE module_name = ?", ("core_analysis_filtering",)).fetchone()['module_id']

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
        module_id_viz_gg = conn.execute("SELECT module_id FROM analysis_modules WHERE module_name = ?", ("core_analysis_visualization_g_g",)).fetchone()['module_id']
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

