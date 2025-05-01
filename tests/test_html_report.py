import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import base64
import sqlite3
import os
import logging  # Import logging

# Adapt imports for refactored code
from pore_analysis.core.database import (
    init_db,
    connect_db,
    register_module,
    register_product,
    store_metric,
    set_simulation_metadata,
)
from pore_analysis.html import generate_html_report
from pore_analysis.summary import generate_summary_from_database  # Use this to generate summary dict

# Get logger for potential use
logger = logging.getLogger(__name__)

# Helper function to create a dummy plot file
def create_dummy_png(filepath, size=(100, 50)):
    # Need Pillow: pip install Pillow
    try:
        from PIL import Image
        img = Image.new('RGB', size, color='red')
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        img.save(filepath)
        run_dir = os.path.dirname(os.path.dirname(filepath))
        return os.path.relpath(filepath, run_dir)
    except ImportError:
        logger.warning("Pillow not installed, cannot create dummy PNG.")
        return None
    except Exception as e:
        logger.error(f"Error creating dummy PNG: {e}")
        return None


@pytest.fixture
def setup_report_db(tmp_path):
    """Sets up a temporary run directory with a populated database for report generation."""
    run_dir = tmp_path / "report_run"
    run_dir.mkdir()
    conn = None
    try:
        conn = init_db(str(run_dir))
        if conn is None:
            pytest.fail("Database initialization failed in fixture.")

        conn.row_factory = sqlite3.Row

        set_simulation_metadata(conn, "run_name", "report_run")
        set_simulation_metadata(conn, "system_name", "test_system")
        set_simulation_metadata(conn, "analysis_status", "success")
        set_simulation_metadata(conn, "is_control_system", "True")

        register_module(conn, "core_analysis_filtering", status='success')
        cur = conn.cursor()
        cur.execute(
            "SELECT module_id FROM analysis_modules WHERE module_name = ?",
            ("core_analysis_filtering",),
        )
        res = cur.fetchone()
        cur.close()
        if res is None:
            pytest.fail("Could not retrieve module_id for core_analysis_filtering")
        module_id_filt = res['module_id']

        # Store various metrics
        store_metric(
            conn,
            "core_analysis_filtering",
            'G_G_AC_Mean_Filt',
            1.0,
            'Å',
            'Desc AC Mean',
        )
        store_metric(
            conn,
            "core_analysis_filtering",
            'G_G_BD_Mean_Filt',
            2.0,
            'Å',
            'Desc BD Mean',
        )
        store_metric(
            conn,
            "core_analysis_filtering",
            'G_G_AC_Std_Filt',
            0.1,
            'Å',
            'Desc AC Std',
        )
        store_metric(
            conn,
            "core_analysis_filtering",
            'G_G_BD_Std_Filt',
            0.2,
            'Å',
            'Desc BD Std',
        )
        store_metric(
            conn,
            "core_analysis_filtering",
            'G_G_AC_Min_Filt',
            0.5,
            'Å',
            'Desc AC Min',
        )
        store_metric(
            conn,
            "core_analysis_filtering",
            'G_G_BD_Min_Filt',
            0.6,
            'Å',
            'Desc BD Min',
        )
        store_metric(
            conn,
            "core_analysis_filtering",
            'G_G_AC_Max_Filt',
            1.5,
            'Å',
            'Desc AC Max',
        )
        store_metric(
            conn,
            "core_analysis_filtering",
            'G_G_BD_Max_Filt',
            2.6,
            'Å',
            'Desc BD Max',
        )

        # Create a dummy plot and register it if possible
        core_viz_dir = run_dir / "core_analysis"
        core_viz_dir.mkdir()
        dummy_gg_plot_path = core_viz_dir / "G_G_Distance_Subunit_Comparison.png"

        rel_plot_path = create_dummy_png(dummy_gg_plot_path)
        if rel_plot_path:
            register_module(conn, "core_analysis_visualization_g_g", status='success')
            cur = conn.cursor()
            cur.execute(
                "SELECT module_id FROM analysis_modules WHERE module_name = ?",
                ("core_analysis_visualization_g_g",),
            )
            res_viz = cur.fetchone()
            cur.close()
            if res_viz is None:
                pytest.fail("Could not retrieve module_id for core_analysis_visualization_g_g")
            module_id_viz_gg = res_viz['module_id']
            register_product(
                conn,
                "core_analysis_visualization_g_g",
                "png",
                "plot",
                rel_plot_path,
                subcategory="subunit_comparison",
                description="Dummy G-G Plot",
            )

        conn.commit()
        yield str(run_dir)

    finally:
        if conn:
            conn.close()


def test_generate_html_report_basic(setup_report_db):
    """Test basic HTML report generation for a control system."""
    run_dir = setup_report_db

    report_path = generate_html_report(run_dir, summary=None)

    assert report_path is not None, "Expected a report path"
    assert os.path.exists(report_path)
    assert Path(report_path).name == "data_analysis_report.html"

    html = Path(report_path).read_text(encoding='utf-8')

    # --- Basic Checks ---
    assert "MD Analysis Report" in html
    assert "report_run" in html
    assert "test_system" in html
    assert "CONTROL SYSTEM (NO TOXIN)" in html

    # --- MODIFIED ASSERTION: Check for the UNESCAPED string as it appears in the template ---
    assert "Overview & Distances" in html  # Check tab name (UNESCAPED)
    # --- END MODIFICATION ---

    # Modify assertions for dropped trailing zeros
    assert "1.0" in html
    assert "2.0" in html
    assert "0.1" in html
    assert "0.6" in html

    assert "COM distance analysis is not applicable." in html

    if (Path(run_dir) / "core_analysis" / "G_G_Distance_Subunit_Comparison.png").exists():
        assert "data:image/png;base64," in html
        assert 'alt="G-G Distance Subunit Comparison (Raw vs Filtered)"' in html
    else:
        logger.warning("Skipping plot embed check as dummy PNG likely wasn't created (Pillow import failed).")
