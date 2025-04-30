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


