import pytest  # pytest.skip("Skipping HTML template tests in CI", allow_module_level=True)
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

    report_path = generate_html_report(run_dir, summary=None)

    # Report should still be generated even with missing data
    assert report_path is not None, "Report generation should run, even if data is missing"
    assert Path(report_path).exists()
    assert Path(report_path).name == "data_analysis_report.html"

    html = Path(report_path).read_text(encoding='utf-8')

    # Check that the report contains the main title
    assert "MD Analysis Report" in html
    # Since data is missing, we expect placeholders like 'N/A'
    assert "N/A" in html
