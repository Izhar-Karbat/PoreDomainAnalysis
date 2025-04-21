import os
from pathlib import Path
from pore_analysis.reporting.html import generate_html_report

def test_generate_html_report(tmp_path):
    # Prepare run directory
    run_dir = tmp_path / "run_html"
    run_dir.mkdir()

    # Minimal summary dict
    run_summary = {
        "RunName": "run_html",
        "AnalysisScriptVersion": "v1.0",
        "IsControlSystem": False
    }

    # Generate HTML report
    report_path = generate_html_report(str(run_dir), run_summary)
    assert report_path is not None

    html_file = Path(report_path)
    assert html_file.exists(), "HTML report file should exist"

    content = html_file.read_text(encoding='utf-8')
    # Basic sanity checks
    assert "<h1>MD Analysis Report</h1>" in content
    assert "run_html_reference.html" not in content  # no erroneous reference
