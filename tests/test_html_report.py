import numpy as np
from pathlib import Path
from pore_analysis.reporting.html import generate_html_report

def test_generate_html_report(tmp_path):
    # Prepare run directory
    run_dir = tmp_path / "run_html"
    run_dir.mkdir()

    # Minimal summary dict with all required overview keys
    run_summary = {
        "SystemName": "sys",
        "RunName": "run_html",
        "RunPath": str(run_dir),
        "AnalysisStatus": "Success",
        "AnalysisTimestamp": "2025-04-21T00:00:00",
        "IsControlSystem": True,
        # G-G filtered stats (overview)
        "G_G_AC_Mean_Filt": 1.0,
        "G_G_BD_Mean_Filt": 2.0,
        "G_G_AC_Std_Filt": 0.1,
        "G_G_BD_Std_Filt": 0.2,
        "G_G_AC_Min_Filt": 0.5,
        "G_G_BD_Min_Filt": 0.6,
    }

    # Generate HTML report
    report_path = generate_html_report(str(run_dir), run_summary)
    assert report_path is not None, "Expected a report path"

    html = Path(report_path).read_text(encoding='utf-8')
    # Sanity checks on content
    assert "<h1>MD Analysis Report</h1>" in html
    assert "sys" in html
    assert "run_html" in html
    assert "Success" in html
