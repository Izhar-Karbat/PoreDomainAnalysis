import numpy as np
from pathlib import Path
from pore_analysis.reporting.html import generate_html_report

def test_full_html_report(tmp_path):
    # 1) Prepare run directory
    run_dir = tmp_path / "run_html"
    run_dir.mkdir()

    # 2) Build a run_summary dict with all required overview + ion stats
    run_summary = {
        "SystemName": "sys",
        "RunName": "run_html",
        "RunPath": str(run_dir),
        "AnalysisStatus": "Success",
        "AnalysisTimestamp": "2025-04-21T00:00:00",
        "IsControlSystem": False,
        # G-G stats for overview tab
        "G_G_AC_Mean_Filt": 1.0,
        "G_G_BD_Mean_Filt": 2.0,
        "G_G_AC_Std_Filt": 0.1,
        "G_G_BD_Std_Filt": 0.2,
        "G_G_AC_Min_Filt": 0.5,
        "G_G_BD_Min_Filt": 0.6,
    }
    # 3) Add all six ion‐occupancy fields that the pore‐ions tab loops over
    for site in ["S0","S1","S2","S3","S4","Cavity"]:
        run_summary[f"Ion_AvgOcc_{site}"] = 0.1

    # 4) Generate the report
    report_path = generate_html_report(str(run_dir), run_summary)
    assert report_path, "Report path should not be None"

    html = Path(report_path).read_text(encoding="utf-8")
    # 5) Basic sanity checks
    assert "<h1>MD Analysis Report</h1>" in html
    assert "Pore Ions" in html
    # Check that each site label appears in the table
    for site in ["S0","S1","S2","S3","S4","Cavity"]:
        assert f"Ion_AvgOcc_{site}" not in html  # field keys aren’t literally shown
        assert site in html                      # but the header “S0”, “S1”, etc. must be
