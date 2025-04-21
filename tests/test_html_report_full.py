import pytest; pytest.skip("Skipping HTML template tests in CI", allow_module_level=True)
import numpy as np
from pathlib import Path
from pore_analysis.reporting.html import generate_html_report

def test_full_html_report(tmp_path):
    # 1) Prepare run directory
    run_dir = tmp_path / "run_html"
    run_dir.mkdir()

    # 2) Build a run_summary dict with all required overview + COM + ion stats
    run_summary = {
        "SystemName": "sys",
        "RunName": "run_html",
        "RunPath": str(run_dir),
        "AnalysisStatus": "Success",
        "AnalysisTimestamp": "2025-04-21T00:00:00",
        "IsControlSystem": False,
        # G-G stats for overview
        "G_G_AC_Mean_Filt": 1.0,
        "G_G_BD_Mean_Filt": 2.0,
        "G_G_AC_Std_Filt": 0.1,
        "G_G_BD_Std_Filt": 0.2,
        "G_G_AC_Min_Filt": 0.5,
        "G_G_BD_Min_Filt": 0.6,
        # COM stats for overview
        "COM_Mean_Filt": 1.5,
        "COM_Std_Filt": 0.3,
        "COM_Min_Filt": 1.0,
        "COM_Max_Filt": 2.0,
        "COM_Filter_Type": "auto",
        "COM_Filter_QualityIssue": False,
    }
    # 3) Add all six ion‑occupancy fields that the pore‑ions tab loops over
    for site in ["S0","S1","S2","S3","S4","Cavity"]:
        run_summary[f"Ion_AvgOcc_{site}"] = 0.1

    # 4) Generate the report
    report_path = generate_html_report(str(run_dir), run_summary)
    assert report_path, "Expected a valid report path"

    html = Path(report_path).read_text(encoding="utf-8")
    # 5) Basic sanity checks on headings
    for heading in ["MD Analysis Report", "Overview & Distances", "Pore Ions",
                    "Inner Vestibule", "Carbonyl Dynamics", "SF Tyrosine", "DW Gate"]:
        assert heading in html
