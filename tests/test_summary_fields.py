import json
import numpy as np
from pathlib import Path
from pore_analysis.reporting.summary import calculate_and_save_run_summary

def test_summary_with_raw_stats(tmp_path):
    # Prepare a run directory
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    core_dir = run_dir / "core_analysis"
    core_dir.mkdir()

    # Create a dummy filtered G-G CSV so summary will write JSON
    filtered_csv = core_dir / "G_G_Distance_Filtered.csv"
    filtered_csv.write_text("G_G_Distance_AC_Filt,G_G_Distance_BD_Filt\n1.0,2.0\n")

    # Provide only raw distance stats and percentiles
    raw_dist_stats = {
        'G_G_AC_Mean_Raw': 1.23,
        'G_G_BD_Mean_Raw': 4.56
    }
    percentile_stats = {
        'G_G_AC_Pctl10_Raw': 0.1,
        'G_G_AC_Pctl90_Raw': 0.9,
        'G_G_BD_Pctl10_Raw': 0.2,
        'G_G_BD_Pctl90_Raw': 0.8
    }

    # Run summary
    status = calculate_and_save_run_summary(
        str(run_dir),
        system_name="sys",
        run_name="run",
        com_analyzed=False,
        filter_info_com={},
        ion_indices=None,
        cavity_water_stats={},
        raw_dist_stats=raw_dist_stats,
        percentile_stats=percentile_stats,
        orientation_rotation_stats={},
        ion_transit_stats={},
        gyration_stats={},
        tyrosine_stats={},
        conduction_stats={},
        dw_gate_stats={},
        is_control_system=True
    )
    # It should succeed now that the critical CSV exists
    assert status is True

    # Inspect generated JSON
    summary_file = run_dir / "analysis_summary.json"
    assert summary_file.exists(), "Summary JSON should be created"
    summary = json.loads(summary_file.read_text())
    assert summary["SystemName"] == "sys"
    assert summary["RunName"] == "run"
    # Check the raw stats and percentiles are present
    assert np.isclose(summary["G_G_AC_Mean_Raw"], 1.23)
    assert np.isclose(summary["G_G_BD_Mean_Raw"], 4.56)
    assert np.isclose(summary["G_G_AC_Pctl10_Raw"], 0.1)
    assert np.isclose(summary["G_G_BD_Pctl90_Raw"], 0.8)
