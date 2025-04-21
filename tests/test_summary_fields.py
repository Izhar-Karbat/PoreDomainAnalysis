import json
import numpy as np
from pore_analysis.reporting.summary import calculate_and_save_run_summary

def test_summary_with_raw_stats(tmp_path):
    # Prepare a run directory
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    # Provide only raw distance stats and percentiles (no CSVs)
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
    # It should at least succeed
    assert status.startswith("Success")

    # Inspect generated JSON
    summary = json.loads((run_dir / "analysis_summary.json").read_text())
    assert summary["SystemName"] == "sys"
    assert summary["RunName"] == "run"
    # Check the raw stats and percentiles made it through
    assert np.isclose(summary["G_G_AC_Mean_Raw"], 1.23)
    assert np.isclose(summary["G_G_BD_Mean_Raw"], 4.56)
    assert np.isclose(summary["G_G_AC_Pctl10_Raw"], 0.1)
    assert np.isclose(summary["G_G_BD_Pctl90_Raw"], 0.8)
