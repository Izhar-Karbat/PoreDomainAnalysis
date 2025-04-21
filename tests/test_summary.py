import json
import pandas as pd
import numpy as np
from pore_analysis.reporting.summary import calculate_and_save_run_summary

def test_calculate_summary_minimal(tmp_path):
    # Set up run directory and core_analysis subfolder
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    core_dir = run_dir / "core_analysis"
    core_dir.mkdir()

    # Create minimal G-G filtered CSV
    df = pd.DataFrame({
        'G_G_Distance_AC_Filt': [1.0, 2.0],
        'G_G_Distance_BD_Filt': [1.5, 2.5]
    })
    df.to_csv(core_dir / "G_G_Distance_Filtered.csv", index=False)

    # Call summary with default/empty dicts
    status = calculate_and_save_run_summary(
        str(run_dir),
        system_name="sys",
        run_name="run",
        com_analyzed=False,
        filter_info_com={},
        ion_indices=None,
        cavity_water_stats={},
        raw_dist_stats={},
        percentile_stats={},
        orientation_rotation_stats={},
        ion_transit_stats={},
        gyration_stats={},
        tyrosine_stats={},
        conduction_stats={},
        dw_gate_stats={},
        is_control_system=True
    )
    assert status is True

    # Load and inspect summary JSON
    summary = json.loads((run_dir / "analysis_summary.json").read_text())
    assert summary["SystemName"] == "sys"
    assert summary["RunName"] == "run"
    # Mean of [1.0,2.0] is 1.5
    assert np.isclose(summary["G_G_AC_Mean_Filt"], 1.5)
    assert summary["AnalysisStatus"] == "Success"
