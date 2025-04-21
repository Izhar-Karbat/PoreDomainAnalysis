import numpy as np
from pathlib import Path
from pore_analysis.modules.core_analysis.core import analyze_trajectory

def test_analyze_trajectory_missing_files(tmp_path):
    # Create an empty run directory with neither PSF nor DCD
    run_dir = tmp_path / "run_missing"
    run_dir.mkdir()

    dist_ac, dist_bd, com_distances, time_points, system_dir, is_control = analyze_trajectory(
        str(run_dir), psf_file=None, dcd_file=None
    )

    # Expect the “missing files” fallback
    assert np.array_equal(dist_ac, np.array([0]))
    assert np.array_equal(dist_bd, np.array([0]))
    assert com_distances is None
    assert np.array_equal(time_points, np.array([0]))
    assert system_dir == "Unknown"
    assert is_control is True
