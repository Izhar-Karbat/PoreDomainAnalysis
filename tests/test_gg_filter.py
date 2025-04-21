import numpy as np
import pandas as pd

from pore_analysis.modules.core_analysis.core import filter_and_save_data

def test_gg_filter_and_save(tmp_path):
    # Prepare a fake run directory
    run_dir = tmp_path / "run1"
    run_dir.mkdir()

    # Synthetic raw distances: a simple trend with noise
    dist_ac = np.array([1.0, 2.0, 3.5, 2.5, 1.5])
    dist_bd = np.array([1.2, 2.1, 3.6, 2.4, 1.3])
    com_distances = None           # not testing COM here
    time_points = np.array([0, 1, 2, 3, 4])
    box_z = None
    is_control_system = True       # so COM branch is skipped

    # Run the filter & save
    (filtered_ac, filtered_bd,
     filtered_com,
     filter_info_g_g, _,
     raw_stats, percent_stats) = filter_and_save_data(
        str(run_dir), dist_ac, dist_bd, com_distances, time_points, box_z, is_control_system
    )

    # 1) The filtered arrays should be numpy arrays of the same length
    assert isinstance(filtered_ac, np.ndarray)
    assert isinstance(filtered_bd, np.ndarray)
    assert filtered_ac.shape == dist_ac.shape
    assert filtered_bd.shape == dist_bd.shape

    # 2) Filter info dict for G-G should contain at least one key
    assert isinstance(filter_info_g_g, dict)
    assert filter_info_g_g, "filter_info_g_g should not be empty"

    # 3) Raw statistics should include G_G_AC_Mean and G_G_BD_Mean
    assert "G_G_AC_Mean" in raw_stats
    assert np.isclose(raw_stats["G_G_AC_Mean"], dist_ac.mean())
    assert "G_G_BD_Mean" in raw_stats
    assert np.isclose(raw_stats["G_G_BD_Mean"], dist_bd.mean())

    # 4) The CSV for filtered G-G distances must exist and have the expected columns
    out_csv = run_dir / "core_analysis" / "G_G_Distance_Filtered.csv"
    assert out_csv.exists(), f"Expected {out_csv} to be created"

    df = pd.read_csv(out_csv)
    expected_cols = {
        "Time (ns)",
        "G_G_Distance_AC_Raw",
        "G_G_Distance_BD_Raw",
        "G_G_Distance_AC_Filt",
        "G_G_Distance_BD_Filt"
    }
    assert expected_cols.issubset(set(df.columns))

    # 5) The filtered values in the CSV match those returned (compare to 4 decimals)
    np.testing.assert_array_almost_equal(
        df["G_G_Distance_AC_Filt"].values,
        filtered_ac,
        decimal=4
    )
    np.testing.assert_array_almost_equal(
        df["G_G_Distance_BD_Filt"].values,
        filtered_bd,
        decimal=4
    )
