import numpy as np
import pandas as pd
from pathlib import Path

from pore_analysis.modules.core_analysis.core import filter_and_save_data

def test_com_filter_and_save(tmp_path):
    run_dir = tmp_path / "run_com"
    run_dir.mkdir()
    
    # Dummy G-G inputs (not checked here)
    dist_ac = np.array([0.0, 0.0, 0.0])
    dist_bd = np.array([0.0, 0.0, 0.0])
    
    # Synthetic COM distances
    com_distances = np.array([1.0, 2.0, 3.0, 2.5])
    time_points = np.array([0, 1, 2, 3])
    box_z = None
    is_control_system = False

    (filtered_ac, filtered_bd, filtered_com,
     _, filter_info_com,
     raw_stats, _) = filter_and_save_data(
        str(run_dir), dist_ac, dist_bd, com_distances, time_points, box_z, is_control_system
    )

    # Assertions
    assert isinstance(filtered_com, np.ndarray)
    assert filtered_com.shape == com_distances.shape
    assert isinstance(filter_info_com, dict) and filter_info_com
    assert "COM_Mean" in raw_stats and np.isclose(raw_stats["COM_Mean"], com_distances.mean())
    assert "COM_Std" in raw_stats and np.isclose(raw_stats["COM_Std"], com_distances.std())

    com_csv = Path(run_dir) / "core_analysis" / "COM_Stability_Filtered.csv"
    assert com_csv.exists()
    df = pd.read_csv(com_csv)
    assert {"Time (ns)", "COM_Distance_Raw", "COM_Distance_Filt"}.issubset(df.columns)
    np.testing.assert_array_almost_equal(df["COM_Distance_Filt"].values, filtered_com, decimal=4)
