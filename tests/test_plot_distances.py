import os
import numpy as np
import matplotlib

# Use a non-interactive backend for tests
matplotlib.use('Agg')

import pytest
from pore_analysis.modules.core_analysis.core import plot_distances

def test_plot_distances_gg(tmp_path):
    # Synthetic data: simple linear trend
    time_points = np.array([0, 1, 2, 3])
    raw_ac = np.array([1.0, 2.0, 3.0, 4.0])
    raw_bd = np.array([1.5, 2.5, 3.5, 4.5])
    filt_ac = raw_ac * 0.9
    filt_bd = raw_bd * 0.9

    output_dir = tmp_path / "plots"
    output_dir.mkdir()

    # Test G-G plot generation
    success = plot_distances(
        time_points,
        raw_ac,
        raw_bd,
        filt_ac,
        filt_bd,
        str(output_dir),
        title_prefix="Test",
        is_gg=True,
        logger=None
    )
    assert success, "plot_distances should return True for valid G-G inputs"

    expected_files = [
        "G_G_Distance_Plot_raw.png",
        "G_G_Distance_Plot.png",
        "G_G_Distance_AC_Comparison.png",
        "G_G_Distance_BD_Comparison.png"
    ]
    for fname in expected_files:
        fpath = output_dir / fname
        assert fpath.exists() and fpath.stat().st_size > 0, f"Expected plot file {fname} to exist and be non-empty"

def test_plot_distances_com(tmp_path):
    # Synthetic data: simple oscillation
    time_points = np.array([0, 1, 2, 3])
    raw_com = np.array([0.0, 1.0, 0.5, 1.5])
    filt_com = raw_com * 0.8

    output_dir = tmp_path / "com_plots"
    output_dir.mkdir()

    # Test COM plot generation
    success = plot_distances(
        time_points,
        raw_com,
        raw_data2=None,
        filtered_data1=filt_com,
        filtered_data2=None,
        output_dir=str(output_dir),
        title_prefix="ComTest",
        is_gg=False,
        logger=None
    )
    assert success, "plot_distances should return True for valid COM inputs"

    expected_files = [
        "COM_Distance_Plot_raw.png",
        "COM_Distance_Plot.png"
    ]
    for fname in expected_files:
        fpath = output_dir / fname
        assert fpath.exists() and fpath.stat().st_size > 0, f"Expected plot file {fname} to exist and be non-empty"
