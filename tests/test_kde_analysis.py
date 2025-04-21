import numpy as np
import matplotlib
matplotlib.use('Agg')
import pytest
from pore_analysis.modules.core_analysis.core import plot_kde_analysis

def test_plot_kde_analysis(tmp_path):
    # Create synthetic data with enough points for KDE
    time_points = np.linspace(0, 9, 10)
    com_distances = np.linspace(1.0, 2.0, 10)
    box_z = None

    # Prepare output directory
    output_dir = tmp_path / "kde_plots"
    output_dir.mkdir()

    # Run the KDE analysis
    success = plot_kde_analysis(time_points, com_distances, box_z, str(output_dir))
    assert success, "plot_kde_analysis should return True for sufficient data"

    # Verify the output file was created and is non-empty
    out_file = output_dir / "COM_Stability_KDE_Analysis.png"
    assert out_file.exists() and out_file.stat().st_size > 0, "KDE plot file should exist and be non-empty"
