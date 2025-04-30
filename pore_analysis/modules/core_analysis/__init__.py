"""
Core analysis module for molecular dynamics trajectory analysis.
This module provides functions for analyzing trajectories, calculating distances,
applying filtering to the data, and visualizing results.

The module has been refactored to:
1. Separate computation from visualization
2. Register analysis products in a database
3. Store metrics for cross-simulation analysis
"""

from .computation import analyze_trajectory, filter_and_save_data
from .visualization import plot_distances, plot_kde_analysis

__all__ = [
    'analyze_trajectory',
    'filter_and_save_data',
    'plot_distances',
    'plot_kde_analysis'
]