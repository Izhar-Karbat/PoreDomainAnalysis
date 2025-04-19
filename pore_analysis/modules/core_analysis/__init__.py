"""
Core analysis module for molecular dynamics trajectory analysis.
This module provides functions for analyzing trajectories, calculating distances,
and applying filtering to the data.
"""

from .core import analyze_trajectory, filter_and_save_data

__all__ = ['analyze_trajectory', 'filter_and_save_data']
