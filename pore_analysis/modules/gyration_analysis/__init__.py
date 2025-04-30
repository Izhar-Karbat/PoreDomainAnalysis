# filename: pore_analysis/modules/gyration_analysis/__init__.py
"""
Gyration Analysis Package

This package provides tools for analyzing the radius of gyration of carbonyl groups
in the selectivity filter of potassium channels, which can indicate flipping events.
Refactored for database integration and separation of computation/visualization.
"""

from .computation import run_gyration_analysis
from .visualization import generate_gyration_plots

__all__ = [
    'run_gyration_analysis',
    'generate_gyration_plots'
]
