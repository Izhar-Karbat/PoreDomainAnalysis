"""
Gyration Analysis Package

This package provides tools for analyzing the radius of gyration of carbonyl groups
in the selectivity filter of potassium channels, which can indicate flipping events.
"""

from .gyration_analysis import analyze_carbonyl_gyration

__all__ = [
    'analyze_carbonyl_gyration'
]
