"""
Core functions for gyration analysis of carbonyl groups in the selectivity filter.

This module is a thin wrapper around the main gyration_analysis.py module,
re-exporting the main entry point function.
"""

from .gyration_analysis import analyze_carbonyl_gyration

__all__ = ['analyze_carbonyl_gyration']
