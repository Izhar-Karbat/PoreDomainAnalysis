"""
Module for analyzing Selectivity Filter (SF) Tyrosine residue dynamics.

Functions:
    - analyze_sf_tyrosine_rotamers: Calculate chi1/chi2 angles and classify rotamers.
"""

__version__ = "1.0.0"

from .tyrosine_rotamers import analyze_sf_tyrosine_rotamers

__all__ = [
    "analyze_sf_tyrosine_rotamers",
] 