"""
Ion Analysis Package

This package provides tools for analyzing K+ ion behavior in potassium channel simulations.
It focuses on tracking ions through the selectivity filter, analyzing occupancy patterns,
and generating relevant statistics and visualizations.

Main entry points:
- track_potassium_ions: Tracks K+ ions through the selectivity filter
- analyze_ion_coordination: Calculates occupancy statistics for binding sites
"""

# Re-export main functions for backward compatibility
from .ion_core import track_potassium_ions
from .ion_occupancy import analyze_ion_coordination
from .binding_sites import visualize_binding_sites_g1_centric
from .filter_structure import find_filter_residues, calculate_tvgyg_sites

# Version info
__version__ = "1.5.0"
