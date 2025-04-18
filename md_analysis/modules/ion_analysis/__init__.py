"""
Package for analyzing K+ ion behavior in potassium channel simulations.

Focuses on:
 - Tracking ions near the selectivity filter (SF).
 - Determining ion occupancy at binding sites.
 - Analyzing coordination with carbonyl oxygens.
 - Characterizing full conduction events and site-to-site transitions.

Main Functions:
 - track_potassium_ions: Initial tracking and Z-position calculation.
 - analyze_ion_coordination: Calculate coordination numbers.
 - analyze_ion_conduction: Analyze full conduction and site transitions.
"""

__version__ = "1.6.0" # Version update

# Import main functions from submodules to expose them at the package level
from .ion_core import track_potassium_ions
from .ion_position import save_ion_position_data, plot_ion_positions
from .coordination import analyze_ion_coordination
from .filter_structure import find_filter_residues
from .ion_conduction import analyze_ion_conduction # Import new function

# Define what gets imported with "from ion_analysis import *"
__all__ = [
    "track_potassium_ions",
    "analyze_ion_coordination",
    "analyze_ion_conduction", # Add new function to export list
    "save_ion_position_data", # Keep utility functions if needed directly
    "plot_ion_positions",
    "find_filter_residues"
]
