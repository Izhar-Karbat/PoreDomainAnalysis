# filename: pore_analysis/modules/ion_analysis/__init__.py
"""
Ion Analysis Module

Analyzes K+ ion behavior in channel simulations, including structure definition,
tracking, occupancy, conduction, and visualization.
"""

# Import the main orchestrator functions
from .computation import run_ion_analysis
from .visualization import generate_ion_plots

# Define what gets imported with "from pore_analysis.modules.ion_analysis import *"
__all__ = [
    'run_ion_analysis',
    'generate_ion_plots'
]

# Version of this specific module (optional)
__version__ = "2.0.0"
