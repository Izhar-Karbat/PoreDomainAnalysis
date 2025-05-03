# pore_analysis/modules/pocket_analysis/__init__.py
"""
Peripheral Pocket Water Analysis Module

Analyzes water molecule classification, occupancy, and residence times
in peripheral channel pockets using a pre-trained Equivariant Transformer model.
"""

# Import the main orchestrator functions once they are defined
from .computation import run_pocket_analysis
from .visualization import generate_pocket_plots

# Define what gets imported with "from pore_analysis.modules.pocket_analysis import *"
__all__ = [
    'run_pocket_analysis',
    'generate_pocket_plots'
]

# Version of this specific module (optional)
__version__ = "1.0.0" # Initial version for this module