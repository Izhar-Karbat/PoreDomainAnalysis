# filename: pore_analysis/modules/dw_gate_analysis/__init__.py
"""
DW-Gate Analysis Module Entry Point.

Exposes the primary functions for running the computation and visualization
steps of the DW-Gate analysis.
"""

# Import the primary functions from the computation and visualization submodules
from .computation import run_dw_gate_analysis
from .visualization import generate_dw_gate_plots

# Define what gets imported with "from pore_analysis.modules.dw_gate_analysis import *"
__all__ = [
    'run_dw_gate_analysis',
    'generate_dw_gate_plots'
]

# Version of this specific module (optional)
__version__ = "2.0.0" # Align with overall suite version or manage separately
