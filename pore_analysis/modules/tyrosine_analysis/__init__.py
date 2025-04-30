# pore_analysis/modules/tyrosine_analysis/__init__.py
"""
SF Tyrosine rotamer analysis module.
"""

from .computation import run_tyrosine_analysis
from .visualization import generate_tyrosine_plots

__all__ = [
    'run_tyrosine_analysis',
    'generate_tyrosine_plots'
]