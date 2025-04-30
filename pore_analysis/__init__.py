"""
Pore Analysis Suite

A comprehensive toolkit for analyzing ion channel structures and dynamics
from molecular dynamics trajectory data.

This package has been refactored with a database-centric approach for:
1. Tracking analysis modules execution
2. Registering analysis products
3. Storing key metrics for cross-simulation analysis
"""

from .core import __version__

__all__ = ['__version__']