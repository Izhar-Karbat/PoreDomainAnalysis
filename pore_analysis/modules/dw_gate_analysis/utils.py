# filename: pore_analysis/modules/dw_gate_analysis/utils.py
"""Utility functions and constants for DW-gate analysis."""

import os
import logging
import matplotlib.pyplot as plt # Keep import if save_plot uses it

logger = logging.getLogger(__name__)

# --- State Constants ---
CLOSED_STATE = "closed"
OPEN_STATE = "open"

# --- Plotting Helpers ---
def save_plot(fig, path, dpi=150): # Adjusted default DPI slightly based on other modules
    """Save matplotlib figure with error handling and directory creation."""
    try:
        plot_dir = os.path.dirname(path)
        if plot_dir:
            os.makedirs(plot_dir, exist_ok=True) # Ensure dir exists
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved plot: {path}")
    except Exception as e:
        logger.error(f"Failed to save plot {path}: {e}", exc_info=True) # Added exc_info for debugging
    finally:
        # Ensure plot is closed even if saving failed
        if fig is not None and plt.fignum_exists(fig.number):
             plt.close(fig)