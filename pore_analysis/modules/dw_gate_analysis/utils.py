"""Utility functions and constants for DW-gate analysis."""

import os
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# --- State Constants ---
CLOSED_STATE = "closed"
OPEN_STATE = "open"

# --- Plotting Helpers ---
def save_plot(fig, path, dpi=300):
    """Save plot with error handling and directory creation."""
    try:
        plot_dir = os.path.dirname(path)
        if plot_dir:
            os.makedirs(plot_dir, exist_ok=True) # Ensure dir exists
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved plot: {path}")
    except Exception as e:
        logger.error(f"Failed to save plot {path}: {e}")
    finally:
        plt.close(fig) # Ensure plot is closed 