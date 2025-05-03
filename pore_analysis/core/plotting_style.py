# pore_analysis/core/plotting_style.py
"""
Defines the unified plotting style configuration for the Pore Analysis Suite.
Includes the STYLE dictionary and the setup_style() function to apply these settings.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# Define uniform style guidelines
STYLE = {
    # Channel/Subunit Colors (Pastel)
    'chain_colors': {
        'A': '#AEC6CF',  # pastel blue
        'B': '#FFD1DC',  # pastel pink
        'C': '#C3FDB8',  # pastel green
        'D': '#FFB347',  # pastel orange
    },
    # Bright colors for overlays
    'bright_colors': {
        'A': '#4682B4',  # steel blue / bright blue
        'B': '#FF6B81',  # bright pink / bright red
        'C': '#32CD32',  # lime green
        'D': '#FF8C00',  # dark orange
    },
    # Semantic state colors
    'state_colors': {
        'closed': '#ADD8E6',  # light blue
        'open': '#FACBAA',    # light orange
    },
    # Line styles
    'line_width': 2,
    'threshold_style': {
        'linestyle': '--',
        'color': 'black',
        'alpha': 0.6,
        'linewidth': 2,
    },
    # Marker styles
    'event_marker': {
        'marker': 'o',
        'markersize': 8,
        'markeredgecolor': 'black',
        'markeredgewidth': 0.5,
    },
    'mean_marker': {
        'marker': 'o',
        'markersize': 8,
        'markerfacecolor': 'red',
        'markeredgecolor': 'white',
        'markeredgewidth': 0.5,
    },
    # Typography - INCREASED FONT SIZES
    'font_family': 'sans-serif',
    'font_sizes': {
        'title': 16, # Kept for potential internal use, but not applied to plots per guidelines
        'axis_label': 14, # Increased from 12
        'tick_label': 12, # Increased from 10
        'annotation': 11, # Increased from 9 (controls legend size)
    },
    # Grid
    'grid': {
        'color': 'lightgrey',
        'alpha': 0.3,
        'linestyle': '-',
    },
}

# Set up the default style
def setup_style():
    """Apply the unified styling to matplotlib and seaborn"""
    # Use seaborn's whitegrid style as a base and customize grid appearance
    sns.set_style("whitegrid", {
        'grid.color': STYLE['grid']['color'],
        'grid.alpha': STYLE['grid']['alpha'],
        'grid.linestyle': STYLE['grid']['linestyle'],
    })

    # Apply font settings from STYLE dictionary
    mpl.rcParams['font.family'] = STYLE['font_family']
    # Note: Titles are handled in HTML, so axes.titlesize is commented out
    # mpl.rcParams['axes.titlesize'] = STYLE['font_sizes']['title']
    mpl.rcParams['axes.labelsize'] = STYLE['font_sizes']['axis_label']
    mpl.rcParams['xtick.labelsize'] = STYLE['font_sizes']['tick_label']
    mpl.rcParams['ytick.labelsize'] = STYLE['font_sizes']['tick_label']
    mpl.rcParams['legend.fontsize'] = STYLE['font_sizes']['annotation']

    # Set default line width
    mpl.rcParams['lines.linewidth'] = STYLE['line_width']

    # Set figure DPI for better rendering quality in saved files and potentially displays
    mpl.rcParams['figure.dpi'] = 100        # DPI for interactive display (if used)
    mpl.rcParams['savefig.dpi'] = 150       # DPI for saved figures (e.g., PNGs for report)

    # Ensure backend is non-interactive ('agg' is good for scripts)
    # plt.switch_backend('agg') # This might be better called once at the start of a script/module using plotting