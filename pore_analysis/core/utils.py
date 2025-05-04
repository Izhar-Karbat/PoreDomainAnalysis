# utils.py
"""
Utility functions for the MD Analysis Script.
"""

import numpy as np
import base64
from io import BytesIO
import math
from datetime import datetime

# Import the config module to access FRAMES_PER_NS dynamically
from pore_analysis.core import config as core_config

def OneLetter(x):
    """Convert three-letter amino acid codes into one-letter codes."""
    d = {
        'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
        'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
        'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
        'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M',
        'HSE': 'H', 'HSD': 'H'  # Handle common histidine variants
    }
    # Handle potential lowercase or mixed case input
    x_upper = x.upper()
    if len(x_upper) % 3 != 0:
        raise ValueError(f'Input length ({len(x_upper)}) must be a multiple of three. Input: {x}')

    out = []
    for i in range(len(x_upper) // 3):
        triplet = x_upper[3 * i: 3 * i + 3]
        if triplet in d:
            out.append(d[triplet])
        else:
            raise ValueError(f"Unknown amino acid code '{triplet}' encountered in input: {x}")
    
    return "".join(out)


def frames_to_time(frame_indices):
    """
    Convert frame indices to time in nanoseconds based on FRAMES_PER_NS.

    Parameters:
    -----------
    frame_indices : numpy.ndarray or list
        Indices of frames.

    Returns:
    --------
    numpy.ndarray
        Time points in nanoseconds.
    """
    # Access the frames-per-nanosecond value dynamically
    fpns = core_config.FRAMES_PER_NS
    if fpns <= 0:
        raise ValueError("FRAMES_PER_NS must be positive.")
    frame_indices_np = np.array(frame_indices)
    return frame_indices_np / fpns


def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string for HTML embedding."""
    try:
        buf = BytesIO()
        # Save with tight bounding box to potentially reduce whitespace
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        return img_str
    except Exception as e:
        # Log the error if logging is configured, otherwise print
        try:
            import logging
            logging.warning(f"Failed to convert figure to base64: {e}", exc_info=True)
        except ImportError:
            print(f"Warning: Failed to convert figure to base64: {e}")
        return ""  # Return empty string on failure


def clean_json_data(data):
    """Recursively cleans data structure for JSON serialization.
       Converts NaN/Infinity to None, numpy types to Python types.
    """
    if isinstance(data, dict):
        return {k: clean_json_data(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return [clean_json_data(item) for item in data]
    # --- Numpy type handling ---
    elif isinstance(data, (np.int8, np.int16, np.int32, np.int64, np.intc, np.intp,
                           np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(data)
    elif isinstance(data, (np.float16, np.float32, np.float64)):
        # Check for nan/inf specifically for numpy floats
        if np.isnan(data) or np.isinf(data):
            return None  # Represent as JSON null
        else:
            return float(data)
    elif isinstance(data, np.ndarray):
        # Convert arrays to lists, applying cleaning to elements
        return [clean_json_data(item) for item in data.tolist()]
    elif isinstance(data, np.bool_):
        return bool(data)
    elif isinstance(data, np.void):  # Handle numpy void type if it appears
        return None
    # --- Standard Python float handling ---
    elif isinstance(data, float):
        if math.isnan(data) or math.isinf(data):
            return None  # Represent as JSON null
        else:
            return float(data)
    elif isinstance(data, datetime):
        return data.isoformat()  # Convert datetime to ISO string

    # Return data unchanged if already serializable or unknown type
    return data
