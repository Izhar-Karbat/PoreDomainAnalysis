# config.py
"""
Configuration settings for the MD Analysis Script.
"""

# --- Global Version ---
Analysis_version = "1.5.0"
# ---------------------

# --- Simulation Parameters ---
# Number of frames per nanosecond in the trajectory
FRAMES_PER_NS = 10

# --- Analysis Parameters ---
# Default distance cutoff for defining contacts (Angstroms)
DEFAULT_CUTOFF = 3.5

# Default stride factor for reading large trajectories (set to 1 to read all frames)
DEFAULT_STRIDE = 1

# --- Water Analysis Parameters ---
# Number of consecutive frames water must be outside cavity to confirm exit
EXIT_BUFFER_FRAMES = 5

# Add any other system-wide constants or default thresholds here
# Example:
# MIN_LEVEL_SIZE_PBC = 30 # Could move from detect_and_correct_multilevel_pbc if desired
