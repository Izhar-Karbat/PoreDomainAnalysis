# config.py
"""
Configuration settings for the MD Analysis Script.
"""

# --- Global Version ---
Analysis_version = "1.6.1"
# ---------------------

# --- Simulation Parameters ---
# Number of frames per nanosecond in the trajectory
FRAMES_PER_NS = 10

# --- Analysis Parameters ---
# Default distance cutoff for defining contacts (Angstroms)
DEFAULT_CUTOFF = 3.5

# Default stride factor for reading large trajectories (set to 1 to read all frames)
DEFAULT_STRIDE = 1

# --- Gyration Analysis Parameters ---
# Threshold (in Angstroms) to define a carbonyl as 'flipped' based on gyration radius
GYRATION_FLIP_THRESHOLD = 4.5
# Minimum consecutive frames in a state to confirm a flip event
GYRATION_FLIP_TOLERANCE_FRAMES = 5

# --- Water Analysis Parameters ---
# Number of consecutive frames water must be outside cavity to confirm exit
EXIT_BUFFER_FRAMES = 5

# --- Ion Analysis Parameters ---
#: Mode for transition tolerance. Options:
#:   'strict'   — two-sided penalty (both past and future windows)
#:   'forward'  — only forward window
#:   'majority' — majority-rule in both windows
ION_TRANSITION_TOLERANCE_MODE: str = "strict"
#: Number of frames in each tolerance window (used for forward and backward checks)
ION_TRANSITION_TOLERANCE_FRAMES: int = 5
#: If True, compute per-site thresholds; otherwise use uniform value
ION_USE_SITE_SPECIFIC_THRESHOLDS: bool = True
#: Uniform occupancy threshold (Å) if site-specific disabled
ION_SITE_OCCUPANCY_THRESHOLD_A: float = 2.0

# Add any other system-wide constants or default thresholds here
# Example:
# MIN_LEVEL_SIZE_PBC = 30 # Could move from detect_and_correct_multilevel_pbc if desired

# --- Constants for Tyrosine Analysis ---
# Tolerance: Minimum frames a new rotamer state must persist to be counted as a transition
TYROSINE_ROTAMER_TOLERANCE_FRAMES = 5

# --- Ion Conduction / Transition Analysis ---
ION_TRANSITION_TOLERANCE_FRAMES = 10 # Frames an ion must stay in a site to count as entry
ION_TRANSITION_TOLERANCE_MODE = 'entry' # Alternative: 'exit', 'midpoint'

# --- DW Gate Analysis ---
DW_GATE_TOLERANCE_FRAMES = 5 # Frames a DW gate must stay open/closed to count as a confirmed event
