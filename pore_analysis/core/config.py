# config.py
"""
Configuration settings for the MD Analysis Script.
"""

# --- Global Version ---
Analysis_version = "2.0.0"
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


# === DW Gate and Tyr-Thr Hydrogen Bond Analysis Parameters ===
# These parameters are used for both DW gate and Tyr-Thr hydrogen bond analysis

# Frames to tolerate when debouncing transitions between states
DW_GATE_TOLERANCE_FRAMES = 5 # Used for both DW gate and Tyr-Thr H-bond debouncing

# Whether to auto-detect reference distances from KDE
DW_GATE_AUTO_DETECT_REFS: bool = True # Used for both DW gate and Tyr-Thr H-bond state detection

# Default reference distances if auto-detection is disabled or fails
# For DW Gate:
DW_GATE_DEFAULT_CLOSED_REF_DIST: float = 2.70 # Default reference distance (Å) for CLOSED state if auto-detect fails
DW_GATE_DEFAULT_OPEN_REF_DIST: float = 4.70 # Default reference distance (Å) for OPEN state if auto-detect fails

# For Tyr-Thr H-bond: 
TYR_THR_DEFAULT_FORMED_REF_DIST: float = 3.5  # Å (H-bond formed)
TYR_THR_DEFAULT_BROKEN_REF_DIST: float = 4.5  # Å (H-bond broken)


# --- Ion Analysis: Site Optimization Parameters ---
#: Maximum distance (Å) to search for density peaks around predefined sites
SITE_OPT_SEARCH_RADIUS: float = 1.5
#: Minimum relative height (fraction of max) for a valid density peak
SITE_OPT_MIN_PEAK_HEIGHT: float = 0.2
#: Number of bins for histogram used in site detection
SITE_OPT_HIST_BINS: int = 100
#: Window size for moving average smoothing of the site density histogram
SITE_OPT_SMOOTH_WINDOW: int = 5

# --- Ion Analysis: HMM Parameters ---
#: Probability for non-adjacent site jumps (keep small)
HMM_EPSILON: float = 1e-6
#: Self-transition probability (favors staying in current state)
HMM_SELF_TRANSITION_P: float = 0.93
#: Emission standard deviation (Å) for Gaussian HMM
HMM_EMISSION_SIGMA: float = 0.8
#: Flicker filter threshold (ns) - states shorter than this may be merged/discarded
HMM_FLICKER_NS: float = 1.0

# --- Ion Analysis: Visualization Parameters ---
#: Occupancy threshold (%) for primary detailed visualization
VIZ_HIGH_OCC_THRESHOLD_PCT: float = 10.0
#: Occupancy threshold (%) for medium detail visualization
VIZ_MEDIUM_OCC_THRESHOLD_PCT: float = 1.0

# Parameters for Tyrosine HMM Analysis ---
#: Define the 9 canonical rotamer states (Chi1, Chi2) using angle centers (t=-180/180, p=60, m=-60)
#: Note: Using 180 for 't' for simplicity in center calculation. Viterbi handles circularity via emission prob.
TYR_HMM_STATES = {
    'mt': (-60, 180), 'mm': (-60, -60), 'mp': (-60, 60),
    'pt': (60, 180),  'pm': (60, -60),  'pp': (60, 60),
    'tt': (180, 180), 'tm': (180, -60), 'tp': (180, 60),
}
#: Order of states used for matrix building (MUST MATCH TYR_HMM_STATES keys)
TYR_HMM_STATE_ORDER = ['mt', 'mm', 'mp', 'pt', 'pm', 'pp', 'tt', 'tm', 'tp']
#: Emission standard deviation (degrees) for Gaussian HMM (applied to Chi1 and Chi2)
#: Assumes isotropic deviation for simplicity. Tune this value (e.g., 15-25 degrees).
TYR_HMM_EMISSION_SIGMA: float = 20.0
#: Probability of staying in the same rotamer state (should be high)
TYR_HMM_SELF_TRANSITION_P: float = 0.95
#: Small probability for non-self transitions
TYR_HMM_EPSILON: float = 1e-5
#: Flicker filter threshold (ns) - HMM states shorter than this may be merged/discarded
TYR_HMM_FLICKER_NS: float = 0.5

# <<<--- START: Pocket Analysis Parameters --->>>
# Sequence for identifying the filter (used to define regions relative to it)
POCKET_ANALYSIS_FILTER_SEQUENCE = "GYG"
# Residues upstream/downstream of the filter sequence to include in graph representation
POCKET_ANALYSIS_N_UPSTREAM = 10
POCKET_ANALYSIS_N_DOWNSTREAM = 10
# Water selection cylinder parameters (relative to filter center)
POCKET_ANALYSIS_CYLINDER_RADIUS = 15.0
POCKET_ANALYSIS_INITIAL_HEIGHT = 20.0
POCKET_ANALYSIS_MIN_WATERS = 100 # Minimum waters to capture initially
# RMSF calculation window size (frames)
POCKET_ANALYSIS_RMSF_WINDOW = 10
# Residence time analysis threshold (frames)
POCKET_ANALYSIS_RESIDENCE_THRESHOLD = 10 # Corresponds to 1 ns if FRAMES_PER_NS=10
# Tolerance window (frames) for smoothing discontinuous water trajectories
POCKET_ANALYSIS_TRAJECTORY_TOLERANCE = 5
# ML Model file paths (relative to the pocket_analysis module directory)
POCKET_ANALYSIS_MODEL_CONFIG_RELPATH = "ml_model/pocket_model_config.json"
POCKET_ANALYSIS_MODEL_WEIGHTS_RELPATH = "ml_model/pocket_model.pth"
# Residence time thresholds for categorization (ns)
POCKET_ANALYSIS_SHORT_LIVED_THRESH_NS = 5.0
POCKET_ANALYSIS_LONG_LIVED_THRESH_NS = 10.0
# <<<--- END: Pocket Analysis Parameters --->>>