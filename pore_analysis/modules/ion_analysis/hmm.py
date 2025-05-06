"""
Core Hidden Markov Model (HMM) logic for ion transition analysis.
Includes Viterbi decoding, state segmentation, flicker filtering,
and validation checks adapted for the Pore Analysis Suite.
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional, Set
from collections import defaultdict

# Import necessary components from the suite (adjust as needed)
# from pore_analysis.core.config import HMM_EPSILON, HMM_SELF_TRANSITION_P, HMM_EMISSION_SIGMA, HMM_FLICKER_NS
# from pore_analysis.core.utils import frames_to_time # If time conversion needed here

logger = logging.getLogger(__name__)

# Warning aggregator for tracking HMM warnings
class WarningAggregator:
    def __init__(self):
        # For tracking warnings by type and affected ions
        self.warning_counts = defaultdict(int)
        self.warning_ions = defaultdict(set)
        self.warning_examples = {}
        # For tracking specific per-ion warnings
        self.invalid_state_warnings = defaultdict(int)
        self.suspicious_state_warnings = defaultdict(int)
        self.rapid_transition_warnings = defaultdict(int)
        
    def add_warning(self, warning_type, ion_name, message=None):
        """Add a warning of a specific type for an ion"""
        self.warning_counts[warning_type] += 1
        self.warning_ions[warning_type].add(ion_name)
        if message and warning_type not in self.warning_examples:
            self.warning_examples[warning_type] = message
            
    def add_invalid_state(self, ion_name, time_idx):
        """Track invalid state index warnings"""
        self.invalid_state_warnings[ion_name] += 1
        warning_type = "invalid_state_index"
        self.warning_counts[warning_type] += 1
        self.warning_ions[warning_type].add(ion_name)
        if warning_type not in self.warning_examples:
            self.warning_examples[warning_type] = f"Invalid state index detected (example: at time {time_idx} for {ion_name})"
    
    def add_suspicious_state(self, ion_name, count=1):
        """Track suspicious state assignment warnings"""
        self.suspicious_state_warnings[ion_name] += count
        warning_type = "suspicious_state_assignment"
        self.warning_counts[warning_type] += count
        self.warning_ions[warning_type].add(ion_name)
    
    def add_rapid_transition(self, ion_name, count=1):
        """Track rapid transition warnings"""
        self.rapid_transition_warnings[ion_name] += count
        warning_type = "rapid_transitions"
        self.warning_counts[warning_type] += count
        self.warning_ions[warning_type].add(ion_name)
        
    def log_summary(self):
        """Log a summary of all warnings"""
        if not self.warning_counts:
            return
            
        logger.info("=== HMM Warning Summary ===")
        
        # Log invalid state warnings
        if self.invalid_state_warnings:
            total = sum(self.invalid_state_warnings.values())
            ion_count = len(self.invalid_state_warnings)
            top_offenders = sorted(self.invalid_state_warnings.items(), key=lambda x: x[1], reverse=True)[:5]
            example = self.warning_examples.get("invalid_state_index", "Invalid state indices detected")
            logger.warning(f"Invalid state indices: {total} occurrences across {ion_count} ions. Example: {example}")
            logger.warning(f"  Top affected ions: " + ", ".join([f"{ion}({count})" for ion, count in top_offenders]))
            
        # Log suspicious state assignment warnings
        if self.suspicious_state_warnings:
            total = sum(self.suspicious_state_warnings.values())
            ion_count = len(self.suspicious_state_warnings)
            top_offenders = sorted(self.suspicious_state_warnings.items(), key=lambda x: x[1], reverse=True)[:5]
            logger.warning(f"Suspicious state assignments: {total} occurrences across {ion_count} ions")
            logger.warning(f"  Top affected ions: " + ", ".join([f"{ion}({count})" for ion, count in top_offenders]))
            
        # Log rapid transition warnings
        if self.rapid_transition_warnings:
            total = sum(self.rapid_transition_warnings.values())
            ion_count = len(self.rapid_transition_warnings)
            top_offenders = sorted(self.rapid_transition_warnings.items(), key=lambda x: x[1], reverse=True)[:5]
            logger.warning(f"Rapid transitions: {total} occurrences across {ion_count} ions")
            logger.warning(f"  Top affected ions: " + ", ".join([f"{ion}({count})" for ion, count in top_offenders]))
            
        # Log any other warning types
        for warning_type, count in self.warning_counts.items():
            if warning_type not in ["invalid_state_index", "suspicious_state_assignment", "rapid_transitions"]:
                ions = self.warning_ions[warning_type]
                example = self.warning_examples.get(warning_type, "")
                logger.warning(f"{warning_type}: {count} occurrences across {len(ions)} ions. {example}")
                
        logger.info("=== End of HMM Warning Summary ===")

# Create a global warning aggregator
warning_aggregator = WarningAggregator()

# --- HMM Core Functions (Adapted from provided code) ---

def build_transition_matrix(nstates: int, self_transition_p: float, epsilon: float) -> Tuple[np.ndarray, np.ndarray, Dict[int, List[int]]]:
    """Builds the adjacent-only transition matrix and its log form."""
    site_centers = np.array([]) # Placeholder: site centers needed for adjacency logic maybe? Or just nstates is enough.
                               # Actually, nstates is enough for the provided logic.

    # Adjacent‑only transition matrix
    A = np.full((nstates, nstates), epsilon)
    # Adjacency list to track which states are considered adjacent
    adjacency = {}
    for i in range(nstates):
        adjacency[i] = []
        neighbors = 0
        for j in (i - 1, i, i + 1):
            if 0 <= j < nstates:
                 neighbors += 1
                 adjacency[i].append(j)

        # Calculate non-self transition probability (split among neighbors excluding self)
        non_self_neighbors = neighbors - 1 if neighbors > 0 else 0
        p_neighbor = (1 - self_transition_p) / non_self_neighbors if non_self_neighbors > 0 else 0

        for j in adjacency[i]:
             if i == j:
                 A[i, j] = self_transition_p
             else:
                 A[i, j] = p_neighbor # Use calculated probability

    # Row‑normalize (handle cases where sum might not be exactly 1 due to epsilon)
    row_sums = A.sum(axis=1, keepdims=True)
    # Avoid division by zero if a row somehow sums to 0
    row_sums[row_sums == 0] = 1.0
    A = A / row_sums
    logA = np.log(A)

    # Verification (optional, can be done during testing)
    # logger.debug("Verifying transition matrix for adjacency-only constraints...")
    # for i in range(nstates):
    #     for j in range(nstates):
    #         if j not in adjacency[i] and A[i, j] > epsilon * 1.1:
    #             logger.error(f"Adjacency constraint violated: A[{i}, {j}] = {A[i, j]}")

    return A, logA, adjacency

def viterbi(obs: np.ndarray, site_centers: np.ndarray, log_pi: np.ndarray, logA: np.ndarray, emission_sigma: float, adjacency: Dict[int, List[int]], site_names: List[str]) -> tuple[np.ndarray, float]:
    """Viterbi decoding for a Gaussian‑emission HMM. Returns path and log-likelihood."""
    T = len(obs)
    nstates = len(site_centers)

    if T == 0:
        return np.array([], dtype=int), 0.0 # Return empty path and 0 likelihood if no observations

    # Log‑emission probability matrix (T × nstates)
    diff = obs[:, None] - site_centers[None, :]
    logB = -0.5 * (diff ** 2) / (emission_sigma ** 2)            - np.log(emission_sigma * np.sqrt(2 * np.pi))

    V   = np.full((T, nstates), -np.inf) # Initialize with -inf for log domain
    ptr = np.zeros((T, nstates), dtype=int)
    V[0] = log_pi + logB[0] # Initial state probability

    # Forward pass
    for t in range(1, T):
        for j in range(nstates):
            # Consider only transitions from allowed previous states (i -> j)
            # logA[:, j] gives log P(j | i) for all i
            scores = V[t - 1] + logA[:, j] # Scores for transitioning from any i to j
            ptr[t, j] = np.argmax(scores) # Best previous state i
            V[t, j]   = scores[ptr[t, j]] + logB[t, j] # Likelihood of best path ending in j at time t

    # Backtrack
    path = np.zeros(T, dtype=int)
    path[-1] = np.argmax(V[-1]) # Most likely final state
    final_log_likelihood = V[-1, path[-1]] # Log likelihood of the best path

    for t in range(T - 2, -1, -1):
        path[t] = ptr[t + 1, path[t + 1]] # Follow pointers back

    # # Optional Adjacency Enforcement during backtrack (commented out - prefer fixing in forward pass or post-processing)
    # # for t in range(T - 2, -1, -1):
    # #     proposed_state = ptr[t + 1, path[t + 1]]
    # #     if proposed_state not in adjacency.get(path[t + 1], []): # Check if proposed is adjacent to next state
    # #         # Find best adjacent state based on V[t] values
    # #         valid_prev_states = [s for s in adjacency.get(path[t+1], []) if s != path[t+1]] # Adjacent states allowed to transition TO path[t+1]
    # #         if not valid_prev_states: # Should not happen with self-transitions allowed
    # #              path[t] = path[t+1] # Stay in the same state if no valid move backward
    # #         else:
    # #              best_prev_score = -np.inf
    # #              best_prev_state = -1
    # #              for prev_s in valid_prev_states:
    # #                   score = V[t, prev_s] # Check likelihood of being in valid prev state at time t
    # #                   if score > best_prev_score:
    # #                        best_prev_score = score
    # #                        best_prev_state = prev_s
    # #              path[t] = best_prev_state
    # #         logger.warning(f"Enforced adjacency constraint during backtrack at t={t}: "
    # #                      f"changed {site_names[proposed_state]} to {site_names[path[t]]} (leading to {site_names[path[t+1]]})")
    # #     else:
    # #         path[t] = proposed_state

    return path, final_log_likelihood

def validate_path(path: np.ndarray, adjacency: Dict[int, List[int]], time_points: np.ndarray, ion_name: str, site_names: List[str]) -> bool:
    """Validate that the path only contains adjacent transitions."""
    valid = True
    for t in range(1, len(path)):
        # Check if current state is adjacent to previous state OR is the same state
        if path[t] != path[t-1] and path[t] not in adjacency.get(path[t-1], []):
            valid = False
            # Use time_points for logging if available and matches length
            time_str = f"{time_points[t]:.3f} ns" if t < len(time_points) else f"index {t}"
            logger.error(f"Non-adjacent transition detected for {ion_name} at {time_str}: "
                         f"{site_names[path[t-1]]} -> {site_names[path[t]]}")
    return valid

def physical_plausibility_check(path: np.ndarray, obs: np.ndarray, site_centers: np.ndarray, time_points: np.ndarray, ion_name: str, site_names: List[str], threshold: float = 2.5) -> Tuple[bool, np.ndarray]:
    """
    Check if the inferred path is physically plausible given the raw data.
    Aggregates warnings for suspicious assignments and returns a boolean flag array.
    """
    plausible = True
    suspicious_flags = np.zeros(len(path), dtype=bool)
    suspicious_count = 0
    invalid_state_count = 0
    suspicious_examples = []

    for t in range(len(path)):
        if path[t] < 0 or path[t] >= len(site_centers): # Should not happen if Viterbi worked
             # Aggregate warning instead of logging each occurrence
             invalid_state_count += 1
             warning_aggregator.add_invalid_state(ion_name, t)
             continue

        assigned_site_center = site_centers[path[t]]
        deviation = abs(obs[t] - assigned_site_center)

        if deviation > threshold:
            suspicious_flags[t] = True
            suspicious_count += 1
            if suspicious_count <= 5: # Store first few examples for later reporting
                 time_str = f"{time_points[t]:.3f} ns" if t < len(time_points) else f"index {t}"
                 suspicious_examples.append(
                     f"at {time_str}: assigned to {site_names[path[t]]} ({assigned_site_center:.2f} Å), "
                     f"raw pos {obs[t]:.2f} Å (Dev: {deviation:.2f} Å > {threshold} Å)"
                 )
            plausible = False # Mark as potentially implausible if any point deviates significantly

    # Log summary for suspicious state assignments if any found
    if suspicious_count > 0:
        # Save just a brief example for aggregation
        if suspicious_examples:
            example_msg = f"Example: {suspicious_examples[0]}"
            warning_aggregator.add_warning("suspicious_state_details", ion_name, example_msg)
        # Add to aggregated counts
        warning_aggregator.add_suspicious_state(ion_name, suspicious_count)
        
        # Log detailed examples immediately only if there are just a few
        if suspicious_count <= 5:
            logger.debug(f"Found {suspicious_count} suspicious site assignments for {ion_name}")
            for example in suspicious_examples:
                logger.debug(f"  {example}")

    # Add check for rapid transitions skipping sites (from original code)
    transitions_checked = 0
    rapid_transition_examples = []
    
    for t in range(1, len(path) - 1):
        if path[t-1] >= 0 and path[t] >= 0 and path[t+1] >= 0: # Ensure valid states
             # Check if jump is > 1 site and the intermediate state is different from start/end
             if abs(path[t+1] - path[t-1]) > 1 and path[t] not in [path[t-1], path[t+1]]:
                 if transitions_checked < 3:  # Store only a few examples
                     time_str = f"{time_points[t]:.3f} ns" if t < len(time_points) else f"index {t}"
                     rapid_transition_examples.append(
                         f"at {time_str}: {site_names[path[t-1]]} -> {site_names[path[t]]} -> {site_names[path[t+1]]}"
                     )
                 transitions_checked += 1
                 plausible = False # Consider this implausible

    # Add rapid transitions to aggregator
    if transitions_checked > 0:
        warning_aggregator.add_rapid_transition(ion_name, transitions_checked)
        # Save an example for the warning summary
        if rapid_transition_examples:
            example_msg = f"Example: {rapid_transition_examples[0]}"
            warning_aggregator.add_warning("rapid_transition_details", ion_name, example_msg)
            
        # Log detailed examples immediately only if there are just a few
        if transitions_checked <= 3:
            logger.debug(f"Found {transitions_checked} suspicious rapid transitions for {ion_name}")
            for example in rapid_transition_examples:
                logger.debug(f"  {example}")

    return plausible, suspicious_flags


def segment_and_filter(path_indices: np.ndarray, site_names: List[str], time_points: np.ndarray, flicker_ns: float, ion_name: str) -> list[dict]:
    """Segment HMM path, apply flicker filter, and collapse identical adjacent segments."""
    labels = [site_names[i] if i >= 0 else "Outside" for i in path_indices] # Handle potential -1 from segmentation
    dt = time_points[1] - time_points[0] if len(time_points) > 1 else 0.0
    flicker_frames = int(flicker_ns / dt) if dt > 0 else 0

    # Initial segmentation
    segs = []
    i = 0
    n = len(labels)
    while i < n:
        j = i + 1
        while j < n and labels[j] == labels[i]:
            j += 1
        segs.append({'label': labels[i], 'start': i, 'end': j - 1})
        i = j

    # Flicker filter
    if flicker_frames > 0:
         filtered = []
         i = 0
         while i < len(segs):
             seg = segs[i]
             length = seg['end'] - seg['start'] + 1

             # Check if segment is short AND has identical neighbours AND is not 'Outside'
             is_short_flicker = (length < flicker_frames and
                                 0 < i < len(segs) - 1 and
                                 segs[i - 1]['label'] == segs[i + 1]['label'] and
                                 seg['label'] != "Outside" and # Don't merge across 'Outside' state
                                 segs[i - 1]['label'] != "Outside")

             if is_short_flicker:
                 prev, nxt = segs[i - 1], segs[i + 1]
                 # Merge with previous segment if it exists in filtered list
                 if filtered and filtered[-1]['start'] == prev['start']:
                     filtered[-1]['end'] = nxt['end'] # Extend the end of the previous segment
                 else:
                     # This case shouldn't happen if logic is correct, but handle defensively
                     merged = {'label': prev['label'], 'start': prev['start'], 'end': nxt['end']}
                     filtered.append(merged)
                 # logger.debug(f"Merged flicker for {ion_name}: {seg['label']} into {prev['label']}")
                 i += 2  # skip current flicker and next segment (which was merged into prev)
             elif length < flicker_frames and seg['label'] != "Outside":
                 # Discard short flickers that cannot be merged (e.g., at start/end or between different states)
                 start_t = time_points[seg['start']] if seg['start'] < len(time_points) else -1
                 end_t = time_points[seg['end']] if seg['end'] < len(time_points) else -1
                 logger.debug(f"Discarding flicker for {ion_name}: {seg['label']} "
                             f"from {start_t:.3f} to {end_t:.3f} ns (length {length} < {flicker_frames} frames)")
                 i += 1
             else: # Keep segment if long enough or 'Outside'
                 filtered.append(seg)
                 i += 1
    else: # No flicker filter applied
         filtered = segs

    # Collapse adjacent identical (handles cases post-filtering or if no filter applied)
    if not filtered: return [] # Handle empty list after filtering
    collapsed = [filtered[0].copy()]
    for seg in filtered[1:]:
        if seg['label'] == collapsed[-1]['label']:
            collapsed[-1]['end'] = seg['end'] # Extend the end time
        else:
            collapsed.append(seg.copy())

    return collapsed


def identify_continuous_segments(ion_data: np.ndarray) -> List[Tuple[int, int]]:
    """
    Identify continuous segments of non-NaN values in the data.
    Handles sequences that start or end with non-NaN values.
    """
    if len(ion_data) == 0:
        return []

    nan_mask = np.isnan(ion_data)
    # Find where the mask changes value (True->False or False->True)
    change_points = np.where(np.diff(nan_mask))[0] + 1

    segments = []
    start_idx = 0 if not nan_mask[0] else None # Start segment at 0 if first value is valid

    for idx in change_points:
        if nan_mask[idx-1] and not nan_mask[idx]: # Transition NaN -> Valid (Start of segment)
            start_idx = idx
        elif not nan_mask[idx-1] and nan_mask[idx]: # Transition Valid -> NaN (End of segment)
            if start_idx is not None:
                 segments.append((start_idx, idx - 1))
                 start_idx = None # Reset start

    # Handle segment that runs to the end of the array
    if start_idx is not None:
        segments.append((start_idx, len(ion_data) - 1))

    return segments


def find_entry_exit_events(ion_data: np.ndarray, time_points: np.ndarray) -> List[Dict[str, Any]]:
    """
    Identify entry and exit events based on NaN transitions.
    Returns event time and index.
    """
    if len(ion_data) < 2: return []

    events = []
    nan_mask = np.isnan(ion_data)

    for i in range(1, len(nan_mask)):
        if nan_mask[i-1] and not nan_mask[i]: # Entry: NaN -> Valid
            events.append({'type': 'entry', 'index': i, 'time': time_points[i]})
        elif not nan_mask[i-1] and nan_mask[i]: # Exit: Valid -> NaN
            events.append({'type': 'exit', 'index': i-1, 'time': time_points[i-1]}) # Use time of last valid point

    return events


def process_ion_with_hmm(
    ion_idx: int,
    ion_name: str,
    ion_z_g1_nan: np.ndarray,
    time_points: np.ndarray,
    site_names: List[str],
    site_centers: np.ndarray,
    log_pi: np.ndarray,
    logA: np.ndarray,
    adjacency: Dict[int, List[int]],
    hmm_params: Dict[str, Any]
) -> Tuple[Optional[List[Dict]], Optional[np.ndarray], Optional[List[Dict]], Optional[np.ndarray]]:
    """
    Processes a single ion's trajectory using HMM, handling NaN segments.

    Args:
        ion_idx: The original index/ID of the ion.
        ion_name: The name of the ion column (e.g., 'Ion_123').
        ion_z_g1_nan: G1-centric Z positions with NaNs when outside filter.
        time_points: Array of time points.
        site_names: List of binding site names in order.
        site_centers: Array of optimized binding site Z centers.
        log_pi: Log initial state probabilities.
        logA: Log transition matrix.
        adjacency: Adjacency dictionary for states.
        hmm_params: Dictionary containing HMM parameters like emission_sigma, flicker_ns.

    Returns:
        Tuple containing:
        - final_transitions: List of collapsed, filtered transition events [{'start_frame', 'end_frame', 'start_time', 'end_time', 'site_label'}].
        - full_hmm_path: Array of HMM state indices for the entire trajectory (-1 for NaN).
        - entry_exit_events: List of detected entry/exit events [{'type', 'index', 'time'}].
        - quality_flags: Array of boolean flags indicating suspicious assignments.
        Returns (None, None, None, None) on critical error.
    """
    logger.info(f"Processing ion {ion_name} (Index: {ion_idx}) with HMM...")
    emission_sigma = hmm_params.get('emission_sigma', 0.8)
    flicker_ns = hmm_params.get('flicker_ns', 1.0)
    n_frames = len(time_points)

    # 1. Identify continuous segments and entry/exit events
    segments = identify_continuous_segments(ion_z_g1_nan)
    entry_exit = find_entry_exit_events(ion_z_g1_nan, time_points)
    logger.debug(f"Ion {ion_name}: Found {len(segments)} segments, {len(entry_exit)} entry/exit events.")

    # 2. Process each segment with Viterbi
    full_hmm_path = np.full(n_frames, -1, dtype=int) # -1 indicates outside filter/NaN
    all_segment_transitions = []

    for seg_idx, (start_frame, end_frame) in enumerate(segments):
         segment_data = ion_z_g1_nan[start_frame : end_frame + 1]
         if len(segment_data) == 0: continue # Skip empty segments

         # Ensure no NaNs within the identified segment (shouldn't happen with identify_continuous_segments)
         if np.any(np.isnan(segment_data)):
              warning_aggregator.add_warning("nan_in_segment", ion_name, 
                                           f"NaN found within supposedly continuous segment {seg_idx+1} for {ion_name}")
              logger.warning(f"NaN found within supposedly continuous segment {seg_idx+1} for ion {ion_name}. Skipping segment.")
              continue

         hmm_path_indices, log_likelihood = viterbi(segment_data, site_centers, log_pi, logA, emission_sigma, adjacency, site_names)

         # Store path for this segment
         if len(hmm_path_indices) == len(segment_data):
              full_hmm_path[start_frame : end_frame + 1] = hmm_path_indices
         else:
              warning_aggregator.add_warning("length_mismatch", ion_name, 
                                          f"Length mismatch between HMM path and segment data for {ion_name}")
              logger.error(f"Length mismatch between HMM path ({len(hmm_path_indices)}) and segment data ({len(segment_data)}) for ion {ion_name}, segment {seg_idx+1}.")
              # Handle error - potentially skip segment or fill with -1? Fill with -1 for safety.
              full_hmm_path[start_frame : end_frame + 1] = -1


    # 3. Validate the full path (across segments - check transitions *at* segment boundaries implicitly)
    # Note: Validation should ideally happen per segment before merging, but let's check the full one too.
    # We need to handle the -1 values representing "Outside" state during validation.
    # Simplified validation for now: check only within segments. Post-processing needed if jumps *between* segments occur.
    is_valid_overall = True
    for start_frame, end_frame in segments:
         segment_path = full_hmm_path[start_frame : end_frame + 1]
         if not validate_path(segment_path, adjacency, time_points[start_frame : end_frame + 1], ion_name, site_names):
              is_valid_overall = False
              warning_aggregator.add_warning("non_adjacent_transition", ion_name, 
                                         f"Non-adjacent transition found within segment for {ion_name}")
              logger.error(f"Non-adjacent transition found within a segment for ion {ion_name}. Check Viterbi/Matrix.")
              # Attempt basic correction? Keep previous state? For now, just log error.

    # 4. Perform plausibility check and get quality flags
    plausible, quality_flags = physical_plausibility_check(full_hmm_path, ion_z_g1_nan, site_centers, time_points, ion_name, site_names)
    if not plausible:
        # The physical_plausibility_check already adds detailed warnings to the aggregator
        logger.debug(f"Physical plausibility concerns raised for ion {ion_name}.")


    # 5. Segment the *full* path (including -1 states) and filter
    # The segment_and_filter function needs the full time_points array for correct time mapping
    final_transitions_dict = segment_and_filter(full_hmm_path, site_names, time_points, flicker_ns, ion_name)

    # Convert final transitions to include time and frame info
    final_transitions = []
    for seg_dict in final_transitions_dict:
         start_f, end_f = seg_dict['start'], seg_dict['end']
         # Ensure frame indices are valid
         if 0 <= start_f < n_frames and 0 <= end_f < n_frames:
              final_transitions.append({
                   'start_frame': start_f,
                   'end_frame': end_f,
                   'start_time': time_points[start_f],
                   'end_time': time_points[end_f],
                   'site_label': seg_dict['label']
              })
         else:
              warning_aggregator.add_warning("invalid_frame_indices", ion_name,
                                          f"Invalid frame indices in filtered segment for {ion_name}")
              logger.warning(f"Invalid frame indices in filtered segment for ion {ion_name}: {seg_dict}")

    logger.info(f"Processing ion {ion_name} complete. Found {len(final_transitions)} final dwell events.")
    return final_transitions, full_hmm_path, entry_exit, quality_flags


# This has been replaced with an active implementation at the top of the file
# The warning_aggregator instance is created there and used throughout the code

