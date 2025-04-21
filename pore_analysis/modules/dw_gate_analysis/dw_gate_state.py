"""
DW-Gate State Analysis: Calculate distances, identify states via KDE, apply RLE debouncing,
build events, calculate statistics, and generate comprehensive plots.
"""

import os
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional # Added Optional

import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array # Added
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors # Keep if needed elsewhere, maybe remove
import seaborn as sns
from scipy.stats import chi2_contingency, kruskal, mannwhitneyu # Added
from scipy.signal import find_peaks # Added
from sklearn.cluster import KMeans # Added
from itertools import combinations # Added

try:
    # Relative imports within the pore_analysis package
    from ..ion_analysis.filter_structure import find_filter_residues # KEEP for initial chain ID
    from .residue_identification import find_gate_residues_for_chain, GateResidues # KEEP for residue ID
    from ...core.utils import frames_to_time # KEEP if needed, maybe replace with dt_ns logic
    # Import specific config values needed
    from ...core.config import (
        DW_GATE_TOLERANCE_FRAMES,
        DEFAULT_CUTOFF,
        FRAMES_PER_NS, # Use this to derive dt_ns
        DW_GATE_AUTO_DETECT_REFS, # New config
        DW_GATE_DEFAULT_CLOSED_REF_DIST, # New config
        DW_GATE_DEFAULT_OPEN_REF_DIST, # New config
    )
except ImportError as e:
    # Use specific logger name if available, else basic config
    try:
        logger = logging.getLogger(__name__)
    except NameError:
        logging.basicConfig()
        logger = logging.getLogger()
    logger.error(f"Error importing dependencies in dw_gate_state.py: {e}")
    # Re-raise to ensure failure if imports are broken
    raise

# --- Logger Setup ---
# Module-specific logger. Configuration is handled by the main script's setup.
logger = logging.getLogger(__name__)

# --- Constants Derived from Config ---
# Use the config values instead of module-level constants
# DISTANCE_THRESHOLD = DEFAULT_CUTOFF # Set inside class init from config
DT_NS = 1.0 / FRAMES_PER_NS if FRAMES_PER_NS > 0 else 0.1 # Calculate dt_ns from config

# Define state names consistently
CLOSED_STATE = "closed"
OPEN_STATE = "open"

# --- Plotting Helpers (Minimalist, adapted) ---
# Keep save_plot, remove setup_plot as figures are created within methods
def save_plot(fig, path, dpi=300): # Increased default DPI
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

# --- NEW DWGateAnalysis Class ---
@dataclass
class GateIndices:
    """Dataclass to hold atom indices for a DW gate in one chain."""
    od1: int
    od2: int
    ne1: int

class DWGateAnalysis:
    """
    Performs DW-gate analysis including KDE reference distance detection,
    RLE debouncing, event building, statistical analysis, and plotting.
    Designed to be integrated into the PoreAnalysis suite.
    """
    def __init__(self, universe: mda.Universe, gate_residues: Dict[str, GateResidues],
                 output_dir: str):
        """
        Initialize the DWGateAnalysis.

        Args:
            universe (mda.Universe): The MDAnalysis Universe object.
            gate_residues (Dict[str, GateResidues]): Dictionary mapping segid to
                                                    identified GateResidues objects.
            output_dir (str): Directory to save plots and intermediate data.
        """
        self.universe = universe
        self.gate_residues = gate_residues # Store the identified residues
        self.output_dir = output_dir # Directory for saving outputs

        # Import config values directly inside __init__ where needed
        from ...core.config import (
            DW_GATE_TOLERANCE_FRAMES,
            DEFAULT_CUTOFF,
            FRAMES_PER_NS,
            DW_GATE_AUTO_DETECT_REFS,
            DW_GATE_DEFAULT_CLOSED_REF_DIST,
            DW_GATE_DEFAULT_OPEN_REF_DIST
        )

        # Extract parameters from imported config values
        self.distance_threshold = DEFAULT_CUTOFF
        self.tolerance_frames = DW_GATE_TOLERANCE_FRAMES
        self.dt_ns = 1.0 / FRAMES_PER_NS if FRAMES_PER_NS > 0 else 0.1 # Calculate from FRAMES_PER_NS
        self.auto_detect_refs = DW_GATE_AUTO_DETECT_REFS
        self.default_closed_ref_dist = DW_GATE_DEFAULT_CLOSED_REF_DIST
        self.default_open_ref_dist = DW_GATE_DEFAULT_OPEN_REF_DIST

        # Initialize state variables
        self.time_axis = None
        self.gate_idx: Dict[str, GateIndices] = {} # Store atom indices here
        self.valid_segids = sorted(list(gate_residues.keys())) # Get segids from input
        self.df_raw = None
        self.df = None # Debounced data
        self.events_df = None
        self.kde_analysis_results = None
        self.stats_results = {} # To store stats like tables, p-values etc.
        self.plot_paths = {} # To store paths of generated plots

        # Pre-calculate time axis
        n_frames = len(self.universe.trajectory)
        self.time_axis = np.arange(n_frames) * self.dt_ns
        logger.info(f"DWGateAnalysis initialized for {len(self.valid_segids)} chains. N_frames={n_frames}, dt={self.dt_ns:.4f} ns.")
        logger.info(f"Params: Threshold={self.distance_threshold} Å, Tolerance={self.tolerance_frames} frames.")
        logger.info(f"Reference dists: AutoDetect={self.auto_detect_refs}, Defaults=({self.default_closed_ref_dist:.2f}, {self.default_open_ref_dist:.2f}) Å.")

        # Identify atoms immediately based on provided gate_residues
        self._identify_dw_gate_atoms()

    def _identify_dw_gate_atoms(self):
        """Populates self.gate_idx using the pre-identified gate_residues."""
        logger.info("Identifying DW-gate atom indices for each valid chain.")
        found_chains = []
        for segid, gate in self.gate_residues.items():
            try:
                asp_atoms = gate.asp_glu_res.atoms.select_atoms("name OD1 OD2 OE1 OE2") # Handle ASP/GLU
                trp_atoms = gate.trp_res.atoms.select_atoms("name NE1")

                # Try to find OD1/OE1 and OD2/OE2 robustly
                od1_idx = asp_atoms.select_atoms("name OD1 OE1").indices
                od2_idx = asp_atoms.select_atoms("name OD2 OE2").indices
                ne1_idx = trp_atoms.indices

                if len(od1_idx) != 1:
                    raise ValueError(f"Could not find exactly one OD1/OE1 atom in {gate.asp_glu_res}")
                if len(od2_idx) != 1:
                    raise ValueError(f"Could not find exactly one OD2/OE2 atom in {gate.asp_glu_res}")
                if len(ne1_idx) != 1:
                     raise ValueError(f"Could not find exactly one NE1 atom in {gate.trp_res}")

                self.gate_idx[segid] = GateIndices(
                    od1=od1_idx[0],
                    od2=od2_idx[0],
                    ne1=ne1_idx[0]
                )
                found_chains.append(segid)
                logger.debug(f"Successfully mapped indices for segid {segid}")
            except Exception as e:
                logger.warning(f"Could not map DW-gate atom indices for segid {segid}: {e}")
                # Remove segid if indices can't be found
                if segid in self.valid_segids:
                    self.valid_segids.remove(segid)

        if not self.valid_segids:
            raise RuntimeError("No valid DW-gates atoms could be mapped after residue identification.")
        logger.info(f"DW-gate atom indices identified for chains: {self.valid_segids}")


    def collect_distances(self):
        """Collects minimum distance between Asp/Glu carboxylates and Trp NE1 for all frames."""
        if not self.valid_segids:
             logger.error("No valid chains to collect distances for.")
             return # Or raise error?

        logger.info("Collecting DW-gate distances for all frames...")
        records = []
        # Use the pre-calculated time axis
        for i, ts in enumerate(tqdm(self.universe.trajectory, desc="DW-gate Distances", total=len(self.time_axis))):
            t_ns = self.time_axis[i]
            pos = ts.positions
            box = ts.dimensions # Needed for distance_array if PBC are relevant

            for segid in self.valid_segids:
                try:
                    idx = self.gate_idx[segid]
                    # Ensure indices are valid for current frame (should be unless topology changes)
                    # Select carboxylate oxygen positions
                    carbox_pos = np.vstack([pos[idx.od1], pos[idx.od2]])
                    # Calculate distance from NE1 to both oxygens
                    dist = distance_array(pos[idx.ne1][None, :], carbox_pos, box=box).min()

                    records.append({
                        "time_ns": t_ns,
                        "frame": i, # Use frame index i directly
                        "chain": segid[-1], # Use simple chain ID (A, B, C, D)
                        "distance": dist,
                        # Initial state assignment (will be refined by KDE/defaults later if needed)
                        "state": CLOSED_STATE if dist <= self.distance_threshold else OPEN_STATE
                    })
                except IndexError:
                     logger.warning(f"Index error accessing atoms for segid {segid} at frame {i}. Skipping.")
                     continue # Skip this chain for this frame
                except KeyError:
                     logger.warning(f"KeyError accessing indices for segid {segid} at frame {i}. Skipping.")
                     continue # Skip this chain for this frame


        if not records:
             logger.error("No distance records were collected.")
             self.df_raw = pd.DataFrame(columns=["time_ns", "frame", "chain", "distance", "state"])
        else:
            self.df_raw = pd.DataFrame(records)
            logger.info(f"Collected {len(self.df_raw)} distance measurements across {len(self.valid_segids)} chains.")

        # Decide whether to calculate or use default reference distances
        if self.auto_detect_refs:
            self.closed_ref_dist, self.open_ref_dist = self.calculate_reference_distances()
        else:
            self.closed_ref_dist = self.default_closed_ref_dist
            self.open_ref_dist = self.default_open_ref_dist
            logger.info(f"Using default reference distances: Closed={self.closed_ref_dist:.2f} Å, Open={self.open_ref_dist:.2f} Å")


    def calculate_reference_distances(self) -> Tuple[float, float]:
        """
        Calculates reference distances from data using KDE and KMeans clustering.
        Updates self.kde_analysis_results and self.plot_paths.
        Returns the calculated (closed_ref_dist, open_ref_dist).
        Uses defaults if calculation fails.
        """
        if self.df_raw is None or self.df_raw.empty:
            logger.warning("Raw distance data not available. Cannot calculate reference distances. Using defaults.")
            return self.default_closed_ref_dist, self.default_open_ref_dist

        import scipy.stats as stats # Keep import local if only used here

        logger.info("Calculating reference distances from data using KDE...")
        results = { 'all_chains': {}, 'per_chain': {} }
        all_peaks = []
        all_dist = self.df_raw.distance.values # Get all distances early for range calculation

        # Determine common x-range for all subplots
        if len(all_dist) > 0:
             x_min_all = np.floor(all_dist.min() - 0.5)
             x_max_all = np.ceil(all_dist.max() + 0.5)
        else:
             x_min_all, x_max_all = 0, 5 # Default range if no data


        # --- Per-chain KDE --- 
        # Create figure *before* loop to manage legends later
        fig = plt.figure(figsize=(12, 9)) # Adjusted figsize for bottom legend

        # Plot 1: Combined distribution
        ax_combined = plt.subplot(3, 2, (1, 2))
        ax_combined.grid(False) # Hide grid

        chains_analyzed = 0
        for i, chain in enumerate(sorted(self.df_raw.chain.unique())):
            dist_arr = self.df_raw.query("chain == @chain").distance.values
            if len(dist_arr) > 10: # Need enough data for KDE
                try:
                    kde = stats.gaussian_kde(dist_arr, bw_method='scott') # Or Silverman?
                    # Use common x-range for evaluation, but keep points reasonable
                    x_kde = np.linspace(max(x_min_all, dist_arr.min() - 0.5), 
                                        min(x_max_all, dist_arr.max() + 0.5), 500)
                    y_kde = kde(x_kde)
                    # Adjust find_peaks parameters if needed
                    peaks, props = find_peaks(y_kde, height=0.05*y_kde.max(), distance=30, prominence=0.05) # Keep prominence
                    peak_distances = x_kde[peaks]
                    peak_heights = y_kde[peaks]
                    all_peaks.extend(peak_distances) # Collect peaks from all chains

                    # Store results
                    results['per_chain'][chain] = {
                        'x_kde': x_kde, 'y_kde': y_kde,
                        'peak_distances': peak_distances, 'peak_heights': peak_heights
                    }
                    logger.debug(f"Chain {chain}: Found {len(peak_distances)} KDE peaks at {np.round(peak_distances, 2)}")

                    # Plotting per-chain KDE (max 4 chains shown)
                    if chains_analyzed < 4:
                        ax_chain = plt.subplot(3, 2, 3 + chains_analyzed)
                        ax_chain.plot(x_kde, y_kde, label=f'_Chain {chain} KDE')
                        ax_chain.scatter(peak_distances, peak_heights, color='red', marker='x', s=50, label='_Peaks')
                        
                        # Add text annotations for peak values
                        for pd, ph in zip(peak_distances, peak_heights):
                            ax_chain.text(pd + 0.05, ph, f'{pd:.1f}Å', fontsize=9, va='bottom', ha='left')
                            
                        ax_chain.set_title(f'Chain {chain}') # NEW Simplified Title
                        ax_chain.set_xlabel('Distance (Å)')
                        ax_chain.set_ylabel('Density')
                        ax_chain.grid(False) # Hide grid
                        ax_chain.set_xlim(x_min_all, x_max_all) # Set common x-axis range
                    chains_analyzed += 1

                except Exception as e:
                    logger.warning(f"KDE calculation failed for chain {chain}: {e}")
            else:
                logger.warning(f"Skipping KDE for chain {chain}: Insufficient data points ({len(dist_arr)}).")
                # Still create subplot placeholder if needed?
                if chains_analyzed < 4:
                     ax_chain = plt.subplot(3, 2, 3 + chains_analyzed)
                     ax_chain.text(0.5, 0.5, f"Chain {chain}\nNo KDE (N={len(dist_arr)})", 
                                    ha='center', va='center', transform=ax_chain.transAxes)
                     ax_chain.set_title(f'Chain {chain}') # NEW Simplified Title
                     ax_chain.set_xlabel('Distance (Å)')
                     ax_chain.set_ylabel('Density')
                     ax_chain.grid(False)
                     ax_chain.set_xlim(x_min_all, x_max_all)
                chains_analyzed += 1


        # --- Combined KDE and Histogram (on the first subplot) ---
        combined_handles, combined_labels = [], [] # For manual legend creation
        if len(all_dist) > 10:
            try:
                # Histogram
                hist_vals, hist_bins = np.histogram(all_dist, bins=50, density=True)
                bin_centers = (hist_bins[:-1] + hist_bins[1:]) / 2
                # Store handle for legend
                bar_container = ax_combined.bar(bin_centers, hist_vals, width=np.diff(hist_bins)[0]*0.9, alpha=0.5, label='Combined Histogram')
                combined_handles.append(bar_container[0]) # Get patch handle
                combined_labels.append('Combined Histogram')

                # Combined KDE
                kde_all = stats.gaussian_kde(all_dist, bw_method='scott')
                x_all = np.linspace(all_dist.min() - 0.5, all_dist.max() + 0.5, 1000)
                y_all = kde_all(x_all)
                peaks_all, _ = find_peaks(y_all, height=0.05*y_all.max(), distance=50, prominence=0.05) # Wider distance for combined?
                peak_distances_all = x_all[peaks_all]
                peak_heights_all = y_all[peaks_all]

                # Store handle for legend
                line_kde, = ax_combined.plot(x_all, y_all, label='Combined KDE', color='k', lw=1.5)
                combined_handles.append(line_kde)
                combined_labels.append('Combined KDE')
                
                # Store handle for legend
                scatter_peaks = ax_combined.scatter(peak_distances_all, peak_heights_all, color='red', marker='x', s=80, label='Combined Peaks')
                combined_handles.append(scatter_peaks)
                combined_labels.append('Combined Peaks')

                results['all_chains'] = {
                    'x_kde': x_all, 'y_kde': y_all,
                    'hist_bins': bin_centers, 'hist_values': hist_vals, # Store hist data
                    'peak_distances': peak_distances_all, 'peak_heights': peak_heights_all
                }
                self.kde_analysis_results = results # Store results object

            except Exception as e:
                 logger.error(f"Combined KDE/Histogram calculation failed: {e}")
                 self.kde_analysis_results = None # Ensure it's None if failed
        else:
            logger.warning("Insufficient combined data points for KDE/Histogram.")
            self.kde_analysis_results = None


        # --- Determine Reference Distances using KMeans on collected peaks ---
        closed_ref = self.default_closed_ref_dist
        open_ref = self.default_open_ref_dist
        if all_peaks:
            unique_peaks = sorted(list(set(all_peaks))) # Use unique peaks
            if len(unique_peaks) >= 2:
                try:
                    # Reshape for KMeans
                    kp = np.array(unique_peaks).reshape(-1, 1)
                    # Fit KMeans with 2 clusters
                    kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(kp) # Explicit n_init
                    centers = sorted(kmeans.cluster_centers_.flatten())
                    closed_ref, open_ref = centers[0], centers[1]
                    logger.info(f"KDE/KMeans derived references: Closed={closed_ref:.2f} Å, Open={open_ref:.2f} Å")

                    # Add KMeans centers to the combined plot - Store handle for legend
                    scatter_kmeans = ax_combined.scatter(centers, [ax_combined.get_ylim()[1]*0.95]*2, color='blue', marker='D', s=100, label='KMeans Centers', zorder=5)
                    combined_handles.append(scatter_kmeans)
                    combined_labels.append('KMeans Centers')

                except Exception as e:
                    logger.warning(f"KMeans clustering on peaks failed: {e}. Using defaults.")
            elif len(unique_peaks) == 1:
                logger.warning(f"Only one unique KDE peak found ({unique_peaks[0]:.2f} Å). Insufficient for clustering. Using defaults.")
            else: # len is 0
                 logger.warning("No KDE peaks found across any chain. Using defaults.")
        else:
            logger.warning("No KDE peaks collected from per-chain analysis. Using defaults.")


        # --- Finalize Combined Plot --- 
        if self.kde_analysis_results: # Only add lines if combined KDE ran
            # Add threshold line - Store handle
             line_thresh = ax_combined.axvline(self.distance_threshold, color='grey', ls=':', lw=1.5, label=f'Threshold ({self.distance_threshold:.1f} Å)')
             combined_handles.append(line_thresh)
             combined_labels.append(f'Threshold ({self.distance_threshold:.1f} Å)')
             
             # Add reference distance lines (final values, either calculated or default) - Store handles
             line_closed = ax_combined.axvline(closed_ref, color='darkred', ls='-', lw=2, label=f'Closed Ref ({closed_ref:.2f} Å)')
             line_open = ax_combined.axvline(open_ref, color='darkgreen', ls='-', lw=2, label=f'Open Ref ({open_ref:.2f} Å)')
             combined_handles.append(line_closed)
             combined_labels.append(f'Closed Ref ({closed_ref:.2f} Å)')
             combined_handles.append(line_open)
             combined_labels.append(f'Open Ref ({open_ref:.2f} Å)')
             
             # ax_combined.legend(fontsize='small') # REMOVED Legend here
             ax_combined.set_title('Combined & Per-Chain DW-Gate Distance Distributions (KDE)')
             ax_combined.set_xlabel('Distance (Å)')
             ax_combined.set_ylabel('Density')
             ax_combined.grid(False) # Ensure grid is off
             ax_combined.set_xlim(x_min_all, x_max_all) # Set common x-limit here too
        else:
             ax_combined.text(0.5, 0.5, 'KDE Analysis Failed or Insufficient Data',
                              ha='center', va='center', transform=ax_combined.transAxes, color='red')


        # --- Create Combined Legend Below Plot ---
        from matplotlib.lines import Line2D
        # Handles/Labels for per-chain elements
        per_chain_handles = [
            Line2D([0], [0], color='tab:blue', lw=2), # Representing the per-chain KDE line
            Line2D([0], [0], marker='x', color='red', markersize=8, linestyle='None') # Representing peaks
        ]
        per_chain_labels = ['Per-Chain KDE', 'Per-Chain Peaks']
        
        # Combine legends
        all_handles = combined_handles + per_chain_handles
        all_labels = combined_labels + per_chain_labels
        
        # Place legend below the figure
        fig.legend(all_handles, all_labels, loc='lower center', 
                   bbox_to_anchor=(0.5, -0.05), # Adjust vertical position below axes
                   ncol=4, # Adjust number of columns as needed
                   fontsize='medium')


        plt.tight_layout(rect=[0, 0.08, 1, 0.95]) # Adjust rect for suptitle and bottom legend space
        # plt.suptitle("DW-Gate Reference Distance Analysis", fontsize=16) # Title is now on subplot

        # Save the plot
        plot_filename = "dw_gate_distance_distribution.png"
        plot_path = os.path.join(self.output_dir, plot_filename)
        save_plot(fig, plot_path) # Save the modified figure
        self.plot_paths['distance_distribution'] = os.path.join("dw_gate_analysis", plot_filename) # Relative path for report

        return closed_ref, open_ref


    def apply_debouncing(self):
        """Applies Run-Length Encoding (RLE) debouncing to the raw states."""
        if self.df_raw is None or self.df_raw.empty:
            logger.error("Cannot apply debouncing: Raw data not available.")
            return

        logger.info(f"Applying RLE debouncing with tolerance={self.tolerance_frames} frames...")
        self.df = self.df_raw.copy() # Start with raw data

        # Map states to numeric for easier processing
        state_map = {CLOSED_STATE: 0, OPEN_STATE: 1}
        inverse_state_map = {v: k for k, v in state_map.items()}

        debounced_states_all = [] # Store debounced states for all chains

        # Process each chain independently
        for chain in sorted(self.df['chain'].unique()):
            chain_indices = self.df['chain'] == chain
            if not chain_indices.any():
                logger.warning(f"No data found for chain {chain} during debouncing.")
                continue

            # Get original states for this chain
            original_numeric_states = self.df.loc[chain_indices, 'state'].map(state_map).tolist()

            # Apply RLE debouncing
            debounced_numeric_states = self._debounce_rle(original_numeric_states)

            # Store the debounced states (still numeric)
            # We need to put them back in the correct order according to the original DataFrame index
            # Create a temporary Series to map back easily
            debounced_series = pd.Series(debounced_numeric_states, index=self.df[chain_indices].index)
            debounced_states_all.append(debounced_series)


        # Concatenate all debounced states and update the DataFrame
        if debounced_states_all:
             all_debounced = pd.concat(debounced_states_all).sort_index()
             # Map back to string representation ('open'/'closed')
             self.df['state'] = all_debounced.map(inverse_state_map)
             logger.info("Debouncing complete.")
        else:
             logger.error("Debouncing failed: No states were processed.")
             # Keep self.df as the raw copy if debouncing fails completely
             self.df = self.df_raw.copy()


    def _debounce_rle(self, states: List[int]) -> List[int]:
        """
        Internal RLE debouncing logic. Merges runs shorter than tolerance
        into the longer neighboring run.

        Args:
            states (List[int]): List of numeric states (e.g., 0 for closed, 1 for open).

        Returns:
            List[int]: Debounced list of numeric states.
        """
        if not states:
            return []

        tol = self.tolerance_frames
        if tol <= 1: # No debouncing if tolerance is 1 or less
             return states

        s = list(states) # Work on a copy
        n = len(s)
        iteration = 0
        max_iterations = n # Safety break to prevent infinite loops

        while iteration < max_iterations:
            iteration += 1
            runs = []
            if not s: break # Should not happen if initial states exist

            # 1. Identify runs
            current_val = s[0]
            start_idx = 0
            for i in range(1, n):
                if s[i] != current_val:
                    runs.append({'value': current_val, 'start': start_idx, 'length': i - start_idx})
                    current_val = s[i]
                    start_idx = i
            runs.append({'value': current_val, 'start': start_idx, 'length': n - start_idx})

            # 2. Find short runs to merge
            merges_made = 0
            indices_to_merge = [i for i, run in enumerate(runs) if run['length'] < tol]

            if not indices_to_merge:
                break # No more short runs found, debouncing complete

            # 3. Merge short runs
            # Iterate backwards to avoid index issues after merging
            for idx in sorted(indices_to_merge, reverse=True):
                current_run = runs[idx]
                val_to_merge = current_run['value']
                start = current_run['start']
                length = current_run['length']

                # Determine the value to merge into
                new_val = -1 # Placeholder for invalid state
                if idx == 0: # First run is short
                    if len(runs) > 1:
                        new_val = runs[1]['value']
                    # else: # Only one run and it's short - keep original value? Or error? Keep original.
                    #    new_val = val_to_merge # No change possible
                elif idx == len(runs) - 1: # Last run is short
                    new_val = runs[idx - 1]['value']
                else: # Middle run is short
                    # Merge into the longer neighbor
                    prev_run = runs[idx - 1]
                    next_run = runs[idx + 1]
                    if prev_run['length'] >= next_run['length']:
                        new_val = prev_run['value']
                    else:
                        new_val = next_run['value']

                # Apply the merge in the state list 's'
                if new_val != -1 and new_val != val_to_merge: # Only merge if neighbor exists and is different
                    for j in range(start, start + length):
                        s[j] = new_val
                    merges_made += 1
                # If new_val is same as original, or no neighbor, no change needed here

            if merges_made == 0:
                 # This can happen if short runs are surrounded by runs of the same value
                 # or are at the ends with no different neighbor.
                 break

        if iteration == max_iterations:
             logger.warning("RLE debouncing reached max iterations. Results might be incomplete.")

        return s

    def build_events(self):
        """Builds continuous events from debounced states, handling potential gaps."""
        if self.df is None or self.df.empty:
            logger.error("Cannot build events: Debounced data frame is missing or empty.")
            return

        logger.info("Building events from debounced states...")
        rows = []
        max_frame_overall = self.df['frame'].max() # Get max frame from data

        # Sort by chain and time/frame is crucial
        df_sorted = self.df.sort_values(['chain', 'frame']).reset_index()

        for ch, sub_df in df_sorted.groupby('chain'):
            if sub_df.empty: continue

            # Detect consecutive blocks of the same state
            sub_df = sub_df.copy() # Avoid SettingWithCopyWarning
            sub_df['block'] = (sub_df['state'] != sub_df['state'].shift()).cumsum()

            chain_events = []
            for _, group in sub_df.groupby('block'):
                start_frame = group['frame'].min()
                end_frame = group['frame'].max()
                frame_count = end_frame - start_frame + 1
                start_time = group['time_ns'].min()
                # End time is start time of *next* block or end of sim if last block
                # Approximate duration using frame count * dt_ns for now
                duration_ns = frame_count * self.dt_ns
                end_time = start_time + duration_ns # Approximate end time

                chain_events.append({
                    'chain': ch,
                    'state': group['state'].iloc[0],
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'frame_count': frame_count,
                    'duration_ns': duration_ns,
                    'start_ns': start_time,
                    'end_ns': end_time # Store approximate end time
                })

            if not chain_events:
                logger.warning(f"No events generated for chain {ch}.")
                continue

            # --- Verify and Fix Coverage for this chain ---
            chain_events_df = pd.DataFrame(chain_events).sort_values('start_frame')
            fixed_events_for_chain = self._verify_and_fix_coverage_per_chain(
                chain_events_df, ch, max_frame_overall
            )
            rows.extend(fixed_events_for_chain)

        if not rows:
            logger.error("Failed to build any events for any chain.")
            self.events_df = pd.DataFrame(columns=[
                'chain', 'state', 'start_frame', 'end_frame',
                'frame_count', 'duration_ns', 'start_ns', 'end_ns'
            ])
        else:
            self.events_df = pd.DataFrame(rows).sort_values(['chain', 'start_frame']).reset_index(drop=True)
            logger.info(f"Built {len(self.events_df)} events across all chains after coverage check.")
            # Final verification (optional)
            self.verify_continuous_coverage() # Logs warnings if issues persist

        # Save events to CSV
        if self.events_df is not None and not self.events_df.empty:
            # --- Add Logging ---
            logger.debug(f"Final events_df head before saving:\n{self.events_df.head().to_string()}")
            logger.debug(f"Final events_df shape: {self.events_df.shape}")
            # --- End Logging ---
            csv_path = os.path.join(self.output_dir, "dw_gate_events.csv")
            try:
                self.events_df.to_csv(csv_path, index=False, float_format="%.3f")
                logger.info(f"Saved DW Gate events data to {csv_path}")
            except Exception as e:
                logger.error(f"Failed to save DW Gate events CSV: {e}")
        else:
             logger.warning("events_df is None or empty; cannot save events CSV.")


    def _verify_and_fix_coverage_per_chain(self, chain_events_df: pd.DataFrame, chain_id: str, max_frame_overall: int) -> List[Dict]:
        """
        Checks for gaps in event coverage for a single chain and fills them
        by extending the neighboring event or using raw data if necessary.
        Returns a list of event dictionaries (original + filled gaps).
        """
        if chain_events_df.empty:
            return []

        events = chain_events_df.to_dict('records')
        fixed_events = []
        last_end_frame = -1

        # 1. Check for gap before the first event
        first_event = events[0]
        if first_event['start_frame'] > 0:
            gap_start_frame = 0
            gap_end_frame = first_event['start_frame'] - 1
            gap_state = first_event['state'] # Assume gap takes state of first event
            gap_count = gap_end_frame - gap_start_frame + 1
            # Try to get actual start/end times from raw data for the gap
            gap_df = self.df_raw[(self.df_raw['chain'] == chain_id) &
                                 (self.df_raw['frame'] >= gap_start_frame) &
                                 (self.df_raw['frame'] <= gap_end_frame)]
            start_ns = gap_df['time_ns'].min() if not gap_df.empty else gap_start_frame * self.dt_ns
            duration_ns = gap_count * self.dt_ns
            end_ns = start_ns + duration_ns

            logger.warning(f"Chain {chain_id}: Found gap at start (Frames 0-{gap_end_frame}). Filling with state '{gap_state}'.")
            fixed_events.append({
                'chain': chain_id, 'state': gap_state,
                'start_frame': gap_start_frame, 'end_frame': gap_end_frame,
                'frame_count': gap_count, 'duration_ns': duration_ns,
                'start_ns': start_ns, 'end_ns': end_ns
            })
            last_end_frame = gap_end_frame


        # 2. Process existing events and check for gaps between them
        for i, event in enumerate(events):
            # Check for gap *before* this event
            if event['start_frame'] > last_end_frame + 1:
                gap_start_frame = last_end_frame + 1
                gap_end_frame = event['start_frame'] - 1
                # Assign state based on the *previous* event's state
                gap_state = fixed_events[-1]['state'] if fixed_events else event['state'] # Fallback to current if first
                gap_count = gap_end_frame - gap_start_frame + 1
                # Try to get actual start/end times
                gap_df = self.df_raw[(self.df_raw['chain'] == chain_id) &
                                     (self.df_raw['frame'] >= gap_start_frame) &
                                     (self.df_raw['frame'] <= gap_end_frame)]
                start_ns = gap_df['time_ns'].min() if not gap_df.empty else gap_start_frame * self.dt_ns
                duration_ns = gap_count * self.dt_ns
                end_ns = start_ns + duration_ns

                logger.warning(f"Chain {chain_id}: Found mid gap (Frames {gap_start_frame}-{gap_end_frame}). Filling with state '{gap_state}'.")
                fixed_events.append({
                    'chain': chain_id, 'state': gap_state,
                    'start_frame': gap_start_frame, 'end_frame': gap_end_frame,
                    'frame_count': gap_count, 'duration_ns': duration_ns,
                    'start_ns': start_ns, 'end_ns': end_ns
                })

            # Add the current event itself
            fixed_events.append(event)
            last_end_frame = event['end_frame']


        # 3. Check for gap after the last event
        if last_end_frame < max_frame_overall:
            gap_start_frame = last_end_frame + 1
            gap_end_frame = max_frame_overall
            gap_state = fixed_events[-1]['state'] # Assume state of last event continues
            gap_count = gap_end_frame - gap_start_frame + 1
             # Try to get actual start/end times
            gap_df = self.df_raw[(self.df_raw['chain'] == chain_id) &
                                 (self.df_raw['frame'] >= gap_start_frame) &
                                 (self.df_raw['frame'] <= gap_end_frame)]
            start_ns = gap_df['time_ns'].min() if not gap_df.empty else gap_start_frame * self.dt_ns
            duration_ns = gap_count * self.dt_ns
            end_ns = start_ns + duration_ns

            logger.warning(f"Chain {chain_id}: Found gap at end (Frames {gap_start_frame}-{gap_end_frame}). Filling with state '{gap_state}'.")
            fixed_events.append({
                'chain': chain_id, 'state': gap_state,
                'start_frame': gap_start_frame, 'end_frame': gap_end_frame,
                'frame_count': gap_count, 'duration_ns': duration_ns,
                'start_ns': start_ns, 'end_ns': end_ns
            })

        return fixed_events


    def verify_continuous_coverage(self):
        """Verifies that the final events cover the full trajectory duration for each chain."""
        if self.events_df is None or self.events_df.empty:
            return True # No events to verify

        maxf = self.df['frame'].max() # Use max frame from debounced data
        is_continuous = True
        for ch, ev in self.events_df.groupby('chain'):
            e = ev.sort_values('start_frame')
            if e.empty: continue

            # Check start
            if e['start_frame'].iloc[0] > 0:
                logger.warning(f"Coverage Issue [Chain {ch}]: Events do not start at frame 0 (starts at {e['start_frame'].iloc[0]}).")
                is_continuous = False

            # Check gaps between events
            for i in range(len(e) - 1):
                end_frame_current = e['end_frame'].iloc[i]
                start_frame_next = e['start_frame'].iloc[i+1]
                if end_frame_current + 1 < start_frame_next:
                    logger.warning(f"Coverage Issue [Chain {ch}]: Gap found between frame {end_frame_current} and {start_frame_next}.")
                    is_continuous = False

            # Check end
            if e['end_frame'].iloc[-1] < maxf:
                logger.warning(f"Coverage Issue [Chain {ch}]: Events do not end at max frame {maxf} (ends at {e['end_frame'].iloc[-1]}).")
                is_continuous = False

        if is_continuous:
            logger.info("Event coverage verified to be continuous for all chains.")
        else:
            logger.error("Event coverage issues detected AFTER gap filling. Check logic.")
        return is_continuous

    def run_statistical_analysis(self):
        """
        Performs statistical analysis on the events data.
        Calculates summary stats, open/closed probabilities, and runs significance tests.
        Stores results (DataFrames, dicts) in self.stats_results.
        """
        if self.events_df is None or self.events_df.empty:
            logger.error("Cannot run statistics: Events data frame not available.")
            self.stats_results = {'Error': 'No events data'}
            return

        logger.info("Running statistical analysis on DW-gate events...")
        # Clear previous results
        self.stats_results = {}
        raw_stats_collector = {} # Temporary dict to hold intermediate results

        # --- 1. Summary Statistics Table ---
        try:
            stats_df = self.events_df.groupby(['chain', 'state'])['duration_ns'].agg(
                Count='size',
                Mean_ns='mean',
                Std_Dev_ns='std',
                Median_ns='median',
                Min_ns='min',
                Max_ns='max',
                Total_Duration_ns='sum' # Also get total time spent in state
            ).reset_index()
            # Fill NaN for Std_Dev if only one event exists
            stats_df['Std_Dev_ns'] = stats_df['Std_Dev_ns'].fillna(0)
            raw_stats_collector['summary_stats_df'] = stats_df

            # --- Add Logging --- 
            if stats_df is not None and not stats_df.empty:
                 logger.debug(f"Calculated summary_stats_df head:\n{stats_df.head().to_string()}")
                 logger.debug(f"summary_stats_df shape: {stats_df.shape}")
            else:
                 logger.warning("summary_stats_df is None or empty after calculation.")
            # --- End Logging --- 

            # REMOVED HTML Generation
            # html_summary_table = stats_df.to_html(...)
            logger.info("Calculated summary statistics.")
        except Exception as e:
            logger.error(f"Failed to calculate summary statistics: {e}")
            # Store error indicator if needed, but avoid storing HTML error message
            raw_stats_collector['summary_stats_error'] = str(e)
            # self.stats_results['summary_table_html'] = "<p>Error calculating summary statistics.</p>"


        # --- 2. Open/Closed Probability and Chi-squared Test ---
        prob_df = None # Initialize
        time_sum_df = None # Initialize
        try:
            # Use the pre-calculated Total_Duration_ns from stats_df if available
            if 'summary_stats_df' in raw_stats_collector:
                 summary_df = raw_stats_collector['summary_stats_df']
                 # Need to handle potential multi-index if groupby was different
                 if all(col in summary_df.columns for col in ['chain', 'state', 'Total_Duration_ns']):
                     time_sum_df = summary_df.pivot(index='chain', columns='state', values='Total_Duration_ns').fillna(0)
                 else:
                     logger.warning("Summary stats DF missing expected columns for pivot. Recalculating time sums.")
                     time_sum_df = None # Force recalculation
            
            if time_sum_df is None:
                logger.debug("Recalculating time sums for probability.")
                time_sum_df = self.events_df.groupby(['chain', 'state'])['duration_ns'].sum().unstack(fill_value=0)

            # Ensure both 'open' and 'closed' columns exist
            if OPEN_STATE not in time_sum_df.columns: time_sum_df[OPEN_STATE] = 0.0
            if CLOSED_STATE not in time_sum_df.columns: time_sum_df[CLOSED_STATE] = 0.0
            # Reorder columns for consistency
            time_sum_df = time_sum_df[[CLOSED_STATE, OPEN_STATE]]

            total_time_per_chain = time_sum_df.sum(axis=1)
            # Avoid division by zero if a chain had no events
            prob_df = time_sum_df.divide(total_time_per_chain, axis=0).fillna(0)
            prob_df['total_time_ns'] = total_time_per_chain # Add total time
            prob_df.reset_index(inplace=True)
            raw_stats_collector['probability_df'] = prob_df # Store the DataFrame for the report

            # Melted version for potential internal use or easy plotting access
            # prob_melted_df = prob_df.melt('chain', var_name='state', value_name='probability')
            # raw_stats_collector['probability_melted_df'] = prob_melted_df

            # Chi-squared test on the time sums (contingency table)
            # Ensure there are at least 2 rows (chains) and 2 columns (states) with non-zero variance?
            if time_sum_df.shape[0] >= 2 and time_sum_df.shape[1] >= 2 and time_sum_df.any(axis=None):
                 chi2_stat, p_chi2, dof, expected = chi2_contingency(time_sum_df.values)
                 raw_stats_collector['chi2_test'] = {'statistic': chi2_stat, 'p_value': p_chi2, 'dof': dof}
                 logger.info(f"Chi-squared test on state durations across chains: chi2={chi2_stat:.3f}, p={p_chi2:.4g}")
            else:
                 logger.warning("Skipping Chi-squared test: Insufficient data dimensions or variance.")
                 raw_stats_collector['chi2_test'] = {'statistic': np.nan, 'p_value': np.nan, 'dof': np.nan}

            # REMOVED HTML Generation
            # html_prob_table = prob_df.to_html(...)
            # self.stats_results['probability_table_html'] = html_prob_table
            # self.stats_results['chi2_results_html'] = f"<p>... {chi2_stat:.3f} ...</p>"

        except Exception as e:
            logger.error(f"Failed to calculate probabilities or run Chi-squared test: {e}")
            raw_stats_collector['probability_error'] = str(e)
            raw_stats_collector['chi2_test'] = {'statistic': np.nan, 'p_value': np.nan, 'dof': np.nan, 'error': str(e)}
            # self.stats_results['probability_table_html'] = "<p>Error calculating probabilities.</p>"
            # self.stats_results['chi2_results_html'] = "<p>Chi-squared test failed.</p>"

        # --- 3. Kruskal-Wallis Test (Compare Open Durations Across Chains) ---
        try:
            chains = sorted(self.events_df['chain'].unique())
            open_duration_lists = [
                self.events_df.loc[(self.events_df['state'] == OPEN_STATE) & (self.events_df['chain'] == c), 'duration_ns'].values
                for c in chains
            ]
            # Filter out chains with no open events or only one event
            valid_open_lists = [lst for lst in open_duration_lists if len(lst) > 1]

            if len(valid_open_lists) >= 2: # Need at least two groups with >1 data point for Kruskal-Wallis
                h_stat, p_kruskal = kruskal(*valid_open_lists)
                raw_stats_collector['kruskal_test'] = {'statistic': h_stat, 'p_value': p_kruskal}
                logger.info(f"Kruskal-Wallis test on OPEN durations across chains: H={h_stat:.3f}, p={p_kruskal:.4g}")
                # REMOVED HTML
                # self.stats_results['kruskal_results_html'] = f"<p>... H-statistic={h_stat:.3f} ...</p>"
            else:
                logger.warning("Skipping Kruskal-Wallis test: Need open events (>1) in at least two chains.")
                raw_stats_collector['kruskal_test'] = {'statistic': np.nan, 'p_value': np.nan, 'skipped': True}
                # self.stats_results['kruskal_results_html'] = "<p>Kruskal-Wallis test skipped (insufficient data).</p>"
        except Exception as e:
            logger.error(f"Failed to run Kruskal-Wallis test: {e}")
            raw_stats_collector['kruskal_test'] = {'statistic': np.nan, 'p_value': np.nan, 'error': str(e)}
            # self.stats_results['kruskal_results_html'] = "<p>Kruskal-Wallis test failed.</p>"


        # --- 4. Pairwise Mann-Whitney U Tests (Compare Open Durations Between Chain Pairs) ---
        comparisons = []
        mw_df = None
        try:
            chains = sorted(self.events_df['chain'].unique())
            if len(chains) >= 2:
                for chain_a, chain_b in combinations(chains, 2):
                    durations_a = self.events_df.loc[(self.events_df['state'] == OPEN_STATE) & (self.events_df['chain'] == chain_a), 'duration_ns']
                    durations_b = self.events_df.loc[(self.events_df['state'] == OPEN_STATE) & (self.events_df['chain'] == chain_b), 'duration_ns']

                    # Check if both chains have sufficient data (e.g., >= 1 event?)
                    if not durations_a.empty and not durations_b.empty:
                         # Perform Mann-Whitney U test
                         u_stat, p_mannwhitney = mannwhitneyu(durations_a, durations_b, alternative='two-sided') # Use two-sided
                         comparisons.append({
                             'Chain 1': chain_a,
                             'Chain 2': chain_b,
                             'U-statistic': u_stat,
                             'p-value': p_mannwhitney
                         })
                    # else: Skip pair if one chain has no open events

            if comparisons:
                mw_df = pd.DataFrame(comparisons)
                # Apply Bonferroni correction (or other method like Benjamini-Hochberg?)
                num_comparisons = len(mw_df)
                mw_df['p-value (Bonferroni)'] = (mw_df['p-value'] * num_comparisons).clip(upper=1.0) # Corrected p-value
                raw_stats_collector['mannwhitney_tests_df'] = mw_df # Store the DataFrame

                logger.info(f"Performed {num_comparisons} pairwise Mann-Whitney U tests on open durations.")

                # REMOVED HTML GENERATION
                # html_mw_table = mw_df.to_html(...)
                # self.stats_results['mannwhitney_table_html'] = html_mw_table
            else:
                 logger.warning("Skipping pairwise Mann-Whitney U tests: Need at least two chains with open events.")
                 raw_stats_collector['mannwhitney_tests_df'] = None # Indicate skipped
                 # self.stats_results['mannwhitney_table_html'] = "<p>Pairwise Mann-Whitney U tests skipped (insufficient data).</p>"

        except Exception as e:
            logger.error(f"Failed to run Mann-Whitney U tests: {e}")
            raw_stats_collector['mannwhitney_tests_error'] = str(e)
            raw_stats_collector['mannwhitney_tests_df'] = None # Indicate failure
            # self.stats_results['mannwhitney_table_html'] = "<p>Pairwise Mann-Whitney U tests failed.</p>"

        # Store the collected raw stats (DataFrames, dicts) in the main results dict
        self.stats_results = raw_stats_collector
        logger.info("Statistical analysis complete. Results stored as DataFrames/Dicts.")


    # --- Plotting Methods (Adapted to save plots and return paths) ---

    def plot_distance_vs_state(self):
        """Plots raw distance vs. debounced state for each chain."""
        if self.df is None or self.df.empty:
            logger.warning("Skipping distance vs state plot: Debounced data not available.")
            return

        sns.set_theme(style='ticks', context='notebook') # Use notebook context for potentially smaller plots
        chains = sorted(self.df['chain'].unique())
        n_chains = len(chains)
        if n_chains == 0: return

        # Create palettes
        pastel_palette = sns.color_palette("pastel", n_colors=n_chains)
        bright_palette = sns.color_palette("bright", n_colors=n_chains)
        chain_color_map_pastel = {ch: col for ch, col in zip(chains, pastel_palette)}
        chain_color_map_bright = {ch: col for ch, col in zip(chains, bright_palette)}

        fig, axes = plt.subplots(n_chains, 1, figsize=(12, 2.5 * n_chains), sharex=True)
        if n_chains == 1: axes = [axes] # Ensure axes is always iterable

        # Find global min/max for consistent y-scale (optional, but good practice)
        y_min = self.df['distance'].min() * 0.9
        y_max = self.df['distance'].max() * 1.1
        if np.isnan(y_min): y_min = 0
        if np.isnan(y_max): y_max = self.distance_threshold * 2 # Default if NaN

        # Reference heights for state indicators (use calculated/default refs)
        state_y = {CLOSED_STATE: self.closed_ref_dist, OPEN_STATE: self.open_ref_dist}

        for i, ch in enumerate(chains):
            ax = axes[i]
            sub = self.df.query("chain == @ch").sort_values('time_ns') # Ensure sorted by time

            if sub.empty:
                 ax.text(0.5, 0.5, f"No data for Chain {ch}", ha='center', va='center', transform=ax.transAxes)
                 continue

            # Plot distance trace with pastel color
            ax.plot(sub['time_ns'], sub['distance'],
                    color=chain_color_map_pastel[ch], alpha=0.8, linewidth=1.0) # Thinner line

            # Plot debounced state bars using event data for efficiency
            chain_events = self.events_df[self.events_df['chain'] == ch]
            for _, event in chain_events.iterrows():
                y = state_y[event['state']]
                # Use event start/end times
                ax.plot([event['start_ns'], event['end_ns']], [y, y], lw=5, # Slightly thinner bars
                        color=chain_color_map_bright[ch], solid_capstyle='butt',
                        label='_nolegend_') # Avoid legend entries for every bar

            # Add threshold line
            ax.axhline(self.distance_threshold, ls=':', color='black', alpha=0.6, linewidth=1.0)

            # Set consistent y-scale and label
            ax.set_ylim(y_min, y_max)
            ax.set_ylabel(f"Chain {ch} (Å)")
            ax.grid(False) # Turn off grid

            # Remove top/right spines
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

        # Finalize plot
        axes[-1].set_xlabel('Time (ns)')
        fig.suptitle('DW-Gate Distance vs. Debounced State', fontsize=14)

        # Add legend manually below the plot
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Mean', markerfacecolor='red', markersize=8, markeredgecolor='white'),
            Line2D([0], [0], color='black', lw=1.5, label='Median'),
            # Maybe add patch for violin?
            # Patch(facecolor='grey', edgecolor='black', alpha=0.6, label='Distribution (Violin)')
        ]
        fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=2, frameon=False)

        plt.suptitle("DW-Gate Event Duration Distributions", fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # Adjust layout for suptitle and legend

        # Save plot
        plot_filename = "dw_gate_distance_vs_state.png"
        plot_path = os.path.join(self.output_dir, plot_filename)
        save_plot(fig, plot_path)
        self.plot_paths['distance_vs_state'] = os.path.join("dw_gate_analysis", plot_filename) # Relative path


    def plot_open_probability(self):
        """Plots open-state probability per chain."""
        if 'probability_df' not in self.stats_results.get('raw_stats', {}):
            logger.warning("Skipping open probability plot: Probability data not available.")
            return

        prob_df = self.stats_results['raw_stats']['probability_df']
        open_prob_df = prob_df[prob_df['state'] == OPEN_STATE].sort_values('chain')
        if open_prob_df.empty:
             logger.warning("Skipping open probability plot: No open state data found.")
             return

        sns.set_theme(style='ticks', context='notebook')
        plt.figure(figsize=(max(6, len(open_prob_df)*1.5), 5)) # Adjust width based on num chains
        ax = sns.barplot(data=open_prob_df, x='chain', y='probability', palette='viridis') # Use a different palette?

        # Add probability values above bars
        for i, p in enumerate(open_prob_df['probability']):
            plt.text(i, p + 0.02, f'{p:.3f}', ha='center', fontsize=10, fontweight='bold')

        plt.ylim(0, 1.1)
        plt.title('DW-Gate Open State Probability per Chain')
        plt.xlabel('Chain')
        plt.ylabel('Probability')
        ax.grid(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.tight_layout()

        # Save plot
        plot_filename = "dw_gate_open_probability.png"
        plot_path = os.path.join(self.output_dir, plot_filename)
        save_plot(plt.gcf(), plot_path)
        self.plot_paths['open_probability'] = os.path.join("dw_gate_analysis", plot_filename) # Relative path


    def plot_state_heatmap(self):
        """Plots DW-Gate state transitions as a heatmap (Closed=Blue, Open=LightBlue)."""
        if self.df is None or self.df.empty:
            logger.warning("Skipping state heatmap plot: Debounced data not available.")
            return

        sns.set_theme(style='white', context='notebook') # White background often looks good for heatmaps

        # Prepare data for heatmap: Pivot debounced data
        df_pivot = None # Initialize df_pivot to None
        time_extent = None # Initialize time_extent
        try:
            # Ensure correct columns and types before pivoting
            # Using drop_duplicates might hide issues if times aren't unique per frame
            heatmap_pivot = self.df[['time_ns', 'chain', 'state']].drop_duplicates()
            if not heatmap_pivot.empty:
                df_pivot = heatmap_pivot.pivot(index="time_ns", columns="chain", values="state")
            else:
                logger.warning("Heatmap source data (time_ns) is empty after drop_duplicates.")

        except Exception as e:
            logger.warning(f"Failed to pivot data for heatmap using time_ns: {e}. Check for duplicate time/chain entries. Trying frame index.")
            # Fallback: maybe use raw data or skip? Let's try to proceed cautiously.
            # Might need resampling if time points aren't perfectly aligned across chains.
            # For now, attempt pivot on frame index as fallback.
            try:
                 logger.warning("Retrying heatmap pivot using frame index.")
                 heatmap_pivot = self.df[['frame', 'chain', 'state']].drop_duplicates()
                 if not heatmap_pivot.empty:
                     df_pivot = heatmap_pivot.pivot(index="frame", columns="chain", values="state")
                     # We'll need to map frame index back to time for the x-axis label later
                     frame_to_time_map = self.df[['frame', 'time_ns']].drop_duplicates().set_index('frame')['time_ns']
                     if not frame_to_time_map.empty:
                          time_extent = [frame_to_time_map.min(), frame_to_time_map.max()]
                     else:
                          logger.warning("Could not create frame_to_time_map for heatmap extent.")
                 else:
                      logger.warning("Heatmap source data (frame) is empty after drop_duplicates.")

            except Exception as e2:
                 logger.error(f"Fallback pivot using frame index also failed: {e2}. Skipping heatmap.")
                 # df_pivot remains None in this case, handled below

        # Check if pivoting failed or resulted in an empty DataFrame
        if df_pivot is None or df_pivot.empty:
            logger.warning("Skipping state heatmap: Pivoted data is None or empty after pivot attempts.")
            return

        # Map states to numeric values (Closed=0, Open=1)
        state_map = {CLOSED_STATE: 0, OPEN_STATE: 1}
        df_numeric = df_pivot.replace(state_map).fillna(-1) # Use -1 for missing data/gaps

        # Create a custom colormap: Missing=Gray, Closed=DarkBlue, Open=LightBlue
        cmap = mcolors.ListedColormap(['#cccccc', '#add8e6', '#00008b']) # Gray, LightBlue, DarkBlue
        bounds = [-1.5, -0.5, 0.5, 1.5] # Define boundaries for the colors
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        # Create figure
        plt.figure(figsize=(12, max(3, 0.5 * len(df_numeric.columns)))) # Adjust height based on chains
        ax = plt.gca()

        # Determine extent (time or frame)
        # Check if time_extent was successfully assigned (i.e., frame-based pivot was used)
        if time_extent is not None:
             extent = [time_extent[0], time_extent[1], -0.5, len(df_numeric.columns)-0.5]
             xlabel = 'Time (ns) [from frame index]' # Clarify label
        elif df_pivot is not None: # Check df_pivot exists before using its index
             # Use original time_ns extent if first pivot worked
             extent = [df_pivot.index.min(), df_pivot.index.max(), -0.5, len(df_numeric.columns)-0.5]
             xlabel = 'Time (ns)'
        else:
            # Should not happen if the earlier check passed, but as a failsafe
            logger.error("Cannot determine heatmap extent because df_pivot is None.")
            return # Cannot proceed


        # Plot the heatmap
        im = ax.imshow(df_numeric.T, aspect='auto', cmap=cmap, norm=norm, interpolation='nearest',
                       extent=extent)

        # Set labels and title
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Chain')
        ax.set_title('DW Gate State Heatmap')

        # Set y-ticks to chain labels (use sorted order from columns)
        ax.set_yticks(np.arange(len(df_numeric.columns)))
        ax.set_yticklabels(sorted(df_numeric.columns))

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, ticks=[-1, 0, 1], orientation='vertical', fraction=0.05, pad=0.04)
        cbar.ax.set_yticklabels(['Missing', 'Open', 'Closed'], fontsize='small')

        # Remove gridlines
        ax.grid(False)

        plt.tight_layout()

        # Save plot
        plot_filename = "dw_gate_state_heatmap.png"
        plot_path = os.path.join(self.output_dir, plot_filename)
        save_plot(plt.gcf(), plot_path)
        self.plot_paths['state_heatmap'] = os.path.join("dw_gate_analysis", plot_filename) # Relative path


    def plot_duration_distributions(self):
        """
        Plots distributions of event durations using violin plots overlaid with
        box plots and swarm plots for individual events. Uses linear y-scale.
        """
        if self.events_df is None or self.events_df.empty:
            logger.warning("Skipping duration distribution plot: Events data not available.")
            return

        sns.set_theme(style='ticks', context='notebook')

        # Create figure with two subplots (Open, Closed) side-by-side
        fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=False) # Don't share Y axis initially

        # Filter events by state
        open_events = self.events_df.query("state == @OPEN_STATE")
        closed_events = self.events_df.query("state == @CLOSED_STATE")

        # Get sorted chains and palettes
        chains = sorted(self.events_df.chain.unique())
        n_chains = len(chains)
        if n_chains == 0: return

        pastel_palette = sns.color_palette("pastel", n_chains)
        bright_palette = sns.color_palette("bright", n_chains)
        chain_color_map_pastel = {ch: col for ch, col in zip(chains, pastel_palette)}
        chain_color_map_bright = {ch: col for ch, col in zip(chains, bright_palette)}

        # Helper function to plot a single panel (modified from original)
        def plot_panel(ax, data, chains, palette_violin, palette_points, chain_map_bright, title):
            if data.empty:
                ax.text(0.5, 0.5, f"No {title.split()[0]} Events", ha='center', va='center', transform=ax.transAxes)
                ax.set_title(title, fontsize=14, fontweight='bold')
                ax.set_xlabel("Chain", fontsize=12)
                return # Skip plotting if no data for this state

            # 1. Violin Plot
            sns.violinplot(
                x="chain", y="duration_ns", data=data, ax=ax, order=chains,
                palette=palette_violin, inner=None, alpha=0.6, # More transparent
                bw_method=0.3, cut=0, scale="width", linewidth=1 # Thinner edge
            )

            # 2. Box Plot (without fliers, customize appearance)
            sns.boxplot(
                x="chain", y="duration_ns", data=data, ax=ax, order=chains,
                width=0.3, color="white", showfliers=False,
                medianprops={"color": "black", "linewidth": 1.5}, # Black median
                boxprops={"facecolor": (0,0,0,0), "edgecolor": "black", "linewidth":1.0}, # Transparent box, thin edge
                whiskerprops={"color": "black", "linewidth": 1.0},
                capprops={"color": "black", "linewidth": 1.0},
                showmeans=True,
                meanprops={"marker":"o", "markerfacecolor":"red", "markeredgecolor":"white", "markersize":7, "zorder":10} # Red mean dot
            )

            # 3. Swarm Plot (use bright palette mapped per chain)
            # Map chain colors for swarmplot explicitly
            swarm_palette = [chain_map_bright.get(ch, "grey") for ch in chains]

            sns.swarmplot(
                x="chain", y="duration_ns", data=data, ax=ax, order=chains,
                palette=swarm_palette, # Use the mapped bright colors
                size=5, # Smaller points
                edgecolor="black", linewidth=0.5, # Thin black edge
                alpha=0.8, zorder=5 # Ensure points are visible
            )

            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.grid(True, axis='y', linestyle=':', alpha=0.7) # Lighter grid
            ax.set_yscale('linear') # Ensure linear scale

            # Adjust y-limit to data range + margin
            if not data.empty:
                 data_max = data['duration_ns'].max() * 1.1
                 ax.set_ylim(bottom=0, top=max(data_max, ax.get_ylim()[1]*0.1)) # Ensure some space if max is low
            else:
                 ax.set_ylim(bottom=0)


            # Add n=<count> text below x-axis ticks
            for i, chain in enumerate(chains):
                n_events = len(data[data['chain'] == chain])
                if n_events > 0:
                    ax.text(i, ax.get_ylim()[0] - (ax.get_ylim()[1]-ax.get_ylim()[0])*0.05, # Position below axis
                            f"n={n_events}", ha='center', va='top', fontsize=9)


        # Plot both panels
        plot_panel(axes[0], open_events, chains, pastel_palette, bright_palette, chain_color_map_bright, "Open State Durations")
        plot_panel(axes[1], closed_events, chains, pastel_palette, bright_palette, chain_color_map_bright, "Closed State Durations")

        # Format axes
        axes[0].set_ylabel("Duration (ns)", fontsize=12)
        axes[1].set_ylabel("")  # No y-label on second plot
        axes[0].set_xlabel("Chain", fontsize=12)
        axes[1].set_xlabel("Chain", fontsize=12)

        # Find common y-limit for both plots for better comparison
        y_max_open = axes[0].get_ylim()[1] if not open_events.empty else 0
        y_max_closed = axes[1].get_ylim()[1] if not closed_events.empty else 0
        common_y_max = max(y_max_open, y_max_closed) * 1.05 # Add 5% margin

        if common_y_max > 0: # Avoid setting limits if no data at all
             axes[0].set_ylim(0, common_y_max)
             axes[1].set_ylim(0, common_y_max)


        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Mean', markerfacecolor='red', markersize=8, markeredgecolor='white'),
            Line2D([0], [0], color='black', lw=1.5, label='Median'),
            # Maybe add patch for violin?
            # Patch(facecolor='grey', edgecolor='black', alpha=0.6, label='Distribution (Violin)')
        ]
        fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=2, frameon=False)

        plt.suptitle("DW-Gate Event Duration Distributions", fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # Adjust layout for suptitle and legend

        # Save plot
        plot_filename = "dw_gate_duration_distributions.png"
        plot_path = os.path.join(self.output_dir, plot_filename)
        save_plot(fig, plot_path)
        self.plot_paths['duration_distributions'] = os.path.join("dw_gate_analysis", plot_filename) # Relative path


    def run_analysis(self) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """
        Executes the full DW-gate analysis pipeline.

        Returns:
            Tuple[Dict[str, Any], Dict[str, str]]:
                - A dictionary containing the statistical results (HTML tables, raw data).
                - A dictionary mapping plot keys to their relative paths.
        """
        logger.info("--- Starting DW-Gate Analysis ---")
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Steps:
        self.collect_distances()    # Also calls calculate_reference_distances if auto_detect_refs is True
        self.apply_debouncing()
        self.build_events()         # Also calls verify/fix coverage and saves events.csv
        self.run_statistical_analysis() # Calculates stats, stores DFs/Dicts

        # Generate remaining plots:
        self.plot_distance_vs_state()
        self.plot_open_probability()
        self.plot_state_heatmap()
        self.plot_duration_distributions()

        logger.info("--- DW-Gate Analysis Finished ---")

        # Return the collected statistics and plot paths
        return self.stats_results, self.plot_paths


# --- Main Entry Point Function (Called by PoreAnalysis main script) ---

def analyse_dw_gates(
    run_dir: str,
    psf_file: str,
    dcd_file: str,
    results: Dict[str, Any] # Existing results dict from main workflow
) -> Dict[str, Any]:
    """
    Main function called by the analysis pipeline to perform DW-gate analysis.

    1. Loads Universe.
    2. Identifies filter and gate residues.
    3. Initializes and runs the DWGateAnalysis class.
    4. Returns statistics and plot paths for reporting.

    Args:
        run_dir (str): Path to the simulation run directory.
        psf_file (str): Path to the topology (PSF) file.
        dcd_file (str): Path to the trajectory (DCD) file.
        results (Dict[str, Any]): Dictionary holding results from previous steps
                                  (e.g., time_points).

    Returns:
        Dict[str, Any]: Dictionary containing DW-gate results, including:
                        'dw_gate_stats': Dictionary of statistical results (HTML tables).
                        'dw_gate_plots': Dictionary mapping plot keys to relative paths.
                        'Error': String message if analysis failed critically.
    """
    logger.info("--- Preparing for DW-Gate Analysis ---")
    dw_results = {} # Initialize dict for DW gate specific results

    # Define output subdirectory for this module
    output_dir = os.path.join(run_dir, "dw_gate_analysis")
    os.makedirs(output_dir, exist_ok=True)

    # --- Load Universe ---
    try:
        logger.debug(f"Loading Topology: {psf_file}, Trajectory: {dcd_file}")
        # Check if universe is already loaded in results? Assume not for now.
        u = mda.Universe(psf_file, dcd_file)
        n_frames = len(u.trajectory)
        if n_frames == 0:
            raise ValueError("Trajectory contains 0 frames.")
        logger.info(f"Universe loaded with {n_frames} frames for DW-gate analysis.")

    except FileNotFoundError as e:
         logger.error(f"Input file not found for DW Gate analysis: {e}")
         dw_results['Error'] = f'Input file not found: {e}'
         return dw_results # Return immediately on critical failure
    except Exception as e:
        logger.error(f"Failed to load Universe for DW Gate analysis: {e}", exc_info=True)
        dw_results['Error'] = f'Universe load failed: {e}'
        return dw_results # Return immediately

    # --- Identify Filter & Gate Residues (Using existing functions) ---
    try:
        # 1. Find Filter Residues (needed by gate identification)
        # Assuming find_filter_residues takes Universe and logger
        filter_res_map = find_filter_residues(u, logger)
        if not filter_res_map:
            raise ValueError("find_filter_residues returned empty or None.")
        logger.info(f"Identified selectivity filter residues for {len(filter_res_map)} chains.")

        # 2. Find Gate Residues for each chain using the filter map
        valid_gates: Dict[str, GateResidues] = {}
        chain_ids = sorted(filter_res_map.keys())
        failed_chains = []
        for segid in chain_ids:
            try:
                gate = find_gate_residues_for_chain(u, segid, filter_res_map)
                if gate: # Ensure it's not None
                    valid_gates[segid] = gate
                else:
                     # Should not happen if find_gate_residues_for_chain raises error on fail
                     logger.warning(f"find_gate_residues_for_chain returned None for segid {segid}")
                     failed_chains.append(segid)
            except ValueError as e:
                logger.warning(f"Skipping chain {segid} due to DW gate residue identification error: {e}")
                failed_chains.append(segid)

        if not valid_gates:
            logger.error("Failed to identify valid DW gate residues for ANY chain. Aborting DW Gate analysis.")
            dw_results['Error'] = 'DW Gate residue ID failed for all chains'
            return dw_results
        if failed_chains:
            logger.warning(f"DW Gate analysis proceeding with {len(valid_gates)} chains (failed: {failed_chains}).")

    except Exception as e:
        logger.error(f"Failed during residue identification for DW Gate analysis: {e}", exc_info=True)
        dw_results['Error'] = f'Residue identification failed: {e}'
        return dw_results

    # --- Run the DWGateAnalysis Class ---
    try:
        analyzer = DWGateAnalysis(
            universe=u,
            gate_residues=valid_gates,
            output_dir=output_dir
        )

        # Run the analysis pipeline within the class
        stats_results, plot_paths = analyzer.run_analysis()

        # Store results for reporting
        dw_results['dw_gate_stats'] = stats_results
        dw_results['dw_gate_plots'] = plot_paths
        # Optionally add key dataframes to results if needed downstream (but prefer summary stats)
        # dw_results['dw_gate_events_df'] = analyzer.events_df
        logger.info("Successfully completed DW-Gate analysis.")

    except Exception as e:
        logger.error(f"An error occurred during DW-Gate analysis execution: {e}", exc_info=True)
        dw_results['Error'] = f'DW-Gate analysis failed: {e}'
        # Optionally include partial results if available
        if 'analyzer' in locals():
            dw_results['dw_gate_stats'] = getattr(analyzer, 'stats_results', {'Error': 'Analysis failed mid-run'})
            dw_results['dw_gate_plots'] = getattr(analyzer, 'plot_paths', {})

    return dw_results 