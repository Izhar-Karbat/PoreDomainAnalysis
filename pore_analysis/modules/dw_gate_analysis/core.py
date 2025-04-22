"""Core orchestration class for DW Gate analysis."""

import logging
import os
from typing import Dict, Any, Tuple, List, Optional

import MDAnalysis as mda
import pandas as pd
import numpy as np

# Import functions from sibling modules
from . import data_collection
from . import signal_processing
from . import state_detection
from . import event_building
from . import statistics
from . import visualization
from .residue_identification import GateResidues # Still needed for type hinting
from pore_analysis.core.config import ( # Import necessary config values
    FRAMES_PER_NS,
    DW_GATE_TOLERANCE_FRAMES,
    DW_GATE_AUTO_DETECT_REFS,
    DW_GATE_DEFAULT_CLOSED_REF_DIST,
    DW_GATE_DEFAULT_OPEN_REF_DIST,
    DEFAULT_CUTOFF # Changed from DW_GATE_DEFAULT_THRESHOLD
)
# Import the function to find filter residues
from pore_analysis.modules.ion_analysis.filter_structure import find_filter_residues

logger = logging.getLogger(__name__)

class DWGateAnalysis:
    """
    Orchestrates the DW-gate analysis workflow by calling functions from
    different modules (data_collection, signal_processing, etc.).
    """
    def __init__(self,
                 psf_file: str,
                 dcd_file: str,
                 output_dir: str,
                 chain_ids: List[str],
                 asp_glu_resname: str = "ASP",
                 asp_glu_resid: int = 27,
                 trp_resname: str = "TRP",
                 trp_resid: int = 31,
                 # Configuration parameters passed directly
                 frames_per_ns: float = FRAMES_PER_NS,
                 tolerance_frames: int = DW_GATE_TOLERANCE_FRAMES,
                 auto_detect_refs: bool = DW_GATE_AUTO_DETECT_REFS,
                 default_closed_ref: float = DW_GATE_DEFAULT_CLOSED_REF_DIST,
                 default_open_ref: float = DW_GATE_DEFAULT_OPEN_REF_DIST,
                 distance_threshold: float = DEFAULT_CUTOFF): # Changed default value
        """
        Initializes the DWGateAnalysis orchestrator.

        Args:
            psf_file: Path to the PSF file.
            dcd_file: Path to the DCD file.
            output_dir: Directory to save analysis outputs (plots, CSVs).
            chain_ids: List of chain/segment IDs to analyze.
            asp_glu_resname: Resname for the Asp/Glu residue.
            asp_glu_resid: Resid for the Asp/Glu residue.
            trp_resname: Resname for the Trp residue.
            trp_resid: Resid for the Trp residue.
            frames_per_ns: Frames per nanosecond for time conversion.
            tolerance_frames: RLE debouncing tolerance in frames.
            auto_detect_refs: Whether to automatically detect ref distances via KDE.
            default_closed_ref: Default closed state reference distance (if not auto-detected).
            default_open_ref: Default open state reference distance (if not auto-detected).
            distance_threshold: Default distance threshold (used for plotting KDE baseline).
        """
        self.psf_file = psf_file
        self.dcd_file = dcd_file
        self.output_dir = output_dir
        self.chain_ids = chain_ids
        self.asp_glu_resname = asp_glu_resname
        self.asp_glu_resid = asp_glu_resid
        self.trp_resname = trp_resname
        self.trp_resid = trp_resid

        # Store config parameters
        self.frames_per_ns = frames_per_ns
        self.dt_ns = 1.0 / frames_per_ns if frames_per_ns > 0 else 0.1 # Handle potential division by zero
        self.tolerance_frames = tolerance_frames
        self.auto_detect_refs = auto_detect_refs
        self.default_closed_ref = default_closed_ref
        self.default_open_ref = default_open_ref
        self.distance_threshold = distance_threshold # Store for KDE plot

        # Initialize attributes to store results
        self.universe: Optional[mda.Universe] = None
        self.gate_residues: Dict[str, GateResidues] = {}
        self.df_distances_wide: Optional[pd.DataFrame] = None # Store wide format distances
        self.df_states_wide: Optional[pd.DataFrame] = None # Store wide format with raw states
        self.df_states_long: Optional[pd.DataFrame] = None # Store long format with debounced states
        self.events_df: Optional[pd.DataFrame] = None
        self.stats_results: Dict[str, Any] = {}
        self.plot_paths: Dict[str, str] = {}
        self.final_closed_ref: float = default_closed_ref # Store final reference used
        self.final_open_ref: float = default_open_ref   # Store final reference used

        logger.info(f"DWGateAnalysis initialized. Output Dir: {self.output_dir}")
        logger.info(f"Residue Selection: {asp_glu_resname}{asp_glu_resid}-{trp_resname}{trp_resid} for chains {self.chain_ids}")
        logger.info(f"Time step (dt_ns): {self.dt_ns:.4f}")
        logger.info(f"Debouncing Tolerance (frames): {self.tolerance_frames}")
        logger.info(f"Reference Detection: Auto={self.auto_detect_refs}, Defaults=({self.default_closed_ref:.2f}, {self.default_open_ref:.2f}) Å")


    def _melt_state_data(self, df_wide: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Melts the wide distance/state DataFrame into a long format.

        Identifies distance (Dist_*) and optionally state (*_state_raw) columns.
        Handles cases where only distance columns are present (for KDE).

        Args:
            df_wide: DataFrame in wide format.

        Returns:
            DataFrame in long format with columns like [frame, time_ns, chain, distance, state_raw]
            (state_raw is optional), or None if melting fails.
        """
        if df_wide is None or df_wide.empty:
            logger.error("Cannot melt data: Input wide DataFrame is empty or None.")
            return None

        # Common ID variables expected
        id_vars_base = ['frame', 'time_ns']
        id_vars = [col for col in id_vars_base if col in df_wide.columns]
        if not id_vars:
            logger.error("Cannot melt data: Missing essential ID columns (frame or time_ns).")
            return None

        # Identify value columns based on prefixes/suffixes
        dist_cols = sorted([col for col in df_wide.columns if col.startswith("Dist_")])
        state_cols = sorted([col for col in df_wide.columns if col.endswith("_state_raw")])

        if not dist_cols:
            logger.error("Cannot melt data: No distance columns (starting with 'Dist_') found.")
            return None

        logger.debug(f"Melting DataFrame. ID vars: {id_vars}, Dist cols: {dist_cols}, State cols: {state_cols}")

        try:
            # Melt distances
            df_dist_long = pd.melt(df_wide, id_vars=id_vars, value_vars=dist_cols,
                                   var_name="dist_col_name", value_name="distance")
            # Extract chain ID from distance column name
            df_dist_long['chain'] = df_dist_long['dist_col_name'].str.replace("Dist_", "")
            df_dist_long = df_dist_long.drop(columns=["dist_col_name"])

            # Melt states if they exist
            if state_cols:
                if len(dist_cols) != len(state_cols):
                     logger.warning(f"Mismatch between number of distance ({len(dist_cols)}) and state ({len(state_cols)}) columns. State melting might be incorrect.")
                     # Attempt merge anyway, but it might fail or produce unexpected results

                df_state_long = pd.melt(df_wide, id_vars=id_vars, value_vars=state_cols,
                                        var_name="state_col_name", value_name="state_raw")
                # Extract chain ID from state column name (more robust extraction)
                # Assumes format ends with _state_raw, extracts part before it
                df_state_long['chain'] = df_state_long['state_col_name'].str.rsplit('_', n=2).str[0]
                df_state_long = df_state_long.drop(columns=["state_col_name"])

                # Merge distance and state data
                merge_cols = id_vars + ['chain']
                # Check if columns exist before merge
                required_merge_cols_dist = merge_cols + ['distance']
                required_merge_cols_state = merge_cols + ['state_raw']
                if not all(c in df_dist_long.columns for c in required_merge_cols_dist):
                     raise ValueError(f"Missing columns for merging distances: expected {required_merge_cols_dist}, got {df_dist_long.columns}")
                if not all(c in df_state_long.columns for c in required_merge_cols_state):
                     raise ValueError(f"Missing columns for merging states: expected {required_merge_cols_state}, got {df_state_long.columns}")

                df_long = pd.merge(df_dist_long[required_merge_cols_dist],
                                   df_state_long[required_merge_cols_state],
                                   on=merge_cols, how='inner') # Use inner merge
                # Select final columns in desired order
                final_cols = id_vars + ['chain', 'distance', 'state_raw']

            else:
                # Only distances were melted
                df_long = df_dist_long
                # Select final columns (without state_raw)
                final_cols = id_vars + ['chain', 'distance']

            # Ensure final columns exist before selection and sorting
            final_cols = [col for col in final_cols if col in df_long.columns]
            df_long = df_long[final_cols].sort_values(['chain'] + id_vars)

            logger.info(f"Successfully melted data to long format. Shape: {df_long.shape}, Columns: {df_long.columns.tolist()}")
            return df_long.reset_index(drop=True)

        except Exception as e:
            logger.error(f"Error during melting of wide DataFrame: {e}", exc_info=True)
            return None


    def run_analysis(self) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """
        Executes the full DW-gate analysis pipeline using modular functions.

        Returns:
            Tuple[Dict[str, Any], Dict[str, str]]:
                - Dictionary containing the statistical results (DataFrames, test dicts).
                - Dictionary mapping plot keys to their relative saved paths.
        """
        # Import modules inside method to potentially avoid circular imports
        from . import data_collection
        from . import signal_processing
        from . import state_detection
        from . import event_building
        from . import statistics
        from . import visualization

        logger.info("--- Starting DW-Gate Analysis Workflow ---")

        # --- 1. Data Collection --- 
        logger.info("Step 1: Data Collection (Load Universe, Identify Filter & Gate Residues)")
        self.universe = data_collection.load_universe(self.psf_file, self.dcd_file)
        if self.universe is None:
            logger.error("Universe loading failed. Aborting analysis.")
            return {'Error': 'Failed to load Universe'}, {}

        # Add frame index attribute if missing (some MDA versions might need this)
        try:
             # Check if the first frame already has the attribute
             _ = self.universe.trajectory[0].frame
        except AttributeError:
             logger.info("Adding 'frame' attribute to trajectory time steps.")
             self.universe.trajectory.add_transformations(lambda ts: setattr(ts, 'frame', ts.frame) or ts)
        except IndexError:
             logger.error("Trajectory appears empty. Cannot proceed.")
             return {'Error': 'Trajectory has 0 frames'}, {}

        # ---> Identify Filter Residues First <---
        logger.info("Identifying selectivity filter residues...")
        filter_res_map = find_filter_residues(self.universe, logger=logger)
        if not filter_res_map:
            logger.error("Failed to identify filter residues. Cannot determine relative DW-gate residues. Aborting.")
            return {'Error': 'Filter residue identification failed'}, {}
        logger.info(f"Identified filter residues for chains: {list(filter_res_map.keys())}")

        # ---> Now identify DW-gate residues relative to the filter <---
        self.gate_residues = data_collection.select_dw_residues(
            universe=self.universe,
            chain_ids=self.chain_ids, # Pass the originally requested chains
            filter_res_map=filter_res_map # Pass the identified filter map
            # Removed redundant args: asp_glu_resname, asp_glu_resid, trp_resname, trp_resid
        )
        if not self.gate_residues:
            logger.error("DW-Gate residue selection failed for all specified chains relative to filter. Aborting analysis.")
            return {'Error': 'Failed to select any DW-gate residues relative to filter'}, {}
        # Update self.chain_ids to only those where residues were successfully found
        self.chain_ids = sorted(list(self.gate_residues.keys()))
        logger.info(f"Successfully selected residues for chains: {self.chain_ids}")

        self.df_distances_wide = data_collection.calculate_dw_distances(
            self.universe, self.gate_residues, self.dt_ns
        )
        if self.df_distances_wide is None or self.df_distances_wide.empty:
            logger.error("Distance calculation failed. Aborting analysis.")
            return {'Error': 'Failed to calculate distances'}, {}
        # Ensure 'frame' column exists (should be added by calculate_dw_distances)
        if 'frame' not in self.df_distances_wide.columns:
             logger.warning("'frame' column missing from calculate_dw_distances output. Adding index as frame.")
             self.df_distances_wide.insert(0, 'frame', np.arange(len(self.df_distances_wide)))
        # Ensure 'time_ns' column exists
        if 'time_ns' not in self.df_distances_wide.columns:
             logger.warning("'time_ns' column missing from calculate_dw_distances output. Estimating from frame and dt_ns.")
             self.df_distances_wide.insert(1, 'time_ns', self.df_distances_wide['frame'] * self.dt_ns)


        # --- 2. Signal Processing & State Detection --- 
        logger.info("Step 2: Signal Processing & State Detection")
        # Melt data for KDE analysis (only need distances)
        dist_cols_for_kde = [f'Dist_{ch}' for ch in self.chain_ids]
        df_for_kde_melt = self.df_distances_wide[['frame', 'time_ns'] + dist_cols_for_kde].copy()
        df_dist_long_for_kde = self._melt_state_data(df_for_kde_melt)

        if df_dist_long_for_kde is None:
             logger.error("Failed to prepare data for KDE analysis (melting failed). Aborting.")
             return {'Error': 'Failed to melt distance data for KDE'}, {}

        if self.auto_detect_refs:
            logger.info("Attempting to auto-detect reference distances via KDE...")
            determined_closed_ref, determined_open_ref, plot_path_kde = \
                signal_processing.determine_reference_distances(
                    df_raw=df_dist_long_for_kde, # Pass melted data with distance column
                    default_closed_ref=self.default_closed_ref,
                    default_open_ref=self.default_open_ref,
                    distance_threshold=self.distance_threshold, # Pass stored config threshold
                    output_dir=self.output_dir,
                    plot_distributions=True # Control plotting via arg if needed later
                )
            self.final_closed_ref = determined_closed_ref
            self.final_open_ref = determined_open_ref
            if plot_path_kde:
                self.plot_paths['distance_distribution'] = plot_path_kde
            else:
                logger.warning("KDE reference distance plot was not generated.")
            logger.info(f"Using final reference distances: Closed={self.final_closed_ref:.2f}, Open={self.final_open_ref:.2f} Å")
        else:
            logger.info(f"Using default reference distances: Closed={self.final_closed_ref:.2f}, Open={self.final_open_ref:.2f} Å")
            # No KDE plot generated if not auto-detecting

        # Assign initial states based on final reference distances
        self.df_states_wide = state_detection.assign_initial_state(
            df_raw_distances=self.df_distances_wide,
            closed_ref_dist=self.final_closed_ref,
            open_ref_dist=self.final_open_ref,
            dist_col_prefix="Dist_",
            state_col_suffix="_state_raw"
        )
        if self.df_states_wide is None or self.df_states_wide.empty:
             logger.error("Initial state assignment failed. Aborting analysis.")
             return {'Error': 'Failed to assign initial states'}, self.plot_paths

        # Melt the wide DataFrame with distances and raw states into long format
        df_long_raw_states = self._melt_state_data(self.df_states_wide)
        if df_long_raw_states is None or 'state_raw' not in df_long_raw_states.columns:
             logger.error("Melting data for debouncing failed or state_raw column missing. Aborting analysis.")
             return {'Error': 'Failed to melt data with raw states for debouncing'}, self.plot_paths

        # Apply RLE debouncing to the long format data
        self.df_states_long = signal_processing.apply_rle_debouncing(
            df_states=df_long_raw_states,
            tolerance_frames=self.tolerance_frames,
            state_col='state_raw', # Input raw state column
            debounced_col='state'   # Output debounced state column name
        )
        if self.df_states_long is None or self.df_states_long.empty:
             logger.error("Debouncing failed. Aborting analysis.")
             return {'Error': 'Debouncing failed'}, self.plot_paths

        # --- 3. Event Building --- 
        logger.info("Step 3: Event Building")
        self.events_df = event_building.build_events_from_states(
            df_states_long=self.df_states_long,
            dt_ns=self.dt_ns,
            state_col='state' # Use the final debounced state column
        )
        if self.events_df is None or self.events_df.empty:
             logger.error("Event building failed. Aborting analysis.")
             # Return plots generated so far (KDE plot)
             return {'Error': 'Event building failed'}, self.plot_paths

        # --- 4. Statistics --- 
        logger.info("Step 4: Statistics Calculation")
        self.stats_results = statistics.calculate_dw_statistics(self.events_df)
        if not self.stats_results or self.stats_results.get('Error'):
             logger.error(f"Statistics calculation failed: {self.stats_results.get('Error', 'Unknown error')}")
             # Continue to plotting if possible, but stats might be missing

        # Save events to CSV (regardless of stats success)
        csv_path = os.path.join(self.output_dir, "dw_gate_events.csv")
        try:
            self.events_df.to_csv(csv_path, index=False, float_format="%.3f")
            logger.info(f"Saved DW Gate events data to {csv_path}")
        except Exception as e:
            logger.error(f"Failed to save DW Gate events CSV: {e}")

        # --- 5. Visualization --- 
        logger.info("Step 5: Generating Visualizations")
        try:
            # Plot distance vs state (needs wide distances/states and events)
            # Pass df_states_wide which has both Dist_X and State_X_raw columns
            plot_path_dist_state = visualization.plot_distance_vs_state(
                df_states=self.df_states_wide, # Contains distances + raw states
                events_df=self.events_df, # Events needed for final state bars
                output_dir=self.output_dir,
                distance_threshold=self.distance_threshold, # Pass stored config threshold
                closed_ref_dist=self.final_closed_ref,
                open_ref_dist=self.final_open_ref,
                state_col='state', # Use final state from events
                dist_col_prefix="Dist_",
                time_col='time_ns'
            )
            if plot_path_dist_state: self.plot_paths['distance_vs_state'] = plot_path_dist_state

            # Plot open probability (needs stats results)
            prob_df = self.stats_results.get('probability_df')
            if isinstance(prob_df, pd.DataFrame) and not prob_df.empty:
                 plot_path_prob = visualization.plot_open_probability(
                     probability_df=prob_df,
                     output_dir=self.output_dir
                 )
                 if plot_path_prob: self.plot_paths['open_probability'] = plot_path_prob
            else:
                 logger.warning("Skipping open probability plot: probability_df missing, empty, or not a DataFrame in stats results.")

            # Plot heatmap (needs long format states with final 'state' column)
            plot_path_heatmap = visualization.plot_state_heatmap(
                df_states=self.df_states_long,
                output_dir=self.output_dir,
                state_col='state',
                time_col='time_ns',
                frame_col='frame'
            )
            if plot_path_heatmap: self.plot_paths['state_heatmap'] = plot_path_heatmap

            # Plot duration distributions (needs events)
            plot_path_durations = visualization.plot_duration_distributions(
                events_df=self.events_df,
                output_dir=self.output_dir
            )
            if plot_path_durations: self.plot_paths['duration_distributions'] = plot_path_durations

        except Exception as e:
            logger.error(f"Error during visualization generation: {e}", exc_info=True)
            # Analysis succeeded, but plotting failed partially or fully

        logger.info("--- DW-Gate Analysis Workflow Finished ---")

        # Return the collected statistics and plot paths
        # Ensure stats_results is a dict, even if errors occurred
        if not isinstance(self.stats_results, dict):
            logger.error(f"Stats results are not a dictionary: {type(self.stats_results)}. Returning empty stats dict.")
            self.stats_results = {'Error': 'Stats results became invalid type'}
            
        return self.stats_results, self.plot_paths

# Placeholder for DWGateAnalysis class removed
