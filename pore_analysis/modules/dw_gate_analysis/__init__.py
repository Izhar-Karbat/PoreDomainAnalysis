"""
DW-Gate Analysis Module Entry Point.
"""

import logging
import os
from typing import Dict, Any
import pandas as pd

import MDAnalysis as mda

# Import the orchestrator class
from .core import DWGateAnalysis

# Import necessary helper functions/config from other parts of the suite
# Assuming find_filter_residues is available for initial chain ID determination
# Adjust the import path based on where find_filter_residues actually lives
try:
    from pore_analysis.modules.ion_analysis.filter_structure import find_filter_residues
except ImportError:
    logger = logging.getLogger(__name__)
    logger.error("Could not import find_filter_residues. DW-Gate analysis relies on it for chain identification.")
    # Define a dummy function to allow execution to proceed but fail gracefully later
    def find_filter_residues(u, logger): return {}

# Import necessary config values (Optional, as DWGateAnalysis imports them too,
# but might be needed for pre-checks or specific setup here)
# from ...core.config import ...

logger = logging.getLogger(__name__)

def analyse_dw_gates(
    run_dir: str,
    psf_file: str,
    dcd_file: str,
    results: Dict[str, Any] # Existing results dict from main workflow (currently unused here)
) -> Dict[str, Any]:
    """
    Entry point function called by the main analysis pipeline to perform DW-gate analysis.

    1. Sets up output directory.
    2. Loads Universe.
    3. Identifies chains using filter residues.
    4. Initializes and runs the DWGateAnalysis class.
    5. Returns statistics and plot paths for reporting.

    Args:
        run_dir (str): Path to the simulation run directory.
        psf_file (str): Path to the topology (PSF) file.
        dcd_file (str): Path to the trajectory (DCD) file.
        results (Dict[str, Any]): Dictionary holding results from previous steps (currently unused).

    Returns:
        Dict[str, Any]: Dictionary containing DW-gate results, compatible with the main
                        reporting structure (e.g., keys 'dw_gate_stats', 'dw_gate_plots').
                        Includes an 'Error' key if analysis failed critically.
    """
    logger.info("--- Initializing DW-Gate Analysis ---")
    dw_module_results = {} # Initialize dict for results from this module

    # Define output subdirectory for this module
    output_subdir = "dw_gate_analysis"
    output_dir_path = os.path.join(run_dir, output_subdir)
    # The DWGateAnalysis constructor will create this directory

    # --- Load Universe (Minimal load here, detailed load in DWGateAnalysis) ---
    # We need the universe primarily to find the filter residues and get chain IDs
    temp_universe = None
    try:
        logger.debug(f"Loading Topology: {psf_file} for initial residue ID.")
        temp_universe = mda.Universe(psf_file)
        logger.debug("Temporary Universe loaded for residue identification.")
    except FileNotFoundError as e:
         logger.error(f"Input PSF file not found for DW Gate analysis: {e}")
         dw_module_results['Error'] = f'Input PSF not found: {e}'
         return dw_module_results # Return immediately on critical failure
    except Exception as e:
        logger.error(f"Failed to load temporary Universe for DW Gate analysis: {e}", exc_info=True)
        dw_module_results['Error'] = f'Temporary Universe load failed: {e}'
        return dw_module_results

    # --- Identify Chains using Filter Residues ---
    chain_ids = []
    try:
        # Assuming find_filter_residues takes Universe and logger
        filter_res_map = find_filter_residues(temp_universe, logger)
        if not filter_res_map:
            raise ValueError("find_filter_residues returned empty or None.")
        chain_ids = sorted(list(filter_res_map.keys()))
        if not chain_ids:
             raise ValueError("No chain IDs derived from filter residues.")
        logger.info(f"Identified potential chains for DW-Gate analysis based on filter residues: {chain_ids}")
    except Exception as e:
        logger.error(f"Failed during filter residue identification for DW Gate analysis: {e}", exc_info=True)
        dw_module_results['Error'] = f'Filter residue identification failed: {e}'
        del temp_universe # Clean up temporary universe
        return dw_module_results
    finally:
        # Clean up the temporary universe object regardless of success/failure
        if temp_universe:
            del temp_universe

    # --- Initialize and Run the DWGateAnalysis Orchestrator ---
    try:
        # Instantiate the main analysis class from core.py
        # It will handle loading the full universe, selecting gate residues for the
        # identified chain_ids, and running the entire workflow.
        # Configuration parameters (tolerance, refs, etc.) are handled by its __init__ using core.config
        analyzer = DWGateAnalysis(
            psf_file=psf_file,
            dcd_file=dcd_file,
            output_dir=output_dir_path,
            chain_ids=chain_ids # Pass the identified chains
            # Residue names/ids use defaults unless passed explicitly here or read from config
            # Config params (frames_per_ns etc) are read from config by __init__
        )

        # Run the analysis pipeline within the class
        stats_results, plot_paths = analyzer.run_analysis()

        # Structure the results for the main analysis summary
        dw_module_results['dw_gate_stats'] = stats_results
        dw_module_results['dw_gate_plots'] = plot_paths
        # Optionally add key summary values directly if needed by aggregate reports
        # e.g., extract overall open probability if available in stats_results
        # Check if stats_results and 'probability_df' key exist and if it's a DataFrame
        if isinstance(stats_results, dict) and isinstance(stats_results.get('probability_df'), pd.DataFrame):
            prob_df = stats_results['probability_df']
            # Use state constants if available, otherwise string literals
            # Assuming OPEN_STATE is defined or imported, e.g., from .state_detection
            # If not, use 'Open' directly for robustness
            open_state_key = getattr(state_detection, 'OPEN_STATE', 'Open')
            # Example: Calculate mean open probability across chains
            if open_state_key in prob_df.columns:
                # Ensure the column contains numeric data before calculating mean
                if pd.api.types.is_numeric_dtype(prob_df[open_state_key]):
                    dw_module_results['mean_open_probability'] = prob_df[open_state_key].mean()
                else:
                    logger.warning(f"Column '{open_state_key}' in probability_df is not numeric. Cannot calculate mean.")

        logger.info("Successfully completed DW-Gate analysis module execution.")

    except Exception as e:
        logger.error(f"An error occurred during DW-Gate analysis execution via DWGateAnalysis class: {e}", exc_info=True)
        dw_module_results['Error'] = f'DW-Gate analysis failed: {e}'
        # Store partial results if available (plot paths might exist even if stats failed)
        if 'analyzer' in locals():
            dw_module_results.setdefault('dw_gate_stats', getattr(analyzer, 'stats_results', {'Error': 'Analysis failed mid-run'}))
            dw_module_results.setdefault('dw_gate_plots', getattr(analyzer, 'plot_paths', {}))

    return dw_module_results

# Expose the main function for the PoreAnalysis suite
__all__ = ['analyse_dw_gates'] 