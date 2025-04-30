# filename: pore_analysis/modules/dw_gate_analysis/state_detection.py
"""Functions for detecting open/closed states from signal data."""

import logging
import pandas as pd
import numpy as np

# Assuming utils is a sibling module for constants
from .utils import CLOSED_STATE, OPEN_STATE

logger = logging.getLogger(__name__)

def assign_initial_state(
    df_raw_distances: pd.DataFrame,
    closed_ref_dist: float,
    open_ref_dist: float,
    dist_col_prefix: str = "Dist_",
    state_col_suffix: str = "_state_raw" # Suffix for the *initial* state assignment
) -> pd.DataFrame:
    """
    Assigns an initial state (closed/open) based on distance relative to reference distances.

    Calculates a midpoint threshold between the closed and open reference distances.
    Assigns CLOSED_STATE if distance <= threshold, OPEN_STATE otherwise.
    Operates on distance columns identified by `dist_col_prefix`.
    Adds new columns for the raw state assignments with `state_col_suffix`.

    Args:
        df_raw_distances: DataFrame containing time and distance columns (e.g., 'Dist_ChainA').
                          Must contain columns identified by `dist_col_prefix`.
        closed_ref_dist: Reference distance for the closed state (determined or default).
        open_ref_dist: Reference distance for the open state (determined or default).
        dist_col_prefix: Prefix identifying the distance columns (default: "Dist_").
        state_col_suffix: Suffix to add for the new raw state columns (default: "_state_raw").

    Returns:
        A new DataFrame with added raw state columns for each distance column found.
        Returns the original DataFrame unmodified if inputs are invalid or no distance columns found.
        Returns None if a critical error occurs during processing.
    """
    if df_raw_distances is None or df_raw_distances.empty:
        logger.error("Input distance DataFrame is empty or None. Cannot assign initial states.")
        return df_raw_distances # Return original df if input invalid

    if closed_ref_dist >= open_ref_dist:
        logger.warning(
            f"Closed reference distance ({closed_ref_dist:.2f}) is not less than open "
            f"reference distance ({open_ref_dist:.2f}). State assignment threshold might be unreliable."
        )
        # Proceed, but log the warning.

    # Create a copy to avoid modifying the original DataFrame passed to the function
    df_with_states = df_raw_distances.copy()
    distance_cols = [col for col in df_with_states.columns if col.startswith(dist_col_prefix)]

    if not distance_cols:
        logger.error(f"No distance columns found with prefix '{dist_col_prefix}'. Cannot assign states.")
        return df_with_states # Return the copy without state columns

    # Calculate the midpoint threshold
    threshold = (closed_ref_dist + open_ref_dist) / 2.0
    logger.info(f"Assigning initial states using midpoint threshold: {threshold:.3f} Å "
                f"(Closed Ref: {closed_ref_dist:.2f}, Open Ref: {open_ref_dist:.2f})")

    num_processed = 0
    for dist_col in distance_cols:
        try:
            # Extract chain identifier from column name (e.g., "Dist_ChainA" -> "ChainA")
            # Handle potential cases where prefix might not be exactly "Dist_" if changed
            if dist_col.startswith(dist_col_prefix):
                chain_id_part = dist_col[len(dist_col_prefix):]
                state_col_name = f"{chain_id_part}{state_col_suffix}" # e.g., "ChainA_state_raw"

                # Apply threshold: closed if <= threshold, open if > threshold
                # Handle NaNs: Assign NaN state if distance is NaN
                df_with_states[state_col_name] = np.where(
                    df_with_states[dist_col].isna(),
                    np.nan, # Keep NaN as NaN
                    np.where(df_with_states[dist_col] <= threshold, CLOSED_STATE, OPEN_STATE)
                )
                logger.debug(f"Assigned initial states to column '{state_col_name}'.")
                num_processed += 1
            else:
                 logger.warning(f"Column '{dist_col}' does not start with expected prefix '{dist_col_prefix}'. Skipping state assignment for this column.")

        except Exception as e:
            logger.error(f"Error assigning initial state for column '{dist_col}': {e}", exc_info=True)
            # If a critical error occurs for one column, maybe return None or raise?
            # For now, log error and continue processing other columns.
            # Consider returning None if any column fails fundamentally.

    if num_processed == 0:
        logger.error(f"Failed to process any distance columns starting with '{dist_col_prefix}'.")
        # Return the original df if no columns were processed.
        return df_raw_distances.copy()

    # Log summary of assigned states for the first processed column if possible
    first_processed_dist_col = next((col for col in distance_cols if f"{col[len(dist_col_prefix):]}{state_col_suffix}" in df_with_states), None)
    if first_processed_dist_col:
         example_state_col = f"{first_processed_dist_col[len(dist_col_prefix):]}{state_col_suffix}"
         logger.debug(f"Initial state counts for {example_state_col}:\n"
                      f"{df_with_states[example_state_col].value_counts(dropna=False)}")

    return df_with_states
