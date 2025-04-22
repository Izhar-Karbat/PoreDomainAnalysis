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
    state_col_suffix: str = "_state_raw"
) -> pd.DataFrame:
    """
    Assigns an initial state (closed/open) based on distance relative to reference distances.

    Calculates a midpoint threshold between the closed and open reference distances.
    Assigns CLOSED_STATE if distance <= threshold, OPEN_STATE otherwise.
    Operates on distance columns identified by `dist_col_prefix`.
    Adds new columns for the raw state assignments with `state_col_suffix`.

    Args:
        df_raw_distances: DataFrame containing time and distance columns (e.g., 'Dist_ChainA').
        closed_ref_dist: Reference distance for the closed state.
        open_ref_dist: Reference distance for the open state.
        dist_col_prefix: Prefix identifying the distance columns (default: "Dist_").
        state_col_suffix: Suffix to add for the new raw state columns (default: "_state_raw").

    Returns:
        A new DataFrame with added raw state columns for each distance column found.
        Returns the original DataFrame unmodified if inputs are invalid or no distance columns found.
    """
    if df_raw_distances is None or df_raw_distances.empty:
        logger.error("Input distance DataFrame is empty or None. Cannot assign initial states.")
        return df_raw_distances.copy() # Return a copy

    if closed_ref_dist >= open_ref_dist:
        logger.warning(
            f"Closed reference distance ({closed_ref_dist:.2f}) is not less than open "
            f"reference distance ({open_ref_dist:.2f}). State assignment might be unreliable."
            f" Using midpoint anyway."
        )
        # Proceed, but the threshold might not be meaningful

    df_with_states = df_raw_distances.copy()
    distance_cols = [col for col in df_with_states.columns if col.startswith(dist_col_prefix)]

    if not distance_cols:
        logger.error(f"No distance columns found with prefix '{dist_col_prefix}'. Cannot assign states.")
        return df_with_states # Return the copy without state columns

    # Calculate the midpoint threshold
    threshold = (closed_ref_dist + open_ref_dist) / 2.0
    logger.info(f"Assigning initial states using midpoint threshold: {threshold:.3f} Å "
                f"(Closed Ref: {closed_ref_dist:.2f}, Open Ref: {open_ref_dist:.2f})")

    for dist_col in distance_cols:
        try:
            # Extract chain identifier from column name (e.g., "Dist_ChainA" -> "ChainA")
            chain_id_part = dist_col[len(dist_col_prefix):]
            state_col_name = f"{chain_id_part}{state_col_suffix}" # e.g., "ChainA_state_raw"

            # Apply threshold: closed if <= threshold, open if > threshold
            # Handle NaNs: Assign NaN state if distance is NaN
            df_with_states[state_col_name] = np.where(
                df_with_states[dist_col].isna(),
                np.nan,
                np.where(df_with_states[dist_col] <= threshold, CLOSED_STATE, OPEN_STATE)
            )
            logger.debug(f"Assigned initial states to column '{state_col_name}'.")

        except Exception as e:
            logger.error(f"Error assigning initial state for column '{dist_col}': {e}", exc_info=True)
            # Optionally, fill the state column with NaN or skip adding it on error
            # For now, let's continue to process other columns if possible

    # Log summary of assigned states for one column if possible
    if distance_cols:
         example_state_col = f"{distance_cols[0][len(dist_col_prefix):]}{state_col_suffix}"
         if example_state_col in df_with_states:
             logger.debug(f"Initial state counts for {example_state_col}:\n"
                          f"{df_with_states[example_state_col].value_counts(dropna=False)}")

    return df_with_states

# Placeholder for state detection functions 