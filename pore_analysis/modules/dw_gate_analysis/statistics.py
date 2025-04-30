# filename: pore_analysis/modules/dw_gate_analysis/statistics.py
"""
Functions for DW gate statistical analysis.
Re-adapted for the refactored Pore Analysis Suite (v2).
This version calculates statistics and returns them, but does not store metrics.
Expects standardized column names as input.
"""

import logging
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from itertools import combinations

# Attempt to import statistical functions, handle missing library
try:
    from scipy.stats import chi2_contingency, kruskal, mannwhitneyu
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    def chi2_contingency(*args, **kwargs): raise ImportError("Scipy not installed.")
    def kruskal(*args, **kwargs): raise ImportError("Scipy not installed.")
    def mannwhitneyu(*args, **kwargs): raise ImportError("Scipy not installed.")
    logger = logging.getLogger(__name__)
    logger.warning("Scipy not installed. Statistical tests requiring it will be skipped.")

# Define state names consistently (required for calculations)
try:
    from .utils import CLOSED_STATE, OPEN_STATE
except ImportError:
    CLOSED_STATE = "closed"
    OPEN_STATE = "open"
    logger = logging.getLogger(__name__)
    logger.warning("Could not import state constants from .utils. Using local definitions.")

logger = logging.getLogger(__name__)


def calculate_dw_statistics(events_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Performs statistical analysis on the DW gate events data.
    Calculates summary stats, open/closed probabilities, and runs significance tests.
    Adapted for the refactored codebase - expects standardized column names
    ('Chain', 'State', 'Duration (ns)') and returns results in a dictionary.

    Args:
        events_df (pd.DataFrame): DataFrame containing processed event data
                                 (columns: 'Chain', 'State', 'Duration (ns)', etc.).

    Returns:
        Dict[str, Any]: A dictionary containing the statistical results:
            - 'summary_stats_df': DataFrame with count, mean, stddev, etc. per chain/state.
            - 'probability_df': DataFrame with open/closed probability per chain.
            - 'chi2_test': Dict with {'statistic', 'p_value', 'dof', 'skipped', 'error'} for state duration vs chain test.
            - 'kruskal_test': Dict with {'statistic', 'p_value', 'skipped', 'error'} for open duration vs chain test.
            - 'mannwhitney_tests_df': DataFrame with pairwise Mann-Whitney U test results.
            - May contain 'Error': 'Reason for failure' if critical errors occur.
    """
    if events_df is None or events_df.empty:
        logger.error("Cannot run statistics: Events data frame is empty or None.")
        return {'Error': 'No events data'}

    # --- Use Standardized Column Names ---
    chain_col = 'Chain'
    state_col = 'State'
    duration_col = 'Duration (ns)'
    required_cols = [chain_col, state_col, duration_col]
    # --- End Use Standardized ---

    if not all(col in events_df.columns for col in required_cols):
         logger.error(f"Cannot run statistics: Events DataFrame missing required columns ({required_cols}). Found: {events_df.columns.tolist()}")
         return {'Error': f'Missing required columns in events data ({required_cols})'}

    logger.info("Running statistical analysis on DW-gate events...")
    stats_results = { # Initialize dict with default/error states for results
        'summary_stats_df': pd.DataFrame(),
        'probability_df': pd.DataFrame(),
        'chi2_test': {'statistic': np.nan, 'p_value': np.nan, 'dof': np.nan, 'skipped': True, 'error': None},
        'kruskal_test': {'statistic': np.nan, 'p_value': np.nan, 'skipped': True, 'error': None},
        'mannwhitney_tests_df': pd.DataFrame(),
        'mannwhitney_tests_error': None
    }

    # --- 1. Summary Statistics ---
    summary_df = pd.DataFrame() # Local variable for intermediate use
    try:
        # Ensure duration is numeric before aggregation
        events_df[duration_col] = pd.to_numeric(events_df[duration_col], errors='coerce')
        # Drop rows where duration became NaN after coercion
        valid_events_df = events_df.dropna(subset=[duration_col]).copy() # Use .copy()

        if valid_events_df.empty:
            logger.warning("No valid numeric durations found after coercion. Cannot calculate summary stats.")
            # stats_results['summary_stats_df'] remains empty DataFrame
        else:
            # Calculate summary statistics using standardized column names
            stats_df = valid_events_df.groupby([chain_col, state_col])[duration_col].agg(
                Count='size',
                Mean_ns='mean',
                Std_Dev_ns='std',
                Median_ns='median',
                Min_ns='min',
                Max_ns='max',
                Total_Duration_ns='sum'
            ).reset_index()
            # Fill NaN for Std_Dev if only one event exists (count == 1)
            stats_df.loc[stats_df['Count'] == 1, 'Std_Dev_ns'] = 0.0
            # Explicitly fill any remaining NaNs in std dev (e.g., if all values were identical)
            stats_df['Std_Dev_ns'] = stats_df['Std_Dev_ns'].fillna(0.0)

            # Store calculated df in results and local var
            stats_results['summary_stats_df'] = stats_df
            summary_df = stats_df # Assign for later use

            if not stats_df.empty:
                 logger.debug(f"Calculated summary_stats_df head:\n{stats_df.head().to_string()}")
                 logger.debug(f"summary_stats_df shape: {stats_df.shape}")
            else:
                 logger.warning("summary_stats_df is empty after calculation.")
        logger.info("Calculated summary statistics.")
    except Exception as e:
        logger.error(f"Failed to calculate summary statistics: {e}", exc_info=True)
        stats_results['summary_stats_error'] = str(e)
        # stats_results['summary_stats_df'] remains empty DataFrame

    # --- 2. Open/Closed Probability and Chi-squared Test ---
    prob_df = pd.DataFrame() # Local variable for intermediate use
    time_sum_df = None
    try:
        # Use valid_events_df which has NaNs dropped from duration
        if not valid_events_df.empty:
             # Use standardized column names
            time_sum_df = valid_events_df.groupby([chain_col, state_col])[duration_col].sum().unstack(fill_value=0)
            # Ensure both states are columns, adding them with 0 if missing
            if OPEN_STATE not in time_sum_df.columns: time_sum_df[OPEN_STATE] = 0.0
            if CLOSED_STATE not in time_sum_df.columns: time_sum_df[CLOSED_STATE] = 0.0
            time_sum_df = time_sum_df[[CLOSED_STATE, OPEN_STATE]] # Ensure consistent order
        else:
            logger.warning("No valid events to calculate time sums for probability.")
            # Create empty DataFrame with expected columns if no valid events
            chains = events_df[chain_col].unique() # Get chains from original potentially non-empty df
            time_sum_df = pd.DataFrame(0.0, index=chains, columns=[CLOSED_STATE, OPEN_STATE])


        total_time_per_chain = time_sum_df.sum(axis=1)
        logger.debug(f"time_sum_df for probability:\n{time_sum_df.to_string()}")
        logger.debug(f"total_time_per_chain:\n{total_time_per_chain.to_string()}")

        # Avoid division by zero
        total_time_per_chain_div = total_time_per_chain.replace(0, np.nan) # Replace 0 with NaN for division
        prob_df_calc = time_sum_df.divide(total_time_per_chain_div, axis=0).fillna(0) # Fill resulting NaN with 0

        prob_df_calc['total_time_ns'] = total_time_per_chain # Add original total time

        # Use standardized column name 'Chain' - reset_index gives 'Chain'
        prob_df_calc = prob_df_calc.reset_index()
        # Rename 'index' to 'Chain' if reset_index didn't name it automatically (depends on pandas version)
        if 'index' in prob_df_calc.columns:
             prob_df_calc.rename(columns={'index': chain_col}, inplace=True)

        stats_results['probability_df'] = prob_df_calc
        prob_df = prob_df_calc # Assign for chi2 test use

        # Chi-squared test on the time sums (contingency table)
        # Use time_sum_df directly as the contingency table
        stats_results['chi2_test']['error'] = 'No duration data' # Default error
        if SCIPY_AVAILABLE and isinstance(time_sum_df, pd.DataFrame) and not time_sum_df.empty:
            # Perform test only if dimensions are valid and contains non-zero values
            if time_sum_df.shape[0] >= 2 and time_sum_df.shape[1] >= 2 and time_sum_df.gt(0).any().any():
                 try:
                     chi2_stat, p_chi2, dof, expected = chi2_contingency(time_sum_df.values)
                     stats_results['chi2_test'] = {'statistic': chi2_stat, 'p_value': p_chi2, 'dof': dof, 'skipped': False, 'error': None}
                     logger.info(f"Chi-squared test on state DURATIONS across chains: chi2={chi2_stat:.3f}, p={p_chi2:.4g}")
                 except ValueError as ve:
                     logger.warning(f"Chi-squared test computation failed (likely low counts/zeros): {ve}. Storing NaN.")
                     stats_results['chi2_test'] = {'statistic': np.nan, 'p_value': np.nan, 'dof': np.nan, 'skipped': False, 'error': str(ve)}
            else:
                 logger.warning("Skipping Chi-squared test: Insufficient data dimensions or variance (or only zeros).")
                 stats_results['chi2_test'] = {'statistic': np.nan, 'p_value': np.nan, 'dof': np.nan, 'skipped': True, 'error': 'Insufficient data dimensions/variance'}
        elif not SCIPY_AVAILABLE:
             logger.warning("Skipping Chi-squared test: Scipy not available.")
             stats_results['chi2_test'] = {'statistic': np.nan, 'p_value': np.nan, 'dof': np.nan, 'skipped': True, 'error': 'Scipy not installed'}

    except Exception as e:
        logger.error(f"Failed to calculate probabilities or run Chi-squared test: {e}", exc_info=True)
        stats_results['probability_error'] = str(e)
        stats_results['probability_df'] = pd.DataFrame() # Ensure key exists as empty DF
        stats_results.setdefault('chi2_test', {})['error'] = str(e)


    # --- 3. Kruskal-Wallis Test (Compare Open Durations Across Chains) ---
    stats_results['kruskal_test']['error'] = 'No data or Scipy missing' # Default error
    if SCIPY_AVAILABLE and 'valid_events_df' in locals() and not valid_events_df.empty: # Check if valid_events_df exists
        try:
             # Use standardized column names
            chains = sorted(valid_events_df[chain_col].unique())
            open_duration_lists = [
                valid_events_df.loc[(valid_events_df[state_col] == OPEN_STATE) & (valid_events_df[chain_col] == c), duration_col].values
                for c in chains
            ]
            # Kruskal-Wallis requires at least one value in each group being compared.
            valid_open_lists = [lst for lst in open_duration_lists if len(lst) >= 1] # Changed from >= 2 to >=1

            if len(valid_open_lists) >= 2: # Need at least two groups with data
                # Further check: ensure at least one group has variance if possible? Optional.
                h_stat, p_kruskal = kruskal(*valid_open_lists)
                stats_results['kruskal_test'] = {'statistic': h_stat, 'p_value': p_kruskal, 'skipped': False, 'error': None}
                logger.info(f"Kruskal-Wallis test on OPEN durations across chains: H={h_stat:.3f}, p={p_kruskal:.4g}")
            else:
                logger.warning("Skipping Kruskal-Wallis test: Need open events in at least two chains.")
                stats_results['kruskal_test']['skipped'] = True
                stats_results['kruskal_test']['error'] = "Need open events in at least two chains"
        except Exception as e:
            logger.error(f"Failed to run Kruskal-Wallis test: {e}", exc_info=True)
            stats_results['kruskal_test'] = {'statistic': np.nan, 'p_value': np.nan, 'skipped': False, 'error': str(e)}
    elif not SCIPY_AVAILABLE:
        logger.warning("Skipping Kruskal-Wallis test: Scipy not available.")
        stats_results['kruskal_test']['error'] = 'Scipy not installed'
    else:
         logger.warning("Skipping Kruskal-Wallis test: No valid events data available.")


    # --- 4. Pairwise Mann-Whitney U Tests (Compare Open Durations Between Chain Pairs) ---
    comparisons = []
    mw_df = pd.DataFrame() # Initialize empty
    stats_results['mannwhitney_tests_df'] = mw_df # Ensure key exists
    stats_results['mannwhitney_tests_error'] = None # Initialize error state

    if SCIPY_AVAILABLE and 'valid_events_df' in locals() and not valid_events_df.empty: # Check if valid_events_df exists
        try:
             # Use standardized column names
            chains = sorted(valid_events_df[chain_col].unique())
            if len(chains) >= 2:
                for chain_a, chain_b in combinations(chains, 2):
                    durations_a = valid_events_df.loc[(valid_events_df[state_col] == OPEN_STATE) & (valid_events_df[chain_col] == chain_a), duration_col]
                    durations_b = valid_events_df.loc[(valid_events_df[state_col] == OPEN_STATE) & (valid_events_df[chain_col] == chain_b), duration_col]

                    # Perform test only if BOTH chains have at least one open event
                    if not durations_a.empty and not durations_b.empty:
                         try:
                             # Mann-Whitney U requires > 0 variance in at least one group typically, but can run otherwise
                             u_stat, p_mannwhitney = mannwhitneyu(durations_a, durations_b, alternative='two-sided')
                             comparisons.append({
                                 'Chain 1': chain_a, 'Chain 2': chain_b,
                                 'U-statistic': u_stat, 'p-value': p_mannwhitney
                             })
                         except ValueError as ve: # Catches issues like identical data
                              logger.warning(f"Mann-Whitney U test failed for pair ({chain_a}, {chain_b}): {ve}. Recording NaN.")
                              comparisons.append({
                                 'Chain 1': chain_a, 'Chain 2': chain_b,
                                 'U-statistic': np.nan, 'p-value': np.nan
                             })
                    # else: Skip pair

            if comparisons:
                mw_df = pd.DataFrame(comparisons)
                num_valid_comparisons = len(mw_df.dropna(subset=['p-value'])) # Count only valid comparisons for correction
                if num_valid_comparisons > 0:
                    mw_df['p-value (Bonferroni)'] = mw_df['p-value'].apply(
                        lambda p: min(p * num_valid_comparisons, 1.0) if pd.notna(p) else np.nan
                    )
                else:
                    mw_df['p-value (Bonferroni)'] = np.nan # Assign NaN if no valid p-values

                stats_results['mannwhitney_tests_df'] = mw_df
                logger.info(f"Performed {len(comparisons)} pairwise Mann-Whitney U tests on open durations (corrected for {num_valid_comparisons} valid tests).")
            else:
                 logger.warning("Skipping pairwise Mann-Whitney U tests: No valid pairs with open events found.")
                 # stats_results['mannwhitney_tests_df'] remains empty DataFrame

        except Exception as e:
            logger.error(f"Failed to run Mann-Whitney U tests: {e}", exc_info=True)
            stats_results['mannwhitney_tests_error'] = str(e)
            # stats_results['mannwhitney_tests_df'] remains empty DataFrame
    elif not SCIPY_AVAILABLE:
        logger.warning("Skipping Mann-Whitney U tests: Scipy not available.")
        stats_results['mannwhitney_tests_error'] = 'Scipy not installed'
    else:
         logger.warning("Skipping Mann-Whitney U tests: No valid events data available.")


    logger.info("Statistical analysis complete.")
    return stats_results