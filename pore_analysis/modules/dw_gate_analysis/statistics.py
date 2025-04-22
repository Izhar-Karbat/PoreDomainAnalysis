"""Functions for DW gate statistical analysis."""

# This file will contain calculate_dw_statistics 

import logging
from typing import Dict, Any
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, kruskal, mannwhitneyu
from itertools import combinations

# Define state names consistently (required for calculations)
CLOSED_STATE = "closed"
OPEN_STATE = "open"

logger = logging.getLogger(__name__)


def calculate_dw_statistics(events_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Performs statistical analysis on the DW gate events data.
    Calculates summary stats, open/closed probabilities, and runs significance tests.

    Args:
        events_df (pd.DataFrame): DataFrame containing processed event data
                                 (columns: chain, state, duration_ns, etc.).

    Returns:
        Dict[str, Any]: A dictionary containing the statistical results:
            - 'summary_stats_df': DataFrame with count, mean, stddev, etc. per chain/state.
            - 'probability_df': DataFrame with open/closed probability per chain.
            - 'chi2_test': Dict with {'statistic', 'p_value', 'dof'} for state vs chain test.
            - 'kruskal_test': Dict with {'statistic', 'p_value'} for open duration vs chain test.
            - 'mannwhitney_tests_df': DataFrame with pairwise Mann-Whitney U test results.
            - Contains keys like 'summary_stats_error', 'probability_error', etc. on failure.
            - May contain 'Error': 'No events data' if input events_df is empty.
    """
    if events_df is None or events_df.empty:
        logger.error("Cannot run statistics: Events data frame not available.")
        return {'Error': 'No events data'}

    logger.info("Running statistical analysis on DW-gate events...")
    stats_results = {} # Initialize dict for results

    # --- 1. Summary Statistics ---
    summary_df = pd.DataFrame() # Initialize empty
    try:
        stats_df = events_df.groupby(['chain', 'state'])['duration_ns'].agg(
            Count='size',
            Mean_ns='mean',
            Std_Dev_ns='std',
            Median_ns='median',
            Min_ns='min',
            Max_ns='max',
            Total_Duration_ns='sum'
        ).reset_index()
        # Fill NaN for Std_Dev if only one event exists
        stats_df['Std_Dev_ns'] = stats_df['Std_Dev_ns'].fillna(0)
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
        stats_results['summary_stats_df'] = pd.DataFrame() # Ensure key exists as empty DF


    # --- 2. Open/Closed Probability and Chi-squared Test ---
    prob_df = pd.DataFrame() # Initialize empty
    time_sum_df = None
    try:
        # Use the calculated summary_df if available and valid
        if not summary_df.empty:
             if all(col in summary_df.columns for col in ['chain', 'state', 'Total_Duration_ns']):
                 try:
                     time_sum_df = summary_df.pivot(index='chain', columns='state', values='Total_Duration_ns').fillna(0)
                 except Exception as pivot_e:
                      logger.warning(f"Pivoting summary_stats_df failed: {pivot_e}. Recalculating time sums.")
                      time_sum_df = None # Force recalculation
             else:
                 logger.warning("Summary stats DF missing expected columns for pivot. Recalculating time sums.")
                 time_sum_df = None # Force recalculation
        else:
             logger.warning("Summary stats DF not available or empty. Recalculating time sums.")
             time_sum_df = None # Force recalculation

        if time_sum_df is None:
            logger.debug("Recalculating time sums for probability directly from events_df.")
            # Ensure events_df is not empty before grouping
            if not events_df.empty:
                time_sum_df = events_df.groupby(['chain', 'state'])['duration_ns'].sum().unstack(fill_value=0)
            else:
                logger.error("Cannot recalculate time sums as events_df is empty.")
                raise ValueError("Cannot calculate probabilities, events_df is empty.") # Raise error to be caught below

        # Ensure both 'open' and 'closed' columns exist after pivot/grouping
        if OPEN_STATE not in time_sum_df.columns: time_sum_df[OPEN_STATE] = 0.0
        if CLOSED_STATE not in time_sum_df.columns: time_sum_df[CLOSED_STATE] = 0.0
        # Reorder columns for consistency
        time_sum_df = time_sum_df[[CLOSED_STATE, OPEN_STATE]]

        total_time_per_chain = time_sum_df.sum(axis=1)

        # --- Debug Logging Added ---
        logger.debug(f"time_sum_df before probability calculation:\n{time_sum_df.to_string()}")
        logger.debug(f"total_time_per_chain before probability calculation:\n{total_time_per_chain.to_string()}")
        # --- End Debug Logging ---

        # Avoid division by zero if a chain had no events
        # Check if total_time_per_chain contains any zeros before dividing
        if (total_time_per_chain == 0).any():
            logger.warning("Found chains with zero total time. Probabilities for these chains will be NaN/0.")
            # Replace 0 with NaN to avoid division error but allow fillna(0) later
            total_time_per_chain_div = total_time_per_chain.replace(0, np.nan)
            prob_df_calc = time_sum_df.divide(total_time_per_chain_div, axis=0).fillna(0)
        else:
            prob_df_calc = time_sum_df.divide(total_time_per_chain, axis=0).fillna(0)

        prob_df_calc['total_time_ns'] = total_time_per_chain # Add original total time
        prob_df_calc.reset_index(inplace=True)
        stats_results['probability_df'] = prob_df_calc
        prob_df = prob_df_calc # Assign for chi2 test use

        # Chi-squared test on the time sums (contingency table)
        # Use time_sum_df directly
        if time_sum_df.shape[0] >= 2 and time_sum_df.shape[1] >= 2 and time_sum_df.gt(0).any().any(): # Check for >0 values
             try:
                 chi2_stat, p_chi2, dof, expected = chi2_contingency(time_sum_df.values)
                 stats_results['chi2_test'] = {'statistic': chi2_stat, 'p_value': p_chi2, 'dof': dof}
                 logger.info(f"Chi-squared test on state durations across chains: chi2={chi2_stat:.3f}, p={p_chi2:.4g}")
             except ValueError as ve:
                 logger.warning(f"Chi-squared test computation failed (likely low counts): {ve}. Storing NaN.")
                 stats_results['chi2_test'] = {'statistic': np.nan, 'p_value': np.nan, 'dof': np.nan, 'error': str(ve)}
        else:
             logger.warning("Skipping Chi-squared test: Insufficient data dimensions or variance (or only zeros).")
             stats_results['chi2_test'] = {'statistic': np.nan, 'p_value': np.nan, 'dof': np.nan, 'skipped': True}

    except Exception as e:
        logger.error(f"Failed to calculate probabilities or run Chi-squared test: {e}", exc_info=True)
        stats_results['probability_error'] = str(e)
        stats_results['probability_df'] = pd.DataFrame() # Ensure key exists as empty DF
        stats_results.setdefault('chi2_test', {})['error'] = str(e) # Add error if chi2 dict exists


    # --- 3. Kruskal-Wallis Test (Compare Open Durations Across Chains) ---
    try:
        # Use events_df passed to function
        chains = sorted(events_df['chain'].unique())
        open_duration_lists = [
            events_df.loc[(events_df['state'] == OPEN_STATE) & (events_df['chain'] == c), 'duration_ns'].values
            for c in chains
        ]
        valid_open_lists = [lst for lst in open_duration_lists if len(lst) > 1] # Require >1 event

        if len(valid_open_lists) >= 2: # Need at least two groups
            h_stat, p_kruskal = kruskal(*valid_open_lists)
            stats_results['kruskal_test'] = {'statistic': h_stat, 'p_value': p_kruskal}
            logger.info(f"Kruskal-Wallis test on OPEN durations across chains: H={h_stat:.3f}, p={p_kruskal:.4g}")
        else:
            logger.warning("Skipping Kruskal-Wallis test: Need open events (>1) in at least two chains.")
            stats_results['kruskal_test'] = {'statistic': np.nan, 'p_value': np.nan, 'skipped': True}
    except Exception as e:
        logger.error(f"Failed to run Kruskal-Wallis test: {e}", exc_info=True)
        stats_results['kruskal_test'] = {'statistic': np.nan, 'p_value': np.nan, 'error': str(e)}


    # --- 4. Pairwise Mann-Whitney U Tests (Compare Open Durations Between Chain Pairs) ---
    comparisons = []
    mw_df = pd.DataFrame() # Initialize empty
    try:
        chains = sorted(events_df['chain'].unique())
        if len(chains) >= 2:
            for chain_a, chain_b in combinations(chains, 2):
                durations_a = events_df.loc[(events_df['state'] == OPEN_STATE) & (events_df['chain'] == chain_a), 'duration_ns']
                durations_b = events_df.loc[(events_df['state'] == OPEN_STATE) & (events_df['chain'] == chain_b), 'duration_ns']

                # Perform test only if BOTH chains have at least one open event
                if not durations_a.empty and not durations_b.empty:
                     try:
                         u_stat, p_mannwhitney = mannwhitneyu(durations_a, durations_b, alternative='two-sided')
                         comparisons.append({
                             'Chain 1': chain_a,
                             'Chain 2': chain_b,
                             'U-statistic': u_stat,
                             'p-value': p_mannwhitney
                         })
                     except ValueError as ve:
                          logger.warning(f"Mann-Whitney U test failed for pair ({chain_a}, {chain_b}): {ve}. Recording NaN.")
                          comparisons.append({
                             'Chain 1': chain_a, 'Chain 2': chain_b,
                             'U-statistic': np.nan, 'p-value': np.nan
                         })
                # else: Skip pair

        if comparisons:
            mw_df = pd.DataFrame(comparisons)
            num_comparisons = len(mw_df)
            if num_comparisons > 0:
                # Apply Bonferroni correction only to non-NaN p-values
                mw_df['p-value (Bonferroni)'] = mw_df['p-value'].apply(
                    lambda p: min(p * num_comparisons, 1.0) if pd.notna(p) else np.nan
                )
            else:
                mw_df['p-value (Bonferroni)'] = np.nan # Should not happen if comparisons exist

            stats_results['mannwhitney_tests_df'] = mw_df
            logger.info(f"Performed {num_comparisons} pairwise Mann-Whitney U tests on open durations.")
        else:
             logger.warning("Skipping pairwise Mann-Whitney U tests: Need at least two chains with open events or no valid comparisons.")
             stats_results['mannwhitney_tests_df'] = pd.DataFrame() # Ensure key exists as empty DF

    except Exception as e:
        logger.error(f"Failed to run Mann-Whitney U tests: {e}", exc_info=True)
        stats_results['mannwhitney_tests_error'] = str(e)
        stats_results['mannwhitney_tests_df'] = pd.DataFrame() # Ensure key exists as empty DF

    logger.info("Statistical analysis complete. Returning raw results.") # Reverted log message
    return stats_results 