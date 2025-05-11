"""
Statistical analysis component for the enhanced report generator.

This module is responsible for analyzing data statistically and providing
significance testing between groups of metrics.
"""

import logging
import numpy as np
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

def calculate_p_value(group1: List[float], group2: List[float]) -> float:
    """
    Calculate p-value for the difference between two groups using Welch's t-test.
    
    Args:
        group1: Values from the first group
        group2: Values from the second group
        
    Returns:
        p-value (float) for the difference between groups
    """
    from scipy import stats
    if len(group1) < 2 or len(group2) < 2:
        return 1.0  # Not enough samples for a meaningful test
    try:
        t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)
        return p_value
    except Exception as e:
        logger.error(f"Error calculating p-value: {e}")
        return 1.0  # Error in calculation

def calculate_effect_size(group1: List[float], group2: List[float]) -> float:
    """
    Calculate Cohen's d effect size for the difference between two groups.
    
    Args:
        group1: Values from the first group
        group2: Values from the second group
        
    Returns:
        Cohen's d effect size (float)
    """
    if len(group1) < 2 or len(group2) < 2:
        return 0.0  # Not enough samples
    try:
        mean1, mean2 = np.mean(group1), np.mean(group2)
        std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        # Pooled standard deviation
        pooled_std = np.sqrt((std1**2 + std2**2) / 2)
        if pooled_std == 0:
            return 0.0
        return abs(mean1 - mean2) / pooled_std
    except Exception as e:
        logger.error(f"Error calculating effect size: {e}")
        return 0.0  # Error in calculation

def interpret_p_value(p: float) -> str:
    """
    Convert a p-value to a human-readable significance description.
    
    Args:
        p: p-value to interpret
        
    Returns:
        String describing the significance level
    """
    if p < 0.001:
        return "highly significant (p < 0.001)"
    elif p < 0.01:
        return f"significant (p = {p:.3f})"
    elif p < 0.05:
        return f"marginally significant (p = {p:.3f})"
    else:
        return f"not statistically significant (p = {p:.3f})"

def interpret_effect_size(d: float) -> str:
    """
    Convert an effect size value to a human-readable description.
    
    Args:
        d: Cohen's d effect size
        
    Returns:
        String describing the effect size magnitude
    """
    if d > 1.2:
        return "very large"
    elif d > 0.8:
        return "large"
    elif d > 0.5:
        return "moderate"
    elif d > 0.2:
        return "small"
    else:
        return "negligible"

def generate_statistical_report(metrics_toxin: Dict, metrics_control: Dict) -> Dict:
    """
    Generate a detailed statistical report comparing toxin and control metrics.
    
    Args:
        metrics_toxin: Dictionary of toxin metrics grouped by category
        metrics_control: Dictionary of control metrics grouped by category
        
    Returns:
        Dictionary containing p-values, effect sizes, and significance descriptions
    """
    # Make copies of the metric data
    toxin_metrics = {k: {sk: v[sk].copy() for sk in v} for k, v in metrics_toxin.items()}
    control_metrics = {k: {sk: v[sk].copy() for sk in v} for k, v in metrics_control.items()}
    
    # Calculate range values for AC subunit
    toxin_ac_range = [max_val - min_val for max_val, min_val in zip(
        toxin_metrics['AC']['max'], toxin_metrics['AC']['min'])]
    control_ac_range = [max_val - min_val for max_val, min_val in zip(
        control_metrics['AC']['max'], control_metrics['AC']['min'])]
    
    # Calculate range values for BD subunit
    toxin_bd_range = [max_val - min_val for max_val, min_val in zip(
        toxin_metrics['BD']['max'], toxin_metrics['BD']['min'])]
    control_bd_range = [max_val - min_val for max_val, min_val in zip(
        control_metrics['BD']['max'], control_metrics['BD']['min'])]
    
    # Calculate p-values
    p_values = {
        'AC': {
            'mean': calculate_p_value(toxin_metrics['AC']['mean'], control_metrics['AC']['mean']),
            'min': calculate_p_value(toxin_metrics['AC']['min'], control_metrics['AC']['min']),
            'max': calculate_p_value(toxin_metrics['AC']['max'], control_metrics['AC']['max']),
            'range': calculate_p_value(toxin_ac_range, control_ac_range)
        },
        'BD': {
            'mean': calculate_p_value(toxin_metrics['BD']['mean'], control_metrics['BD']['mean']),
            'min': calculate_p_value(toxin_metrics['BD']['min'], control_metrics['BD']['min']),
            'max': calculate_p_value(toxin_metrics['BD']['max'], control_metrics['BD']['max']),
            'range': calculate_p_value(toxin_bd_range, control_bd_range)
        }
    }
    
    # Calculate effect sizes
    effect_sizes = {
        'AC': {
            'mean': calculate_effect_size(toxin_metrics['AC']['mean'], control_metrics['AC']['mean']),
            'min': calculate_effect_size(toxin_metrics['AC']['min'], control_metrics['AC']['min']),
            'max': calculate_effect_size(toxin_metrics['AC']['max'], control_metrics['AC']['max']),
            'range': calculate_effect_size(toxin_ac_range, control_ac_range)
        },
        'BD': {
            'mean': calculate_effect_size(toxin_metrics['BD']['mean'], control_metrics['BD']['mean']),
            'min': calculate_effect_size(toxin_metrics['BD']['min'], control_metrics['BD']['min']),
            'max': calculate_effect_size(toxin_metrics['BD']['max'], control_metrics['BD']['max']),
            'range': calculate_effect_size(toxin_bd_range, control_bd_range)
        }
    }
    
    # Generate significance descriptions
    significance_notes = {
        'mean': f"The difference in mean G-G distances between toxin and control is {interpret_p_value(p_values['AC']['mean'])} for the A:C subunit pair and {interpret_p_value(p_values['BD']['mean'])} for B:D. The effect size is {interpret_effect_size(effect_sizes['AC']['mean'])} for A:C and {interpret_effect_size(effect_sizes['BD']['mean'])} for B:D.",
        'min': f"The reduction in minimum G-G distances is {interpret_p_value(p_values['AC']['min'])} for A:C and {interpret_p_value(p_values['BD']['min'])} for B:D subunit pairs, with {interpret_effect_size(effect_sizes['AC']['min'])} and {interpret_effect_size(effect_sizes['BD']['min'])} effect sizes respectively.",
        'max': f"The increase in maximum G-G distances is {interpret_p_value(p_values['AC']['max'])} for A:C and {interpret_p_value(p_values['BD']['max'])} for B:D, with {interpret_effect_size(effect_sizes['AC']['max'])} and {interpret_effect_size(effect_sizes['BD']['max'])} effect sizes respectively.",
        'range': f"The increased range in G-G distances is {interpret_p_value(p_values['AC']['range'])} for A:C and {interpret_p_value(p_values['BD']['range'])} for B:D, with {interpret_effect_size(effect_sizes['AC']['range'])} and {interpret_effect_size(effect_sizes['BD']['range'])} effect sizes respectively."
    }
    
    return {
        'p_values': p_values,
        'effect_sizes': effect_sizes,
        'significance_notes': significance_notes,
        'ranges': {
            'toxin': {
                'AC': toxin_ac_range,
                'BD': toxin_bd_range
            },
            'control': {
                'AC': control_ac_range,
                'BD': control_bd_range
            }
        }
    }