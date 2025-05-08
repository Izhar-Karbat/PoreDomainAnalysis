"""
Comparative analysis module for cross-analysis suite.

This module implements statistical tests and comparisons between toxin-bound
and control systems based on extracted metrics.
"""

import os
import sqlite3
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

class ComparativeAnalysis:
    """
    Performs statistical tests and comparisons between toxin-bound and control systems.
    """
    
    def __init__(self, meta_db_path: str, output_dir: str):
        """
        Initialize the comparative analysis module.
        
        Args:
            meta_db_path: Path to the meta-database file
            output_dir: Directory for saving plots and results
        """
        self.meta_db_path = meta_db_path
        self.output_dir = output_dir
        self.meta_conn = None
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Set Seaborn style for plots
        sns.set(style="whitegrid")
        
    def connect(self) -> bool:
        """
        Connects to the meta-database.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self.meta_conn = sqlite3.connect(self.meta_db_path)
            self.meta_conn.row_factory = sqlite3.Row
            logger.info(f"Connected to meta-database: {self.meta_db_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to meta-database: {e}")
            return False
    
    def get_metrics_dataframe(self, metric_names: Optional[List[str]] = None, 
                             categories: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Retrieves metrics as a pandas DataFrame for analysis.
        
        Args:
            metric_names: Optional list of specific metric names to include
            categories: Optional list of metric categories to include
        
        Returns:
            pd.DataFrame: DataFrame with metrics data
        """
        if not self.meta_conn:
            logger.error("Meta-database not connected")
            return pd.DataFrame()
        
        cursor = self.meta_conn.cursor()
        
        # Build the query based on parameters
        query = """
            SELECT s.system_id, s.system_name, s.system_type, s.run_name, 
                   m.metric_name, m.metric_category, m.value, m.units
            FROM systems s
            JOIN aggregated_metrics m ON s.system_id = m.system_id
            WHERE s.is_included = 1
        """
        params = []
        
        if metric_names:
            placeholders = ', '.join(['?'] * len(metric_names))
            query += f" AND m.metric_name IN ({placeholders})"
            params.extend(metric_names)
        
        if categories:
            placeholders = ', '.join(['?'] * len(categories))
            query += f" AND m.metric_category IN ({placeholders})"
            params.extend(categories)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        # Convert to DataFrame
        data = []
        for row in rows:
            data.append({
                'system_id': row['system_id'],
                'system_name': row['system_name'],
                'system_type': row['system_type'],
                'run_name': row['run_name'],
                'metric_name': row['metric_name'],
                'category': row['metric_category'],
                'value': row['value'],
                'units': row['units']
            })
        
        return pd.DataFrame(data)
    
    def compare_metric(self, metric_name: str) -> Dict[str, Any]:
        """
        Compare a single metric between toxin and control groups.
        
        Args:
            metric_name: Name of the metric to compare
        
        Returns:
            Dictionary with test results
        """
        try:
            df = self.get_metrics_dataframe(metric_names=[metric_name])
            
            if df.empty:
                logger.warning(f"No data found for metric: {metric_name}")
                return {'status': 'error', 'message': 'No data found'}
            
            # Split by group
            toxin_df = df[df['system_type'] == 'toxin']
            control_df = df[df['system_type'] == 'control']
            
            if toxin_df.empty or control_df.empty:
                logger.warning(f"Missing data for one of the groups for metric: {metric_name}")
                return {'status': 'error', 'message': 'Missing data for one group'}
            
            # Get values
            toxin_values = toxin_df['value'].values
            control_values = control_df['value'].values
            
            # Check that values are valid for statistical analysis
            try:
                if np.isnan(toxin_values).any() or np.isnan(control_values).any():
                    logger.warning(f"NaN values found for metric: {metric_name}")
                    return {'status': 'error', 'message': 'NaN values found'}
            except TypeError:
                # Handle case where isnan is not supported for the data type
                logger.warning(f"Non-numeric data type found for metric: {metric_name}")
                return {'status': 'error', 'message': 'Non-numeric data type'}
                
            # Ensure values are numeric
            try:
                toxin_values = toxin_values.astype(float)
                control_values = control_values.astype(float)
            except (ValueError, TypeError):
                logger.warning(f"Non-numeric values found for metric: {metric_name}")
                return {'status': 'error', 'message': 'Non-numeric values found'}
            
            # Get units (assuming same units for the same metric)
            units = df['units'].iloc[0] if not df['units'].iloc[0] is None else ''
            
            # Basic statistics
            toxin_mean = np.mean(toxin_values)
            toxin_std = np.std(toxin_values)
            control_mean = np.mean(control_values)
            control_std = np.std(control_values)
            
            # Perform statistical test (Mann-Whitney U test - nonparametric)
            u_stat, p_value = stats.mannwhitneyu(toxin_values, control_values, alternative='two-sided')
            
            # Calculate effect size (Cohen's d)
            pooled_std = np.sqrt(((len(toxin_values) - 1) * toxin_std**2 + 
                                (len(control_values) - 1) * control_std**2) / 
                               (len(toxin_values) + len(control_values) - 2))
            effect_size = np.abs(toxin_mean - control_mean) / pooled_std if pooled_std > 0 else 0
            
            # Prepare result
            result = {
                'status': 'success',
                'metric_name': metric_name,
                'units': units,
                'toxin_mean': toxin_mean,
                'toxin_std': toxin_std,
                'toxin_n': len(toxin_values),
                'control_mean': control_mean,
                'control_std': control_std,
                'control_n': len(control_values),
                'difference': toxin_mean - control_mean,
                'percent_change': ((toxin_mean - control_mean) / control_mean * 100) if control_mean != 0 else 0,
                'test_name': 'Mann-Whitney U',
                'test_statistic': u_stat,
                'p_value': p_value,
                'effect_size': effect_size,
                'effect_size_name': "Cohen's d",
                'is_significant': p_value < 0.05,
                'effect_magnitude': 'large' if effect_size > 0.8 else 'medium' if effect_size > 0.5 else 'small' if effect_size > 0.2 else 'negligible'
            }
        except Exception as e:
            logger.error(f"Error comparing metric {metric_name}: {e}")
            return {'status': 'error', 'message': str(e)}
        
        # Convert NumPy types to Python types for JSON serialization
        serializable_result = self._numpy_to_python_types(result)
        
        # Store result in database
        self._store_analysis_result(
            analysis_name=f"Comparison of {metric_name}",
            analysis_type="Mann-Whitney U test",
            p_value=float(p_value),  # Ensure p_value is a Python float
            effect_size=float(effect_size),  # Ensure effect_size is a Python float
            details=json.dumps(serializable_result)
        )
        
        return result
    
    def _numpy_to_python_types(self, obj):
        """
        Convert NumPy types to standard Python types for JSON serialization.
        
        Args:
            obj: Object to convert
            
        Returns:
            Object with NumPy types converted to standard Python types
        """
        if isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8,
                           np.int16, np.int32, np.int64, np.uint8,
                           np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.bool_) or (isinstance(obj, bool) and not isinstance(obj, int)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._numpy_to_python_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._numpy_to_python_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._numpy_to_python_types(item) for item in obj)
        else:
            return obj
            
    def _store_analysis_result(self, analysis_name: str, analysis_type: str, 
                              p_value: float, effect_size: float, 
                              details: str) -> int:
        """
        Store analysis result in the database.
        
        Args:
            analysis_name: Name of the analysis
            analysis_type: Type of statistical test or comparison
            p_value: Statistical significance
            effect_size: Effect size measure
            details: JSON string with detailed results
        
        Returns:
            int: ID of the inserted row
        """
        if not self.meta_conn:
            logger.error("Meta-database not connected")
            return -1
        
        try:
            # Make sure p_value and effect_size are Python floats
            p_value = float(p_value) if p_value is not None else 0.0
            effect_size = float(effect_size) if effect_size is not None else 0.0
            
            # Ensure details is a valid JSON string
            if not isinstance(details, str):
                logger.warning(f"Details is not a string: {type(details)}. Converting to JSON.")
                try:
                    # Try to convert to JSON if it's not already a string
                    details_obj = self._numpy_to_python_types(details)
                    details = json.dumps(details_obj)
                except Exception as e:
                    logger.error(f"Failed to convert details to JSON: {e}")
                    details = json.dumps({"error": "Failed to serialize details"})
            
            cursor = self.meta_conn.cursor()
            cursor.execute(
                """INSERT INTO comparative_analysis 
                   (analysis_name, analysis_type, p_value, effect_size, details, timestamp) 
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (analysis_name, analysis_type, p_value, effect_size, 
                 details, datetime.now().isoformat())
            )
            
            self.meta_conn.commit()
            return cursor.lastrowid
        except Exception as e:
            logger.error(f"Error storing analysis result: {e}")
            return -1
    
    def plot_metric_comparison(self, metric_name: str, 
                              plot_type: str = 'boxplot') -> Optional[str]:
        """
        Create a comparison plot for a metric.
        
        Args:
            metric_name: Name of the metric to plot
            plot_type: Type of plot ('boxplot', 'violin', 'bar', 'pointplot')
        
        Returns:
            str: Path to the saved plot file, or None if failed
        """
        df = self.get_metrics_dataframe(metric_names=[metric_name])
        
        if df.empty:
            logger.warning(f"No data found for metric: {metric_name}")
            return None
        
        # Get units
        units = df['units'].iloc[0] if not df['units'].iloc[0] is None else ''
        units_str = f" ({units})" if units else ""
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Plot based on type
        if plot_type == 'boxplot':
            ax = sns.boxplot(x='system_type', y='value', hue='system_type', data=df, palette='Set2', legend=False)
            sns.stripplot(x='system_type', y='value', data=df, color='black', alpha=0.5, jitter=True)
        elif plot_type == 'violin':
            ax = sns.violinplot(x='system_type', y='value', hue='system_type', data=df, palette='Set2', inner='point', legend=False)
            sns.stripplot(x='system_type', y='value', data=df, color='black', alpha=0.5, jitter=True)
        elif plot_type == 'bar':
            ax = sns.barplot(x='system_type', y='value', hue='system_type', data=df, palette='Set2', 
                           errorbar=('ci', 95), legend=False)
        elif plot_type == 'pointplot':
            ax = sns.pointplot(x='system_type', y='value', hue='system_type', data=df, palette='Set2', 
                             errorbar=('ci', 95), dodge=True, legend=False)
        else:
            logger.warning(f"Unknown plot type: {plot_type}, defaulting to boxplot")
            ax = sns.boxplot(x='system_type', y='value', hue='system_type', data=df, palette='Set2', legend=False)
        
        # Add annotations
        plt.xlabel('System Type')
        plt.ylabel(f'{metric_name}{units_str}')
        plt.title(f'Comparison of {metric_name} between Toxin and Control Systems')
        
        # Add statistical annotation if possible
        try:
            result = self.compare_metric(metric_name)
            if result['status'] == 'success':
                if result['is_significant']:
                    p_text = f"p = {result['p_value']:.4f} *"
                else:
                    p_text = f"p = {result['p_value']:.4f} (n.s.)"
                
                # Add text annotation
                y_pos = df['value'].max() * 1.05
                plt.text(0.5, y_pos, p_text, ha='center', fontsize=12)
        except Exception as e:
            logger.warning(f"Failed to add statistical annotation: {e}")
        
        # Save the plot
        plot_filename = f"{metric_name.replace(' ', '_')}_comparison_{plot_type}.png"
        plot_path = os.path.join(self.output_dir, plot_filename)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()
        
        # Register the plot in the database
        if self.meta_conn:
            cursor = self.meta_conn.cursor()
            cursor.execute(
                """INSERT INTO comparative_plots 
                   (plot_name, plot_type, category, subcategory, relative_path, timestamp) 
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (f"Comparison of {metric_name}", plot_type, "metric_comparison", 
                 metric_name, os.path.relpath(plot_path, os.path.dirname(self.meta_db_path)), 
                 datetime.now().isoformat())
            )
            self.meta_conn.commit()
        
        logger.info(f"Saved {plot_type} plot for {metric_name} to {plot_path}")
        return plot_path
    
    def plot_correlation_matrix(self, metric_names: List[str], 
                               split_by_type: bool = True) -> Optional[str]:
        """
        Create a correlation matrix for selected metrics.
        
        Args:
            metric_names: List of metrics to include
            split_by_type: Whether to create separate matrices for toxin/control
        
        Returns:
            str: Path to the saved plot file, or None if failed
        """
        df = self.get_metrics_dataframe(metric_names=metric_names)
        
        if df.empty:
            logger.warning(f"No data found for metrics: {metric_names}")
            return None
        
        # Prepare data for correlation analysis
        pivot_df = df.pivot_table(
            index=['system_id', 'system_name', 'system_type', 'run_name'],
            columns=['metric_name'], 
            values='value',
            aggfunc='first'
        ).reset_index()
        
        if split_by_type:
            # Create separate correlation matrices for toxin and control
            toxin_df = pivot_df[pivot_df['system_type'] == 'toxin']
            control_df = pivot_df[pivot_df['system_type'] == 'control']
            
            # Drop non-numeric columns
            toxin_corr = toxin_df.drop(columns=['system_id', 'system_name', 'system_type', 'run_name']).corr()
            control_corr = control_df.drop(columns=['system_id', 'system_name', 'system_type', 'run_name']).corr()
            
            # Create figure with two subplots
            fig, axes = plt.subplots(1, 2, figsize=(16, 7))
            
            # Plot toxin correlation matrix
            sns.heatmap(toxin_corr, annot=True, cmap='coolwarm', fmt='.2f', 
                      linewidths=0.5, ax=axes[0], vmin=-1, vmax=1)
            axes[0].set_title('Toxin Systems Correlation Matrix')
            
            # Plot control correlation matrix
            sns.heatmap(control_corr, annot=True, cmap='coolwarm', fmt='.2f', 
                      linewidths=0.5, ax=axes[1], vmin=-1, vmax=1)
            axes[1].set_title('Control Systems Correlation Matrix')
            
            plot_filename = "correlation_matrix_split.png"
        else:
            # Create combined correlation matrix
            corr_df = pivot_df.drop(columns=['system_id', 'system_name', 'system_type', 'run_name']).corr()
            
            # Create figure
            plt.figure(figsize=(10, 8))
            
            # Plot correlation matrix
            sns.heatmap(corr_df, annot=True, cmap='coolwarm', fmt='.2f', 
                      linewidths=0.5, vmin=-1, vmax=1)
            plt.title('Correlation Matrix for Selected Metrics')
            
            plot_filename = "correlation_matrix_combined.png"
        
        # Save the plot
        plot_path = os.path.join(self.output_dir, plot_filename)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()
        
        # Register the plot in the database
        if self.meta_conn:
            cursor = self.meta_conn.cursor()
            cursor.execute(
                """INSERT INTO comparative_plots 
                   (plot_name, plot_type, category, subcategory, relative_path, timestamp) 
                   VALUES (?, ?, ?, ?, ?, ?)""",
                ("Correlation Matrix", "heatmap", "correlation", 
                 "correlation_matrix", os.path.relpath(plot_path, os.path.dirname(self.meta_db_path)), 
                 datetime.now().isoformat())
            )
            self.meta_conn.commit()
        
        logger.info(f"Saved correlation matrix plot to {plot_path}")
        return plot_path
    
    def run_batch_comparisons(self, categories: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run comparative analysis for multiple metrics based on categories.
        
        Args:
            categories: Optional list of metric categories to analyze
        
        Returns:
            Dictionary with summary of results
        """
        # Get metrics based on categories
        if categories:
            df = self.get_metrics_dataframe(categories=categories)
        else:
            df = self.get_metrics_dataframe()
        
        if df.empty:
            logger.warning("No metrics found for batch comparison")
            return {'status': 'error', 'message': 'No metrics found'}
        
        # Get unique metrics
        unique_metrics = df['metric_name'].unique()
        logger.info(f"Running batch comparisons for {len(unique_metrics)} metrics")
        
        results = []
        significant_results = []
        
        # Run comparisons for each metric
        for metric in unique_metrics:
            try:
                # Get result and ensure it's properly serializable
                result = self.compare_metric(metric)
                
                # Skip metrics that failed to compare
                if result['status'] != 'success':
                    logger.warning(f"Comparison failed for metric: {metric} - {result.get('message', 'Unknown error')}")
                    continue
                
                # Ensure all NumPy types are converted to Python types
                result = self._numpy_to_python_types(result)
                results.append(result)
                
                # Create plot for the metric
                try:
                    plot_path = self.plot_metric_comparison(metric)
                    if plot_path:
                        result['plot_path'] = plot_path
                except Exception as plot_error:
                    logger.warning(f"Failed to create plot for {metric}: {plot_error}")
                
                # Add to significant results if p < 0.05
                if result.get('is_significant', False):
                    significant_results.append(result)
            except Exception as e:
                logger.error(f"Error processing metric {metric}: {e}")
                # Continue with the next metric instead of failing entirely
                continue
        
        # Sort results by significance and effect size
        sorted_results = sorted(results, 
                               key=lambda x: (not x['is_significant'], x['p_value'], -x['effect_size']))
        
        # Create summary table (if at least 3 metrics were compared)
        if len(results) >= 3:
            self._create_summary_table(sorted_results)
        
        # Create correlation matrix (if at least 3 metrics were compared)
        if len(results) >= 3:
            metric_names = [r['metric_name'] for r in results]
            self.plot_correlation_matrix(metric_names, split_by_type=True)
        
        summary = {
            'status': 'success',
            'total_metrics': len(unique_metrics),
            'successful_comparisons': len(results),
            'significant_differences': len(significant_results),
            'results': sorted_results
        }
        
        # Convert any NumPy types to standard Python types
        serializable_summary = self._numpy_to_python_types(summary)
        
        return serializable_summary
    
    def _create_summary_table(self, results: List[Dict[str, Any]]) -> Optional[str]:
        """
        Create a summary table of comparison results.
        
        Args:
            results: List of comparison results
        
        Returns:
            str: Path to the saved plot file, or None if failed
        """
        # Create DataFrame from results
        data = []
        for r in results:
            data.append({
                'Metric': r['metric_name'],
                'Units': r['units'],
                'Toxin Mean': r['toxin_mean'],
                'Control Mean': r['control_mean'],
                'Difference': r['difference'],
                'Change (%)': r['percent_change'],
                'p-value': r['p_value'],
                'Significant': r['is_significant'],
                'Effect Size': r['effect_size'],
                'Magnitude': r['effect_magnitude']
            })
        
        df = pd.DataFrame(data)
        
        # Sort by significance and effect size
        df = df.sort_values(by=['Significant', 'p-value', 'Effect Size'], 
                           ascending=[False, True, False])
        
        # Create figure
        plt.figure(figsize=(12, len(df) * 0.5 + 2))
        
        # Prepare data for the table - handle rounding only for numeric columns
        table_data = df.copy()
        
        # Convert 'Significant' to string to avoid rounding issues
        if 'Significant' in table_data.columns:
            table_data['Significant'] = table_data['Significant'].map({True: 'Yes', False: 'No'})
            
        # Round numeric columns only
        for col in table_data.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns:
            table_data[col] = table_data[col].round(4)
            
        # Get cell colors based on significance
        if 'Significant' in df.columns:
            cell_colors = np.where(
                df['Significant'].map({True: True, False: False, 'Yes': True, 'No': False}).values[:, np.newaxis], 
                np.full((len(df), df.shape[1]), 'lightyellow'), 
                np.full((len(df), df.shape[1]), 'white')
            )
        else:
            cell_colors = np.full((len(df), df.shape[1]), 'white')
            
        # Plot as table
        table = plt.table(
            cellText=table_data.values,
            colLabels=df.columns,
            cellLoc='center',
            loc='center',
            cellColours=cell_colors
        )
        
        # Adjust table appearance
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Remove axes
        plt.axis('off')
        plt.title('Summary of Metric Comparisons between Toxin and Control Systems', 
                fontsize=14, pad=20)
        
        # Save the plot
        plot_filename = "comparison_summary_table.png"
        plot_path = os.path.join(self.output_dir, plot_filename)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Register the plot in the database
        if self.meta_conn:
            cursor = self.meta_conn.cursor()
            cursor.execute(
                """INSERT INTO comparative_plots 
                   (plot_name, plot_type, category, subcategory, relative_path, timestamp) 
                   VALUES (?, ?, ?, ?, ?, ?)""",
                ("Comparison Summary Table", "table", "summary", 
                 "comparison_table", os.path.relpath(plot_path, os.path.dirname(self.meta_db_path)), 
                 datetime.now().isoformat())
            )
            self.meta_conn.commit()
        
        logger.info(f"Saved comparison summary table to {plot_path}")
        return plot_path
    
    def close(self):
        """Closes the database connection."""
        if self.meta_conn:
            self.meta_conn.close()
            logger.info("Closed meta-database connection")