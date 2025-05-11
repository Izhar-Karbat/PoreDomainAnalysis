"""
Data access component for the enhanced report generator.

This module is responsible for:
1. Loading data from the aggregated enhanced_cross_analysis.db database
2. Comparing metrics between toxin and control groups
3. Providing data structures for report generation
"""

import sqlite3
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from scipy import stats

from ..core.data_models import MetricResult, RunMetadata

logger = logging.getLogger(__name__)


class DataRepository:
    """
    Provides access to the aggregated database and methods for data analysis.
    """
    
    def __init__(self, db_path: Path):
        """
        Initialize the DataRepository with the path to the aggregated database.
        
        Args:
            db_path: Path to the enhanced_cross_analysis.db file
        """
        self.db_path = db_path
        if not self.db_path.exists():
            raise FileNotFoundError(f"Aggregated database not found: {self.db_path}. Please run DataAggregator first.")
        logger.info(f"DataRepository initialized with DB: {self.db_path}")

    def _get_system_ids_by_group(self, group_type: str) -> List[int]:
        """
        Helper to get system_ids for a given group_type.
        
        Args:
            group_type: Type of group ('toxin-bound' or 'toxin-free')
            
        Returns:
            List of system IDs for the specified group
        """
        ids = []
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT system_id FROM systems WHERE group_type = ?", (group_type,))
            ids = [row['system_id'] for row in cursor.fetchall()]
            conn.close()
        except sqlite3.Error as e:
            logger.error(f"Database error while fetching system IDs for group {group_type}: {e}")
        return ids

    def load_run_metadata(self) -> RunMetadata:
        """
        Loads high-level metadata about the compared runs from the aggregated DB.
        
        Returns:
            RunMetadata object containing information about the runs
        """
        logger.info(f"Loading run metadata from {self.db_path}")
        control_run_names: List[str] = []
        toxin_run_names: List[str] = []
        toxin_name_common = None
        channel_name_common = None

        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Get control run names
            cursor.execute("SELECT run_name FROM systems WHERE group_type = 'toxin-free'")
            control_run_names = [row['run_name'] for row in cursor.fetchall()]

            # Get toxin run names
            cursor.execute("SELECT run_name FROM systems WHERE group_type = 'toxin-bound'")
            toxin_run_names = [row['run_name'] for row in cursor.fetchall()]
            
            # Try to get toxin_name and channel_name (if available)
            # This is a simplified approach for the MVP
            if toxin_run_names:
                # Attempt to extract toxin name from run name
                toxin_name_parts = [name.split('_') for name in toxin_run_names if 'toxin' in name.lower()]
                if toxin_name_parts:
                    # Look for common elements in the names that might be the toxin identifier
                    common_parts = set.intersection(*[set(parts) for parts in toxin_name_parts])
                    toxin_candidates = [part for part in common_parts if 'toxin' in part.lower()]
                    if toxin_candidates:
                        toxin_name_common = toxin_candidates[0]

            conn.close()
        except sqlite3.Error as e:
            logger.error(f"Database error loading run metadata: {e}")

        return RunMetadata(
            control_run_ids=control_run_names,
            toxin_run_ids=toxin_run_names,
            toxin_name=toxin_name_common or "UnknownToxin",
            channel_name=channel_name_common or "UnknownChannel",
            simulation_parameters={}  # Could be aggregated if consistent
        )

    def load_metrics_for_stage1_overview(self) -> List[MetricResult]:
        """
        Loads high-level summary data for the Stage 1 "System Overview" tab.
        This doesn't perform toxin vs control comparison but gives simple counts and averages.
        
        Returns:
            List of MetricResult objects with high-level metrics
        """
        logger.info(f"Loading high-level metrics for Stage 1 overview from {self.db_path}")
        overview_metrics = []
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Count of toxin-bound systems
            cursor.execute("SELECT COUNT(*) FROM systems WHERE group_type = 'toxin-bound'")
            num_toxin_systems = cursor.fetchone()[0]
            overview_metrics.append(MetricResult(
                name="Number of Toxin-Bound Systems", 
                value_toxin=float(num_toxin_systems), 
                value_control=0, 
                units="systems", 
                difference=0)
            )

            # Count of toxin-free (control) systems
            cursor.execute("SELECT COUNT(*) FROM systems WHERE group_type = 'toxin-free'")
            num_control_systems = cursor.fetchone()[0]
            overview_metrics.append(MetricResult(
                name="Number of Toxin-Free (Control) Systems", 
                value_toxin=0, 
                value_control=float(num_control_systems), 
                units="systems", 
                difference=0)
            )
            
            # Average trajectory length for toxin systems
            cursor.execute("SELECT AVG(trajectory_total_frames) FROM systems WHERE group_type = 'toxin-bound'")
            avg_frames_toxin_row = cursor.fetchone()
            if avg_frames_toxin_row and avg_frames_toxin_row[0] is not None:
                overview_metrics.append(MetricResult(
                    name="Avg. Trajectory Frames (Toxin)", 
                    value_toxin=float(avg_frames_toxin_row[0]), 
                    value_control=0, 
                    units="frames", 
                    difference=0)
                )

            # Average trajectory length for control systems
            cursor.execute("SELECT AVG(trajectory_total_frames) FROM systems WHERE group_type = 'toxin-free'")
            avg_frames_control_row = cursor.fetchone()
            if avg_frames_control_row and avg_frames_control_row[0] is not None:
                overview_metrics.append(MetricResult(
                    name="Avg. Trajectory Frames (Control)", 
                    value_toxin=0, 
                    value_control=float(avg_frames_control_row[0]), 
                    units="frames", 
                    difference=0)
                )

            # Total number of metrics collected
            cursor.execute("SELECT COUNT(DISTINCT metric_name) FROM aggregated_metrics")
            total_metrics = cursor.fetchone()[0]
            overview_metrics.append(MetricResult(
                name="Total Unique Metrics Collected", 
                value_toxin=float(total_metrics), 
                value_control=float(total_metrics), 
                units="metrics", 
                difference=0)
            )

            conn.close()
        except sqlite3.Error as e:
            logger.error(f"Database error loading overview metrics: {e}")
        
        return overview_metrics

    def load_and_compare_metrics(self) -> List[MetricResult]:
        """
        Fetches metrics from the aggregated DB, performs comparisons between
        toxin and control groups, and returns a list of MetricResult objects.
        This is for Stage 2 and detailed reports.
        
        Returns:
            List of MetricResult objects with comparison data
        """
        logger.info(f"Loading and comparing metrics from {self.db_path}")
        compared_metrics: List[MetricResult] = []

        toxin_system_ids = self._get_system_ids_by_group('toxin-bound')
        control_system_ids = self._get_system_ids_by_group('toxin-free')

        if not toxin_system_ids or not control_system_ids:
            logger.warning("Not enough systems in both toxin and control groups for comparison. Aborting metric comparison.")
            return []

        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Get all unique metric names
            cursor.execute("SELECT DISTINCT metric_name, units FROM aggregated_metrics")
            unique_metric_specs = [(row['metric_name'], row['units']) for row in cursor.fetchall()]

            for metric_name, units in unique_metric_specs:
                # Fetch values for toxin group
                cursor.execute(f"""
                    SELECT value FROM aggregated_metrics
                    WHERE metric_name = ? AND system_id IN ({','.join(['?']*len(toxin_system_ids))})
                """, (metric_name, *toxin_system_ids))
                toxin_values = [row['value'] for row in cursor.fetchall() if row['value'] is not None]

                # Fetch values for control group
                cursor.execute(f"""
                    SELECT value FROM aggregated_metrics
                    WHERE metric_name = ? AND system_id IN ({','.join(['?']*len(control_system_ids))})
                """, (metric_name, *control_system_ids))
                control_values = [row['value'] for row in cursor.fetchall() if row['value'] is not None]

                if not toxin_values or not control_values:
                    logger.debug(f"Skipping metric '{metric_name}': not enough data in one or both groups for comparison.")
                    continue

                mean_toxin = np.mean(toxin_values)
                mean_control = np.mean(control_values)
                difference = mean_toxin - mean_control
                
                p_value = None
                significant = False
                effect_size = None
                
                # Perform statistical tests if enough samples
                if len(toxin_values) > 1 and len(control_values) > 1:
                    # Welch's t-test (assumes unequal variances by default)
                    ttest_result = stats.ttest_ind(toxin_values, control_values, equal_var=False, nan_policy='omit')
                    p_value = ttest_result.pvalue
                    
                    # Significance threshold (configurable)
                    if p_value < 0.05:  # placeholder
                        significant = True
                    
                    # Calculate effect size (Cohen's d)
                    pooled_std = np.sqrt((np.std(toxin_values, ddof=1)**2 + np.std(control_values, ddof=1)**2) / 2)
                    if pooled_std > 0:
                        effect_size = abs(mean_toxin - mean_control) / pooled_std
                else:
                    logger.debug(f"Skipping p-value calculation for '{metric_name}' due to insufficient sample size in one/both groups.")

                compared_metrics.append(
                    MetricResult(
                        name=metric_name,
                        value_control=float(mean_control),
                        value_toxin=float(mean_toxin),
                        units=units or "",
                        difference=float(difference),
                        p_value=float(p_value) if p_value is not None else None,
                        significant=significant,
                        effect_size=float(effect_size) if effect_size is not None else None,
                        # description and priority_score to be filled later by AnalysisFilter/SectionBuilder
                    )
                )
            conn.close()
        except sqlite3.Error as e:
            logger.error(f"Database error during metric loading and comparison: {e}")
        except Exception as general_exception:
            logger.error(f"General error during metric loading and comparison: {general_exception}")

        if not compared_metrics:
            logger.warning("No metrics were successfully compared. Detailed report might be empty.")
        else:
            logger.info(f"Successfully compared {len(compared_metrics)} unique metrics.")
        return compared_metrics