"""
Analysis selection component for the enhanced report generator.

This module is responsible for:
1. Filtering metrics based on significance, effect size, and priority
2. Grouping metrics into logical report sections
"""

import logging
import csv
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple

from ..core.data_models import MetricResult, ReportSection

logger = logging.getLogger(__name__)


class AnalysisFilter:
    """
    Filters metrics based on statistical significance, effect size, and priority.
    """
    
    def __init__(self, 
                 priority_metrics: Optional[List[str]] = None,
                 p_value_threshold: float = 0.05,
                 effect_size_threshold: float = 0.5,
                 max_metrics_per_section: int = 10):
        """
        Initialize the AnalysisFilter with filtering parameters.
        
        Args:
            priority_metrics: List of metric names to prioritize
            p_value_threshold: Maximum p-value for statistical significance
            effect_size_threshold: Minimum effect size (Cohen's d) for relevance
            max_metrics_per_section: Maximum number of metrics to include per section
        """
        self.priority_metrics = set(priority_metrics or [])
        self.p_value_threshold = p_value_threshold
        self.effect_size_threshold = effect_size_threshold
        self.max_metrics_per_section = max_metrics_per_section
        logger.info(f"AnalysisFilter initialized with p_value_threshold={p_value_threshold}, "
                    f"effect_size_threshold={effect_size_threshold}")

    def filter_metrics(self, metrics: List[MetricResult]) -> List[MetricResult]:
        """
        Filter and prioritize metrics based on significance and effect size.
        
        Args:
            metrics: List of MetricResult objects to filter
            
        Returns:
            Filtered and prioritized list of MetricResult objects
        """
        if not metrics:
            logger.warning("No metrics provided for filtering")
            return []

        logger.info(f"Filtering {len(metrics)} metrics")
        
        # Step 1: Assign priority scores to metrics
        scored_metrics = []
        for metric in metrics:
            score = 0.0
            
            # Priority based on statistical significance
            if metric.p_value is not None and metric.p_value < self.p_value_threshold:
                score += 3.0
                
            # Priority based on effect size
            if metric.effect_size is not None and metric.effect_size > self.effect_size_threshold:
                score += 2.0
                
            # Priority based on being in the priority_metrics list
            if metric.name in self.priority_metrics:
                score += 5.0
                
            # Add a small weight for the absolute difference magnitude
            if metric.difference != 0 and metric.value_control != 0:
                rel_diff = abs(metric.difference) / abs(metric.value_control)
                score += min(rel_diff * 2.0, 2.0)  # Cap at 2.0
                
            # Assign the score and add to the list
            metric.priority_score = score
            scored_metrics.append(metric)
            
        # Step 2: Sort metrics by priority score (descending)
        sorted_metrics = sorted(scored_metrics, key=lambda m: m.priority_score, reverse=True)
        
        # For Stage 1, we might return all metrics or a limited set for testing
        if len(sorted_metrics) > 50:  # Arbitrary limit for testing
            logger.info(f"Limiting to top 50 metrics by priority score")
            return sorted_metrics[:50]
        
        return sorted_metrics


class SectionBuilder:
    """
    Groups metrics into logical report sections based on glossary mappings.
    """
    
    def __init__(self, glossary_file_path: Path):
        """
        Initialize the SectionBuilder with a glossary file path.
        
        Args:
            glossary_file_path: Path to the glossary_mapping.csv file
        """
        self.glossary_file_path = glossary_file_path
        self.prefix_to_section: Dict[str, Dict[str, str]] = {}
        self._load_glossary()
        
    def _load_glossary(self):
        """Load the glossary mapping from CSV file."""
        try:
            if not self.glossary_file_path.exists():
                logger.warning(f"Glossary file not found: {self.glossary_file_path}")
                # Create a minimal default mapping
                self.prefix_to_section = {
                    "DW_": {
                        "title": "DW Gate Analysis",
                        "description": "Analysis of the DW gate dynamics and conformations."
                    },
                    "Ion_": {
                        "title": "Ion Pathway Analysis",
                        "description": "Analysis of ion conduction and occupancy in the channel."
                    },
                    "Tyr_": {
                        "title": "Tyrosine Analysis",
                        "description": "Analysis of tyrosine residue conformations and interactions."
                    },
                    "Overview": {
                        "title": "System Overview",
                        "description": "High-level summary of the analyzed systems."
                    }
                }
                return
                
            with open(self.glossary_file_path, newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    if 'prefix' in row and 'section_title' in row:
                        self.prefix_to_section[row['prefix']] = {
                            "title": row['section_title'],
                            "description": row.get('section_description', ''),
                            "metric_template": row.get('metric_description_template', '')
                        }
                        
            logger.info(f"Loaded {len(self.prefix_to_section)} section mappings from glossary")
        except Exception as e:
            logger.error(f"Error loading glossary file: {e}")
            # Fallback to defaults
            self.prefix_to_section = {
                "DW_": {
                    "title": "DW Gate Analysis",
                    "description": "Analysis of the DW gate dynamics and conformations."
                },
                "Ion_": {
                    "title": "Ion Pathway Analysis",
                    "description": "Analysis of ion conduction and occupancy in the channel."
                }
            }
            
    def _get_section_for_metric(self, metric_name: str) -> Tuple[str, str]:
        """
        Determine which section a metric belongs to based on its name prefix.
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            Tuple of (section_title, section_description)
        """
        for prefix, section_info in self.prefix_to_section.items():
            if metric_name.startswith(prefix):
                return section_info["title"], section_info.get("description", "")
                
        # Default section for metrics that don't match any prefix
        return "Other Metrics", "Metrics that don't fall into specific categories."
        
    def group_metrics_into_sections(self, metrics: List[MetricResult]) -> List[ReportSection]:
        """
        Group metrics into logical report sections based on prefix mappings.
        
        Args:
            metrics: List of MetricResult objects to group
            
        Returns:
            List of ReportSection objects, each containing related metrics
        """
        if not metrics:
            logger.warning("No metrics provided for grouping into sections")
            return []
            
        logger.info(f"Grouping {len(metrics)} metrics into sections")
        
        # Group metrics by section title
        sections_dict: Dict[str, List[MetricResult]] = {}
        section_descriptions: Dict[str, str] = {}
        
        for metric in metrics:
            section_title, section_desc = self._get_section_for_metric(metric.name)
            
            if section_title not in sections_dict:
                sections_dict[section_title] = []
                section_descriptions[section_title] = section_desc
                
            sections_dict[section_title].append(metric)
            
        # Create ReportSection objects from the grouped metrics
        report_sections = []
        for title, section_metrics in sections_dict.items():
            # Sort metrics within section by priority score
            sorted_section_metrics = sorted(section_metrics, key=lambda m: m.priority_score, reverse=True)
            
            report_sections.append(
                ReportSection(
                    title=title,
                    description=section_descriptions.get(title, ""),
                    metrics=sorted_section_metrics
                )
            )
            
        # Sort sections by importance (based on highest metric priority in each section)
        if report_sections:
            report_sections.sort(
                key=lambda s: max((m.priority_score for m in s.metrics), default=0),
                reverse=True
            )
            
        logger.info(f"Created {len(report_sections)} report sections")
        return report_sections