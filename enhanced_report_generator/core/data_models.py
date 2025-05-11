"""
Core data models for the enhanced report generator.

This module defines the primary data structures used throughout the report generation pipeline:
- MetricResult: Represents a single metric with comparison data between toxin and control groups
- ReportSection: Represents a thematic section in the report containing related metrics
- RunMetadata: Contains high-level metadata about the simulation runs being compared
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path


@dataclass
class MetricResult:
    """
    Represents a single metric with comparison data between toxin and control groups.
    """
    name: str
    value_toxin: float  # Mean value for toxin group
    value_control: float  # Mean value for control group
    units: str = ""  # Units of measurement (%, Å, etc.)
    difference: float = 0.0  # Absolute difference (value_toxin - value_control)
    p_value: Optional[float] = None  # Statistical significance (p-value)
    significant: bool = False  # Whether difference is statistically significant
    effect_size: Optional[float] = None  # Effect size (e.g., Cohen's d)
    description: str = ""  # Human-readable description of this metric
    priority_score: float = 0.0  # Score for prioritizing visualization/reporting


@dataclass
class ReportSection:
    """
    Represents a thematic section in the report containing related metrics.
    """
    title: str  # Section title
    description: str = ""  # Section description
    metrics: List[MetricResult] = field(default_factory=list)  # Metrics in this section
    plots: List[Path] = field(default_factory=list)  # Paths to generated plots for this section
    ai_interpretation: str = ""  # AI-generated interpretation text for this section


@dataclass
class RunMetadata:
    """
    Contains high-level metadata about the simulation runs being compared.
    """
    control_run_ids: List[str]  # IDs of control (toxin-free) runs
    toxin_run_ids: List[str]  # IDs of toxin-bound runs
    toxin_name: str = ""  # Name of the toxin (if available)
    channel_name: str = ""  # Name of the channel (if available)
    simulation_parameters: Dict[str, Any] = field(default_factory=dict)  # Shared simulation parameters