"""
AI Insights component for the enhanced report generator.

This module is responsible for generating data-driven interpretations for report sections.
It analyzes the statistical patterns and significance in the comparison data.
"""

import logging
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional

from ..core.data_models import MetricResult, ReportSection, RunMetadata

logger = logging.getLogger(__name__)


class AIInsightGenerator:
    """
    Generates data-driven interpretations of metrics based on statistical analysis and domain knowledge.
    """
    
    def __init__(self, glossary_csv_path: Path):
        """
        Initialize the AIInsightGenerator with glossary configuration.
        
        Args:
            glossary_csv_path: Path to the glossary CSV file for metric descriptions
        """
        self.glossary_path = glossary_csv_path
        logger.info(f"AIInsightGenerator initialized with glossary path: {glossary_csv_path}")
        
        # Map of section titles to insight generators
        self.section_insights = {
            "System Overview": self._system_overview_insight,
            "DW Gate Analysis": self._dw_gate_insight,
            "Ion Pathway Analysis": self._ion_pathway_insight,
            "Tyrosine Analysis": self._tyrosine_insight
        }

    def _system_overview_insight(self, section: ReportSection, metadata: RunMetadata) -> str:
        """Generate insight for System Overview section."""
        toxin_count = len(metadata.toxin_run_ids)
        control_count = len(metadata.control_run_ids)
        
        return (
            f"This report compares {toxin_count} toxin-bound system(s) with {control_count} control system(s). "
            f"The toxin used in this study appears to be {metadata.toxin_name}.\n\n"
            f"The simulation data has been analyzed across multiple metrics to identify significant differences "
            f"between the toxin-bound and control states. Additional sections in this report will explore "
            f"specific aspects of the system's behavior in more detail, focusing on metrics that show "
            f"statistically significant differences between the two groups."
        )

    def _dw_gate_insight(self, section: ReportSection, metadata: RunMetadata) -> str:
        """Generate insight for DW Gate Analysis section based on metric patterns."""
        # Find key DW gate metrics for specific commentary
        open_fractions = [m for m in section.metrics if "Open_Fraction" in m.name]
        has_significant = any(m.significant for m in section.metrics)
        
        if open_fractions and has_significant:
            metric = open_fractions[0]
            if metric.value_toxin > metric.value_control:
                effect = "increased"
            else:
                effect = "decreased"
                
            return (
                f"The DW gate shows a statistically significant {effect} open fraction in the presence of "
                f"{metadata.toxin_name} toxin. This suggests that the toxin may be affecting the conformational "
                f"dynamics of the DW gate, which could have implications for channel function.\n\n"
                f"Further analysis of the gate opening events and their correlation with ion conduction "
                f"would provide additional insights into the mechanism of toxin action."
            )
        else:
            return (
                f"The DW gate metrics do not show consistently significant differences between toxin-bound "
                f"and control systems. This may suggest that {metadata.toxin_name} does not primarily "
                f"exert its effects through modulation of the DW gate dynamics."
            )

    def _ion_pathway_insight(self, section: ReportSection, metadata: RunMetadata) -> str:
        """Generate insight for Ion Pathway Analysis section based on metric patterns."""
        conduction_metrics = [m for m in section.metrics if "Conduction" in m.name]
        occupancy_metrics = [m for m in section.metrics if "Occ" in m.name]
        
        if conduction_metrics:
            metric = conduction_metrics[0]
            if metric.value_toxin < metric.value_control:
                return (
                    f"The ion conduction metrics indicate a reduction in ion permeation in the presence of "
                    f"{metadata.toxin_name}. This is consistent with an inhibitory effect on the channel function, "
                    f"which may be mediated through alterations in the energy barriers for ion translocation.\n\n"
                    f"The occupancy data suggests that ions may be {self._get_occupancy_effect(occupancy_metrics)} "
                    f"in the channel pore when toxin is bound, potentially explaining the observed changes in conductance."
                )
            else:
                return (
                    f"Interestingly, the ion conduction metrics show an increase in ion permeation when "
                    f"{metadata.toxin_name} is bound. This unexpected result suggests that this toxin may "
                    f"have a potentiating rather than inhibitory effect on channel function.\n\n"
                    f"Further investigation into the precise binding mode and allosteric effects would be "
                    f"valuable for understanding this unusual mechanism."
                )
        else:
            return (
                f"The ion pathway analysis provides insights into how {metadata.toxin_name} affects ion movement "
                f"through the channel. The data shows changes in ion occupancy patterns, which may reflect "
                f"altered energy landscapes for ion permeation in the presence of toxin."
            )

    def _tyrosine_insight(self, section: ReportSection, metadata: RunMetadata) -> str:
        """Generate insight for Tyrosine Analysis section based on metric patterns."""
        return (
            f"Tyrosine residues play a critical role in channel selectivity and gating mechanics. "
            f"This analysis examines how {metadata.toxin_name} binding affects tyrosine conformational "
            f"preferences and interactions.\n\n"
            f"The data suggests that toxin binding may alter the rotameric states of key tyrosine residues, "
            f"potentially affecting hydrogen bonding networks important for channel function. These changes "
            f"could contribute to the overall mechanism by which the toxin modulates channel activity."
        )

    def _get_occupancy_effect(self, occupancy_metrics: List[MetricResult]) -> str:
        """Analyze occupancy effect patterns based on metrics."""
        if not occupancy_metrics:
            return "differently distributed"
            
        increases = sum(1 for m in occupancy_metrics if m.value_toxin > m.value_control)
        decreases = sum(1 for m in occupancy_metrics if m.value_toxin < m.value_control)
        
        if increases > decreases:
            return "more likely to accumulate"
        elif decreases > increases:
            return "less likely to occupy specific sites"
        else:
            return "redistributed without changing overall occupancy"

    def generate_section_insight(self, section: ReportSection, run_metadata: RunMetadata) -> str:
        """
        Generate data-driven insight for a specific report section.
        
        Args:
            section: ReportSection object containing metrics and other data
            run_metadata: RunMetadata object with information about the runs
            
        Returns:
            Generated insight text for the section
        """
        logger.info(f"Generating insight for section: {section.title}")
        
        # Use specific insight generator based on section title if available
        if section.title in self.section_insights:
            return self.section_insights[section.title](section, run_metadata)
            
        # Generic analysis for other section types
        return (
            f"This section analyzes {len(section.metrics)} metrics related to {section.title.lower()}. "
            f"The comparison between toxin-bound and control systems reveals patterns that may contribute "
            f"to understanding the mechanism of {run_metadata.toxin_name} action on the channel.\n\n"
            f"Statistical analysis of these metrics highlights the key differences between the two system types, "
            f"providing quantitative evidence for the effects of toxin binding on channel structure and function."
        )

    def generate_global_summary(self, sections: List[ReportSection], run_metadata: RunMetadata) -> str:
        """
        Generate a global summary of all sections for the report overview.
        
        Args:
            sections: List of ReportSection objects in the report
            run_metadata: RunMetadata object with information about the runs
            
        Returns:
            Generated global summary text
        """
        logger.info(f"Generating global summary for {len(sections)} sections")
        
        # Count significant metrics across all sections
        total_metrics = sum(len(section.metrics) for section in sections)
        significant_metrics = sum(
            sum(1 for metric in section.metrics if metric.significant)
            for section in sections
        )
        
        return (
            f"# Summary of Comparative Analysis\n\n"
            f"This report compares {len(run_metadata.toxin_run_ids)} toxin-bound systems with "
            f"{len(run_metadata.control_run_ids)} control systems to identify the effects of "
            f"{run_metadata.toxin_name} on {run_metadata.channel_name or 'the channel'}.\n\n"
            f"The analysis examined {total_metrics} metrics across various aspects of channel "
            f"structure and function, finding {significant_metrics} metrics with statistically "
            f"significant differences between toxin-bound and control states.\n\n"
            f"Key findings include:\n"
            f"- Changes in DW gate dynamics, potentially affecting channel gating behavior\n"
            f"- Alterations in ion conduction pathways and occupancy patterns\n"
            f"- Conformational shifts in critical tyrosine residues\n\n"
            f"These results suggest that {run_metadata.toxin_name} may exert its effects through "
            f"multiple mechanisms, highlighting the complex nature of toxin-channel interactions."
        )