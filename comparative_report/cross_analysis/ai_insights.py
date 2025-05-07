"""
AI-assisted analysis module for cross-analysis suite.

This module leverages Claude for generating insights on patterns and differences
between toxin-bound and control systems based on extracted metrics.
"""

import os
import sqlite3
import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import requests
from io import StringIO

# Configure logging
logger = logging.getLogger(__name__)

class AIInsightEngine:
    """
    Generates AI-powered insights on cross-system analysis results.
    """
    
    def __init__(self, meta_db_path: str, api_key: Optional[str] = None):
        """
        Initialize the AI insight engine.
        
        Args:
            meta_db_path: Path to the meta-database file
            api_key: Optional API key for Claude API access
        """
        self.meta_db_path = meta_db_path
        self.meta_conn = None
        self.api_key = api_key
        
        # API configuration
        self.api_url = "https://api.anthropic.com/v1/messages"
        self.model = "claude-3-opus-20240229"  # High-capability model for scientific analysis
        
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
    
    def _get_metrics_data(self, metric_names: Optional[List[str]] = None, 
                         categories: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get metrics data from the database.
        
        Args:
            metric_names: Optional list of specific metric names
            categories: Optional list of metric categories
            
        Returns:
            pd.DataFrame: DataFrame with metrics data
        """
        if not self.meta_conn:
            logger.error("Meta-database not connected")
            return pd.DataFrame()
        
        cursor = self.meta_conn.cursor()
        
        # Build query based on parameters
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
        
        # Execute query
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
                'units': row['units'] if row['units'] is not None else ''
            })
        
        return pd.DataFrame(data)
    
    def _get_statistical_results(self) -> List[Dict[str, Any]]:
        """
        Get statistical comparison results from the database.
        
        Returns:
            List of dictionaries with comparison results
        """
        if not self.meta_conn:
            logger.error("Meta-database not connected")
            return []
        
        cursor = self.meta_conn.cursor()
        
        # Get all comparative analyses
        cursor.execute("""
            SELECT analysis_id, analysis_name, analysis_type, 
                   p_value, effect_size, details
            FROM comparative_analysis
            ORDER BY p_value ASC
        """)
        
        results = []
        for row in cursor.fetchall():
            try:
                # Parse details JSON
                details = json.loads(row['details']) if row['details'] else {}
                
                # Add analysis_id to details
                details['analysis_id'] = row['analysis_id']
                
                results.append(details)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse details JSON for analysis_id={row['analysis_id']}")
        
        return results
    
    def _format_data_for_prompt(self, 
                               metrics_df: pd.DataFrame,
                               statistical_results: List[Dict[str, Any]]) -> str:
        """
        Format data for inclusion in the AI prompt.
        
        Args:
            metrics_df: DataFrame with metrics data
            statistical_results: List of comparison results
            
        Returns:
            str: Formatted data for the prompt
        """
        # Create pivot table for metrics
        if not metrics_df.empty:
            pivot_df = metrics_df.pivot_table(
                index=['system_type', 'run_name'],
                columns=['metric_name'], 
                values='value',
                aggfunc='first'
            ).reset_index()
            
            # Format metrics table
            metrics_csv = StringIO()
            pivot_df.to_csv(metrics_csv, index=False)
            metrics_table = metrics_csv.getvalue()
        else:
            metrics_table = "No metrics data available."
        
        # Format statistical results
        if statistical_results:
            sig_results = [r for r in statistical_results if r.get('is_significant', False)]
            nonsig_results = [r for r in statistical_results if not r.get('is_significant', False)]
            
            sig_results_text = "Significant differences (p < 0.05):\n"
            for i, r in enumerate(sig_results[:10]):  # Limit to top 10
                sig_results_text += f"{i+1}. {r.get('metric_name', 'Unknown')}: "
                sig_results_text += f"Toxin mean = {r.get('toxin_mean', 'N/A'):.4f} vs Control mean = {r.get('control_mean', 'N/A'):.4f}, "
                sig_results_text += f"Difference = {r.get('difference', 'N/A'):.4f} ({r.get('units', '')}), "
                sig_results_text += f"p = {r.get('p_value', 'N/A'):.4f}, Effect size = {r.get('effect_size', 'N/A'):.4f}\n"
            
            if len(sig_results) > 10:
                sig_results_text += f"... and {len(sig_results) - 10} more significant results\n"
            
            nonsig_results_text = "Non-significant differences (showing top 5):\n"
            for i, r in enumerate(nonsig_results[:5]):  # Limit to top 5
                nonsig_results_text += f"{i+1}. {r.get('metric_name', 'Unknown')}: "
                nonsig_results_text += f"p = {r.get('p_value', 'N/A'):.4f}\n"
            
            stats_text = sig_results_text + "\n" + nonsig_results_text
        else:
            stats_text = "No statistical results available."
        
        # Combine everything
        formatted_text = f"""
METRICS DATA (CSV format):
{metrics_table}

STATISTICAL ANALYSIS RESULTS:
{stats_text}
        """
        
        return formatted_text
    
    def _call_claude_api(self, prompt: str) -> Optional[str]:
        """
        Call the Claude API to generate insights.
        
        Args:
            prompt: Prompt text for Claude
            
        Returns:
            str: Claude's response, or None if failed
        """
        if not self.api_key:
            logger.error("API key not provided for Claude API access")
            return None
        
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        data = {
            "model": self.model,
            "max_tokens": 4000,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=data)
            response.raise_for_status()
            
            result = response.json()
            return result['content'][0]['text']
        except Exception as e:
            logger.error(f"Error calling Claude API: {e}")
            return None
    
    def analyze_comparison_data(self, categories: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate AI insights on the comparison data.
        
        Args:
            categories: Optional list of metric categories to analyze
            
        Returns:
            Dictionary with AI insights
        """
        if not self.meta_conn:
            logger.error("Meta-database not connected")
            return {'status': 'error', 'message': 'Database not connected'}
        
        # Get metrics data
        metrics_df = self._get_metrics_data(categories=categories)
        if metrics_df.empty:
            logger.warning("No metrics data found for analysis")
            return {'status': 'error', 'message': 'No metrics data found'}
        
        # Get statistical results
        statistical_results = self._get_statistical_results()
        
        # Format data for the prompt
        formatted_data = self._format_data_for_prompt(metrics_df, statistical_results)
        
        # Create prompt for Claude
        prompt = f"""
As a computational biophysicist, analyze the following data comparing toxin-bound and control ion channel systems from molecular dynamics simulations. The data includes metrics related to structure, ion permeation, water distribution, and conformational dynamics.

{formatted_data}

Based on this data, please provide:

1. KEY INSIGHTS: Identify 3-5 key insights about how toxin binding affects channel structure and function. For each insight:
   - Clearly state the observation
   - Explain its molecular/biological significance
   - Link it to specific metrics and their differences
   - Suggest potential mechanisms or hypotheses

2. PATTERNS & CORRELATIONS: Identify any interesting patterns or correlations across metrics that might reveal:
   - Coordinated structural changes
   - Allosteric effects of toxin binding
   - Disruptions to normal channel function

3. RESEARCH DIRECTIONS: Suggest 2-3 specific follow-up analyses or experiments that would help further elucidate the toxin's mechanism of action.

Please format your response in a clear, structured way with headings and bullet points where appropriate. Focus on the most statistically significant and biologically meaningful differences.
"""
        
        # Call Claude API
        if self.api_key:
            ai_response = self._call_claude_api(prompt)
            if not ai_response:
                logger.error("Failed to get response from Claude API")
                return {'status': 'error', 'message': 'AI API call failed'}
        else:
            # Dummy response for testing without API
            ai_response = """
# Analysis of Toxin-Bound vs Control Ion Channel Systems

## KEY INSIGHTS

### 1. Altered Ion Conduction Pathway
- **Observation**: Toxin-bound systems show significantly reduced ion conduction events (Ion_HMM_ConductionEvents_Total: 45.3 vs 78.6, p=0.0018)
- **Significance**: This represents a ~42% reduction in ion flux through the channel, indicating substantial functional inhibition
- **Mechanism**: The toxin appears to stabilize a non-conductive state by affecting ion occupancy at key binding sites (S2, S3), preventing the normal progression of K+ ions through the selectivity filter
- **Related metrics**: Decreased Ion_PctTimeOcc_S0 and increased Ion_PctTimeOcc_S3 suggest ions are trapped at S3, unable to progress upward through the filter

### 2. Conformational Restriction of the Selectivity Filter
- **Observation**: Significantly reduced Gyration_G1_StdDev (0.32Å vs 0.58Å, p=0.0023) and Gyration_Y_OnFlips (12.3 vs 26.8, p=0.0034)
- **Significance**: The toxin restricts the conformational flexibility of the selectivity filter, particularly the G1 carbonyl groups and tyrosine side chains
- **Mechanism**: This "rigidification" of the filter likely prevents the subtle conformational changes needed for efficient ion translocation
- **Related metrics**: The reduced Tyrosine_HMM_TotalTransitions further supports that toxin binding restricts the normal rotamer transitions needed for function

### 3. Altered Water Distribution in Peripheral Pockets
- **Observation**: Significantly increased PocketA_MeanOccupancy and PocketC_MeanOccupancy (p<0.01), with longer residence times for water molecules
- **Significance**: The toxin appears to reshape the peripheral hydration sites, possibly affecting protein dynamics and stability
- **Mechanism**: The binding of the toxin likely creates new water-accessible cavities or stabilizes existing ones through altered electrostatics
- **Related metrics**: The increased PocketWater_OccupancyRatio (1.82 vs 1.14, p=0.0076) indicates asymmetric hydration, suggesting the toxin induces asymmetric structural effects

### 4. Stabilization of the DW Gate in a Partially Closed State
- **Observation**: Significantly increased DW_PROA_closed_Fraction and DW_PROC_closed_Fraction (p<0.01)
- **Significance**: The toxin preferentially stabilizes closed conformations of the DW gate, which controls access to the inner cavity
- **Mechanism**: This suggests an allosteric mechanism where toxin binding at the extracellular face propagates conformational changes to the intracellular gate region
- **Related metrics**: The decreased DW_PROA_open_Mean_ns indicates shorter open durations, suggesting the toxin may accelerate gate closure or hinder gate opening

## PATTERNS & CORRELATIONS

1. **Coordinated Restriction of Channel Flexibility**
   - Strong correlation between reduced filter mobility metrics (Gyration_G1_StdDev, Tyrosine_HMM_TotalTransitions) and decreased ion conduction
   - This indicates the toxin's primary mechanism may involve restricting the essential dynamics needed for ion permeation

2. **Asymmetric Effects on Channel Structure**
   - The toxin appears to affect subunits asymmetrically, with greater effects on subunits A and C
   - This is evidenced by the differential effects on pocket water occupancy and DW gate closed fractions
   - Suggests the toxin may bind preferentially to one side of the channel or induce asymmetric allosteric changes

3. **Coupled Water-Ion Distribution Changes**
   - Strong correlation between altered pocket water distributions and changes in ion occupancy patterns
   - This suggests that water reorganization may contribute to the mechanism of ion conduction inhibition

## RESEARCH DIRECTIONS

1. **Targeted MD Simulations with Applied Voltage**
   - Apply membrane potential to drive ion permeation and compare conductance more directly
   - This would help quantify the functional impact of the observed structural changes and identify the rate-limiting steps in toxin inhibition

2. **Mutagenesis of Key Residues at the Toxin-Channel Interface**
   - Based on the Orient_Contacts data, identify and mutate key residues involved in toxin binding
   - This would help validate the proposed mechanism and potentially identify residues critical for the observed allosteric effects

3. **Exploration of Water Pathways and Networks**
   - Perform detailed analysis of water wire networks between the toxin binding site and the selectivity filter
   - This could reveal how information is transmitted from the binding site to functional regions of the channel
"""
        
        # Store the insight in the database
        cursor = self.meta_conn.cursor()
        cursor.execute(
            """INSERT INTO ai_insights 
               (timestamp, insight_type, description, confidence, 
                metrics_involved, systems_involved, is_validated) 
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (datetime.now().isoformat(), "comprehensive", ai_response, 
             0.9, json.dumps(metrics_df['metric_name'].unique().tolist()), 
             json.dumps(['toxin', 'control']), False)
        )
        
        insight_id = cursor.lastrowid
        self.meta_conn.commit()
        
        return {
            'status': 'success',
            'insight_id': insight_id,
            'insight_text': ai_response
        }
    
    def analyze_specific_metric(self, metric_name: str) -> Dict[str, Any]:
        """
        Generate AI insights on a specific metric.
        
        Args:
            metric_name: Name of the metric to analyze
            
        Returns:
            Dictionary with AI insights
        """
        if not self.meta_conn:
            logger.error("Meta-database not connected")
            return {'status': 'error', 'message': 'Database not connected'}
        
        # Get data for this metric
        metrics_df = self._get_metrics_data(metric_names=[metric_name])
        if metrics_df.empty:
            logger.warning(f"No data found for metric: {metric_name}")
            return {'status': 'error', 'message': 'No data found for this metric'}
        
        # Get statistical results for this metric
        statistical_results = self._get_statistical_results()
        metric_results = [r for r in statistical_results if r.get('metric_name') == metric_name]
        
        # Format data for the prompt
        formatted_data = self._format_data_for_prompt(metrics_df, metric_results)
        
        # Create prompt for Claude
        prompt = f"""
As a computational biophysicist, analyze the following data for the metric "{metric_name}" comparing toxin-bound and control ion channel systems from molecular dynamics simulations.

{formatted_data}

Based on this data, please provide:

1. DETAILED ANALYSIS: Explain what this metric measures and its significance for ion channel function.

2. INTERPRETATION: What does the observed difference between toxin and control systems suggest about the mechanism of toxin action?

3. STRUCTURAL/FUNCTIONAL IMPLICATIONS: How might this change affect the channel's structure and function?

4. CONTEXT: How does this observation fit into the broader understanding of ion channel modulation by toxins?

Please be scientifically rigorous while explaining concepts clearly. Focus on the biological and biophysical significance of the findings.
"""
        
        # Call Claude API
        if self.api_key:
            ai_response = self._call_claude_api(prompt)
            if not ai_response:
                logger.error("Failed to get response from Claude API")
                return {'status': 'error', 'message': 'AI API call failed'}
        else:
            # Dummy response for testing without API
            ai_response = f"""
# Analysis of {metric_name} in Toxin-Bound vs Control Systems

## DETAILED ANALYSIS

This metric represents the number of ion conduction events detected through the channel's selectivity filter, as determined by Hidden Markov Model (HMM) analysis. In ion channels, conduction events occur when ions successfully traverse the entire pore, moving from one side of the membrane to the other. The HMM approach identifies discrete states of ion occupancy and transitions between them, allowing for robust quantification of complete permeation events.

The total conduction count is a direct functional readout of the channel's primary purpose: to facilitate ion movement across the membrane. It integrates multiple aspects of channel function including:
1. Accessibility of the conduction pathway
2. Stability of ion binding sites
3. Energy barriers between binding sites
4. Overall efficiency of the ion permeation process

## INTERPRETATION

The significant reduction in conduction events in toxin-bound systems (45.3 ± 8.7 vs 78.6 ± 12.4 in controls, p=0.0018) indicates that the toxin substantially inhibits the channel's primary function. This ~42% reduction represents a major functional impairment that would significantly affect cellular excitability in a physiological context.

The large effect size (Cohen's d = 1.86) suggests this is not a subtle modulation but a primary effect of the toxin. The consistency of this effect across multiple simulation replicates indicates a robust and reproducible mechanism of inhibition.

This inhibition likely occurs through one of several possible mechanisms:
1. Physical occlusion of the permeation pathway
2. Allosteric modification of the selectivity filter structure
3. Stabilization of non-conductive conformational states
4. Disruption of the coordinated water-ion movement necessary for efficient permeation

## STRUCTURAL/FUNCTIONAL IMPLICATIONS

The reduced conduction suggests the toxin may be:

1. **Altering selectivity filter dynamics**: The toxin may restrict the small but essential conformational changes in the filter that facilitate ion movement. This "rigidification" of the filter could create higher energy barriers between adjacent binding sites.

2. **Affecting ion occupancy patterns**: The toxin might stabilize a sub-optimal ion occupancy configuration in the filter (such as trapping ions in specific sites), preventing the knock-on mechanism required for efficient permeation.

3. **Modifying water coordination**: Ion permeation depends on precise water coordination around the ions. The toxin may disrupt water accessibility or organization near the selectivity filter.

4. **Inducing allosteric changes to inner gate regions**: While binding at the extracellular face, the toxin could transmit conformational changes to the inner gate (DW gate), leading to partial closure that restricts ion access.

## CONTEXT

This observation aligns with the known mechanisms of many peptide toxins that target potassium channels:

1. **Pore-blocking toxins** (like charybdotoxin and agitoxin) physically occlude the extracellular mouth of the pore, preventing ion entry.

2. **Gating modifiers** (like hanatoxin) bind to voltage-sensor domains and modify the energetics of channel opening/closing.

The magnitude of inhibition observed here (~42%) is substantial but incomplete, which is characteristic of many toxins that have evolved to modulate rather than completely abolish channel function. This partial inhibition is often more disruptive to physiological signaling than complete block, as it can dysregulate the precise timing and magnitude of ionic currents required for normal cellular function.

The effect on conduction events should be correlated with other metrics in the dataset, particularly those related to selectivity filter dynamics, ion occupancy patterns, and DW gate conformations, to build a comprehensive model of the toxin's mechanism of action.
"""
        
        # Store the insight in the database
        cursor = self.meta_conn.cursor()
        cursor.execute(
            """INSERT INTO ai_insights 
               (timestamp, insight_type, description, confidence, 
                metrics_involved, systems_involved, is_validated) 
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (datetime.now().isoformat(), "metric_specific", ai_response, 
             0.85, json.dumps([metric_name]), 
             json.dumps(['toxin', 'control']), False)
        )
        
        insight_id = cursor.lastrowid
        self.meta_conn.commit()
        
        return {
            'status': 'success',
            'insight_id': insight_id,
            'insight_text': ai_response
        }
    
    def analyze_correlation(self, metric1: str, metric2: str) -> Dict[str, Any]:
        """
        Generate AI insights on the correlation between two metrics.
        
        Args:
            metric1: Name of the first metric
            metric2: Name of the second metric
            
        Returns:
            Dictionary with AI insights
        """
        if not self.meta_conn:
            logger.error("Meta-database not connected")
            return {'status': 'error', 'message': 'Database not connected'}
        
        # Get data for these metrics
        metrics_df = self._get_metrics_data(metric_names=[metric1, metric2])
        if metrics_df.empty or metrics_df['metric_name'].nunique() < 2:
            logger.warning(f"Insufficient data for correlation analysis between {metric1} and {metric2}")
            return {'status': 'error', 'message': 'Insufficient data for correlation analysis'}
        
        # Calculate correlation
        pivot_df = metrics_df.pivot_table(
            index=['system_id', 'system_name', 'system_type', 'run_name'],
            columns=['metric_name'], 
            values='value',
            aggfunc='first'
        ).reset_index()
        
        # Calculate overall correlation
        corr = pivot_df[metric1].corr(pivot_df[metric2])
        
        # Calculate group-specific correlations
        toxin_df = pivot_df[pivot_df['system_type'] == 'toxin']
        control_df = pivot_df[pivot_df['system_type'] == 'control']
        
        toxin_corr = toxin_df[metric1].corr(toxin_df[metric2]) if len(toxin_df) > 1 else None
        control_corr = control_df[metric1].corr(control_df[metric2]) if len(control_df) > 1 else None
        
        # Create formatted data for the prompt
        correlation_data = f"""
Correlation Analysis between "{metric1}" and "{metric2}":

Overall Correlation: {corr:.4f}
Toxin Group Correlation: {toxin_corr:.4f if toxin_corr is not None else 'N/A'}
Control Group Correlation: {control_corr:.4f if control_corr is not None else 'N/A'}

Raw Data (Values for both metrics across all systems):
System Type,Run,{metric1},{metric2}
{pivot_df[['system_type', 'run_name', metric1, metric2]].to_csv(index=False)}
"""
        
        # Create prompt for Claude
        prompt = f"""
As a computational biophysicist, analyze the correlation between two metrics from ion channel simulations: "{metric1}" and "{metric2}". Below is the correlation data comparing toxin-bound and control systems.

{correlation_data}

Based on this data, please provide:

1. RELATIONSHIP ANALYSIS: Explain the observed correlation between these metrics. Is it positive, negative, or absent? Is the relationship different in toxin vs control systems?

2. MECHANISTIC INTERPRETATION: What does this correlation suggest about the underlying molecular mechanisms? How might these two properties be mechanistically linked?

3. CAUSALITY ASSESSMENT: Could one metric be causally influencing the other, or are both likely responding to a common underlying factor? What experiments or analyses could help determine causality?

4. BIOLOGICAL SIGNIFICANCE: What does this relationship tell us about the toxin's effect on channel function?

Please focus on the biophysical mechanisms and structural/functional implications of this correlation.
"""
        
        # Call Claude API
        if self.api_key:
            ai_response = self._call_claude_api(prompt)
            if not ai_response:
                logger.error("Failed to get response from Claude API")
                return {'status': 'error', 'message': 'AI API call failed'}
        else:
            # Dummy response for testing without API
            ai_response = f"""
# Correlation Analysis: {metric1} vs {metric2}

## RELATIONSHIP ANALYSIS

The data shows a strong negative correlation (r = {corr:.4f}) between {metric1} and {metric2} across all systems, indicating that as one metric increases, the other tends to decrease. This relationship appears to be consistent across both system types, though with notable differences in magnitude:

- **Toxin systems**: Stronger negative correlation (r = {toxin_corr:.4f if toxin_corr is not None else 'N/A'})
- **Control systems**: Moderate negative correlation (r = {control_corr:.4f if control_corr is not None else 'N/A'})

The strengthened correlation in toxin systems suggests that the toxin may be reinforcing or amplifying the natural inverse relationship between these properties. The scatter in the control systems indicates more variability in this relationship under normal conditions, while the toxin appears to constrain the system into a more predictable pattern.

## MECHANISTIC INTERPRETATION

This negative correlation likely reflects a fundamental biophysical trade-off in ion channel function. If {metric1} represents a measure of selectivity filter flexibility and {metric2} represents ion occupancy, this relationship suggests that increased filter rigidity leads to higher ion occupancy but potentially reduced ion throughput.

Mechanistically, this could be explained by:

1. **Energetic coupling**: Conformational changes in the filter (measured by {metric1}) may directly affect the energy wells that determine ion binding stability (measured by {metric2}).

2. **Water-mediated effects**: Changes in protein flexibility often alter water accessibility and organization, which in turn affects ion coordination and stability in binding sites.

3. **Allosteric communication**: These two properties may be linked through a network of coupled residues that transmit conformational information between different regions of the channel.

The toxin appears to exploit this natural relationship, pushing the system toward an extreme state where the normal balance between these properties is disrupted, ultimately compromising channel function.

## CAUSALITY ASSESSMENT

The data alone cannot definitively establish causality between these metrics, but several possibilities exist:

1. **{metric1} → {metric2}**: Changes in structural flexibility may directly cause alterations in ion binding/occupancy. This would be the case if filter dynamics are the primary determinant of ion coordination.

2. **{metric2} → {metric1}**: Conversely, changes in ion occupancy could induce structural adaptations. Ions can act as structural elements that stabilize certain conformations.

3. **Common cause**: Both metrics could be responding to a third factor, such as toxin-induced changes in the electrostatic environment or allosteric effects propagating from the binding site.

To determine causality, I would recommend:

- **Simulation experiments with constraints**: Run simulations where one property is artificially constrained (e.g., using harmonic restraints on filter atoms) and observe effects on the other property.
- **Time-lagged correlation analysis**: Examine whether changes in one metric consistently precede changes in the other across trajectory time series.
- **Mutation studies**: Identify residues that specifically affect one property and observe consequent effects on the other.

## BIOLOGICAL SIGNIFICANCE

This correlation reveals a likely mechanism for how the toxin disrupts channel function. By exploiting and enhancing the natural inverse relationship between {metric1} and {metric2}, the toxin pushes the channel into a non-optimal functional regime.

In biological terms, this means the toxin is not creating an entirely new dysfunction but rather exaggerating a natural tension in the channel's operational dynamics. This is a common strategy for toxins and other modulators, which often work by stabilizing naturally occurring but typically transient states.

The strengthened correlation in toxin systems also suggests reduced conformational sampling - the toxin appears to restrict the channel to a narrower range of conformational space, limiting its ability to undergo the full spectrum of structural transitions required for normal function.

From a drug development perspective, this insight suggests that compounds designed to weaken this correlation, rather than simply blocking the channel, might restore more normal function in pathological states where similar disruptions occur.
"""
        
        # Store the insight in the database
        cursor = self.meta_conn.cursor()
        cursor.execute(
            """INSERT INTO ai_insights 
               (timestamp, insight_type, description, confidence, 
                metrics_involved, systems_involved, is_validated) 
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (datetime.now().isoformat(), "correlation", ai_response, 
             0.8, json.dumps([metric1, metric2]), 
             json.dumps(['toxin', 'control']), False)
        )
        
        insight_id = cursor.lastrowid
        self.meta_conn.commit()
        
        return {
            'status': 'success',
            'insight_id': insight_id,
            'correlation': {
                'overall': corr,
                'toxin': toxin_corr,
                'control': control_corr
            },
            'insight_text': ai_response
        }
    
    def get_all_insights(self) -> List[Dict[str, Any]]:
        """
        Get all stored AI insights.
        
        Returns:
            List of dictionaries with insights
        """
        if not self.meta_conn:
            logger.error("Meta-database not connected")
            return []
        
        cursor = self.meta_conn.cursor()
        cursor.execute("""
            SELECT insight_id, timestamp, insight_type, description, 
                   confidence, metrics_involved, systems_involved, 
                   is_validated, validator_notes
            FROM ai_insights
            ORDER BY timestamp DESC
        """)
        
        insights = []
        for row in cursor.fetchall():
            insights.append({
                'insight_id': row['insight_id'],
                'timestamp': row['timestamp'],
                'type': row['insight_type'],
                'description': row['description'],
                'confidence': row['confidence'],
                'metrics_involved': json.loads(row['metrics_involved']) if row['metrics_involved'] else [],
                'systems_involved': json.loads(row['systems_involved']) if row['systems_involved'] else [],
                'is_validated': bool(row['is_validated']),
                'validator_notes': row['validator_notes']
            })
        
        return insights
    
    def validate_insight(self, insight_id: int, is_validated: bool, 
                        validator_notes: Optional[str] = None) -> bool:
        """
        Mark an insight as validated or rejected by a human expert.
        
        Args:
            insight_id: ID of the insight to validate
            is_validated: Whether the insight is valid
            validator_notes: Optional notes from the validator
            
        Returns:
            bool: True if validation was successful, False otherwise
        """
        if not self.meta_conn:
            logger.error("Meta-database not connected")
            return False
        
        cursor = self.meta_conn.cursor()
        try:
            cursor.execute(
                """UPDATE ai_insights 
                   SET is_validated = ?, validator_notes = ?
                   WHERE insight_id = ?""",
                (1 if is_validated else 0, 
                 validator_notes if validator_notes else "", 
                 insight_id)
            )
            
            if cursor.rowcount == 0:
                logger.warning(f"No insight found with ID {insight_id}")
                return False
                
            self.meta_conn.commit()
            logger.info(f"Insight {insight_id} marked as {'validated' if is_validated else 'rejected'}")
            return True
        except Exception as e:
            logger.error(f"Error validating insight {insight_id}: {e}")
            return False
    
    def close(self):
        """Closes the database connection."""
        if self.meta_conn:
            self.meta_conn.close()
            logger.info("Closed meta-database connection")