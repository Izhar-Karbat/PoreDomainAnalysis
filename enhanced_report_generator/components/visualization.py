"""
Visualization component for the enhanced report generator.

This module is responsible for generating plots and visualizations for the report.
For the MVP, it implements a minimal set of standard plots.
"""

import logging
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from ..core.data_models import MetricResult, ReportSection

logger = logging.getLogger(__name__)

# Set matplotlib style
plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams['figure.figsize'] = (10, 6)
mpl.rcParams['font.size'] = 12


class VisualizationEngine:
    """
    Generates visualizations and plots for the report sections.
    """
    
    def __init__(self, output_plots_dir: Path):
        """
        Initialize the VisualizationEngine with an output directory for plots.
        
        Args:
            output_plots_dir: Directory where plot files will be saved
        """
        self.output_plots_dir = output_plots_dir
        self.output_plots_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"VisualizationEngine initialized with output directory: {output_plots_dir}")
        
        # Define plot colors
        self.toxin_color = '#d63384'  # Pink/purple for toxin
        self.control_color = '#198754'  # Green for control

    def generate_plots_for_section(self, section: ReportSection) -> List[Path]:
        """
        Generate appropriate plots for a given report section.
        
        Args:
            section: ReportSection object containing metrics to visualize
            
        Returns:
            List of paths to generated plot files
        """
        logger.info(f"Generating plots for section: {section.title}")
        
        if not section.metrics:
            logger.warning(f"No metrics found in section '{section.title}'. Skipping plot generation.")
            return []
            
        plot_paths = []
        
        # For the MVP, generate 1-2 standard plots per section based on section type
        if section.title == "System Overview":
            # For overview, we might not need plots in the MVP
            pass
            
        elif section.title == "DW Gate Analysis":
            # Generate plots specific to DW gate metrics
            open_fraction_metrics = [m for m in section.metrics if "Open_Fraction" in m.name]
            if open_fraction_metrics:
                plot_path = self._create_bar_comparison_plot(
                    metrics=open_fraction_metrics[:4],  # Limit to avoid too many bars
                    title="DW Gate Opening Fractions",
                    ylabel="Open Fraction (%)",
                    filename=f"{section.title.lower().replace(' ', '_')}_open_fractions.png"
                )
                if plot_path:
                    plot_paths.append(plot_path)
                    
        elif section.title == "Ion Pathway Analysis":
            # Create plots for ion pathway metrics
            occupancy_metrics = [m for m in section.metrics if "Occ" in m.name]
            if occupancy_metrics:
                plot_path = self._create_grouped_bar_plot(
                    metrics=occupancy_metrics[:6],  # Limit to top sites
                    title="Ion Occupancy by Site",
                    ylabel="Average Occupancy",
                    filename=f"{section.title.lower().replace(' ', '_')}_occupancy.png"
                )
                if plot_path:
                    plot_paths.append(plot_path)
                    
            conduction_metrics = [m for m in section.metrics if "Conduction" in m.name]
            if conduction_metrics:
                plot_path = self._create_bar_comparison_plot(
                    metrics=conduction_metrics[:3],
                    title="Ion Conduction Metrics",
                    ylabel="Value",
                    filename=f"{section.title.lower().replace(' ', '_')}_conduction.png"
                )
                if plot_path:
                    plot_paths.append(plot_path)
                    
        else:
            # Generic plots for other section types
            # Choose top 5 metrics by priority score
            top_metrics = sorted(section.metrics, key=lambda m: m.priority_score, reverse=True)[:5]
            if top_metrics:
                plot_path = self._create_bar_comparison_plot(
                    metrics=top_metrics,
                    title=f"Top Metrics for {section.title}",
                    ylabel="Value",
                    filename=f"{section.title.lower().replace(' ', '_')}_top_metrics.png"
                )
                if plot_path:
                    plot_paths.append(plot_path)
                    
        logger.info(f"Generated {len(plot_paths)} plots for section '{section.title}'")
        return plot_paths

    def _create_bar_comparison_plot(self, metrics: List[MetricResult], title: str, ylabel: str, filename: str) -> Optional[Path]:
        """
        Create a simple bar plot comparing toxin vs control for each metric.
        
        Args:
            metrics: List of MetricResult objects to plot
            title: Plot title
            ylabel: Y-axis label
            filename: Output filename
            
        Returns:
            Path to the saved plot file or None if generation failed
        """
        try:
            if not metrics:
                return None
                
            # Set up the plot
            fig, ax = plt.subplots(figsize=(12, 7))
            
            # Calculate positions for bars
            num_metrics = len(metrics)
            index = np.arange(num_metrics)
            bar_width = 0.35
            
            # Create bars
            toxin_bars = ax.bar(
                index - bar_width/2, 
                [m.value_toxin for m in metrics], 
                bar_width,
                label='Toxin-Bound', 
                color=self.toxin_color, 
                alpha=0.8
            )
            
            control_bars = ax.bar(
                index + bar_width/2, 
                [m.value_control for m in metrics], 
                bar_width,
                label='Control', 
                color=self.control_color, 
                alpha=0.8
            )
            
            # Add labels and title
            ax.set_xlabel('Metric')
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.set_xticks(index)
            
            # Format x-tick labels: use shortened metric names
            x_labels = []
            for m in metrics:
                # Format depends on metric name pattern
                parts = m.name.split('_')
                if len(parts) > 2:
                    # Use last 2 parts for most metrics (e.g., DW_PROA_Open_Fraction -> Open_Fraction)
                    label = '_'.join(parts[-2:])
                else:
                    label = m.name
                x_labels.append(label)
                
            ax.set_xticklabels(x_labels, rotation=45, ha='right')
            
            # Add legend
            ax.legend()
            
            # Add significance indicators
            for i, metric in enumerate(metrics):
                if metric.significant:
                    # Add a star above the bars for significant differences
                    max_val = max(metric.value_toxin, metric.value_control)
                    ax.text(i, max_val * 1.05, '*', ha='center', fontsize=16)
                    
            # Add values on bars
            for i, bar in enumerate(toxin_bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height * 1.02,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=9)
                        
            for i, bar in enumerate(control_bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height * 1.02,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=9)
            
            # Adjust layout and save
            plt.tight_layout()
            output_path = self.output_plots_dir / filename
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            
            logger.info(f"Created bar comparison plot: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating bar comparison plot: {e}")
            return None

    def _create_grouped_bar_plot(self, metrics: List[MetricResult], title: str, ylabel: str, filename: str) -> Optional[Path]:
        """
        Create a grouped bar plot for metrics with similar patterns.
        
        Args:
            metrics: List of MetricResult objects to plot
            title: Plot title
            ylabel: Y-axis label
            filename: Output filename
            
        Returns:
            Path to the saved plot file or None if generation failed
        """
        try:
            if not metrics:
                return None
                
            # Set up the plot
            fig, ax = plt.subplots(figsize=(12, 7))
            
            # Calculate positions for bars
            num_metrics = len(metrics)
            index = np.arange(num_metrics)
            bar_width = 0.35
            
            # Create bars
            toxin_bars = ax.bar(
                index - bar_width/2, 
                [m.value_toxin for m in metrics], 
                bar_width,
                label='Toxin-Bound', 
                color=self.toxin_color, 
                alpha=0.8
            )
            
            control_bars = ax.bar(
                index + bar_width/2, 
                [m.value_control for m in metrics], 
                bar_width,
                label='Control', 
                color=self.control_color, 
                alpha=0.8
            )
            
            # Add labels and title
            ax.set_xlabel('Site/Location')
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.set_xticks(index)
            
            # Extract site names from metrics
            # For ion occupancy, names often have site identifiers (S0, S1, etc.)
            x_labels = []
            for m in metrics:
                parts = m.name.split('_')
                site_part = None
                for part in parts:
                    if any(site_marker in part for site_marker in ['S0', 'S1', 'S2', 'S3', 'S4', 'S5', 'SF', 'Cav']):
                        site_part = part
                        break
                        
                if site_part:
                    x_labels.append(site_part)
                else:
                    # Fallback to last part of name
                    x_labels.append(parts[-1])
                    
            ax.set_xticklabels(x_labels)
            
            # Add legend
            ax.legend()
            
            # Add significance indicators
            for i, metric in enumerate(metrics):
                if metric.significant:
                    # Add a star above the bars for significant differences
                    max_val = max(metric.value_toxin, metric.value_control)
                    ax.text(i, max_val * 1.05, '*', ha='center', fontsize=16)
            
            # Adjust layout and save
            plt.tight_layout()
            output_path = self.output_plots_dir / filename
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            
            logger.info(f"Created grouped bar plot: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating grouped bar plot: {e}")
            return None