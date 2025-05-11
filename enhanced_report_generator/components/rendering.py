"""
Rendering component for the enhanced report generator.

This module is responsible for:
1. Rendering HTML reports from templates
2. Converting HTML reports to PDF (simple implementation for MVP)
3. Managing template loading and rendering
"""

import logging
import shutil
import datetime
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from jinja2 import Environment, FileSystemLoader

from ..core.data_models import ReportSection, RunMetadata
from .stats_analysis import generate_statistical_report

logger = logging.getLogger(__name__)


class ReportLayoutRenderer:
    """
    Renders report templates with data into final reports.
    """
    
    def __init__(self, templates_dir: Path, output_dir: Path, assets_dir: Path):
        """
        Initialize the ReportLayoutRenderer with directories for templates, output, and assets.
        
        Args:
            templates_dir: Directory containing HTML templates
            output_dir: Directory where reports will be saved
            assets_dir: Directory containing assets (CSS, JS, images)
        """
        self.templates_dir = templates_dir
        self.output_dir = output_dir
        self.assets_dir = assets_dir
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up Jinja2 environment
        self.env = Environment(
            loader=FileSystemLoader(self.templates_dir),
            autoescape=True
        )
        
        logger.info(f"ReportLayoutRenderer initialized with templates_dir={templates_dir}, output_dir={output_dir}")

    def render_html_report(self,
                           report_title: str,
                           sections: List[ReportSection],
                           run_metadata: RunMetadata,
                           global_ai_summary: Optional[str] = None,
                           template_name: str = "system_overview_template.html",
                           output_filename: Optional[str] = None,
                           enhanced_data: Optional[Dict] = None) -> Path:
        """
        Render an HTML report using a template.

        Args:
            report_title: Title of the report
            sections: List of ReportSection objects to include in the report
            run_metadata: RunMetadata object with information about the runs
            global_ai_summary: Optional global AI-generated summary for the report
            template_name: Name of the template file to use
            output_filename: Optional custom output filename
            enhanced_data: Optional dictionary with additional data for enhanced templates

        Returns:
            Path to the generated HTML report
        """
        logger.info(f"Rendering HTML report using template: {template_name}")

        # Load the template
        template = self.env.get_template(template_name)

        # Generate timestamp
        generation_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Prepare context for template
        context = {
            "report_title": report_title,
            "sections": sections,
            "section": sections[0] if sections else None,  # For single-section templates
            "run_metadata": run_metadata,
            "global_ai_summary": global_ai_summary,
            "generation_timestamp": generation_timestamp
        }

        # Add enhanced data if provided
        if enhanced_data:
            context.update(enhanced_data)

        # Render the template
        html_content = template.render(**context)

        # Determine output filename
        if not output_filename:
            safe_title = report_title.lower().replace(" ", "_").replace("/", "_")
            output_filename = f"{safe_title}_{generation_timestamp.replace(' ', '_').replace(':', '-')}.html"

        # Write the output file
        output_path = self.output_dir / output_filename
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        # Copy assets to output directory
        self._copy_assets()

        logger.info(f"HTML report generated at: {output_path}")
        return output_path

    def _prepare_data_for_react_component(self, gg_data_raw: Dict, gg_sim_points_raw: Dict) -> Dict:
        """
        Transforms raw data from _prepare_gg_distance_data and _prepare_gg_simulation_points
        into the structures expected by the GGDistanceTabbedComparison React component.

        Args:
            gg_data_raw: Raw data dictionary from _prepare_gg_distance_data
            gg_sim_points_raw: Raw data dictionary from _prepare_gg_simulation_points

        Returns:
            Dictionary containing JSON-formatted data ready for React component
        """
        logger.info("Preparing data for React GG distance comparison component")

        # 1. Prepare 'mockData' structure for the React component
        react_mock_data = {
            "toxin": gg_data_raw.get('toxin', {}),
            "control": gg_data_raw.get('control', {})
        }

        # 2. Prepare 'statisticalData' structure
        react_statistical_data = {
            "mean": {}, "min": {}, "max": {}, "range": {}
        }

        raw_p_values = gg_data_raw.get('p_values', {})
        raw_effect_sizes = gg_data_raw.get('effect_sizes', {})

        for metric_type in ["mean", "min", "max", "range"]:
            for subunit_pair in ["AC", "BD"]:
                # Keys for toxin data
                toxin_key = f"toxin-{subunit_pair}"  # e.g. "toxin-AC"

                p_value = raw_p_values.get(subunit_pair, {}).get(metric_type)
                effect_size = raw_effect_sizes.get(subunit_pair, {}).get(metric_type)

                if p_value is not None and effect_size is not None:
                    if metric_type not in react_statistical_data:
                        react_statistical_data[metric_type] = {}
                    react_statistical_data[metric_type][toxin_key] = {
                        "pValue": p_value,
                        "effectSize": effect_size
                    }

        # 3. Simulation points should already be in correct structure
        react_simulation_points = gg_sim_points_raw

        return {
            "gg_react_mock_data_json": json.dumps(react_mock_data),
            "gg_react_statistical_data_json": json.dumps(react_statistical_data),
            "gg_react_simulation_points_json": json.dumps(react_simulation_points),
            "gg_significance_notes": gg_data_raw.get('significance_notes', {}),
            "gg_num_systems": gg_data_raw.get('num_systems', 'N/A')
        }

    def render_tabbed_report(self,
                             report_title: str,
                             sections: List[ReportSection],
                             run_metadata: RunMetadata,
                             db_path: Path,
                             output_filename: Optional[str] = None) -> Path:
        """
        Render a tabbed HTML report with multiple analysis sections.

        Args:
            report_title: Title of the report
            sections: List of ReportSection objects to include in the report
            run_metadata: RunMetadata object with information about the runs
            db_path: Path to the database for extracting additional data
            output_filename: Optional custom output filename

        Returns:
            Path to the generated HTML report
        """
        logger.info(f"Rendering tabbed HTML report for multiple analyses")

        # Prepare G-G distance data for the Pore Geometry tab
        gg_data_raw = self._prepare_gg_distance_data(db_path, run_metadata)
        gg_sim_points_raw = self._prepare_gg_simulation_points(db_path, run_metadata)

        # Transform data for the React component
        react_data = self._prepare_data_for_react_component(gg_data_raw, gg_sim_points_raw)

        # Prepare data for the new fluctuation tab
        gg_fluctuation_data = self._prepare_gg_fluctuation_data(db_path, run_metadata)

        # Generate timestamp
        generation_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Determine output filename
        if not output_filename:
            safe_title = report_title.lower().replace(" ", "_").replace("/", "_")
            output_filename = f"{safe_title}.html"

        # Call render_html_report with enhanced data
        return self.render_html_report(
            report_title=report_title,
            sections=sections,
            run_metadata=run_metadata,
            template_name="tabbed_report_template.html",
            output_filename=output_filename,
            enhanced_data={
                # Keep original data for backward compatibility
                "gg_data": gg_data_raw,
                "gg_sim_points": gg_sim_points_raw,

                # Add React component data
                "gg_react_mock_data_json": react_data["gg_react_mock_data_json"],
                "gg_react_statistical_data_json": react_data["gg_react_statistical_data_json"],
                "gg_react_simulation_points_json": react_data["gg_react_simulation_points_json"],
                "gg_significance_notes": react_data["gg_significance_notes"],
                "gg_num_systems": react_data["gg_num_systems"],

                # Add data for G-G fluctuation tab
                "gg_fluctuation_summary": gg_fluctuation_data["summary_fluctuations"],
                "gg_fluctuation_stats": gg_fluctuation_data["stats"],
                "gg_react_fluctuation_summary_json": json.dumps(gg_fluctuation_data["summary_fluctuations"]),
                "gg_react_fluctuation_individual_json": json.dumps(gg_fluctuation_data["individual_fluctuations"]),
                "gg_react_fluctuation_stats_json": json.dumps(gg_fluctuation_data["stats"]),
                "gg_fluctuation_significance_notes": gg_fluctuation_data["significance_notes"]
            }
        )

    def _prepare_gg_distance_data(self, db_path: Path, run_metadata: RunMetadata) -> Dict:
        """
        Prepare G-G distance data for the Pore Geometry tab.

        Args:
            db_path: Path to the database
            run_metadata: RunMetadata object with information about the runs

        Returns:
            Dictionary with G-G distance metrics for toxin and control systems
        """
        logger.info(f"Preparing G-G distance data from {db_path}")

        import sqlite3
        import numpy as np

        # Connect to the database
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Get toxin and control system IDs
        cursor.execute("SELECT system_id FROM systems WHERE group_type = 'toxin-bound'")
        toxin_system_ids = [row['system_id'] for row in cursor.fetchall()]

        cursor.execute("SELECT system_id FROM systems WHERE group_type = 'toxin-free'")
        control_system_ids = [row['system_id'] for row in cursor.fetchall()]

        # Group metrics for toxin systems
        toxin_metrics = {
            'AC': {
                'mean': [],
                'min': [],
                'max': []
            },
            'BD': {
                'mean': [],
                'min': [],
                'max': []
            }
        }

        # Group metrics for control systems
        control_metrics = {
            'AC': {
                'mean': [],
                'min': [],
                'max': []
            },
            'BD': {
                'mean': [],
                'min': [],
                'max': []
            }
        }

        # Get G-G distance metrics for toxin systems
        for system_id in toxin_system_ids:
            cursor.execute("""
                SELECT metric_name, value FROM aggregated_metrics
                WHERE system_id = ? AND metric_name IN
                ('G_G_AC_Mean_Filt', 'G_G_BD_Mean_Filt', 'G_G_AC_Min_Filt', 'G_G_BD_Min_Filt',
                 'G_G_AC_Max_Filt', 'G_G_BD_Max_Filt')
            """, (system_id,))

            for row in cursor.fetchall():
                if 'G_G_AC_Mean_Filt' in row['metric_name']:
                    toxin_metrics['AC']['mean'].append(row['value'])
                elif 'G_G_BD_Mean_Filt' in row['metric_name']:
                    toxin_metrics['BD']['mean'].append(row['value'])
                elif 'G_G_AC_Min_Filt' in row['metric_name']:
                    toxin_metrics['AC']['min'].append(row['value'])
                elif 'G_G_BD_Min_Filt' in row['metric_name']:
                    toxin_metrics['BD']['min'].append(row['value'])
                elif 'G_G_AC_Max_Filt' in row['metric_name']:
                    toxin_metrics['AC']['max'].append(row['value'])
                elif 'G_G_BD_Max_Filt' in row['metric_name']:
                    toxin_metrics['BD']['max'].append(row['value'])

        # Get G-G distance metrics for control systems
        for system_id in control_system_ids:
            cursor.execute("""
                SELECT metric_name, value FROM aggregated_metrics
                WHERE system_id = ? AND metric_name IN
                ('G_G_AC_Mean_Filt', 'G_G_BD_Mean_Filt', 'G_G_AC_Min_Filt', 'G_G_BD_Min_Filt',
                 'G_G_AC_Max_Filt', 'G_G_BD_Max_Filt')
            """, (system_id,))

            for row in cursor.fetchall():
                if 'G_G_AC_Mean_Filt' in row['metric_name']:
                    control_metrics['AC']['mean'].append(row['value'])
                elif 'G_G_BD_Mean_Filt' in row['metric_name']:
                    control_metrics['BD']['mean'].append(row['value'])
                elif 'G_G_AC_Min_Filt' in row['metric_name']:
                    control_metrics['AC']['min'].append(row['value'])
                elif 'G_G_BD_Min_Filt' in row['metric_name']:
                    control_metrics['BD']['min'].append(row['value'])
                elif 'G_G_AC_Max_Filt' in row['metric_name']:
                    control_metrics['AC']['max'].append(row['value'])
                elif 'G_G_BD_Max_Filt' in row['metric_name']:
                    control_metrics['BD']['max'].append(row['value'])

        # Calculate summary statistics for toxin systems
        toxin_summary = {
            'AC': {
                'mean': {'value': np.mean(toxin_metrics['AC']['mean']), 'stdDev': np.std(toxin_metrics['AC']['mean'])},
                'min': {'value': np.mean(toxin_metrics['AC']['min']), 'stdDev': np.std(toxin_metrics['AC']['min'])},
                'max': {'value': np.mean(toxin_metrics['AC']['max']), 'stdDev': np.std(toxin_metrics['AC']['max'])},
                'range': {'value': np.mean([m-n for m, n in zip(toxin_metrics['AC']['max'], toxin_metrics['AC']['min'])]),
                         'stdDev': np.std([m-n for m, n in zip(toxin_metrics['AC']['max'], toxin_metrics['AC']['min'])])}
            },
            'BD': {
                'mean': {'value': np.mean(toxin_metrics['BD']['mean']), 'stdDev': np.std(toxin_metrics['BD']['mean'])},
                'min': {'value': np.mean(toxin_metrics['BD']['min']), 'stdDev': np.std(toxin_metrics['BD']['min'])},
                'max': {'value': np.mean(toxin_metrics['BD']['max']), 'stdDev': np.std(toxin_metrics['BD']['max'])},
                'range': {'value': np.mean([m-n for m, n in zip(toxin_metrics['BD']['max'], toxin_metrics['BD']['min'])]),
                         'stdDev': np.std([m-n for m, n in zip(toxin_metrics['BD']['max'], toxin_metrics['BD']['min'])])}
            }
        }

        # Calculate summary statistics for control systems
        control_summary = {
            'AC': {
                'mean': {'value': np.mean(control_metrics['AC']['mean']), 'stdDev': np.std(control_metrics['AC']['mean'])},
                'min': {'value': np.mean(control_metrics['AC']['min']), 'stdDev': np.std(control_metrics['AC']['min'])},
                'max': {'value': np.mean(control_metrics['AC']['max']), 'stdDev': np.std(control_metrics['AC']['max'])},
                'range': {'value': np.mean([m-n for m, n in zip(control_metrics['AC']['max'], control_metrics['AC']['min'])]),
                         'stdDev': np.std([m-n for m, n in zip(control_metrics['AC']['max'], control_metrics['AC']['min'])])}
            },
            'BD': {
                'mean': {'value': np.mean(control_metrics['BD']['mean']), 'stdDev': np.std(control_metrics['BD']['mean'])},
                'min': {'value': np.mean(control_metrics['BD']['min']), 'stdDev': np.std(control_metrics['BD']['min'])},
                'max': {'value': np.mean(control_metrics['BD']['max']), 'stdDev': np.std(control_metrics['BD']['max'])},
                'range': {'value': np.mean([m-n for m, n in zip(control_metrics['BD']['max'], control_metrics['BD']['min'])]),
                         'stdDev': np.std([m-n for m, n in zip(control_metrics['BD']['max'], control_metrics['BD']['min'])])}
            }
        }

        # Generate statistical report
        stats_report = generate_statistical_report(toxin_metrics, control_metrics)

        # Add result to the summary data
        return {
            'toxin': toxin_summary,
            'control': control_summary,
            'p_values': stats_report['p_values'],
            'effect_sizes': stats_report['effect_sizes'],
            'significance_notes': stats_report['significance_notes'],
            'num_systems': min(len(toxin_system_ids), len(control_system_ids))
        }

    def _prepare_gg_simulation_points(self, db_path: Path, run_metadata: RunMetadata) -> Dict:
        """
        Prepare individual simulation points for the G-G distance visualization.

        Args:
            db_path: Path to the database
            run_metadata: RunMetadata object with information about the runs

        Returns:
            Dictionary with individual simulation data points
        """
        logger.info(f"Preparing G-G simulation points from {db_path}")

        import sqlite3

        # Connect to the database
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Get toxin system data
        cursor.execute("""
            SELECT s.run_name, m.metric_name, m.value
            FROM aggregated_metrics m
            JOIN systems s ON m.system_id = s.system_id
            WHERE s.group_type = 'toxin-bound'
            AND m.metric_name IN
            ('G_G_AC_Mean_Filt', 'G_G_BD_Mean_Filt', 'G_G_AC_Min_Filt', 'G_G_BD_Min_Filt',
             'G_G_AC_Max_Filt', 'G_G_BD_Max_Filt')
            ORDER BY s.run_name, m.metric_name
        """)

        # Organize toxin system data
        toxin_points = {
            'AC': {
                'mean': [],
                'min': [],
                'max': [],
                'range': []
            },
            'BD': {
                'mean': [],
                'min': [],
                'max': [],
                'range': []
            }
        }

        # Track metric pairs for range calculation
        toxin_pairs = {}

        for row in cursor.fetchall():
            if 'G_G_AC_Mean_Filt' in row['metric_name']:
                toxin_points['AC']['mean'].append(row['value'])
            elif 'G_G_BD_Mean_Filt' in row['metric_name']:
                toxin_points['BD']['mean'].append(row['value'])
            elif 'G_G_AC_Min_Filt' in row['metric_name']:
                toxin_points['AC']['min'].append(row['value'])
                if row['run_name'] not in toxin_pairs:
                    toxin_pairs[row['run_name']] = {}
                toxin_pairs[row['run_name']]['AC_min'] = row['value']
            elif 'G_G_BD_Min_Filt' in row['metric_name']:
                toxin_points['BD']['min'].append(row['value'])
                if row['run_name'] not in toxin_pairs:
                    toxin_pairs[row['run_name']] = {}
                toxin_pairs[row['run_name']]['BD_min'] = row['value']
            elif 'G_G_AC_Max_Filt' in row['metric_name']:
                toxin_points['AC']['max'].append(row['value'])
                if row['run_name'] not in toxin_pairs:
                    toxin_pairs[row['run_name']] = {}
                toxin_pairs[row['run_name']]['AC_max'] = row['value']
            elif 'G_G_BD_Max_Filt' in row['metric_name']:
                toxin_points['BD']['max'].append(row['value'])
                if row['run_name'] not in toxin_pairs:
                    toxin_pairs[row['run_name']] = {}
                toxin_pairs[row['run_name']]['BD_max'] = row['value']

        # Calculate ranges for toxin systems
        for run_name, metrics in toxin_pairs.items():
            if 'AC_max' in metrics and 'AC_min' in metrics:
                toxin_points['AC']['range'].append(metrics['AC_max'] - metrics['AC_min'])
            if 'BD_max' in metrics and 'BD_min' in metrics:
                toxin_points['BD']['range'].append(metrics['BD_max'] - metrics['BD_min'])

        # Get control system data
        cursor.execute("""
            SELECT s.run_name, m.metric_name, m.value
            FROM aggregated_metrics m
            JOIN systems s ON m.system_id = s.system_id
            WHERE s.group_type = 'toxin-free'
            AND m.metric_name IN
            ('G_G_AC_Mean_Filt', 'G_G_BD_Mean_Filt', 'G_G_AC_Min_Filt', 'G_G_BD_Min_Filt',
             'G_G_AC_Max_Filt', 'G_G_BD_Max_Filt')
            ORDER BY s.run_name, m.metric_name
        """)

        # Organize control system data
        control_points = {
            'AC': {
                'mean': [],
                'min': [],
                'max': [],
                'range': []
            },
            'BD': {
                'mean': [],
                'min': [],
                'max': [],
                'range': []
            }
        }

        # Track metric pairs for range calculation
        control_pairs = {}

        for row in cursor.fetchall():
            if 'G_G_AC_Mean_Filt' in row['metric_name']:
                control_points['AC']['mean'].append(row['value'])
            elif 'G_G_BD_Mean_Filt' in row['metric_name']:
                control_points['BD']['mean'].append(row['value'])
            elif 'G_G_AC_Min_Filt' in row['metric_name']:
                control_points['AC']['min'].append(row['value'])
                if row['run_name'] not in control_pairs:
                    control_pairs[row['run_name']] = {}
                control_pairs[row['run_name']]['AC_min'] = row['value']
            elif 'G_G_BD_Min_Filt' in row['metric_name']:
                control_points['BD']['min'].append(row['value'])
                if row['run_name'] not in control_pairs:
                    control_pairs[row['run_name']] = {}
                control_pairs[row['run_name']]['BD_min'] = row['value']
            elif 'G_G_AC_Max_Filt' in row['metric_name']:
                control_points['AC']['max'].append(row['value'])
                if row['run_name'] not in control_pairs:
                    control_pairs[row['run_name']] = {}
                control_pairs[row['run_name']]['AC_max'] = row['value']
            elif 'G_G_BD_Max_Filt' in row['metric_name']:
                control_points['BD']['max'].append(row['value'])
                if row['run_name'] not in control_pairs:
                    control_pairs[row['run_name']] = {}
                control_pairs[row['run_name']]['BD_max'] = row['value']

        # Calculate ranges for control systems
        for run_name, metrics in control_pairs.items():
            if 'AC_max' in metrics and 'AC_min' in metrics:
                control_points['AC']['range'].append(metrics['AC_max'] - metrics['AC_min'])
            if 'BD_max' in metrics and 'BD_min' in metrics:
                control_points['BD']['range'].append(metrics['BD_max'] - metrics['BD_min'])

        return {
            'toxin': toxin_points,
            'control': control_points
        }

    def _prepare_gg_fluctuation_data(self, db_path: Path, run_metadata: RunMetadata) -> Dict:
        """
        Prepare G-G distance fluctuation data (standard deviation metrics) for the fluctuation tab.

        Args:
            db_path: Path to the database
            run_metadata: RunMetadata object with information about the runs

        Returns:
            Dictionary with G-G distance fluctuation metrics and statistics
        """
        logger.info(f"Preparing G-G fluctuation data from {db_path}")

        import sqlite3
        import numpy as np
        from .stats_analysis import calculate_p_value, calculate_effect_size

        # Connect to the database
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Define the metrics we're looking for (standard deviation metrics)
        metric_names = {
            "AC": "G_G_AC_Std_Filt",  # Standard deviation for AC subunit
            "BD": "G_G_BD_Std_Filt"   # Standard deviation for BD subunit
        }

        # Initialize data structures
        fluctuation_data = {
            'toxin': {'AC': [], 'BD': []},
            'control': {'AC': [], 'BD': []}
        }

        # Get toxin system data
        for subunit, metric_name in metric_names.items():
            cursor.execute("""
                SELECT s.run_name, m.value
                FROM aggregated_metrics m
                JOIN systems s ON m.system_id = s.system_id
                WHERE s.group_type = 'toxin-bound'
                AND m.metric_name = ?
                ORDER BY s.run_name
            """, (metric_name,))

            for row in cursor.fetchall():
                fluctuation_data['toxin'][subunit].append(row['value'])

        # Get control system data
        for subunit, metric_name in metric_names.items():
            cursor.execute("""
                SELECT s.run_name, m.value
                FROM aggregated_metrics m
                JOIN systems s ON m.system_id = s.system_id
                WHERE s.group_type = 'toxin-free'
                AND m.metric_name = ?
                ORDER BY s.run_name
            """, (metric_name,))

            for row in cursor.fetchall():
                fluctuation_data['control'][subunit].append(row['value'])

        # Calculate summary statistics for each group and subunit
        summary_stats = {'toxin': {}, 'control': {}}

        for group in ['toxin', 'control']:
            for subunit in ['AC', 'BD']:
                values = fluctuation_data[group][subunit]
                # Handle case where we might have no data
                if not values:
                    summary_stats[group][subunit] = {
                        'value': np.nan,
                        'stdDev': np.nan
                    }
                else:
                    summary_stats[group][subunit] = {
                        'value': np.mean(values),         # Mean of std deviations
                        'stdDev': np.std(values, ddof=1)  # Std dev of std deviations
                    }

        # Calculate p-values and effect sizes for toxin vs control
        p_values = {}
        effect_sizes = {}

        for subunit in ['AC', 'BD']:
            toxin_values = fluctuation_data['toxin'][subunit]
            control_values = fluctuation_data['control'][subunit]

            # Only calculate if we have enough data points
            if len(toxin_values) > 1 and len(control_values) > 1:
                p_values[subunit] = calculate_p_value(toxin_values, control_values)
                effect_sizes[subunit] = calculate_effect_size(toxin_values, control_values)
            else:
                p_values[subunit] = None
                effect_sizes[subunit] = None

        # Generate significance interpretations
        if p_values['AC'] is not None and p_values['BD'] is not None:
            from ..components.stats_analysis import interpret_p_value, interpret_effect_size
            significance_notes = {
                'AC': f"The difference in G-G distance fluctuations between toxin and control is {interpret_p_value(p_values['AC'])} for the A:C subunit pair with {interpret_effect_size(effect_sizes['AC'])} effect size.",
                'BD': f"The difference in G-G distance fluctuations between toxin and control is {interpret_p_value(p_values['BD'])} for the B:D subunit pair with {interpret_effect_size(effect_sizes['BD'])} effect size."
            }
        else:
            significance_notes = {
                'AC': "Not enough data for statistical analysis of A:C fluctuations.",
                'BD': "Not enough data for statistical analysis of B:D fluctuations."
            }

        return {
            "summary_fluctuations": summary_stats,                # For bar chart
            "individual_fluctuations": fluctuation_data,          # For scatter points
            "stats": {
                "p_values": p_values,                             # For significance
                "effect_sizes": effect_sizes                      # For effect size
            },
            "significance_notes": significance_notes,             # For text description
            "num_systems": min(len(run_metadata.toxin_run_ids), len(run_metadata.control_run_ids))
        }

    def _copy_assets(self):
        """Copy assets to the output directory."""
        output_assets_dir = self.output_dir / "assets"
        output_assets_dir.mkdir(exist_ok=True)
        
        # Copy CSS files
        for css_file in self.assets_dir.glob("*.css"):
            shutil.copy(css_file, output_assets_dir)
            
        # Copy JS files
        for js_file in self.assets_dir.glob("*.js"):
            shutil.copy(js_file, output_assets_dir)
            
        # Copy image files
        for img_file in self.assets_dir.glob("*.{png,jpg,jpeg,gif,svg}"):
            shutil.copy(img_file, output_assets_dir)
            
        logger.info(f"Assets copied to {output_assets_dir}")

    def render_pdf_report(self, html_report_path: Path) -> Optional[Path]:
        """
        Convert an HTML report to PDF.
        
        Args:
            html_report_path: Path to the HTML report
            
        Returns:
            Path to the generated PDF report or None if conversion failed
        """
        logger.info(f"Converting HTML report to PDF: {html_report_path}")
        
        # Simple implementation for MVP - just log that PDF creation would happen here
        # In a full implementation, this would use a library like weasyprint or a service like wkhtmltopdf
        
        pdf_path = html_report_path.with_suffix('.pdf')
        
        try:
            # Placeholder for actual PDF generation
            # weasyprint.HTML(filename=str(html_report_path)).write_pdf(str(pdf_path))
            
            # For MVP, just create an empty file or log that this would be implemented
            logger.info(f"PDF generation not implemented in MVP. Would create: {pdf_path}")
            
            # Uncomment to create empty PDF placeholder
            # with open(pdf_path, 'w') as f:
            #     f.write("PDF placeholder - conversion not implemented in MVP")
            
            return pdf_path
        except Exception as e:
            logger.error(f"Error generating PDF: {e}")
            return None
            
    def generate_streamlit_dashboard_data(self, sections: List[ReportSection], output_path: Path) -> bool:
        """
        Generate data files for a Streamlit dashboard.
        
        Args:
            sections: List of ReportSection objects to include in the dashboard
            output_path: Path to save the dashboard data
            
        Returns:
            True if generation was successful, False otherwise
        """
        logger.info(f"Generating Streamlit dashboard data: {output_path}")
        
        # Placeholder for MVP - in full implementation, would generate JSON/CSV data for Streamlit
        
        try:
            # Placeholder for actual dashboard data generation
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Just create a placeholder text file for MVP
            with open(output_path / "dashboard_data.txt", "w") as f:
                f.write(f"Dashboard data would go here. Generated on {datetime.datetime.now()}\n")
                f.write(f"Would include {len(sections)} sections of data.\n")
                
            logger.info(f"Streamlit dashboard data placeholder created at: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error generating Streamlit dashboard data: {e}")
            return False