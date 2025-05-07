"""
HTML report generation module for cross-analysis suite.

This module creates a comprehensive HTML report comparing toxin-bound
and control systems, integrating metrics, plots, and AI insights.
"""

import os
import sqlite3
import json
import logging
import base64
import markdown
from typing import Dict, List, Any, Optional
from datetime import datetime
from jinja2 import Environment, FileSystemLoader, select_autoescape

# Configure logging
logger = logging.getLogger(__name__)

class HTMLReportGenerator:
    """
    Generates HTML reports for cross-system analysis.
    """
    
    def __init__(self, meta_db_path: str, output_dir: str, 
                template_dir: Optional[str] = None):
        """
        Initialize the HTML report generator.
        
        Args:
            meta_db_path: Path to the meta-database file
            output_dir: Directory for saving the report
            template_dir: Optional directory containing custom templates
        """
        self.meta_db_path = meta_db_path
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up template directory
        if template_dir and os.path.exists(template_dir):
            self.template_dir = template_dir
        else:
            # Use default template directory
            self.template_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 'templates'
            )
            
            # Create default template directory if it doesn't exist
            if not os.path.exists(self.template_dir):
                os.makedirs(self.template_dir, exist_ok=True)
                
                # Create default template if it doesn't exist
                self._create_default_template()
        
        self.meta_conn = None
    
    def _create_default_template(self):
        """Create default template files if they don't exist."""
        # Main template
        main_template_path = os.path.join(self.template_dir, 'cross_report_template.html')
        if not os.path.exists(main_template_path):
            with open(main_template_path, 'w') as f:
                f.write("""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        h1, h2, h3, h4 {
            color: #2c3e50;
        }
        h1 {
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }
        h2 {
            border-bottom: 1px solid #ddd;
            padding-bottom: 5px;
            margin-top: 30px;
        }
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 30px;
        }
        .header-info {
            flex: 1;
        }
        .timestamp {
            color: #7f8c8d;
            font-size: 0.9em;
        }
        .summary-box {
            background-color: #f8f9fa;
            border-left: 4px solid #3498db;
            padding: 15px;
            margin: 20px 0;
            border-radius: 4px;
        }
        .metric-comparison {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            margin: 20px 0;
        }
        .metric-card {
            flex: 1;
            min-width: 300px;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 15px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .metric-title {
            font-size: 1.2em;
            font-weight: bold;
            margin-bottom: 10px;
            color: #2980b9;
        }
        .plot-container {
            margin: 20px 0;
            text-align: center;
        }
        .plot-container img {
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            padding: 12px 15px;
            border-bottom: 1px solid #ddd;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
            font-weight: bold;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
        .significant {
            color: #e74c3c;
            font-weight: bold;
        }
        .tabs {
            display: flex;
            margin-top: 20px;
            border-bottom: 1px solid #ddd;
        }
        .tab {
            padding: 10px 20px;
            cursor: pointer;
            border: 1px solid #ddd;
            border-bottom: none;
            border-radius: 5px 5px 0 0;
            margin-right: 5px;
            background-color: #f8f9fa;
        }
        .tab.active {
            background-color: #fff;
            border-bottom: 1px solid #fff;
            margin-bottom: -1px;
            font-weight: bold;
        }
        .tab-content {
            display: none;
            padding: 20px;
            border: 1px solid #ddd;
            border-top: none;
            border-radius: 0 0 5px 5px;
        }
        .tab-content.active {
            display: block;
        }
        .insight-box {
            background-color: #eff8ff;
            border-left: 4px solid #3498db;
            padding: 15px;
            margin: 20px 0;
            border-radius: 4px;
        }
        .insight-title {
            font-weight: bold;
            color: #2980b9;
            margin-bottom: 10px;
        }
        .badge {
            display: inline-block;
            padding: 3px 7px;
            border-radius: 3px;
            font-size: 0.8em;
            font-weight: bold;
            margin-right: 5px;
        }
        .badge-toxin {
            background-color: #e74c3c;
            color: white;
        }
        .badge-control {
            background-color: #2ecc71;
            color: white;
        }
        .category-badge {
            background-color: #f1c40f;
            color: #333;
        }
    </style>
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            // Tab functionality
            const tabs = document.querySelectorAll('.tab');
            const tabContents = document.querySelectorAll('.tab-content');
            
            tabs.forEach(tab => {
                tab.addEventListener('click', () => {
                    // Remove active class from all tabs and contents
                    tabs.forEach(t => t.classList.remove('active'));
                    tabContents.forEach(c => c.classList.remove('active'));
                    
                    // Add active class to clicked tab and corresponding content
                    tab.classList.add('active');
                    const tabId = tab.getAttribute('data-tab');
                    document.getElementById(tabId).classList.add('active');
                });
            });
            
            // Activate first tab by default
            if (tabs.length > 0) {
                tabs[0].click();
            }
        });
    </script>
</head>
<body>
    <div class="header">
        <div class="header-info">
            <h1>{{ title }}</h1>
            <p class="timestamp">Generated on: {{ timestamp }}</p>
        </div>
    </div>
    
    <div class="summary-box">
        <h2>Analysis Summary</h2>
        <p>This report compares {{ toxin_count }} toxin-bound and {{ control_count }} control systems, analyzing metrics across multiple categories.</p>
        <p><strong>Significant differences found:</strong> {{ significant_count }} metrics show statistically significant differences (p < 0.05).</p>
    </div>
    
    <!-- Tabs Navigation -->
    <div class="tabs">
        <div class="tab active" data-tab="tab-overview">Overview</div>
        <div class="tab" data-tab="tab-structure">Structure</div>
        <div class="tab" data-tab="tab-ions">Ion Analysis</div>
        <div class="tab" data-tab="tab-water">Water Analysis</div>
        <div class="tab" data-tab="tab-gates">Gates & Dynamics</div>
        <div class="tab" data-tab="tab-ai">AI Insights</div>
    </div>
    
    <!-- Tab Contents -->
    <div id="tab-overview" class="tab-content active">
        <h2>System Overview</h2>
        
        <!-- Systems Table -->
        <h3>Systems Included in Analysis</h3>
        <table>
            <thead>
                <tr>
                    <th>System Name</th>
                    <th>Type</th>
                    <th>Frames</th>
                    <th>Time (ns)</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
                {% for system in systems %}
                <tr>
                    <td>{{ system.system_name }}</td>
                    <td><span class="badge badge-{{ system.system_type }}">{{ system.system_type }}</span></td>
                    <td>{{ system.frame_count }}</td>
                    <td>{{ system.time_ns }}</td>
                    <td>{{ system.analysis_status }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        
        <!-- Summary Plots -->
        <h3>Key Comparison Metrics</h3>
        <div class="plot-container">
            {% if plots.summary_table %}
            <img src="data:image/png;base64,{{ plots.summary_table }}" alt="Summary of metric comparisons">
            {% else %}
            <p>No summary table available.</p>
            {% endif %}
        </div>
        
        <!-- Correlation Matrix -->
        <h3>Metric Correlations</h3>
        <div class="plot-container">
            {% if plots.correlation_matrix %}
            <img src="data:image/png;base64,{{ plots.correlation_matrix }}" alt="Correlation matrix">
            {% else %}
            <p>No correlation matrix available.</p>
            {% endif %}
        </div>
    </div>
    
    <div id="tab-structure" class="tab-content">
        <h2>Structural Analysis</h2>
        
        <!-- Structure Metrics Comparison -->
        <h3>Key Structure Metrics</h3>
        <div class="metric-comparison">
            {% for metric in structure_metrics %}
            <div class="metric-card">
                <div class="metric-title">{{ metric.metric_name }}</div>
                <p><strong>Toxin:</strong> {{ metric.toxin_mean|round(4) }} ± {{ metric.toxin_std|round(4) }} {{ metric.units }}</p>
                <p><strong>Control:</strong> {{ metric.control_mean|round(4) }} ± {{ metric.control_std|round(4) }} {{ metric.units }}</p>
                <p><strong>Change:</strong> {{ metric.percent_change|round(2) }}% ({{ "+" if metric.difference > 0 else "" }}{{ metric.difference|round(4) }} {{ metric.units }})</p>
                <p><strong>p-value:</strong> <span {% if metric.is_significant %}class="significant"{% endif %}>{{ metric.p_value|round(4) }}</span></p>
                {% if metric.plot_path %}
                <div class="plot-container">
                    <img src="data:image/png;base64,{{ plots[metric.metric_name] }}" alt="{{ metric.metric_name }} comparison">
                </div>
                {% endif %}
            </div>
            {% endfor %}
        </div>
    </div>
    
    <div id="tab-ions" class="tab-content">
        <h2>Ion Permeation Analysis</h2>
        
        <!-- Ion Metrics Comparison -->
        <h3>Key Ion Metrics</h3>
        <div class="metric-comparison">
            {% for metric in ion_metrics %}
            <div class="metric-card">
                <div class="metric-title">{{ metric.metric_name }}</div>
                <p><strong>Toxin:</strong> {{ metric.toxin_mean|round(4) }} ± {{ metric.toxin_std|round(4) }} {{ metric.units }}</p>
                <p><strong>Control:</strong> {{ metric.control_mean|round(4) }} ± {{ metric.control_std|round(4) }} {{ metric.units }}</p>
                <p><strong>Change:</strong> {{ metric.percent_change|round(2) }}% ({{ "+" if metric.difference > 0 else "" }}{{ metric.difference|round(4) }} {{ metric.units }})</p>
                <p><strong>p-value:</strong> <span {% if metric.is_significant %}class="significant"{% endif %}>{{ metric.p_value|round(4) }}</span></p>
                {% if metric.plot_path %}
                <div class="plot-container">
                    <img src="data:image/png;base64,{{ plots[metric.metric_name] }}" alt="{{ metric.metric_name }} comparison">
                </div>
                {% endif %}
            </div>
            {% endfor %}
        </div>
    </div>
    
    <div id="tab-water" class="tab-content">
        <h2>Water Distribution Analysis</h2>
        
        <!-- Water Metrics Comparison -->
        <h3>Key Water Metrics</h3>
        <div class="metric-comparison">
            {% for metric in water_metrics %}
            <div class="metric-card">
                <div class="metric-title">{{ metric.metric_name }}</div>
                <p><strong>Toxin:</strong> {{ metric.toxin_mean|round(4) }} ± {{ metric.toxin_std|round(4) }} {{ metric.units }}</p>
                <p><strong>Control:</strong> {{ metric.control_mean|round(4) }} ± {{ metric.control_std|round(4) }} {{ metric.units }}</p>
                <p><strong>Change:</strong> {{ metric.percent_change|round(2) }}% ({{ "+" if metric.difference > 0 else "" }}{{ metric.difference|round(4) }} {{ metric.units }})</p>
                <p><strong>p-value:</strong> <span {% if metric.is_significant %}class="significant"{% endif %}>{{ metric.p_value|round(4) }}</span></p>
                {% if metric.plot_path %}
                <div class="plot-container">
                    <img src="data:image/png;base64,{{ plots[metric.metric_name] }}" alt="{{ metric.metric_name }} comparison">
                </div>
                {% endif %}
            </div>
            {% endfor %}
        </div>
    </div>
    
    <div id="tab-gates" class="tab-content">
        <h2>Gates & Dynamics Analysis</h2>
        
        <!-- DW Gate & Dynamics Metrics Comparison -->
        <h3>Key Gate Metrics</h3>
        <div class="metric-comparison">
            {% for metric in gate_metrics %}
            <div class="metric-card">
                <div class="metric-title">{{ metric.metric_name }}</div>
                <p><strong>Toxin:</strong> {{ metric.toxin_mean|round(4) }} ± {{ metric.toxin_std|round(4) }} {{ metric.units }}</p>
                <p><strong>Control:</strong> {{ metric.control_mean|round(4) }} ± {{ metric.control_std|round(4) }} {{ metric.units }}</p>
                <p><strong>Change:</strong> {{ metric.percent_change|round(2) }}% ({{ "+" if metric.difference > 0 else "" }}{{ metric.difference|round(4) }} {{ metric.units }})</p>
                <p><strong>p-value:</strong> <span {% if metric.is_significant %}class="significant"{% endif %}>{{ metric.p_value|round(4) }}</span></p>
                {% if metric.plot_path %}
                <div class="plot-container">
                    <img src="data:image/png;base64,{{ plots[metric.metric_name] }}" alt="{{ metric.metric_name }} comparison">
                </div>
                {% endif %}
            </div>
            {% endfor %}
        </div>
    </div>
    
    <div id="tab-ai" class="tab-content">
        <h2>AI-Generated Insights</h2>
        
        {% for insight in ai_insights %}
        <div class="insight-box">
            <div class="insight-title">{{ insight.title }}</div>
            <div>{{ insight.content|safe }}</div>
            <p><em>Confidence: {{ insight.confidence|round(2) }}</em></p>
            <p>
                {% for metric in insight.metrics_involved %}
                <span class="badge category-badge">{{ metric }}</span>
                {% endfor %}
            </p>
        </div>
        {% endfor %}
    </div>
    
    <footer>
        <p>Generated by Pore Analysis Cross-Analysis Suite on {{ timestamp }}</p>
    </footer>
</body>
</html>
""")
                logger.info(f"Created default template at {main_template_path}")
    
    def connect(self) -> bool:
        """
        Connect to the meta-database.
        
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
    
    def _get_systems_data(self) -> List[Dict[str, Any]]:
        """
        Get systems data from the database.
        
        Returns:
            List of dictionaries with system information
        """
        if not self.meta_conn:
            logger.error("Meta-database not connected")
            return []
        
        cursor = self.meta_conn.cursor()
        cursor.execute("""
            SELECT system_id, system_name, system_type, run_name, 
                   db_path, analysis_status, frame_count, time_ns, 
                   is_included, metadata
            FROM systems
            WHERE is_included = 1
            ORDER BY system_type, run_name
        """)
        
        systems = []
        for row in cursor.fetchall():
            systems.append({
                'system_id': row['system_id'],
                'system_name': row['system_name'],
                'system_type': row['system_type'],
                'run_name': row['run_name'],
                'db_path': row['db_path'],
                'analysis_status': row['analysis_status'],
                'frame_count': row['frame_count'],
                'time_ns': row['time_ns'],
                'metadata': json.loads(row['metadata']) if row['metadata'] else {}
            })
        
        return systems
    
    def _get_comparison_results(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get comparison results from the database, organized by category.
        
        Returns:
            Dictionary with results grouped by category
        """
        if not self.meta_conn:
            logger.error("Meta-database not connected")
            return {}
        
        cursor = self.meta_conn.cursor()
        cursor.execute("""
            SELECT analysis_id, analysis_name, analysis_type, 
                   p_value, effect_size, details
            FROM comparative_analysis
            ORDER BY p_value ASC
        """)
        
        # Initialize categories
        categories = {
            'structure': [],
            'ion': [],
            'water': [],
            'dw_gate': [],
            'carbonyl': [],
            'tyrosine': [],
            'pocket': [],
            'toxin': [],
            'other': []
        }
        
        # Process results
        for row in cursor.fetchall():
            try:
                details = json.loads(row['details']) if row['details'] else {}
                if 'status' in details and details['status'] == 'success':
                    metric_name = details.get('metric_name', '')
                    
                    # Determine category based on metric name or metric_category in aggregated_metrics
                    if metric_name:
                        cursor.execute(
                            """SELECT metric_category FROM aggregated_metrics 
                               WHERE metric_name = ? LIMIT 1""", (metric_name,)
                        )
                        category_row = cursor.fetchone()
                        if category_row:
                            category = category_row['metric_category']
                        else:
                            # Guess category from metric name
                            if 'G_G_' in metric_name or 'COM_' in metric_name:
                                category = 'structure'
                            elif 'Ion_' in metric_name:
                                category = 'ion'
                            elif 'InnerVestibule_' in metric_name:
                                category = 'water'
                            elif 'DW_' in metric_name:
                                category = 'dw_gate'
                            elif 'Gyration_' in metric_name:
                                category = 'carbonyl'
                            elif 'Tyr_' in metric_name:
                                category = 'tyrosine'
                            elif 'Pocket' in metric_name:
                                category = 'pocket'
                            elif 'Orient_' in metric_name:
                                category = 'toxin'
                            else:
                                category = 'other'
                            
                        # Add to appropriate category
                        if category in categories:
                            categories[category].append(details)
                        else:
                            categories['other'].append(details)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse details JSON for analysis_id={row['analysis_id']}")
        
        return categories
    
    def _get_ai_insights(self) -> List[Dict[str, Any]]:
        """
        Get AI insights from the database.
        
        Returns:
            List of dictionaries with AI insights
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
            WHERE is_validated = 1 OR validator_notes IS NULL
            ORDER BY confidence DESC
        """)
        
        insights = []
        for row in cursor.fetchall():
            # Convert markdown description to HTML
            description_html = markdown.markdown(row['description'])
            
            # Create title based on insight type
            if row['insight_type'] == 'comprehensive':
                title = "Comprehensive Analysis of Toxin Effects"
            elif row['insight_type'] == 'metric_specific':
                metrics = json.loads(row['metrics_involved']) if row['metrics_involved'] else []
                title = f"Analysis of {metrics[0] if metrics else 'Specific Metric'}"
            elif row['insight_type'] == 'correlation':
                metrics = json.loads(row['metrics_involved']) if row['metrics_involved'] else []
                if len(metrics) >= 2:
                    title = f"Correlation Analysis: {metrics[0]} vs {metrics[1]}"
                else:
                    title = "Correlation Analysis"
            else:
                title = f"{row['insight_type'].title()} Analysis"
            
            insights.append({
                'insight_id': row['insight_id'],
                'title': title,
                'type': row['insight_type'],
                'content': description_html,
                'confidence': float(row['confidence']),
                'metrics_involved': json.loads(row['metrics_involved']) if row['metrics_involved'] else [],
                'systems_involved': json.loads(row['systems_involved']) if row['systems_involved'] else [],
                'is_validated': bool(row['is_validated']),
                'validator_notes': row['validator_notes']
            })
        
        return insights
    
    def _get_plots_data(self) -> Dict[str, str]:
        """
        Get plots as base64 encoded strings.
        
        Returns:
            Dictionary mapping plot names to base64 encoded strings
        """
        if not self.meta_conn:
            logger.error("Meta-database not connected")
            return {}
        
        cursor = self.meta_conn.cursor()
        cursor.execute("""
            SELECT plot_id, plot_name, plot_type, category, 
                   subcategory, relative_path
            FROM comparative_plots
        """)
        
        plots = {}
        for row in cursor.fetchall():
            try:
                # Get absolute path
                plot_path = os.path.join(
                    os.path.dirname(self.meta_db_path), 
                    row['relative_path']
                )
                
                # Check if file exists
                if os.path.exists(plot_path):
                    # Read file and encode as base64
                    with open(plot_path, 'rb') as f:
                        encoded = base64.b64encode(f.read()).decode('utf-8')
                    
                    # Use subcategory or metric name as key
                    if row['subcategory'] == 'comparison_table':
                        plots['summary_table'] = encoded
                    elif row['subcategory'] == 'correlation_matrix':
                        plots['correlation_matrix'] = encoded
                    elif 'comparison' in row['plot_type']:
                        # Extract metric name from the plot name (assumes "Comparison of X" format)
                        plot_name = row['plot_name']
                        if plot_name.startswith('Comparison of '):
                            metric_name = plot_name[len('Comparison of '):]
                            plots[metric_name] = encoded
                    else:
                        plots[row['subcategory']] = encoded
            except Exception as e:
                logger.warning(f"Failed to process plot {row['plot_name']}: {e}")
        
        return plots
    
    def generate_report(self, title: str) -> str:
        """
        Generate HTML report.
        
        Args:
            title: Title for the report
            
        Returns:
            str: Path to the generated HTML file
        """
        if not self.meta_conn:
            logger.error("Meta-database not connected")
            return ""
        
        # Prepare data for the template
        systems = self._get_systems_data()
        toxin_count = sum(1 for s in systems if s['system_type'] == 'toxin')
        control_count = sum(1 for s in systems if s['system_type'] == 'control')
        
        comparison_results = self._get_comparison_results()
        ai_insights = self._get_ai_insights()
        plots = self._get_plots_data()
        
        # Count significant differences
        all_results = []
        for category, results in comparison_results.items():
            all_results.extend(results)
        
        significant_count = sum(1 for r in all_results if r.get('is_significant', False))
        
        # Set up Jinja environment
        env = Environment(
            loader=FileSystemLoader(self.template_dir),
            autoescape=select_autoescape(['html'])
        )
        
        # Load template
        template = env.get_template('cross_report_template.html')
        
        # Render template
        html_content = template.render(
            title=title,
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            systems=systems,
            toxin_count=toxin_count,
            control_count=control_count,
            significant_count=significant_count,
            structure_metrics=comparison_results.get('structure', []),
            ion_metrics=comparison_results.get('ion', []),
            water_metrics=comparison_results.get('water', []) + comparison_results.get('pocket', []),
            gate_metrics=(comparison_results.get('dw_gate', []) + 
                         comparison_results.get('carbonyl', []) + 
                         comparison_results.get('tyrosine', [])),
            toxin_metrics=comparison_results.get('toxin', []),
            ai_insights=ai_insights,
            plots=plots
        )
        
        # Save HTML file
        output_path = os.path.join(self.output_dir, 'cross_analysis_report.html')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Generated HTML report at {output_path}")
        return output_path
    
    def close(self):
        """Close the database connection."""
        if self.meta_conn:
            self.meta_conn.close()
            logger.info("Closed meta-database connection")