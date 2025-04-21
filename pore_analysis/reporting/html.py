# reporting.py
"""
Functions for generating analysis reports (HTML for single runs, PPT for summaries).
"""

import os
import logging
import base64
import traceback
from datetime import datetime
import numpy as np
import pandas as pd
import json # <<< Add json import
import jinja2 # For HTML templating
from collections import defaultdict
import matplotlib.pyplot as plt # Keep for plt.close if needed, though plots are loaded now
import seaborn as sns # Keep for plot style consistency settings if applied here
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ..domain.models import DomainCalculationResult, ResidueDomainAffiliation
from ..config import config
from ..utils.plotly_utils import format_fig


# Import from other modules
try:
    from pore_analysis.core.utils import fig_to_base64 # Corrected import path
    # Plotting functions are no longer called directly here for HTML report
    # from core_analysis import plot_pore_diameter, plot_com_positions, plot_filtering_comparison, plot_kde_analysis
except ImportError as e:
    print(f"Error importing dependency modules in reporting/html.py: {e}") # Updated filename in error message
    raise

# Get a logger for this module
logger = logging.getLogger(__name__)

# Define the new base HTML template structure with tabs
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ run_summary.RunName }} - MD Analysis Report</title>
    <style>
        body { font-family: sans-serif; line-height: 1.6; margin: 0; background-color: #f4f4f4; color: #333; }
        .container { max-width: 1200px; margin: 20px auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 0 15px rgba(0,0,0,0.1); }
        h1, h2, h3 { color: #0056b3; padding-bottom: 5px; }
        h1 { text-align: center; border-bottom: 2px solid #0056b3; margin-bottom: 20px;}
        h2 { border-bottom: 2px solid #0056b3; margin-top: 30px; }
        h3 { border-bottom: 1px solid #ddd; margin-top: 25px; }
        .section { margin-bottom: 30px; padding-bottom: 20px; border-bottom: 1px solid #eee; }
        .section:last-child { border-bottom: none; }
        .plot-container { margin: 20px 0; text-align: center; }
        .plot-container img { max-width: 95%; height: auto; border: 1px solid #ddd; padding: 5px; background: #fff; margin-bottom: 5px; }
        .stats-table { width: 100%; border-collapse: collapse; margin: 20px 0; font-size: 0.9em;}
        .stats-table th, .stats-table td { padding: 8px 10px; border: 1px solid #ddd; text-align: left; }
        .stats-table th { background-color: #e9ecef; font-weight: bold; }
        .stats-table tr:nth-child(even) { background-color: #f8f9fa; }
        .two-column { display: flex; flex-wrap: wrap; gap: 20px; }
        .column { flex: 1; min-width: 48%; }
        .info-box { background-color: #e7f1ff; border-left: 5px solid #0056b3; padding: 15px; margin: 15px 0; border-radius: 4px; font-size: 0.9em;}
        .warning-box { background-color: #fff3cd; border-left: 5px solid #ffc107; padding: 15px; margin: 15px 0; border-radius: 4px; font-size: 0.9em;}
        .warning-box h4 { color: #856404; margin-top: 0; }
        .error-box { background-color: #f8d7da; border-left: 5px solid #dc3545; padding: 15px; margin: 15px 0; border-radius: 4px; font-size: 0.9em;}
        .error-box h4 { color: #721c24; margin-top: 0; }
        .control-system-banner { background-color: #17a2b8; color: white; text-align: center; padding: 10px; margin: 15px 0; border-radius: 4px; font-weight: bold; font-size: 1.1em; }
        pre { background-color: #f8f9fa; padding: 10px; border: 1px solid #ddd; border-radius: 4px; white-space: pre-wrap; word-wrap: break-word; font-size: 0.9em; max-height: 300px; overflow-y: auto;}
        .footer { text-align: center; margin-top: 30px; font-size: 0.8em; color: #666; }
        td, th { word-wrap: break-word; } /* Prevent table cell overflow */

        /* Tab Styles */
        .tab-nav { list-style: none; padding: 0; margin: 0 0 20px 0; border-bottom: 2px solid #0056b3; display: flex; }
        .tab-nav li { margin-right: 5px; }
        .tab-nav a {
            display: inline-block;
            padding: 10px 15px;
            text-decoration: none;
            color: #0056b3;
            background-color: #f0f0f0;
            border: 1px solid #ccc;
            border-bottom: none;
            border-radius: 5px 5px 0 0;
            position: relative;
            bottom: -2px; /* Align with container border */
            transition: background-color 0.2s ease;
        }
        .tab-nav a.active {
            background-color: white;
            border-color: #0056b3 #0056b3 white;
            border-bottom: 2px solid white; /* Cover the container border */
            color: #333;
            font-weight: bold;
        }
        .tab-nav a:hover:not(.active) {
            background-color: #e0e0e0;
        }
        .tab-content { display: none; } /* Hide content by default */
        .tab-content.active { display: block; } /* Show active content */

    </style>
</head>
<body>
    <div class="container">
        <h1>MD Analysis Report</h1>
        {% if run_summary.IsControlSystem %}
        <div class="control-system-banner">CONTROL SYSTEM (NO TOXIN)</div>
        {% endif %}
        <p class="footer">Generated: {{ generation_timestamp }} | Analysis Version: {{ run_summary.AnalysisScriptVersion }}</p>

        <!-- Tab Navigation -->
        <ul class="tab-nav">
            <li><a href="#tab-overview" class="tab-link active">Overview & Distances</a></li>
            <li><a href="#tab-toxin" class="tab-link">Toxin Interface</a></li>
            <li><a href="#tab-pore-ions" class="tab-link">Pore Ions</a></li>
            <li><a href="#tab-inner-vestibule" class="tab-link">Inner Vestibule</a></li>
            <li><a href="#tab-carbonyl" class="tab-link">Carbonyl Dynamics</a></li>
            <li><a href="#tab-tyrosine" class="tab-link">SF Tyrosine</a></li>
            <li><a href="#tab-dw-gate" class="tab-link">DW Gate</a></li>
        </ul>

        <!-- Tab Content Containers -->
        <div id="tab-overview" class="tab-content active">
            {% include '_tab_overview.html' %}
        </div>

        <div id="tab-toxin" class="tab-content">
            {% include '_tab_toxin.html' %}
        </div>

        <div id="tab-pore-ions" class="tab-content">
            {% include '_tab_pore_ions.html' %}
        </div>

        <div id="tab-inner-vestibule" class="tab-content">
            {% include '_tab_inner_vestibule.html' %}
        </div>

        <div id="tab-carbonyl" class="tab-content">
            {% include '_tab_carbonyl.html' %}
        </div>

        <div id="tab-tyrosine" class="tab-content">
            {% include '_tab_tyrosine.html' %}
        </div>

        <div id="tab-dw-gate" class="tab-content">
            {% include '_tab_dw_gates.html' %}
        </div>

        <div class="footer">End of Report</div>
    </div>

    <!-- JavaScript for Tab Switching -->
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const tabLinks = document.querySelectorAll('.tab-link');
            const tabContents = document.querySelectorAll('.tab-content');

            tabLinks.forEach(link => {
                link.addEventListener('click', function(event) {
                    event.preventDefault(); // Prevent default anchor behavior

                    // Deactivate all tabs and content
                    tabLinks.forEach(l => l.classList.remove('active'));
                    tabContents.forEach(c => c.classList.remove('active'));

                    // Activate the clicked tab and corresponding content
                    this.classList.add('active');
                    const targetId = this.getAttribute('href').substring(1);
                    document.getElementById(targetId).classList.add('active');
                });
            });
        });
    </script>
</body>
</html>
"""


def _load_image_base64(image_path):
    """Loads an image file and returns its base64 encoded string."""
    if image_path and os.path.exists(image_path):
        try:
            with open(image_path, 'rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')
        except Exception as e:
            logger.warning(f"Error loading image {image_path}: {e}")
    return None

def generate_html_report(run_dir, run_summary):
    """
    Generates a comprehensive HTML report for a single run.

    Relies on existing plot files and the summary dictionary.

    Args:
        run_dir (str): Path to the specific run directory.
        run_summary (dict): Dictionary containing summary statistics for the run,
                            loaded from 'analysis_summary.json'.

    Returns:
        str | None: Path to the generated HTML report, or None on failure.
    """
    logger.info(f"Generating HTML report for {run_summary.get('RunName', 'Unknown Run')}...")

    # --- Add config values to summary for template access ---
    # (Do this carefully, only add what's needed by the template)
    try:
        from pore_analysis.core.config import GYRATION_FLIP_THRESHOLD, GYRATION_FLIP_TOLERANCE_FRAMES, FRAMES_PER_NS
        run_summary['GYRATION_FLIP_THRESHOLD'] = GYRATION_FLIP_THRESHOLD
        run_summary['GYRATION_FLIP_TOLERANCE_FRAMES'] = GYRATION_FLIP_TOLERANCE_FRAMES
        run_summary['FRAMES_PER_NS'] = FRAMES_PER_NS
    except ImportError:
        logger.warning("Could not import config values for HTML template.")

    # --- Load Base64 Encoded Images ---
    img_data = {}
    # Expected plot filenames relative to run_dir
    PLOT_FILES = {
        "Raw_Distances": "core_analysis/Raw_Distances.csv",
        # Use G_G_ prefix for keys and filenames
        "G_G_Distance_Plot_raw": "core_analysis/G_G_Distance_Plot_raw.png",
        "G_G_Distance_Plot": "core_analysis/G_G_Distance_Plot.png",
        "G_G_Distance_AC_Comparison": "core_analysis/G_G_Distance_AC_Comparison.png",
        "G_G_Distance_BD_Comparison": "core_analysis/G_G_Distance_BD_Comparison.png",
        "COM_Distance_Plot_raw": "core_analysis/COM_Distance_Plot_raw.png",
        "COM_Distance_Plot": "core_analysis/COM_Distance_Plot.png",
        "COM_Stability_Plot_raw": "core_analysis/COM_Stability_Plot_raw.png",
        "COM_Stability_Plot": "core_analysis/COM_Stability_Plot.png",
        "COM_Stability_Comparison": "core_analysis/COM_Stability_Comparison.png",
        "COM_Stability_KDE_Analysis": "core_analysis/COM_Stability_KDE_Analysis.png",
        # Orientation/Contact plots (now in orientation_contacts/)
        "Toxin_Orientation_Angle": "orientation_contacts/Toxin_Orientation_Angle.png",
        "Toxin_Rotation_Components": "orientation_contacts/Toxin_Rotation_Components.png",
        "Toxin_Channel_Contacts": "orientation_contacts/Toxin_Channel_Contacts.png",
        "Toxin_Channel_Residue_Contact_Map_Full": "orientation_contacts/Toxin_Channel_Residue_Contact_Map_Full.png",
        "Toxin_Channel_Residue_Contact_Map_Focused": "orientation_contacts/Toxin_Channel_Residue_Contact_Map_Focused.png",
        # Ion plots (now in ion_analysis/)
        "binding_sites_g1_centric_visualization": "ion_analysis/binding_sites_g1_centric_visualization.png",
        "K_Ion_Combined_Plot": "ion_analysis/K_Ion_Combined_Plot.png",
        "K_Ion_Occupancy_Heatmap": "ion_analysis/K_Ion_Occupancy_Heatmap.png",
        "K_Ion_Average_Occupancy": "ion_analysis/K_Ion_Average_Occupancy.png",
        # Change key and filename for the transition/overlay plot
        "K_Ion_Overlayed_Transitions": "ion_analysis/K_Ion_Overlayed_Transitions.png",
        # Water plots (now in inner_vestibule_analysis/)
        "Inner_Vestibule_Count_Plot": "inner_vestibule_analysis/Inner_Vestibule_Count_Plot.png",
        "Inner_Vestibule_Residence_Hist": "inner_vestibule_analysis/Inner_Vestibule_Residence_Hist.png",
        # Gyration analysis plots (now in gyration_analysis/)
        "G1_gyration_radii": "gyration_analysis/G1_gyration_radii.png",
        "Y_gyration_radii": "gyration_analysis/Y_gyration_radii.png",
        "Flip_Duration_Distribution": "gyration_analysis/Flip_Duration_Distribution.png",
        # Tyrosine analysis plots (NEW)
        "SF_Tyrosine_Dihedrals": "tyrosine_analysis/SF_Tyrosine_Dihedrals.png",
        "SF_Tyrosine_Rotamer_Scatter": "tyrosine_analysis/SF_Tyrosine_Rotamer_Scatter.png",
        "SF_Tyrosine_Rotamer_Population": "tyrosine_analysis/SF_Tyrosine_Rotamer_Population.png",
        # DW Gate analysis plots (NEW)
        "dw_distance_timeseries": "dw_gate_analysis/dw_distance_timeseries.png",
        "dw_gate_state_heatmap": "dw_gate_analysis/dw_gate_state_heatmap.png",
        "dw_gate_closed_fraction_bar": "dw_gate_analysis/dw_gate_closed_fraction_bar.png",
        "dw_duration_analysis": "dw_gate_analysis/dw_gate_duration_analysis.png",
        "dw_distance_state_overlay": "dw_gate_analysis/dw_distance_state_overlay.png",
    }

    logger.debug("Loading images for HTML report...")
    for key, filename in PLOT_FILES.items():
        img_path = os.path.join(run_dir, filename)
        img_data[key] = _load_image_base64(img_path)
        if img_data[key] is None:
            logger.debug(f"Plot file not found or load failed: {img_path}")

    # --- Load Binding Site Data ---
    binding_site_data = None
    binding_site_path = os.path.join(run_dir, 'binding_site_positions_g1_centric.txt')
    if os.path.exists(binding_site_path):
        try:
            with open(binding_site_path, 'r') as f:
                binding_site_data = f.read()
            logger.debug("Loaded binding site position data.")
        except Exception as e:
            logger.warning(f"Error reading binding site data from {binding_site_path}: {e}")

    # --- Render HTML ---
    try:
        logger.debug("Rendering HTML template...")
        # Use FileSystemLoader to find partial templates
        # Assuming the script runs from the workspace root or similar
        # Adjust the path if necessary, e.g., based on __file__
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # workspace_root = os.path.dirname(os.path.dirname(script_dir)) # Go up two levels (reporting -> pore_analysis -> root)
        # ^^^ This assumes script is in reporting/. Workspace root is likely run_dir's parent or grandparent?
        # Let's load templates relative to this script's location instead.
        templates_dir = os.path.join(script_dir, 'templates')
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(templates_dir),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )
        # Load the base template directly from the string variable
        template = env.from_string(HTML_TEMPLATE)

        # Prepare context for rendering
        render_context = {
            'run_summary': run_summary,
            'plots': img_data,
            'binding_site_data': binding_site_data,
            'generation_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'DISTANCE_THRESHOLD': run_summary.get('DWhbond_threshold', 3.5)
        }

        rendered_html = template.render(render_context)
        logger.debug("HTML rendering complete.")

        # Save the report
        report_filename = f"{run_summary.get('RunName', 'report')}_analysis_report.html"
        report_path = os.path.join(run_dir, report_filename)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(rendered_html)

        logger.info(f"Generated HTML report at: {report_path}")
        return report_path

    except jinja2.exceptions.TemplateError as e:
        logger.error(f"Error rendering Jinja2 template: {e}", exc_info=True)
        error_report_path = os.path.join(run_dir, f"{run_summary.get('RunName', 'report')}_report_TEMPLATE_ERROR.html")
        with open(error_report_path, 'w') as f: f.write(f"<html><body><h1>Template Error</h1><pre>{traceback.format_exc()}</pre></body></html>")
        return None
    except Exception as e:
        logger.error(f"Unexpected error generating HTML report: {e}", exc_info=True)
        error_report_path = os.path.join(run_dir, f"{run_summary.get('RunName', 'report')}_report_ERROR.html")
        with open(error_report_path, 'w') as f: f.write(f"<html><body><h1>Report Generation Error</h1><pre>{traceback.format_exc()}</pre></body></html>")
        return None

# --- PowerPoint Report Generation ---
# Note: PowerPoint functionality has been moved to presentation.py
