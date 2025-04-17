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
from pptx import Presentation
from pptx.util import Inches, Pt
from collections import defaultdict
import matplotlib.pyplot as plt # Keep for plt.close if needed, though plots are loaded now
import seaborn as sns # Keep for plot style consistency settings if applied here


# Import from other modules
try:
    from md_analysis.core.utils import fig_to_base64 # Corrected import path
    # Plotting functions are no longer called directly here for HTML report
    # from core_analysis import plot_pore_diameter, plot_com_positions, plot_filtering_comparison, plot_kde_analysis
except ImportError as e:
    print(f"Error importing dependency modules in reporting/html.py: {e}") # Updated filename in error message
    raise

# Get a logger for this module
logger = logging.getLogger(__name__)

# --- HTML Report Generation ---

# Define the HTML template structure here
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ run_summary.RunName }} - MD Analysis Report</title>
    <style>
        body { font-family: sans-serif; line-height: 1.6; margin: 20px; background-color: #f4f4f4; color: #333; }
        .container { max-width: 1200px; margin: auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 0 15px rgba(0,0,0,0.1); }
        h1, h2, h3 { color: #0056b3; border-bottom: 2px solid #0056b3; padding-bottom: 5px; }
        h1 { text-align: center; }
        .section { margin-bottom: 30px; padding-bottom: 20px; border-bottom: 1px solid #eee; }
        .section:last-child { border-bottom: none; }
        .plot-container { margin: 20px 0; text-align: center; }
        .plot-container img { max-width: 95%; height: auto; border: 1px solid #ddd; padding: 5px; background: #fff; margin-bottom: 5px; }
        .stats-table { width: 100%; border-collapse: collapse; margin: 20px 0; font-size: 0.9em;}
        .stats-table th, .stats-table td { padding: 8px 10px; border: 1px solid #ddd; text-align: left; }
        .stats-table th { background-color: #e9ecef; font-weight: bold; }
        .stats-table tr:nth-child(even) { background-color: #f8f9fa; }
        .two-column { display: flex; flex-wrap: wrap; gap: 20px; }
        .column { flex: 1; min-width: 48%; } /* Adjusted min-width */
        .info-box { background-color: #e7f1ff; border-left: 5px solid #0056b3; padding: 15px; margin: 15px 0; border-radius: 4px; font-size: 0.9em;}
        .warning-box { background-color: #fff3cd; border-left: 5px solid #ffc107; padding: 15px; margin: 15px 0; border-radius: 4px; font-size: 0.9em;}
        .warning-box h4 { color: #856404; margin-top: 0; }
        .error-box { background-color: #f8d7da; border-left: 5px solid #dc3545; padding: 15px; margin: 15px 0; border-radius: 4px; font-size: 0.9em;}
        .error-box h4 { color: #721c24; margin-top: 0; }
        .control-system-banner { background-color: #17a2b8; color: white; text-align: center; padding: 10px; margin: 15px 0; border-radius: 4px; font-weight: bold; font-size: 1.1em; }
        pre { background-color: #f8f9fa; padding: 10px; border: 1px solid #ddd; border-radius: 4px; white-space: pre-wrap; word-wrap: break-word; font-size: 0.9em; max-height: 300px; overflow-y: auto;}
        .footer { text-align: center; margin-top: 30px; font-size: 0.8em; color: #666; }
        td, th { word-wrap: break-word; } /* Prevent table cell overflow */
    </style>
</head>
<body>
    <div class="container">
        <h1>MD Analysis Report</h1>
        {% if run_summary.IsControlSystem %}
        <div class="control-system-banner">CONTROL SYSTEM (NO TOXIN)</div>
        {% endif %}
        <p class="footer">Generated: {{ generation_timestamp }} | Analysis Version: {{ run_summary.AnalysisScriptVersion }}</p>

        <div class="section">
            <h2>Run Information</h2>
            <table class="stats-table">
                <tr><th>System</th><td>{{ run_summary.SystemName }}</td></tr>
                <tr><th>Run</th><td>{{ run_summary.RunName }}</td></tr>
                <tr><th>Path</th><td>{{ run_summary.RunPath }}</td></tr>
                <tr><th>System Type</th><td>{% if run_summary.IsControlSystem %}Control (No Toxin){% else %}Toxin-Channel Complex{% endif %}</td></tr>
                <tr><th>Analysis Status</th><td>{{ run_summary.AnalysisStatus }}</td></tr>
                <tr><th>Analysis Timestamp</th><td>{{ run_summary.AnalysisTimestamp }}</td></tr>
            </table>
        </div>

        <div class="section">
            <h2>G-G Distance Analysis</h2>
            <table class="stats-table">
              <thead><tr><th>Metric</th><th>Filtered A:C</th><th>Filtered B:D</th></tr></thead>
              <tbody>
                <tr><td>Mean (Å)</td><td>{{ "%.3f"|format(run_summary.GG_AC_Mean_Filt) if run_summary.GG_AC_Mean_Filt is not none else 'N/A' }}</td><td>{{ "%.3f"|format(run_summary.GG_BD_Mean_Filt) if run_summary.GG_BD_Mean_Filt is not none else 'N/A' }}</td></tr>
                <tr><td>Std Dev (Å)</td><td>{{ "%.3f"|format(run_summary.GG_AC_Std_Filt) if run_summary.GG_AC_Std_Filt is not none else 'N/A' }}</td><td>{{ "%.3f"|format(run_summary.GG_BD_Std_Filt) if run_summary.GG_BD_Std_Filt is not none else 'N/A' }}</td></tr>
                <tr><td>Min (Å)</td><td>{{ "%.3f"|format(run_summary.GG_AC_Min_Filt) if run_summary.GG_AC_Min_Filt is not none else 'N/A' }}</td><td>{{ "%.3f"|format(run_summary.GG_BD_Min_Filt) if run_summary.GG_BD_Min_Filt is not none else 'N/A' }}</td></tr>
              </tbody>
            </table>
            <div class="two-column">
                <div class="column plot-container">
                    <h3>Raw G-G Distance</h3>
                    {% if img_data.GG_Distance_Plot_raw %} <img src="data:image/png;base64,{{ img_data.GG_Distance_Plot_raw }}" alt="Raw G-G Distance Plot"> {% else %} <p><i>Plot not available.</i></p> {% endif %}
                </div>
                <div class="column plot-container">
                    <h3>Filtered G-G Distance</h3>
                    {% if img_data.GG_Distance_Plot %} <img src="data:image/png;base64,{{ img_data.GG_Distance_Plot }}" alt="Filtered G-G Distance Plot"> {% else %} <p><i>Plot not available.</i></p> {% endif %}
                </div>
            </div>
            {% if img_data.G_G_Distance_AC_Comparison %}
            <div class="plot-container">
                <h3>A:C Distance Filtering Detail</h3>
                <img src="data:image/png;base64,{{ img_data.G_G_Distance_AC_Comparison }}" alt="A:C Filtering Comparison">
            </div>
            {% endif %}
            {% if img_data.G_G_Distance_BD_Comparison %}
            <div class="plot-container">
                <h3>B:D Distance Filtering Detail</h3>
                <img src="data:image/png;base64,{{ img_data.G_G_Distance_BD_Comparison }}" alt="B:D Filtering Comparison">
            </div>
            {% endif %}
        </div>

        {% if not run_summary.IsControlSystem and run_summary.COM_Mean_Filt is not none %} {# Only show for non-control systems with COM data #}
        <div class="section">
            <h2>COM Distance Analysis (Toxin-Channel Stability)</h2>
            <div class="info-box">
                Filter Type: {{ run_summary.COM_Filter_Type | default('N/A') }} |
                Quality Issue Detected: {{ 'Yes' if run_summary.COM_Filter_QualityIssue else 'No' }}
            </div>
            <table class="stats-table">
              <thead><tr><th>Metric</th><th>Filtered COM</th></tr></thead>
              <tbody>
                <tr><td>Mean (Å)</td><td>{{ "%.3f"|format(run_summary.COM_Mean_Filt) }}</td></tr>
                <tr><td>Std Dev (Å)</td><td>{{ "%.3f"|format(run_summary.COM_Std_Filt) }}</td></tr>
                <tr><td>Min (Å)</td><td>{{ "%.3f"|format(run_summary.COM_Min_Filt) }}</td></tr>
                <tr><td>Max (Å)</td><td>{{ "%.3f"|format(run_summary.COM_Max_Filt) }}</td></tr>
              </tbody>
            </table>
            <div class="two-column">
                <div class="column plot-container">
                    <h3>Raw COM Distance</h3>
                    {% if img_data.COM_Stability_Plot_raw %} <img src="data:image/png;base64,{{ img_data.COM_Stability_Plot_raw }}" alt="Raw COM Stability Plot"> {% else %} <p><i>Plot not available.</i></p> {% endif %}
                </div>
                <div class="column plot-container">
                    <h3>Filtered COM Distance</h3>
                     {% if img_data.COM_Stability_Plot %} <img src="data:image/png;base64,{{ img_data.COM_Stability_Plot }}" alt="Filtered COM Stability Plot"> {% else %} <p><i>Plot not available.</i></p> {% endif %}
                </div>
            </div>
            {% if img_data.COM_Stability_Comparison %}
            <div class="plot-container">
                <h3>COM Distance Filtering Detail</h3>
                <img src="data:image/png;base64,{{ img_data.COM_Stability_Comparison }}" alt="COM Filtering Comparison">
            </div>
            {% endif %}
            {% if img_data.COM_Stability_KDE_Analysis %}
            <div class="plot-container">
                <h3>COM Level Analysis (KDE on Raw Data)</h3>
                <img src="data:image/png;base64,{{ img_data.COM_Stability_KDE_Analysis }}" alt="COM KDE Analysis">
            </div>
            {% endif %}
        </div>
        {% elif run_summary.IsControlSystem %}
        <div class="section">
            <h2>COM Distance Analysis</h2>
            <div class="info-box">
                This is a control system without toxin. COM distance analysis is not applicable.
            </div>
        </div>
        {% else %}
        <div class="section"><p><i>COM distance analysis not performed or data unavailable.</i></p></div>
        {% endif %}

        {% if not run_summary.IsControlSystem and run_summary.COM_Mean_Filt is not none %} {# Only show for non-control systems #}
        <div class="section">
            <h2>Toxin Orientation & Interface Analysis</h2>
             <table class="stats-table">
               <thead><tr><th>Metric</th><th>Value</th></tr></thead>
               <tbody>
                 <tr><td>Mean Orientation Angle (°)</td><td>{{ "%.2f"|format(run_summary.Orient_Angle_Mean) if run_summary.Orient_Angle_Mean is not none else 'N/A' }}</td></tr>
                 <tr><td>Std Dev Orientation Angle (°)</td><td>{{ "%.2f"|format(run_summary.Orient_Angle_Std) if run_summary.Orient_Angle_Std is not none else 'N/A' }}</td></tr>
                 <tr><td>Mean Atom Contacts</td><td>{{ "%.1f"|format(run_summary.Orient_Contacts_Mean) if run_summary.Orient_Contacts_Mean is not none else 'N/A' }}</td></tr>
                 <tr><td>Std Dev Atom Contacts</td><td>{{ "%.1f"|format(run_summary.Orient_Contacts_Std) if run_summary.Orient_Contacts_Std is not none else 'N/A' }}</td></tr>
               </tbody>
             </table>
            <div class="two-column">
                <div class="column plot-container">
                    <h3>Orientation Angle</h3>
                    {% if img_data.Toxin_Orientation_Angle %} <img src="data:image/png;base64,{{ img_data.Toxin_Orientation_Angle }}" alt="Toxin Orientation Angle"> {% else %} <p><i>Plot not available.</i></p> {% endif %}
                </div>
                <div class="column plot-container">
                    <h3>Rotation Components</h3>
                    {% if img_data.Toxin_Rotation_Components %} <img src="data:image/png;base64,{{ img_data.Toxin_Rotation_Components }}" alt="Toxin Rotation Components"> {% else %} <p><i>Plot not available.</i></p> {% endif %}
                </div>
            </div>
             <div class="plot-container">
                 <h3>Total Atom Contacts</h3>
                 {% if img_data.Toxin_Channel_Contacts %} <img src="data:image/png;base64,{{ img_data.Toxin_Channel_Contacts }}" alt="Toxin-Channel Contacts"> {% else %} <p><i>Plot not available.</i></p> {% endif %}
             </div>
             <div class="plot-container">
                 <h3>Focused Residue Contact Map</h3>
                 {% if img_data.Toxin_Channel_Residue_Contact_Map_Focused %} <img src="data:image/png;base64,{{ img_data.Toxin_Channel_Residue_Contact_Map_Focused }}" alt="Focused Contact Map"> {% else %} <p><i>Plot not available.</i></p> {% endif %}
             </div>
             <div class="plot-container">
                 <h3>Full Residue Contact Map</h3>
                 {% if img_data.Toxin_Channel_Residue_Contact_Map_Full %} <img src="data:image/png;base64,{{ img_data.Toxin_Channel_Residue_Contact_Map_Full }}" alt="Full Contact Map"> {% else %} <p><i>Plot not available.</i></p> {% endif %}
             </div>
        </div>
        {% elif run_summary.IsControlSystem %}
        <div class="section">
            <h2>Toxin Orientation & Interface Analysis</h2>
            <div class="info-box">
                This is a control system without toxin. Orientation and interface analysis is not applicable.
            </div>
        </div>
        {% endif %}

        <div class="section">
            <h2>K+ Ion Analysis (Selectivity Filter)</h2>
            {% if run_summary.Ion_Count > 0 %} {# Check if ions were tracked #}
                 <div class="info-box">
                    Tracked {{ run_summary.Ion_Count }} unique K+ ions near the filter.<br>
                    Coordinates relative to G1 Cα Z-plane = 0 Å.
                 </div>
                 {% if binding_site_data %}
                 <div class="info-box">
                     <h3>Binding Site Positions (G1 Cα = 0 Å)</h3>
                     <pre>{{ binding_site_data }}</pre>
                 </div>
                 {% endif %}

                 {% if img_data.binding_sites_g1_centric_visualization %}
                 <div class="plot-container">
                     <h3>Binding Site Visualization</h3>
                     <img src="data:image/png;base64,{{ img_data.binding_sites_g1_centric_visualization }}" alt="Binding Sites Visualization">
                 </div>
                 {% endif %}

                 {% if img_data.K_Ion_Combined_Plot %}
                 <div class="plot-container">
                    <h3>Ion Positions & Density</h3>
                    <img src="data:image/png;base64,{{ img_data.K_Ion_Combined_Plot }}" alt="K+ Ion Positions and Density">
                 </div>
                 {% endif %}

                 {% if img_data.K_Ion_Occupancy_Heatmap %}
                 <div class="plot-container">
                    <h3>Site Occupancy Heatmap</h3>
                    <img src="data:image/png;base64,{{ img_data.K_Ion_Occupancy_Heatmap }}" alt="K+ Ion Occupancy Heatmap">
                 </div>
                 {% endif %}

                 {% if img_data.K_Ion_Average_Occupancy %}
                 <div class="plot-container">
                    <h3>Average Site Occupancy</h3>
                    <img src="data:image/png;base64,{{ img_data.K_Ion_Average_Occupancy }}" alt="Average K+ Ion Occupancy">
                 </div>
                 {% endif %}

                 <h3>Site Occupancy Statistics</h3>
                 <table class="stats-table">
                     <thead><tr><th>Site</th><th>Mean Occ.</th><th>Max Occ.</th><th>% Time Occ. (>0)</th></tr></thead>
                     <tbody>
                     {% for site in ['S0', 'S1', 'S2', 'S3', 'S4', 'Cavity'] %}
                         <tr>
                             <td>{{ site }}</td>
                             <td>{{ "%.3f"|format(run_summary['Ion_AvgOcc_'~site]) if run_summary['Ion_AvgOcc_'~site] is not none else 'N/A' }}</td>
                             <td>{{ "%d"|format(run_summary['Ion_MaxOcc_'~site]) if run_summary['Ion_MaxOcc_'~site] is not none else 'N/A' }}</td>
                             <td>{{ "%.1f"|format(run_summary['Ion_PctTimeOcc_'~site]) if run_summary['Ion_PctTimeOcc_'~site] is not none else 'N/A' }}%</td>
                         </tr>
                     {% endfor %}
                     </tbody>
                 </table>
            {% else %}
                <p><i>K+ ion analysis not performed or no ions tracked near filter.</i></p>
            {% endif %}
        </div>

        <div class="section">
            <h2>Inner Vestibule Analysis</h2>
             {% if run_summary.InnerVestibule_MeanOcc is defined and run_summary.InnerVestibule_MeanOcc is not none %} {# Check if inner vestibule stats exist #}
                 <div class="analysis-section">
                   <h3>Inner Vestibule Water Dynamics</h3>
                   <table class="stats-table">
                     <thead><tr><th>Metric</th><th>Value</th></tr></thead>
                     <tbody>
                         <tr><td>Mean Occupancy</td><td>{{ "%.2f"|format(run_summary.InnerVestibule_MeanOcc) }}</td></tr>
                         <tr><td>Std Dev Occupancy</td><td>{{ "%.2f"|format(run_summary.InnerVestibule_StdOcc) if run_summary.InnerVestibule_StdOcc is defined else 'N/A' }}</td></tr>
                         <tr><td>Avg Residence Time (ns)</td><td>{{ "%.3f"|format(run_summary.InnerVestibule_AvgResidenceTime_ns) if run_summary.InnerVestibule_AvgResidenceTime_ns is defined else 'N/A' }}</td></tr>
                         <tr><td>Total Confirmed Exit Events</td><td>{{ "%d"|format(run_summary.InnerVestibule_TotalExitEvents) if run_summary.InnerVestibule_TotalExitEvents is defined else 'N/A' }}</td></tr>
                         <tr><td>Exchange Rate (/ns)</td><td>{{ "%.4f"|format(run_summary.InnerVestibule_ExchangeRatePerNs) if run_summary.InnerVestibule_ExchangeRatePerNs is defined else 'N/A' }}</td></tr>
                     </tbody>
                   </table>
                 </div>
                 <div class="two-column">
                    <div class="column plot-container">
                        <h3>Inner Vestibule Water Count</h3>
                         {% if img_data.Inner_Vestibule_Count_Plot %} <img src="data:image/png;base64,{{ img_data.Inner_Vestibule_Count_Plot }}" alt="Inner Vestibule Water Count Plot"> {% else %} <p><i>Plot not available.</i></p> {% endif %}
                    </div>
                    <div class="column plot-container">
                        <h3>Residence Time Distribution</h3>
                        {% if img_data.Inner_Vestibule_Residence_Hist %} <img src="data:image/png;base64,{{ img_data.Inner_Vestibule_Residence_Hist }}" alt="Inner Vestibule Water Residence Histogram"> {% else %} <p><i>Plot not available.</i></p> {% endif %}
                    </div>
                 </div>
             {% else %}
                 <p><i>Inner vestibule analysis not performed or data unavailable.</i></p>
             {% endif %}
        </div>

        {# <<< ADDED SECTION for Carbonyl Gyration >>> #}
        <div class="section">
            <h2>Carbonyl Gyration Analysis (Selectivity Filter G1)</h2>
             {% if run_summary.Gyration_G1_Mean is not none %} {# Check if gyration analysis was performed #}
                 <div class="info-box">
                    Gyration radius (ρ) measures the distance between the G1 carbonyl oxygen atoms and the pore center.
                    Changes in this radius can indicate carbonyl flipping events that may affect ion permeation.
                 </div>
                 <table class="stats-table">
                    <thead><tr><th>Metric</th><th>G1 Glycine</th></tr></thead>
                    <tbody>
                        <tr><td>Mean Gyration Radius (Å)</td>
                            <td>{{ "%.3f"|format(run_summary.Gyration_G1_Mean) }}</td>
                        </tr>
                        <tr><td>Std Dev (Å)</td>
                            <td>{{ "%.3f"|format(run_summary.Gyration_G1_Std) }}</td>
                        </tr>
                        <tr><td>Detected Flips</td>
                            <td>{{ run_summary.Gyration_Flips }}</td>
                        </tr>
                        <tr><td>Maximum Change (Å)</td>
                            <td>{{ "%.3f"|format(run_summary.Gyration_MaxChange) if run_summary.Gyration_MaxChange is not none else 'N/A' }}</td>
                        </tr>
                    </tbody>
                 </table>
                 <div class="plot-container">
                     <h3>G1 Gyration Radii</h3>
                     {% if img_data.G1_gyration_radii %} <img src="data:image/png;base64,{{ img_data.G1_gyration_radii }}" alt="G1 Gyration Radii"> {% else %} <p><i>Plot not available.</i></p> {% endif %}
                 </div>
                 <div class="info-box">
                    <p><strong>Interpretation:</strong> Carbonyl flipping is indicated by sudden changes in gyration radius.
                    {% if run_summary.Gyration_Flips > 5 %}
                    A high number of flips ({{ run_summary.Gyration_Flips }}) suggests significant instability in the selectivity filter,
                    which may disrupt ion coordination and permeation.
                    {% elif run_summary.Gyration_Flips > 0 %}
                    {{ run_summary.Gyration_Flips }} flip(s) were detected, which may temporarily affect ion coordination.
                    {% else %}
                    No significant flips were detected, suggesting a stable selectivity filter conformation.
                    {% endif %}
                    </p>
                 </div>
             {% else %}
                 <p><i>Carbonyl gyration analysis not performed or data unavailable.</i></p>
             {% endif %}
        </div>

        <div class="footer">End of Report</div>
    </div>
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

    # --- Load Base64 Encoded Images ---
    img_data = {}
    plot_files = {
        # Core distance plots (now in core_analysis/)
        "GG_Distance_Plot_raw": "core_analysis/GG_Distance_Plot_raw.png",
        "GG_Distance_Plot": "core_analysis/GG_Distance_Plot.png",
        "G_G_Distance_AC_Comparison": "core_analysis/G_G_Distance_AC_Comparison.png",
        "G_G_Distance_BD_Comparison": "core_analysis/G_G_Distance_BD_Comparison.png",
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
        # Water plots (now in inner_vestibule_analysis/)
        "Inner_Vestibule_Count_Plot": "inner_vestibule_analysis/Inner_Vestibule_Count_Plot.png",
        "Inner_Vestibule_Residence_Hist": "inner_vestibule_analysis/Inner_Vestibule_Residence_Hist.png",
        # Gyration analysis plots (now in gyration_analysis/)
        "G1_gyration_radii": "gyration_analysis/G1_gyration_radii.png",
    }

    logger.debug("Loading images for HTML report...")
    for key, filename in plot_files.items():
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
        env = jinja2.Environment(loader=jinja2.BaseLoader(), autoescape=jinja2.select_autoescape(['html', 'xml']))
        template = env.from_string(HTML_TEMPLATE)

        # Prepare context for rendering
        render_context = {
            'run_summary': run_summary, # Pass the loaded summary dictionary
            'img_data': img_data,
            'binding_site_data': binding_site_data,
            'generation_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
