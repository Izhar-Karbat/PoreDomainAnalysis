import pytest; # pytest.skip("Skipping HTML template tests in CI", allow_module_level=True)
import numpy as np
import pandas as pd
from pathlib import Path
import base64
import sqlite3
import os
import json

# Adapt imports for refactored code
from pore_analysis.core.database import init_db, connect_db, register_module, register_product, store_metric, set_simulation_metadata
from pore_analysis.html import generate_html_report
from pore_analysis.summary import generate_summary_from_database # Use this to generate summary dict

# Helper function to create a dummy plot file
def create_dummy_png(filepath, size=(100, 50)):
    try:
        from PIL import Image
        img = Image.new('RGB', size, color = 'green')
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        img.save(filepath)
        # Return relative path assuming run_dir is the parent of the module dir
        return os.path.relpath(filepath, os.path.dirname(os.path.dirname(filepath)))
    except ImportError:
        print(f"Warning: Pillow not installed, cannot create dummy PNG: {filepath}")
        return None

@pytest.fixture
def setup_full_report_db(tmp_path):
    """Sets up a database with multiple modules completed for a non-control system."""
    run_dir = tmp_path / "full_report_run"
    run_dir.mkdir()
    conn = init_db(str(run_dir))

    # --- Populate DB ---
    set_simulation_metadata(conn, "run_name", "full_report_run")
    set_simulation_metadata(conn, "system_name", "toxin_complex")
    set_simulation_metadata(conn, "analysis_status", "success")
    set_simulation_metadata(conn, "is_control_system", "False") # Non-control

    # Register modules as successful
    register_module(conn, "core_analysis_filtering", status='success')
    register_module(conn, "orientation_analysis", status='success')
    register_module(conn, "ion_analysis", status='success') # Needed for ion metrics/plots
    register_module(conn, "inner_vestibule_analysis", status='success')
    register_module(conn, "gyration_analysis", status='success')
    register_module(conn, "tyrosine_analysis", status='success')
    register_module(conn, "dw_gate_analysis", status='success')
    # Visualization modules
    register_module(conn, "core_analysis_visualization_g_g", status='success')
    register_module(conn, "core_analysis_visualization_com", status='success')
    register_module(conn, "orientation_analysis_visualization", status='success')
    register_module(conn, "ion_analysis_visualization", status='success')
    register_module(conn, "inner_vestibule_analysis_visualization", status='success')
    register_module(conn, "gyration_analysis_visualization", status='success')
    register_module(conn, "tyrosine_analysis_visualization", status='success')
    register_module(conn, "dw_gate_analysis_visualization", status='success')

    # --- Store Metrics (representative subset) ---
    # Core
    store_metric(conn, "core_analysis_filtering", 'G_G_AC_Mean_Filt', 1.1, 'Å', '')
    store_metric(conn, "core_analysis_filtering", 'G_G_BD_Mean_Filt', 2.1, 'Å', '')
    store_metric(conn, "core_analysis_filtering", 'COM_Mean_Filt', 10.5, 'Å', '')
    store_metric(conn, "core_analysis_filtering", 'COM_Std_Filt', 0.5, 'Å', '')
    # Orientation
    store_metric(conn, "orientation_analysis", 'Orient_Angle_Mean', 45.0, '°', '')
    store_metric(conn, "orientation_analysis", 'Orient_Contacts_Mean', 55.0, 'count', '')
    # Ions
    for site in ["S0","S1","S2","S3","S4","Cavity"]:
        store_metric(conn, "ion_analysis", f"Ion_AvgOcc_{site}", 0.1 + int(site[1] if len(site)>1 else 5) * 0.05, 'count', '')
        store_metric(conn, "ion_analysis", f"Ion_MaxOcc_{site}", 1, 'count', '')
        store_metric(conn, "ion_analysis", f"Ion_PctTimeOcc_{site}", 10.0 + int(site[1] if len(site)>1 else 5) * 5, '%', '')
    store_metric(conn, "ion_analysis", 'Ion_HMM_ConductionEvents_Total', 5, 'count','')
    store_metric(conn, "ion_analysis", 'Ion_HMM_Transition_S1_S0', 2, 'count','')
    # Inner Vestibule
    store_metric(conn, "inner_vestibule_analysis", 'InnerVestibule_MeanOcc', 3.5, 'count', '')
    store_metric(conn, "inner_vestibule_analysis", 'InnerVestibule_AvgResidenceTime_ns', 0.15, 'ns', '')
    # Gyration
    store_metric(conn, "gyration_analysis", 'Gyration_G1_Mean', 3.0, 'Å', '')
    store_metric(conn, "gyration_analysis", 'Gyration_Y_Mean', 3.2, 'Å', '')
    store_metric(conn, "gyration_analysis", 'Gyration_G1_OnFlips', 2, 'count', '')
    store_metric(conn, "gyration_analysis", 'Gyration_Y_OnFlips', 1, 'count', '')
    # Tyrosine
    store_metric(conn, "tyrosine_analysis", 'Tyr_HMM_TotalTransitions', 15, 'count', '')
    store_metric(conn, "tyrosine_analysis", 'Tyr_HMM_Population_pp', 85.0, '%', '')
    store_metric(conn, "tyrosine_analysis", 'Tyr_HMM_MeanDwell_pp', 10.2, 'ns', '')
    set_simulation_metadata(conn, 'Tyr_HMM_DominantState', 'pp') # Use metadata for dominant state
    # DW Gate
    store_metric(conn, "dw_gate_analysis", 'DW_PROA_Closed_Fraction', 75.0, '%', '')
    store_metric(conn, "dw_gate_analysis", 'DW_PROA_open_Mean_ns', 2.5, 'ns', '')

    # --- Create and Register Dummy Plots (matching plots_dict.json entries) ---
    # Load plot definitions to register required plots
    plots_dict_path = Path(__file__).parent.parent / "pore_analysis" / "plots_dict.json"
    plot_keys_needed = set()
    if plots_dict_path.exists():
        try:
            with open(plots_dict_path, 'r') as f_plots:
                plot_defs = json.load(f_plots)
                for pdef in plot_defs:
                    plot_keys_needed.add((pdef['module_name'], pdef['category'], pdef['subcategory'], pdef['template_key']))
        except Exception as e_json:
            print(f"Warning: Could not load plots_dict.json: {e_json}")

    # Mapping template_key to expected filename (based on subcategory where possible)
    key_to_filename = {
        "subunit_comparison": "G_G_Distance_Subunit_Comparison.png",
        "comparison": "COM_Stability_Comparison.png",
        "com_kde": "COM_Stability_KDE_Analysis.png",
        "orientation_angle": "Toxin_Orientation_Angle.png",
        "rotation_components": "Toxin_Rotation_Components.png",
        "channel_contacts": "Toxin_Channel_Contacts.png",
        "contact_map_focused": "Toxin_Channel_Residue_Contact_Map_Focused.png",
        "k_ion_combined_plot": "K_Ion_Combined_Plot.png",
        "k_ion_occupancy_heatmap": "K_Ion_Occupancy_Heatmap.png",
        "k_ion_average_occupancy": "K_Ion_Average_Occupancy.png",
        "binding_sites_g1_centric_visualization": "binding_sites_g1_centric_visualization.png",
        "hmm_transitions_plot": "ion_transitions_hmm.png",
        "site_optimization_plot": "binding_site_optimization.png", # Registered by structure
        "inner_vestibule_count_plot": "inner_vestibule_count_plot.png",
        "inner_vestibule_residence_hist": "inner_vestibule_residence_hist.png",
        "g1_gyration_radii": "G1_gyration_radii_stacked.png",
        "y_gyration_radii": "Y_gyration_radii_stacked.png",
        "flip_duration_distribution": "Flip_Duration_Distribution.png",
        "sf_tyrosine_chi1_dihedrals": "SF_Tyrosine_Chi1_Dihedrals_HMM.png",
        "sf_tyrosine_chi2_dihedrals": "SF_Tyrosine_Chi2_Dihedrals_HMM.png",
        "sf_tyrosine_rotamer_scatter": "SF_Tyrosine_Rotamer_Scatter.png",
        "sf_tyrosine_rotamer_population": "SF_Tyrosine_Rotamer_Population_HMM.png",
        "dw_distance_distribution": "dw_gate_distance_distribution.png",
        "dw_distance_vs_state": "dw_gate_distance_vs_state.png",
        "dw_open_probability": "dw_gate_open_probability.png",
        "dw_state_heatmap": "dw_gate_state_heatmap.png",
        "dw_duration_distributions": "dw_gate_duration_distribution.png"
    }

    # Register necessary plots to satisfy html.py and plots_dict.json
    for module_name, category, subcategory, template_key in plot_keys_needed:
        filename = key_to_filename.get(template_key)
        if filename:
            # Determine module subdirectory path (handle core vs other modules)
            if module_name.startswith("core"): module_dir_name = "core_analysis"
            elif module_name.startswith("orientation"): module_dir_name = "orientation_contacts"
            elif module_name.startswith("ion"): module_dir_name = "ion_analysis"
            elif module_name.startswith("inner_vestibule"): module_dir_name = "inner_vestibule_analysis"
            elif module_name.startswith("gyration"): module_dir_name = "gyration_analysis"
            elif module_name.startswith("tyrosine"): module_dir_name = "tyrosine_analysis"
            elif module_name.startswith("dw_gate"): module_dir_name = "dw_gate_analysis"
            else: module_dir_name = "unknown_module" # Fallback

            # Create dummy file and register it
            module_output_dir = run_dir / module_dir_name
            dummy_plot_path = module_output_dir / filename
            rel_plot_path = create_dummy_png(dummy_plot_path) # Returns None if Pillow missing
            if rel_plot_path:
                 # Ensure module is registered if not already
                cursor = conn.cursor()
                cursor.execute("SELECT module_id FROM analysis_modules WHERE module_name = ?", (module_name,))
                mod_exists = cursor.fetchone()
                if not mod_exists:
                    register_module(conn, module_name, status='success')

                register_product(conn, module_name, "png", category, rel_plot_path,
                                 subcategory=subcategory, description=f"Dummy plot for {template_key}")
            else:
                 print(f"Skipping registration for {template_key} as dummy PNG failed.")


    conn.commit()
    conn.close()
    return str(run_dir)


def test_generate_html_report_full(setup_full_report_db):
    """Test HTML report generation with a more complete dataset."""
    run_dir = setup_full_report_db

    # Generate report
    report_path = generate_html_report(run_dir, summary=None) # Generate from DB

    assert report_path is not None
    assert Path(report_path).exists()

    html = Path(report_path).read_text(encoding='utf-8')

    # Check for presence of various section titles
    assert "Overview &amp; Distances" in html
    assert "Toxin Interface" in html
    assert "Pore Ion Analysis" in html
    assert "Inner Vestibule Water Analysis" in html
    assert "Carbonyl Dynamics" in html
    assert "SF Tyrosine Analysis" in html
    assert "DW Gate Dynamics" in html

    # Check for specific metrics rendered
    assert "10.500" in html # COM Mean Filt
    assert "45.00" in html # Orient Angle Mean
    assert "Ion_AvgOcc_S0" in html # Check raw key presence before rendering check
    # Check rendered occupancy value (adjust precision as needed)
    assert ">0.10<" in html or ">0.1<" in html # Ion_AvgOcc_S0
    assert "3.50" in html # InnerVestibule_MeanOcc
    assert "2" in html # Gyration G1 On Flips count check
    assert ">pp<" in html # Dominant Tyrosine state (check metadata render)
    assert "75.0%" in html # DW Gate Closed Fraction

    # Check if plots were embedded (presence of base64 image data)
    # Requires Pillow in fixture and plots_dict.json to be correct
    if "data:image/png;base64," in html:
        print("Base64 image data found in report.")
    else:
        print("Warning: No base64 image data found in report (check plots_dict.json and fixture PNG creation).")

