# filename: pore_analysis/modules/gyration_analysis/visualization.py
"""
Visualization functions for carbonyl gyration analysis.
Generates plots based on data stored in the database.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import sqlite3
from typing import Dict, Optional, List, Any

# Import from core modules
try:
    from pore_analysis.core.plotting_style import STYLE, setup_style
    from pore_analysis.core.database import (
        register_module, update_module_status, get_product_path, register_product,
        get_module_status
    )
    from pore_analysis.core.logging import setup_system_logger
    # Import config values needed for plotting
    from pore_analysis.core.config import GYRATION_FLIP_THRESHOLD
    # Import function to get config values stored in DB (if needed for annotations)
    # from pore_analysis.core.database import get_config_parameters
except ImportError as e:
    print(f"Error importing dependency modules in gyration_analysis/visualization.py: {e}")
    raise

logger = logging.getLogger(__name__) # Use module-level logger

# Apply standard plotting style
setup_style()


def _plot_gyration_time_series(
    radii_csv_path: str,
    events_csv_path: Optional[str], # Optional path to flip events
    residue_type: str, # 'G1' or 'Y'
    output_dir: str,
    run_dir: str,
    db_conn: sqlite3.Connection,
    module_name: str # The visualization module name
    ) -> Optional[str]:
    """
    Generates gyration radius time series plot for G1 or Y, with each chain
    on a separate subplot stacked vertically. Adds background shading for
    'normal' and 'flipped' states based on GYRATION_FLIP_THRESHOLD.
    Optionally marks flip events loaded from a separate CSV.
    Adheres to plotting guidelines (no Python titles).
    """
    if not os.path.exists(radii_csv_path):
        logger.error(f"{residue_type.upper()} radii data file not found: {radii_csv_path}")
        return None

    try:
        df_radii = pd.read_csv(radii_csv_path)
        if 'Time (ns)' not in df_radii.columns:
            raise ValueError("Missing 'Time (ns)' column in radii CSV.")
        time_points = df_radii['Time (ns)'].values
        # Identify columns for the specific residue type
        dist_col_prefix = f"PRO" # Assuming columns like PROA_G1, PROB_G1 etc. Adjust if needed.
        chain_cols = sorted([col for col in df_radii.columns if col.endswith(f'_{residue_type.upper()}')])
        chains = sorted([col.split('_')[0] for col in chain_cols]) # Extract chain names like 'PROA'

        if not chain_cols or not chains:
            logger.warning(f"No '{residue_type.upper()}' columns found in {radii_csv_path}. Skipping plot.")
            return None
        n_chains = len(chains)

    except Exception as e:
        logger.error(f"Failed to load radii data or identify chains from {radii_csv_path}: {e}", exc_info=True)
        return None

    # Load event data if path provided
    df_events = None
    if events_csv_path and os.path.exists(events_csv_path):
        try:
            df_events_all = pd.read_csv(events_csv_path)
            # Filter events for the current residue type
            df_events = df_events_all[df_events_all['residue'] == residue_type.upper()].copy()
            if 'time_ns' not in df_events.columns or 'type' not in df_events.columns or 'chain' not in df_events.columns:
                 logger.warning("Events CSV missing 'time_ns', 'type', or 'chain' column. Cannot mark events.")
                 df_events = None # Treat as if no events file
        except Exception as e:
            logger.error(f"Failed to load or parse events data from {events_csv_path}: {e}")
            df_events = None # Continue without marking events

    # --- Plotting ---
    try:
        # Create stacked subplots, sharing the X axis
        fig, axes = plt.subplots(n_chains, 1, figsize=(12, 2.5 * n_chains), sharex=True)
        if n_chains == 1:
            axes = [axes] # Ensure axes is always iterable

        # Determine consistent Y-axis limits across all chains for this residue type
        all_radii_flat = df_radii[chain_cols].values.flatten()
        all_radii_flat = all_radii_flat[~np.isnan(all_radii_flat)] # Remove NaNs
        if len(all_radii_flat) > 0:
            y_min_global = np.min(all_radii_flat) * 0.95
            y_max_global = np.max(all_radii_flat) * 1.05
            # Ensure threshold is visible
            y_max_global = max(y_max_global, GYRATION_FLIP_THRESHOLD * 1.1)
            y_min_global = min(y_min_global, GYRATION_FLIP_THRESHOLD * 0.9)
        else:
            # Default range if no valid data
            y_min_global = 0
            y_max_global = GYRATION_FLIP_THRESHOLD * 1.5

        # Use consistent chain colors from STYLE
        chain_color_map = {}
        for chain_label in chains:
             chain_suffix = chain_label[-1] # Assumes last char is A, B, C, or D
             # Use bright colors for the main trace
             chain_color_map[chain_label] = STYLE['bright_colors'].get(chain_suffix, 'grey')


        # Plot data for each chain on its subplot
        for i, chain_label in enumerate(chains):
            ax = axes[i]
            col_name = f"{chain_label}_{residue_type.upper()}"
            radii = df_radii[col_name].values
            mask = np.isfinite(radii) & np.isfinite(time_points) # Ensure time is also finite

            # Filter data based on mask
            time_masked = time_points[mask]
            radii_masked = radii[mask]

            # Set consistent Y limits BEFORE plotting for background span
            ax.set_ylim(y_min_global, y_max_global)

            # Check if there's data to plot after masking
            if time_masked.size > 0 and radii_masked.size > 0:
                ax.plot(time_masked, radii_masked,
                        label=f'{chain_label} {residue_type.upper()}', # Label for potential future legend use
                        color=chain_color_map.get(chain_label, 'grey'), # Use mapped color
                        linewidth=STYLE['line_width'] * 0.8, zorder=1) # Slightly thinner line, zorder=1
            else:
                 ax.text(0.5, 0.5, f"No data for {chain_label}", ha='center', va='center', transform=ax.transAxes)

            # Add threshold line to each subplot
            ax.axhline(y=GYRATION_FLIP_THRESHOLD, **STYLE['threshold_style'], zorder=2)
            # Add text label for threshold only on the top plot for clarity
            if i == 0:
                 # Determine text x-position based on time_points array size
                 text_x_pos = time_points[-1] * 0.98 if time_points.size > 0 else 0.98
                 ax.text(text_x_pos,
                         GYRATION_FLIP_THRESHOLD,
                         f' Flip Thr: {GYRATION_FLIP_THRESHOLD:.2f} Å',
                         color=STYLE['threshold_style']['color'],
                         ha='right', va='bottom', fontsize=STYLE['font_sizes']['annotation']*0.9, zorder=3)

            # --- Add background shading ---
            ymin_ax, ymax_ax = ax.get_ylim() # Use established y-limits

            # Define colors for shading (use state_colors from STYLE or define new ones)
            normal_color = STYLE['state_colors'].get('closed', '#ADD8E6') # Default light blue
            flipped_color = STYLE['state_colors'].get('open', '#FACBAA') # Default light orange
            shade_alpha = 0.15 # Make it quite transparent

            # Normal region (< threshold)
            ax.axhspan(ymin_ax, GYRATION_FLIP_THRESHOLD, facecolor=normal_color, alpha=shade_alpha, zorder=0)

            # Flipped region (> threshold)
            ax.axhspan(GYRATION_FLIP_THRESHOLD, ymax_ax, facecolor=flipped_color, alpha=shade_alpha, zorder=0)
            # --- End background shading ---

            # Mark events for this specific chain (plot AFTER background)
            if df_events is not None:
                 chain_events = df_events[df_events['chain'] == chain_label]
                 on_events = chain_events[chain_events['type'] == 'on']
                 off_events = chain_events[chain_events['type'] == 'off']
                 # Add vertical lines for events, using the chain's color but different style/alpha
                 event_color = chain_color_map.get(chain_label, 'grey')
                 for t in on_events['time_ns']: ax.axvline(t, color=event_color, ls='--', alpha=0.5, lw=1.0, zorder=1) # 'On' flips dashed
                 for t in off_events['time_ns']: ax.axvline(t, color=event_color, ls=':', alpha=0.5, lw=1.0, zorder=1) # 'Off' flips dotted

            # Configure axes
            ax.set_ylabel(f"{chain_label}\nRadius (Å)", fontsize=STYLE['font_sizes']['axis_label']*0.9) # Slightly smaller axis label
            ax.tick_params(axis='y', labelsize=STYLE['font_sizes']['tick_label'])
            ax.grid(axis='y', linestyle=STYLE['grid']['linestyle'], alpha=STYLE['grid']['alpha'], color=STYLE['grid']['color'], zorder=0) # Keep grid behind shading

            # Remove top/right spines for cleaner look like the example
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

            # Hide x-axis labels for all but the bottom plot
            if i < n_chains - 1:
                plt.setp(ax.get_xticklabels(), visible=False)
                ax.tick_params(axis='x', bottom=False) # Hide ticks too

        # Set label only on the last (bottom) axis
        axes[-1].set_xlabel('Time (ns)', fontsize=STYLE['font_sizes']['axis_label'])
        axes[-1].tick_params(axis='x', labelsize=STYLE['font_sizes']['tick_label'])

        # No figure title (suptitle) per guidelines
        # fig.suptitle(f'{residue_type.upper()} Carbonyl Gyration Radius per Chain', fontsize=STYLE['font_sizes']['title'])

        plt.tight_layout()
        # Adjust spacing between subplots if needed
        plt.subplots_adjust(hspace=0.15) # Reduce vertical space

        # --- Save Figure ---
        plot_filename = f"{residue_type.upper()}_gyration_radii_stacked.png" # Keep original filename
        plot_filepath = os.path.join(output_dir, plot_filename)
        rel_path = os.path.relpath(plot_filepath, run_dir)

        fig.savefig(plot_filepath, dpi=150)
        plt.close(fig)
        logger.info(f"Saved stacked {residue_type.upper()} gyration plot to {plot_filepath}")

        # --- Register product ---
        # Use the original subcategory names expected by html.py/plots_dict.json
        subcategory_key = f"{residue_type.lower()}_gyration" # e.g., g1_gyration or y_gyration
        register_product(db_conn, module_name, "png", "plot", rel_path,
                         subcategory=subcategory_key,
                         description=f"Stacked time series of {residue_type.upper()} carbonyl gyration radius per chain with state background.")
        return rel_path

    except Exception as e:
        logger.error(f"Failed to generate stacked {residue_type.upper()} gyration plot: {e}", exc_info=True)
        if 'fig' in locals() and plt.fignum_exists(fig.number): plt.close(fig)
        return None

# --- Main Visualization Orchestrator (generate_gyration_plots) ---
# This function remains unchanged as it already calls the plotting function
# twice with different residue_type arguments. The internal logic of the
# plotting function is what has been modified.

def generate_gyration_plots(
    run_dir: str,
    db_conn: sqlite3.Connection
) -> Dict[str, Any]:
    """
    Generates all standard plots for the gyration analysis module.
    Retrieves necessary data file paths from the database.

    Args:
        run_dir: Path to the specific run directory.
        db_conn: Active database connection.

    Returns:
        Dictionary containing status and paths to generated plots.
    """
    module_name = "gyration_analysis_visualization"
    start_time = time.time()
    register_module(db_conn, module_name, status='running')
    logger_local = setup_system_logger(run_dir) # Use a different name to avoid conflict
    if logger_local is None: logger_local = logging.getLogger(__name__) # Fallback

    results: Dict[str, Any] = {'status': 'failed', 'plots': {}}
    output_dir = os.path.join(run_dir, "gyration_analysis") # Ensure output dir matches computation
    os.makedirs(output_dir, exist_ok=True)

    # --- Check computation status ---
    comp_status = get_module_status(db_conn, "gyration_analysis")
    if comp_status != 'success':
        results['status'] = 'skipped'
        results['error'] = f"Skipping visualization: Computation status was '{comp_status}'."
        logger_local.warning(results['error'])
        update_module_status(db_conn, module_name, 'skipped', error_message=results['error'])
        return results

    # --- Retrieve Data File Paths ---
    g1_radii_rel = get_product_path(db_conn, 'csv', 'data', 'g1_gyration_data', 'gyration_analysis')
    y_radii_rel = get_product_path(db_conn, 'csv', 'data', 'y_gyration_data', 'gyration_analysis')
    events_rel = get_product_path(db_conn, 'csv', 'data', 'gyration_flip_events', 'gyration_analysis') # Optional

    g1_radii_abs = os.path.join(run_dir, g1_radii_rel) if g1_radii_rel else None
    y_radii_abs = os.path.join(run_dir, y_radii_rel) if y_radii_rel else None
    events_abs = os.path.join(run_dir, events_rel) if events_rel else None

    # --- Generate Plots ---
    plots_generated = 0
    plots_failed = 0

    # Plot G1 Radii (will now be stacked)
    if g1_radii_abs:
        g1_plot_path = _plot_gyration_time_series(g1_radii_abs, events_abs, 'G1', output_dir, run_dir, db_conn, module_name)
        if g1_plot_path:
            # Use the key expected by plots_dict.json / html.py
            results['plots']['g1_gyration_radii'] = g1_plot_path
            plots_generated += 1
        else: plots_failed += 1
    else: logger_local.warning("Skipping G1 gyration plot: Radii data path not found.")

    # Plot Y Radii (will now be stacked)
    if y_radii_abs:
        y_plot_path = _plot_gyration_time_series(y_radii_abs, events_abs, 'Y', output_dir, run_dir, db_conn, module_name)
        if y_plot_path:
            # Use the key expected by plots_dict.json / html.py
            results['plots']['y_gyration_radii'] = y_plot_path
            plots_generated += 1
        else: plots_failed += 1
    else: logger_local.warning("Skipping Y gyration plot: Radii data path not found.")

    # Plot Flip Durations (remains unchanged)
    # Assume _plot_flip_duration_distribution exists and is correct
    # If it doesn't exist, define it or remove this call
    try:
        from .visualization import _plot_flip_duration_distribution # Try importing if defined locally
        if events_abs:
            duration_plot_path = _plot_flip_duration_distribution(events_abs, output_dir, run_dir, db_conn, module_name)
            if duration_plot_path:
                # Use the key expected by plots_dict.json / html.py
                results['plots']['flip_duration_distribution'] = duration_plot_path
                plots_generated += 1
            else: plots_failed += 1
        else: logger_local.warning("Skipping flip duration plot: Events data path not found.")
    except ImportError:
         logger_local.warning("Function _plot_flip_duration_distribution not found. Skipping duration plot.")
    except NameError:
         logger_local.warning("Function _plot_flip_duration_distribution not defined. Skipping duration plot.")


    # --- Finalize ---
    exec_time = time.time() - start_time
    final_status = 'success' if plots_failed == 0 and plots_generated > 0 else ('failed' if plots_failed > 0 else 'skipped')
    error_msg = f"{plots_failed} plot(s) failed to generate." if plots_failed > 0 else None

    update_module_status(db_conn, module_name, final_status, execution_time=exec_time, error_message=error_msg)
    logger_local.info(f"--- Gyration Analysis Visualization finished in {exec_time:.2f} seconds (Status: {final_status}) ---")
    results['status'] = final_status
    if error_msg: results['error'] = error_msg

    return results


# --- Helper function for flip duration plot (if needed) ---
# Add the definition of _plot_flip_duration_distribution here if it's not
# already defined in the file or imported correctly.
# Example placeholder:
def _plot_flip_duration_distribution(
    events_csv_path: str, # Path to gyration_flip_events.csv
    output_dir: str,
    run_dir: str,
    db_conn: sqlite3.Connection,
    module_name: str # The visualization module name
    ) -> Optional[str]:
    """
    Plots the distribution of confirmed flip durations for G1 and Y.
    (Implementation copied from previous context if needed)
    """
    if not events_csv_path or not os.path.exists(events_csv_path):
        logger.warning(f"Flip events data file not found or path not provided: {events_csv_path}. Skipping duration plot.")
        return None

    try:
        df_events = pd.read_csv(events_csv_path)
        df_durations = df_events[df_events['type'] == 'off'].copy()
        if 'duration_ns' not in df_durations.columns or 'residue' not in df_durations.columns:
             logger.warning("Events CSV missing 'duration_ns' or 'residue' column. Cannot plot durations.")
             return None
        df_durations.dropna(subset=['duration_ns'], inplace=True)
        if df_durations.empty:
            logger.info("No valid flip duration data found to plot.")
            return None
        df_durations.rename(columns={'residue': 'Residue', 'duration_ns': 'Duration (ns)'}, inplace=True)

    except Exception as e:
        logger.error(f"Failed to load or process events data from {events_csv_path}: {e}", exc_info=True)
        return None

    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        pastel_palette_map = {'G1': STYLE['chain_colors'].get('B'), 'Y': STYLE['chain_colors'].get('A')}
        bright_palette_map = {'G1': STYLE['bright_colors'].get('B'), 'Y': STYLE['bright_colors'].get('A')}

        sns.violinplot(x='Residue', y='Duration (ns)', data=df_durations, order=['G1', 'Y'],
                       inner=None, palette=pastel_palette_map, cut=0, ax=ax)
        sns.swarmplot(x='Residue', y='Duration (ns)', data=df_durations, order=['G1', 'Y'],
                      size=STYLE['event_marker']['markersize']*0.8, palette=bright_palette_map,
                      edgecolor=STYLE['event_marker']['markeredgecolor'],
                      linewidth=STYLE['event_marker']['markeredgewidth'], ax=ax)

        ax.set_ylabel('Duration in Flipped State (ns)')
        ax.set_xlabel('Residue Type')
        ax.grid(True, axis='y', linestyle=STYLE['grid']['linestyle'], alpha=STYLE['grid']['alpha'])
        plt.tight_layout()

        plot_filename = "Flip_Duration_Distribution.png"
        plot_filepath = os.path.join(output_dir, plot_filename)
        rel_path = os.path.relpath(plot_filepath, run_dir)

        fig.savefig(plot_filepath, dpi=150)
        plt.close(fig)
        logger.info(f"Saved flip duration distribution plot to {plot_filepath}")

        register_product(db_conn, module_name, "png", "plot", rel_path,
                         subcategory="flip_duration",
                         description="Distribution of confirmed carbonyl flip durations.")
        return rel_path

    except Exception as e:
        logger.error(f"Failed to generate flip duration plot: {e}", exc_info=True)
        if 'fig' in locals() and plt.fignum_exists(fig.number): plt.close(fig)
        return None
