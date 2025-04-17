#!/usr/bin/env python3
"""
Pore Symmetry Analysis for K+ Channel MD Simulations

This script analyzes the symmetry of K+ channel pores by comparing A-C and B-D distances.
It processes G-G distance CSV files from both toxin and control systems, calculates symmetry
metrics, and generates visualizations to highlight differences between the systems.

The script:
1. Finds G_G_Distance_Filtered.csv files in toxin and control directories
2. Calculates symmetry metrics (AC/BD ratio, AC-BD difference, ellipticity)
3. Produces time series plots, distributions, and statistical comparisons
4. Generates a comprehensive visual report
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.gridspec import GridSpec
import argparse
from tqdm import tqdm

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['figure.dpi'] = 150

def find_gg_distance_files(base_dirs):
    """Find all G_G_Distance_Filtered.csv files in the specified directories."""
    all_files = {}
    for dir_type, base_dir in base_dirs.items():
        pattern = os.path.join(base_dir, "**", "G_G_Distance_Filtered.csv")
        files = glob.glob(pattern, recursive=True)
        all_files[dir_type] = files
        print(f"Found {len(files)} {dir_type} G-G distance files")
    return all_files

def load_gg_data(file_path):
    """Load G-G distance data from a CSV file and return as a DataFrame."""
    try:
        df = pd.read_csv(file_path)
        # Check if the required columns exist
        required_cols = ['Time (ns)', 'A_C_Distance_Filtered', 'B_D_Distance_Filtered']
        if not all(col in df.columns for col in required_cols):
            print(f"Warning: Missing required columns in {file_path}")
            return None
        
        # Keep only the required columns
        df = df[required_cols]
        
        # Handle NaN values
        # 1. First check if we have too many NaNs
        nan_percentage = df[['A_C_Distance_Filtered', 'B_D_Distance_Filtered']].isna().mean().mean() * 100
        if nan_percentage > 20:  # If more than 20% of distance values are NaN
            print(f"Warning: {file_path} has {nan_percentage:.1f}% NaN values, skipping")
            return None
        
        # 2. For remaining files with acceptable NaN levels, interpolate
        df = df.interpolate(method='linear')
        
        # Add the file path
        df['source_file'] = file_path
        
        # Extract run information from path
        path_parts = file_path.split(os.sep)
        run_name = None
        # Look for R1, R2, etc. in path parts
        for part in path_parts:
            if part.startswith('R') and len(part) <= 3 and part[1:].isdigit():
                run_name = part
                break
        
        # If we couldn't find R1-R10 pattern, use the parent directory name
        if run_name is None:
            parent_dir = os.path.basename(os.path.dirname(file_path))
            run_name = parent_dir
        
        df['run_name'] = run_name
        
        # Determine system type (toxin or control)
        if 'toxin' in file_path.lower():
            df['system_type'] = 'Toxin'
        elif 'control' in file_path.lower():
            df['system_type'] = 'Control'
        else:
            df['system_type'] = 'Unknown'
        
        return df
    
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def calculate_symmetry_metrics(df):
    """
    Calculate various symmetry metrics for the pore.
    
    Args:
        df: DataFrame with A_C_Distance_Filtered and B_D_Distance_Filtered columns
        
    Returns:
        DataFrame with added symmetry metrics
    """
    # Calculate the ratio AC/BD (values > 1 mean AC > BD)
    df['ac_bd_ratio'] = df['A_C_Distance_Filtered'] / df['B_D_Distance_Filtered']
    
    # Calculate the difference AC-BD (positive values mean AC > BD)
    df['ac_bd_diff'] = df['A_C_Distance_Filtered'] - df['B_D_Distance_Filtered']
    
    # Calculate ellipticity as max(AC,BD)/min(AC,BD)
    df['ellipticity'] = np.maximum(df['A_C_Distance_Filtered'], df['B_D_Distance_Filtered']) / \
                        np.minimum(df['A_C_Distance_Filtered'], df['B_D_Distance_Filtered'])
    
    # Calculate pore area (approximating as ellipse)
    df['pore_area'] = np.pi * (df['A_C_Distance_Filtered']/2) * (df['B_D_Distance_Filtered']/2)
    
    # Calculate average diameter
    df['avg_diameter'] = (df['A_C_Distance_Filtered'] + df['B_D_Distance_Filtered']) / 2
    
    # Dominant axis (1 if AC > BD, -1 if BD > AC, 0 if equal)
    df['dominant_axis'] = np.sign(df['ac_bd_diff'])
    
    return df

def process_gg_files(file_dict):
    """
    Process all G-G distance files and combine data.
    
    Args:
        file_dict: Dictionary with keys 'toxin' and 'control', values are lists of file paths
        
    Returns:
        DataFrame with combined data from all files
    """
    all_data = []
    
    for system_type, files in file_dict.items():
        for file_path in tqdm(files, desc=f"Processing {system_type} files"):
            df = load_gg_data(file_path)
            if df is not None:
                # Calculate symmetry metrics
                df = calculate_symmetry_metrics(df)
                all_data.append(df)
    
    if not all_data:
        raise ValueError("No valid data found in any of the files")
    
    # Combine all dataframes
    combined_df = pd.concat(all_data, ignore_index=True)
    
    return combined_df

def calculate_statistics(df):
    """
    Calculate statistics for each run and system type.
    
    Args:
        df: DataFrame with symmetry metrics
        
    Returns:
        DataFrame with statistics per run
    """
    # Group by run and calculate statistics
    metrics = ['A_C_Distance_Filtered', 'B_D_Distance_Filtered', 
               'ac_bd_ratio', 'ac_bd_diff', 'ellipticity', 
               'pore_area', 'avg_diameter']
    
    stats_list = []
    
    # First, calculate per-run statistics
    for (run_name, system_type), group in df.groupby(['run_name', 'system_type']):
        run_stats = {'run_name': run_name, 'system_type': system_type}
        
        for metric in metrics:
            run_stats[f'{metric}_mean'] = group[metric].mean()
            run_stats[f'{metric}_std'] = group[metric].std()
            run_stats[f'{metric}_min'] = group[metric].min()
            run_stats[f'{metric}_max'] = group[metric].max()
            
            # Calculate the percentage of frames where AC > BD
            if metric == 'ac_bd_diff':
                run_stats['pct_ac_dominant'] = (group[metric] > 0).mean() * 100
                run_stats['pct_bd_dominant'] = (group[metric] < 0).mean() * 100
        
        stats_list.append(run_stats)
    
    run_stats_df = pd.DataFrame(stats_list)
    
    # Now calculate overall system type statistics
    system_stats = []
    
    for system_type, group in run_stats_df.groupby('system_type'):
        sys_stats = {'system_type': system_type}
        
        for col in run_stats_df.columns:
            if col in ['run_name', 'system_type']:
                continue
            
            sys_stats[f'{col}_mean'] = group[col].mean()
            sys_stats[f'{col}_std'] = group[col].std()
            
            # For important metrics, also calculate statistical significance
            if col.endswith('_mean') and col.split('_mean')[0] in metrics:
                base_metric = col.split('_mean')[0]
                
                # Only calculate if we have both toxin and control data
                if len(run_stats_df['system_type'].unique()) > 1:
                    toxin_values = run_stats_df[run_stats_df['system_type'] == 'Toxin'][col]
                    control_values = run_stats_df[run_stats_df['system_type'] == 'Control'][col]
                    
                    if len(toxin_values) >= 2 and len(control_values) >= 2:
                        try:
                            t_stat, p_value = stats.ttest_ind(toxin_values, control_values, equal_var=False)
                            sys_stats[f'{base_metric}_p_value'] = p_value
                            sys_stats[f'{base_metric}_significant'] = p_value < 0.05
                        except:
                            sys_stats[f'{base_metric}_p_value'] = None
                            sys_stats[f'{base_metric}_significant'] = None
        
        system_stats.append(sys_stats)
    
    system_stats_df = pd.DataFrame(system_stats)
    
    return run_stats_df, system_stats_df

def plot_symmetry_time_series(df, output_dir):
    """
    Create a time series plot showing pore symmetry metrics over time.
    
    Args:
        df: DataFrame with symmetry metrics
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample every Nth point to avoid overcrowding (adjust based on your data)
    sample_interval = max(1, len(df) // 10000)
    
    # Create one plot per run for better clarity
    for (run_name, system_type), group in df.groupby(['run_name', 'system_type']):
        sampled_data = group.iloc[::sample_interval].copy()
        
        # Create the figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Plot 1: AC and BD distances
        ax1.plot(sampled_data['Time (ns)'], sampled_data['A_C_Distance_Filtered'], 
                 label='A-C Distance', color='blue', alpha=0.8)
        ax1.plot(sampled_data['Time (ns)'], sampled_data['B_D_Distance_Filtered'], 
                 label='B-D Distance', color='green', alpha=0.8)
        ax1.set_ylabel('Distance (Å)', fontsize=12)
        ax1.set_title(f'G-G Distances Over Time - {system_type} {run_name}', fontsize=14)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Symmetry metrics
        ax2.plot(sampled_data['Time (ns)'], sampled_data['ac_bd_diff'], 
                 label='AC-BD Difference', color='red', alpha=0.8)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_ylabel('AC-BD Difference (Å)', fontsize=12)
        ax2.set_xlabel('Time (ns)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)
        
        # Add text with statistics
        stats_text = (
            f"Mean AC: {group['A_C_Distance_Filtered'].mean():.2f} ± {group['A_C_Distance_Filtered'].std():.2f} Å\n"
            f"Mean BD: {group['B_D_Distance_Filtered'].mean():.2f} ± {group['B_D_Distance_Filtered'].std():.2f} Å\n"
            f"Mean AC-BD: {group['ac_bd_diff'].mean():.2f} ± {group['ac_bd_diff'].std():.2f} Å\n"
            f"Ellipticity: {group['ellipticity'].mean():.2f} ± {group['ellipticity'].std():.2f}\n"
            f"AC > BD: {(group['ac_bd_diff'] > 0).mean() * 100:.1f}% of frames"
        )
        
        # Add a text box with statistics
        ax2.text(0.02, 0.05, stats_text, transform=ax2.transAxes, fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'symmetry_time_series_{system_type}_{run_name}.png'))
        plt.close()

def plot_symmetry_distributions(df, output_dir):
    """
    Create distribution plots for symmetry metrics.
    
    Args:
        df: DataFrame with symmetry metrics
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create distribution plots comparing toxin vs control
    if 'Toxin' in df['system_type'].values and 'Control' in df['system_type'].values:
        # Define metrics to plot
        metrics = [
            ('ac_bd_diff', 'AC-BD Difference (Å)'),
            ('ellipticity', 'Ellipticity (max/min)'),
            ('pore_area', 'Pore Area (Å²)'),
            ('avg_diameter', 'Average Diameter (Å)')
        ]
        
        # Create a 2x2 grid of distribution plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i, (metric, label) in enumerate(metrics):
            ax = axes[i]
            
            # Create distribution plot
            sns.kdeplot(data=df, x=metric, hue='system_type', fill=True, alpha=0.5, ax=ax)
            
            # Add mean lines
            for system, color in zip(['Toxin', 'Control'], ['blue', 'orange']):
                mean_val = df[df['system_type'] == system][metric].mean()
                ax.axvline(mean_val, color=color, linestyle='--', 
                          label=f'{system} Mean: {mean_val:.3f}')
            
            ax.set_xlabel(label, fontsize=12)
            ax.set_ylabel('Density', fontsize=12)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Distribution of Pore Symmetry Metrics: Toxin vs Control', fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        plt.savefig(os.path.join(output_dir, 'symmetry_distributions.png'))
        plt.close()
        
        # Create a violin plot for AC-BD difference
        plt.figure(figsize=(8, 6))
        sns.violinplot(data=df, x='system_type', y='ac_bd_diff', inner='box')
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.xlabel('System Type', fontsize=12)
        plt.ylabel('AC-BD Difference (Å)', fontsize=12)
        plt.title('Distribution of Pore Asymmetry (AC-BD Difference)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'ac_bd_diff_violin.png'))
        plt.close()

def create_summary_figure(run_stats_df, system_stats_df, output_dir):
    """
    Create a comprehensive summary figure comparing toxin and control systems.
    
    Args:
        run_stats_df: DataFrame with statistics per run
        system_stats_df: DataFrame with statistics per system type
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if we have both toxin and control data
    if not all(system in run_stats_df['system_type'].values for system in ['Toxin', 'Control']):
        print("Cannot create summary figure - need both toxin and control data")
        return
    
    # Create a large figure with multiple panels
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(3, 3, figure=fig)
    
    # Panel 1: AC vs BD mean distances (scatter plot)
    ax1 = fig.add_subplot(gs[0, 0])
    sns.scatterplot(data=run_stats_df, x='A_C_Distance_Filtered_mean', y='B_D_Distance_Filtered_mean', 
                   hue='system_type', style='system_type', s=100, ax=ax1)
    
    # Add diagonal line (symmetry)
    lims = [
        np.min([ax1.get_xlim(), ax1.get_ylim()]),  # min of both axes
        np.max([ax1.get_xlim(), ax1.get_ylim()]),  # max of both axes
    ]
    ax1.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
    ax1.set_xlabel('Mean A-C Distance (Å)', fontsize=12)
    ax1.set_ylabel('Mean B-D Distance (Å)', fontsize=12)
    ax1.set_title('Pore Dimensions', fontsize=13)
    
    # Panel 2: Ellipticity and AC dominance (double y-axis)
    ax2 = fig.add_subplot(gs[0, 1])
    bars1 = sns.barplot(data=run_stats_df, x='system_type', y='ellipticity_mean', 
                      errorbar='sd', ax=ax2, alpha=0.7, color='skyblue')
    ax2.set_ylabel('Mean Ellipticity (max/min)', fontsize=12)
    ax2.set_title('Pore Ellipticity & AC Dominance', fontsize=13)
    
    # Add second y-axis for AC dominance percentage
    ax2_twin = ax2.twinx()
    bars2 = sns.barplot(data=run_stats_df, x='system_type', y='pct_ac_dominant', 
                      errorbar='sd', ax=ax2_twin, alpha=0.5, color='salmon')
    ax2_twin.set_ylabel('% Frames with AC > BD', fontsize=12)
    
    # Add legend manually
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='skyblue', alpha=0.7, label='Ellipticity'),
        Patch(facecolor='salmon', alpha=0.5, label='AC Dominance %')
    ]
    ax2.legend(handles=legend_elements, loc='upper right')
    
    # Panel 3: AC-BD difference comparison
    ax3 = fig.add_subplot(gs[0, 2])
    sns.boxplot(data=run_stats_df, x='system_type', y='ac_bd_diff_mean', ax=ax3)
    sns.stripplot(data=run_stats_df, x='system_type', y='ac_bd_diff_mean', 
                 ax=ax3, color='black', alpha=0.5)
    ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax3.set_ylabel('Mean AC-BD Difference (Å)', fontsize=12)
    ax3.set_title('Pore Asymmetry', fontsize=13)
    
    # Panel 4: Per-run symmetry metrics
    ax4 = fig.add_subplot(gs[1, :2])
    
    # Reshape data for better visualization
    plot_data = []
    for _, row in run_stats_df.iterrows():
        plot_data.append({
            'run_name': row['run_name'],
            'system_type': row['system_type'],
            'metric': 'AC Distance',
            'value': row['A_C_Distance_Filtered_mean']
        })
        plot_data.append({
            'run_name': row['run_name'],
            'system_type': row['system_type'],
            'metric': 'BD Distance',
            'value': row['B_D_Distance_Filtered_mean']
        })
        plot_data.append({
            'run_name': row['run_name'],
            'system_type': row['system_type'],
            'metric': 'Ellipticity',
            'value': row['ellipticity_mean']
        })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create a grouped bar plot
    sns.barplot(data=plot_df, x='run_name', y='value', hue='metric', 
               ax=ax4, palette=['blue', 'green', 'red'])
    
    # Split plot by system type with clear visual separation
    toxin_runs = run_stats_df[run_stats_df['system_type'] == 'Toxin']['run_name'].tolist()
    control_runs = run_stats_df[run_stats_df['system_type'] == 'Control']['run_name'].tolist()
    
    # Add vertical line to separate toxin and control runs
    if toxin_runs and control_runs:
        # Find the position to draw the separator (between last toxin and first control)
        toxin_pos = [i for i, run in enumerate(plot_df['run_name'].unique()) if run in toxin_runs]
        control_pos = [i for i, run in enumerate(plot_df['run_name'].unique()) if run in control_runs]
        
        if toxin_pos and control_pos:
            separator_x = (max(toxin_pos) + min(control_pos)) / 2
            ax4.axvline(x=separator_x, color='black', linestyle='--', alpha=0.5)
    
    ax4.set_xlabel('Run', fontsize=12)
    ax4.set_ylabel('Value', fontsize=12)
    ax4.set_title('Per-Run Metrics Comparison', fontsize=13)
    ax4.legend(title='Metric', fontsize=10)
    
    # Panel 5: Mean pore shape visualization (ellipse)
    ax5 = fig.add_subplot(gs[1, 2])
    
    # Calculate mean AC and BD for both system types
    toxin_data = run_stats_df[run_stats_df['system_type'] == 'Toxin']
    control_data = run_stats_df[run_stats_df['system_type'] == 'Control']
    
    toxin_ac = toxin_data['A_C_Distance_Filtered_mean'].mean() / 2
    toxin_bd = toxin_data['B_D_Distance_Filtered_mean'].mean() / 2
    control_ac = control_data['A_C_Distance_Filtered_mean'].mean() / 2
    control_bd = control_data['B_D_Distance_Filtered_mean'].mean() / 2
    
    # Create ellipses to represent pore shapes
    from matplotlib.patches import Ellipse
    
    # Define center coordinates
    toxin_center = (0, 0.5)
    control_center = (0, -0.5)
    
    # Create ellipses (AC is x-axis, BD is y-axis)
    toxin_ellipse = Ellipse(toxin_center, width=toxin_ac*2, height=toxin_bd*2, 
                           angle=0, alpha=0.6, facecolor='blue', edgecolor='navy', linewidth=2)
    control_ellipse = Ellipse(control_center, width=control_ac*2, height=control_bd*2, 
                             angle=0, alpha=0.6, facecolor='orange', edgecolor='darkorange', linewidth=2)
    
    ax5.add_patch(toxin_ellipse)
    ax5.add_patch(control_ellipse)
    
    # Add annotations
    ax5.annotate('Toxin', xy=(toxin_center[0]+0.1, toxin_center[1]), 
                xytext=(toxin_center[0]+max(toxin_ac, toxin_bd)+0.2, toxin_center[1]),
                fontsize=12, fontweight='bold', color='blue',
                arrowprops=dict(facecolor='blue', shrink=0.05, width=1.5, headwidth=8))
    
    ax5.annotate('Control', xy=(control_center[0]+0.1, control_center[1]), 
                xytext=(control_center[0]+max(control_ac, control_bd)+0.2, control_center[1]),
                fontsize=12, fontweight='bold', color='darkorange',
                arrowprops=dict(facecolor='darkorange', shrink=0.05, width=1.5, headwidth=8))
    
    # Add dimension labels
    ax5.annotate(f'AC: {toxin_ac*2:.2f} Å', 
                xy=(0, toxin_center[1]+toxin_bd+0.1), 
                ha='center', va='bottom', fontsize=10, color='blue')
    ax5.annotate(f'BD: {toxin_bd*2:.2f} Å', 
                xy=(toxin_ac+0.1, toxin_center[1]), 
                ha='left', va='center', fontsize=10, color='blue')
    
    ax5.annotate(f'AC: {control_ac*2:.2f} Å', 
                xy=(0, control_center[1]-control_bd-0.1), 
                ha='center', va='top', fontsize=10, color='darkorange')
    ax5.annotate(f'BD: {control_bd*2:.2f} Å', 
                xy=(control_ac+0.1, control_center[1]), 
                ha='left', va='center', fontsize=10, color='darkorange')
    
    # Set axis limits and labels
    max_radius = max(toxin_ac, toxin_bd, control_ac, control_bd) + 0.5
    ax5.set_xlim(-max_radius, max_radius)
    ax5.set_ylim(-max_radius-0.5, max_radius+0.5)
    ax5.set_aspect('equal')
    ax5.set_title('Average Pore Shape', fontsize=13)
    ax5.set_xlabel('A-C Axis (Å)', fontsize=12)
    ax5.set_ylabel('B-D Axis (Å)', fontsize=12)
    ax5.grid(True, alpha=0.3)
    
    # Panel 6: Summary statistics table
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('off')
    
    # Extract relevant statistics from system_stats_df
    if len(system_stats_df) > 0:
        table_data = []
        metrics = [
            ('A_C_Distance_Filtered', 'AC Distance (Å)'),
            ('B_D_Distance_Filtered', 'BD Distance (Å)'),
            ('ac_bd_ratio', 'AC/BD Ratio'),
            ('ac_bd_diff', 'AC-BD Difference (Å)'),
            ('ellipticity', 'Ellipticity'),
            ('pore_area', 'Pore Area (Å²)'),
            ('avg_diameter', 'Average Diameter (Å)'),
            ('pct_ac_dominant', '% Frames AC > BD')
        ]
        
        for metric_key, metric_name in metrics:
            row = [metric_name]
            
            # Add data for toxin
            toxin_row = system_stats_df[system_stats_df['system_type'] == 'Toxin'].iloc[0] if 'Toxin' in system_stats_df['system_type'].values else None
            if toxin_row is not None:
                mean_key = f'{metric_key}_mean_mean' if f'{metric_key}_mean_mean' in toxin_row else f'{metric_key}_mean'
                std_key = f'{metric_key}_mean_std' if f'{metric_key}_mean_std' in toxin_row else f'{metric_key}_std'
                
                if mean_key in toxin_row and std_key in toxin_row:
                    row.append(f"{toxin_row[mean_key]:.3f} ± {toxin_row[std_key]:.3f}")
                else:
                    row.append('N/A')
            else:
                row.append('N/A')
                
            # Add data for control
            control_row = system_stats_df[system_stats_df['system_type'] == 'Control'].iloc[0] if 'Control' in system_stats_df['system_type'].values else None
            if control_row is not None:
                mean_key = f'{metric_key}_mean_mean' if f'{metric_key}_mean_mean' in control_row else f'{metric_key}_mean'
                std_key = f'{metric_key}_mean_std' if f'{metric_key}_mean_std' in control_row else f'{metric_key}_std'
                
                if mean_key in control_row and std_key in control_row:
                    row.append(f"{control_row[mean_key]:.3f} ± {control_row[std_key]:.3f}")
                else:
                    row.append('N/A')
            else:
                row.append('N/A')
            
            # Add difference calculation
            if toxin_row is not None and control_row is not None:
                mean_key_t = f'{metric_key}_mean_mean' if f'{metric_key}_mean_mean' in toxin_row else f'{metric_key}_mean'
                mean_key_c = f'{metric_key}_mean_mean' if f'{metric_key}_mean_mean' in control_row else f'{metric_key}_mean'
                
                if mean_key_t in toxin_row and mean_key_c in control_row:
                    diff = toxin_row[mean_key_t] - control_row[mean_key_c]
                    pct_change = (diff / abs(control_row[mean_key_c])) * 100 if control_row[mean_key_c] != 0 else float('inf')
                    
                    # Check if there's a p-value for this metric
                    p_value_key = f'{metric_key}_p_value'
                    if p_value_key in toxin_row:
                        p_value = toxin_row[p_value_key]
                        sig_marker = '*' if p_value < 0.05 else ''
                        row.append(f"{diff:.3f} ({pct_change:.1f}%){sig_marker}")
                    else:
                        row.append(f"{diff:.3f} ({pct_change:.1f}%)")
                else:
                    row.append('N/A')
            else:
                row.append('N/A')
            
            table_data.append(row)
        
        # Create the table
        table = ax6.table(
            cellText=table_data,
            colLabels=['Metric', 'Toxin', 'Control', 'Difference (%)'],
            loc='center',
            cellLoc='center'
        )
        
        # Set table properties
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)  # Make rows taller
        
        # Add a note about significance
        ax6.text(0.01, 0.01, "* p < 0.05", transform=ax6.transAxes, fontsize=9, fontstyle='italic')
        
        # Set title
        ax6.set_title('Summary Statistics', fontsize=14, pad=20)
    
    # Add an overall title to the figure
    fig.suptitle('Pore Symmetry Analysis: Toxin vs Control', fontsize=16, y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, 'pore_symmetry_summary.png'), dpi=150)
    plt.savefig(os.path.join(output_dir, 'pore_symmetry_summary.svg'), format='svg')
    plt.close()
    
    print(f"Summary figure saved to {os.path.join(output_dir, 'pore_symmetry_summary.png')}")

def save_statistics_csv(run_stats_df, system_stats_df, output_dir):
    """
    Save statistics to CSV files for further analysis.
    
    Args:
        run_stats_df: DataFrame with statistics per run
        system_stats_df: DataFrame with statistics per system type
        output_dir: Directory to save CSV files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save run-level statistics
    run_stats_df.to_csv(os.path.join(output_dir, 'pore_symmetry_run_stats.csv'), index=False)
    
    # Save system-level statistics
    system_stats_df.to_csv(os.path.join(output_dir, 'pore_symmetry_system_stats.csv'), index=False)
    
    print(f"Statistics saved to CSV files in {output_dir}")

def generate_html_report(run_stats_df, system_stats_df, output_dir):
    """
    Generate a simple HTML report with the results.
    
    Args:
        run_stats_df: DataFrame with statistics per run
        system_stats_df: DataFrame with statistics per system type
        output_dir: Directory to save the report
    """
    os.makedirs(output_dir, exist_ok=True)
    
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Pore Symmetry Analysis Report</title>
        <style>
            body { font-family: Arial, sans-serif; line-height: 1.6; margin: 20px; }
            h1, h2, h3 { color: #2c3e50; }
            table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
            th, td { text-align: left; padding: 8px; border: 1px solid #ddd; }
            th { background-color: #f2f2f2; }
            tr:nth-child(even) { background-color: #f9f9f9; }
            .significant { background-color: #d4edda; }
            .image-container { margin: 20px 0; text-align: center; }
            .image-container img { max-width: 100%; height: auto; border: 1px solid #ddd; }
            .summary { background-color: #e9f7fe; padding: 15px; border-radius: 5px; margin: 20px 0; }
        </style>
    </head>
    <body>
        <h1>Pore Symmetry Analysis Report</h1>
        
        <div class="summary">
            <h2>Key Findings</h2>
    """
    
    # Add key differences between toxin and control
    toxin_data = system_stats_df[system_stats_df['system_type'] == 'Toxin'].iloc[0] if 'Toxin' in system_stats_df['system_type'].values else None
    control_data = system_stats_df[system_stats_df['system_type'] == 'Control'].iloc[0] if 'Control' in system_stats_df['system_type'].values else None
    
    if toxin_data is not None and control_data is not None:
        # AC-BD difference
        ac_bd_diff_toxin = toxin_data.get('ac_bd_diff_mean_mean', None)
        ac_bd_diff_control = control_data.get('ac_bd_diff_mean_mean', None)
        
        if ac_bd_diff_toxin is not None and ac_bd_diff_control is not None:
            diff = ac_bd_diff_toxin - ac_bd_diff_control
            html_content += f"<p><strong>Pore Asymmetry (AC-BD):</strong> Toxin systems show "
            if ac_bd_diff_toxin > ac_bd_diff_control:
                html_content += f"greater asymmetry with AC-BD = {ac_bd_diff_toxin:.3f} Å compared to {ac_bd_diff_control:.3f} Å in control systems.</p>"
            else:
                html_content += f"lesser asymmetry with AC-BD = {ac_bd_diff_toxin:.3f} Å compared to {ac_bd_diff_control:.3f} Å in control systems.</p>"
        
        # Ellipticity
        ellipticity_toxin = toxin_data.get('ellipticity_mean_mean', None)
        ellipticity_control = control_data.get('ellipticity_mean_mean', None)
        
        if ellipticity_toxin is not None and ellipticity_control is not None:
            html_content += f"<p><strong>Pore Ellipticity:</strong> Toxin systems show "
            if ellipticity_toxin > ellipticity_control:
                html_content += f"more elliptical pores ({ellipticity_toxin:.3f}) compared to control systems ({ellipticity_control:.3f}).</p>"
            else:
                html_content += f"less elliptical pores ({ellipticity_toxin:.3f}) compared to control systems ({ellipticity_control:.3f}).</p>"
        
        # AC dominance
        ac_dom_toxin = toxin_data.get('pct_ac_dominant_mean', None)
        ac_dom_control = control_data.get('pct_ac_dominant_mean', None)
        
        if ac_dom_toxin is not None and ac_dom_control is not None:
            html_content += f"<p><strong>AC Axis Dominance:</strong> In toxin systems, the AC distance is greater than BD in {ac_dom_toxin:.1f}% of frames, "
            html_content += f"compared to {ac_dom_control:.1f}% in control systems.</p>"
    
    html_content += """
        </div>
        
        <h2>Summary Figures</h2>
        
        <div class="image-container">
            <h3>Overall Summary</h3>
            <img src="pore_symmetry_summary.png" alt="Pore Symmetry Summary">
            <p>Comprehensive comparison of pore symmetry metrics between toxin and control systems.</p>
        </div>
        
        <div class="image-container">
            <h3>Asymmetry Distribution</h3>
            <img src="ac_bd_diff_violin.png" alt="AC-BD Difference Violin Plot">
            <p>Distribution of pore asymmetry (AC-BD difference) in toxin vs control systems.</p>
        </div>
        
        <div class="image-container">
            <h3>Metric Distributions</h3>
            <img src="symmetry_distributions.png" alt="Symmetry Metrics Distributions">
            <p>Distribution of key symmetry metrics comparing toxin and control systems.</p>
        </div>
        
        <h2>System Statistics</h2>
        <table>
            <tr>
                <th>System Type</th>
                <th>AC Distance (Å)</th>
                <th>BD Distance (Å)</th>
                <th>AC-BD Difference (Å)</th>
                <th>Ellipticity</th>
                <th>% Frames AC > BD</th>
            </tr>
    """
    
    # Add rows for each system type
    for _, row in system_stats_df.iterrows():
        system_type = row['system_type']
        
        # Get relevant values with appropriate fallbacks
        ac_mean = row.get('A_C_Distance_Filtered_mean_mean', row.get('A_C_Distance_Filtered_mean', 'N/A'))
        bd_mean = row.get('B_D_Distance_Filtered_mean_mean', row.get('B_D_Distance_Filtered_mean', 'N/A'))
        diff_mean = row.get('ac_bd_diff_mean_mean', row.get('ac_bd_diff_mean', 'N/A'))
        ellip_mean = row.get('ellipticity_mean_mean', row.get('ellipticity_mean', 'N/A'))
        ac_dom = row.get('pct_ac_dominant_mean', 'N/A')
        
        # Format values
        if isinstance(ac_mean, (int, float)): ac_mean = f"{ac_mean:.3f}"
        if isinstance(bd_mean, (int, float)): bd_mean = f"{bd_mean:.3f}"
        if isinstance(diff_mean, (int, float)): diff_mean = f"{diff_mean:.3f}"
        if isinstance(ellip_mean, (int, float)): ellip_mean = f"{ellip_mean:.3f}"
        if isinstance(ac_dom, (int, float)): ac_dom = f"{ac_dom:.1f}%"
        
        html_content += f"""
            <tr>
                <td><strong>{system_type}</strong></td>
                <td>{ac_mean}</td>
                <td>{bd_mean}</td>
                <td>{diff_mean}</td>
                <td>{ellip_mean}</td>
                <td>{ac_dom}</td>
            </tr>
        """
    
    html_content += """
        </table>
        
        <h2>Run Statistics</h2>
        <table>
            <tr>
                <th>Run</th>
                <th>System Type</th>
                <th>AC Distance (Å)</th>
                <th>BD Distance (Å)</th>
                <th>AC-BD Difference (Å)</th>
                <th>Ellipticity</th>
                <th>% Frames AC > BD</th>
            </tr>
    """
    
    # Add rows for each run
    for _, row in run_stats_df.iterrows():
        run_name = row['run_name']
        system_type = row['system_type']
        
        # Get relevant values with appropriate fallbacks
        ac_mean = row.get('A_C_Distance_Filtered_mean', 'N/A')
        bd_mean = row.get('B_D_Distance_Filtered_mean', 'N/A')
        diff_mean = row.get('ac_bd_diff_mean', 'N/A')
        ellip_mean = row.get('ellipticity_mean', 'N/A')
        ac_dom = row.get('pct_ac_dominant', 'N/A')
        
        # Format values
        if isinstance(ac_mean, (int, float)): ac_mean = f"{ac_mean:.3f}"
        if isinstance(bd_mean, (int, float)): bd_mean = f"{bd_mean:.3f}"
        if isinstance(diff_mean, (int, float)): diff_mean = f"{diff_mean:.3f}"
        if isinstance(ellip_mean, (int, float)): ellip_mean = f"{ellip_mean:.3f}"
        if isinstance(ac_dom, (int, float)): ac_dom = f"{ac_dom:.1f}%"
        
        html_content += f"""
            <tr>
                <td>{run_name}</td>
                <td>{system_type}</td>
                <td>{ac_mean}</td>
                <td>{bd_mean}</td>
                <td>{diff_mean}</td>
                <td>{ellip_mean}</td>
                <td>{ac_dom}</td>
            </tr>
        """
    
    html_content += """
        </table>
        
        <h2>Individual Time Series</h2>
        <p>Time series plots showing G-G distances and asymmetry for each run are available in the output directory.</p>
        
        <hr>
        <p><em>Report generated at: """ + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S') + """</em></p>
    </body>
    </html>
    """
    
    # Write HTML to file
    with open(os.path.join(output_dir, 'pore_symmetry_report.html'), 'w') as f:
        f.write(html_content)
    
    print(f"HTML report saved to {os.path.join(output_dir, 'pore_symmetry_report.html')}")

def main():
    parser = argparse.ArgumentParser(description='Analyze pore symmetry in K+ channel simulations.')
    parser.add_argument('--toxin_dir', default='/home/labs/bmeitan/karbati/rCs1/toxin/toxin',
                        help='Directory containing toxin system data')
    parser.add_argument('--control_dir', default='/home/labs/bmeitan/karbati/Cs1_AF3v/control/control',
                        help='Directory containing control system data')
    parser.add_argument('--output_dir', default='pore_symmetry_analysis',
                        help='Directory for saving output files')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find G-G distance files
    base_dirs = {'toxin': args.toxin_dir, 'control': args.control_dir}
    file_dict = find_gg_distance_files(base_dirs)
    
    # Check if we found any files
    if not file_dict['toxin'] and not file_dict['control']:
        print("No G-G distance files found. Check the specified directories.")
        return
    
    # Process the files
    print("Processing G-G distance files...")
    combined_df = process_gg_files(file_dict)
    
    # Calculate statistics
    print("Calculating statistics...")
    run_stats_df, system_stats_df = calculate_statistics(combined_df)
    
    # Create plots
    print("Creating plots...")
    plot_symmetry_time_series(combined_df, os.path.join(args.output_dir, 'time_series'))
    plot_symmetry_distributions(combined_df, args.output_dir)
    create_summary_figure(run_stats_df, system_stats_df, args.output_dir)
    
    # Save statistics
    save_statistics_csv(run_stats_df, system_stats_df, args.output_dir)
    
    # Generate HTML report
    generate_html_report(run_stats_df, system_stats_df, args.output_dir)
    
    print(f"Analysis complete. Results saved to {args.output_dir}")

if __name__ == '__main__':
    main()
