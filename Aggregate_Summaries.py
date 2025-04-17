#!/usr/bin/env python3
"""
Aggregate and compare analysis_summary.json files from toxin and control systems.
Generates a compact JSON file suitable for sharing and further analysis.
"""

import os
import glob
import json
import numpy as np
import argparse
from scipy import stats
from datetime import datetime

def find_summary_files(base_dirs):
    """Find all analysis_summary.json files in the specified directories."""
    all_files = []
    for base_dir in base_dirs:
        pattern = os.path.join(base_dir, "**", "analysis_summary.json")
        files = glob.glob(pattern, recursive=True)
        all_files.extend(files)
    return all_files

def load_summary_data(summary_files):
    """Load all summary data from the specified files."""
    data = []
    for file_path in summary_files:
        try:
            with open(file_path, 'r') as f:
                summary = json.load(f)
                
            # Add file information
            summary['FilePath'] = file_path
            
            # Determine system type based on path
            if 'control' in file_path.lower():
                summary['SystemType'] = 'Control'
            elif 'toxin' in file_path.lower():
                summary['SystemType'] = 'Toxin'
            else:
                summary['SystemType'] = 'Unknown'
                
            data.append(summary)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return data

def calculate_group_statistics(summary_data):
    """Calculate statistics for toxin and control groups."""
    # Separate toxin and control data
    toxin_data = [item for item in summary_data if item['SystemType'] == 'Toxin']
    control_data = [item for item in summary_data if item['SystemType'] == 'Control']
    
    print(f"Found {len(toxin_data)} toxin systems and {len(control_data)} control systems")
    
    # Key metrics to analyze
    key_metrics = [
        # G-G Distance metrics
        'GG_AC_Mean_Filt', 'GG_AC_Std_Filt', 'GG_BD_Mean_Filt', 'GG_BD_Std_Filt',
        
        # COM metrics (only for toxin)
        'COM_Mean_Filt', 'COM_Std_Filt',
        
        # Ion metrics
        'Ion_Count',
        'Ion_AvgOcc_S0', 'Ion_AvgOcc_S1', 'Ion_AvgOcc_S2', 
        'Ion_AvgOcc_S3', 'Ion_AvgOcc_S4', 'Ion_AvgOcc_Cavity',
        
        # Water metrics
        'CavityWater_MeanOcc', 'CavityWater_StdOcc', 
        'CavityWater_AvgResidenceTime_ns', 'CavityWater_ExchangeRatePerNs'
    ]
    
    # Calculate statistics for each group
    stats = {
        'toxin': {},
        'control': {},
        'comparison': {},
        'metadata': {
            'toxin_count': len(toxin_data),
            'control_count': len(control_data),
            'timestamp': datetime.now().isoformat()
        }
    }
    
    # Helper function to safely get values from a list of dictionaries
    def safe_get_values(data_list, key):
        values = []
        for item in data_list:
            val = item.get(key)
            if val is not None and val != "null" and not isinstance(val, str):
                try:
                    values.append(float(val))
                except (ValueError, TypeError):
                    pass
        return np.array(values) if values else np.array([])
    
    # Helper function to calculate basic statistics
    def calc_basic_stats(values):
        if len(values) == 0:
            return {'count': 0, 'mean': None, 'std': None, 'min': None, 'max': None}
        
        return {
            'count': len(values),
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values))
        }
    
    # Calculate statistics for each metric
    for metric in key_metrics:
        toxin_values = safe_get_values(toxin_data, metric)
        control_values = safe_get_values(control_data, metric)
        
        # Store stats for toxin group
        stats['toxin'][metric] = calc_basic_stats(toxin_values)
        
        # Store stats for control group
        stats['control'][metric] = calc_basic_stats(control_values)
        
        # Calculate comparison statistics if both groups have data
        if len(toxin_values) > 0 and len(control_values) > 0:
            toxin_mean = np.mean(toxin_values)
            control_mean = np.mean(control_values)
            difference = toxin_mean - control_mean
            
            # Calculate percentage change relative to control
            pct_change = (difference / abs(control_mean)) * 100 if control_mean != 0 else None
            
            # Perform t-test if enough samples
            p_value = None
            t_stat = None
            significant = None
            
            if len(toxin_values) >= 2 and len(control_values) >= 2:
                try:
                    t_stat, p_value = stats.ttest_ind(toxin_values, control_values, equal_var=False)
                    significant = p_value < 0.05
                except Exception as e:
                    print(f"Error calculating t-test for {metric}: {e}")
            
            stats['comparison'][metric] = {
                'difference': float(difference) if difference is not None else None,
                'percent_change': float(pct_change) if pct_change is not None else None,
                't_statistic': float(t_stat) if t_stat is not None else None,
                'p_value': float(p_value) if p_value is not None else None,
                'significant': significant
            }
    
    # Add run names for reference
    stats['metadata']['toxin_runs'] = [item.get('RunName', 'Unknown') for item in toxin_data]
    stats['metadata']['control_runs'] = [item.get('RunName', 'Unknown') for item in control_data]
    
    return stats

def main():
    parser = argparse.ArgumentParser(description='Aggregate MD analysis summary files and produce concise comparison.')
    parser.add_argument('--toxin_dir', default='/home/labs/bmeitan/karbati/rCs1/toxin/toxin',
                        help='Directory containing toxin system data')
    parser.add_argument('--control_dir', default='/home/labs/bmeitan/karbati/Cs1_AF3v/control/control',
                        help='Directory containing control system data')
    parser.add_argument('--output', default='toxin_control_aggregate.json',
                        help='Output JSON file with aggregated data')
    
    args = parser.parse_args()
    
    # Find all summary files
    print("Finding summary files...")
    summary_files = find_summary_files([args.toxin_dir, args.control_dir])
    print(f"Found {len(summary_files)} summary files")
    
    if not summary_files:
        print("No summary files found. Check the specified directories.")
        return
    
    # Load summary data
    print("Loading summary data...")
    summary_data = load_summary_data(summary_files)
    print(f"Loaded data from {len(summary_data)} files")
    
    # Calculate statistics
    print("Calculating group statistics...")
    stats = calculate_group_statistics(summary_data)
    
    # Save result to JSON
    print(f"Saving aggregated data to {args.output}...")
    with open(args.output, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Aggregated data saved to {args.output}")
    print("Done!")

if __name__ == "__main__":
    main()
