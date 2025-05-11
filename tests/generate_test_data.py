#!/usr/bin/env python3
"""
Test data generator for the enhanced report generator.

This script creates a directory structure with dummy analysis_registry.db files
that can be used to test the enhanced_report_generator package.
"""

import os
import shutil
import sqlite3
import random
import numpy as np
from pathlib import Path

# Base metrics that will be generated for each test system
BASE_METRICS = [
    # DW Gate metrics
    ("DW_PROA_Open_Fraction", "dw_gate_analysis", "%"),
    ("DW_PROB_Open_Fraction", "dw_gate_analysis", "%"),
    ("DW_PROC_Open_Fraction", "dw_gate_analysis", "%"),
    ("DW_PROD_Open_Fraction", "dw_gate_analysis", "%"),
    ("DW_PROA_MeanOpenTime", "dw_gate_analysis", "ns"),
    ("DW_PROB_MeanOpenTime", "dw_gate_analysis", "ns"),
    ("DW_PROC_MeanOpenTime", "dw_gate_analysis", "ns"),
    ("DW_PROD_MeanOpenTime", "dw_gate_analysis", "ns"),
    
    # Ion pathway metrics
    ("Ion_AvgOcc_S0", "ion_analysis", "count"),
    ("Ion_AvgOcc_S1", "ion_analysis", "count"),
    ("Ion_AvgOcc_S2", "ion_analysis", "count"),
    ("Ion_AvgOcc_S3", "ion_analysis", "count"),
    ("Ion_AvgOcc_S4", "ion_analysis", "count"),
    ("Ion_HMM_ConductionEvents_Total", "ion_analysis", "count"),
    ("Ion_HMM_ConductionRate", "ion_analysis", "events/ns"),
    
    # Tyrosine metrics
    ("Tyr_HMM_Population_Rotamer1", "tyrosine_analysis", "%"),
    ("Tyr_HMM_Population_Rotamer2", "tyrosine_analysis", "%"),
    ("Tyr_HMM_Transitions", "tyrosine_analysis", "count"),
    ("Tyr_HBond_Occupancy", "tyrosine_analysis", "%"),
    
    # Center of mass metrics
    ("COM_Mean_Filter", "core_analysis", "Å"),
    ("COM_Mean_SelectivityFilter", "core_analysis", "Å"),
    ("COM_Mean_CentralCavity", "core_analysis", "Å"),
]


def create_test_db(db_path, run_name, is_control=False, frames=1000, seed=None):
    """
    Create a test analysis_registry.db file with dummy data.
    
    Args:
        db_path: Path to the database file to create
        run_name: Run name for metadata
        is_control: Whether this is a control (toxin-free) system
        frames: Number of frames in the trajectory
        seed: Random seed for reproducibility
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Create database schema
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create metadata table
    cursor.execute("""
    CREATE TABLE simulation_metadata (
        key TEXT PRIMARY KEY,
        value TEXT
    )
    """)
    
    # Create metrics table
    cursor.execute("""
    CREATE TABLE metrics (
        metric_name TEXT,
        value REAL,
        units TEXT,
        module_name TEXT
    )
    """)
    
    # Insert metadata
    system_name = f"{'control' if is_control else 'toxin'}_system_{run_name}"
    metadata = [
        ('run_name', run_name),
        ('system_name', system_name),
        ('psf_file', f"/path/to/{system_name}.psf"),
        ('dcd_file', f"/path/to/{system_name}.dcd"),
        ('analysis_start_frame', '0'),
        ('analysis_end_frame', str(frames)),
        ('trajectory_total_frames', str(frames)),
        ('is_control_system', str(is_control))
    ]
    
    cursor.executemany("INSERT INTO simulation_metadata VALUES (?, ?)", metadata)
    
    # Insert metrics with biased values based on system type
    metrics_data = []
    
    # Base difference between toxin and control systems
    # We'll make the control and toxin systems have different values to ensure
    # the comparisons will show significant differences
    for metric_name, module_name, units in BASE_METRICS:
        # Generate base value
        base_value = random.uniform(0.1, 100.0)
        
        # Add bias based on system type
        if is_control:
            # Control systems
            if "Open_Fraction" in metric_name:
                # Lower open fraction in control
                value = base_value * 0.7
            elif "Ion_HMM_Conduction" in metric_name:
                # Higher conduction in control
                value = base_value * 1.5
            elif "Tyr_HMM_Population_Rotamer1" in metric_name:
                # Different rotamer populations
                value = base_value * 1.3
            else:
                # Small random variation
                value = base_value * random.uniform(0.9, 1.1)
        else:
            # Toxin systems
            if "Open_Fraction" in metric_name:
                # Higher open fraction with toxin
                value = base_value * 1.3
            elif "Ion_HMM_Conduction" in metric_name:
                # Lower conduction with toxin
                value = base_value * 0.6
            elif "Tyr_HMM_Population_Rotamer1" in metric_name:
                # Different rotamer populations
                value = base_value * 0.7
            else:
                # Small random variation
                value = base_value * random.uniform(0.9, 1.1)
        
        # Round to reasonable precision
        value = round(value, 3)
        
        # Add some randomness for variability between runs
        if not is_control:
            # Add more variability to toxin runs for realistic data
            noise = random.uniform(-value * 0.15, value * 0.15)
            value += noise
        
        metrics_data.append((metric_name, value, units, module_name))
    
    cursor.executemany("INSERT INTO metrics VALUES (?, ?, ?, ?)", metrics_data)
    
    conn.commit()
    conn.close()


def create_test_dataset(base_dir, num_control=3, num_toxin=3):
    """
    Create a test dataset with control and toxin systems.
    
    Args:
        base_dir: Base directory for the test dataset
        num_control: Number of control systems to create
        num_toxin: Number of toxin systems to create
    """
    # Create base directories
    base_path = Path(base_dir)
    control_dir = base_path / "control_systems"
    toxin_dir = base_path / "toxin_systems"
    
    control_dir.mkdir(parents=True, exist_ok=True)
    toxin_dir.mkdir(parents=True, exist_ok=True)
    
    # Create control systems
    for i in range(1, num_control + 1):
        run_name = f"run{i}_control"
        run_dir = control_dir / run_name
        run_dir.mkdir(exist_ok=True)
        
        db_path = run_dir / "analysis_registry.db"
        create_test_db(db_path, run_name, is_control=True, frames=1000 + i * 100, seed=i)
        print(f"Created control system: {run_name}")
    
    # Create toxin systems
    for i in range(1, num_toxin + 1):
        run_name = f"run{i}_toxin"
        run_dir = toxin_dir / run_name
        run_dir.mkdir(exist_ok=True)
        
        db_path = run_dir / "analysis_registry.db"
        create_test_db(db_path, run_name, is_control=False, frames=1000 + i * 100, seed=i + 100)
        print(f"Created toxin system: {run_name}")
    
    print(f"Test dataset created in {base_dir}")
    print(f"- {num_control} control systems")
    print(f"- {num_toxin} toxin systems")


if __name__ == "__main__":
    # Default test dataset location
    test_data_dir = Path("test_data")
    
    # Remove existing test data directory if it exists
    if test_data_dir.exists():
        shutil.rmtree(test_data_dir)
    
    # Create new test dataset
    create_test_dataset(test_data_dir, num_control=3, num_toxin=3)
    
    print("\nTo run the enhanced report generator on this test dataset:")
    print(f"python -m enhanced_report_generator.main --sim_roots {test_data_dir}/control_systems {test_data_dir}/toxin_systems --output_dir ./test_report --title 'Test Analysis'")