#!/usr/bin/env python3
"""
Test script for the enhanced report generator.

This script generates test data and runs the enhanced report generator on it.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Add parent directory to path so we can import the enhanced_report_generator package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from generate_test_data import create_test_dataset

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def run_test(output_dir="test_report", clean=True):
    """
    Run a test of the enhanced report generator.
    
    Args:
        output_dir: Directory to save the reports
        clean: Whether to remove existing test data and output directory
    """
    logger.info("Starting test of enhanced report generator")
    
    # Define paths
    test_data_dir = Path("test_data")
    output_dir = Path(output_dir)
    
    # Clean up existing directories if requested
    if clean:
        import shutil
        if test_data_dir.exists():
            logger.info(f"Removing existing test data directory: {test_data_dir}")
            shutil.rmtree(test_data_dir)
        if output_dir.exists():
            logger.info(f"Removing existing output directory: {output_dir}")
            shutil.rmtree(output_dir)
    
    # Create test data
    logger.info("Generating test data...")
    create_test_dataset(test_data_dir, num_control=3, num_toxin=3)
    
    # Run Stage 1 (Overview)
    logger.info("Running Stage 1: Overview report generation...")
    try:
        cmd = [
            sys.executable,
            "-m", "enhanced_report_generator.main",
            "--sim_roots", str(test_data_dir / "control_systems"), str(test_data_dir / "toxin_systems"),
            "--output_dir", str(output_dir),
            "--title", "Test Analysis",
            "--stage", "overview",
            "--log_level", "INFO"
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        logger.info("Stage 1 completed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Stage 1 failed with error: {e}")
        return False
    
    # Verify overview report was created
    overview_html = output_dir / "test_analysis_system_overview.html"
    if not overview_html.exists():
        logger.error(f"Overview report not found: {overview_html}")
        return False
    logger.info(f"Overview report generated: {overview_html}")
    
    # Run Stage 2 (Detailed)
    logger.info("Running Stage 2: Detailed report generation...")
    try:
        cmd = [
            sys.executable,
            "-m", "enhanced_report_generator.main",
            "--output_dir", str(output_dir),
            "--title", "Test Analysis",
            "--stage", "detailed",
            "--log_level", "INFO"
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        logger.info("Stage 2 completed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Stage 2 failed with error: {e}")
        return False
    
    # Verify detailed report was created
    detailed_html = output_dir / "test_analysis_detailed_report.html"
    if not detailed_html.exists():
        logger.error(f"Detailed report not found: {detailed_html}")
        return False
    logger.info(f"Detailed report generated: {detailed_html}")
    
    # Verify plot directory was created and contains plots
    plots_dir = output_dir / "plots"
    if not plots_dir.exists() or not any(plots_dir.iterdir()):
        logger.warning(f"No plots found in {plots_dir}")
    else:
        num_plots = len(list(plots_dir.glob("*.png")))
        logger.info(f"Found {num_plots} plots in {plots_dir}")
    
    logger.info("Test completed successfully!")
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the enhanced report generator")
    parser.add_argument(
        "--output_dir", type=str, default="test_report",
        help="Directory to save the reports"
    )
    parser.add_argument(
        "--no-clean", action="store_true",
        help="Don't remove existing test data and output directory"
    )
    
    args = parser.parse_args()
    
    success = run_test(output_dir=args.output_dir, clean=not args.no_clean)
    sys.exit(0 if success else 1)