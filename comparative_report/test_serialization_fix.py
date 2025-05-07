#!/usr/bin/env python3
"""
Test script to validate the numpy to Python type conversion in comparative analysis
"""

import os
import sys
import logging
from cross_analysis.comparative_analysis import ComparativeAnalysis

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_single_metric_comparison(meta_db_path, output_dir, metric_name):
    """Test comparison of a single metric with the serialization fix."""
    try:
        # Initialize the comparative analysis module
        analyzer = ComparativeAnalysis(meta_db_path, output_dir)
        
        # Connect to the database
        if not analyzer.connect():
            logger.error("Failed to connect to meta-database")
            return False
            
        # Compare a single metric
        logger.info(f"Testing comparison of metric: {metric_name}")
        result = analyzer.compare_metric(metric_name)
        
        # Log the result
        if result['status'] == 'success':
            logger.info(f"Successfully compared metric {metric_name}")
            logger.info(f"Result type: {type(result)}")
            logger.info(f"Toxin mean: {result['toxin_mean']} ({type(result['toxin_mean'])})")
            logger.info(f"Control mean: {result['control_mean']} ({type(result['control_mean'])})")
            logger.info(f"p-value: {result['p_value']} ({type(result['p_value'])})")
            logger.info(f"Is significant: {result['is_significant']} ({type(result['is_significant'])})")
            return True
        else:
            logger.error(f"Failed to compare metric {metric_name}: {result.get('message', 'Unknown error')}")
            return False
    except Exception as e:
        logger.error(f"Exception during test: {e}")
        return False
    finally:
        # Close the connection
        if 'analyzer' in locals() and analyzer:
            analyzer.close()

def main():
    """Main function."""
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <meta_db_path> <metric_name>")
        return 1
        
    meta_db_path = sys.argv[1]
    metric_name = sys.argv[2]
    output_dir = os.path.dirname(meta_db_path)
    
    # Test the metric comparison
    success = test_single_metric_comparison(meta_db_path, output_dir, metric_name)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())