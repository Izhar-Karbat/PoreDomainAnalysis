#!/usr/bin/env python3
"""
Main script for the Cross-Analysis Suite.

This script ties together all components of the cross-analysis suite to enable
systematic comparisons between toxin-bound and control systems, with optional
AI-assisted insights.
"""

import os
import sys
import logging
import argparse
from datetime import datetime
import json

# Import all modules from the cross-analysis package
from cross_analysis.extract_metrics import MetricsExtractor
from cross_analysis.comparative_analysis import ComparativeAnalysis
from cross_analysis.ai_insights import AIInsightEngine
from cross_analysis.html_report import HTMLReportGenerator

# Configure logging
def setup_logging(log_dir: str, level=logging.INFO):
    """Set up logging configuration."""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"cross_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return log_file

# Parse command line arguments
def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Cross-Analysis Suite for comparing toxin-bound and control ion channel systems.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # Main directory arguments
    parser.add_argument('--toxin-dir', required=True, 
                      help="Directory containing toxin-bound systems")
    parser.add_argument('--control-dir', required=True,
                      help="Directory containing control systems")
    parser.add_argument('--output-dir', default='cross_analysis_results',
                      help="Directory for storing results and reports")
    
    # Database options
    parser.add_argument('--meta-db', default='cross_analysis.db',
                      help="Path to the meta-database file (will be created if it doesn't exist)")
    parser.add_argument('--reinit-db', action='store_true',
                      help="Reinitialize the meta-database (deletes existing data)")
    
    # Analysis options
    parser.add_argument('--extract-only', action='store_true',
                      help="Only extract metrics without running analysis")
    parser.add_argument('--skip-extraction', action='store_true',
                      help="Skip metrics extraction, use existing database")
    parser.add_argument('--categories', nargs='+',
                      help="Specific metric categories to analyze (space-separated)")
    
    # AI options
    parser.add_argument('--enable-ai', action='store_true',
                      help="Enable AI-assisted analysis (requires API key)")
    parser.add_argument('--api-key', 
                      help="API key for Claude (if not provided, will look for CLAUDE_API_KEY env var)")
    
    # Output options
    parser.add_argument('--report-title', default="Toxin vs Control: Cross-System Analysis",
                      help="Title for the HTML report")
    parser.add_argument('--no-report', action='store_true',
                      help="Skip generating HTML report")
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                      default='INFO', help="Set the logging level")
    
    return parser.parse_args()

def main():
    """Main function for the cross-analysis suite."""
    # Parse arguments
    args = parse_arguments()
    
    # Create output directory
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up logging
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    log_file = setup_logging(output_dir, log_level)
    logger = logging.getLogger(__name__)
    
    # Show start banner
    logger.info("=" * 80)
    logger.info("Starting Cross-Analysis Suite")
    logger.info("=" * 80)
    logger.info(f"Toxin directory: {args.toxin_dir}")
    logger.info(f"Control directory: {args.control_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Log file: {log_file}")
    
    # Initialize meta-database path
    if os.path.isabs(args.meta_db):
        meta_db_path = args.meta_db
    else:
        meta_db_path = os.path.join(output_dir, args.meta_db)
    
    # If reinit-db flag is set, delete existing database
    if args.reinit_db and os.path.exists(meta_db_path):
        logger.warning(f"Deleting existing meta-database at {meta_db_path}")
        os.remove(meta_db_path)
    
    # Step 1: Extract metrics from individual system databases
    if not args.skip_extraction:
        logger.info("Step 1: Extracting metrics from system databases")
        extractor = MetricsExtractor(args.toxin_dir, args.control_dir, meta_db_path)
        
        if not extractor.connect_meta_db():
            logger.error("Failed to connect to meta-database")
            return 1
        
        # Scan for systems
        systems = extractor.scan_systems()
        if not systems:
            logger.error("No systems found in the specified directories")
            extractor.close()
            return 1
        
        # Register systems in the database
        count = extractor.register_systems()
        logger.info(f"Registered {count} systems in the meta-database")
        
        # Extract metrics
        if args.categories:
            logger.info(f"Extracting metrics for categories: {', '.join(args.categories)}")
            total_metrics = extractor.extract_metrics(args.categories)
        else:
            logger.info("Extracting all metrics")
            total_metrics = extractor.extract_metrics()
        
        logger.info(f"Extracted {total_metrics} metrics from {len(systems)} systems")
        extractor.close()
        
        # If extract-only flag is set, stop here
        if args.extract_only:
            logger.info("Extraction-only mode. Skipping analysis.")
            return 0
    else:
        logger.info("Skipping metrics extraction as requested")
    
    # Step 2: Run comparative analysis
    logger.info("Step 2: Running comparative analysis")
    analyzer = ComparativeAnalysis(meta_db_path, output_dir)
    
    if not analyzer.connect():
        logger.error("Failed to connect to meta-database for analysis")
        return 1
    
    # Run batch comparisons
    if args.categories:
        logger.info(f"Running comparative analysis for categories: {', '.join(args.categories)}")
        results = analyzer.run_batch_comparisons(args.categories)
    else:
        logger.info("Running comparative analysis for all metrics")
        results = analyzer.run_batch_comparisons()
    
    # Log summary of results
    if results.get('status') == 'success':
        logger.info(f"Completed analysis of {results['total_metrics']} metrics")
        logger.info(f"Found {results['significant_differences']} significant differences")
    else:
        logger.error("Comparative analysis failed")
        analyzer.close()
        return 1
    
    # Step 3: Generate AI insights (if enabled)
    if args.enable_ai:
        logger.info("Step 3: Generating AI insights")
        
        # Get API key (from args or environment variable)
        api_key = args.api_key or os.environ.get('CLAUDE_API_KEY')
        if not api_key:
            logger.warning("No API key provided for AI analysis. Skipping this step.")
        else:
            ai_engine = AIInsightEngine(meta_db_path, api_key)
            
            if not ai_engine.connect():
                logger.error("Failed to connect to meta-database for AI analysis")
                analyzer.close()
                return 1
            
            # Generate comprehensive insights
            logger.info("Generating comprehensive insights")
            insights = ai_engine.analyze_comparison_data(args.categories)
            
            if insights.get('status') == 'success':
                logger.info(f"Generated comprehensive insights (ID: {insights['insight_id']})")
                
                # Generate insights for top significant metrics
                if results.get('status') == 'success' and results.get('results'):
                    significant_results = [r for r in results['results'] if r.get('is_significant', False)]
                    if significant_results:
                        top_metrics = significant_results[:min(3, len(significant_results))]
                        for metric in top_metrics:
                            metric_name = metric.get('metric_name')
                            if metric_name:
                                logger.info(f"Generating insights for metric: {metric_name}")
                                metric_insights = ai_engine.analyze_specific_metric(metric_name)
                                if metric_insights.get('status') == 'success':
                                    logger.info(f"Generated insights for {metric_name} (ID: {metric_insights['insight_id']})")
            else:
                logger.warning("Failed to generate AI insights")
            
            ai_engine.close()
    else:
        logger.info("Skipping AI insights generation (not enabled)")
    
    # Step 4: Generate HTML report
    if not args.no_report:
        logger.info("Step 4: Generating HTML report")
        report_gen = HTMLReportGenerator(meta_db_path, output_dir)
        
        if not report_gen.connect():
            logger.error("Failed to connect to meta-database for report generation")
            analyzer.close()
            return 1
        
        # Generate report
        report_path = report_gen.generate_report(args.report_title)
        if report_path:
            logger.info(f"Generated HTML report: {report_path}")
        else:
            logger.error("Failed to generate HTML report")
        
        report_gen.close()
    else:
        logger.info("Skipping HTML report generation as requested")
    
    # Close connections
    analyzer.close()
    
    # Show completion banner
    logger.info("=" * 80)
    logger.info("Cross-Analysis Suite completed successfully")
    logger.info("=" * 80)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())