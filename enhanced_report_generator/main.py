"""
Main orchestrator module for the enhanced report generator.

This module provides the entry point for running the report generation process,
handling command-line arguments, and orchestrating the pipeline stages.
"""

import argparse
import logging
import datetime
import sys
from pathlib import Path

from .core.data_models import ReportSection
from .components.data_aggregator import DataAggregator
from .components.data_access import DataRepository
from .components.analysis_selection import AnalysisFilter, SectionBuilder
from .components.ai_insights import AIInsightGenerator
from .components.visualization import VisualizationEngine
from .components.rendering import ReportLayoutRenderer

# Define paths
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = Path.cwd() / "enhanced_report_output"
DEFAULT_TEMPLATES_DIR = BASE_DIR / "templates"
DEFAULT_CONFIG_DIR = BASE_DIR / "config"
DEFAULT_GLOSSARY_PATH = DEFAULT_CONFIG_DIR / "glossary_mapping.csv"
DEFAULT_ASSETS_DIR = BASE_DIR / "assets"
DEFAULT_AGGREGATED_DB_NAME = "enhanced_cross_analysis.db"


def setup_logging(log_level: str = "INFO", log_file: str = "enhanced_report_generator.log"):
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level (INFO, DEBUG, etc.)
        log_file: Path to the log file
    """
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode="a")
        ]
    )
    
    # Reduce verbosity for matplotlib and other libraries
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)


def ensure_essential_files():
    """Ensure that essential files exist or create them with defaults."""
    # Ensure glossary file exists
    if not DEFAULT_GLOSSARY_PATH.exists():
        DEFAULT_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(DEFAULT_GLOSSARY_PATH, "w", newline="") as f:
            f.write("prefix,section_title,section_description,metric_description_template\n")
            f.write("Overview,System Overview,\"High-level summary of analyzed systems.\",\"Parameter {metric_name}: {value_toxin} {units}\"\n")
            f.write("DW_,DW Gate Analysis,\"Analysis of the DW gate dynamics and conformations.\",\"{metric_name}: {value_toxin} vs {value_control} {units} (p={p_value})\"\n")
            f.write("Ion_,Ion Pathway Analysis,\"Analysis of ion conduction and occupancy in the channel.\",\"{metric_name}: {value_toxin} vs {value_control} {units} (p={p_value})\"\n")
        logging.info(f"Created default glossary at {DEFAULT_GLOSSARY_PATH}")
    
    # Ensure templates directory exists
    DEFAULT_TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Ensure assets directory exists
    DEFAULT_ASSETS_DIR.mkdir(parents=True, exist_ok=True)


def run_stage1_pipeline(output_dir: Path, aggregated_db_name: str, report_title: str, toxin_dir: Path, control_dir: Path, force_aggregation: bool = False, generate_html: bool = True):
    """
    Run Stage 1 of the pipeline: Data Aggregation and System Overview Report.

    Args:
        output_dir: Directory to save outputs
        aggregated_db_name: Name of the SQLite database file
        report_title: Title for the report
        toxin_dir: Directory containing toxin simulation data
        control_dir: Directory containing control simulation data
        force_aggregation: Whether to force re-aggregation even if DB exists
        generate_html: Whether to generate the HTML report (default: True)

    Returns:
        dict: Status report with information about the pipeline execution
    """
    logger = logging.getLogger(__name__)
    logger.info("--- Starting Stage 1: Data Aggregation & System Overview Report ---")

    status_report = {
        "success": False,
        "errors": [],
        "warnings": [],
        "output_files": {},
        "metrics_info": {
            "toxin_systems": 0,
            "control_systems": 0,
            "total_metrics": 0
        }
    }

    # Prepare output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregated_db_path = output_dir / aggregated_db_name
    plots_output_dir = output_dir / "plots"
    plots_output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure essential files exist
    ensure_essential_files()

    # --- 1. Data Aggregation ---
    aggregation_needed = force_aggregation or not aggregated_db_path.exists()

    if aggregation_needed:
        logger.info("Running Data Aggregation...")
        sim_dirs = []
        if toxin_dir and toxin_dir.exists():
            sim_dirs.append(toxin_dir)
            logger.info(f"Found toxin directory: {toxin_dir}")
        else:
            warning = f"Toxin directory not found or not specified: {toxin_dir}"
            logger.warning(warning)
            status_report["warnings"].append(warning)

        if control_dir and control_dir.exists():
            sim_dirs.append(control_dir)
            logger.info(f"Found control directory: {control_dir}")
        else:
            warning = f"Control directory not found or not specified: {control_dir}"
            logger.warning(warning)
            status_report["warnings"].append(warning)

        if not sim_dirs:
            error = "No valid simulation directories found. Cannot proceed with aggregation."
            logger.error(error)
            status_report["errors"].append(error)
            return status_report

        try:
            aggregator = DataAggregator(aggregated_db_path=aggregated_db_path)
            aggregator.run(root_dirs_for_discovery=sim_dirs, clear_existing=force_aggregation)

            if not aggregated_db_path.exists():
                error = f"Aggregation failed. Database not created at {aggregated_db_path}."
                logger.error(error)
                status_report["errors"].append(error)
                return status_report

        except Exception as e:
            error = f"Error during data aggregation: {str(e)}"
            logger.error(error)
            status_report["errors"].append(error)
            return status_report
    else:
        logger.info(f"Using existing database at {aggregated_db_path}")

    # --- 2. Generate System Overview ---
    try:
        # Initialize components
        data_repo = DataRepository(db_path=aggregated_db_path)

        # Load run metadata and generate overview metrics
        run_meta = data_repo.load_run_metadata()
        overview_metrics = data_repo.load_metrics_for_stage1_overview()

        # Update status report with metrics info
        status_report["metrics_info"]["toxin_systems"] = len(run_meta.toxin_run_ids)
        status_report["metrics_info"]["control_systems"] = len(run_meta.control_run_ids)

        # Check for data completeness
        if not run_meta.toxin_run_ids:
            warning = "No toxin-bound systems found in the database. Comparative analysis will not be possible."
            logger.warning(warning)
            status_report["warnings"].append(warning)

        if not run_meta.control_run_ids:
            warning = "No toxin-free (control) systems found in the database. Comparative analysis will not be possible."
            logger.warning(warning)
            status_report["warnings"].append(warning)

        # Generate HTML report if requested
        if generate_html:
            ai_generator = AIInsightGenerator(glossary_csv_path=DEFAULT_GLOSSARY_PATH)
            report_renderer = ReportLayoutRenderer(
                templates_dir=DEFAULT_TEMPLATES_DIR,
                output_dir=output_dir,
                assets_dir=DEFAULT_ASSETS_DIR
            )

            # Create overview section
            system_overview = ReportSection(
                title="System Overview",
                description="This section provides a high-level summary of the simulation systems included in this comparative analysis.",
                metrics=overview_metrics,
                plots=[],
                ai_interpretation=ai_generator.generate_section_insight(
                    ReportSection(title="System Overview", metrics=overview_metrics),
                    run_meta
                ) if overview_metrics else "No system data available for overview analysis."
            )

            # Render HTML report
            output_filename = f"{report_title.lower().replace(' ', '_')}_system_overview.html"
            html_path = report_renderer.render_html_report(
                report_title=f"{report_title} - System Overview",
                sections=[system_overview],
                run_metadata=run_meta,
                global_ai_summary=None,
                template_name="system_overview_template.html",
                output_filename=output_filename
            )

            logger.info(f"System overview report generated at {html_path}")
            status_report["output_files"]["overview_report"] = str(html_path)

        status_report["success"] = True

    except Exception as e:
        error = f"Error in Stage 1 pipeline: {str(e)}"
        logger.error(error)
        status_report["errors"].append(error)

    logger.info("--- Stage 1 Pipeline Finished ---")
    return status_report


def main():
    """Main entry point function."""
    parser = argparse.ArgumentParser(description="Enhanced Report Generator")

    parser.add_argument(
        "--toxin-dir", type=Path,
        help="Directory containing toxin simulation data."
    )
    parser.add_argument(
        "--control-dir", type=Path,
        help="Directory containing control simulation data."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Directory to save the generated reports and plots."
    )
    parser.add_argument(
        "--aggregated-db-name", type=str, default=DEFAULT_AGGREGATED_DB_NAME,
        help="Name of the SQLite database file within the output directory."
    )
    parser.add_argument(
        "--title", type=str, default="Cross-Simulation Analysis",
        help="Base title for the generated reports."
    )
    parser.add_argument(
        "--force-aggregate", action="store_true",
        help="Force re-aggregation of data even if database exists."
    )
    parser.add_argument(
        "--log-level", type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO',
        help="Set the logging level."
    )
    parser.add_argument(
        "--report-path", type=Path,
        help="Path to save a JSON report of the pipeline execution results."
    )
    parser.add_argument(
        "--tabbed-report", action="store_true",
        help="Generate a tabbed report with multiple analysis sections."
    )

    args = parser.parse_args()

    # Set up logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    logger.info(f"Enhanced Report Generator started")

    try:
        # Run pipeline
        if args.tabbed_report:
            logger.info("Generating tabbed report with multiple analysis sections")
            status = run_tabbed_report_pipeline(
                args.output_dir,
                args.aggregated_db_name,
                args.title,
                args.toxin_dir,
                args.control_dir,
                args.force_aggregate
            )
        else:
            logger.info("Running Stage 1 pipeline (system overview only)")
            status = run_stage1_pipeline(
                args.output_dir,
                args.aggregated_db_name,
                args.title,
                args.toxin_dir,
                args.control_dir,
                args.force_aggregate
            )

        # Display results to user
        if status["success"]:
            logger.info("Pipeline completed successfully.")
            if status["warnings"]:
                logger.warning("Completed with warnings:")
                for warning in status["warnings"]:
                    logger.warning(f"  - {warning}")

            # Print metrics summary
            logger.info("Data summary:")
            logger.info(f"  - Toxin-bound systems: {status['metrics_info']['toxin_systems']}")
            logger.info(f"  - Control systems: {status['metrics_info']['control_systems']}")

            # Print output files
            logger.info("Output files:")
            for file_type, file_path in status["output_files"].items():
                logger.info(f"  - {file_type}: {file_path}")

            # Human in the loop message
            if status["metrics_info"]["toxin_systems"] == 0 or status["metrics_info"]["control_systems"] == 0:
                logger.error("\nComparative analysis cannot proceed: Missing data for toxin and/or control systems.")
                logger.error("Human intervention required to correct the data categorization.")
                logger.error("Please review the generated report and provide guidance on how to proceed.")
            else:
                logger.info("\nReport generation completed successfully.")
                if not args.tabbed_report:
                    logger.info("For a more comprehensive analysis, run again with --tabbed-report flag to include additional sections.")
        else:
            logger.error("Pipeline failed.")
            for error in status["errors"]:
                logger.error(f"  - {error}")
            sys.exit(1)

        # Save status report if requested
        if args.report_path:
            import json
            with open(args.report_path, 'w') as f:
                json.dump(status, f, indent=2)
            logger.info(f"Pipeline status report saved to {args.report_path}")

    except Exception as e:
        logger.exception(f"Pipeline failed with error: {e}")
        sys.exit(1)


def run_tabbed_report_pipeline(output_dir: Path, aggregated_db_name: str, report_title: str, toxin_dir: Path, control_dir: Path, force_aggregation: bool = False):
    """
    Run the pipeline to generate a tabbed report with multiple analysis sections.

    Args:
        output_dir: Directory to save outputs
        aggregated_db_name: Name of the SQLite database file
        report_title: Title for the report
        toxin_dir: Directory containing toxin simulation data
        control_dir: Directory containing control simulation data
        force_aggregation: Whether to force re-aggregation even if DB exists

    Returns:
        dict: Status report with information about the pipeline execution
    """
    logger = logging.getLogger(__name__)
    logger.info("--- Starting Tabbed Report Pipeline ---")

    # First run Stage 1 to ensure we have the data aggregated
    status = run_stage1_pipeline(
        output_dir,
        aggregated_db_name,
        report_title,
        toxin_dir,
        control_dir,
        force_aggregation,
        generate_html=False  # Don't generate the HTML for Stage 1
    )

    if not status["success"]:
        logger.error("Stage 1 pipeline failed, cannot proceed with tabbed report.")
        return status

    # Check if we have both toxin and control data
    if status["metrics_info"]["toxin_systems"] == 0 or status["metrics_info"]["control_systems"] == 0:
        logger.warning("Missing toxin or control data, but continuing with tabbed report generation.")

    # Create status report for the tabbed report
    tabbed_status = {
        "success": False,
        "errors": [],
        "warnings": status["warnings"],
        "output_files": {},
        "metrics_info": status["metrics_info"]
    }

    try:
        # Get data from Stage 1
        aggregated_db_path = output_dir / aggregated_db_name

        # Initialize components
        data_repo = DataRepository(db_path=aggregated_db_path)
        ai_generator = AIInsightGenerator(glossary_csv_path=DEFAULT_GLOSSARY_PATH)
        report_renderer = ReportLayoutRenderer(
            templates_dir=DEFAULT_TEMPLATES_DIR,
            output_dir=output_dir,
            assets_dir=DEFAULT_ASSETS_DIR
        )

        # Load run metadata and overview metrics
        run_meta = data_repo.load_run_metadata()
        overview_metrics = data_repo.load_metrics_for_stage1_overview()

        # Create overview section
        system_overview = ReportSection(
            title="System Overview",
            description="This section provides a high-level summary of the simulation systems included in this comparative analysis.",
            metrics=overview_metrics,
            plots=[],
            ai_interpretation=ai_generator.generate_section_insight(
                ReportSection(title="System Overview", metrics=overview_metrics),
                run_meta
            ) if overview_metrics else "No system data available for overview analysis."
        )

        # Render tabbed HTML report
        output_filename = f"{report_title.lower().replace(' ', '_')}.html"
        html_path = report_renderer.render_tabbed_report(
            report_title=report_title,
            sections=[system_overview],
            run_metadata=run_meta,
            db_path=aggregated_db_path,
            output_filename=output_filename
        )

        logger.info(f"Tabbed report generated at {html_path}")
        tabbed_status["output_files"]["tabbed_report"] = str(html_path)
        tabbed_status["success"] = True

    except Exception as e:
        error = f"Error generating tabbed report: {str(e)}"
        logger.error(error)
        tabbed_status["errors"].append(error)

    logger.info("--- Tabbed Report Pipeline Finished ---")
    return tabbed_status


if __name__ == "__main__":
    main()