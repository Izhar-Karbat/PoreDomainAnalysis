# tests/test_integration_ci_data.py
# (Renamed to distinguish from the tmp_path version)

import pytest
import subprocess
import sys
import os
from pathlib import Path
import sqlite3 # For DB check

# Define the path to the test data relative to the project root
# Assumes pytest is run from the project root directory
TEST_DATA_DIR = Path("tests/data")

@pytest.mark.skipif(not TEST_DATA_DIR.exists() or not (TEST_DATA_DIR / "step5_input.psf").exists() or not (TEST_DATA_DIR / "MD_Aligned.dcd").exists(), reason="Test data not found in tests/data/")
def test_main_py_with_ci_data():
    """
    Test running main.py end-to-end using data from the ./tests/data directory,
    similar to the CI workflow.
    """
    run_dir = TEST_DATA_DIR # Target the actual data directory

    # Clean up previous analysis results if they exist
    db_file = run_dir / "analysis_registry.db"
    report_file = run_dir / "data_analysis_report.html"
    log_file = run_dir / f"{run_dir.name}_analysis.log" # Adjust log filename if needed
    summary_file = run_dir / "analysis_summary.json"
    module_dirs = [d for d in run_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]

    if db_file.exists(): db_file.unlink()
    if report_file.exists(): report_file.unlink()
    if log_file.exists(): log_file.unlink()
    if summary_file.exists(): summary_file.unlink()
    for mod_dir in module_dirs:
        if mod_dir.name != 'data': # Avoid deleting the data itself!
            try:
                # Check if analysis output exists before attempting removal
                # Example: Check for a common output file or directory structure
                if (mod_dir / 'output.csv').exists() or list(mod_dir.glob('*.png')):
                     print(f"Removing old analysis outputs in {mod_dir}...")
                     # Implement safer cleanup if needed, e.g., remove specific files
                     # For now, just log - uncomment shutil if needed
                     # import shutil
                     # shutil.rmtree(mod_dir)
                else:
                     print(f"No apparent analysis output to clean in {mod_dir}.")
            except Exception as e:
                print(f"Warning: Could not clean module directory {mod_dir}: {e}")


    # Run the main script targeting the data directory
    # Add --reinit-db to ensure a clean database state for the test
    cmd = [
        sys.executable, "-m", "pore_analysis.main",
        "--folder", str(run_dir),
        "--reinit-db", # Ensure clean DB state for this test run
        "--report"     # Generate the report
        # Add --no-plots if you want to speed it up further in CI
        # "--no-plots"
    ]
    print(f"Running command: {' '.join(cmd)}")
    # Increase timeout if analysis takes longer
    result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=600) # 10 min timeout

    print("--- STDOUT ---")
    print(result.stdout)
    print("--- STDERR ---")
    print(result.stderr)

    # Assertions
    assert result.returncode == 0, f"main.py failed with stderr:\n{result.stderr}"
    assert db_file.exists(), "Database file was not created in tests/data/"
    assert report_file.exists(), "HTML report file was not created in tests/data/"

    # Check basic content of the report
    content = report_file.read_text(encoding="utf-8")
    assert ">MD Analysis Report</h1>" in content

    # Check DB status
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("SELECT value FROM simulation_metadata WHERE key = 'analysis_status'")
    status_row = cursor.fetchone()
    conn.close()
    assert status_row is not None and status_row[0] == 'success', "Analysis status in DB is not 'success'"
