import subprocess
import sys
from pathlib import Path
import pytest

def test_html_integration(tmp_path):
    # Skip if no test data present
    data_dir = Path(__file__).parent / "data"
    if not data_dir.exists():
        pytest.skip("No test data found in tests/data for integration test")

    # Copy data to a temp folder
    work_dir = tmp_path / "data"
    subprocess.run(["cp", "-r", str(data_dir), str(work_dir)], check=True)

    # Run the pipeline with HTML report
    cmd = [
        sys.executable, "-m", "pore_analysis.main",
        "--folder", str(work_dir),
        "--report"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    # Check that the HTML report file exists
    run_name = work_dir.name
    report_file = work_dir / f"{run_name}_analysis_report.html"
    assert report_file.exists(), f"Expected HTML report at {report_file}"

    content = report_file.read_text(encoding="utf-8")
    # Basic checks that the report loaded key sections
    assert "<h1>MD Analysis Report</h1>" in content
    assert "Pore Ions" in content
    assert "Inner Vestibule" in content
    assert "DW Gate" in content
