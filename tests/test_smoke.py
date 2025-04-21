import subprocess
import sys
import os

def test_smoke():
    """
    Smoke‑test PoreDomainAnalysis on the sample data.
    """
    repo_root = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(repo_root, 'tests', 'data')
    cmd = [
        sys.executable, '-m', 'pore_analysis.main',
        '--folder', data_dir,
        '--no-report'
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, (
        f"Smoke test failed (exit {result.returncode}):\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )
