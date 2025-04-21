#!/usr/bin/env python3
"""
Smoke test for PoreDomainAnalysis on full trajectory,
with automated cleanup and real‑time log streaming.
"""

import subprocess
import os
import sys
import shutil

def clean_data_dir(data_dir):
    """
    Remove all files and directories in data_dir except the two input files.
    """
    keep = {'MD_Aligned.dcd', 'step5_input.psf'}
    for entry in os.listdir(data_dir):
        path = os.path.join(data_dir, entry)
        if entry in keep:
            continue
        try:
            if os.path.isdir(path) and not os.path.islink(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
        except Exception as e:
            print(f"Warning: could not remove {path}: {e}", file=sys.stderr)

def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(repo_root, 'tests', 'data')

    # 1. Clean up old outputs
    print(f"Cleaning previous outputs in {data_dir} …")
    clean_data_dir(data_dir)

    # 2. Build and run the analysis command
    cmd = [
        sys.executable, '-m', 'pore_analysis.main',
        '--folder', data_dir,
        '--no-report'
    ]
    print("Running sample test with full trajectory:")
    print("  " + " ".join(cmd))
    try:
        # Stream stdout/stderr directly so you see progress live
        subprocess.check_call(cmd)
        print("\n✅ Sample test passed.")
    except subprocess.CalledProcessError as e:
        print("\n❌ Sample test FAILED", file=sys.stderr)
        sys.exit(e.returncode)

if __name__ == "__main__":
    main()
