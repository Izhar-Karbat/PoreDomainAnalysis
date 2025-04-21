from pathlib import Path
from pore_analysis.reporting.html import generate_html_report

def test_generate_html_report_error(tmp_path):
    # Prepare run directory without required summary fields
    run_dir = tmp_path / "run_err"
    run_dir.mkdir()

    # Minimal summary dict missing keys to force template error
    run_summary = {"RunName": "err_run"}

    # generate_html_report should catch the TemplateError and return None
    report_path = generate_html_report(str(run_dir), run_summary)
    assert report_path is None

    # A fallback error report should be created
    error_html = run_dir / "err_run_report_TEMPLATE_ERROR.html"
    assert error_html.exists(), "Error HTML file should exist after template failure"

    # Check it contains the "Template Error" header
    content = error_html.read_text(encoding="utf-8")
    assert "<h1>Template Error</h1>" in content
