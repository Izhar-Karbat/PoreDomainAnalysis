#!/bin/bash
# Example usage of the Cross-Analysis Suite

# Define directories
TOXIN_DIR="/home/labs/bmeitan/karbati/rCs1/toxin"
CONTROL_DIR="/home/labs/bmeitan/karbati/rCs1/control"
OUTPUT_DIR="./analysis_results"

# Basic usage - extract and analyze all metrics
echo "Running full analysis..."
python cross_analysis_main.py \
  --toxin-dir "$TOXIN_DIR" \
  --control-dir "$CONTROL_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --report-title "Comparison of Toxin-Bound and Control Ion Channels"

# Extract only (useful for initial setup or testing)
echo "Extracting metrics only..."
python cross_analysis_main.py \
  --toxin-dir "$TOXIN_DIR" \
  --control-dir "$CONTROL_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --extract-only

# Focus on specific metric categories
echo "Analyzing structure and ion metrics..."
python cross_analysis_main.py \
  --toxin-dir "$TOXIN_DIR" \
  --control-dir "$CONTROL_DIR" \
  --output-dir "$OUTPUT_DIR/focused_analysis" \
  --categories structure ion \
  --report-title "Structure and Ion Analysis"

# Using existing analysis without re-extraction
echo "Generating report from existing database..."
python cross_analysis_main.py \
  --toxin-dir "$TOXIN_DIR" \
  --control-dir "$CONTROL_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --skip-extraction \
  --report-title "Analysis Report (No Re-extraction)"

# With AI insights (if you have a Claude API key)
# echo "Running analysis with AI insights..."
# export CLAUDE_API_KEY="your-api-key-here"
# python cross_analysis_main.py \
#   --toxin-dir "$TOXIN_DIR" \
#   --control-dir "$CONTROL_DIR" \
#   --output-dir "$OUTPUT_DIR/with_ai" \
#   --enable-ai \
#   --report-title "Analysis with AI-Generated Insights"

echo "Examples complete. Check the output directories for results."