#!/bin/bash

# Base directory where the R1, R2, ... R10 folders are located
BASE_DIR="~/rCs1/toxin/toxin"
# Path to your analysis script
ANALYSIS_SCRIPT="main_analyzer.py" 
# Python interpreter command
PYTHON_CMD="python3"

echo "Starting batch analysis..."

# Loop through numbers 1 to 10
for i in {1..10}
do
  # Construct the full path to the current run folder
  RUN_FOLDER="${BASE_DIR}/R${i}"
  # Use eval to handle the tilde (~) expansion correctly
  EVAL_RUN_FOLDER=$(eval echo ${RUN_FOLDER}) 

  echo "----------------------------------"
  echo "Processing folder: ${EVAL_RUN_FOLDER}"
  echo "----------------------------------"

  # Check if the folder exists
  if [ -d "${EVAL_RUN_FOLDER}" ]; then
    # Run the analysis command
    # Using --force_rerun as requested
    ${PYTHON_CMD} ${ANALYSIS_SCRIPT} --folder "${EVAL_RUN_FOLDER}" --force_rerun
    
    # Check the exit status of the python script (optional)
    if [ $? -eq 0 ]; then
      echo "Successfully processed ${EVAL_RUN_FOLDER}"
    else
      echo "WARNING: Analysis script returned an error for ${EVAL_RUN_FOLDER}"
    fi
  else
    echo "WARNING: Folder ${EVAL_RUN_FOLDER} not found. Skipping."
  fi
  
  echo # Add a blank line for readability
done

echo "----------------------------------"
echo "Batch analysis finished."
echo "----------------------------------"
