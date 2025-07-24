#!/bin/bash
# run_pipeline.sh (Corrected Version)

set -e # Exit immediately if a command fails

# --- 1. SETUP & CONFIGURATION ---
echo "--- Installing Dependencies ---"
pip install -r requirements.txt --quiet

echo "--- Configuring Git ---"
git config --global user.name "Kaggle CI Bot"
git config --global user.email "ci-runner@kaggle.com"

echo "--- Setting up DVC Credentials ---"
echo "${GDRIVE_CREDENTIALS_DATA_B64}" | base64 -d > gdrive-credentials.json

# --- 2. RUN DVC PIPELINE ---
echo "--- Pulling DVC data ---"
dvc pull --force

echo "--- Reproducing DVC pipeline ---"
dvc repro

# --- 3. PUSH RESULTS ---
echo "--- Updating DVC tracking for mlruns ---"
# THIS IS THE CRITICAL FIX:
# Explicitly tell DVC to update the pointer for the mlruns directory
# based on the new contents created by 'dvc repro'.
dvc add mlruns

echo "--- Pushing DVC artifacts to remote storage ---"
# Now that mlruns.dvc is updated, this command knows to upload the new data.
dvc push

echo "--- Committing results back to Git ---"
# The 'git add' command now correctly stages the updated mlruns.dvc file
git add .
git commit -m "CI: Automated run, updated mlruns" || echo "No new changes to commit."
git push

echo "--- Workflow Finished Successfully ---"