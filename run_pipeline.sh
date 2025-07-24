#!/bin/bash
# run_pipeline.sh

set -e # Exit immediately if a command fails

# --- 1. SETUP & CONFIGURATION ---
echo "--- Installing Dependencies ---"
pip install -r requirements.txt --quiet

echo "--- Configuring Git ---"
# The remote 'origin' is already set correctly by 'git clone'
git config --global user.name "Kaggle CI Bot"
git config --global user.email "ci-runner@kaggle.com"

echo "--- Setting up DVC Credentials ---"
# The notebook created this file for us before calling the script
echo "${GDRIVE_CREDENTIALS_DATA_B64}" | base64 -d > gdrive-credentials.json

# --- 2. RUN DVC PIPELINE ---
echo "--- Pulling DVC data ---"
dvc pull --force

echo "--- Reproducing DVC pipeline ---"
dvc repro

# --- 3. PUSH RESULTS ---
echo "--- Pushing DVC artifacts to remote storage ---"
dvc push

echo "--- Committing results back to Git ---"
git add .
git commit -m "CI: Automated run from Kaggle GPU" || echo "No new changes to commit."
git push

echo "--- Workflow Finished Successfully ---"