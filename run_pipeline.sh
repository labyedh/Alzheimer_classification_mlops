#!/bin/bash
# run_pipeline.sh (Final, Robust Version)

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
dvc add mlruns

echo "--- Pushing DVC artifacts to remote storage ---"
dvc push

echo "--- Committing results back to Git ---"
git add .
# Use '|| true' to prevent script failure if there are no changes
git commit -m "CI: Automated run from Kaggle GPU" || true 

# --- THE FIX for the race condition ---
echo "--- Syncing with remote before pushing ---"
# Pull any changes that happened on the remote while this job was running.
# --rebase avoids a merge commit, keeping the history clean.
git pull --rebase

echo "--- Pushing results to Git ---"
git push

echo "--- Workflow Finished Successfully ---"