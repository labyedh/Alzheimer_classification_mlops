#!/bin/bash
# run_pipeline.sh

set -e # Exit immediately if a command fails

echo "--- Installing Dependencies ---"
pip install -r requirements.txt --quiet

echo "--- Configuring Git for Push ---"
git config --global user.name "Kaggle CI Runner"
git config --global user.email "ci-runner@kaggle.com"

# --- START OF FIX ---
# Switch the remote URL from the HTTPS (used for clone) to SSH (for push)
# This forces Git to use the SSH deploy key we set up.
GIT_USERNAME="labyedh"
GIT_REPO="Alzheimer_classification_mlops"
git remote set-url origin "git@github.com:${GIT_USERNAME}/${GIT_REPO}.git"
echo "Git remote URL is now set to SSH."
# --- END OF FIX ---

echo "--- Configuring DVC ---"
# The credentials file was created by the notebook before calling this script
dvc remote modify storage gdrive_use_service_account true
dvc remote modify storage gdrive_service_account_json_file_path gdrive-credentials.json

# --- RUN DVC PIPELINE ---
echo "--- Pulling DVC data ---"
dvc pull --force

echo "--- Reproducing DVC pipeline ---"
dvc repro

# --- PUSH RESULTS ---
echo "--- Pushing DVC artifacts to remote storage ---"
dvc push

echo "--- Committing results back to Git ---"
git add .
git commit -m "CI: Automated run from Kaggle GPU" || echo "No new changes to commit."
git push

echo "--- Workflow Finished Successfully ---"