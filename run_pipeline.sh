#!/bin/bash
# run_pipeline.sh

set -e # Exit immediately if a command fails

echo "--- Installing Dependencies ---"
pip install -r requirements.txt --quiet

echo "--- Configuring Git for Push ---"
git config --global user.name "Kaggle CI Runner"
git config --global user.email "ci-runner@kaggle.com"
# Switch to SSH for secure push
git remote set-url origin git@github.com:labyedh/Alzheimer_classification_mlops.git

echo "--- Configuring DVC ---"
dvc remote modify storage gdrive_use_service_account true
dvc remote modify storage gdrive_service_account_json_file_path gdrive-credentials.json

echo "--- Pulling DVC data ---"
dvc pull --force

echo "--- Reproducing DVC pipeline ---"
dvc repro

echo "--- Pushing DVC artifacts ---"
dvc push

echo "--- Committing results back to Git ---"
git add .
git commit -m "CI: Automated run from Kaggle GPU" || echo "No new changes to commit."
git push

echo "--- Workflow Finished Successfully ---"