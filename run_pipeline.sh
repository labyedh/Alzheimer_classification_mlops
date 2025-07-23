#!/bin/bash
# run_pipeline.sh

# Exit immediately if a command fails
set -e

# --- 1. SETUP & CONFIGURATION ---
echo "--- Setting up SSH for GitHub ---"
mkdir -p ~/.ssh

# --- START OF FIX ---
# Use 'printf' to write the key. It is more reliable than 'echo'
# for preserving whitespace and ensuring a trailing newline.
# The GIT_DEPLOY_KEY_B64 is passed from the notebook.
printf "%s\n" "$(echo "${GIT_DEPLOY_KEY_B64}" | base64 -d)" > ~/.ssh/id_ed25519
# --- END OF FIX ---

chmod 600 ~/.ssh/id_ed25519
ssh-keyscan github.com >> ~/.ssh/known_hosts

echo "--- Configuring Git ---"
git config --global user.name "Kaggle CI Runner"
git config --global user.email "ci-runner@kaggle.com"
git remote set-url origin git@github.com:labyedh/Alzheimer_classification_mlops.git

echo "--- Setting up DVC Credentials ---"
echo "${GDRIVE_CREDENTIALS_DATA_B64}" | base64 -d > gdrive-credentials.json

echo "--- Installing Dependencies ---"
pip install -r requirements.txt --quiet

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