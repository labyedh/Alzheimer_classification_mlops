#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- 1. SETUP & CONFIGURATION ---
echo "--- Setting up SSH for GitHub ---"
# Decode the base64 encoded deploy key and set up SSH
mkdir -p ~/.ssh
echo -e "${GIT_DEPLOY_KEY_B64}" | base64 -d > ~/.ssh/id_ed25519
chmod 600 ~/.ssh/id_ed25519 # Set strict permissions
ssh-keyscan github.com >> ~/.ssh/known_hosts

echo "--- Configuring Git ---"
git config --global user.name "Kaggle CI Runner"
git config --global user.email "kaggle-ci@github.com"
# The git remote is already configured from the checkout

echo "--- Setting up DVC Credentials ---"
# Decode the base64 encoded service account key
echo -e "${GDRIVE_CREDENTIALS_DATA_B64}" | base64 -d > gdrive-credentials.json

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
# Commit only if there are changes to commit
git diff-index --quiet HEAD || git commit -m "CI: Automated run from Kaggle GPU"
git push

echo "--- Workflow Finished Successfully ---"