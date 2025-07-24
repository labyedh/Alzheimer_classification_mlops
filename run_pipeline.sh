#!/bin/bash
# run_pipeline.sh
# This script automates the entire ML workflow: setup, training, and pushing results.

# Exit immediately if any command fails, ensuring the script stops on error.
set -e

# --- 1. SETUP & CONFIGURATION ---
echo "--- Setting up SSH for GitHub ---"
# This section creates the SSH key from the secret passed by the Kaggle notebook.
# This key is required for the final 'git push'.
mkdir -p ~/.ssh
printf "%s\n" "$(echo "${GIT_DEPLOY_KEY_B64}" | base64 -d)" > ~/.ssh/id_ed25519
chmod 600 ~/.ssh/id_ed25519
ssh-keyscan github.com >> ~/.ssh/known_hosts

echo "--- Configuring Git ---"
# Set the author for the commit that will be made by this script.
git config --global user.name "Kaggle CI Bot"
git config --global user.email "kaggle-bot@users.noreply.github.com"

# Ensure the git remote is set to use the SSH URL for pushing.
GIT_USERNAME="labyedh"
GIT_REPO="Alzheimer_classification_mlops"
git remote set-url origin "git@github.com:${GIT_USERNAME}/${GIT_REPO}.git"

echo "--- Setting up DVC Credentials ---"
# Create the Google Drive credentials file from the secret.
echo "${GDRIVE_CREDENTIALS_DATA_B64}" | base64 -d > gdrive-credentials.json

echo "--- Installing Dependencies ---"
pip install -r requirements.txt --quiet


# --- 2. RUN THE DVC PIPELINE ---
echo "--- Pulling DVC data ---"
dvc pull --force

echo "--- Reproducing DVC pipeline (Training & Evaluation) ---"
dvc repro


# --- 3. PUSH ARTIFACTS AND RESULTS ---
echo "--- Pushing DVC artifacts (models, plots) to remote storage ---"
# This uploads the large files to your Google Drive.
dvc push

echo "--- Committing results (dvc.lock, metrics) back to Git ---"
# Stage all changes that DVC made (like dvc.lock) and any new metrics.
git add .

# Create the commit. The '|| true' part prevents an error if no files changed.
git commit -m "CI: Automated run from Kaggle GPU" || true


# --- THE FINAL STEP: PUSH TO GITHUB ---
echo "--- Pushing code and DVC updates to GitHub repository ---"
# This command pushes the commit you just made back to your main branch.
git push


echo "--- Workflow Finished Successfully ---"