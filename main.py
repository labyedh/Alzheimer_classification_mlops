# main.py

import argparse
import os
import yaml
from loguru import logger
from src.utils import setup_logging

# Import the core logic functions from your scripts folder
# This works because of the __init__.py file in the src folder
from scripts.train import train
from scripts.evaluate import evaluate
from src import config

def main():
    """
    Main entry point for running the audio classification pipeline manually.
    This script orchestrates the training and evaluation processes based on user commands.
    """
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Audio Classification Training and Evaluation Pipeline.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        "stage",
        type=str,
        choices=["train", "evaluate", "full-pipeline"],
        help="Which stage of the pipeline to run:\n"
             "  train          - Runs the K-Fold cross-validation training only.\n"
             "  evaluate       - Evaluates the trained models on the test set.\n"
             "  full-pipeline  - Runs both training and evaluation sequentially."
    )
    
    parser.add_argument(
        "--feature",
        type=str,
        choices=["logmel", "mfcc"],
        help="Temporarily override the 'feature_type' in params.yaml for this run."
    )

    args = parser.parse_args()

    # --- Load and Optionally Override Configuration ---
    with open("params.yaml", 'r') as f:
        params = yaml.safe_load(f)
    
    # If the --feature flag is used, update the params.yaml file
    # This ensures consistency between manual runs and DVC runs
    if args.feature and args.feature != params.get('feature_type'):
        logger.info(f"Overriding feature_type from params.yaml. Setting to '{args.feature}'.")
        params['feature_type'] = args.feature
        with open("params.yaml", 'w') as f:
            yaml.dump(params, f)

    feature_type = params['feature_type']
    logger.info(f"--- Running pipeline for FEATURE_TYPE: {feature_type.upper()} ---")

    # --- Create necessary directories ---
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    os.makedirs(config.METRICS_DIR, exist_ok=True)

    # --- Execute Pipeline Stages ---
    if args.stage == "train":
        logger.info(">>> Running Training Stage <<<")
        train()
    
    elif args.stage == "evaluate":
        logger.info(">>> Running Evaluation Stage <<<")
        evaluate()
        
    elif args.stage == "full-pipeline":
        logger.info(">>> Running Full Pipeline (Train -> Evaluate) <<<")
        
        # Run Training Stage
        logger.info("--- Stage 1: Training ---")
        train()
        
        # Run Evaluation Stage
        logger.info("--- Stage 2: Evaluation ---")
        evaluate()
        
    logger.info("--- Pipeline execution finished. ---")

if __name__ == "__main__":
    setup_logging() 
    main()