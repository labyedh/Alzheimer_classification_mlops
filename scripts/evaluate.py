import os
import yaml
import json
import torch
from loguru import logger
import mlflow
import numpy as np

from src import config
from src.datasets import get_test_loader
from src.cnn_lstm import MODELS
from src.test_utils import evaluate_classifier

def evaluate():
    with open("params.yaml", 'r') as f:
        params = yaml.safe_load(f)
    feature_type = params['feature_type']
    cfg = config.get_config_for_feature(feature_type)
    
    test_loader = get_test_loader(cfg['dataset_path'], config.BATCH_SIZE)
    if not test_loader:
        logger.warning("No test data to evaluate.")
        return

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    runs = mlflow.search_runs(experiment_names=[config.EXPERIMENT_NAME], order_by=["start_time DESC"], max_results=1)
    run_id = runs.iloc[0].run_id

    with mlflow.start_run(run_id=run_id):
        logger.info(f"--- Running Final Evaluation in MLflow Run ID: {run_id} ---")

        all_reports = []
        for fold in range(config.K_FOLDS):
            model_path = cfg['model_save_path_template'].format(fold=fold)
            if not os.path.exists(model_path):
                logger.warning(f"Model for fold {fold} not found. Skipping.")
                continue

            ModelClass = MODELS[cfg['model_name']]
            model = ModelClass().to(config.DEVICE)
            model.load_state_dict(torch.load(model_path))
            
            logger.info(f"--- Evaluating model from fold {fold+1} ---")
            report = evaluate_classifier(model, test_loader, config.DEVICE, f"{feature_type}_fold{fold}")
            all_reports.append(report['weighted avg'])
        
        avg_f1 = np.mean([r['f1-score'] for r in all_reports]) if all_reports else 0
        mlflow.log_metric("avg_test_f1_score", avg_f1)
        
        final_metrics = {'average_test_f1': avg_f1}
        with open(config.METRICS_FILE_TEST, 'w') as f:
            json.dump(final_metrics, f, indent=4)
        
        logger.info(f"Evaluation finished. Average Test F1 on all folds: {avg_f1:.4f}")

if __name__ == "__main__":
    evaluate()