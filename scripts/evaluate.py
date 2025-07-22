import os
import yaml
import json
import torch
import numpy as np
from loguru import logger
import mlflow
from mlflow.models.signature import infer_signature
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import config
from src.datasets import get_test_loader
from src.cnn_lstm import MODELS
from src.test_utils import evaluate_single_model, get_predictions, log_final_ensemble_metrics
from src.utils import setup_logging

def evaluate():
    setup_logging()
    
    with open("params.yaml", 'r') as f:
        params = yaml.safe_load(f)
    feature_type = params['feature_type']
    cfg = config.get_config_for_feature(feature_type)
    
    test_loader = get_test_loader(cfg['dataset_path'], config.BATCH_SIZE)
    if not test_loader:
        return

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    runs = mlflow.search_runs(experiment_names=[config.EXPERIMENT_NAME], filter_string=f"tags.mlflow.runName = '{feature_type}_train_run'", order_by=["start_time DESC"], max_results=1)
    if runs.empty:
        logger.error(f"No MLflow run found for '{feature_type}_train_run'.")
        return
    run_id = runs.iloc[0].run_id

    with mlflow.start_run(run_id=run_id):
        logger.info(f"--- Running Final Evaluation in MLflow Run ID: {run_id} ---")
        
        best_model_path, best_model_loss = None, float('inf')
        all_fold_outputs, true_labels = [], None

        for fold in range(config.K_FOLDS):
            model_path = cfg['model_save_path_template'].format(fold=fold)
            if not os.path.exists(model_path):
                continue

            ModelClass = MODELS[cfg['model_name']]
            model = ModelClass().to(config.DEVICE)
            model.load_state_dict(torch.load(model_path))
            
            metrics = evaluate_single_model(model, test_loader, config.DEVICE)
            if metrics['loss'] < best_model_loss:
                best_model_loss = metrics['loss']
                best_model_path = model_path
            
            outputs, labels = get_predictions(model, test_loader, config.DEVICE)
            all_fold_outputs.append(outputs)
            if true_labels is None: true_labels = labels
        
        if best_model_path:
            logger.info(f"Registering best model from '{best_model_path}'")
            BestModelClass = MODELS[cfg['model_name']]
            best_model = BestModelClass()
            best_model.load_state_dict(torch.load(best_model_path))
            input_sample, _ = next(iter(test_loader))
            signature = infer_signature(input_sample.numpy(), best_model(input_sample).detach().numpy())
            
            mlflow.pytorch.log_model(
                pytorch_model=best_model,
                artifact_path="best-model",
                signature=signature,
                registered_model_name=f"{feature_type}-classifier"
            )
        
        if all_fold_outputs:
            stacked_outputs = torch.stack(all_fold_outputs)
            mean_outputs = torch.mean(stacked_outputs, dim=0)
            ensemble_preds = (torch.sigmoid(mean_outputs) > 0.5).long().numpy().flatten()
            
            avg_test_f1 = log_final_ensemble_metrics(true_labels.numpy(), ensemble_preds, feature_type)
            
            final_metrics = {'ensemble_test_f1_score': avg_test_f1}
            with open(config.METRICS_FILE_TEST, 'w') as f:
                json.dump(final_metrics, f, indent=4)
            logger.info(f"Evaluation finished. Ensemble Test F1-Score: {avg_test_f1:.4f}")

if __name__ == "__main__":
    evaluate()