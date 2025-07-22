import os
import yaml
import json
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, SubsetRandomSampler
from loguru import logger
import mlflow
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import config
from src.datasets import get_full_dataset
from src.cnn_lstm import MODELS
from src.train_utils import train_one_fold

def train():
    with open("params.yaml", 'r') as f:
        params = yaml.safe_load(f)
    feature_type = params['feature_type']
    cfg = config.get_config_for_feature(feature_type)
    
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.METRICS_DIR, exist_ok=True)
    
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.EXPERIMENT_NAME)
    with mlflow.start_run(run_name=f"{feature_type}_train_run") as run:
        mlflow.log_params(params)
        mlflow.log_param("k_folds", config.K_FOLDS)

        dataset = get_full_dataset(cfg['dataset_path'])
        kfold = KFold(n_splits=config.K_FOLDS, shuffle=True, random_state=42)
        
        fold_f1_scores = []
        for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
            train_loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, sampler=SubsetRandomSampler(train_ids))
            val_loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, sampler=SubsetRandomSampler(val_ids))
            
            ModelClass = MODELS[cfg['model_name']]
            model = ModelClass().to(config.DEVICE)
            
            fold_model_path = cfg['model_save_path_template'].format(fold=fold)
            _, best_f1 = train_one_fold(model, train_loader, val_loader, fold_model_path, fold+1)
            fold_f1_scores.append(best_f1)
        
        avg_val_f1 = sum(fold_f1_scores) / len(fold_f1_scores)
        metrics = {'avg_kfold_val_f1': avg_val_f1}
        mlflow.log_metric("avg_kfold_val_f1", avg_val_f1)
        
        with open(config.METRICS_FILE_TRAIN, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        logger.info(f"K-Fold training finished. Average Val F1: {avg_val_f1:.4f}")

if __name__ == "__main__":
    train()