import os
import yaml
import json
import torch
import numpy as np
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
from src.utils import plot_and_save_history, setup_logging

def train():
    setup_logging()
    
    with open("params.yaml", 'r') as f:
        params = yaml.safe_load(f)
    feature_type = params['feature_type']
    cfg = config.get_config_for_feature(feature_type)
    
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    os.makedirs(config.METRICS_DIR, exist_ok=True)
    
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.EXPERIMENT_NAME)
    
    with mlflow.start_run(run_name=f"{feature_type}_train_run") as run:
        logger.info(f"Starting MLflow Run ID: {run.info.run_id}")
        mlflow.log_params(params)
        mlflow.log_param("k_folds", config.K_FOLDS)

        dataset = get_full_dataset(cfg['dataset_path'])
        kfold = KFold(n_splits=config.K_FOLDS, shuffle=True, random_state=42)
        
        fold_metrics = {'f1': [], 'acc': [], 'loss': []}

        for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
            fold_num = fold + 1
            logger.info(f"--- Starting FOLD {fold_num}/{config.K_FOLDS} ---")
            
            train_loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, sampler=SubsetRandomSampler(train_ids))
            val_loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, sampler=SubsetRandomSampler(val_ids))
            
            ModelClass = MODELS[cfg['model_name']]
            model = ModelClass().to(config.DEVICE)
            
            fold_model_path = cfg['model_save_path_template'].format(fold=fold)
            
            history, _ = train_one_fold(model, train_loader, val_loader, fold_model_path, fold_num)
            
            if not history['val_f1']:
                logger.warning(f"Fold {fold_num} did not complete any epochs. Skipping.")
                continue
            
            plot_and_save_history(history, feature_type, fold_num, config.PLOTS_DIR)
            plot_path = os.path.join(config.PLOTS_DIR, f"training_curves_{feature_type}_fold{fold_num}.png")
            mlflow.log_artifact(plot_path, f"fold_{fold_num}_plots")

            best_epoch_index = np.argmax(history['val_f1'])
            fold_metrics['f1'].append(history['val_f1'][best_epoch_index])
            fold_metrics['acc'].append(history['val_acc'][best_epoch_index])
            fold_metrics['loss'].append(history['val_loss'][best_epoch_index])
        
        if fold_metrics['f1']:
            avg_val_f1 = np.mean(fold_metrics['f1'])
            dvc_metrics = {'avg_kfold_val_f1': avg_val_f1}
            with open(config.METRICS_FILE_TRAIN, 'w') as f:
                json.dump(dvc_metrics, f, indent=4)
            logger.info(f"K-Fold training finished. Average Val F1: {avg_val_f1:.4f}")
        else:
            logger.error("No folds completed. No metrics to report.")

if __name__ == "__main__":
    train()