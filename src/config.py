import os
import torch

# --- SHARED PROJECT CONSTANTS ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K_FOLDS = 1
BATCH_SIZE = 16
LEARNING_RATE = 0.0001
EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10

# --- DIRECTORY & FILE PATHS ---
DATA_DIR = "data"
MODEL_DIR = "models"
PLOTS_DIR = "plots"
METRICS_DIR = "metrics"
METRICS_FILE_TRAIN = os.path.join(METRICS_DIR, "train_metrics.json")
METRICS_FILE_TEST = os.path.join(METRICS_DIR, "test_metrics.json")

CLASS_NAMES = ["AD", "Control"]

# --- MLflow Configuration ---
MLFLOW_TRACKING_URI = "file:./mlruns"
EXPERIMENT_NAME = "Dementia_Audio_Classification"

# --- FEATURE-SPECIFIC SETTINGS ---
FEATURE_CONFIGS = {
    "logmel": {
        "dataset_filename": "log_mel_data.pkl",
        "model_name": "CNN_LSTM_LogMel",
    },
    "mfcc": {
        "dataset_filename": "mfcc_data.pkl",
        "model_name": "CNN_LSTM_MFCC",
    }
}

def get_config_for_feature(feature_type: str) -> dict:
    if feature_type not in FEATURE_CONFIGS:
        raise ValueError(f"Invalid feature_type: '{feature_type}'.")
    
    config = FEATURE_CONFIGS[feature_type].copy()
    config['dataset_path'] = os.path.join(DATA_DIR, config['dataset_filename'])
    config['model_save_path_template'] = os.path.join(MODEL_DIR, f"model_{feature_type}_fold_{{fold}}.pth")
    return config