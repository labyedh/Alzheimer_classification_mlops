import os
import torch
import torch.nn as nn
import pandas as pd
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import mlflow

from src import config

def get_predictions(model, data_loader, device):
    model.eval()
    all_outputs, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            all_outputs.append(outputs.cpu())
            all_labels.append(labels.cpu())
    return torch.cat(all_outputs), torch.cat(all_labels)

def evaluate_single_model(model, data_loader, device):
    criterion = nn.BCEWithLogitsLoss()
    outputs, labels = get_predictions(model, data_loader, device)
    loss = criterion(outputs, labels.float().unsqueeze(1)).item()
    preds = (torch.sigmoid(outputs) > 0.5).long().numpy().flatten()
    true_labels = labels.numpy()
    accuracy = accuracy_score(true_labels, preds)
    f1 = f1_score(true_labels, preds, average='weighted', zero_division=0)
    return {"loss": loss, "accuracy": accuracy, "f1_score": f1}

def log_final_ensemble_metrics(true_labels, predictions, feature_type):
    logger.info("\n--- Final Ensemble Test Set Performance ---")
    report_dict = classification_report(true_labels, predictions, target_names=config.CLASS_NAMES, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    logger.info("\nClassification Report:\n%s", report_df)

    mlflow.log_metric("ensemble_test_accuracy", report_dict['accuracy'])
    mlflow.log_metric("ensemble_test_f1_weighted", report_dict['weighted avg']['f1-score'])
    
    report_path = os.path.join(config.PLOTS_DIR, f"test_report_{feature_type}.csv")
    report_df.to_csv(report_path)
    mlflow.log_artifact(report_path, "evaluation_reports")
    
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=config.CLASS_NAMES, yticklabels=config.CLASS_NAMES)
    plt.ylabel('True Labels'); plt.xlabel('Predicted Labels'); plt.title(f'Ensemble Confusion Matrix ({feature_type})')
    
    cm_path = os.path.join(config.PLOTS_DIR, f"confusion_matrix_{feature_type}.png")
    plt.savefig(cm_path)
    plt.show(); plt.close()
    mlflow.log_artifact(cm_path, "evaluation_plots")
    
    return report_dict['weighted avg']['f1-score']