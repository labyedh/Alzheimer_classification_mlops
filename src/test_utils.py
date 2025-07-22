import torch
import os
import pandas as pd
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import mlflow

from src import config

def evaluate_classifier(model, test_loader, device, feature_type):
    model.eval()
    true_labels, predictions = [], []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = (torch.sigmoid(outputs) > 0.5).long()
            predictions.extend(preds.cpu().numpy().flatten())
            true_labels.extend(labels.cpu().numpy().flatten())

    report_dict = classification_report(true_labels, predictions, target_names=config.CLASS_NAMES, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    logger.info("\nClassification Report:\n%s", report_df)
    
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=config.CLASS_NAMES, yticklabels=config.CLASS_NAMES)
    plt.ylabel('True Labels'); plt.xlabel('Predicted Labels'); plt.title(f'Test Confusion Matrix ({feature_type})')
    
    cm_path = os.path.join(config.PLOTS_DIR, f"confusion_matrix_{feature_type}.png")
    plt.savefig(cm_path)
    plt.show()
    mlflow.log_artifact(cm_path, "evaluation_plots")
    
    return report_dict