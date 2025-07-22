import os
import torch
import matplotlib.pyplot as plt
from loguru import logger
import sys

def plot_and_save_history(history, feature_type, fold_num, plot_dir):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f'Metrics for Fold {fold_num} ({feature_type.upper()})', fontsize=16)

    ax1.plot(epochs, history['train_loss'], 'o-', label='Train Loss')
    ax1.plot(epochs, history['val_loss'], 'o-', label='Val Loss')
    ax1.set_title('Loss Curves'); ax1.set_xlabel('Epochs'); ax1.set_ylabel('Loss')
    ax1.legend(); ax1.grid(True)
    
    ax2.plot(epochs, history['train_acc'], 'o-', label='Train Accuracy')
    ax2.plot(epochs, history['val_acc'], 'o-', label='Val Accuracy')
    ax2.plot(epochs, history['train_f1'], 's--', label='Train F1-Score')
    ax2.plot(epochs, history['val_f1'], 's--', label='Val F1-Score')
    ax2.set_title('Accuracy & F1-Score Curves'); ax2.set_xlabel('Epochs'); ax2.set_ylabel('Score')
    ax2.legend(); ax2.grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(plot_dir, f"training_curves_{feature_type}_fold{fold_num}.png")
    plt.savefig(save_path)
    logger.info(f"Saved training plot for fold {fold_num} to {save_path}")
    plt.close()

def setup_logging():
    logger.remove()
    logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>", level="INFO")