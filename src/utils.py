# src/utils.py

import os
import re
import torch
import json
import matplotlib.pyplot as plt
from loguru import logger
import sys

def plot_and_save_history(history, feature_type, fold_num, plot_dir):
    """
    Plots training and validation curves for a single fold and saves the figure.

    Args:
        history (dict): A dictionary containing lists of metrics for each epoch.
        feature_type (str): The type of feature used (e.g., 'logmel', 'mfcc').
        fold_num (int): The current fold number (1-based).
        plot_dir (str): The directory to save the plot.
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f'Metrics for Fold {fold_num} ({feature_type.upper()})', fontsize=16)

    # Plot Loss
    ax1.plot(epochs, history['train_loss'], 'o-', label='Train Loss')
    ax1.plot(epochs, history['val_loss'], 'o-', label='Val Loss')
    ax1.set_title('Loss Curves')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot Accuracy and F1-Score
    ax2.plot(epochs, history['train_acc'], 'o-', label='Train Accuracy')
    ax2.plot(epochs, history['val_acc'], 'o-', label='Val Accuracy')
    ax2.plot(epochs, history['train_f1'], 's--', label='Train F1-Score')
    ax2.plot(epochs, history['val_f1'], 's--', label='Val F1-Score')
    ax2.set_title('Accuracy & F1-Score Curves')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Score')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make space for suptitle
    
    # Save the figure
    save_path = os.path.join(plot_dir, f"training_curves_{feature_type}_fold{fold_num}.png")
    plt.savefig(save_path)
    logger.info(f"Saved training plot for fold {fold_num} to {save_path}")
    plt.close() # Close the figure to free up memory


def load_checkpoint(model, checkpoint_path):
    """
    Loads a trained model checkpoint from a .pth file.

    Args:
        model (torch.nn.Module): An instance of the model architecture.
        checkpoint_path (str): Path to the saved checkpoint file.

    Returns:
        torch.nn.Module: The model with loaded weights.
    """
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
        logger.info(f"Successfully loaded checkpoint from {checkpoint_path}")
    except FileNotFoundError:
        logger.error(f"Checkpoint file not found at: {checkpoint_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
        raise
        
    return model


def setup_logging():
    """
    Configures the Loguru logger for consistent formatting and output.
    """
    logger.remove() # Remove default handler
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    logger.info("Logging is set up.")