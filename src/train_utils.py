import os
import torch
import torch.nn as nn
import mlflow
from tqdm import tqdm
from loguru import logger
from sklearn.metrics import accuracy_score, f1_score
from src import config
import numpy as np
def train_one_fold(model, train_loader, val_loader, fold_model_path, fold_num):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'train_f1': [], 'val_f1': []}
    best_val_f1 = 0.0
    patience_counter = 0

    for epoch in range(config.EPOCHS):
        model.train()
        train_preds, train_targets, train_losses = [], [], []
        for inputs, labels in tqdm(train_loader, desc=f"Fold {fold_num} Epoch {epoch+1}", leave=False):
            inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.float().unsqueeze(1))
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            preds = (torch.sigmoid(outputs) > 0.5).long()
            train_preds.extend(preds.cpu().numpy().flatten())
            train_targets.extend(labels.cpu().numpy().flatten())
        
        model.eval()
        val_preds, val_targets, val_losses = [], [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
                outputs = model(inputs)
                val_losses.append(criterion(outputs, labels.float().unsqueeze(1)).item())
                preds = (torch.sigmoid(outputs) > 0.5).long()
                val_preds.extend(preds.cpu().numpy().flatten())
                val_targets.extend(labels.cpu().numpy().flatten())

        history['train_loss'].append(np.mean(train_losses))
        history['val_loss'].append(np.mean(val_losses))
        history['train_acc'].append(accuracy_score(train_targets, train_preds))
        history['val_acc'].append(accuracy_score(val_targets, val_preds))
        history['train_f1'].append(f1_score(train_targets, train_preds, zero_division=0))
        history['val_f1'].append(f1_score(val_targets, val_preds, zero_division=0))
        
        mlflow.log_metrics({
            f"fold_{fold_num}_val_acc": history['val_acc'][-1],
            f"fold_{fold_num}_val_f1": history['val_f1'][-1],
        }, step=epoch)

        if history['val_f1'][-1] > best_val_f1:
            best_val_f1 = history['val_f1'][-1]
            torch.save(model.state_dict(), fold_model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                logger.info(f"Early stopping at epoch {epoch+1}.")
                break
                
    logger.info(f"Fold {fold_num} best validation F1: {best_val_f1:.4f}")
    
    if not os.path.exists(fold_model_path):
        torch.save(model.state_dict(), fold_model_path)
        logger.warning(f"No improvement in fold {fold_num}; saving model from last epoch.")

    return history, best_val_f1