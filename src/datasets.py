import pickle
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from loguru import logger

class AudioDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.from_numpy(features).float()
        self.labels = torch.from_numpy(labels).long()
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def get_full_dataset(dataset_path):
    try:
        with open(dataset_path, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        logger.error(f"Dataset not found at {dataset_path}")
        raise

    X_list = [data.get(k, np.array([])) for k in ['X_train', 'X_val', 'X_test', 'X_train_norm', 'X_val_norm', 'X_test_norm']]
    y_list = [data.get(k, np.array([])) for k in ['y_train', 'y_val', 'y_test']]
    
    X_all = np.concatenate([x for x in X_list if x.size > 0], axis=0)
    y_all = np.concatenate([y for y in y_list if y.size > 0], axis=0)
    
    return AudioDataset(X_all, y_all)

def get_test_loader(dataset_path, batch_size):
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)

    X_test = data.get('X_test_norm', data.get('X_test'))
    y_test = data.get('y_test')
    
    if X_test is None or X_test.size == 0:
        logger.warning("Test set is empty.")
        return None

    test_dataset = AudioDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader