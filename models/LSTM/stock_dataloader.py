import torch
import numpy as np
from torch.utils.data import Dataset

class StockDataloader(Dataset):
    # PyTorch Dataset for stock time series data

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        if len(y.shape) > 1:
            y = y.reshape(-1)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        print(f"Dataset created: X shape = {self.X.shape}, y shape = {self.y.shape}")

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
