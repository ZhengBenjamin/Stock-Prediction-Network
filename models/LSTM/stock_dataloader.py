import torch
import numpy as np

from torch.utils.data import Dataset
from typing import Optional


class StockDataloader(Dataset):
    """Dataset wrapper for stock sequences.

    X: shape (n_samples, seq_len, n_features)
    y: shape (n_samples, seq_len) -> converted to (n_samples, seq_len, 1)
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        # Ensure target has shape (n_samples, seq_len, 1)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
