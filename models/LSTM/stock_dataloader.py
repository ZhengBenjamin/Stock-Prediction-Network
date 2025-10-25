import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
from typing import Optional, List

class StockDataloader(Dataset):
    """Needed for DataLoader in trainer.py"""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
