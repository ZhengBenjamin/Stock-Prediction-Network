import numpy as np

from .model import *
from .stock_dataloader import *
from utils.stock_preprocessor import StockPreprocessor
from torch.utils.data import DataLoader

from typing import Optional, List

class Trainer:
    
    def __init__(self, 
                 hidden_size: Optional[int] = 64, 
                 num_layers: Optional[int] = 2, 
                 dropout: Optional[int] = 0.2,
                 sequence_length: Optional[int] = 1000,
                 batch_size: Optional[int] = 32,
                 lr: Optional[float] = 0.001):
        
        self.preprocessor = StockPreprocessor(sequence_length=sequence_length)
        self.lr = lr

        self.X, self.y = self._prepare_data()
        self.stock_dataloader = StockDataloader(self.X, self.y)

        self.data = StockDataloader(self.X, self.y)
        self.loader = DataLoader(self.data, batch_size, shuffle=True)

        self.device = torch.device("cuda")

        self.model = Model(self.X.shape[2], hidden_size, num_layers, dropout).to(self.device)

    def train(self, epochs: Optional[int] = 10000):
        
        for epoch in range(epochs):
            self.model.train()
            loss = 0

            criterion = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

            for X_batch, y_batch in self.loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                loss += loss.item()
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss/len(self.loader):.6f}")

    def _prepare_data(self):
        data = self.preprocessor.get_normalized_data()
        X = data[:, :-1, :]  # All but last time step
        y = data[:, 1:, 4]   # Close price at next time step

        return X, y