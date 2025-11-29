import torch
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
                 dropout: Optional[float] = 0.2,
                 sequence_length: Optional[int] = 1000,
                 batch_size: Optional[int] = 32,
                 lr: Optional[float] = 0.001):
        
        self.preprocessor = StockPreprocessor(sequence_length=sequence_length)
        self.lr = lr

        # data to predict the next close after the provided sequence.
        # X : num_stocks, seq_len-1, num_features)
        # y : (num_stocks, 1) -> the next close value after the last timestep in X
        self.X, self.y = self._prepare_data()
        self.stock_dataloader = StockDataloader(self.X, self.y)

        self.data = StockDataloader(self.X, self.y)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loader = DataLoader(
            self.data,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True if self.device.type == "cuda" else False,
        )

        self.model = Model(self.X.shape[2], hidden_size, num_layers, dropout).to(self.device)
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def train(self, epochs: Optional[int] = 10000):
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0

            for X_batch, y_batch in self.loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(X_batch)           # (batch, 1)
                loss = self.criterion(outputs, y_batch) # (batch, 1)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            if epoch % 100 == 0:
                avg_loss = running_loss / max(1, len(self.loader))
                print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")

    def _prepare_data(self):
        data = self.preprocessor.get_normalized_data()
        # close feature idx 4 for autoregressive forecasting
        close = data[:, :, 4:5] # (S, T, 1)
        X = close[:, :-1, :] # (S, T-1, 1)
        y_last_close = close[:, -1, 0] # (S,)

        return X, y_last_close

    def forecast(self, horizon: int, stock_idx: int = 0) -> float:
        """Predict the normalized close price horizon days ahead via iterative forecasting.

        Returns the denormalized price for the horizon-th day ahead for the selected stock.
        """
        self.model.eval()
        with torch.no_grad():
            # Seed with the training sequence for the selected stock
            seq = torch.tensor(self.X[stock_idx], dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, L, 1)
            last_pred = None
            for _ in range(int(horizon)):
                last_pred = self.model(seq) # (1, 1) normalized
                # Slide the window and append the new prediction as next timestep
                seq = torch.cat([seq[:, 1:, :], last_pred.unsqueeze(1)], dim=1)

            # Denormalize the final prediction to price units
            pred_norm = last_pred.squeeze(0).squeeze(0).detach().cpu().numpy()
            pred_price = float(self.preprocessor.denormalize_close(pred_norm))
            return pred_price

    def evaluate(self, horizon: int = 1, stock_idx: int = 0) -> None:
        """Run a simple evaluation: print last known close and horizon-ahead predicted close in price units."""
        # Last known close (normalized) from the dataset
        last_close_norm = self.preprocessor.normalized_data[stock_idx, -1, 4]
        last_close_price = float(self.preprocessor.denormalize_close(np.array([last_close_norm]))[0])

        pred_price = self.forecast(horizon=horizon, stock_idx=stock_idx)
        print(f"\nEvaluation (stock #{stock_idx}):")
        print(f"  Last known close: {last_close_price:.4f}")
        print(f"  Predicted close {horizon} day(s) ahead: {pred_price:.4f}\n")