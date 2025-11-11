"""LSTM model trainer for stock price prediction."""

import numpy as np
import torch
import os

from .model import Model
from .stock_dataloader import StockDataloader
from utils.stock_preprocessor import StockPreprocessor
from torch.utils.data import DataLoader
from typing import Optional, Tuple


class Trainer:
    """Trainer class for the LSTM model."""

    def __init__(
        self,
        hidden_size: Optional[int] = 64,
        num_layers: Optional[int] = 2,
        dropout: Optional[float] = 0.2,
        sequence_length: Optional[int] = 1000,
        batch_size: Optional[int] = 32,
        lr: Optional[float] = 0.001
    ):
        """Initialize the trainer with model parameters."""
        self.preprocessor = StockPreprocessor(sequence_length=sequence_length)
        self.lr = lr

        self.X, self.y = self._prepare_data()
        self.stock_dataloader = StockDataloader(self.X, self.y)

        self.data = StockDataloader(self.X, self.y)
        self.loader = DataLoader(self.data, batch_size, shuffle=True)

        # Set device to CUDA if available, else CPU
        self.device = (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")
        )

        # Initialize model and move to device
        self.model = Model(
            self.X.shape[2], hidden_size, num_layers, dropout
        ).to(self.device)

        # Create checkpoints directory
        self.checkpoint_dir = "checkpoints/lstm"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_loss = float('inf')

    def train(self, epochs: Optional[int] = 10000):

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0

            for X_batch, y_batch in self.loader:
                # Move batch data to the appropriate device
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(self.loader)
            print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")

            # Save checkpoint if best loss
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                checkpoint_path = os.path.join(
                    self.checkpoint_dir, "best_model.pt"
                )
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": self.model.state_dict(),
                        "loss": avg_loss,
                        "input_size": self.X.shape[2],
                    },
                    checkpoint_path,
                )
                print(f"Checkpoint saved to {checkpoint_path}")

    def _prepare_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare and normalize training data."""
        data = self.preprocessor.get_normalized_data()
        X = data[:, :-1, :]  # All but last time step
        y = data[:, 1:, 4]   # Close price at next time step

        return X, y
