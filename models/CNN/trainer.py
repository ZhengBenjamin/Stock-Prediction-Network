"""CNN trainer for stock prediction (many-to-many)."""

import torch
from typing import Optional, Tuple
from torch.utils.data import DataLoader

from models.CNN.cnn_model import CNNModel
from models.LSTM.stock_dataloader import StockDataloader
from utils.stock_preprocessor import StockPreprocessor


class CNNTrainer:
    """Trainer that uses the CNNModel and the existing StockDataloader.

    The trainer mirrors the LSTM trainer interface but uses a 1D CNN
    that returns one prediction per timestep (batch, seq_len, 1).
    """

    def __init__(
        self,
        hidden_channels: Optional[int] = 64,
        kernel_size: int = 3,
        dropout: float = 0.2,
        sequence_length: Optional[int] = 1000,
        batch_size: Optional[int] = 32,
        lr: Optional[float] = 1e-3,
    ):
        self.preprocessor = StockPreprocessor(sequence_length=sequence_length)
        self.lr = lr

        self.X, self.y = self._prepare_data()
        self.data = StockDataloader(self.X, self.y)
        self.loader = DataLoader(self.data, batch_size, shuffle=True)

        self.device = (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")
        )

        # CNN expects input_size as number of features
        input_size = self.X.shape[2]
        self.model = CNNModel(
            input_size=input_size,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            dropout=dropout,
        ).to(self.device)

    def train(self, epochs: Optional[int] = 1000):
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0

            for X_batch, y_batch in self.loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(f"Epoch {epoch}, Loss: {running_loss/len(self.loader):.6f}")

    def _prepare_data(self) -> Tuple:
        data = self.preprocessor.get_normalized_data()
        X = data[:, :-1, :]
        y = data[:, 1:, 4]
        return X, y
