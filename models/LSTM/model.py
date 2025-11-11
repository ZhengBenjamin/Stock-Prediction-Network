import torch
import torch.nn as nn
import numpy as np

from typing import Optional


class Model(nn.Module):
    """LSTM model that predicts a value for each timestep (many-to-many)."""

    def __init__(
        self,
        input_size: int,
        hidden_size: Optional[int] = 64,
        num_layers: Optional[int] = 2,
        dropout: Optional[float] = 0.2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        # Predict one value per timestep
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: tensor of shape (batch, seq_len, input_size)

        Returns:
            Tensor of shape (batch, seq_len, 1)
        """
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        # Apply the linear layer to each timestep's output to get
        # a prediction for every timestep (many-to-many).
        # out has shape (batch, seq_len, hidden_size) ->
        # after fc: (batch, seq_len, 1)
        out = self.fc(out)

        return out

