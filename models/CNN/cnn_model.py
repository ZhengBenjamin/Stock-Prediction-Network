import torch
import torch.nn as nn

from typing import Optional


class CNNModel(nn.Module):
    """Simple 1D-CNN that predicts one value per timestep.

    Expects input x of shape (batch, seq_len, input_size).
    Returns tensor of shape (batch, seq_len, 1).
    """

    def __init__(
        self,
        input_size: int,
        hidden_channels: Optional[int] = 64,
        kernel_size: int = 3,
        dropout: Optional[float] = 0.2,
    ):
        super().__init__()
        padding = kernel_size // 2

        # Conv1d uses (batch, channels, seq_len)
        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels=input_size,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            # project to single value per timestep
            nn.Conv1d(
                in_channels=hidden_channels,
                out_channels=1,
                kernel_size=1,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size) -> (batch, input_size, seq_len)
        x = x.permute(0, 2, 1)
        out = self.net(x)  # (batch, 1, seq_len)
        out = out.permute(0, 2, 1)  # -> (batch, seq_len, 1)
        return out
