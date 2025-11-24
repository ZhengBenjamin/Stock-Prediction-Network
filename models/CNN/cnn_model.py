import torch
import torch.nn as nn
from typing import Optional

class CNNModel(nn.Module):
    # 1D-CNN that predicts one value per timestep.
    def __init__(
        self,
        input_size: int,
        hidden_channels: Optional[int] = 64,
        kernel_size: int = 3,
        dropout: Optional[float] = 0.2,
    ):
        super().__init__()
        padding = kernel_size // 2

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
            nn.Conv1d(
                in_channels=hidden_channels,
                out_channels=1,
                kernel_size=1,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        out = self.net(x)
        out = out.permute(0, 2, 1)
        return out