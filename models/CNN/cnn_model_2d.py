import torch
import torch.nn as nn
from typing import Optional


class CNN2DModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_channels: Optional[int] = 32,
        kernel_time: int = 3,
        kernel_feat: int = 3,
        dropout: Optional[float] = 0.2,
    ):
        super().__init__()
        kt = kernel_time
        kf = kernel_feat
        pt = kt // 2
        pf = kf // 2

        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=hidden_channels,
                kernel_size=(kt, kf),
                padding=(pt, pf),
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=(kt, kf),
                padding=(pt, pf),
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            # single channel per spatial location
            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=1,
                kernel_size=1,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        out = self.net(x)
        
        out = out.mean(dim=-1, keepdim=True)
        
        out = out.squeeze(1)
        return out
