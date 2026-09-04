from __future__ import annotations

import math

import torch
from torch import nn


class ImageEncoder(nn.Module):
    def __init__(self, output_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, output_dim),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("image must have shape [B, 3, H, W]")
        return self.network(image)


def sinusoidal_positions(length: int, dimension: int, device: torch.device) -> torch.Tensor:
    position = torch.arange(length, device=device).float().unsqueeze(1)
    scale = torch.exp(
        torch.arange(0, dimension, 2, device=device).float() * (-math.log(10000) / dimension)
    )
    encoding = torch.zeros(length, dimension, device=device)
    encoding[:, 0::2] = torch.sin(position * scale)
    encoding[:, 1::2] = torch.cos(position * scale[: encoding[:, 1::2].shape[1]])
    return encoding
