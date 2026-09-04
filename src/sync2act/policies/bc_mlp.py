from __future__ import annotations

import torch
from torch import nn

from .common import ImageEncoder


class BCMLP(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, input_mode: str = "state_only", hidden_dim: int = 128
    ):
        super().__init__()
        if input_mode not in {"state_only", "image_state"}:
            raise ValueError("input_mode must be state_only or image_state")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.input_mode = input_mode
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 64), nn.ReLU()
        )
        self.image_encoder = ImageEncoder(64) if input_mode == "image_state" else None
        self.head = nn.Sequential(
            nn.Linear(128 if self.image_encoder else 64, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(
        self, state: torch.Tensor, image: torch.Tensor | None = None, **_: torch.Tensor
    ) -> torch.Tensor:
        if state.ndim != 2 or state.shape[1] != self.state_dim:
            raise ValueError(f"state must have shape [B, {self.state_dim}]")
        features = self.state_encoder(state)
        if self.image_encoder is not None:
            if image is None:
                raise ValueError("image is required in image_state mode")
            features = torch.cat([self.image_encoder(image), features], dim=-1)
        return self.head(features).unsqueeze(1)
