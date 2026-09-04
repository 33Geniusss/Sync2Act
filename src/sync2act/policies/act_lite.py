from __future__ import annotations

import torch
from torch import nn

from .common import ImageEncoder, sinusoidal_positions


class ACTLite(nn.Module):
    """Small action-query transformer; deliberately omits ACT's CVAE latent path."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        horizon: int = 16,
        hidden_dim: int = 96,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.state_dim, self.action_dim, self.horizon, self.hidden_dim = (
            state_dim,
            action_dim,
            horizon,
            hidden_dim,
        )
        self.image_encoder = ImageEncoder(hidden_dim)
        self.state_encoder = nn.Linear(state_dim, hidden_dim)
        self.action_queries = nn.Parameter(torch.randn(horizon, hidden_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            hidden_dim, num_heads, hidden_dim * 4, dropout, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers)
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, action_dim)
        )

    def observation_tokens(
        self,
        state: torch.Tensor,
        image: torch.Tensor,
        quality: torch.Tensor | None = None,
        missing: torch.Tensor | None = None,
        time_offset: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del quality, missing, time_offset
        if state.ndim != 2 or state.shape[1] != self.state_dim:
            raise ValueError(f"state must have shape [B, {self.state_dim}]")
        return torch.stack([self.image_encoder(image), self.state_encoder(state)], dim=1)

    def forward(
        self,
        state: torch.Tensor,
        image: torch.Tensor,
        quality: torch.Tensor | None = None,
        missing: torch.Tensor | None = None,
        time_offset: torch.Tensor | None = None,
        **_: torch.Tensor,
    ) -> torch.Tensor:
        observations = self.observation_tokens(state, image, quality, missing, time_offset)
        batch = state.shape[0]
        queries = self.action_queries.unsqueeze(0).expand(batch, -1, -1)
        tokens = torch.cat([observations, queries], dim=1)
        tokens = tokens + sinusoidal_positions(tokens.shape[1], self.hidden_dim, tokens.device)
        encoded = self.transformer(tokens)
        return self.output_head(encoded[:, -self.horizon :])
