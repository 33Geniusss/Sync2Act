from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .act_lite import ACTLite


def quality_weighted_action_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    quality: torch.Tensor,
    padding_mask: torch.Tensor | None = None,
    loss_type: str = "mse",
) -> torch.Tensor:
    element = (
        F.mse_loss(prediction, target, reduction="none")
        if loss_type == "mse"
        else F.smooth_l1_loss(prediction, target, reduction="none")
    )
    step_loss = element.mean(dim=-1)
    weights = quality.to(step_loss.dtype)
    if padding_mask is not None:
        weights = weights * (~padding_mask).to(step_loss.dtype)
    denominator = weights.sum()
    return torch.where(
        denominator > 0,
        (step_loss * weights).sum() / denominator.clamp_min(1e-8),
        prediction.sum() * 0.0,
    )


class QualityAwareACT(ACTLite):
    def __init__(self, *args, use_quality_features: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_quality_features = use_quality_features
        self.quality_encoder = nn.Sequential(
            nn.Linear(3, self.hidden_dim), nn.ReLU(), nn.Linear(self.hidden_dim, self.hidden_dim)
        )

    def observation_tokens(self, state, image, quality=None, missing=None, time_offset=None):
        tokens = super().observation_tokens(state, image)
        if not self.use_quality_features:
            return tokens
        batch = state.shape[0]
        quality = torch.ones(batch, 1, device=state.device) if quality is None else quality[:, :1]
        missing = torch.zeros(batch, 1, device=state.device) if missing is None else missing[:, :1]
        time_offset = (
            torch.zeros(batch, 1, device=state.device)
            if time_offset is None
            else time_offset[:, :1]
        )
        quality_token = self.quality_encoder(
            torch.cat([quality, missing, time_offset], dim=-1)
        ).unsqueeze(1)
        return torch.cat([tokens, quality_token], dim=1)
