from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .act_lite import ACTLite


def quality_weighted_action_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    action_label_quality: torch.Tensor,
    padding_mask: torch.Tensor | None = None,
    loss_type: str = "mse",
) -> torch.Tensor:
    element = (
        F.mse_loss(prediction, target, reduction="none")
        if loss_type == "mse"
        else F.smooth_l1_loss(prediction, target, reduction="none")
    )
    step_loss = element.mean(dim=-1)
    weights = action_label_quality.to(step_loss.dtype)
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
        self.image_quality_encoder = nn.Sequential(
            nn.Linear(3, self.hidden_dim), nn.ReLU(), nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        self.state_quality_encoder = nn.Sequential(
            nn.Linear(3, self.hidden_dim), nn.ReLU(), nn.Linear(self.hidden_dim, self.hidden_dim)
        )

    def observation_tokens(
        self,
        state,
        image,
        quality=None,
        missing=None,
        time_offset=None,
        image_quality=None,
        state_quality=None,
        image_missing=None,
        state_missing=None,
        image_time_offset=None,
        state_time_offset=None,
    ):
        tokens = super().observation_tokens(state, image)
        if not self.use_quality_features:
            return tokens
        batch = state.shape[0]
        camera_count = tokens.shape[1] - 1
        legacy_quality = (
            torch.ones(batch, 1, device=state.device) if quality is None else quality[:, :1]
        )
        image_quality = legacy_quality if image_quality is None else image_quality
        if image_quality.ndim == 1:
            image_quality = image_quality.unsqueeze(1)
        if image_quality.shape[1] == 1 and camera_count > 1:
            image_quality = image_quality.expand(-1, camera_count)
        if image_quality.shape != (batch, camera_count):
            raise ValueError(f"image_quality must have shape [B, {camera_count}]")
        state_quality = legacy_quality if state_quality is None else state_quality
        if state_quality.ndim == 1:
            state_quality = state_quality.unsqueeze(1)
        if state_quality.shape != (batch, 1):
            raise ValueError("state_quality must have shape [B, 1]")
        legacy_missing = (
            torch.zeros(batch, 1, device=state.device) if missing is None else missing[:, :1]
        )
        legacy_offset = (
            torch.zeros(batch, 1, device=state.device)
            if time_offset is None
            else time_offset[:, :1]
        )
        image_missing = legacy_missing if image_missing is None else image_missing
        image_time_offset = legacy_offset if image_time_offset is None else image_time_offset
        if image_missing.ndim == 1:
            image_missing = image_missing.unsqueeze(1)
        if image_time_offset.ndim == 1:
            image_time_offset = image_time_offset.unsqueeze(1)
        if image_missing.shape[1] == 1 and camera_count > 1:
            image_missing = image_missing.expand(-1, camera_count)
        if image_time_offset.shape[1] == 1 and camera_count > 1:
            image_time_offset = image_time_offset.expand(-1, camera_count)
        if image_missing.shape != (batch, camera_count):
            raise ValueError(f"image_missing must have shape [B, {camera_count}]")
        if image_time_offset.shape != (batch, camera_count):
            raise ValueError(f"image_time_offset must have shape [B, {camera_count}]")
        state_missing = legacy_missing if state_missing is None else state_missing
        state_time_offset = legacy_offset if state_time_offset is None else state_time_offset
        if state_missing.ndim == 1:
            state_missing = state_missing.unsqueeze(1)
        if state_time_offset.ndim == 1:
            state_time_offset = state_time_offset.unsqueeze(1)
        if state_missing.shape != (batch, 1):
            raise ValueError("state_missing must have shape [B, 1]")
        if state_time_offset.shape != (batch, 1):
            raise ValueError("state_time_offset must have shape [B, 1]")
        image_quality = image_quality.to(dtype=tokens.dtype, device=tokens.device)
        state_quality = state_quality.to(dtype=tokens.dtype, device=tokens.device)
        image_missing = image_missing.to(dtype=tokens.dtype, device=tokens.device)
        state_missing = state_missing.to(dtype=tokens.dtype, device=tokens.device)
        image_time_offset = image_time_offset.to(dtype=tokens.dtype, device=tokens.device)
        state_time_offset = state_time_offset.to(dtype=tokens.dtype, device=tokens.device)
        image_metadata = torch.stack([image_quality, image_missing, image_time_offset], dim=-1)
        state_metadata = torch.cat([state_quality, state_missing, state_time_offset], dim=-1)
        camera_tokens = tokens[:, :camera_count] * image_quality.unsqueeze(-1)
        camera_tokens = camera_tokens + self.image_quality_encoder(image_metadata)
        state_token = tokens[:, camera_count:] * state_quality.unsqueeze(-1)
        state_token = state_token + self.state_quality_encoder(state_metadata).unsqueeze(1)
        return torch.cat([camera_tokens, state_token], dim=1)
