from __future__ import annotations

from typing import Any, TypeAlias

import torch

Episode: TypeAlias = dict[str, Any]
REQUIRED_KEYS = {
    "observation.image",
    "observation.state",
    "action",
    "timestamp",
    "quality_score",
    "missing_mask",
}


def clone_episode(episode: Episode) -> Episode:
    return {
        key: value.clone() if torch.is_tensor(value) else value for key, value in episode.items()
    }


def validate_episode(episode: Episode) -> int:
    missing = REQUIRED_KEYS - episode.keys()
    if missing:
        raise ValueError(f"Episode is missing keys: {sorted(missing)}")
    lengths = {key: int(episode[key].shape[0]) for key in REQUIRED_KEYS}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Episode fields have inconsistent lengths: {lengths}")
    image = episode["observation.image"]
    state = episode["observation.state"]
    action = episode["action"]
    single_camera = image.ndim == 4 and image.shape[1] == 3
    multi_camera = image.ndim == 5 and image.shape[1] > 0 and image.shape[2] == 3
    if not (single_camera or multi_camera):
        raise ValueError(
            "observation.image must have shape [T, 3, H, W] or [T, cameras, 3, H, W]"
        )
    if state.ndim != 2 or action.ndim != 2:
        raise ValueError("state and action must have shape [T, D]")
    if episode["timestamp"].ndim != 1:
        raise ValueError("timestamp must have shape [T]")
    return next(iter(lengths.values()))
