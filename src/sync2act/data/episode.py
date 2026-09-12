from __future__ import annotations

import copy
from typing import Any, TypeAlias

import torch

Episode: TypeAlias = dict[str, Any]

BASE_REQUIRED_KEYS = {
    "observation.image",
    "observation.state",
    "action",
    "timestamp",
}
MODAL_QUALITY_KEYS = {"image_quality", "state_quality", "action_label_quality"}
MODAL_OFFSET_KEYS = {"image_time_offset", "state_time_offset", "action_label_time_offset"}
MODAL_MISSING_KEYS = {
    "image_missing_mask",
    "state_missing_mask",
    "action_label_missing_mask",
}
# Public compatibility constant. Validation also accepts legacy episodes and upgrades them in place.
REQUIRED_KEYS = BASE_REQUIRED_KEYS | MODAL_QUALITY_KEYS | MODAL_OFFSET_KEYS | MODAL_MISSING_KEYS


def clone_episode(episode: Episode) -> Episode:
    """Clone tensors and nested metadata so corruptions never mutate their input."""
    return {
        key: value.clone() if torch.is_tensor(value) else copy.deepcopy(value)
        for key, value in episode.items()
    }


def _camera_count(image: torch.Tensor) -> int:
    return 1 if image.ndim == 4 else int(image.shape[1])


def _zeros(length: int, *, like: torch.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
    return torch.zeros(length, dtype=dtype or like.dtype, device=like.device)


def ensure_modal_metadata(episode: Episode) -> Episode:
    """Upgrade a clean or legacy episode to the modality-specific schema."""
    missing_base = BASE_REQUIRED_KEYS - episode.keys()
    if missing_base:
        raise ValueError(f"Episode is missing keys: {sorted(missing_base)}")
    length = int(episode["action"].shape[0])
    image = episode["observation.image"]
    camera_count = _camera_count(image)
    timestamp = episode["timestamp"]

    legacy_quality = episode.get("quality_score")
    if legacy_quality is None:
        legacy_quality = torch.ones(length, dtype=torch.float32, device=timestamp.device)
    else:
        legacy_quality = legacy_quality.float()
    episode.setdefault(
        "image_quality", legacy_quality.unsqueeze(1).expand(-1, camera_count).clone()
    )
    episode.setdefault("state_quality", legacy_quality.clone())
    episode.setdefault("action_label_quality", legacy_quality.clone())

    legacy_offset = episode.get("time_offset")
    if legacy_offset is None:
        legacy_offset = _zeros(length, like=timestamp)
    episode.setdefault(
        "image_time_offset", legacy_offset.unsqueeze(1).expand(-1, camera_count).clone()
    )
    episode.setdefault("state_time_offset", legacy_offset.clone())
    episode.setdefault("action_label_time_offset", legacy_offset.clone())

    legacy_missing = episode.get("missing_mask")
    if legacy_missing is None:
        legacy_missing = _zeros(length, like=timestamp, dtype=torch.bool)
    else:
        legacy_missing = legacy_missing.bool()
    episode.setdefault(
        "image_missing_mask", legacy_missing.unsqueeze(1).expand(-1, camera_count).clone()
    )
    episode.setdefault("state_missing_mask", legacy_missing.clone())
    episode.setdefault("action_label_missing_mask", legacy_missing.clone())

    # Allow older multi-camera episodes that carried [T, 1] image metadata.
    for key in ("image_quality", "image_time_offset", "image_missing_mask"):
        value = episode[key]
        if value.shape == (length, 1) and camera_count > 1:
            episode[key] = value.expand(-1, camera_count).clone()
    refresh_legacy_metadata(episode, validate=False)
    return episode


def ensure_modal_quality(episode: Episode) -> Episode:
    """Backward-compatible alias for the full modality metadata upgrade."""
    return ensure_modal_metadata(episode)


def _max_abs_signed(values: torch.Tensor) -> torch.Tensor:
    indices = values.abs().argmax(dim=1, keepdim=True)
    return values.gather(1, indices).squeeze(1)


def refresh_legacy_metadata(episode: Episode, *, validate: bool = True) -> Episode:
    """Refresh legacy scalar summaries without losing modality-specific information."""
    metadata_keys = MODAL_QUALITY_KEYS | MODAL_OFFSET_KEYS | MODAL_MISSING_KEYS
    if any(key not in episode for key in metadata_keys):
        ensure_modal_metadata(episode)
        return episode
    episode["quality_score"] = torch.stack(
        [
            episode["image_quality"].amin(dim=1),
            episode["state_quality"],
            episode["action_label_quality"],
        ],
        dim=1,
    ).amin(dim=1)
    episode["missing_mask"] = torch.stack(
        [
            episode["image_missing_mask"].any(dim=1),
            episode["state_missing_mask"],
            episode["action_label_missing_mask"],
        ],
        dim=1,
    ).any(dim=1)
    episode["time_offset"] = _max_abs_signed(
        torch.cat(
            [
                episode["image_time_offset"],
                episode["state_time_offset"].unsqueeze(1),
                episode["action_label_time_offset"].unsqueeze(1),
            ],
            dim=1,
        )
    )
    if validate:
        validate_episode(episode)
    return episode


def refresh_legacy_quality(episode: Episode) -> Episode:
    """Backward-compatible alias that refreshes all scalar summaries."""
    return refresh_legacy_metadata(episode)


def _validate_tensor(
    episode: Episode,
    key: str,
    shape: tuple[int, ...],
    *,
    finite: bool = False,
    unit_interval: bool = False,
    boolean: bool = False,
) -> None:
    value = episode[key]
    if not torch.is_tensor(value) or tuple(value.shape) != shape:
        raise ValueError(f"{key} must have shape {list(shape)}")
    if boolean and value.dtype != torch.bool:
        raise ValueError(f"{key} must have boolean dtype")
    if finite and not torch.isfinite(value).all():
        raise ValueError(f"{key} must contain only finite values")
    if unit_interval and ((value < 0).any() or (value > 1).any()):
        raise ValueError(f"{key} must contain values in [0, 1]")


def validate_episode(episode: Episode) -> int:
    ensure_modal_metadata(episode)
    length = int(episode["action"].shape[0])
    camera_count = _camera_count(episode["observation.image"])
    tracked_keys = REQUIRED_KEYS | {"quality_score", "missing_mask", "time_offset"}
    lengths = {
        key: int(value.shape[0])
        for key, value in episode.items()
        if key in tracked_keys and torch.is_tensor(value) and value.ndim
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Episode fields have inconsistent lengths: {lengths}")

    image = episode["observation.image"]
    state = episode["observation.state"]
    action = episode["action"]
    single_camera = image.ndim == 4 and image.shape[1] == 3
    multi_camera = image.ndim == 5 and image.shape[1] > 0 and image.shape[2] == 3
    if not (single_camera or multi_camera):
        raise ValueError("observation.image must have shape [T, 3, H, W] or [T, cameras, 3, H, W]")
    if state.ndim != 2 or action.ndim != 2:
        raise ValueError("state and action must have shape [T, D]")
    _validate_tensor(episode, "timestamp", (length,), finite=True)
    _validate_tensor(
        episode, "image_quality", (length, camera_count), finite=True, unit_interval=True
    )
    for key in ("state_quality", "action_label_quality", "quality_score"):
        _validate_tensor(episode, key, (length,), finite=True, unit_interval=True)
    _validate_tensor(episode, "image_time_offset", (length, camera_count), finite=True)
    for key in ("state_time_offset", "action_label_time_offset", "time_offset"):
        _validate_tensor(episode, key, (length,), finite=True)
    _validate_tensor(episode, "image_missing_mask", (length, camera_count), boolean=True)
    for key in ("state_missing_mask", "action_label_missing_mask", "missing_mask"):
        _validate_tensor(episode, key, (length,), boolean=True)
    return length
