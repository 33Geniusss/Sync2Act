from __future__ import annotations

import copy
from typing import Any

import torch

from sync2act.data.episode import (
    Episode,
    clone_episode,
    ensure_modal_metadata,
    refresh_legacy_metadata,
    validate_episode,
)


def _shift_time_offset(timestamp: torch.Tensor, shift: int) -> torch.Tensor:
    """Return source-time minus reference-time for a frame shift."""
    length = timestamp.shape[0]
    if shift == 0:
        return torch.zeros_like(timestamp)
    indices = torch.arange(length, device=timestamp.device)
    source_indices = indices - int(shift)
    valid = (source_indices >= 0) & (source_indices < length)
    clamped = source_indices.clamp(0, length - 1)
    offset = timestamp[clamped] - timestamp
    if length > 1:
        frame_interval = torch.median(timestamp[1:] - timestamp[:-1])
        offset[~valid] = -float(shift) * frame_interval
    else:
        offset[~valid] = 0
    return offset


def quality_from_time_offset(
    time_offset: torch.Tensor, reference_timestamp: torch.Tensor
) -> torch.Tensor:
    """Convert temporal misalignment into severity-aware quality in [0, 1]."""
    if reference_timestamp.shape[0] > 1:
        intervals = (reference_timestamp[1:] - reference_timestamp[:-1]).abs()
        frame_interval = torch.median(intervals).clamp_min(
            torch.finfo(reference_timestamp.dtype).eps
        )
    else:
        frame_interval = torch.tensor(
            1e-6,
            dtype=reference_timestamp.dtype,
            device=reference_timestamp.device,
        )
    return torch.exp(-time_offset.abs() / (2.0 * frame_interval)).clamp(0.0, 1.0)


def _record(
    result: Episode,
    kind: str,
    seed: int,
    params: dict[str, Any],
    indices: torch.Tensor,
    *,
    missing_indices: torch.Tensor | None = None,
) -> Episode:
    refresh_legacy_metadata(result, validate=False)
    previous = result.get("provenance")
    if (
        previous
        and isinstance(previous, dict)
        and "source" in previous
        and "source_provenance" not in result
    ):
        result["source_provenance"] = copy.deepcopy(previous)
    event = {
        "type": kind,
        "seed": int(seed),
        "parameters": copy.deepcopy(params),
        "affected_indices": indices.cpu().tolist(),
        "missing_indices": ([] if missing_indices is None else missing_indices.cpu().tolist()),
    }
    result.setdefault("corruption_events", []).append(event)
    # Retain the last-event view for older GUI/report consumers.
    result["provenance"] = copy.deepcopy(event)
    validate_episode(result)
    return result


def _camera_indices(episode: Episode, camera_index: int | None) -> list[int]:
    image = episode["observation.image"]
    camera_count = 1 if image.ndim == 4 else int(image.shape[1])
    if camera_index is None:
        return list(range(camera_count))
    if not 0 <= int(camera_index) < camera_count:
        raise ValueError(f"camera_index must be in [0, {camera_count - 1}]")
    return [int(camera_index)]


def _image_views(image: torch.Tensor) -> torch.Tensor:
    return image.unsqueeze(1) if image.ndim == 4 else image


def temporal_shift(
    episode: Episode,
    target: str = "image",
    shift: int = 1,
    boundary: str = "mask",
    seed: int = 0,
    camera_index: int | None = None,
) -> Episode:
    key = {"image": "observation.image", "state": "observation.state", "action": "action"}.get(
        target
    )
    if key is None:
        raise ValueError("target must be image, state, or action")
    if boundary not in {"drop", "repeat", "zero", "mask"}:
        raise ValueError("boundary must be drop, repeat, zero, or mask")
    if target != "image" and camera_index is not None:
        raise ValueError("camera_index is only valid for image temporal shifts")
    result = clone_episode(episode)
    length = validate_episode(result)
    ensure_modal_metadata(result)
    amount = abs(int(shift))
    if amount >= length:
        raise ValueError("Absolute shift must be smaller than episode length")
    cameras = _camera_indices(result, camera_index) if target == "image" else []
    params = {
        "target": target,
        "shift": int(shift),
        "boundary": boundary,
        "camera_indices": cameras,
    }
    if amount == 0:
        return _record(
            result,
            "temporal_shift",
            seed,
            params,
            torch.empty(0, dtype=torch.long),
        )

    source = episode[key]
    shifted_offset = _shift_time_offset(episode["timestamp"], int(shift))
    shifted_quality = quality_from_time_offset(shifted_offset, episode["timestamp"])
    boundary_indices = torch.arange(amount) if shift > 0 else torch.arange(length - amount, length)

    if target == "image":
        destination_view = _image_views(result[key])
        source_view = _image_views(source)
        if shift > 0:
            destination_view[amount:, cameras] = source_view[:-amount, cameras]
            edge = source_view[0, cameras]
        else:
            destination_view[:-amount, cameras] = source_view[amount:, cameras]
            edge = source_view[-1, cameras]
        result["image_time_offset"][:, cameras] += shifted_offset.unsqueeze(1)
        result["image_quality"][:, cameras] *= shifted_quality.unsqueeze(1)
    else:
        if shift > 0:
            result[key][amount:] = source[:-amount]
            edge = source[0]
        else:
            result[key][:-amount] = source[amount:]
            edge = source[-1]
        offset_key = "state_time_offset" if target == "state" else "action_label_time_offset"
        quality_key = "state_quality" if target == "state" else "action_label_quality"
        result[offset_key] += shifted_offset
        result[quality_key] *= shifted_quality

    if boundary == "drop":
        keep = slice(amount, None) if shift > 0 else slice(None, -amount)
        for field, value in list(result.items()):
            if torch.is_tensor(value) and value.ndim and value.shape[0] == length:
                result[field] = value[keep].clone()
        affected = torch.arange(length - amount)
        missing_indices = torch.empty(0, dtype=torch.long)
    else:
        affected = torch.arange(length)
        missing_indices = boundary_indices
        if target == "image":
            destination_view = _image_views(result[key])
            if boundary == "repeat":
                destination_view[boundary_indices[:, None], cameras] = edge
            elif boundary in {"zero", "mask"}:
                destination_view[boundary_indices[:, None], cameras] = 0
                result["image_quality"][boundary_indices[:, None], cameras] = 0.0
            result["image_missing_mask"][boundary_indices[:, None], cameras] = True
        else:
            if boundary == "repeat":
                result[key][boundary_indices] = edge
            elif boundary in {"zero", "mask"}:
                result[key][boundary_indices] = 0
                quality_key = "state_quality" if target == "state" else "action_label_quality"
                result[quality_key][boundary_indices] = 0.0
            missing_key = "state_missing_mask" if target == "state" else "action_label_missing_mask"
            result[missing_key][boundary_indices] = True
    return _record(
        result,
        "temporal_shift",
        seed,
        params,
        affected,
        missing_indices=missing_indices,
    )


def frame_drop(
    episode: Episode,
    probability: float = 0.1,
    replacement: str = "previous",
    seed: int = 0,
    camera_index: int | None = None,
) -> Episode:
    if not 0 <= probability <= 1:
        raise ValueError("probability must be in [0, 1]")
    if replacement not in {"previous", "zero", "interpolation"}:
        raise ValueError("replacement must be previous, zero, or interpolation")
    result = clone_episode(episode)
    length = validate_episode(result)
    cameras = _camera_indices(result, camera_index)
    generator = torch.Generator().manual_seed(seed)
    dropped = torch.rand(length, generator=generator) < probability
    indices = dropped.nonzero(as_tuple=False).flatten()
    images = _image_views(result["observation.image"])
    original = _image_views(episode["observation.image"])
    for index in indices.tolist():
        if replacement == "zero":
            images[index, cameras] = 0
        elif replacement == "previous":
            images[index, cameras] = original[max(0, index - 1), cameras]
        else:
            before, after = max(0, index - 1), min(length - 1, index + 1)
            average = (original[before, cameras].float() + original[after, cameras].float()) / 2
            images[index, cameras] = (
                average.round().to(images.dtype) if not images.is_floating_point() else average
            )
    result["image_missing_mask"][indices[:, None], cameras] = True
    result["image_quality"][indices[:, None], cameras] *= 0.25
    return _record(
        result,
        "frame_drop",
        seed,
        {
            "probability": probability,
            "replacement": replacement,
            "camera_indices": cameras,
        },
        indices,
        missing_indices=indices,
    )


def action_noise(
    episode: Episode,
    mode: str = "gaussian",
    sigma: float = 0.1,
    probability: float = 0.05,
    delay: int = 1,
    random_delay: bool = False,
    seed: int = 0,
) -> Episode:
    if mode not in {"gaussian", "spike", "delay"}:
        raise ValueError("mode must be gaussian, spike, or delay")
    result = clone_episode(episode)
    length = validate_episode(result)
    generator = torch.Generator().manual_seed(seed)
    if mode == "gaussian":
        noise = (
            torch.randn(result["action"].shape, generator=generator, dtype=result["action"].dtype)
            * sigma
        )
        result["action"] += noise
        indices = torch.arange(length)
        result["action_label_quality"][indices] *= 1.0 - min(0.9, abs(sigma))
    elif mode == "spike":
        mask = torch.rand(length, generator=generator) < probability
        indices = mask.nonzero(as_tuple=False).flatten()
        spikes = torch.randn((len(indices), result["action"].shape[1]), generator=generator) * sigma
        result["action"][indices] += spikes
        result["action_label_quality"][indices] *= 0.25
    else:
        actual_delay = (
            int(torch.randint(0, max(1, delay) + 1, (1,), generator=generator).item())
            if random_delay
            else int(delay)
        )
        if actual_delay < 0 or actual_delay >= length:
            raise ValueError("delay must satisfy 0 <= delay < episode length")
        if actual_delay:
            result["action"][actual_delay:] = episode["action"][:-actual_delay]
            result["action"][:actual_delay] = episode["action"][0]
        frame_indices = torch.arange(length, device=episode["timestamp"].device)
        source_indices = (frame_indices - actual_delay).clamp_min(0)
        indices = frame_indices[source_indices != frame_indices]
        delay_offset = episode["timestamp"][source_indices] - episode["timestamp"]
        result["action_label_time_offset"] += delay_offset
        result["action_label_quality"] *= quality_from_time_offset(
            delay_offset, episode["timestamp"]
        )
        delay = actual_delay
    return _record(
        result,
        "action_noise",
        seed,
        {
            "mode": mode,
            "sigma": sigma,
            "probability": probability,
            "delay": delay,
            "random_delay": random_delay,
            "random_delay_scope": "episode" if random_delay else "fixed",
        },
        indices,
    )


def state_anomaly(
    episode: Episode,
    mode: str = "spike",
    probability: float = 0.05,
    magnitude: float = 3.0,
    dimension: int = 0,
    duration: int = 4,
    constant: float = 0.0,
    jitter_std: float = 0.002,
    seed: int = 0,
) -> Episode:
    if mode not in {"spike", "missing", "stuck", "timestamp_jitter"}:
        raise ValueError("Unsupported state anomaly mode")
    result = clone_episode(episode)
    length = validate_episode(result)
    generator = torch.Generator().manual_seed(seed)
    missing_indices = torch.empty(0, dtype=torch.long)
    if mode == "spike":
        mask = torch.rand(length, generator=generator) < probability
        indices = mask.nonzero(as_tuple=False).flatten()
        noise = (
            torch.randn((len(indices), result["observation.state"].shape[1]), generator=generator)
            * magnitude
        )
        result["observation.state"][indices] += noise
    elif mode in {"missing", "stuck"}:
        duration = max(1, min(duration, length))
        start = int(torch.randint(0, length - duration + 1, (1,), generator=generator).item())
        indices = torch.arange(start, start + duration)
        if mode == "missing":
            result["observation.state"][indices] = 0
            result["state_missing_mask"][indices] = True
            missing_indices = indices
        else:
            if not 0 <= dimension < result["observation.state"].shape[1]:
                raise ValueError("dimension is out of range")
            result["observation.state"][indices, dimension] = constant
    else:
        jitter = torch.randn((length,), generator=generator) * jitter_std
        result["timestamp"] += jitter
        result["timestamp"] = torch.cummax(result["timestamp"], dim=0).values
        jitter_offset = result["timestamp"] - episode["timestamp"]
        result["state_time_offset"] += jitter_offset
        result["state_quality"] *= quality_from_time_offset(jitter_offset, episode["timestamp"])
        indices = torch.arange(length)
    if mode != "timestamp_jitter":
        result["state_quality"][indices] *= 0.25
    return _record(
        result,
        "state_anomaly",
        seed,
        {
            "mode": mode,
            "probability": probability,
            "magnitude": magnitude,
            "dimension": dimension,
            "duration": duration,
            "constant": constant,
            "jitter_std": jitter_std,
        },
        indices,
        missing_indices=missing_indices,
    )


def apply_corruption(episode: Episode, config: dict[str, Any]) -> Episode:
    config = dict(config)
    kind = config.pop("type")
    functions = {
        "temporal_shift": temporal_shift,
        "frame_drop": frame_drop,
        "action_noise": action_noise,
        "state_anomaly": state_anomaly,
    }
    if kind not in functions:
        raise ValueError(f"Unknown corruption type: {kind}")
    return functions[kind](episode, **config)
