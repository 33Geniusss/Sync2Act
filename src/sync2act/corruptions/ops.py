from __future__ import annotations

from typing import Any

import torch

from sync2act.data.episode import Episode, clone_episode, validate_episode


def _record(
    result: Episode, kind: str, seed: int, params: dict[str, Any], indices: torch.Tensor
) -> Episode:
    result["provenance"] = {
        "type": kind,
        "seed": seed,
        "parameters": params,
        "affected_indices": indices.cpu().tolist(),
    }
    validate_episode(result)
    return result


def temporal_shift(
    episode: Episode,
    target: str = "image",
    shift: int = 1,
    boundary: str = "mask",
    seed: int = 0,
) -> Episode:
    key = {"image": "observation.image", "state": "observation.state", "action": "action"}.get(
        target
    )
    if key is None:
        raise ValueError("target must be image, state, or action")
    if boundary not in {"drop", "repeat", "zero", "mask"}:
        raise ValueError("boundary must be drop, repeat, zero, or mask")
    result = clone_episode(episode)
    length = validate_episode(result)
    amount = abs(int(shift))
    if amount >= length:
        raise ValueError("Absolute shift must be smaller than episode length")
    if amount == 0:
        return _record(
            result,
            "temporal_shift",
            seed,
            {"target": target, "shift": 0, "boundary": boundary},
            torch.empty(0, dtype=torch.long),
        )
    source = episode[key]
    if shift > 0:
        result[key][amount:] = source[:-amount]
        affected = torch.arange(amount)
        edge_value = source[0]
    else:
        result[key][:-amount] = source[amount:]
        affected = torch.arange(length - amount, length)
        edge_value = source[-1]
    if boundary == "drop":
        keep = slice(amount, None) if shift > 0 else slice(None, -amount)
        for field in list(result):
            if (
                torch.is_tensor(result[field])
                and result[field].ndim
                and result[field].shape[0] == length
            ):
                result[field] = result[field][keep].clone()
        affected = torch.arange(length - amount)
        result["quality_score"] *= 0.5
    else:
        shifted_indices = torch.arange(length)
        result["quality_score"][shifted_indices] *= 0.5
        if boundary == "repeat":
            result[key][affected] = edge_value
        elif boundary in {"zero", "mask"}:
            result[key][affected] = 0
        result["missing_mask"][affected] = True
        if boundary in {"zero", "mask"}:
            result["quality_score"][affected] = 0.0
        affected = shifted_indices
    return _record(
        result,
        "temporal_shift",
        seed,
        {"target": target, "shift": shift, "boundary": boundary},
        affected,
    )


def frame_drop(
    episode: Episode,
    probability: float = 0.1,
    replacement: str = "previous",
    seed: int = 0,
) -> Episode:
    if not 0 <= probability <= 1:
        raise ValueError("probability must be in [0, 1]")
    if replacement not in {"previous", "zero", "interpolation"}:
        raise ValueError("replacement must be previous, zero, or interpolation")
    result = clone_episode(episode)
    length = validate_episode(result)
    generator = torch.Generator().manual_seed(seed)
    dropped = torch.rand(length, generator=generator) < probability
    indices = dropped.nonzero(as_tuple=False).flatten()
    images = result["observation.image"]
    original = episode["observation.image"]
    for index in indices.tolist():
        if replacement == "zero":
            images[index].zero_()
        elif replacement == "previous":
            images[index] = original[max(0, index - 1)]
        else:
            before, after = max(0, index - 1), min(length - 1, index + 1)
            images[index] = (original[before] + original[after]) / 2
    result["missing_mask"][dropped] = True
    result["quality_score"][dropped] *= 0.25
    return _record(
        result,
        "frame_drop",
        seed,
        {"probability": probability, "replacement": replacement},
        indices,
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
        penalty = min(0.9, abs(sigma))
    elif mode == "spike":
        mask = torch.rand(length, generator=generator) < probability
        indices = mask.nonzero(as_tuple=False).flatten()
        spikes = torch.randn((len(indices), result["action"].shape[1]), generator=generator) * sigma
        result["action"][indices] += spikes
        penalty = 0.75
    else:
        if random_delay:
            actual_delay = int(
                torch.randint(0, max(1, delay) + 1, (1,), generator=generator).item()
            )
        else:
            actual_delay = int(delay)
        if actual_delay < 0 or actual_delay >= length:
            raise ValueError("delay must satisfy 0 <= delay < episode length")
        if actual_delay:
            result["action"][actual_delay:] = episode["action"][:-actual_delay]
            result["action"][:actual_delay] = episode["action"][0]
        indices = torch.arange(actual_delay, dtype=torch.long)
        delay = actual_delay
        penalty = 0.5
    result["quality_score"][indices] *= 1.0 - penalty
    params = {
        "mode": mode,
        "sigma": sigma,
        "probability": probability,
        "delay": delay,
        "random_delay": random_delay,
    }
    return _record(result, "action_noise", seed, params, indices)


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
            result["missing_mask"][indices] = True
        else:
            if not 0 <= dimension < result["observation.state"].shape[1]:
                raise ValueError("dimension is out of range")
            result["observation.state"][indices, dimension] = constant
    else:
        jitter = torch.randn((length,), generator=generator) * jitter_std
        result["timestamp"] += jitter
        result["timestamp"] = torch.cummax(result["timestamp"], dim=0).values
        indices = torch.arange(length)
    result["quality_score"][indices] *= 0.25
    params = {
        "mode": mode,
        "probability": probability,
        "magnitude": magnitude,
        "dimension": dimension,
        "duration": duration,
        "constant": constant,
        "jitter_std": jitter_std,
    }
    return _record(result, "state_anomaly", seed, params, indices)


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
