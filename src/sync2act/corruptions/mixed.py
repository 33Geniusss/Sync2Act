from __future__ import annotations

import copy
from collections import Counter
from typing import Any

import torch

from sync2act.data.episode import (
    Episode,
    clone_episode,
    ensure_modal_metadata,
    refresh_legacy_metadata,
    validate_episode,
)

from .ops import apply_corruption

_MERGED_TENSOR_FIELDS = {
    "observation.image",
    "observation.state",
    "action",
    "timestamp",
    "image_quality",
    "state_quality",
    "action_label_quality",
    "image_time_offset",
    "state_time_offset",
    "action_label_time_offset",
    "image_missing_mask",
    "state_missing_mask",
    "action_label_missing_mask",
}


def _component_schedule(
    segment_count: int, components: list[dict[str, Any]], generator: torch.Generator
) -> list[int]:
    weights = torch.tensor([float(item.get("weight", 1.0)) for item in components])
    if (weights < 0).any() or float(weights.sum()) <= 0:
        raise ValueError("Mixture component weights must be non-negative with a positive sum")
    raw = weights / weights.sum() * segment_count
    counts = torch.floor(raw).to(torch.long)
    remaining = segment_count - int(counts.sum())
    if remaining:
        order = torch.argsort(raw - counts, descending=True)
        counts[order[:remaining]] += 1
    schedule = torch.cat(
        [torch.full((int(count),), index, dtype=torch.long) for index, count in enumerate(counts)]
    )
    if segment_count > 1:
        schedule = schedule[torch.randperm(segment_count, generator=generator)]
    return schedule.tolist()


def build_mixed_quality_dataset(
    episodes: list[Episode],
    components: list[dict[str, Any]],
    *,
    segment_length: int = 16,
    seed: int = 7,
) -> tuple[list[Episode], dict[str, Any]]:
    """Construct one training set containing clean and differently corrupted segments.

    Components have ``name``, ``weight`` and either a corruption ``config`` or ``None`` for
    clean data. Segment assignments are shuffled globally but allocated with deterministic,
    near-exact proportions.
    """
    if not episodes:
        raise ValueError("At least one episode is required")
    if not components:
        raise ValueError("At least one mixture component is required")
    if segment_length < 1:
        raise ValueError("segment_length must be positive")
    names = [str(item["name"]) for item in components]
    if len(set(names)) != len(names):
        raise ValueError("Mixture component names must be unique")
    for item in components:
        config = item.get("config")
        if config is not None and "type" not in config:
            raise ValueError(f"Mixture component {item['name']} has no corruption type")

    segments: list[tuple[int, int, int]] = []
    for episode_index, episode in enumerate(episodes):
        length = validate_episode(episode)
        segments.extend(
            (episode_index, start, min(length, start + segment_length))
            for start in range(0, length, segment_length)
        )
    generator = torch.Generator().manual_seed(seed)
    schedule = _component_schedule(len(segments), components, generator)
    assignments: dict[int, list[tuple[int, int, int]]] = {}
    for (episode_index, start, end), component_index in zip(segments, schedule, strict=True):
        assignments.setdefault(episode_index, []).append((start, end, component_index))

    mixed: list[Episode] = []
    frame_counts: Counter[str] = Counter()
    segment_counts: Counter[str] = Counter()
    for episode_index, episode in enumerate(episodes):
        destination = clone_episode(episode)
        ensure_modal_metadata(destination)
        destination["corruption_events"] = copy.deepcopy(destination.get("corruption_events", []))
        cached: dict[int, Episode] = {}
        for start, end, component_index in assignments.get(episode_index, []):
            component = components[component_index]
            name = str(component["name"])
            frame_counts[name] += end - start
            segment_counts[name] += 1
            config = component.get("config")
            if config is None:
                continue
            if component_index not in cached:
                cached[component_index] = apply_corruption(
                    episode,
                    {
                        **copy.deepcopy(config),
                        "seed": seed + episode_index * 1009 + component_index * 97,
                    },
                )
            source = cached[component_index]
            for field in _MERGED_TENSOR_FIELDS:
                destination[field][start:end] = source[field][start:end]
            source_event = copy.deepcopy(source["corruption_events"][-1])
            source_event["mixture_component"] = name
            source_event["segment"] = [start, end]
            source_event["affected_indices"] = [
                index for index in source_event["affected_indices"] if start <= index < end
            ]
            source_event["missing_indices"] = [
                index for index in source_event.get("missing_indices", []) if start <= index < end
            ]
            destination["corruption_events"].append(source_event)
            destination["provenance"] = copy.deepcopy(source_event)
        destination["mixture"] = {
            "seed": seed,
            "segment_length": segment_length,
            "assignments": [
                {
                    "start": start,
                    "end": end,
                    "component": names[component_index],
                }
                for start, end, component_index in assignments.get(episode_index, [])
            ],
        }
        refresh_legacy_metadata(destination)
        mixed.append(destination)

    manifest = {
        "schema_version": 1,
        "seed": seed,
        "segment_length": segment_length,
        "components": copy.deepcopy(components),
        "episode_count": len(episodes),
        "segment_count": len(segments),
        "segment_counts": dict(segment_counts),
        "frame_counts": dict(frame_counts),
    }
    return mixed, manifest


def transform_quality_annotations(
    episodes: list[Episode], mode: str, *, seed: int = 7
) -> list[Episode]:
    """Create constant or shuffled quality controls without changing sensor/action data."""
    if mode not in {"identity", "constant", "shuffled"}:
        raise ValueError("quality annotation mode must be identity, constant, or shuffled")
    fields = [
        "image_quality",
        "state_quality",
        "action_label_quality",
        "image_time_offset",
        "state_time_offset",
        "action_label_time_offset",
        "image_missing_mask",
        "state_missing_mask",
        "action_label_missing_mask",
    ]
    metadata_fields = set(fields) | {"quality_score", "time_offset", "missing_mask"}
    transformed = []
    for source in episodes:
        ensure_modal_metadata(source)
        transformed.append(
            {
                key: (
                    value.clone()
                    if torch.is_tensor(value) and key in metadata_fields
                    else copy.deepcopy(value)
                    if not torch.is_tensor(value)
                    else value
                )
                for key, value in source.items()
            }
        )
    if mode == "identity":
        return transformed
    if mode == "constant":
        for episode in transformed:
            for field in fields:
                value = episode[field]
                if "quality" in field:
                    value.fill_(1)
                else:
                    value.zero_()
            episode["quality_transform"] = {"mode": mode, "seed": seed}
            refresh_legacy_metadata(episode)
        return transformed

    lengths = [validate_episode(episode) for episode in transformed]
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(sum(lengths), generator=generator)
    for field in fields:
        combined = torch.cat([episode[field] for episode in transformed], dim=0)
        shuffled = combined[permutation]
        cursor = 0
        for episode, length in zip(transformed, lengths, strict=True):
            episode[field] = shuffled[cursor : cursor + length].clone()
            cursor += length
    for episode in transformed:
        episode["quality_transform"] = {"mode": mode, "seed": seed}
        refresh_legacy_metadata(episode)
    return transformed
