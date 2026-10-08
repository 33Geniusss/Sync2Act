"""Paired offline observation-corruption evaluation shared by GUI and studies."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from sync2act.corruptions import build_mixed_quality_dataset
from sync2act.data.dataset import NormalizationStats
from sync2act.data.episode import (
    Episode,
    clone_episode,
    ensure_modal_metadata,
    refresh_legacy_metadata,
)

from .evaluator import evaluate_policy


@dataclass(frozen=True)
class TestCorruptionConfig:
    __test__ = False
    condition: str = "clean"
    shift: int = 2
    delay_fraction: float = 0.4
    missing_fraction: float = 0.1
    segment_length: int = 16
    seed: int = 107
    camera_index: int | None = None
    quality_mode: str = "oracle"

    def validate(self):
        if self.condition not in {"clean", "image", "state", "mixed"}:
            raise ValueError("Test condition must be clean, image, state, or mixed")
        if self.quality_mode not in {"oracle", "unknown"}:
            raise ValueError("Quality mode must be oracle or unknown")
        for value in (self.delay_fraction, self.missing_fraction):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Segment fractions must be finite and between 0 and 1")
        if self.delay_fraction + self.missing_fraction > 1 + 1e-12:
            raise ValueError("Delay and missing fractions must sum to at most 1")
        if not isinstance(self.shift, int) or self.shift < 1:
            raise ValueError("Delay must be a positive integer; future frames are forbidden")
        if not isinstance(self.segment_length, int) or self.segment_length < 1:
            raise ValueError("Segment length must be a positive integer")
        if self.camera_index is not None and (
            not isinstance(self.camera_index, int) or self.camera_index < 0
        ):
            raise ValueError("Camera index must be non-negative or None for all cameras")
        if not isinstance(self.seed, int) or not 0 <= self.seed < 2**32:
            raise ValueError("Seed must be an integer in [0, 2**32)")


def preset_config(condition: str, *, severity: str = "medium", **kwargs) -> TestCorruptionConfig:
    levels = {
        "light": (1, 0.2, 0.05),
        "medium": (2, 0.4, 0.1),
        "heavy": (4, 0.6, 0.2),
    }
    if severity not in levels:
        raise ValueError("Severity must be light, medium, or heavy")
    shift, delay, missing = levels[severity]
    if condition == "mixed":
        delay, missing = {"light": (0.3, 0.1), "medium": (0.6, 0.2), "heavy": (0.7, 0.3)}[severity]
    return TestCorruptionConfig(
        **{"condition": condition, "shift": shift, "delay_fraction": delay,
           "missing_fraction": missing, **kwargs}
    )


@dataclass
class PreparedTestCondition:
    episodes: list[Episode]
    manifest: dict
    affected: torch.Tensor


def prepare_test_condition(
    episodes: list[Episode], config: TestCorruptionConfig
) -> PreparedTestCondition:
    """Clone held-out episodes; never change labels, timestamps or episode order.

    Ratios allocate disjoint segments, not independent per-frame Bernoulli masks.
    Positive shifts read only past frames from the same episode, with masked edges.
    """
    config.validate()
    if not episodes:
        raise ValueError("A non-empty held-out test partition is required")
    clean = [ensure_modal_metadata(clone_episode(episode)) for episode in episodes]
    components = [{"name": "clean", "weight": max(0.0, 1 - config.delay_fraction - config.missing_fraction), "config": None}]
    targets = ["image", "state"] if config.condition == "mixed" else [config.condition]
    if config.condition == "clean":
        damaged = clean
        mixture = {"frame_counts": {"clean": sum(len(e["action"]) for e in clean)}}
    else:
        for target in targets:
            camera = {"camera_index": config.camera_index} if target == "image" else {}
            for kind, weight, operation in (
                ("delay", config.delay_fraction, {"type": "temporal_shift", "shift": config.shift, "boundary": "mask"}),
                ("missing", config.missing_fraction, {"type": "modality_missing", "probability": 1.0}),
            ):
                if weight:
                    components.append({"name": f"{target}_{kind}", "weight": weight / len(targets),
                                       "config": {**operation, "target": target, **camera}})
        damaged, mixture = build_mixed_quality_dataset(
            clean, components, segment_length=config.segment_length, seed=config.seed
        )
    affected_masks, records = [], []
    for index, (original, modified) in enumerate(zip(clean, damaged, strict=True)):
        for key in ("action", "timestamp", "action_label_quality", "action_label_time_offset", "action_label_missing_mask"):
            if not torch.equal(original[key], modified[key]):
                raise RuntimeError(f"Test corruption altered protected field: {key}")
        mask = torch.zeros(len(original["action"]), dtype=torch.bool)
        for assignment in modified.get("mixture", {}).get("assignments", []):
            if assignment["component"] != "clean":
                mask[assignment["start"]:assignment["end"]] = True
        affected_masks.append(mask)
        records.append({
            "episode": index, "frames": len(mask), "affected_frames": int(mask.sum()),
            "assignments": modified.get("mixture", {}).get("assignments", []),
            "events": modified.get("corruption_events", []),
            "image_delay_ms_max": float(modified["image_time_offset"].abs().max()) * 1000,
            "state_delay_ms_max": float(modified["state_time_offset"].abs().max()) * 1000,
        })
        if config.quality_mode == "unknown":
            for modality in ("image", "state"):
                modified[f"{modality}_quality"].fill_(1)
                modified[f"{modality}_missing_mask"].zero_()
                modified[f"{modality}_time_offset"].zero_()
            refresh_legacy_metadata(modified)
    affected = torch.cat(affected_masks)
    manifest = {
        "schema_version": 1, "config": asdict(config), "scope": "offline-only",
        "labels": "unchanged original test actions", "causal": True,
        "quality_information": "known synthetic corruption (Oracle)" if config.quality_mode == "oracle" else "all observation metadata set to normal",
        "mixture": mixture, "episodes": records, "frames": len(affected),
        "affected_frames": int(affected.sum()), "actual_affected_fraction": float(affected.float().mean()),
    }
    return PreparedTestCondition(damaged, manifest, affected)


def write_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def evaluate_test_condition(
    model, prepared: PreparedTestCondition, *, stats: NormalizationStats,
    output_dir: str | Path, clean_metrics: dict | None = None, device="cpu",
    batch_size=32, stop_event=None, progress=None,
) -> dict:
    if stats is None:
        raise ValueError("Frozen training normalization statistics are required")
    output = Path(output_dir)
    metrics = evaluate_policy(
        model, prepared.episodes, device=device, batch_size=batch_size, stats=stats,
        output_dir=output, frame_groups={"affected": prepared.affected, "unaffected": ~prepared.affected},
        stop_event=stop_event, progress=progress,
        vectorized_batches=True,
    )
    if clean_metrics is not None:
        baseline = clean_metrics["action_mse"]
        metrics["mse_ratio_to_clean_test"] = metrics["action_mse"] / baseline if baseline > 0 else None
        metrics["mse_delta_from_clean_test"] = metrics["action_mse"] - baseline
    write_json(output / "corruption.json", prepared.manifest)
    write_json(output / "metrics.json", metrics)
    return metrics


def evaluate_paired(
    model, episodes, config: TestCorruptionConfig, *, stats: NormalizationStats,
    output_dir: str | Path, device="cpu", batch_size=32, stop_event=None, progress=None,
    provenance: dict | None = None,
) -> dict:
    """Always recompute a clean baseline with the same weights and frozen statistics."""
    if stats is None:
        raise ValueError("Frozen training normalization statistics are required")
    config.validate()
    output = Path(output_dir)
    common = {"stats": stats, "device": device, "batch_size": batch_size,
              "stop_event": stop_event, "progress": progress}
    clean = evaluate_test_condition(model, prepare_test_condition(episodes, TestCorruptionConfig()),
                                    output_dir=output / "clean", **common)
    prepared = prepare_test_condition(episodes, config)
    tested = evaluate_test_condition(model, prepared, output_dir=output / "test",
                                     clean_metrics=clean, **common)
    result = {"schema_version": 1, "clean": clean, "test": tested,
              "test_config": asdict(config), "test_manifest": prepared.manifest,
              "normalization": stats.to_dict(), "provenance": provenance or {},
              "output_dir": str(output.resolve())}
    write_json(output / "comparison.json", result)
    return result
