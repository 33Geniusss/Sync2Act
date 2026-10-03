from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sync2act import __version__

CHECKPOINT_SCHEMA_VERSION = 3
QUALITY_SCHEMA_VERSION = 2
RESUME_STATE_VERSION = 1


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def model_signature(model: torch.nn.Module) -> str:
    specification = {
        "class": f"{model.__class__.__module__}.{model.__class__.__qualname__}",
        "state": {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in model.state_dict().items()
        },
    }
    return _canonical_hash(specification)


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def checkpoint_metadata(model: torch.nn.Module, config: dict) -> dict:
    return {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "quality_schema_version": QUALITY_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "sync2act_version": __version__,
        "git_commit": _git_commit(),
        "model_class": f"{model.__class__.__module__}.{model.__class__.__qualname__}",
        "model_signature": model_signature(model),
        "config_signature": _canonical_hash(config),
        "experiment_signature": config.get("experiment_signature"),
        "quality_formula": "exp(-abs(modality_time_offset)/(2*median_frame_interval))",
    }


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any,
    step: int,
    epoch: int,
    config: dict,
    stats: dict | None = None,
    history: list[dict] | None = None,
    *,
    resume_state: dict | None = None,
    best: dict | None = None,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": checkpoint_metadata(model, config),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "step": step,
        "epoch": epoch,
        "config": config,
        "stats": stats,
        "history": history or [],
        "checkpoint_kind": "last",
        "resume_state": resume_state,
        "best": best,
    }
    return atomic_save(payload, target)


def atomic_save(payload: dict, target: str | Path) -> Path:
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    try:
        with temporary.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def best_checkpoint_path(last_path: str | Path) -> Path:
    path = Path(last_path)
    return path.with_name(f"{path.stem}.best.pt")


def save_best_checkpoint(path, model, config, stats, best, history) -> Path:
    """Export the selected weights. This artifact is intentionally evaluation-only."""
    return atomic_save(
        {
            "metadata": checkpoint_metadata(model, config),
            "model": best["model"],
            "optimizer": None,
            "scheduler": None,
            "step": best["step"],
            "epoch": best["epoch"],
            "config": config,
            "stats": stats,
            "history": history[: best["epoch"] + 1],
            "checkpoint_kind": "best",
            "selection_metric": "validation_loss",
            "selection_value": best["validation_loss"],
            "resume_state": None,
        },
        path,
    )


def capture_rng_state(device: torch.device) -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
    }


def restore_rng_state(state: dict, device: torch.device) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if device.type == "cuda":
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA device count changed; exact RNG restoration is unavailable")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def inspect_checkpoint(
    path: str | Path,
    *,
    expected_model: torch.nn.Module | None = None,
    experiment_signature: str | None = None,
    require_resume: bool = False,
) -> tuple[bool, str, dict | None]:
    target = Path(path)
    if not target.is_file():
        return False, "checkpoint is missing", None
    try:
        payload = torch.load(target, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, EOFError, ValueError) as error:
        return False, f"checkpoint cannot be read: {error}", None
    metadata = payload.get("metadata") or {}
    if metadata.get("checkpoint_schema_version") not in {2, CHECKPOINT_SCHEMA_VERSION}:
        return False, "checkpoint schema version does not match", payload
    if metadata.get("quality_schema_version") != QUALITY_SCHEMA_VERSION:
        return False, "quality schema version does not match", payload
    if expected_model is not None and metadata.get("model_signature") != model_signature(
        expected_model
    ):
        return False, "model signature does not match", payload
    if (
        experiment_signature is not None
        and metadata.get("experiment_signature") != experiment_signature
    ):
        return False, "experiment signature does not match", payload
    if require_resume and (
        payload.get("checkpoint_kind") != "last"
        or (payload.get("resume_state") or {}).get("version") != RESUME_STATE_VERSION
    ):
        return False, "checkpoint lacks exact-resume state; evaluation only", payload
    return True, "compatible", payload


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    map_location: str | torch.device = "cpu",
) -> dict:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    metadata = payload.get("metadata") or {}
    signature = metadata.get("model_signature")
    if signature is not None and signature != model_signature(model):
        raise ValueError(
            "Checkpoint model signature does not match the requested model architecture"
        )
    try:
        model.load_state_dict(payload["model"])
    except RuntimeError as error:
        raise ValueError(
            f"Checkpoint parameters are incompatible with the model: {error}"
        ) from error
    if optimizer is not None and payload.get("optimizer"):
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler"):
        scheduler.load_state_dict(payload["scheduler"])
    return payload
