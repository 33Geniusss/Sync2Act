from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path


def installation_root() -> Path:
    """Return the folder that owns the installed app or the source checkout."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def datasets_root() -> Path:
    target = Path.home() / "Sync2Act" / "datasets"
    target.mkdir(parents=True, exist_ok=True)
    return target


def huggingface_cache_root() -> Path:
    """Return the app-owned Hub cache, kept outside individual datasets."""
    target = Path.home() / "Sync2Act" / ".cache" / "huggingface"
    target.mkdir(parents=True, exist_ok=True)
    return target


def models_root() -> Path:
    target = installation_root() / "models"
    target.mkdir(parents=True, exist_ok=True)
    return target


def safe_name(value: str, fallback: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._")
    return name or fallback


def default_dataset_path(repo_id: str) -> Path:
    dataset_name = safe_name(repo_id.rstrip("/").rsplit("/", 1)[-1], "dataset")
    return datasets_root() / dataset_name if repo_id.strip() else datasets_root()


def new_model_checkpoint_path(
    model_name: str,
    data_condition: str,
    *,
    now: datetime | None = None,
) -> Path:
    timestamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    stem = "_".join(
        (
            safe_name(model_name, "model"),
            safe_name(data_condition, "clean"),
            timestamp,
        )
    )
    root = models_root()
    candidate = root / f"{stem}.pt"
    suffix = 2
    while candidate.exists():
        candidate = root / f"{stem}_{suffix:02d}.pt"
        suffix += 1
    return candidate
