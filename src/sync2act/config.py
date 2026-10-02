from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

# All entry points resolve their explicit overrides against these defaults.
MODEL_DEFAULTS = {
    "name": "bc_mlp",
    "state_dim": 6,
    "action_dim": 3,
    "input_mode": "image_state",
    "hidden_dim": 64,
    "horizon": 8,
    "num_layers": 1,
    "num_heads": 4,
    "dropout": 0.0,
}
TRAINING_DEFAULTS = {
    "epochs": 3,
    "batch_size": 32,
    "weight_decay": 1e-4,
    "validation_split": 0.15,
    "test_split": 0.15,
    "seed": 7,
    "device": "auto",
    "num_workers": 0,
    "max_steps": None,
    "checkpoint_name": "checkpoint.pt",
}
TRAINING_PROTOCOL_VERSION = 2


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Configuration root must be a mapping")
    return data


def save_config(config: dict[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return target


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def resolve_training_config(
    config: dict[str, Any],
    *,
    model_overrides: dict | None = None,
    training_overrides: dict | None = None,
    source: str = "config",
    override_source: str = "runtime",
) -> dict[str, Any]:
    """Resolve YAML, GUI and study settings without changing the caller's mapping.

    Priority: shared defaults < supplied configuration < explicit overrides.
    Persist provenance alongside the resolved values, including derived values.
    """
    import torch

    result = deepcopy(config)
    supplied_model = result.get("model", {})
    supplied_training = result.get("training", {})
    model = deep_merge(MODEL_DEFAULTS, supplied_model)
    model.update(model_overrides or {})
    name = model["name"]
    is_bc = name == "bc_mlp"
    quality_input = name in {
        "quality_act",
        "quality_act_input",
        "quality_act_full",
        "quality_act_shuffled",
        "quality_act_constant",
        "quality_input",
        "quality_full",
        "quality_shuffled",
        "quality_constant",
    }
    weighted = name in {
        "quality_act",
        "quality_act_weighted",
        "quality_weighted_loss",
        "quality_act_full",
        "quality_act_shuffled",
        "quality_act_constant",
        "quality_full",
        "quality_shuffled",
        "quality_constant",
    }
    model.setdefault("use_quality_features", quality_input)
    if is_bc:
        model["horizon"] = 1
    defaults = {
        **TRAINING_DEFAULTS,
        "training_protocol_version": TRAINING_PROTOCOL_VERSION,
        "learning_rate": 0.003 if is_bc else 0.001,
        "loss": "mse" if is_bc else "smooth_l1",
        "lambda_smooth": 0.0 if is_bc else 0.01,
        "grad_clip": 0.0 if is_bc else 1.0,
        "quality_weighted_loss": weighted,
        "quality_transform": (
            "shuffled"
            if name.endswith("shuffled")
            else "constant"
            if name.endswith("constant")
            else "identity"
        ),
    }
    training = deep_merge(defaults, supplied_training)
    training.update(training_overrides or {})
    if training["device"] == "auto":
        training["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    if not is_bc and "horizon" in training and training["horizon"] != model["horizon"]:
        raise ValueError("training.horizon must equal model.horizon")
    training["horizon"] = model["horizon"]
    training.setdefault("split_seed", training["seed"])
    if training["loss"] not in {"mse", "smooth_l1"}:
        raise ValueError("training.loss must be mse or smooth_l1")
    if training["quality_transform"] not in {"identity", "constant", "shuffled"}:
        raise ValueError("Unknown quality_transform")
    for key in ("epochs", "batch_size", "horizon"):
        if int(training[key]) != training[key] or training[key] < 1:
            raise ValueError(f"training.{key} must be a positive integer")
    for key in ("learning_rate", "weight_decay", "lambda_smooth", "grad_clip"):
        training[key] = float(training[key])
        if not math.isfinite(training[key]) or training[key] < 0:
            raise ValueError(f"training.{key} must be finite and non-negative")
    if int(training["num_workers"]) != training["num_workers"] or training["num_workers"] < 0:
        raise ValueError("num_workers must be a non-negative integer")
    val, test = float(training["validation_split"]), float(training["test_split"])
    if not 0 < val < 1 or not 0 <= test < 1 or val + test >= 1:
        raise ValueError("Require validation_split > 0, test_split >= 0 and sum < 1")
    for key in ("state_dim", "action_dim", "hidden_dim", "num_layers", "num_heads"):
        if int(model[key]) != model[key] or model[key] < 1:
            raise ValueError(f"model.{key} must be a positive integer")
    if not is_bc and model["hidden_dim"] % model["num_heads"]:
        raise ValueError("hidden_dim must be divisible by num_heads")
    sources = deepcopy(config.get("config_sources", supplied_training.get("config_sources", {})))
    for section, values, supplied, overrides in (
        ("model", model, supplied_model, model_overrides or {}),
        ("training", training, supplied_training, training_overrides or {}),
    ):
        for key in values:
            if key in {"model", "config_sources", "dataset_config"}:
                continue
            path = f"{section}.{key}"
            if key in overrides:
                sources[path] = override_source
            elif key in supplied:
                sources.setdefault(path, source)
            else:
                sources.setdefault(path, "shared defaults")
    sources["training.horizon"] = "model.horizon"
    training["training_protocol_version"] = TRAINING_PROTOCOL_VERSION
    training["model"] = model
    training["config_sources"] = sources
    if "dataset" in result:
        training["dataset_config"] = deepcopy(result["dataset"])
    result.update(model=model, training=training, config_sources=sources)
    return result
