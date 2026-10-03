"""Resume compatibility checks for the deterministic episode-window training path."""

from __future__ import annotations

import hashlib
import json

import torch

from sync2act.data.episode import REQUIRED_KEYS

from .checkpoint import CHECKPOINT_SCHEMA_VERSION, RESUME_STATE_VERSION, model_signature


def training_data_signature(train_episodes, validation_episodes) -> str:
    """Hash tensor content and episode order, including images and quality annotations.

    A memoryview avoids allocating a second bytes copy of large image tensors.
    Called once per invocation, never on every checkpoint write.
    """
    digest = hashlib.sha256()
    for label, episodes in (("train", train_episodes), ("validation", validation_episodes)):
        digest.update(f"{label}:{len(episodes)}".encode())
        for episode in episodes:
            for key in sorted(REQUIRED_KEYS):
                tensor = episode[key].detach().cpu().contiguous()
                digest.update(json.dumps([key, str(tensor.dtype), list(tensor.shape)]).encode())
                digest.update(memoryview(tensor.view(torch.uint8).numpy()).cast("B"))
    return digest.hexdigest()


def validate_resume(payload, model, config, stats, data_signature, batches_per_epoch) -> dict:
    if (
        payload.get("config", {}).get("training_protocol_version")
        != config["training_protocol_version"]
    ):
        raise ValueError("Checkpoint uses a different training protocol; start a new run")
    state = payload.get("resume_state") or {}
    if (
        payload.get("metadata", {}).get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION
        or state.get("version") != RESUME_STATE_VERSION
        or payload.get("checkpoint_kind") != "last"
    ):
        raise ValueError(
            "Checkpoint is evaluation-only or lacks exact-resume state; start a new run"
        )
    if payload["metadata"]["model_signature"] != model_signature(model):
        raise ValueError("Checkpoint model architecture changed")
    required = {
        "epoch",
        "next_batch",
        "batches_per_epoch",
        "order",
        "sampler_rng_state",
        "loader_rng_state",
        "rng_state",
        "train_loss_sums",
        "data_signature",
        "training_seconds",
    }
    if not required <= state.keys() or not payload.get("optimizer") or not payload.get("scheduler"):
        raise ValueError("Checkpoint is missing optimizer, scheduler, or resume progress")
    previous = payload["config"]
    critical = (
        "model",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "loss",
        "lambda_smooth",
        "grad_clip",
        "seed",
        "split_seed",
        "horizon",
        "quality_weighted_loss",
        "quality_transform",
        "episode_split",
        "device",
        "training_protocol_version",
    )
    changed = [key for key in critical if previous.get(key) != config.get(key)]
    if changed:
        raise ValueError(f"Cannot resume with changed training settings: {', '.join(changed)}")
    if payload["stats"] != stats:
        raise ValueError("Normalization statistics changed; cannot resume this run")
    if state["data_signature"] != data_signature:
        raise ValueError("Training or validation data changed; cannot resume this run")
    if state["batches_per_epoch"] != batches_per_epoch:
        raise ValueError("Number of training batches changed")
    if not 0 <= state["epoch"] <= previous["epochs"]:
        raise ValueError("Invalid checkpoint epoch cursor")
    if not 0 <= state["next_batch"] <= batches_per_epoch:
        raise ValueError("Invalid checkpoint batch cursor")
    if config["epochs"] != previous["epochs"]:
        if config["resume_schedule"] != "restart":
            raise ValueError(
                "epochs changed; use resume_schedule='restart' at a completed epoch boundary"
            )
        if state["next_batch"] != 0 or config["epochs"] <= previous["epochs"]:
            raise ValueError(
                "Schedule restart requires an epoch boundary and a larger epochs target"
            )
    elif config["resume_schedule"] == "restart":
        raise ValueError("Schedule restart requires a larger epochs target")
    if config.get("max_steps") and config["max_steps"] <= payload["step"]:
        raise ValueError("max_steps must exceed the saved global step; clear it to continue")
    if len(payload["history"]) != state["epoch"]:
        raise ValueError("Checkpoint history and epoch cursor disagree")
    if payload["step"] != state["epoch"] * batches_per_epoch + state["next_batch"]:
        raise ValueError("Checkpoint optimizer step and batch cursor disagree")
    return state
