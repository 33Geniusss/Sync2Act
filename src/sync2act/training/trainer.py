from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from sync2act.config import resolve_training_config
from sync2act.data.dataset import EpisodeWindowDataset, NormalizationStats
from sync2act.data.episode import Episode

from .checkpoint import load_checkpoint, save_checkpoint
from .losses import EpochLossAccumulator, prediction_loss_sums
from .setup import describe_policy, prepare_training_episodes, seed_everything


@dataclass
class TrainingResult:
    history: list[dict]
    checkpoint: Path
    training_seconds: float
    stopped: bool


def _move_batch(batch: dict, device: torch.device) -> dict:
    moved = {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
    if moved["image"].dtype == torch.uint8:
        moved["image"] = moved["image"].float().div_(255.0)
    return moved


def _loss(model, batch, config, *, return_sums=False):
    prediction = model(
        state=batch["state"],
        image=batch["image"],
        image_quality=batch["image_quality"],
        state_quality=batch["state_quality"],
        image_missing=batch["image_missing"],
        state_missing=batch["state_missing"],
        image_time_offset=batch["image_time_offset"],
        state_time_offset=batch["state_time_offset"],
        missing=batch["missing"],
        time_offset=batch["time_offset"],
    )
    target = batch["actions"][:, : prediction.shape[1]]
    padding = batch["padding_mask"][:, : prediction.shape[1]]
    action_label_quality = batch["action_label_quality"][:, : prediction.shape[1]]
    sums = prediction_loss_sums(prediction, target, padding, action_label_quality, config)
    values = sums.means(float(config.get("lambda_smooth", 0.0)))
    return (*values, sums) if return_sums else values


def train_policy(
    model: torch.nn.Module,
    episodes: list[Episode],
    config: dict,
    output_dir: str | Path = "runs/latest",
    progress: Callable[[dict], None] | None = None,
    stop_event: threading.Event | None = None,
    resume_from: str | Path | None = None,
    validation_episodes: list[Episode] | None = None,
    normalization_stats: NormalizationStats | None = None,
) -> TrainingResult:
    model_settings = config.get("model") or describe_policy(model)
    config = resolve_training_config(
        {"model": model_settings, "training": config}, source="train_policy"
    )["training"]
    seed = int(config["seed"])
    seed_everything(seed)
    device = torch.device(config["device"])
    model.to(device)
    horizon = int(config.get("horizon", getattr(model, "horizon", 1)))
    if resume_from is not None:
        resume_payload = torch.load(Path(resume_from), map_location="cpu", weights_only=False)
        if (
            resume_payload.get("config", {}).get("training_protocol_version")
            != config["training_protocol_version"]
        ):
            raise ValueError("Checkpoint uses a different training protocol; start a new run")
        if normalization_stats is None and resume_payload.get("stats"):
            normalization_stats = NormalizationStats.from_dict(resume_payload["stats"])
    if validation_episodes is None:
        # Low-level callers supply an unsplit dataset. Reserve validation episodes;
        # frontends additionally reserve test episodes before calling this function.
        prepared = prepare_training_episodes(episodes, {**config, "test_split": 0.0})
        episodes, validation_episodes = prepared.train, prepared.validation
        config["episode_split"] = prepared.split
        config["test_split"] = 0.0
        if normalization_stats is None:
            normalization_stats = prepared.stats
    dataset = EpisodeWindowDataset(episodes, horizon=horizon, stats=normalization_stats)
    train_set = dataset
    val_set = EpisodeWindowDataset(validation_episodes, horizon=horizon, stats=dataset.stats)
    loader = DataLoader(
        train_set,
        batch_size=int(config.get("batch_size", 32)),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        pin_memory=device.type == "cuda",
        num_workers=int(config["num_workers"]),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=int(config.get("batch_size", 32)),
        pin_memory=device.type == "cuda",
        num_workers=int(config["num_workers"]),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 1e-3)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, int(config.get("epochs", 3)))
    )
    step, start_epoch, history = 0, 0, []
    if resume_from:
        payload = load_checkpoint(resume_from, model, optimizer, scheduler, device)
        step, start_epoch, history = (
            int(payload["step"]),
            int(payload["epoch"]) + 1,
            list(payload.get("history", [])),
        )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_name = str(config.get("checkpoint_name", "checkpoint.pt"))
    if Path(checkpoint_name).name != checkpoint_name or not checkpoint_name.endswith(".pt"):
        raise ValueError("checkpoint_name must be a .pt filename without a directory")
    checkpoint_path = output / checkpoint_name
    snapshot = {
        "model": config["model"],
        "training": {
            k: v
            for k, v in config.items()
            if k not in {"model", "dataset_config", "config_sources"}
        },
        "dataset": config.get("dataset_config", {}),
        "config_sources": config.get("config_sources", {}),
        "normalization_stats": dataset.stats.to_dict(),
    }
    checkpoint_path.with_suffix(".config.json").write_text(
        json.dumps(snapshot, indent=2), encoding="utf-8"
    )
    started = time.perf_counter()
    stopped = False
    epochs = int(config.get("epochs", 3))
    max_steps = config.get("max_steps")
    for epoch in range(start_epoch, epochs):
        model.train()
        train_totals = EpochLossAccumulator()
        for batch in loader:
            if stop_event and stop_event.is_set():
                stopped = True
                break
            batch = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss, action_loss, smoothness, sums = _loss(model, batch, config, return_sums=True)
            loss.backward()
            if config.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["grad_clip"]))
            optimizer.step()
            step += 1
            train_totals.update(sums)
            event = {
                "epoch": epoch,
                "step": step,
                "train_loss": float(loss.detach()),
                "action_loss": float(action_loss.detach()),
                "smoothness_loss": float(smoothness.detach()),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "eta_seconds": max(
                    0.0,
                    (time.perf_counter() - started) / step * max(0, epochs * len(loader) - step),
                ),
            }
            if progress:
                progress(event)
            if max_steps and step >= int(max_steps):
                stopped = True
                break
        model.eval()
        validation_totals = EpochLossAccumulator()
        with torch.no_grad():
            for batch in val_loader:
                batch = _move_batch(batch, device)
                *_, sums = _loss(model, batch, config, return_sums=True)
                validation_totals.update(sums)
        train_mean, train_action, train_smooth = train_totals.means(config["lambda_smooth"])
        val_mean, val_action, val_smooth = validation_totals.means(config["lambda_smooth"])
        record = {
            "epoch": epoch,
            "step": step,
            "train_loss": train_mean,
            "validation_loss": val_mean,
            "train_action_loss": train_action,
            "train_smoothness_loss": train_smooth,
            "validation_action_loss": val_action,
            "validation_smoothness_loss": val_smooth,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        scheduler.step()
        save_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            scheduler,
            step,
            epoch,
            config,
            dataset.stats.to_dict(),
            history,
        )
        if progress:
            progress({**record, "checkpoint": str(checkpoint_path), "epoch_complete": True})
        if stopped:
            break
    return TrainingResult(history, checkpoint_path, time.perf_counter() - started, stopped)
