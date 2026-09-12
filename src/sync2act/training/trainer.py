from __future__ import annotations

import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split

from sync2act.data.dataset import EpisodeWindowDataset, NormalizationStats
from sync2act.data.episode import Episode
from sync2act.policies.quality_act import quality_weighted_action_loss

from .checkpoint import load_checkpoint, save_checkpoint


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


def _loss(model, batch, config):
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
    use_weighting = config.get("quality_weighted_loss", False)
    if use_weighting:
        action_loss = quality_weighted_action_loss(
            prediction,
            target,
            action_label_quality,
            padding,
            config.get("loss", "mse"),
        )
    else:
        valid = (~padding).unsqueeze(-1).expand_as(prediction)
        per_element = (
            F.mse_loss(prediction, target, reduction="none")
            if config.get("loss", "mse") == "mse"
            else F.smooth_l1_loss(prediction, target, reduction="none")
        )
        action_loss = per_element[valid].mean() if valid.any() else prediction.sum() * 0
    if prediction.shape[1] > 1:
        smoothness = (prediction[:, 1:] - prediction[:, :-1]).square().mean()
    else:
        smoothness = prediction.sum() * 0
    total = action_loss + float(config.get("lambda_smooth", 0.0)) * smoothness
    return total, action_loss, smoothness


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
    seed = int(config.get("seed", 7))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    horizon = int(config.get("horizon", getattr(model, "horizon", 1)))
    if resume_from is not None and normalization_stats is None:
        resume_payload = torch.load(Path(resume_from), map_location="cpu", weights_only=False)
        if resume_payload.get("stats"):
            normalization_stats = NormalizationStats.from_dict(resume_payload["stats"])
    dataset = EpisodeWindowDataset(episodes, horizon=horizon, stats=normalization_stats)
    if validation_episodes is None:
        val_size = max(1, int(len(dataset) * float(config.get("validation_split", 0.15))))
        train_size = len(dataset) - val_size
        train_set, val_set = random_split(
            dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed)
        )
    else:
        train_set = dataset
        val_set = EpisodeWindowDataset(validation_episodes, horizon=horizon, stats=dataset.stats)
    loader = DataLoader(
        train_set,
        batch_size=int(config.get("batch_size", 32)),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_set,
        batch_size=int(config.get("batch_size", 32)),
        pin_memory=device.type == "cuda",
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
    started = time.perf_counter()
    stopped = False
    epochs = int(config.get("epochs", 3))
    max_steps = config.get("max_steps")
    for epoch in range(start_epoch, epochs):
        model.train()
        train_total = 0.0
        batches = 0
        for batch in loader:
            if stop_event and stop_event.is_set():
                stopped = True
                break
            batch = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss, action_loss, smoothness = _loss(model, batch, config)
            loss.backward()
            if config.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["grad_clip"]))
            optimizer.step()
            step += 1
            batches += 1
            train_total += float(loss.detach())
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
        validation = []
        with torch.no_grad():
            for batch in val_loader:
                batch = _move_batch(batch, device)
                validation.append(float(_loss(model, batch, config)[0]))
        record = {
            "epoch": epoch,
            "step": step,
            "train_loss": train_total / max(1, batches),
            "validation_loss": sum(validation) / max(1, len(validation)),
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
