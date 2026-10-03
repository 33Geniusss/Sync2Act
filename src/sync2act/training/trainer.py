from __future__ import annotations

import json
import math
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

from .checkpoint import (
    RESUME_STATE_VERSION,
    best_checkpoint_path,
    capture_rng_state,
    load_checkpoint,
    restore_rng_state,
    save_best_checkpoint,
    save_checkpoint,
)
from .losses import EpochLossAccumulator, prediction_loss_sums
from .resume import training_data_signature, validate_resume
from .setup import describe_policy, prepare_training_episodes, seed_everything


@dataclass
class TrainingResult:
    history: list[dict]
    checkpoint: Path
    training_seconds: float
    stopped: bool
    best_checkpoint: Path | None = None
    best_validation_loss: float | None = None

    @property
    def evaluation_checkpoint(self) -> Path:
        return self.best_checkpoint or self.checkpoint


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
    """Train with resumable batch boundaries; leave model at the latest weights.

    Best weights are exported separately and selected by complete-epoch validation
    loss. Resume requires unchanged data and optimization settings. The built-in
    EpisodeWindowDataset has deterministic fetching, including with worker prefetch.
    """
    model_settings = config.get("model") or describe_policy(model)
    config = resolve_training_config(
        {"model": model_settings, "training": config}, source="train_policy"
    )["training"]
    seed = int(config["seed"])
    seed_everything(seed)
    device = torch.device(config["device"])
    horizon = int(config["horizon"])
    resume_payload = None
    if resume_from is not None:
        resume_payload = torch.load(Path(resume_from), map_location="cpu", weights_only=False)
        if normalization_stats is None and resume_payload.get("stats"):
            normalization_stats = NormalizationStats.from_dict(resume_payload["stats"])
    if validation_episodes is None:
        prepared = prepare_training_episodes(episodes, {**config, "test_split": 0.0})
        episodes, validation_episodes = prepared.train, prepared.validation
        config["episode_split"] = prepared.split
        config["test_split"] = 0.0
        if normalization_stats is None:
            normalization_stats = prepared.stats
    dataset = EpisodeWindowDataset(episodes, horizon=horizon, stats=normalization_stats)
    val_set = EpisodeWindowDataset(validation_episodes, horizon=horizon, stats=dataset.stats)
    stats = dataset.stats.to_dict()
    batch_size = int(config["batch_size"])
    batches_per_epoch = math.ceil(len(dataset) / batch_size)
    data_signature = training_data_signature(episodes, validation_episodes)
    state = None
    if resume_payload is not None:
        state = validate_resume(
            resume_payload, model, config, stats, data_signature, batches_per_epoch
        )

    output = Path(output_dir)
    checkpoint_name = str(config["checkpoint_name"])
    if (
        Path(checkpoint_name).name != checkpoint_name
        or not checkpoint_name.endswith(".pt")
        or checkpoint_name.endswith(".best.pt")
    ):
        raise ValueError("checkpoint_name must be a latest .pt filename, not a .best.pt file")
    checkpoint_path = output / checkpoint_name
    best_path = best_checkpoint_path(checkpoint_path)
    if resume_from is None and (checkpoint_path.exists() or best_path.exists()):
        raise FileExistsError("Checkpoint already exists; use resume_from or a new output/name")
    if resume_from is not None and checkpoint_path.resolve() != Path(resume_from).resolve():
        if checkpoint_path.exists() or best_path.exists():
            raise FileExistsError("Resume destination already contains another checkpoint")

    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    epochs = int(config["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    sampler_generator = torch.Generator().manual_seed(seed)
    epoch, next_batch, step = 0, 0, 0
    history, best = [], None
    order, loader_rng_state = None, None
    train_totals = EpochLossAccumulator()
    elapsed_before = 0.0
    config["scheduler_plan"] = {
        "start_epoch": 0,
        "T_max": epochs,
        "initial_lr": config["learning_rate"],
    }
    if state is not None:
        load_checkpoint(resume_from, model, optimizer, scheduler, device)
        epoch, next_batch, step = (
            int(state["epoch"]),
            int(state["next_batch"]),
            int(resume_payload["step"]),
        )
        history, best = list(resume_payload["history"]), resume_payload.get("best")
        order, loader_rng_state = state["order"], state["loader_rng_state"]
        if order is not None and (
            order.numel() != len(dataset)
            or not torch.equal(order.sort().values, torch.arange(len(dataset)))
        ):
            raise ValueError("Checkpoint sample permutation does not match this dataset")
        if next_batch and order is None:
            raise ValueError("Partial epoch is missing its sample permutation")
        sampler_generator.set_state(state["sampler_rng_state"].cpu())
        train_totals.load_state_dict(state["train_loss_sums"], device)
        elapsed_before = float(state["training_seconds"])
        config["scheduler_plan"] = resume_payload["config"]["scheduler_plan"]
        if config["resume_schedule"] == "restart":
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] = float(config["learning_rate"])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - epoch)
            config["scheduler_plan"] = {
                "start_epoch": epoch,
                "T_max": epochs - epoch,
                "initial_lr": config["learning_rate"],
            }
        # A schedule restart is a one-time request; subsequent resumes are strict.
        config["resume_schedule"] = "strict"
        restore_rng_state(state["rng_state"], device)

    output.mkdir(parents=True, exist_ok=True)
    snapshot = {
        "model": config["model"],
        "training": {
            k: v
            for k, v in config.items()
            if k not in {"model", "dataset_config", "config_sources"}
        },
        "dataset": config.get("dataset_config", {}),
        "config_sources": config.get("config_sources", {}),
        "normalization_stats": stats,
    }
    checkpoint_path.with_suffix(".config.json").write_text(
        json.dumps(snapshot, indent=2), encoding="utf-8"
    )
    # Embedded best weights make copying only the latest checkpoint sufficient.
    if best is not None:
        save_best_checkpoint(best_path, model, config, stats, best, history)
    started, session_start_step = time.perf_counter(), step
    stopped = False
    max_steps = config["max_steps"]
    interval = int(config["checkpoint_interval_steps"])

    def should_stop():
        return bool((stop_event and stop_event.is_set()) or (max_steps and step >= int(max_steps)))

    def persist(status):
        resume_state = {
            "version": RESUME_STATE_VERSION,
            "status": status,
            "epoch": epoch,
            "next_batch": next_batch,
            "batches_per_epoch": batches_per_epoch,
            "order": order,
            "sampler_rng_state": sampler_generator.get_state(),
            # State immediately before constructing this epoch's loader iterator.
            "loader_rng_state": loader_rng_state,
            "rng_state": capture_rng_state(device),
            "train_loss_sums": train_totals.state_dict(),
            "data_signature": data_signature,
            "training_seconds": elapsed_before + time.perf_counter() - started,
        }
        save_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            scheduler,
            step,
            epoch if next_batch else epoch - 1,
            config,
            stats,
            history,
            resume_state=resume_state,
            best=best,
        )

    # Also leaves a recovery point before the first optimizer step.
    persist("complete" if epoch >= epochs else "running")
    while epoch < epochs:
        if should_stop():
            stopped = True
            persist("paused")
            break
        model.train()
        if order is None:
            order = torch.randperm(len(dataset), generator=sampler_generator)
            loader_rng_state = torch.Generator().manual_seed(seed + epoch).get_state()
        loader_generator = torch.Generator()
        loader_generator.set_state(loader_rng_state.cpu())
        remaining_batches = [
            order[i : i + batch_size].tolist()
            for i in range(next_batch * batch_size, len(dataset), batch_size)
        ]
        loader = DataLoader(
            dataset,
            batch_sampler=remaining_batches,
            generator=loader_generator,
            pin_memory=device.type == "cuda",
            num_workers=int(config["num_workers"]),
        )
        for batch in loader:
            if should_stop():
                stopped = True
                break
            batch = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss, action_loss, smoothness, sums = _loss(model, batch, config, return_sums=True)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Non-finite training loss; resume from the last saved checkpoint"
                )
            loss.backward()
            if config["grad_clip"]:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            optimizer.step()
            step += 1
            next_batch += 1
            train_totals.update(sums)
            if progress:
                progress(
                    {
                        "epoch": epoch,
                        "step": step,
                        "train_loss": float(loss.detach()),
                        "action_loss": float(action_loss.detach()),
                        "smoothness_loss": float(smoothness.detach()),
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "eta_seconds": max(
                            0.0,
                            (time.perf_counter() - started)
                            / max(1, step - session_start_step)
                            * (epochs * batches_per_epoch - step),
                        ),
                    }
                )
            if next_batch < batches_per_epoch:
                if should_stop():
                    stopped = True
                    break
                if interval and step % interval == 0:
                    persist("running")
        if next_batch < batches_per_epoch:
            persist("paused")
            stopped = True
            break

        # Validate and advance the scheduler only after ALL batches of this epoch.
        model.eval()
        validation_totals = EpochLossAccumulator()
        val_loader = DataLoader(
            val_set,
            batch_size=batch_size,
            num_workers=int(config["num_workers"]),
            pin_memory=device.type == "cuda",
            generator=torch.Generator().manual_seed(seed + epoch + 1_000_003),
        )
        with torch.no_grad():
            for batch in val_loader:
                *_, sums = _loss(model, _move_batch(batch, device), config, return_sums=True)
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
        if math.isfinite(val_mean) and (best is None or val_mean < best["validation_loss"]):
            best = {
                "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "validation_loss": val_mean,
                "epoch": epoch,
                "step": step,
            }
            save_best_checkpoint(best_path, model, config, stats, best, history)
        scheduler.step()
        epoch += 1
        next_batch, order, loader_rng_state = 0, None, None
        train_totals = EpochLossAccumulator()
        persist("complete" if epoch == epochs else "paused" if should_stop() else "running")
        if progress:
            progress(
                {
                    **record,
                    "checkpoint": str(checkpoint_path),
                    "best_checkpoint": str(best_path) if best else None,
                    "epoch_complete": True,
                }
            )
    return TrainingResult(
        history,
        checkpoint_path,
        elapsed_before + time.perf_counter() - started,
        stopped,
        best_path if best is not None else None,
        best["validation_loss"] if best is not None else None,
    )
