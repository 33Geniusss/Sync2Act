from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from sync2act.data.dataset import EpisodeWindowDataset, NormalizationStats
from sync2act.data.episode import Episode

from .temporal import ensemble_action_chunks


def _episode_trajectory_metrics(
    prediction: torch.Tensor,
    episode_ids: list[int],
    steps: list[int],
) -> tuple[float, float]:
    """Compute discrete trajectory changes without crossing episode boundaries."""
    velocities: list[torch.Tensor] = []
    accelerations: list[torch.Tensor] = []
    episode_tensor = torch.tensor(episode_ids)
    step_tensor = torch.tensor(steps)

    for episode_id in episode_tensor.unique(sorted=True):
        indices = torch.nonzero(episode_tensor == episode_id, as_tuple=False).flatten()
        indices = indices[torch.argsort(step_tensor[indices])]
        local_prediction = prediction[indices]
        if len(local_prediction) < 2:
            continue
        velocity = local_prediction[1:] - local_prediction[:-1]
        velocities.append(velocity)
        if len(velocity) >= 2:
            accelerations.append(velocity[1:] - velocity[:-1])

    smoothness = float(torch.cat(velocities).square().mean()) if velocities else 0.0
    jerk = float(torch.cat(accelerations).abs().mean()) if accelerations else 0.0
    return smoothness, jerk


def _prediction_metrics(prediction, target, episode_ids, steps) -> dict:
    error = prediction - target
    smoothness, jerk = _episode_trajectory_metrics(prediction, episode_ids, steps)
    per_episode = []
    for episode_id in sorted(set(episode_ids)):
        mask = torch.tensor([item == episode_id for item in episode_ids])
        local_error = error[mask]
        per_episode.append(
            {
                "episode": episode_id,
                "mse": float(local_error.square().mean()),
                "mae": float(local_error.abs().mean()),
            }
        )
    return {
        "action_mse": float(error.square().mean()),
        "action_mae": float(error.abs().mean()),
        "trajectory_smoothness": smoothness,
        "jerk": jerk,
        "per_episode": per_episode,
    }


def evaluate_policy(
    model: torch.nn.Module,
    episodes: list[Episode],
    device: str = "cpu",
    batch_size: int = 32,
    stats: NormalizationStats | None = None,
    output_dir: str | Path | None = None,
    *,
    temporal_decay: float | None = None,
    save_action_chunks: bool = False,
) -> dict:
    """Evaluate first-step actions, or causal temporal fusion with a paired baseline.

    Temporal fusion uses the same forward outputs as the first-step baseline.
    This remains offline evaluation on recorded observations, not a rollout.
    """
    if temporal_decay is not None and not 0 < temporal_decay <= 1:
        raise ValueError("temporal_decay must be in (0, 1]")
    if save_action_chunks and output_dir is None:
        raise ValueError("save_action_chunks requires output_dir")
    model = model.to(device).eval()
    horizon = getattr(model, "horizon", 1)
    dataset = EpisodeWindowDataset(episodes, horizon=horizon, stats=stats)
    loader = DataLoader(dataset, batch_size=batch_size, pin_memory=str(device).startswith("cuda"))
    predictions, targets, episode_ids, steps, latencies = [], [], [], [], []
    chunks = []
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            if image.dtype == torch.uint8:
                image = image.float().div_(255.0)
            if str(device).startswith("cuda"):
                torch.cuda.synchronize()
            start = time.perf_counter()
            prediction = model(
                state=batch["state"].to(device),
                image=image,
                image_quality=batch["image_quality"].to(device),
                state_quality=batch["state_quality"].to(device),
                image_missing=batch["image_missing"].to(device),
                state_missing=batch["state_missing"].to(device),
                image_time_offset=batch["image_time_offset"].to(device),
                state_time_offset=batch["state_time_offset"].to(device),
                missing=batch["missing"].to(device),
                time_offset=batch["time_offset"].to(device),
            )
            if temporal_decay is not None or save_action_chunks:
                full_prediction = prediction.cpu()
                chunks.append(
                    full_prediction * dataset.stats.action_std + dataset.stats.action_mean
                )
                prediction = full_prediction[:, 0]
            else:
                prediction = prediction[:, 0].cpu()
            if str(device).startswith("cuda"):
                torch.cuda.synchronize()
            latencies.extend(
                [(time.perf_counter() - start) * 1000 / len(prediction)] * len(prediction)
            )
            predictions.append(prediction * dataset.stats.action_std + dataset.stats.action_mean)
            targets.append(
                batch["actions"][:, 0] * dataset.stats.action_std + dataset.stats.action_mean
            )
            episode_ids.extend(batch["episode_index"].tolist())
            steps.extend(batch["step"].tolist())
    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    first_step = prediction
    baseline_metrics = _prediction_metrics(first_step, target, episode_ids, steps)
    aggregation_seconds = 0.0
    if temporal_decay is not None:
        start = time.perf_counter()
        prediction = ensemble_action_chunks(torch.cat(chunks), episode_ids, steps, temporal_decay)
        aggregation_seconds = time.perf_counter() - start
    latency = torch.tensor(latencies)
    metrics = {
        "evaluation_scope": "offline-only",
        **(
            baseline_metrics
            if temporal_decay is None
            else _prediction_metrics(prediction, target, episode_ids, steps)
        ),
        "latency_p50_ms": float(torch.quantile(latency, 0.5)),
        "latency_p95_ms": float(torch.quantile(latency, 0.95)),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    if temporal_decay is not None:
        metrics.update(
            {
                "action_selection": "causal_temporal_ensemble",
                "temporal_decay": temporal_decay,
                "first_step": baseline_metrics,
                "aggregation_seconds": aggregation_seconds,
                "latency_scope": "batched forward and transfers, excludes offline temporal aggregation; not control latency",
            }
        )
    if output_dir:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "episode": episode_ids,
                "step": steps,
                **{f"prediction_{i}": prediction[:, i].numpy() for i in range(prediction.shape[1])},
                **(
                    {
                        f"first_step_{i}": first_step[:, i].numpy()
                        for i in range(first_step.shape[1])
                    }
                    if temporal_decay is not None
                    else {}
                ),
                **{f"target_{i}": target[:, i].numpy() for i in range(target.shape[1])},
            }
        ).to_csv(output / "predictions.csv", index=False)
        pd.DataFrame(metrics["per_episode"]).to_csv(output / "per_episode.csv", index=False)
        if temporal_decay is not None:
            pd.DataFrame(baseline_metrics["per_episode"]).to_csv(
                output / "first_step_per_episode.csv", index=False
            )
        if save_action_chunks:
            torch.save(
                {
                    "schema_version": 1,
                    "action_scale": "original_dataset_units",
                    "chunks": torch.cat(chunks),
                    "target": target,
                    "episode_ids": episode_ids,
                    "steps": steps,
                },
                output / "action_chunks.pt",
            )
    return metrics
