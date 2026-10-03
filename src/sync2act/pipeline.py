from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from pathlib import Path

import torch

from sync2act.config import resolve_training_config
from sync2act.data.dataset import NormalizationStats
from sync2act.data.synthetic import generate_demo_episodes
from sync2act.evaluation import evaluate_policy
from sync2act.policies import build_policy
from sync2act.reporting import generate_report
from sync2act.training import load_checkpoint, train_policy
from sync2act.training.setup import build_training_policy, prepare_training_episodes

DEFAULT_DEMO_CONFIG = {
    "model": {"name": "bc_mlp", "state_dim": 6, "action_dim": 3, "input_mode": "image_state"},
    "training": {
        "epochs": 2,
        "batch_size": 32,
        "learning_rate": 0.003,
        "seed": 7,
        "device": "cpu",
        "max_steps": 12,
    },
    "dataset": {"num_episodes": 3, "length": 32, "seed": 7},
}


def make_demo_dataset(config: dict | None = None):
    settings = (config or {}).get("dataset", config or {})
    allowed = {"num_episodes", "length", "state_dim", "action_dim", "image_size", "seed"}
    return generate_demo_episodes(
        **{key: value for key, value in settings.items() if key in allowed}
    )


def _demo_dataset_settings(config: dict) -> dict:
    defaults = {
        name: parameter.default
        for name, parameter in inspect.signature(generate_demo_episodes).parameters.items()
    }
    return {**defaults, **{k: v for k, v in config.get("dataset", {}).items() if k in defaults}}


def run_training(
    config: dict,
    output_dir: str | Path,
    progress: Callable[[dict], None] | None = None,
    stop_event=None,
    resume_from=None,
):
    """Train on prepared episodes and return (model, clean test episodes, result)."""
    config = {**config, "dataset": _demo_dataset_settings(config)}
    config = resolve_training_config(config, source="YAML/pipeline")
    training_config = config["training"]
    prepared = prepare_training_episodes(
        make_demo_dataset(config), training_config, config.get("corruption")
    )
    if not prepared.test:
        raise ValueError(
            "The CLI/pipeline workflow requires test_split > 0 for held-out evaluation"
        )
    training_config["episode_split"] = prepared.split
    training_config["corruption"] = config.get("corruption")
    model = build_training_policy(config["model"], training_config)
    result = train_policy(
        model,
        prepared.train,
        training_config,
        output_dir,
        progress,
        stop_event,
        resume_from,
        validation_episodes=prepared.validation,
        normalization_stats=prepared.stats,
    )
    load_checkpoint(result.evaluation_checkpoint, model)
    # Callers evaluate only the clean held-out partition, using checkpoint statistics.
    return model, prepared.test, result


def run_evaluation(config: dict, checkpoint: str | Path, output_dir: str | Path):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_policy(saved["config"].get("model", config.get("model", config)))
    payload = load_checkpoint(checkpoint, model)
    stats = NormalizationStats.from_dict(payload["stats"]) if payload.get("stats") else None
    episodes = make_demo_dataset(config)
    saved_config = payload["config"]
    if _demo_dataset_settings(config) == saved_config.get("dataset_config"):
        episodes = [episodes[i] for i in saved_config["episode_split"]["test"]]
    metrics = evaluate_policy(
        model,
        episodes,
        config.get("evaluation", {}).get("device", "cpu"),
        stats=stats,
        output_dir=output_dir,
    )
    metrics["training_seconds"] = config.get("training_seconds")
    output = Path(output_dir)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def run_demo(
    output_dir: str | Path = "runs/demo", progress: Callable[[dict], None] | None = None
) -> dict:
    output = Path(output_dir)
    model, episodes, training = run_training(DEFAULT_DEMO_CONFIG, output, progress)
    payload = torch.load(training.checkpoint, map_location="cpu", weights_only=False)
    metrics = evaluate_policy(
        model, episodes, stats=NormalizationStats.from_dict(payload["stats"]), output_dir=output
    )
    metrics["training_seconds"] = training.training_seconds
    run = {
        "name": "bundled-demo",
        "model": "bc_mlp",
        "corruption": "clean",
        "seed": 7,
        "config": payload["config"],
        "checkpoint": str(training.evaluation_checkpoint),
        "resume_checkpoint": str(training.checkpoint),
        "checkpoint_selection": "best_validation_loss"
        if training.best_checkpoint
        else "latest_unvalidated",
        "metrics": metrics,
    }
    (output / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    report = generate_report([run], output / "report.html")
    return {"run": run, "report": str(report)}


def run_benchmark(config: dict, output_root: str | Path) -> list[dict]:
    output_root = Path(output_root)
    results = []
    models = config.get("models", ["bc_mlp", "act_lite", "quality_act"])
    corruptions = config.get("corruptions", [{"name": "clean"}])
    seeds = config.get("seeds", [7, 17, 27])
    for model_name in models:
        for corruption in corruptions:
            for seed in seeds:
                run_name = f"{model_name}-{corruption.get('name', corruption.get('type', 'clean'))}-s{seed}"
                run_config = {
                    "model": {**config.get("model_defaults", {}), "name": model_name},
                    "training": {**config.get("training", {}), "seed": seed},
                    "dataset": config.get("dataset", {}),
                }
                if corruption.get("type"):
                    run_config["corruption"] = {
                        key: value for key, value in corruption.items() if key != "name"
                    }
                model, episodes, training = run_training(run_config, output_root / run_name)
                payload = torch.load(training.checkpoint, map_location="cpu", weights_only=False)
                metrics = evaluate_policy(
                    model,
                    episodes,
                    device=payload["config"]["device"],
                    stats=NormalizationStats.from_dict(payload["stats"]),
                    output_dir=output_root / run_name,
                )
                metrics["training_seconds"] = training.training_seconds
                run = {
                    "name": run_name,
                    "model": model_name,
                    "corruption": corruption.get("name", "clean"),
                    "seed": seed,
                    "config": payload["config"],
                    "checkpoint": str(training.evaluation_checkpoint),
                    "resume_checkpoint": str(training.checkpoint),
                    "checkpoint_selection": "best_validation_loss"
                    if training.best_checkpoint
                    else "latest_unvalidated",
                    "metrics": metrics,
                }
                (output_root / run_name / "run.json").write_text(
                    json.dumps(run, indent=2), encoding="utf-8"
                )
                results.append(run)
    generate_report(results, output_root / "benchmark.html")
    return results
