from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from sync2act.corruptions import apply_corruption
from sync2act.data.dataset import NormalizationStats
from sync2act.data.synthetic import generate_demo_episodes
from sync2act.evaluation import evaluate_policy
from sync2act.policies import build_policy
from sync2act.reporting import generate_report
from sync2act.training import load_checkpoint, train_policy

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


def run_training(
    config: dict,
    output_dir: str | Path,
    progress: Callable[[dict], None] | None = None,
    stop_event=None,
    resume_from=None,
):
    episodes = make_demo_dataset(config)
    corruption = config.get("corruption")
    if corruption:
        episodes = [
            apply_corruption(
                episode,
                {
                    **corruption,
                    "seed": int(corruption.get("seed", config.get("training", {}).get("seed", 7)))
                    + index,
                },
            )
            for index, episode in enumerate(episodes)
        ]
    model_config = config.get("model", config)
    model = build_policy(model_config)
    training_config = dict(config.get("training", config))
    training_config["model"] = model_config
    result = train_policy(
        model,
        episodes,
        training_config,
        output_dir,
        progress,
        stop_event,
        resume_from,
    )
    return model, episodes, result


def run_evaluation(config: dict, checkpoint: str | Path, output_dir: str | Path):
    model = build_policy(config.get("model", config))
    payload = load_checkpoint(checkpoint, model)
    stats = NormalizationStats.from_dict(payload["stats"]) if payload.get("stats") else None
    episodes = make_demo_dataset(config)
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
    metrics = evaluate_policy(model, episodes, stats=None, output_dir=output)
    metrics["training_seconds"] = training.training_seconds
    run = {
        "name": "bundled-demo",
        "model": "bc_mlp",
        "corruption": "clean",
        "seed": 7,
        "config": DEFAULT_DEMO_CONFIG,
        "checkpoint": str(training.checkpoint),
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
                metrics = evaluate_policy(model, episodes, output_dir=output_root / run_name)
                metrics["training_seconds"] = training.training_seconds
                run = {
                    "name": run_name,
                    "model": model_name,
                    "corruption": corruption.get("name", "clean"),
                    "seed": seed,
                    "config": run_config,
                    "metrics": metrics,
                }
                (output_root / run_name / "run.json").write_text(
                    json.dumps(run, indent=2), encoding="utf-8"
                )
                results.append(run)
    generate_report(results, output_root / "benchmark.html")
    return results
