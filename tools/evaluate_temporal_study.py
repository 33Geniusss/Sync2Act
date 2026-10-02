"""Reevaluate saved study checkpoints with causal temporal ensembling; no training.

python tools/evaluate_temporal_study.py --device cuda --decay 0.7
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sync2act.data.dataset import NormalizationStats
from sync2act.data.lerobot import load_local_lerobot_dataset
from sync2act.evaluation.evaluator import evaluate_policy
from sync2act.policies.factory import build_policy
from sync2act.reporting.temporal import write_temporal_report


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def compare_historical(
    original: Path, paired: Path, historical_mse: float, current_mse: float
) -> dict:
    old, new = pd.read_csv(original), pd.read_csv(paired)
    if not old[["episode", "step"]].equals(new[["episode", "step"]]):
        raise RuntimeError("Historical evaluation sample ordering changed")
    prediction_columns = [key for key in old if key.startswith("prediction_")]
    target_columns = [key for key in old if key.startswith("target_")]
    old_pred = old[prediction_columns].to_numpy()
    new_pred = new[
        [key.replace("prediction_", "first_step_") for key in prediction_columns]
    ].to_numpy()
    target_match = np.allclose(old[target_columns], new[target_columns], rtol=1e-5, atol=1e-6)
    prediction_match = np.allclose(old_pred, new_pred, rtol=1e-4, atol=2e-5)
    mse_match = bool(np.isclose(historical_mse, current_mse, rtol=1e-4, atol=1e-9))
    audit = {
        "passed": bool(target_match and prediction_match and mse_match),
        "target_match": bool(target_match),
        "prediction_match": bool(prediction_match),
        "mse_match": mse_match,
        "prediction_max_abs_difference": float(np.abs(old_pred - new_pred).max()),
        "historical_mse": historical_mse,
        "recomputed_mse": current_mse,
        "prediction_rtol": 1e-4,
        "prediction_atol": 2e-5,
        "mse_rtol": 1e-4,
        "mse_atol": 1e-9,
        "target_rtol": 1e-5,
        "target_atol": 1e-6,
    }
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=Path("runs/mixed_modality_damage_v1_3_0"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--decay", type=float, default=0.7)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if not 0 < args.decay <= 1:
        parser.error("decay must be in (0, 1]")
    torch.set_num_threads(4)
    output = args.study / "temporal_ensemble"
    source_files = [
        Path(__file__),
        Path("src/sync2act/evaluation/evaluator.py"),
        Path("src/sync2act/evaluation/temporal.py"),
        Path("src/sync2act/policies/act_lite.py"),
        Path("src/sync2act/policies/quality_act.py"),
        Path("src/sync2act/policies/common.py"),
        Path("src/sync2act/data/dataset.py"),
        Path("src/sync2act/data/lerobot.py"),
    ]
    source_hashes = {str(p): file_hash(p) for p in source_files}
    run_paths = sorted(args.study.glob("*/*/*/seed-*/run.json"))
    study = json.loads((args.study / "study.json").read_text(encoding="utf-8"))
    expected = (
        len(study["datasets"])
        * len(study["models"])
        * len(study["conditions"])
        * len(study["seeds"])
    )
    if len(run_paths) != expected:
        raise RuntimeError(f"Expected {expected} original runs, found {len(run_paths)}")
    evaluations = []
    for manifest_path in sorted(args.study.glob("*/dataset_manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        dataset_name = manifest["name"]
        dataset_runs = [p for p in run_paths if p.relative_to(args.study).parts[0] == dataset_name]
        test_episodes = None
        data_root = Path(manifest["loaded_metadata"]["root"])
        inventory = [
            {
                "path": str(p.relative_to(data_root)),
                "size": p.stat().st_size,
                "mtime_ns": p.stat().st_mtime_ns,
            }
            for sub in ["meta", "data", "videos"]
            for p in sorted((data_root / sub).rglob("*"))
            if p.is_file()
        ]
        for path in dataset_runs:
            run = json.loads(path.read_text(encoding="utf-8"))
            if run.get("status") != "complete":
                raise RuntimeError(f"Incomplete source run: {path}")
            checkpoint = path.parent / run["checkpoint"]
            relative = path.parent.relative_to(args.study)
            destination = output / relative
            signature_input = {
                "checkpoint_sha256": file_hash(checkpoint),
                "run_sha256": file_hash(path),
                "historical_predictions_sha256": file_hash(path.parent / "predictions.csv"),
                "manifest_sha256": file_hash(manifest_path),
                "source_hashes": source_hashes,
                "data_inventory": inventory,
                "decay": args.decay,
                "device": args.device,
                "torch": torch.__version__,
            }
            signature = hashlib.sha256(
                json.dumps(signature_input, sort_keys=True).encode()
            ).hexdigest()
            record_path = destination / "evaluation.json"
            if record_path.exists():
                saved = json.loads(record_path.read_text(encoding="utf-8"))
                artifacts = [
                    "predictions.csv",
                    "per_episode.csv",
                    "first_step_per_episode.csv",
                    "action_chunks.pt",
                ]
                if (
                    saved.get("signature") == signature
                    and saved["baseline_audit"]["passed"]
                    and all((destination / a).is_file() for a in artifacts)
                ):
                    evaluations.append(saved)
                    print(f"[{len(evaluations)}/{expected}] reuse {relative}", flush=True)
                    continue
            if args.report_only:
                raise RuntimeError(f"Missing or stale temporal evaluation: {record_path}")
            if test_episodes is None:
                print(f"Loading {dataset_name} from {data_root}", flush=True)
                last_percent = -10

                def progress(event):
                    nonlocal last_percent
                    if event.get("percent", 0) >= last_percent + 10:
                        last_percent = event["percent"]
                        print(f"  {event}", flush=True)

                cache_signature = hashlib.sha256(
                    json.dumps(
                        {
                            "manifest": manifest,
                            "inventory": inventory,
                            "loader_sha256": file_hash(Path("src/sync2act/data/lerobot.py")),
                        },
                        sort_keys=True,
                    ).encode()
                ).hexdigest()
                cache = output / "_data_cache" / f"{dataset_name}-{cache_signature[:16]}.pt"
                if cache.exists():
                    cached = torch.load(cache, map_location="cpu", weights_only=False)
                    test_episodes, loaded = cached["episodes"], cached["metadata"]
                else:
                    episodes, loaded = load_local_lerobot_dataset(
                        data_root,
                        progress=progress,
                        max_image_size=manifest["loaded_metadata"]["max_image_size"],
                        episode_limit=manifest["loaded_metadata"].get("episode_limit"),
                        frames_per_episode_limit=manifest["loaded_metadata"].get(
                            "frames_per_episode_limit"
                        ),
                        episode_positions=manifest["split"]["test"],
                    )
                    by_position = dict(
                        zip(sorted(manifest["split"]["test"]), episodes, strict=True)
                    )
                    test_episodes = [by_position[i] for i in manifest["split"]["test"]]
                    del episodes, by_position
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    temporary = cache.with_suffix(".pt.tmp")
                    torch.save({"episodes": test_episodes, "metadata": loaded}, temporary)
                    temporary.replace(cache)
                for key in [
                    "source_episodes",
                    "source_frames",
                    "image_keys",
                    "image_shape",
                    "state_dim",
                    "action_dim",
                ]:
                    if loaded[key] != manifest["loaded_metadata"][key]:
                        raise RuntimeError(f"Dataset metadata changed: {key}")
            if run["split"] != manifest["split"]:
                raise RuntimeError("Run split differs from dataset manifest")
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            model = build_policy(run["model_config"])
            model.load_state_dict(payload["model"], strict=True)
            if payload["stats"] != manifest["normalization_stats"]:
                raise RuntimeError("Checkpoint normalization differs from frozen study statistics")
            start = time.perf_counter()
            metrics = evaluate_policy(
                model,
                test_episodes,
                device=args.device,
                batch_size=run["training_config"]["batch_size"],
                stats=NormalizationStats.from_dict(payload["stats"]),
                output_dir=destination,
                temporal_decay=args.decay,
                save_action_chunks=True,
            )
            audit = compare_historical(
                path.parent / "predictions.csv",
                destination / "predictions.csv",
                run["metrics"]["action_mse"],
                metrics["first_step"]["action_mse"],
            )
            saved = {
                "schema_version": 1,
                "signature": signature,
                "provenance": signature_input,
                "source_run": str(path.resolve()),
                "dataset": dataset_name,
                "model": run["model"],
                "condition": run["condition"],
                "seed": run["seed"],
                "test_episode_indices": manifest["split"]["test"],
                "created_at": datetime.now(UTC).isoformat(),
                "metrics": metrics,
                "baseline_audit": audit,
                "evaluation_wall_seconds": time.perf_counter() - start,
            }
            write_json(record_path, saved)
            if not audit["passed"]:
                raise RuntimeError(f"Historical baseline mismatch: {record_path}: {audit}")
            evaluations.append(saved)
            print(
                f"[{len(evaluations)}/{expected}] {relative}: MSE ratio={metrics['action_mse'] / metrics['first_step']['action_mse']:.6f}",
                flush=True,
            )
            del model, payload
        del test_episodes
    rows = []
    for record in evaluations:
        metrics = record["metrics"]
        row = {key: record[key] for key in ["dataset", "model", "condition", "seed"]}
        for key in ["action_mse", "action_mae", "trajectory_smoothness", "jerk"]:
            row[f"first_step_{key}"] = metrics["first_step"][key]
            row[f"ensemble_{key}"] = metrics[key]
            row[f"{key}_ratio"] = (
                metrics[key] / metrics["first_step"][key] if metrics["first_step"][key] else None
            )
        row["baseline_prediction_max_abs_difference"] = record["baseline_audit"][
            "prediction_max_abs_difference"
        ]
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "results.csv", index=False)
    protocol = {
        "schema_version": 1,
        "scope": "offline-only, same saved checkpoints, no retraining",
        "expected_runs": expected,
        "completed_runs": len(rows),
        "decay": args.decay,
        "decay_selection": "fixed existing default before evaluation; not tuned on test data",
        "baseline_audits_passed": sum(r["baseline_audit"]["passed"] for r in evaluations),
        "aggregation": "per-run ensemble/first-step ratios, equal-weight arithmetic means; never pool raw MSE across datasets",
        "created_at": datetime.now(UTC).isoformat(),
        "report_date": "2026-10-01",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": args.device,
        "gpu": torch.cuda.get_device_name() if args.device.startswith("cuda") else None,
        "source_hashes": source_hashes,
        "latency_note": "No real-time or closed-loop latency claim; offline batch fusion measured separately.",
    }
    write_json(output / "summary.json", protocol)
    write_temporal_report(args.study, frame, protocol)
    print(f"Complete: {output / 'report.html'}", flush=True)


if __name__ == "__main__":
    main()
