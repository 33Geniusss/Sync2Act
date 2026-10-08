"""Evaluate recorded study checkpoints on corrupted held-out observations, without training.

python tools/evaluate_test_corruption_study.py --pilot --device cuda
python tools/evaluate_test_corruption_study.py --device cuda
python tools/evaluate_test_corruption_study.py --report-only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sync2act.data.dataset import NormalizationStats
from sync2act.data.lerobot import load_local_lerobot_dataset
from sync2act.evaluation.test_corruption import (
    TestCorruptionConfig,
    evaluate_test_condition,
    prepare_test_condition,
    preset_config,
    write_json,
)
from sync2act.policies import build_policy
from sync2act.reporting.test_corruption import write_test_corruption_report

ROOT = Path(__file__).resolve().parents[1]


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def historical_audit(original, recomputed, historical_mse, current_mse):
    old, new = pd.read_csv(original), pd.read_csv(recomputed)
    if not old[["episode", "step"]].equals(new[["episode", "step"]]):
        raise RuntimeError("Historical test sample ordering changed")
    predictions = [c for c in old if c.startswith("prediction_")]
    targets = [c for c in old if c.startswith("target_")]
    audit = {
        "targets_match": bool(np.allclose(old[targets], new[targets], rtol=1e-5, atol=1e-6)),
        "predictions_match": bool(np.allclose(old[predictions], new[predictions], rtol=1e-4, atol=2e-5)),
        "mse_match": bool(np.isclose(historical_mse, current_mse, rtol=1e-4, atol=1e-9)),
        "prediction_max_abs_difference": float(np.abs(old[predictions].to_numpy() - new[predictions].to_numpy()).max()),
        "historical_mse": historical_mse, "recomputed_mse": current_mse,
        "tolerances": {"prediction_rtol": 1e-4, "prediction_atol": 2e-5,
                       "target_rtol": 1e-5, "target_atol": 1e-6, "mse_rtol": 1e-4, "mse_atol": 1e-9},
    }
    audit["passed"] = all(audit[k] for k in ("targets_match", "predictions_match", "mse_match"))
    return audit


def load_test_data(study, manifest, output):
    root = Path(manifest["loaded_metadata"]["root"])
    inventory = [{"path": str(p.relative_to(root)), "size": p.stat().st_size,
                  "mtime_ns": p.stat().st_mtime_ns}
                 for sub in ("meta", "data", "videos") for p in sorted((root / sub).rglob("*")) if p.is_file()]
    if not inventory:
        raise FileNotFoundError(f"Original dataset is unavailable: {root}")
    # Match the existing temporal study cache exactly; otherwise load original files.
    cache_key = signature({"manifest": manifest, "inventory": inventory,
                           "loader_sha256": file_hash(ROOT / "src/sync2act/data/lerobot.py")})
    old_cache = study / "temporal_ensemble/_data_cache" / f"{manifest['name']}-{cache_key[:16]}.pt"
    cache = output / "_data_cache" / f"{manifest['name']}-{cache_key[:16]}.pt"
    candidate = cache if cache.exists() else old_cache
    if candidate.exists():
        loaded = torch.load(candidate, map_location="cpu", weights_only=False)
        episodes, metadata = loaded["episodes"], loaded["metadata"]
    else:
        print(f"Loading held-out episodes: {manifest['name']}", flush=True)
        episodes, metadata = load_local_lerobot_dataset(
            root, max_image_size=manifest["loaded_metadata"]["max_image_size"],
            episode_limit=manifest["loaded_metadata"].get("episode_limit"),
            frames_per_episode_limit=manifest["loaded_metadata"].get("frames_per_episode_limit"),
            episode_positions=manifest["split"]["test"],
        )
        by_index = dict(zip(sorted(manifest["split"]["test"]), episodes, strict=True))
        episodes = [by_index[i] for i in manifest["split"]["test"]]
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"episodes": episodes, "metadata": metadata}, cache)
    for key in ("source_episodes", "source_frames", "image_keys", "image_shape", "state_dim", "action_dim"):
        if metadata[key] != manifest["loaded_metadata"][key]:
            raise RuntimeError(f"Dataset metadata changed: {key}")
    if len(episodes) != len(manifest["split"]["test"]):
        raise RuntimeError("Test episode count differs from recorded split")
    digest = hashlib.sha256()
    for episode in episodes:
        for key in sorted(episode):
            value = episode[key]
            if torch.is_tensor(value):
                digest.update(str((key, value.dtype, tuple(value.shape))).encode())
                digest.update(value.contiguous().numpy().tobytes())
    return episodes, {"cache_signature": cache_key, "test_tensor_sha256": digest.hexdigest()}


def test_cases(run, damage_seeds, pilot=False):
    yield "baseline", "clean", TestCorruptionConfig()
    for severity in (["medium"] if pilot else ["medium", "light", "heavy"]):
        if severity != "medium" and not (
            run["model"] in {"act_lite", "quality_full"} and
            run["condition"] in {"clean", "mixed_three_corruptions"}
        ):
            continue
        for condition in ("image", "state", "mixed"):
            modes = ["oracle"]
            if severity == "medium" and run["model"] in {"quality_input", "quality_full"}:
                modes.append("unknown")
            for seed in damage_seeds:
                for mode in modes:
                    config = preset_config(condition, severity=severity, seed=seed, quality_mode=mode)
                    group = "main" if severity == "medium" and mode == "oracle" else "metadata" if mode == "unknown" else "severity"
                    yield group, severity, config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=Path("runs/mixed_modality_damage_v1_3_0"))
    parser.add_argument("--output", type=Path, default=Path("runs/test_time_corruption_v1"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if args.report_only:
        summary = json.loads((args.output / "summary.json").read_text(encoding="utf-8"))
        if len(summary["record_paths"]) != summary["expected_evaluations"]:
            raise RuntimeError("Saved study is incomplete")
        records = [json.loads((args.output / p).read_text(encoding="utf-8")) for p in summary["record_paths"]]
        for relative, record in zip(summary["record_paths"], records, strict=True):
            directory = (args.output / relative).parent
            if record["signature"] != signature(record["provenance"]):
                raise RuntimeError(f"Invalid evaluation signature: {directory}")
            for name, digest in record["artifact_hashes"].items():
                if not (directory / name).is_file() or file_hash(directory / name) != digest:
                    raise RuntimeError(f"Missing or modified artifact: {directory / name}")
        write_test_corruption_report(records, summary, args.output)
        return
    torch.set_num_threads(4)
    study = json.loads((args.study / "study.json").read_text(encoding="utf-8"))
    paths = sorted(args.study.glob("*/*/*/seed-*/run.json"))
    expected_sources = len(study["datasets"]) * len(study["models"]) * len(study["conditions"]) * len(study["seeds"])
    if len(paths) != expected_sources:
        raise RuntimeError(f"Expected {expected_sources} source runs, found {len(paths)}")
    sources = []
    for path in paths:
        run = json.loads(path.read_text(encoding="utf-8"))
        if args.pilot and not (run["dataset"] == "xarm_lift_medium" and run["model"] in {"act_lite", "quality_full"}
                               and run["condition"] in {"clean", "mixed_three_corruptions"} and run["seed"] == 7):
            continue
        if run.get("status") != "complete":
            raise RuntimeError(f"Incomplete source run: {path}")
        sources.append((path, run))
    damage_seeds = [107] if args.pilot else [107, 117, 127]
    total = sum(len(list(test_cases(run, damage_seeds, args.pilot))) for _, run in sources)
    code_files = [Path(__file__), *sorted((ROOT / "src/sync2act/evaluation").glob("*.py")),
                  *sorted((ROOT / "src/sync2act/corruptions").glob("*.py")),
                  *sorted((ROOT / "src/sync2act/policies").glob("*.py")),
                  ROOT / "src/sync2act/data/dataset.py", ROOT / "src/sync2act/data/episode.py"]
    source_hashes = {str(p.resolve().relative_to(ROOT)): file_hash(p) for p in code_files}
    runtime = {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
               "device": args.device, "gpu": torch.cuda.get_device_name() if args.device.startswith("cuda") else None}
    records, record_paths = [], []
    started = time.perf_counter()
    for dataset in sorted({run["dataset"] for _, run in sources}):
        manifest_path = args.study / dataset / "dataset_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        episodes, data_evidence = load_test_data(args.study, manifest, args.output)
        # Conditions are generated once per dataset and shared across every checkpoint.
        prepared_cache, baselines = {}, {}
        tasks = [(path, run, group, severity, config)
                 for path, run in sources if run["dataset"] == dataset
                 for group, severity, config in test_cases(run, damage_seeds, args.pilot)]
        tasks.sort(key=lambda task: (task[2] != "baseline", signature(asdict(task[4])), str(task[0])))
        for path, run, group, severity, config in tasks:
            if run["split"] != manifest["split"]:
                raise RuntimeError(f"Run test split changed: {path}")
            checkpoint = path.parent / run["checkpoint"]
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if payload["stats"] != manifest["normalization_stats"]:
                raise RuntimeError(f"Frozen normalization changed: {checkpoint}")
            model = build_policy(run["model_config"])
            model.load_state_dict(payload["model"], strict=True)
            stats = NormalizationStats.from_dict(payload["stats"])
            base_provenance = {"checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": file_hash(checkpoint),
                               "source_run_sha256": file_hash(path), "manifest_sha256": file_hash(manifest_path),
                               "historical_predictions_sha256": file_hash(path.parent / "predictions.csv"),
                               "data": data_evidence, "source_hashes": source_hashes, "runtime": runtime,
                               "normalization": payload["stats"], "test_episode_indices": manifest["split"]["test"]}
            baseline = baselines.get(str(path))
            case = "clean" if group == "baseline" else f"{severity}-{config.condition}-{config.quality_mode}-s{config.seed}"
            relative = path.parent.relative_to(args.study)
            destination = args.output / "evaluations" / relative / case
            record_path = destination / "evaluation.json"
            provenance = {**base_provenance, "test_config": asdict(config)}
            key = signature(provenance)
            record = None
            if record_path.exists():
                saved = json.loads(record_path.read_text(encoding="utf-8"))
                if saved.get("signature") != key:
                    raise RuntimeError(f"Stale evaluation; use a new output directory: {record_path}")
                if all((destination / name).is_file() and file_hash(destination / name) == digest
                       for name, digest in saved["artifact_hashes"].items()):
                    record = saved
            if record is None:
                prep_key = signature(asdict(config))
                if prep_key not in prepared_cache:
                    # Bound RAM even for four-camera datasets; deterministic regeneration is cheap.
                    if len(prepared_cache) >= 1:
                        prepared_cache.clear()
                    prepared_cache[prep_key] = prepare_test_condition(episodes, config)
                start = time.perf_counter()
                metrics = evaluate_test_condition(
                    model, prepared_cache[prep_key], stats=stats, output_dir=destination,
                    clean_metrics=baseline, device=args.device,
                    batch_size=run["training_config"]["batch_size"],
                )
                audit = None
                if group == "baseline":
                    audit = historical_audit(path.parent / "predictions.csv", destination / "predictions.csv",
                                             run["metrics"]["action_mse"], metrics["action_mse"])
                    if not audit["passed"]:
                        write_json(destination / "failed_baseline_audit.json", audit)
                        raise RuntimeError(f"Clean baseline differs from history: {destination}: {audit}")
                record = {"schema_version": 1, "signature": key, "provenance": provenance,
                          "dataset": dataset, "model": run["model"], "train_condition": run["condition"],
                          "train_seed": run["seed"], "group": group, "severity": severity,
                          "test_condition": config.condition, "quality_mode": config.quality_mode,
                          "damage_seed": config.seed if group != "baseline" else None,
                          "metrics": metrics, "baseline_audit": audit,
                          "actual_affected_fraction": prepared_cache[prep_key].manifest["actual_affected_fraction"],
                          "created_at": datetime.now(UTC).isoformat(),
                          "evaluation_wall_seconds": time.perf_counter() - start,
                          "artifact_hashes": {name: file_hash(destination / name) for name in
                                              ("predictions.csv", "per_episode.csv", "corruption.json", "metrics.json")}}
                write_json(record_path, record)
            if group == "baseline":
                if not record["baseline_audit"]["passed"]:
                    raise RuntimeError("Cached baseline audit failed")
                baselines[str(path)] = record["metrics"]
            records.append(record)
            record_paths.append(str(record_path.relative_to(args.output)))
            print(f"[{len(records)}/{total}] {relative} / {case}: MSE={record['metrics']['action_mse']:.7g} ({time.perf_counter()-started:.1f}s elapsed)", flush=True)
            del model, payload
        del episodes, prepared_cache
    summary = {"schema_version": 1, "scope": "offline-only; same historical weights; no retraining",
               "pilot": args.pilot, "source_study": str(args.study.resolve()), "source_checkpoint_count": len(sources),
               "expected_evaluations": total, "completed_evaluations": len(records),
               "baseline_audits_passed": sum(bool(r["baseline_audit"] and r["baseline_audit"]["passed"]) for r in records),
               "damage_seeds": damage_seeds, "runtime": runtime, "source_hashes": source_hashes,
               "record_paths": record_paths, "created_at": datetime.now(UTC).isoformat(),
               "wall_seconds": time.perf_counter() - started,
               "selection": "All historical checkpoints at medium severity; Oracle/unknown paired for quality_input and quality_full; light/heavy for ACT-Lite and quality_full trained clean or mixed.",
               "quality_protocol": "All six training variants receive true observation metadata in Oracle tests. Training shuffled/constant controls are not inference metadata interventions.",
               "aggregation": "Average damage seeds within each training seed, then average training seeds; report separate SDs. Never pool raw MSE across datasets.",
               "severity_note": "Light/medium/heavy jointly vary delay and damaged fraction; not a single-factor sensitivity estimate."}
    write_json(args.output / "summary.json", summary)
    write_test_corruption_report(records, summary, args.output)
    print(f"Completed {len(records)} evaluations: {args.output / 'report.html'}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("Interrupted; completed signed evaluations can be resumed with the same command.")
