import importlib.util
import json
from pathlib import Path

import pytest
import torch

from sync2act.config import load_config, resolve_training_config, save_config
from sync2act.data.synthetic import generate_demo_episodes


def _load_study_runner():
    path = Path(__file__).resolve().parents[2] / "tools" / "run_real_dataset_study.py"
    spec = importlib.util.spec_from_file_location("sync2act_real_dataset_study", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load study runner from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study_runner = _load_study_runner()


def test_requested_mixed_damage_condition_weights_and_missing_semantics():
    conditions = {item["name"]: item for item in study_runner.corruption_conditions([], 30.0)}

    assert list(conditions) == study_runner.CONDITION_ORDER
    expected = {
        "mixed_image_damaged": {
            "clean": 0.50,
            "image_shift2": 0.40,
            "image_missing": 0.10,
        },
        "mixed_state_damaged": {
            "clean": 0.50,
            "state_shift2": 0.40,
            "state_missing": 0.10,
        },
        "mixed_action_damaged": {
            "clean": 0.50,
            "action_shift2": 0.40,
            "action_missing": 0.10,
        },
        "mixed_three_corruptions": {
            "clean": 0.20,
            "image_shift2": 0.20,
            "state_shift2": 0.20,
            "action_shift2": 0.25,
            "image_missing": 0.05,
            "state_missing": 0.05,
            "action_missing": 0.05,
        },
    }
    for name, weights in expected.items():
        components = conditions[name]["components"]
        assert {item["name"]: item["weight"] for item in components} == weights
        assert sum(item["weight"] for item in components) == pytest.approx(1.0, abs=1e-12)
        for item in components:
            if item["name"].endswith("_missing"):
                assert item["config"] == {
                    "type": "modality_missing",
                    "target": item["name"].removesuffix("_missing"),
                    "probability": 1.0,
                }


def test_bare_report_only_reuses_saved_study_configuration(tmp_path: Path, monkeypatch) -> None:
    saved = {
        "started_at": "2026-09-12T00:00:00+00:00",
        "datasets": [study_runner.DATASETS[0]],
        "models": ["act_lite"],
        "conditions": ["clean"],
        "seeds": [7, 17, 27],
        "split_seed": 11,
        "episode_limit": None,
        "frames_per_episode_limit": None,
        "max_image_size": 64,
        "epochs": 10,
        "bc_batch_size": 64,
        "act_batch_size": 256,
        "device": "cpu",
        "mixture_segment_length": 16,
    }
    (tmp_path / "study.json").write_text(json.dumps(saved), encoding="utf-8")
    runs = [
        {
            "dataset": "xarm_lift_medium",
            "model": "act_lite",
            "condition": "clean",
            "seed": seed,
        }
        for seed in saved["seeds"]
    ]
    captured = {}
    monkeypatch.setattr(study_runner, "_load_compatible_runs", lambda *_: (runs, []))

    def fake_write_report(loaded_runs, study, output_root):
        captured["runs"] = loaded_runs
        captured["study"] = study
        return output_root / "report.html"

    monkeypatch.setattr(study_runner, "write_report", fake_write_report)
    args = study_runner.parser().parse_args(["--output", str(tmp_path), "--report-only"])

    result = study_runner.run_study(args)

    assert result == tmp_path / "report.html"
    assert len(captured["runs"]) == 3
    assert captured["study"]["seeds"] == [7, 17, 27]
    assert captured["study"]["episode_limit"] is None
    assert captured["study"]["frames_per_episode_limit"] is None
    assert captured["study"]["epochs"] == 10
    assert captured["study"]["act_batch_size"] == 256
    assert captured["study"]["started_at"] == saved["started_at"]


def test_study_yaml_and_shared_resolver_have_same_model_and_training_values():
    raw = load_config(Path(__file__).resolve().parents[2] / "configs/train/real_dataset_study.yaml")
    model = study_runner.model_config("quality_full", 6, 3, raw)
    actual = study_runner.training_config("quality_full", 7, 10, "cpu", model, 64, 256, raw)
    expected = resolve_training_config(
        raw,
        model_overrides={"name": "quality_full", "state_dim": 6, "action_dim": 3},
        training_overrides={"seed": 7, "device": "cpu"},
    )
    assert model == expected["model"]
    for key in (
        "epochs",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "horizon",
        "loss",
        "lambda_smooth",
        "quality_weighted_loss",
        "quality_transform",
        "num_workers",
        "validation_split",
        "test_split",
        "split_seed",
    ):
        assert actual[key] == expected["training"][key]


def test_old_study_cannot_be_overwritten_by_new_training(tmp_path):
    manifest = tmp_path / "study.json"
    original = '{"source_fingerprint": "historical", "epochs": 10}'
    manifest.write_text(original)
    args = study_runner.parser().parse_args(["--output", str(tmp_path)])
    with pytest.raises(ValueError, match="new --output"):
        study_runner.run_study(args)
    assert manifest.read_text() == original


def test_study_yaml_cli_overrides_and_real_training_roundtrip(tmp_path, monkeypatch):
    episodes = generate_demo_episodes(num_episodes=6, length=8, image_size=8)
    metadata = {
        "episodes": 6,
        "frames": 48,
        "source_episodes": 6,
        "source_frames": 48,
        "state_dim": 6,
        "action_dim": 3,
        "camera_count": 1,
        "fps": 30,
    }
    monkeypatch.setattr(
        study_runner,
        "validate_single_task_dataset",
        lambda *_: {"total_tasks": 1, "task_labels": ["test"]},
    )
    monkeypatch.setattr(
        study_runner, "load_local_lerobot_dataset", lambda *a, **k: (episodes, metadata)
    )
    config_path = save_config(
        {
            "model": {"name": "act_lite", "horizon": 3, "hidden_dim": 16},
            "training": {
                "epochs": 5,
                "batch_size": 32,
                "learning_rate": 0.0007,
                "validation_split": 0.2,
                "test_split": 0.2,
            },
            "study": {"seeds": [7], "episode_limit": None, "frames_per_episode_limit": None},
        },
        tmp_path / "study.yaml",
    )
    output = tmp_path / "runs"
    args = study_runner.parser().parse_args(
        [
            "--config",
            str(config_path),
            "--output",
            str(output),
            "--skip-download",
            "--datasets",
            "xarm_lift_medium",
            "--models",
            "act_lite",
            "--conditions",
            "clean",
            "mixed_state_damaged",
            "--device",
            "cpu",
            "--epochs",
            "1",
            "--act-batch-size",
            "4",
        ]
    )
    study_runner.run_study(args)
    manifests = sorted(output.glob("*/*/*/*/run.json"))
    assert len(manifests) == 2
    stats = []
    for path in manifests:
        run = json.loads(path.read_text())
        payload = torch.load(path.parent / "checkpoint.pt", weights_only=False)
        assert payload["config"] == run["training_config"]
        cfg = payload["config"]
        assert run["checkpoint"] == "checkpoint.best.pt"
        assert run["resume_checkpoint"] == "checkpoint.pt"
        selected = torch.load(path.parent / run["checkpoint"], weights_only=False)
        assert selected["checkpoint_kind"] == "best"
        assert run["checkpoint_selection"] == "best_validation_loss"
        assert cfg["epochs"] == 1 and cfg["batch_size"] == 4
        assert cfg["horizon"] == 3 and cfg["model"]["hidden_dim"] == 16
        assert cfg["learning_rate"] == 0.0007
        assert len(cfg["episode_split"]["train"]) == 4
        assert (path.parent / "checkpoint.config.json").is_file()
        stats.append(payload["stats"])
    assert stats[0] == stats[1]
    # Same resolved inputs reuse the checkpoints without another optimizer step.
    times = [(p.parent / "checkpoint.pt").stat().st_mtime_ns for p in manifests]
    study_runner.run_study(args)
    assert times == [(p.parent / "checkpoint.pt").stat().st_mtime_ns for p in manifests]
    report_args = study_runner.parser().parse_args(["--output", str(output), "--report-only"])
    study_runner.run_study(report_args)
    saved_study = json.loads((output / "study.json").read_text())
    assert saved_study["epochs"] == 1 and saved_study["act_batch_size"] == 4
    assert saved_study["validation_split"] == saved_study["test_split"] == 0.2
