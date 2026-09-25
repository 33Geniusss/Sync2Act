import importlib.util
import json
from pathlib import Path


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
    conditions = {
        item["name"]: item for item in study_runner.corruption_conditions([], 30.0)
    }

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
        assert sum(item["weight"] for item in components) == 1.0
        for item in components:
            if item["name"].endswith("_missing"):
                assert item["config"] == {
                    "type": "modality_missing",
                    "target": item["name"].removesuffix("_missing"),
                    "probability": 1.0,
                }


def test_bare_report_only_reuses_saved_study_configuration(
    tmp_path: Path, monkeypatch
) -> None:
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
    args = study_runner.parser().parse_args(
        ["--output", str(tmp_path), "--report-only"]
    )

    result = study_runner.run_study(args)

    assert result == tmp_path / "report.html"
    assert len(captured["runs"]) == 3
    assert captured["study"]["seeds"] == [7, 17, 27]
    assert captured["study"]["episode_limit"] is None
    assert captured["study"]["frames_per_episode_limit"] is None
    assert captured["study"]["epochs"] == 10
    assert captured["study"]["act_batch_size"] == 256
    assert captured["study"]["started_at"] == saved["started_at"]
