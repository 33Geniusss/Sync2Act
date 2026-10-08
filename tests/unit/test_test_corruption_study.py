import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from sync2act.reporting.test_corruption import write_test_corruption_report


def runner():
    path = Path(__file__).resolve().parents[2] / "tools/evaluate_test_corruption_study.py"
    spec = importlib.util.spec_from_file_location("test_corruption_study_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prespecified_study_matrix_and_distinct_damage_seeds():
    study = runner()
    models = ["act_lite", "quality_input", "quality_weighted_loss", "quality_full", "quality_shuffled", "quality_constant"]
    conditions = ["clean", "mixed_image_damaged", "mixed_state_damaged", "mixed_action_damaged", "mixed_three_corruptions"]
    counts = {}
    for _dataset in range(3):
        for model in models:
            for condition in conditions:
                for seed in (7, 17, 27):
                    run = {"model": model, "condition": condition, "seed": seed}
                    cases = list(study.test_cases(run, [107, 117, 127]))
                    assert cases[0][0] == "baseline"
                    for group, _, config in cases:
                        counts[group] = counts.get(group, 0) + 1
                        if group != "baseline":
                            assert config.seed not in (7, 17, 27)
                            assert config.condition in ("image", "state", "mixed")
    assert counts == {"baseline": 270, "main": 2430, "metadata": 810, "severity": 648}


def test_historical_audit_rejects_label_or_prediction_changes(tmp_path):
    study = runner()
    original = pd.DataFrame({"episode": [0, 0], "step": [0, 1], "prediction_0": [1, 2], "target_0": [2, 3]})
    original.to_csv(tmp_path / "old.csv", index=False)
    original.to_csv(tmp_path / "new.csv", index=False)
    assert study.historical_audit(tmp_path / "old.csv", tmp_path / "new.csv", 1, 1)["passed"]
    original.loc[0, "target_0"] = 20
    original.to_csv(tmp_path / "new.csv", index=False)
    assert not study.historical_audit(tmp_path / "old.csv", tmp_path / "new.csv", 1, 1)["passed"]


def test_report_averages_damage_seeds_before_training_seeds_and_is_bilingual(tmp_path):
    records = []
    for training_seed, ratios in ((7, [1.0, 3.0]), (17, [5.0, 7.0])):
        for damage_seed, ratio in zip((107, 117), ratios, strict=True):
            records.append({"dataset": "toy", "model": "act_lite", "train_condition": "clean",
                            "train_seed": training_seed, "group": "main", "severity": "medium",
                            "test_condition": "state", "quality_mode": "oracle", "damage_seed": damage_seed,
                            "actual_affected_fraction": 0.5,
                            "metrics": {"action_mse": ratio, "action_mae": 0.5, "mse_ratio_to_clean_test": ratio,
                                        "mse_delta_from_clean_test": ratio - 1,
                                        "frame_groups": {name: {"frames": 1, "action_mse": ratio, "action_mae": 0.5}
                                                         for name in ("affected", "unaffected")}}})
    summary = {"completed_evaluations": 4, "expected_evaluations": 4, "baseline_audits_passed": 0, "pilot": True}
    write_test_corruption_report(records, summary, tmp_path)
    aggregate = pd.read_csv(tmp_path / "aggregate.csv").iloc[0]
    assert aggregate.mse_ratio == 4
    assert aggregate.training_seed_sd == pytest.approx(8**0.5)
    assert aggregate.mean_within_training_seed_damage_sd == pytest.approx(2**0.5)
    assert aggregate.training_seed_count == 2
    assert len(json.loads((tmp_path / "results.json").read_text())) == 4
    for name in ("report.html", "report_en.html", "severity_curves.svg", "severity_curves.png"):
        assert (tmp_path / name).stat().st_size > 0
    assert "Small pilot only" in (tmp_path / "report_en.html").read_text(encoding="utf-8")
