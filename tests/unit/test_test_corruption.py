import json
import threading

import pandas as pd
import pytest
import torch

from sync2act.data.dataset import compute_stats
from sync2act.data.episode import clone_episode
from sync2act.data.synthetic import generate_demo_episodes
from sync2act.evaluation.test_corruption import (
    TestCorruptionConfig,
    evaluate_paired,
    prepare_test_condition,
    preset_config,
)


def episodes():
    data = generate_demo_episodes(num_episodes=3, length=40, image_size=8)
    for e in data:
        e["observation.image"] = e["observation.image"].unsqueeze(1).repeat(1, 2, 1, 1, 1)
    return data


def test_corruption_preserves_original_labels_timestamps_and_unselected_camera():
    data = episodes()
    originals = [clone_episode(e) for e in data]
    config = preset_config("image", camera_index=1, seed=107)
    prepared = prepare_test_condition(data, config)
    assert prepared.affected.any()
    for before, original, damaged in zip(originals, data, prepared.episodes, strict=True):
        for key, value in before.items():
            if torch.is_tensor(value):
                assert torch.equal(value, original[key])
        for key in ("action", "timestamp", "observation.state"):
            assert torch.equal(damaged[key], original[key])
        assert torch.equal(damaged["observation.image"][:, 0], original["observation.image"][:, 0])
    repeated = prepare_test_condition(data, config)
    assert repeated.manifest == prepared.manifest
    assert torch.equal(repeated.affected, prepared.affected)
    changed_seed = prepare_test_condition(data, preset_config("image", camera_index=1, seed=108))
    assert changed_seed.manifest["episodes"] != prepared.manifest["episodes"]


def test_delay_is_past_only_and_boundary_masked_without_label_shift():
    data = episodes()
    result = prepare_test_condition(data, TestCorruptionConfig(
        condition="state", delay_fraction=1, missing_fraction=0, shift=2,
    ))
    for original, damaged in zip(data, result.episodes, strict=True):
        assert torch.equal(damaged["observation.state"][2:], original["observation.state"][:-2])
        assert damaged["state_missing_mask"][:2].all()
        assert (damaged["state_time_offset"] <= 0).all()
        assert torch.equal(damaged["action"], original["action"])


def test_unknown_metadata_does_not_repair_damaged_observations_or_mutate_oracle():
    data = episodes()
    oracle = prepare_test_condition(data, preset_config("mixed"))
    unknown = prepare_test_condition(data, preset_config("mixed", quality_mode="unknown"))
    assert torch.equal(oracle.affected, unknown.affected)
    for a, b in zip(oracle.episodes, unknown.episodes, strict=True):
        for key in ("observation.image", "observation.state", "action"):
            assert torch.equal(a[key], b[key])
        for modality in ("image", "state"):
            assert (b[f"{modality}_quality"] == 1).all()
            assert not b[f"{modality}_missing_mask"].any()
            assert not b[f"{modality}_time_offset"].any()
    assert any((e["state_quality"] < 1).any() for e in oracle.episodes)


@pytest.mark.parametrize("overrides", [
    {"condition": "action"}, {"shift": -1}, {"shift": 0},
    {"delay_fraction": 0.8, "missing_fraction": 0.3},
    {"missing_fraction": float("nan")}, {"quality_mode": "guess"},
])
def test_invalid_protocol_is_rejected(overrides):
    with pytest.raises(ValueError):
        prepare_test_condition(episodes(), TestCorruptionConfig(**overrides))


class StatePolicy(torch.nn.Module):
    horizon = 1

    def forward(self, state, **kwargs):
        return state[:, :3].unsqueeze(1)


def test_paired_evaluation_uses_frozen_stats_and_same_targets(tmp_path):
    data = episodes()
    stats = compute_stats(generate_demo_episodes(num_episodes=2, length=15, seed=81))
    result = evaluate_paired(StatePolicy(), data, preset_config("state"),
                             stats=stats, output_dir=tmp_path, batch_size=13)
    clean = pd.read_csv(tmp_path / "clean/predictions.csv")
    dirty = pd.read_csv(tmp_path / "test/predictions.csv")
    targets = [c for c in clean if c.startswith("target_")]
    pd.testing.assert_frame_equal(clean[["episode", "step", *targets]], dirty[["episode", "step", *targets]])
    assert not clean["prediction_0"].equals(dirty["prediction_0"])
    assert result["normalization"] == stats.to_dict()
    assert result["test"]["mse_ratio_to_clean_test"] == pytest.approx(
        result["test"]["action_mse"] / result["clean"]["action_mse"])
    groups = result["test"]["frame_groups"]
    assert sum(group["frames"] for group in groups.values()) == len(clean)
    weighted_mse = sum(g["action_mse"] * g["frames"] for g in groups.values()) / len(clean)
    assert weighted_mse == pytest.approx(result["test"]["action_mse"])
    assert json.loads((tmp_path / "comparison.json").read_text())["test_config"]["condition"] == "state"
    with pytest.raises(ValueError, match="normalization"):
        evaluate_paired(StatePolicy(), data, preset_config("state"), stats=None, output_dir=tmp_path)


def test_cancellation_does_not_claim_completed_comparison(tmp_path):
    stop = threading.Event()
    stop.set()
    data = episodes()
    with pytest.raises(InterruptedError):
        evaluate_paired(StatePolicy(), data, preset_config("state"), stats=compute_stats(data),
                        output_dir=tmp_path, stop_event=stop)
    assert not (tmp_path / "comparison.json").exists()


@pytest.mark.parametrize("batch_size", [1, 13, 47, 512])
def test_vectorized_batches_match_original_window_loader(batch_size):
    from torch.utils.data import DataLoader

    from sync2act.data.dataset import EpisodeWindowDataset
    from sync2act.evaluation.evaluator import _evaluation_batches

    prepared = prepare_test_condition(episodes(), preset_config("mixed"))
    stats = compute_stats(prepared.episodes)
    dataset = EpisodeWindowDataset(prepared.episodes, horizon=8, stats=stats)
    original = list(DataLoader(dataset, batch_size=batch_size))
    vectorized = list(_evaluation_batches(prepared.episodes, stats, batch_size))
    assert len(original) == len(vectorized)
    for a, b in zip(original, vectorized, strict=True):
        for key in b:
            expected = a[key][:, :1] if key == "actions" else a[key]
            torch.testing.assert_close(expected, b[key], check_dtype=False, rtol=0, atol=0)
