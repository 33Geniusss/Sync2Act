from copy import deepcopy

import pytest
import torch

from sync2act.config import resolve_training_config
from sync2act.data.dataset import compute_stats
from sync2act.data.synthetic import generate_demo_episodes
from sync2act.training.setup import build_training_policy, prepare_training_episodes


def test_shared_resolution_has_explicit_precedence_and_provenance():
    raw = {
        "model": {"name": "act_lite", "horizon": 4},
        "training": {"epochs": 5, "batch_size": 16, "seed": 11, "device": "cpu"},
    }
    original = deepcopy(raw)
    resolved = resolve_training_config(
        raw, training_overrides={"epochs": 8}, source="YAML", override_source="CLI"
    )
    assert raw == original
    assert resolved["training"]["epochs"] == 8
    assert resolved["training"]["batch_size"] == 16
    assert resolved["training"]["horizon"] == 4
    assert resolved["training"]["split_seed"] == 11
    assert resolved["training"]["num_workers"] == 0
    assert resolved["config_sources"]["training.epochs"] == "CLI"
    assert resolved["config_sources"]["training.batch_size"] == "YAML"
    assert resolved["config_sources"]["training.weight_decay"] == "shared defaults"
    assert resolve_training_config(resolved) == resolved


def test_conflicting_horizons_are_rejected():
    with pytest.raises(ValueError, match="horizon"):
        resolve_training_config(
            {"model": {"name": "act_lite", "horizon": 8}, "training": {"horizon": 16}}
        )


def test_seed_is_applied_before_model_initialization():
    config = resolve_training_config({"model": {"name": "act_lite"}, "training": {"seed": 23}})
    a = build_training_policy(config["model"], config["training"])
    torch.randn(100)
    b = build_training_policy(config["model"], config["training"])
    assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())


def test_partition_freezes_clean_stats_and_never_corrupts_holdouts():
    clean = generate_demo_episodes(num_episodes=6, length=8, image_size=8)
    config = resolve_training_config({"training": {"seed": 17, "split_seed": 7}})["training"]
    prepared = prepare_training_episodes(
        clean,
        config,
        {
            "type": "state_anomaly",
            "mode": "spike",
            "probability": 1.0,
            "magnitude": 1000,
        },
    )
    groups = [set(prepared.split[key]) for key in ("train", "validation", "test")]
    assert set.union(*groups) == set(range(6))
    assert all(not groups[i] & groups[j] for i in range(3) for j in range(i))
    expected = compute_stats([clean[i] for i in prepared.split["train"]])
    assert torch.equal(prepared.stats.state_mean, expected.state_mean)
    assert not torch.equal(compute_stats(prepared.train).state_mean, expected.state_mean)
    for key in ("validation", "test"):
        assert all(prepared.working[i] is clean[i] for i in prepared.split[key])
    assert prepare_training_episodes(clean, {**config, "seed": 27}).split == prepared.split
