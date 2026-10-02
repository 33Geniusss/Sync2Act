"""Shared episode preparation and seeded model initialization for every frontend."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch

from sync2act.corruptions import apply_corruption, transform_quality_annotations
from sync2act.data.dataset import NormalizationStats, compute_stats
from sync2act.data.episode import Episode
from sync2act.data.split import split_episode_indices
from sync2act.policies import build_policy


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_training_policy(model_config: dict, training_config: dict):
    seed_everything(int(training_config["seed"]))
    return build_policy(model_config)


def describe_policy(model) -> dict:
    """Recover constructor settings for callers supplying an already-built policy."""
    common = {"state_dim": model.state_dim, "action_dim": model.action_dim}
    if not hasattr(model, "horizon"):
        return {
            **common,
            "name": "bc_mlp",
            "input_mode": model.input_mode,
            "hidden_dim": model.state_encoder[0].out_features,
            "horizon": 1,
        }
    layer = model.transformer.layers[0]
    return {
        **common,
        "name": "quality_act" if hasattr(model, "use_quality_features") else "act_lite",
        "hidden_dim": model.hidden_dim,
        "horizon": model.horizon,
        "num_layers": len(model.transformer.layers),
        "num_heads": layer.self_attn.num_heads,
        "dropout": layer.dropout.p,
        "use_quality_features": getattr(model, "use_quality_features", False),
    }


@dataclass
class PreparedEpisodes:
    train: list[Episode]
    validation: list[Episode]
    test: list[Episode]
    split: dict[str, list[int]]
    stats: NormalizationStats
    working: list[Episode]


def prepare_training_episodes(
    clean_episodes: list[Episode],
    training_config: dict,
    corruption: dict | None = None,
) -> PreparedEpisodes:
    """Split first, freeze clean training statistics, then damage only training episodes."""
    split = split_episode_indices(
        len(clean_episodes),
        training_config["validation_split"],
        training_config["test_split"],
        training_config.get("split_seed", training_config["seed"]),
    )
    clean_train = [clean_episodes[i] for i in split["train"]]
    stats = compute_stats(clean_train)
    working = list(clean_episodes)
    if corruption:
        for i in split["train"]:
            working[i] = apply_corruption(
                clean_episodes[i],
                {**corruption, "seed": int(corruption.get("seed", training_config["seed"])) + i},
            )
    train = [working[i] for i in split["train"]]
    mode = training_config.get("quality_transform", "identity")
    if mode != "identity":
        train = transform_quality_annotations(train, mode, seed=training_config["seed"] + 7919)
        for i, episode in zip(split["train"], train, strict=True):
            working[i] = episode
    return PreparedEpisodes(
        train,
        [clean_episodes[i] for i in split["validation"]],
        [clean_episodes[i] for i in split["test"]],
        split,
        stats,
        working,
    )
