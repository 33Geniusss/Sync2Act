import torch

from sync2act.corruptions import (
    action_noise,
    apply_corruption,
    frame_drop,
    state_anomaly,
    temporal_shift,
)
from sync2act.data import clone_episode, generate_demo_episodes, validate_episode


def test_synthetic_episode_contract():
    episodes = generate_demo_episodes(num_episodes=2, length=12, seed=2)
    assert len(episodes) == 2
    assert validate_episode(episodes[0]) == 12
    assert episodes[0]["observation.image"].shape == (12, 3, 32, 32)


def test_corruptions_are_reproducible_and_non_mutating():
    original = generate_demo_episodes(num_episodes=1, length=24)[0]
    snapshot = clone_episode(original)
    configs = [
        {"type": "temporal_shift", "target": "image", "shift": -2, "boundary": "mask", "seed": 4},
        {"type": "frame_drop", "probability": 0.4, "replacement": "interpolation", "seed": 4},
        {"type": "action_noise", "mode": "gaussian", "sigma": 0.2, "seed": 4},
        {"type": "state_anomaly", "mode": "spike", "probability": 0.4, "seed": 4},
    ]
    for config in configs:
        first, second = apply_corruption(original, config), apply_corruption(original, config)
        for key in [
            "observation.image",
            "observation.state",
            "action",
            "timestamp",
            "quality_score",
            "missing_mask",
        ]:
            assert torch.equal(first[key], second[key])
            assert torch.equal(original[key], snapshot[key])
        assert first["provenance"]["seed"] == 4


def test_quality_and_masks_match_damage():
    episode = generate_demo_episodes(num_episodes=1, length=20)[0]
    dropped = frame_drop(episode, probability=1.0, replacement="zero", seed=1)
    assert dropped["missing_mask"].all()
    assert (dropped["quality_score"] < 1).all()
    assert (dropped["observation.image"] == 0).all()
    shifted = temporal_shift(episode, target="state", shift=3, boundary="mask")
    assert shifted["missing_mask"][:3].all()
    assert not shifted["missing_mask"][3:].any()


def test_all_damage_variants_keep_episode_boundary():
    episode = generate_demo_episodes(num_episodes=1, length=20)[0]
    variants = [
        temporal_shift(episode, "action", 2, "drop"),
        action_noise(episode, "spike", sigma=2, probability=0.5, seed=1),
        action_noise(episode, "delay", delay=2, random_delay=True, seed=1),
        state_anomaly(episode, "missing", duration=3, seed=1),
        state_anomaly(episode, "stuck", dimension=1, duration=3, seed=1),
        state_anomaly(episode, "timestamp_jitter", seed=1),
    ]
    assert validate_episode(variants[0]) == 18
    assert all(validate_episode(item) in {18, 20} for item in variants)
