import pytest
import torch

from sync2act.corruptions import (
    action_noise,
    apply_corruption,
    build_mixed_quality_dataset,
    frame_drop,
    state_anomaly,
    temporal_shift,
    transform_quality_annotations,
)
from sync2act.data import clone_episode, generate_demo_episodes, validate_episode
from sync2act.data.dataset import EpisodeWindowDataset


def test_synthetic_episode_contract():
    episodes = generate_demo_episodes(num_episodes=2, length=12, seed=2)
    assert len(episodes) == 2
    assert validate_episode(episodes[0]) == 12
    assert episodes[0]["observation.image"].shape == (12, 3, 32, 32)
    assert episodes[0]["image_quality"].shape == (12, 1)
    assert episodes[0]["state_quality"].eq(1).all()
    assert episodes[0]["action_label_quality"].eq(1).all()


def test_multi_camera_episode_contract():
    episode = generate_demo_episodes(num_episodes=1, length=12, seed=2)[0]
    episode["observation.image"] = episode["observation.image"].unsqueeze(1).repeat(1, 3, 1, 1, 1)
    assert validate_episode(episode) == 12


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
            "time_offset",
            "image_time_offset",
            "state_time_offset",
            "action_label_time_offset",
            "image_quality",
            "state_quality",
            "action_label_quality",
            "quality_score",
            "missing_mask",
            "image_missing_mask",
            "state_missing_mask",
            "action_label_missing_mask",
        ]:
            assert torch.equal(first[key], second[key])
            assert torch.equal(original[key], snapshot[key])
        assert first["provenance"]["seed"] == 4


def test_quality_and_masks_match_damage():
    episode = generate_demo_episodes(num_episodes=1, length=20)[0]
    dropped = frame_drop(episode, probability=1.0, replacement="zero", seed=1)
    assert dropped["missing_mask"].all()
    assert (dropped["quality_score"] < 1).all()
    assert (dropped["image_quality"] < 1).all()
    assert dropped["state_quality"].eq(1).all()
    assert dropped["action_label_quality"].eq(1).all()
    assert (dropped["observation.image"] == 0).all()
    shifted = temporal_shift(episode, target="state", shift=3, boundary="mask")
    assert shifted["missing_mask"][:3].all()
    assert not shifted["missing_mask"][3:].any()
    assert shifted["image_quality"].eq(1).all()
    assert (shifted["state_quality"] < 1).all()
    assert shifted["action_label_quality"].eq(1).all()


def test_temporal_shift_records_signed_time_offset_in_seconds():
    episode = generate_demo_episodes(num_episodes=1, length=20)[0]
    delayed = temporal_shift(episode, target="state", shift=2, boundary="mask")
    advanced = temporal_shift(episode, target="image", shift=-2, boundary="repeat")

    torch.testing.assert_close(delayed["time_offset"], torch.full((20,), -0.1))
    torch.testing.assert_close(delayed["state_time_offset"], torch.full((20,), -0.1))
    assert delayed["image_time_offset"].eq(0).all()
    assert delayed["action_label_time_offset"].eq(0).all()
    torch.testing.assert_close(advanced["time_offset"], torch.full((20,), 0.1))
    sample = EpisodeWindowDataset([delayed], horizon=2)[5]
    assert sample["time_offset"].item() == pytest.approx(-0.1)
    assert sample["image_quality"].shape == (1,)
    assert sample["state_quality"].shape == (1,)
    assert sample["action_label_quality"].shape == (2,)


def test_temporal_shift_quality_decreases_with_severity():
    episode = generate_demo_episodes(num_episodes=1, length=20)[0]
    shift_one = temporal_shift(episode, target="state", shift=1, boundary="repeat")
    shift_three = temporal_shift(episode, target="state", shift=3, boundary="repeat")

    expected_one = torch.exp(torch.tensor(-0.5))
    expected_three = torch.exp(torch.tensor(-1.5))
    torch.testing.assert_close(shift_one["state_quality"], expected_one.expand(20))
    torch.testing.assert_close(shift_three["state_quality"], expected_three.expand(20))
    assert shift_three["state_quality"].mean() < shift_one["state_quality"].mean()


def test_action_delay_marks_every_misaligned_frame_and_records_offset():
    episode = generate_demo_episodes(num_episodes=1, length=20)[0]
    delayed = action_noise(episode, mode="delay", delay=2, random_delay=False, seed=1)

    assert delayed["provenance"]["affected_indices"] == list(range(1, 20))
    assert delayed["quality_score"][0].item() == 1.0
    assert delayed["image_quality"].eq(1).all()
    assert delayed["state_quality"].eq(1).all()
    expected = torch.tensor([0.0, -0.05] + [-0.1] * 18)
    torch.testing.assert_close(delayed["time_offset"], expected)
    torch.testing.assert_close(delayed["action_label_time_offset"], expected)
    expected_quality = torch.exp(-expected.abs() / 0.1)
    torch.testing.assert_close(delayed["action_label_quality"], expected_quality)
    torch.testing.assert_close(delayed["quality_score"], expected_quality)


def test_opposite_modal_shifts_do_not_cancel_and_provenance_accumulates():
    episode = generate_demo_episodes(num_episodes=1, length=20)[0]
    state_shifted = temporal_shift(episode, target="state", shift=2, boundary="repeat")
    combined = temporal_shift(state_shifted, target="action", shift=-2, boundary="repeat")

    torch.testing.assert_close(combined["state_time_offset"], torch.full((20,), -0.1))
    torch.testing.assert_close(combined["action_label_time_offset"], torch.full((20,), 0.1))
    assert combined["time_offset"].abs().min().item() == pytest.approx(0.1)
    assert [event["parameters"]["target"] for event in combined["corruption_events"]] == [
        "state",
        "action",
    ]


def test_camera_specific_drop_only_marks_selected_camera():
    episode = generate_demo_episodes(num_episodes=1, length=8)[0]
    episode["observation.image"] = episode["observation.image"].unsqueeze(1).repeat(1, 2, 1, 1, 1)
    dropped = frame_drop(episode, probability=1.0, replacement="zero", camera_index=1, seed=1)
    assert not dropped["image_missing_mask"][:, 0].any()
    assert dropped["image_missing_mask"][:, 1].all()
    assert dropped["image_quality"][:, 0].eq(1).all()
    assert dropped["image_quality"][:, 1].eq(0.25).all()


def test_mixed_quality_builder_and_annotation_controls():
    episodes = generate_demo_episodes(num_episodes=2, length=16)
    components = [
        {"name": "clean", "weight": 0.5, "config": None},
        {
            "name": "state_shift",
            "weight": 0.5,
            "config": {
                "type": "temporal_shift",
                "target": "state",
                "shift": 2,
                "boundary": "repeat",
            },
        },
    ]
    mixed, manifest = build_mixed_quality_dataset(episodes, components, segment_length=4, seed=5)
    assert manifest["segment_counts"] == {"clean": 4, "state_shift": 4}
    assert any((episode["state_quality"] < 1).any() for episode in mixed)
    assert all(episode["image_quality"].eq(1).all() for episode in mixed)

    shuffled = transform_quality_annotations(mixed, "shuffled", seed=4)
    constant = transform_quality_annotations(mixed, "constant", seed=4)
    original_quality = torch.cat([episode["state_quality"] for episode in mixed])
    shuffled_quality = torch.cat([episode["state_quality"] for episode in shuffled])
    assert torch.equal(original_quality.sort().values, shuffled_quality.sort().values)
    assert all(episode["quality_score"].eq(1).all() for episode in constant)
    assert all(episode["time_offset"].eq(0).all() for episode in constant)


def test_quality_values_are_validated():
    episode = generate_demo_episodes(num_episodes=1, length=8)[0]
    episode["state_quality"][2] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        validate_episode(episode)


def test_timestamp_jitter_records_actual_time_offset():
    episode = generate_demo_episodes(num_episodes=1, length=20)[0]
    jittered = state_anomaly(episode, mode="timestamp_jitter", jitter_std=0.01, seed=3)

    torch.testing.assert_close(
        jittered["time_offset"], jittered["timestamp"] - episode["timestamp"]
    )
    assert jittered["time_offset"].abs().max() > 0
    expected_quality = torch.exp(-jittered["time_offset"].abs() / 0.1)
    torch.testing.assert_close(jittered["state_quality"], expected_quality)


def test_uint8_frame_interpolation_does_not_overflow():
    episode = generate_demo_episodes(num_episodes=1, length=3)[0]
    episode["observation.image"] = torch.full((3, 3, 4, 4), 250, dtype=torch.uint8)
    dropped = frame_drop(episode, probability=1.0, replacement="interpolation", seed=1)
    assert dropped["observation.image"].eq(250).all()


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
