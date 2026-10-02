import json

import pytest
import torch

from sync2act.corruptions import state_anomaly
from sync2act.data import generate_demo_episodes
from sync2act.data.dataset import compute_stats
from sync2act.pipeline import run_demo, run_training
from sync2act.policies import BCMLP, ACTLite
from sync2act.training import load_checkpoint, train_policy


def test_bc_training_and_resume(tmp_path):
    episodes = generate_demo_episodes(num_episodes=2, length=18, image_size=16)
    config = {
        "epochs": 2,
        "max_steps": 8,
        "batch_size": 8,
        "learning_rate": 0.01,
        "device": "cpu",
        "seed": 3,
        "checkpoint_name": "bc_mlp_clean_test.pt",
    }
    model = BCMLP(6, 3, hidden_dim=32)
    result = train_policy(model, episodes, config, tmp_path / "first")
    assert result.checkpoint.exists()
    assert result.checkpoint.name == "bc_mlp_clean_test.pt"
    restored = BCMLP(6, 3, hidden_dim=32)
    payload = load_checkpoint(result.checkpoint, restored)
    resumed = train_policy(
        restored,
        episodes,
        {**config, "epochs": 3, "max_steps": None},
        tmp_path / "second",
        resume_from=result.checkpoint,
    )
    assert resumed.history[-1]["step"] > payload["step"]


def test_bc_can_overfit_a_tiny_batch():
    torch.manual_seed(9)
    state = torch.randn(8, 6)
    target = torch.stack(
        [state[:, 0] - state[:, 1], state[:, 2] * 0.5, -state[:, 3]], dim=1
    ).unsqueeze(1)
    model = BCMLP(6, 3, hidden_dim=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    initial = torch.nn.functional.mse_loss(model(state), target).item()
    for _ in range(100):
        loss = torch.nn.functional.mse_loss(model(state), target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert torch.nn.functional.mse_loss(model(state), target).item() < initial * 0.05


def test_act_lite_training_steps_lower_loss(tmp_path):
    episodes = generate_demo_episodes(num_episodes=2, length=16, image_size=16)
    model = ACTLite(6, 3, horizon=4, hidden_dim=32, num_layers=1)
    events = []
    train_policy(
        model,
        episodes,
        {
            "epochs": 3,
            "batch_size": 8,
            "learning_rate": 0.003,
            "horizon": 4,
            "device": "cpu",
            "seed": 5,
        },
        tmp_path,
        events.append,
    )
    losses = [event["train_loss"] for event in events if not event.get("epoch_complete")]
    assert min(losses[-4:]) < max(losses[:4])


def test_training_accepts_all_camera_views(tmp_path):
    episodes = generate_demo_episodes(num_episodes=2, length=8, image_size=16)
    for episode in episodes:
        first = episode["observation.image"]
        episode["observation.image"] = torch.stack([first, 1.0 - first], dim=1)
    model = BCMLP(6, 3, input_mode="image_state", hidden_dim=16)
    result = train_policy(
        model,
        episodes,
        {
            "epochs": 1,
            "max_steps": 1,
            "batch_size": 4,
            "learning_rate": 0.001,
            "device": "cpu",
            "seed": 2,
        },
        tmp_path,
    )
    assert result.checkpoint.exists()


def test_explicit_clean_validation_uses_training_statistics(tmp_path):
    episodes = generate_demo_episodes(num_episodes=3, length=12, image_size=16)
    train_episodes = [episodes[0], episodes[1]]
    validation_episodes = [episodes[2]]
    model = BCMLP(6, 3, hidden_dim=16)
    result = train_policy(
        model,
        train_episodes,
        {
            "epochs": 1,
            "max_steps": 1,
            "batch_size": 8,
            "learning_rate": 0.001,
            "device": "cpu",
            "seed": 4,
        },
        tmp_path,
        validation_episodes=validation_episodes,
    )
    payload = torch.load(result.checkpoint, map_location="cpu", weights_only=False)
    expected_mean = torch.cat([episode["observation.state"] for episode in train_episodes]).mean(0)
    assert torch.allclose(torch.tensor(payload["stats"]["state_mean"]), expected_mean)


def test_corrupted_training_can_use_frozen_clean_statistics(tmp_path):
    clean = generate_demo_episodes(num_episodes=2, length=12, image_size=16)
    clean_stats = compute_stats(clean)
    corrupted = [
        state_anomaly(episode, mode="spike", probability=1.0, magnitude=100.0, seed=index)
        for index, episode in enumerate(clean)
    ]
    result = train_policy(
        BCMLP(6, 3, hidden_dim=16),
        corrupted,
        {
            "epochs": 1,
            "max_steps": 1,
            "batch_size": 8,
            "learning_rate": 0.001,
            "device": "cpu",
            "seed": 4,
        },
        tmp_path,
        normalization_stats=clean_stats,
    )
    payload = torch.load(result.checkpoint, map_location="cpu", weights_only=False)
    assert torch.allclose(torch.tensor(payload["stats"]["state_mean"]), clean_stats.state_mean)


def test_resume_reuses_checkpoint_statistics(tmp_path):
    clean = generate_demo_episodes(num_episodes=2, length=12, image_size=16)
    config = {
        "epochs": 1,
        "max_steps": 1,
        "batch_size": 8,
        "learning_rate": 0.001,
        "device": "cpu",
        "seed": 4,
    }
    first = train_policy(BCMLP(6, 3, hidden_dim=16), clean, config, tmp_path / "first")
    first_payload = torch.load(first.checkpoint, map_location="cpu", weights_only=False)
    corrupted = [
        state_anomaly(episode, mode="spike", probability=1.0, magnitude=100.0, seed=index)
        for index, episode in enumerate(clean)
    ]
    resumed = train_policy(
        BCMLP(6, 3, hidden_dim=16),
        corrupted,
        {**config, "epochs": 2, "max_steps": None},
        tmp_path / "resumed",
        resume_from=first.checkpoint,
    )
    resumed_payload = torch.load(resumed.checkpoint, map_location="cpu", weights_only=False)
    assert resumed_payload["stats"] == first_payload["stats"]


def test_end_to_end_demo_writes_real_artifacts(tmp_path):
    result = run_demo(tmp_path / "demo")
    run = result["run"]
    assert run["metrics"]["evaluation_scope"] == "offline-only"
    assert "rollout_success_rate" not in run["metrics"]
    assert "rollout_return" not in run["metrics"]
    assert "rollout_completion_steps" not in run["metrics"]
    assert (tmp_path / "demo" / "checkpoint.pt").exists()
    assert (tmp_path / "demo" / "report.html").exists()
    assert json.loads((tmp_path / "demo" / "run.json").read_text())["metrics"]["action_mse"] >= 0


def test_yaml_pipeline_reproducible_and_evaluates_clean_holdout(tmp_path):
    from sync2act.config import load_config, save_config

    config_path = save_config(
        {
            "model": {"name": "bc_mlp", "hidden_dim": 16},
            "dataset": {"num_episodes": 6, "length": 8, "image_size": 8, "seed": 7},
            "training": {"epochs": 1, "batch_size": 7, "device": "cpu", "seed": 11},
            "corruption": {
                "type": "state_anomaly",
                "mode": "spike",
                "probability": 1.0,
                "magnitude": 1000,
            },
        },
        tmp_path / "train.yaml",
    )
    config = load_config(config_path)
    model_a, test, a = run_training(config, tmp_path / "a")
    torch.randn(23)
    model_b, _, b = run_training(config, tmp_path / "b")
    assert all(
        torch.equal(value, model_b.state_dict()[key]) for key, value in model_a.state_dict().items()
    )
    assert a.history == b.history
    saved = torch.load(a.checkpoint, weights_only=False)
    split = saved["config"]["episode_split"]
    clean = generate_demo_episodes(**config["dataset"])
    assert len(test) == len(split["test"]) == 1
    assert torch.equal(test[0]["action"], clean[split["test"][0]]["action"])
    expected = compute_stats([clean[i] for i in split["train"]])
    assert saved["stats"] == expected.to_dict()
    snapshot = json.loads(a.checkpoint.with_suffix(".config.json").read_text())
    assert snapshot["training"]["episode_split"] == split
    assert snapshot["model"]["hidden_dim"] == 16


def test_low_level_training_splits_whole_episodes_before_normalizing(tmp_path):
    episodes = generate_demo_episodes(num_episodes=3, length=8, image_size=8)
    result = train_policy(
        BCMLP(6, 3, hidden_dim=16),
        episodes,
        {"epochs": 1, "batch_size": 8, "device": "cpu"},
        tmp_path,
    )
    saved = torch.load(result.checkpoint, weights_only=False)
    split = saved["config"]["episode_split"]
    assert len(split["validation"]) == 1 and len(split["train"]) == 2
    assert not set(split["train"]) & set(split["validation"])
    assert saved["stats"] == compute_stats([episodes[i] for i in split["train"]]).to_dict()
    assert saved["config"]["model"]["hidden_dim"] == 16


def test_resume_rejects_old_loss_protocol_without_writing_new_artifacts(tmp_path):
    episodes = generate_demo_episodes(num_episodes=2, length=4, image_size=8)
    config = {"epochs": 1, "batch_size": 4, "device": "cpu"}
    first = train_policy(BCMLP(6, 3, hidden_dim=16), episodes, config, tmp_path / "first")
    payload = torch.load(first.checkpoint, weights_only=False)
    payload["config"].pop("training_protocol_version")
    legacy_path = tmp_path / "legacy.pt"
    torch.save(payload, legacy_path)
    with pytest.raises(ValueError, match="different training protocol"):
        train_policy(
            BCMLP(6, 3, hidden_dim=16), episodes, config, tmp_path / "new", resume_from=legacy_path
        )
    assert not (tmp_path / "new").exists()
