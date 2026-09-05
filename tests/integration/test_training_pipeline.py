import json

import torch

from sync2act.data import generate_demo_episodes
from sync2act.pipeline import run_demo
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
    expected_mean = torch.cat(
        [episode["observation.state"] for episode in train_episodes]
    ).mean(0)
    assert torch.allclose(torch.tensor(payload["stats"]["state_mean"]), expected_mean)


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
