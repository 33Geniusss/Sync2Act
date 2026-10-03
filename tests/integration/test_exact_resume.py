import random
import threading
from copy import deepcopy

import numpy as np
import pytest
import torch

from sync2act.config import resolve_training_config
from sync2act.data.synthetic import generate_demo_episodes
from sync2act.policies import ACTLite
from sync2act.training import load_checkpoint, train_policy
from sync2act.training.checkpoint import capture_rng_state, restore_rng_state
from sync2act.training.losses import LossSums
from sync2act.training.setup import build_training_policy, seed_everything


class RandomizedACT(ACTLite):
    """Exercise Python/NumPy RNG as well as Transformer dropout during training."""

    def forward(self, *args, **kwargs):
        output = super().forward(*args, **kwargs)
        if self.training:
            output = output * (0.9 + 0.05 * random.random() + 0.05 * np.random.random())
        return output


def settings():
    return resolve_training_config(
        {
            "model": {"name": "act_lite", "hidden_dim": 16, "horizon": 3, "dropout": 0.25},
            "training": {
                "epochs": 3,
                "batch_size": 4,
                "seed": 19,
                "device": "cpu",
                "checkpoint_interval_steps": 2,
            },
        }
    )


def make_model(config):
    seed_everything(config["training"]["seed"])
    return RandomizedACT(6, 3, hidden_dim=16, horizon=3, dropout=0.25, num_layers=1)


def assert_nested_equal(a, b):
    if torch.is_tensor(a):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_nested_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for left, right in zip(a, b, strict=True):
            assert_nested_equal(left, right)
    else:
        assert a == b


@pytest.mark.parametrize("pause_step", [1, 4, 6])
def test_mid_epoch_and_boundary_resume_match_continuous_training(tmp_path, pause_step):
    config = settings()
    episodes = generate_demo_episodes(num_episodes=3, length=7, image_size=8)
    # Two training episodes: four batches/epoch, including a short last batch.
    full = train_policy(make_model(config), episodes, config["training"], tmp_path / "full")
    paused = train_policy(
        make_model(config),
        episodes,
        {**config["training"], "max_steps": pause_step},
        tmp_path / "partial",
    )
    checkpoint = torch.load(paused.checkpoint, weights_only=False)
    assert checkpoint["resume_state"]["next_batch"] == pause_step % 4
    assert len(checkpoint["history"]) == pause_step // 4
    assert checkpoint["scheduler"]["last_epoch"] == pause_step // 4
    # Disturb every RNG before rebuilding the resumed model.
    random.random(), np.random.random(), torch.randn(100)
    resumed = train_policy(
        make_model(config),
        episodes,
        config["training"],
        tmp_path / "resumed",
        resume_from=paused.checkpoint,
    )
    expected = torch.load(full.checkpoint, weights_only=False)
    actual = torch.load(resumed.checkpoint, weights_only=False)
    for key in ("model", "optimizer", "scheduler", "history", "best", "step"):
        assert_nested_equal(expected[key], actual[key])
    assert_nested_equal(expected["resume_state"]["rng_state"], actual["resume_state"]["rng_state"])
    assert actual["step"] == 12
    assert actual["resume_state"]["status"] == "complete"
    assert resumed.best_checkpoint.is_file()


def test_stop_event_before_first_batch_and_repeated_resumes(tmp_path):
    config = settings()
    episodes = generate_demo_episodes(num_episodes=3, length=7, image_size=8)
    reference = train_policy(make_model(config), episodes, config["training"], tmp_path / "full")
    stop = threading.Event()
    stop.set()
    paused = train_policy(
        make_model(config), episodes, config["training"], tmp_path / "paused", stop_event=stop
    )
    assert paused.stopped and not paused.history and paused.best_checkpoint is None
    for limit in [2, 3, 8, None]:
        paused = train_policy(
            make_model(config),
            episodes,
            {**config["training"], "max_steps": limit},
            tmp_path / "paused",
            resume_from=paused.checkpoint,
        )
    a = torch.load(reference.checkpoint, weights_only=False)
    b = torch.load(paused.checkpoint, weights_only=False)
    assert_nested_equal(a["model"], b["model"])
    assert a["history"] == b["history"]


def test_periodic_checkpoint_recovers_after_unexpected_failure(tmp_path):
    config = settings()
    episodes = generate_demo_episodes(num_episodes=3, length=7, image_size=8)
    reference = train_policy(make_model(config), episodes, config["training"], tmp_path / "full")

    def fail(event):
        if event["step"] == 3 and not event.get("epoch_complete"):
            raise RuntimeError("simulated interruption")

    with pytest.raises(RuntimeError, match="simulated"):
        train_policy(make_model(config), episodes, config["training"], tmp_path / "failed", fail)
    latest = tmp_path / "failed" / "checkpoint.pt"
    assert torch.load(latest, weights_only=False)["step"] == 2
    recovered = train_policy(
        make_model(config), episodes, config["training"], tmp_path / "recovered", resume_from=latest
    )
    a = torch.load(reference.checkpoint, weights_only=False)
    b = torch.load(recovered.checkpoint, weights_only=False)
    assert_nested_equal(a["model"], b["model"])
    assert a["history"] == b["history"]


def test_best_is_frozen_and_survives_moving_only_latest(tmp_path, monkeypatch):
    from sync2act.training import trainer

    config = settings()
    episodes = generate_demo_episodes(num_episodes=3, length=3, image_size=8)
    original_loss = trainer._loss
    values = iter([0.1, 0.4, 0.4])

    def controlled_validation(model, batch, cfg, *, return_sums=False):
        if model.training:
            return original_loss(model, batch, cfg, return_sums=return_sums)
        sums = LossSums(
            torch.tensor(next(values)), torch.tensor(1.0), torch.tensor(0.0), torch.tensor(0.0)
        )
        losses = sums.means(cfg["lambda_smooth"])
        return (*losses, sums) if return_sums else losses

    monkeypatch.setattr(trainer, "_loss", controlled_validation)
    first = train_policy(
        make_model(config), episodes, {**config["training"], "max_steps": 2}, tmp_path / "first"
    )
    best_before = torch.load(first.best_checkpoint, weights_only=False)
    first.best_checkpoint.unlink()  # Latest is self-contained, even without its sibling.
    resumed = train_policy(
        make_model(config),
        episodes,
        config["training"],
        tmp_path / "moved",
        resume_from=first.checkpoint,
    )
    best_after = torch.load(resumed.best_checkpoint, weights_only=False)
    latest = torch.load(resumed.checkpoint, weights_only=False)
    assert best_after["epoch"] == 0 and latest["epoch"] == 2
    assert best_after["selection_metric"] == "validation_loss"
    assert_nested_equal(best_before["model"], best_after["model"])
    assert any(not torch.equal(v, latest["model"][k]) for k, v in best_after["model"].items())
    with pytest.raises(ValueError, match="evaluation-only"):
        train_policy(
            make_model(config),
            episodes,
            config["training"],
            tmp_path / "wrong",
            resume_from=resumed.best_checkpoint,
        )


def test_changed_data_and_training_settings_are_rejected(tmp_path):
    config = settings()
    episodes = generate_demo_episodes(num_episodes=3, length=7, image_size=8)
    first = train_policy(
        make_model(config), episodes, {**config["training"], "max_steps": 1}, tmp_path / "first"
    )
    for changes in [{"batch_size": 2}, {"learning_rate": 0.02}, {"epochs": 4}]:
        with pytest.raises(ValueError, match="changed"):
            train_policy(
                make_model(config),
                episodes,
                {**config["training"], **changes},
                tmp_path / "rejected",
                resume_from=first.checkpoint,
            )
    changed = deepcopy(episodes)
    changed[0]["observation.image"][0].zero_()
    with pytest.raises(ValueError, match="data changed"):
        train_policy(
            make_model(config),
            changed,
            config["training"],
            tmp_path / "rejected",
            resume_from=first.checkpoint,
        )
    assert not (tmp_path / "rejected").exists()


def test_explicit_schedule_extension_can_itself_resume_exactly(tmp_path):
    config = settings()
    episodes = generate_demo_episodes(num_episodes=3, length=7, image_size=8)
    original = {**config["training"], "epochs": 1}
    first = train_policy(make_model(config), episodes, original, tmp_path / "first")
    extend = {**config["training"], "resume_schedule": "restart"}
    full = train_policy(
        make_model(config),
        episodes,
        extend,
        tmp_path / "extended-full",
        resume_from=first.checkpoint,
    )
    partial = train_policy(
        make_model(config),
        episodes,
        {**extend, "max_steps": 6},
        tmp_path / "extended-partial",
        resume_from=first.checkpoint,
    )
    assert partial.history[-1]["epoch"] == 0
    resumed = train_policy(
        make_model(config),
        episodes,
        config["training"],
        tmp_path / "resumed",
        resume_from=partial.checkpoint,
    )
    a, b = [torch.load(result.checkpoint, weights_only=False) for result in (full, resumed)]
    assert_nested_equal(a["model"], b["model"])
    assert a["history"] == b["history"]
    assert b["config"]["scheduler_plan"]["T_max"] == 2
    assert b["history"][1]["learning_rate"] == config["training"]["learning_rate"]


def test_rng_capture_restore_roundtrip():
    device = torch.device("cpu")
    state = capture_rng_state(device)
    expected = random.random(), np.random.rand(3), torch.rand(3)
    restore_rng_state(state, device)
    assert_nested_equal(expected, (random.random(), np.random.rand(3), torch.rand(3)))


def test_legacy_checkpoint_remains_loadable_for_evaluation(tmp_path):
    config = settings()
    model = build_training_policy(config["model"], config["training"])
    path = tmp_path / "legacy.pt"
    torch.save({"metadata": {"checkpoint_schema_version": 2}, "model": model.state_dict()}, path)
    restored = build_training_policy(config["model"], config["training"])
    load_checkpoint(path, restored)
    assert_nested_equal(model.state_dict(), restored.state_dict())


def test_worker_prefetch_does_not_advance_saved_batch_cursor(tmp_path):
    config = settings()
    config["training"].update(num_workers=1, epochs=1)
    episodes = generate_demo_episodes(num_episodes=3, length=7, image_size=8)
    full = train_policy(make_model(config), episodes, config["training"], tmp_path / "full")
    partial = train_policy(
        make_model(config), episodes, {**config["training"], "max_steps": 1}, tmp_path / "partial"
    )
    resumed = train_policy(
        make_model(config),
        episodes,
        config["training"],
        tmp_path / "resumed",
        resume_from=partial.checkpoint,
    )
    a, b = [torch.load(result.checkpoint, weights_only=False) for result in (full, resumed)]
    assert_nested_equal(a["model"], b["model"])
    assert a["history"] == b["history"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_dropout_resume_matches_with_deterministic_kernels(tmp_path, monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        config = settings()
        config["training"].update(device="cuda", epochs=2)
        episodes = generate_demo_episodes(num_episodes=3, length=7, image_size=8)
        full = train_policy(make_model(config), episodes, config["training"], tmp_path / "full")
        partial = train_policy(
            make_model(config),
            episodes,
            {**config["training"], "max_steps": 1},
            tmp_path / "partial",
        )
        resumed = train_policy(
            make_model(config),
            episodes,
            config["training"],
            tmp_path / "resumed",
            resume_from=partial.checkpoint,
        )
        a, b = [
            torch.load(r.checkpoint, map_location="cpu", weights_only=False)
            for r in (full, resumed)
        ]
        assert_nested_equal(a["model"], b["model"])
        assert_nested_equal(a["optimizer"], b["optimizer"])
        assert a["history"] == b["history"]
        assert_nested_equal(
            a["resume_state"]["rng_state"]["cuda"], b["resume_state"]["rng_state"]["cuda"]
        )
    finally:
        torch.use_deterministic_algorithms(previous)
