import importlib.util
from pathlib import Path

import pandas as pd
import pytest
import torch

from sync2act.data.synthetic import generate_demo_episodes
from sync2act.evaluation.evaluator import evaluate_policy
from sync2act.evaluation.temporal import ensemble_action_chunks, receding_horizon_actions


def test_chunk_fusion_matches_streaming_and_resets_at_episode_boundaries():
    chunks = torch.arange(6 * 3 * 2, dtype=torch.float32).reshape(6, 3, 2)
    expected = torch.cat(
        [
            receding_horizon_actions(list(chunks[:4]), 0.5),
            receding_horizon_actions(list(chunks[4:]), 0.5),
        ]
    )
    order = torch.tensor([4, 2, 0, 5, 3, 1])
    actual = ensemble_action_chunks(
        chunks[order],
        [1, 0, 0, 1, 0, 0],
        [0, 2, 0, 1, 3, 1],
        0.5,
    )
    assert torch.allclose(actual, expected[order])
    # The first frame in the second episode has no contribution from episode 0.
    assert torch.equal(actual[0], chunks[4, 0])


def test_future_observations_cannot_change_past_fused_actions():
    chunks = torch.randn(7, 4, 2)
    before = ensemble_action_chunks(chunks, [0] * 7, list(range(7)))
    chunks[4:] = 100000
    after = ensemble_action_chunks(chunks, [0] * 7, list(range(7)))
    assert torch.equal(before[:4], after[:4])


def test_gaps_use_frame_distance_and_do_not_use_out_of_horizon_predictions():
    chunks = torch.tensor(
        [[[1.0], [2.0], [3.0]], [[9.0], [10.0], [11.0]], [[20.0], [21.0], [22.0]]]
    )
    actual = ensemble_action_chunks(chunks, [0, 0, 0], [0, 2, 5], decay=0.5)
    assert actual[:, 0].tolist() == pytest.approx([1, (3 * 0.25 + 9) / 1.25, 20])


@pytest.mark.parametrize("decay", [0, -0.1, 1.1, float("nan")])
def test_invalid_decay_rejected(decay):
    with pytest.raises(ValueError):
        ensemble_action_chunks(torch.zeros(1, 2, 1), [0], [0], decay)


def test_duplicate_steps_rejected_and_horizon_one_is_identity():
    chunks = torch.randn(3, 1, 2)
    assert torch.equal(ensemble_action_chunks(chunks, [0] * 3, [0, 1, 2]), chunks[:, 0])
    with pytest.raises(ValueError, match="duplicate"):
        ensemble_action_chunks(chunks, [0] * 3, [0, 1, 1])


class ChunkPolicy(torch.nn.Module):
    horizon = 3

    def forward(self, state, **kwargs):
        return state[:, None, :3] + state.new_tensor([0, 2, 7])[None, :, None]


def test_evaluator_pairing_batch_boundaries_and_saved_chunks(tmp_path):
    episodes = generate_demo_episodes(num_episodes=2, length=5)
    model = ChunkPolicy()
    baseline = evaluate_policy(model, episodes, batch_size=2)
    fused = evaluate_policy(
        model,
        episodes,
        batch_size=2,
        temporal_decay=0.5,
        save_action_chunks=True,
        output_dir=tmp_path,
    )
    different_batch = evaluate_policy(model, episodes, batch_size=7, temporal_decay=0.5)
    for key in ["action_mse", "action_mae", "trajectory_smoothness", "jerk"]:
        assert fused["first_step"][key] == pytest.approx(baseline[key])
        assert fused[key] == pytest.approx(different_batch[key])
    saved = torch.load(tmp_path / "action_chunks.pt", weights_only=True)
    expected = ensemble_action_chunks(saved["chunks"], saved["episode_ids"], saved["steps"], 0.5)
    table = pd.read_csv(tmp_path / "predictions.csv")
    assert torch.allclose(
        torch.tensor(table[["prediction_0", "prediction_1", "prediction_2"]].values),
        expected.double(),
    )
    assert saved["chunks"].shape == (10, 3, 3)
    assert not torch.equal(expected, saved["chunks"][:, 0])


def test_historical_baseline_audit_detects_changed_predictions_and_targets(tmp_path):
    path = Path(__file__).resolve().parents[2] / "tools/evaluate_temporal_study.py"
    spec = importlib.util.spec_from_file_location("temporal_study", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    old = pd.DataFrame(
        {"episode": [0, 0], "step": [0, 1], "prediction_0": [1.0, 3.0], "target_0": [2.0, 4.0]}
    )
    new = old.rename(columns={"prediction_0": "first_step_0"})
    original, paired = tmp_path / "old.csv", tmp_path / "new.csv"
    old.to_csv(original, index=False)
    new.to_csv(paired, index=False)
    assert module.compare_historical(original, paired, 1.0, 1.0)["passed"]
    new.loc[0, "first_step_0"] = 2
    new.to_csv(paired, index=False)
    assert not module.compare_historical(original, paired, 1.0, 1.0)["prediction_match"]
    new.loc[0, "first_step_0"] = 1
    new.loc[0, "target_0"] = 5
    new.to_csv(paired, index=False)
    assert not module.compare_historical(original, paired, 1.0, 1.0)["target_match"]
    new.iloc[::-1].to_csv(paired, index=False)
    with pytest.raises(RuntimeError, match="ordering"):
        module.compare_historical(original, paired, 1.0, 1.0)
