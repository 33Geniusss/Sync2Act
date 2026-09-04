import pytest
import torch

from sync2act.evaluation.evaluator import _episode_trajectory_metrics


def test_trajectory_metrics_never_cross_episode_boundaries():
    prediction = torch.tensor([[0.0], [1.0], [3.0], [100.0], [102.0]])
    episode_ids = [0, 0, 0, 1, 1]
    steps = [0, 1, 2, 0, 1]

    smoothness, jerk = _episode_trajectory_metrics(
        prediction, episode_ids, steps
    )

    # Episode-local velocities are [1, 2] and [2]. The 97-unit boundary jump
    # between episodes must not contribute to either metric.
    assert smoothness == pytest.approx(3.0)
    assert jerk == pytest.approx(1.0)


def test_trajectory_metrics_sort_frames_by_step_within_each_episode():
    prediction = torch.tensor([[3.0], [0.0], [1.0]])

    smoothness, jerk = _episode_trajectory_metrics(
        prediction, episode_ids=[4, 4, 4], steps=[2, 0, 1]
    )

    assert smoothness == pytest.approx(2.5)
    assert jerk == pytest.approx(1.0)
