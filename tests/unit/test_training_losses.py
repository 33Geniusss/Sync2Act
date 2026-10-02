import pytest
import torch

from sync2act.training.losses import EpochLossAccumulator, prediction_loss_sums


def test_padding_never_contributes_to_smoothness_or_its_gradient():
    prediction = torch.tensor([[[1.0], [3.0], [100.0], [-100.0]]], requires_grad=True)
    padding = torch.tensor([[False, False, True, True]])
    sums = prediction_loss_sums(
        prediction, torch.zeros_like(prediction), padding, torch.ones(1, 4), {"loss": "mse"}
    )
    total, action, smooth = sums.means(0.25)
    assert action.item() == 5
    assert smooth.item() == 4
    assert total.item() == 6
    total.backward()
    assert torch.equal(prediction.grad[0, 2:], torch.zeros(2, 1))


@pytest.mark.parametrize("horizon", [1, 4])
def test_zero_valid_targets_and_pairs_remain_differentiable(horizon):
    prediction = torch.randn(2, horizon, 3, requires_grad=True)
    sums = prediction_loss_sums(
        prediction,
        torch.zeros_like(prediction),
        torch.ones(2, horizon, dtype=torch.bool),
        torch.zeros(2, horizon),
        {"quality_weighted_loss": True},
    )
    loss, _, smooth = sums.means(0.1)
    assert loss.item() == smooth.item() == 0
    loss.backward()
    assert torch.equal(prediction.grad, torch.zeros_like(prediction))


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("loss_type", ["mse", "smooth_l1"])
def test_epoch_loss_is_invariant_to_uneven_batches_and_padding(weighted, loss_type):
    torch.manual_seed(31)
    prediction = torch.randn(7, 4, 3)
    target = torch.randn_like(prediction)
    padding = torch.tensor([[False] * n + [True] * (4 - n) for n in [4, 1, 3, 2, 4, 1, 2]])
    quality = torch.tensor([[0.0, 0.2, 0.8, 1.0]]).expand(7, -1)
    config = {"loss": loss_type, "quality_weighted_loss": weighted}
    full = prediction_loss_sums(prediction, target, padding, quality, config).means(0.17)
    for batch_size in [1, 2, 4, 7]:
        accumulator = EpochLossAccumulator()
        for i in range(0, 7, batch_size):
            batch = slice(i, i + batch_size)
            accumulator.update(
                prediction_loss_sums(
                    prediction[batch], target[batch], padding[batch], quality[batch], config
                )
            )
        assert accumulator.means(0.17) == pytest.approx([v.item() for v in full], rel=1e-6)


def test_weighted_epoch_uses_quality_sum_even_when_below_one():
    accumulator = EpochLossAccumulator()
    for prediction, quality in [(1.0, 0.1), (3.0, 0.2), (999.0, 0.0)]:
        p = torch.tensor([[[prediction]]])
        accumulator.update(
            prediction_loss_sums(
                p,
                torch.zeros_like(p),
                torch.zeros(1, 1, dtype=torch.bool),
                torch.tensor([[quality]]),
                {"quality_weighted_loss": True, "loss": "mse"},
            )
        )
    assert accumulator.means(0)[0] == pytest.approx((0.1 + 9 * 0.2) / 0.3)
