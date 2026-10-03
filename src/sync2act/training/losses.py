from __future__ import annotations

from dataclasses import dataclass

import torch

from sync2act.policies.quality_act import action_loss_sums


@dataclass
class LossSums:
    action_sum: torch.Tensor
    action_weight: torch.Tensor
    smooth_sum: torch.Tensor
    smooth_count: torch.Tensor

    def means(self, lambda_smooth: float):
        action = self.action_sum / self.action_weight.clamp_min(1e-8)
        smooth = self.smooth_sum / self.smooth_count.clamp_min(1)
        return action + lambda_smooth * smooth, action, smooth


def prediction_loss_sums(prediction, target, padding, quality, config) -> LossSums:
    weights = quality if config.get("quality_weighted_loss", False) else torch.ones_like(quality)
    action_sum, action_weight = action_loss_sums(
        prediction, target, weights, padding, config.get("loss", "mse")
    )
    valid_pairs = (~padding[:, 1:]) & (~padding[:, :-1])
    pair_error = (prediction[:, 1:] - prediction[:, :-1]).square().mean(dim=-1)
    smooth_sum = pair_error.masked_fill(~valid_pairs, 0).sum()
    return LossSums(action_sum, action_weight, smooth_sum, valid_pairs.sum())


class EpochLossAccumulator:
    """Sum each term's numerator/denominator; never average batch means."""

    def __init__(self):
        self.totals = None

    def update(self, sums: LossSums):
        values = torch.stack(
            [
                sums.action_sum.detach().double(),
                sums.action_weight.detach().double(),
                sums.smooth_sum.detach().double(),
                sums.smooth_count.detach().double(),
            ]
        )
        self.totals = values if self.totals is None else self.totals + values

    def means(self, lambda_smooth: float) -> tuple[float, float, float]:
        if self.totals is None:
            return 0.0, 0.0, 0.0
        return tuple(float(value) for value in LossSums(*self.totals).means(lambda_smooth))

    def state_dict(self) -> dict:
        return {"totals": None if self.totals is None else self.totals.detach().cpu().clone()}

    def load_state_dict(self, state: dict, device: torch.device) -> None:
        totals = state["totals"]
        self.totals = None if totals is None else totals.to(device).clone()
