from __future__ import annotations

import torch


class TemporalEnsembler:
    def __init__(self, decay: float = 0.7):
        if not 0 < decay <= 1:
            raise ValueError("decay must be in (0, 1]")
        self.decay = decay
        self.predictions: dict[int, list[tuple[int, torch.Tensor]]] = {}

    def add(self, start_step: int, chunk: torch.Tensor) -> None:
        for offset, action in enumerate(chunk):
            self.predictions.setdefault(start_step + offset, []).append(
                (start_step, action.detach().clone())
            )

    def action(self, step: int) -> torch.Tensor:
        candidates = self.predictions.get(step)
        if not candidates:
            raise KeyError(f"No prediction for step {step}")
        newest = max(origin for origin, _ in candidates)
        weights = torch.tensor(
            [self.decay ** (newest - origin) for origin, _ in candidates],
            device=candidates[0][1].device,
        )
        actions = torch.stack([action for _, action in candidates])
        return (actions * weights[:, None]).sum(0) / weights.sum()


def receding_horizon_actions(chunks: list[torch.Tensor], decay: float = 0.7) -> torch.Tensor:
    ensemble = TemporalEnsembler(decay)
    output = []
    for step, chunk in enumerate(chunks):
        ensemble.add(step, chunk)
        output.append(ensemble.action(step))
    return torch.stack(output)
