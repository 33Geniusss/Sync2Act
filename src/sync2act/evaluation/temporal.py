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


def ensemble_action_chunks(
    chunks: torch.Tensor,
    episode_ids: list[int],
    steps: list[int],
    decay: float = 0.7,
) -> torch.Tensor:
    """Causal, episode-local fusion of [frame, horizon, action] predictions.

    Row order is preserved. Only chunks starting at or before the target step
    contribute; missing start steps and predictions beyond an episode are ignored.
    Weights are decay ** (target_step - source_step), matching TemporalEnsembler
    when a new chunk is produced at every observed step.
    """
    if not 0 < decay <= 1:
        raise ValueError("decay must be in (0, 1]")
    if chunks.ndim != 3 or chunks.shape[1] < 1:
        raise ValueError("chunks must have shape [frames, positive horizon, actions]")
    if len(episode_ids) != len(chunks) or len(steps) != len(chunks):
        raise ValueError("episode_ids and steps must match the number of chunks")
    ids = torch.as_tensor(episode_ids, device=chunks.device)
    times = torch.as_tensor(steps, device=chunks.device)
    output = torch.empty_like(chunks[:, 0])
    for episode in ids.unique():
        indices = (ids == episode).nonzero(as_tuple=False).flatten()
        indices = indices[torch.argsort(times[indices])]
        local_steps = times[indices].contiguous()
        if len(local_steps) > 1 and (local_steps[1:] == local_steps[:-1]).any():
            raise ValueError("duplicate steps within an episode")
        local_chunks = chunks[indices]
        total = torch.zeros_like(local_chunks[:, 0])
        denominator = chunks.new_zeros((len(indices), 1))
        for age in range(chunks.shape[1]):
            starts = local_steps - age
            source = torch.searchsorted(local_steps, starts)
            valid = (source < len(indices)) & (
                local_steps[source.clamp_max(len(indices) - 1)] == starts
            )
            weight = decay**age
            total[valid] += weight * local_chunks[source[valid], age]
            denominator[valid] += weight
        output[indices] = total / denominator
    return output
