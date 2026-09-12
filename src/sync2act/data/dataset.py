from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from .episode import Episode, ensure_modal_metadata, validate_episode


@dataclass
class NormalizationStats:
    state_mean: torch.Tensor
    state_std: torch.Tensor
    action_mean: torch.Tensor
    action_std: torch.Tensor

    def to_dict(self) -> dict[str, list[float]]:
        return {key: getattr(self, key).tolist() for key in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, data: dict[str, list[float]]) -> NormalizationStats:
        return cls(**{key: torch.tensor(value, dtype=torch.float32) for key, value in data.items()})


def compute_stats(episodes: list[Episode]) -> NormalizationStats:
    states = torch.cat([episode["observation.state"] for episode in episodes])
    actions = torch.cat([episode["action"] for episode in episodes])
    return NormalizationStats(
        states.mean(0),
        states.std(0).clamp_min(1e-5),
        actions.mean(0),
        actions.std(0).clamp_min(1e-5),
    )


class EpisodeWindowDataset(Dataset):
    def __init__(
        self, episodes: list[Episode], horizon: int = 1, stats: NormalizationStats | None = None
    ):
        if not episodes:
            raise ValueError("At least one episode is required")
        self.episodes = episodes
        self.horizon = horizon
        self.stats = stats if stats is not None else compute_stats(episodes)
        self.indices: list[tuple[int, int]] = []
        for episode_index, episode in enumerate(episodes):
            ensure_modal_metadata(episode)
            length = validate_episode(episode)
            self.indices.extend((episode_index, step) for step in range(length))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        episode_index, step = self.indices[index]
        episode = self.episodes[episode_index]
        length = episode["action"].shape[0]
        end = min(length, step + self.horizon)
        count = end - step
        actions = torch.zeros(self.horizon, episode["action"].shape[1])
        action_label_quality = torch.zeros(self.horizon)
        action_label_time_offset = torch.zeros(self.horizon)
        action_label_missing = torch.ones(self.horizon, dtype=torch.bool)
        padding = torch.ones(self.horizon, dtype=torch.bool)
        actions[:count] = episode["action"][step:end]
        action_label_quality[:count] = episode["action_label_quality"][step:end]
        action_label_time_offset[:count] = episode["action_label_time_offset"][step:end]
        action_label_missing[:count] = episode["action_label_missing_mask"][step:end]
        padding[:count] = False
        state = (episode["observation.state"][step] - self.stats.state_mean) / self.stats.state_std
        actions = (actions - self.stats.action_mean) / self.stats.action_std
        image = episode["observation.image"][step]
        return {
            "image": image,
            "state": state,
            "actions": actions,
            "image_quality": episode["image_quality"][step].float(),
            "state_quality": episode["state_quality"][step].float().reshape(1),
            "action_label_quality": action_label_quality,
            "quality": action_label_quality,
            "image_missing": episode["image_missing_mask"][step],
            "state_missing": episode["state_missing_mask"][step].reshape(1),
            "action_label_missing": action_label_missing,
            "image_time_offset": episode["image_time_offset"][step].float(),
            "state_time_offset": episode["state_time_offset"][step].float().reshape(1),
            "action_label_time_offset": action_label_time_offset.float(),
            # Legacy aliases remain available to old policies and external adapters.
            "missing": episode["missing_mask"][step].float().reshape(1),
            "time_offset": episode["time_offset"][step].float().reshape(1),
            "padding_mask": padding,
            "episode_index": torch.tensor(episode_index),
            "step": torch.tensor(step),
        }
