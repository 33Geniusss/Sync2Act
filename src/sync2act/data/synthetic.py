from __future__ import annotations

import math

import torch

from .episode import Episode


def generate_demo_episodes(
    num_episodes: int = 4,
    length: int = 48,
    state_dim: int = 6,
    action_dim: int = 3,
    image_size: int = 32,
    seed: int = 7,
) -> list[Episode]:
    """Generate deterministic reaching-like demonstrations with rendered target dots."""
    generator = torch.Generator().manual_seed(seed)
    episodes: list[Episode] = []
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, image_size), torch.linspace(-1, 1, image_size), indexing="ij"
    )
    for episode_id in range(num_episodes):
        time = torch.linspace(0, 1, length)
        phase = episode_id * 0.37
        base = torch.stack(
            [torch.sin(2 * math.pi * time + phase), torch.cos(2 * math.pi * time + phase)], dim=1
        )
        extra = torch.randn((length, max(0, state_dim - 2)), generator=generator) * 0.03
        state = torch.cat([base, extra], dim=1)[:, :state_dim]
        target = torch.stack([torch.cos(math.pi * time), torch.sin(math.pi * time)], dim=1)
        velocity = target - state[:, :2]
        third = velocity[:, :1] * velocity[:, 1:2]
        action_base = torch.cat([velocity, third], dim=1)
        if action_dim > 3:
            action_base = torch.cat([action_base, torch.zeros(length, action_dim - 3)], dim=1)
        action = action_base[:, :action_dim].clamp(-1, 1)
        images = []
        for t in range(length):
            cx, cy = state[t, 0].item(), state[t, 1].item()
            tx, ty = target[t, 0].item(), target[t, 1].item()
            robot = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / 0.035)
            goal = torch.exp(-((xx - tx) ** 2 + (yy - ty) ** 2) / 0.025)
            images.append(torch.stack([robot, goal, 0.25 * (robot + goal)]))
        episodes.append(
            {
                "observation.image": torch.stack(images).float(),
                "observation.state": state.float(),
                "action": action.float(),
                "timestamp": (time * (length - 1) / 20.0).float(),
                "time_offset": torch.zeros(length, dtype=torch.float32),
                "image_time_offset": torch.zeros(length, 1, dtype=torch.float32),
                "state_time_offset": torch.zeros(length, dtype=torch.float32),
                "action_label_time_offset": torch.zeros(length, dtype=torch.float32),
                "image_quality": torch.ones(length, 1, dtype=torch.float32),
                "state_quality": torch.ones(length, dtype=torch.float32),
                "action_label_quality": torch.ones(length, dtype=torch.float32),
                "quality_score": torch.ones(length, dtype=torch.float32),
                "missing_mask": torch.zeros(length, dtype=torch.bool),
                "image_missing_mask": torch.zeros(length, 1, dtype=torch.bool),
                "state_missing_mask": torch.zeros(length, dtype=torch.bool),
                "action_label_missing_mask": torch.zeros(length, dtype=torch.bool),
                "episode_id": episode_id,
            }
        )
    return episodes
