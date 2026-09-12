from __future__ import annotations

import torch


def split_episode_indices(
    episode_count: int,
    validation_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 7,
) -> dict[str, list[int]]:
    """Return deterministic, mutually exclusive episode-level split indices."""
    if episode_count < 1:
        raise ValueError("At least one episode is required")
    if not 0 <= validation_split < 1 or not 0 <= test_split < 1:
        raise ValueError("Validation and test splits must be in [0, 1)")
    if validation_split + test_split >= 1:
        raise ValueError("Validation split plus test split must be smaller than 1")

    requested_holdouts = int(validation_split > 0) + int(test_split > 0)
    if episode_count <= requested_holdouts:
        raise ValueError(
            f"At least {requested_holdouts + 1} episodes are required for non-empty "
            "train, validation, and test partitions"
        )

    validation_count = max(1, int(episode_count * validation_split)) if validation_split > 0 else 0
    test_count = max(1, int(episode_count * test_split)) if test_split > 0 else 0
    while validation_count + test_count >= episode_count:
        if validation_count >= test_count and validation_count > int(validation_split > 0):
            validation_count -= 1
        elif test_count > int(test_split > 0):
            test_count -= 1
        else:
            raise ValueError("The requested split leaves no training episodes")

    order = torch.randperm(episode_count, generator=torch.Generator().manual_seed(seed)).tolist()
    validation_end = validation_count
    test_end = validation_end + test_count
    return {
        "train": order[test_end:],
        "validation": order[:validation_end],
        "test": order[validation_end:test_end],
    }
