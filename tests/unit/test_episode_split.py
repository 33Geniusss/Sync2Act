import pytest

from sync2act.data import split_episode_indices


def test_episode_split_is_deterministic_disjoint_and_complete():
    first = split_episode_indices(20, validation_split=0.2, test_split=0.15, seed=9)
    second = split_episode_indices(20, validation_split=0.2, test_split=0.15, seed=9)
    assert first == second
    assert len(first["train"]) == 13
    assert len(first["validation"]) == 4
    assert len(first["test"]) == 3
    partitions = [set(first[name]) for name in ("train", "validation", "test")]
    assert not partitions[0] & partitions[1]
    assert not partitions[0] & partitions[2]
    assert not partitions[1] & partitions[2]
    assert set.union(*partitions) == set(range(20))


def test_episode_split_rejects_impossible_or_overlapping_ratios():
    with pytest.raises(ValueError, match="At least 3 episodes"):
        split_episode_indices(2, validation_split=0.15, test_split=0.15)
    with pytest.raises(ValueError, match="smaller than 1"):
        split_episode_indices(10, validation_split=0.5, test_split=0.5)
