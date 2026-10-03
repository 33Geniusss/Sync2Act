import pytest
import torch

from sync2act.training.checkpoint import atomic_save


def test_failed_atomic_replacement_preserves_previous_checkpoint(tmp_path, monkeypatch):
    target = tmp_path / "checkpoint.pt"
    atomic_save({"step": 3}, target)
    original = target.read_bytes()

    def fail(*args):
        raise OSError("simulated disk failure")

    monkeypatch.setattr("sync2act.training.checkpoint.os.replace", fail)
    with pytest.raises(OSError, match="disk failure"):
        atomic_save({"step": 4}, target)
    assert target.read_bytes() == original
    assert torch.load(target, weights_only=False)["step"] == 3
    assert not target.with_suffix(".pt.tmp").exists()
