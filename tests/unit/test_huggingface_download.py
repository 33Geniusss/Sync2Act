from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from sync2act.data.huggingface import download_dataset_snapshot


def test_download_dataset_snapshot_uses_dataset_repo_and_reports_progress(tmp_path, monkeypatch):
    received = {}

    def fake_snapshot_download(**kwargs):
        received.update(kwargs)
        bar = kwargs["tqdm_class"](total=2, disable=True)
        bar.update(1)
        bar.update(1)
        bar.close()

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=fake_snapshot_download)
    )
    events = []
    target = download_dataset_snapshot(
        "owner/example", tmp_path / "dataset", "v1", events.append
    )

    assert target == (tmp_path / "dataset").resolve()
    assert received["repo_id"] == "owner/example"
    assert received["repo_type"] == "dataset"
    assert received["revision"] == "v1"
    assert received["local_dir"] == str(target)
    assert events[-1]["percent"] == 100
    assert any(event["total"] == 2 for event in events)


@pytest.mark.parametrize("repo_id", ["", "missing-owner"])
def test_download_dataset_snapshot_rejects_invalid_repo_id(tmp_path, repo_id):
    with pytest.raises(ValueError, match="owner/dataset"):
        download_dataset_snapshot(repo_id, tmp_path)
