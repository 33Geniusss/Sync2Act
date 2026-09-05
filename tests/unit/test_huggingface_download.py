from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from sync2act.data.huggingface import download_dataset_snapshot


def test_download_dataset_snapshot_uses_dataset_repo_and_reports_progress(tmp_path, monkeypatch):
    received = {}
    snapshot = tmp_path / "hub" / "snapshot"
    (snapshot / "meta").mkdir(parents=True)
    (snapshot / "data" / "chunk-000").mkdir(parents=True)
    (snapshot / "meta" / "info.json").write_text("{}", encoding="utf-8")
    (snapshot / "data" / "chunk-000" / "file-000.parquet").write_bytes(b"parquet")

    def fake_snapshot_download(**kwargs):
        received.update(kwargs)
        bar = kwargs["tqdm_class"](total=2, disable=True)
        bar.update(1)
        bar.update(1)
        bar.close()
        return str(snapshot)

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=fake_snapshot_download)
    )
    events = []
    target = download_dataset_snapshot(
        "owner/example",
        tmp_path / "dataset",
        "v1",
        events.append,
        cache_dir=tmp_path / "cache",
    )

    assert target == (tmp_path / "dataset").resolve()
    assert received["repo_id"] == "owner/example"
    assert received["repo_type"] == "dataset"
    assert received["revision"] == "v1"
    assert "local_dir" not in received
    assert received["cache_dir"] == str((tmp_path / "cache").resolve())
    assert (target / "meta" / "info.json").read_text(encoding="utf-8") == "{}"
    assert (target / "data" / "chunk-000" / "file-000.parquet").read_bytes() == b"parquet"
    assert not (target / ".cache").exists()
    assert events[-1]["percent"] == 100
    assert any(event["total"] == 2 for event in events)


@pytest.mark.parametrize("repo_id", ["", "missing-owner"])
def test_download_dataset_snapshot_rejects_invalid_repo_id(tmp_path, repo_id):
    with pytest.raises(ValueError, match="owner/dataset"):
        download_dataset_snapshot(repo_id, tmp_path)


def test_successful_download_removes_legacy_local_cache(tmp_path, monkeypatch):
    snapshot = tmp_path / "hub" / "snapshot"
    snapshot.mkdir(parents=True)
    (snapshot / "README.md").write_text("dataset", encoding="utf-8")
    target = tmp_path / "dataset"
    legacy = target / ".cache" / "huggingface" / "download"
    legacy.mkdir(parents=True)
    (legacy / "old.incomplete").write_bytes(b"partial")

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=lambda **_kwargs: str(snapshot)),
    )

    download_dataset_snapshot("owner/example", target, cache_dir=tmp_path / "cache")

    assert (target / "README.md").read_text(encoding="utf-8") == "dataset"
    assert not (target / ".cache").exists()
