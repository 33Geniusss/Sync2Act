from __future__ import annotations

import os
import shutil
import threading
from collections.abc import Callable
from pathlib import Path

from sync2act.paths import huggingface_cache_root

ProgressCallback = Callable[[dict], None]


class DownloadCancelled(RuntimeError):
    """Raised when a dataset snapshot download is cancelled by the caller."""


class _NullProgressStream:
    def write(self, value: str) -> int:
        return len(value)

    def flush(self) -> None:
        pass


def _progress_tqdm(progress: ProgressCallback | None, cancel_event: threading.Event | None):
    from tqdm.auto import tqdm

    class SnapshotProgress(tqdm):
        def __init__(self, *args, **kwargs):
            # Windowed PyInstaller apps have no stderr; tqdm otherwise calls None.write().
            kwargs["file"] = _NullProgressStream()
            super().__init__(*args, **kwargs)

        def _emit(self) -> None:
            if progress is None:
                return
            # Hub 1.x also applies the custom tqdm class to byte-transfer bars.
            # Keep the GUI stable by reporting only the snapshot's file-count bar.
            if str(getattr(self, "unit", "it")) == "B":
                return
            total = int(getattr(self, "total", 0) or 0)
            current = int(getattr(self, "n", 0))
            percent = int(100 * current / total) if total else 0
            progress(
                {
                    "current": current,
                    "total": total,
                    "percent": min(100, max(0, percent)),
                    "description": str(getattr(self, "desc", "") or "Downloading files"),
                }
            )

        def update(self, n: int | float = 1):
            if cancel_event is not None and cancel_event.is_set():
                raise DownloadCancelled("Dataset download cancelled")
            changed = super().update(n)
            self._emit()
            return changed

        def close(self) -> None:
            self._emit()
            super().close()

    return SnapshotProgress


def _materialize_snapshot(
    snapshot: Path,
    target: Path,
    progress: ProgressCallback | None,
    cancel_event: threading.Event | None,
) -> None:
    """Copy a cached Hub snapshot into the requested dataset directory.

    Hard links avoid a second copy when the cache and target share a filesystem.
    A regular copy is used for other filesystems and filesystems without hard-link
    support. Per-file temporary names keep interrupted copies out of the dataset.
    """
    files = sorted(path for path in snapshot.rglob("*") if path.is_file())
    total = len(files)
    for current, source in enumerate(files, start=1):
        if cancel_event is not None and cancel_event.is_set():
            raise DownloadCancelled("Dataset download cancelled")
        relative = source.relative_to(snapshot)
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.sync2act-copying")
        temporary.unlink(missing_ok=True)
        try:
            os.link(source.resolve(), temporary)
        except OSError:
            shutil.copy2(source, temporary)
        os.replace(temporary, destination)
        if progress:
            progress(
                {
                    "current": current,
                    "total": total,
                    "percent": int(100 * current / total) if total else 100,
                    "description": "Preparing dataset files",
                }
            )


def _remove_legacy_local_cache(target: Path) -> None:
    """Remove metadata left by the former local_dir-based downloader."""
    legacy = target / ".cache" / "huggingface"
    if legacy.exists():
        shutil.rmtree(legacy, ignore_errors=True)
    cache_parent = target / ".cache"
    try:
        cache_parent.rmdir()
    except OSError:
        pass


def download_dataset_snapshot(
    repo_id: str,
    target_dir: str | Path,
    revision: str = "main",
    progress: ProgressCallback | None = None,
    cancel_event: threading.Event | None = None,
    *,
    cache_dir: str | Path | None = None,
) -> Path:
    """Download one Hugging Face dataset repository into an explicit local folder."""
    repo_id = repo_id.strip()
    revision = revision.strip() or "main"
    if not repo_id or "/" not in repo_id:
        raise ValueError("Enter a dataset repository ID such as 'owner/dataset'.")
    target = Path(target_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    cache = (
        Path(cache_dir).expanduser().resolve()
        if cache_dir is not None
        else huggingface_cache_root()
    )
    cache.mkdir(parents=True, exist_ok=True)
    if cancel_event is not None and cancel_event.is_set():
        raise DownloadCancelled("Dataset download cancelled")

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - dependency guard for partial installs
        raise RuntimeError(
            "Hugging Face support is not installed. Install the project dependencies first."
        ) from exc

    if progress:
        progress(
            {
                "current": 0,
                "total": 0,
                "percent": 0,
                "description": "Reading repository file list",
            }
        )
    snapshot = Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            cache_dir=str(cache),
            # Hub 1.x can race its Windows symlink capability probe across worker
            # threads, causing WinError 1314 before its documented copy fallback.
            max_workers=1 if os.name == "nt" else 8,
            tqdm_class=_progress_tqdm(progress, cancel_event),
        )
    )
    _materialize_snapshot(snapshot, target, progress, cancel_event)
    _remove_legacy_local_cache(target)
    if progress:
        progress(
            {
                "current": 1,
                "total": 1,
                "percent": 100,
                "description": "Download complete",
            }
        )
    return target
