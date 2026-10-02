from __future__ import annotations

import threading
import traceback
from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from sync2act.config import resolve_training_config
from sync2act.data.huggingface import DownloadCancelled, download_dataset_snapshot
from sync2act.data.lerobot import DatasetLoadCancelled, load_local_lerobot_dataset
from sync2act.training import train_policy
from sync2act.training.setup import build_training_policy


class TrainingWorker(QObject):
    progress = Signal(dict)
    finished = Signal(object, object)
    failed = Signal(str)

    def __init__(
        self,
        episodes,
        model_config: dict,
        training_config: dict,
        output_dir: Path,
        resume_from=None,
        validation_episodes=None,
        normalization_stats=None,
    ):
        super().__init__()
        self.episodes = episodes
        self.model_config = model_config
        self.training_config = training_config
        self.output_dir = output_dir
        self.resume_from = resume_from
        self.validation_episodes = validation_episodes
        self.normalization_stats = normalization_stats
        self.stop_event = threading.Event()

    @Slot()
    def run(self):
        try:
            resolved = resolve_training_config(
                {"model": self.model_config, "training": self.training_config}, source="GUI"
            )
            self.training_config = resolved["training"]
            model = build_training_policy(resolved["model"], self.training_config)
            result = train_policy(
                model,
                self.episodes,
                self.training_config,
                self.output_dir,
                self.progress.emit,
                self.stop_event,
                self.resume_from,
                self.validation_episodes,
                self.normalization_stats,
            )
            self.finished.emit(model, result)
        except Exception:
            self.failed.emit(traceback.format_exc())

    @Slot()
    def stop(self):
        self.stop_event.set()


class DatasetDownloadWorker(QObject):
    progress = Signal(dict)
    finished = Signal(str)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, repo_id: str, target_dir: Path, revision: str = "main"):
        super().__init__()
        self.repo_id = repo_id
        self.target_dir = target_dir
        self.revision = revision
        self.stop_event = threading.Event()

    @Slot()
    def run(self):
        try:
            target = download_dataset_snapshot(
                self.repo_id,
                self.target_dir,
                self.revision,
                self.progress.emit,
                self.stop_event,
            )
            self.finished.emit(str(target))
        except DownloadCancelled:
            self.cancelled.emit()
        except Exception:
            self.failed.emit(traceback.format_exc())

    @Slot()
    def stop(self):
        self.stop_event.set()


class DatasetLoadWorker(QObject):
    progress = Signal(dict)
    finished = Signal(object, dict)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, root: Path):
        super().__init__()
        self.root = root
        self.stop_event = threading.Event()

    @Slot()
    def run(self):
        try:
            episodes, metadata = load_local_lerobot_dataset(
                self.root, self.progress.emit, self.stop_event
            )
            self.finished.emit(episodes, metadata)
        except DatasetLoadCancelled:
            self.cancelled.emit()
        except Exception:
            self.failed.emit(traceback.format_exc())

    @Slot()
    def stop(self):
        self.stop_event.set()
