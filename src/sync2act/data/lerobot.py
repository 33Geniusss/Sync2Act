from __future__ import annotations

import json
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image

from .episode import Episode, validate_episode

ProgressCallback = Callable[[dict], None]


class DatasetLoadCancelled(RuntimeError):
    """Raised when local dataset loading is cancelled."""


def _emit(progress: ProgressCallback | None, percent: int, description: str) -> None:
    if progress:
        progress({"percent": percent, "description": description})


def _check_cancel(cancel_event: threading.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise DatasetLoadCancelled("Dataset loading cancelled")


def load_local_lerobot_dataset(
    root: str | Path,
    progress: ProgressCallback | None = None,
    cancel_event: threading.Event | None = None,
    max_image_size: int = 128,
    episode_limit: int | None = None,
    frames_per_episode_limit: int | None = None,
) -> tuple[list[Episode], dict]:
    """Load a LeRobot v3 local repository into Sync2Act's episode contract."""
    root = Path(root).expanduser().resolve()
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise ValueError(f"Not a LeRobot dataset: missing {info_path}")
    _check_cancel(cancel_event)
    _emit(progress, 1, "Reading LeRobot metadata")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    features = info.get("features", {})
    required = {"observation.state", "action", "episode_index", "timestamp"}
    missing = required - features.keys()
    if missing:
        raise ValueError(f"LeRobot dataset is missing features: {sorted(missing)}")
    video_keys = [key for key, value in features.items() if value.get("dtype") == "video"]
    if not video_keys:
        raise ValueError("LeRobot dataset has no video observation feature")
    if "observation.image" in video_keys:
        video_keys = [
            "observation.image",
            *[key for key in video_keys if key != "observation.image"],
        ]

    parquet_paths = sorted(root.glob("data/chunk-*/file-*.parquet"))
    if not parquet_paths:
        raise ValueError(f"No LeRobot parquet files found under {root / 'data'}")
    tables = []
    for index, path in enumerate(parquet_paths):
        _check_cancel(cancel_event)
        tables.append(pd.read_parquet(path))
        _emit(
            progress,
            2 + int(13 * (index + 1) / len(parquet_paths)),
            f"Reading parquet {index + 1}/{len(parquet_paths)}",
        )
    table = pd.concat(tables, ignore_index=True)
    required_columns = {"observation.state", "action", "episode_index", "timestamp"}
    missing_columns = required_columns - set(table.columns)
    if missing_columns:
        raise ValueError(f"Parquet data is missing columns: {sorted(missing_columns)}")

    source_frame_count = len(table)
    source_episode_indices = table["episode_index"].to_numpy(dtype=np.int64)
    unique_source_episodes = list(dict.fromkeys(source_episode_indices.tolist()))
    if episode_limit is not None:
        if episode_limit <= 0:
            raise ValueError("episode_limit must be positive")
        unique_source_episodes = unique_source_episodes[:episode_limit]
    if frames_per_episode_limit is not None and frames_per_episode_limit <= 0:
        raise ValueError("frames_per_episode_limit must be positive")
    selected_positions: list[int] = []
    for episode_index in unique_source_episodes:
        positions = np.flatnonzero(source_episode_indices == episode_index)
        if frames_per_episode_limit is not None:
            positions = positions[:frames_per_episode_limit]
        selected_positions.extend(positions.tolist())
    if not selected_positions:
        raise ValueError("Dataset selection contains no frames")
    selected_positions_array = np.asarray(sorted(selected_positions), dtype=np.int64)
    selected_mask = np.zeros(source_frame_count, dtype=np.bool_)
    selected_mask[selected_positions_array] = True
    decode_stop = int(selected_positions_array[-1]) + 1
    table = table.iloc[selected_positions_array].reset_index(drop=True)

    image_shapes: dict[str, list[int]] = {}
    for image_key in video_keys:
        image_shape = features[image_key].get("shape", [])
        if len(image_shape) != 3:
            raise ValueError(f"Video feature {image_key} must have [H, W, C] shape")
        height, width, channels = map(int, image_shape)
        if channels != 3:
            raise ValueError(f"Video feature {image_key} must be RGB")
        image_shapes[image_key] = [height, width, channels]

    source_height, source_width, channels = image_shapes[video_keys[0]]
    if max_image_size <= 0:
        raise ValueError("max_image_size must be positive")
    resize_scale = min(1.0, max_image_size / max(source_height, source_width))
    target_height = max(1, round(source_height * resize_scale))
    target_width = max(1, round(source_width * resize_scale))

    try:
        import av
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("Install the 'av' package to decode LeRobot video") from exc
    frame_count = len(table)
    images = np.empty(
        (frame_count, len(video_keys), 3, target_height, target_width), dtype=np.uint8
    )
    total_camera_frames = decode_stop * len(video_keys)
    for camera_index, image_key in enumerate(video_keys):
        height, width, _ = image_shapes[image_key]
        video_paths = sorted((root / "videos" / Path(image_key)).glob("chunk-*/*.mp4"))
        if not video_paths:
            raise ValueError(f"No MP4 files found for video feature {image_key}")
        camera_buffer = images[:, camera_index]
        source_cursor = 0
        selected_cursor = 0
        for video_path in video_paths:
            _check_cancel(cancel_event)
            with av.open(str(video_path)) as container:
                for frame in container.decode(video=0):
                    _check_cancel(cancel_event)
                    if source_cursor >= source_frame_count:
                        raise ValueError(
                            f"Video feature {image_key} contains more frames than parquet data"
                        )
                    if selected_mask[source_cursor]:
                        rgb = frame.to_ndarray(format="rgb24")
                        if rgb.shape[:2] != (height, width):
                            raise ValueError(
                                f"Decoded frame shape {rgb.shape} for {image_key} does not match "
                                f"metadata {image_shapes[image_key]}"
                            )
                        if (height, width) != (target_height, target_width):
                            rgb = np.asarray(
                                Image.fromarray(rgb).resize(
                                    (target_width, target_height), Image.Resampling.BILINEAR
                                )
                            )
                        camera_buffer[selected_cursor] = np.transpose(rgb, (2, 0, 1))
                        selected_cursor += 1
                    source_cursor += 1
                    completed = camera_index * decode_stop + source_cursor
                    if source_cursor % 128 == 0 or source_cursor == decode_stop:
                        _emit(
                            progress,
                            15 + int(75 * completed / total_camera_frames),
                            f"Decoding camera {camera_index + 1}/{len(video_keys)}: "
                            f"{selected_cursor}/{frame_count} selected frames",
                        )
                    if source_cursor >= decode_stop:
                        break
            if source_cursor >= decode_stop:
                break
        if selected_cursor != frame_count:
            raise ValueError(
                f"Video feature {image_key} has {selected_cursor} selected frames but expected "
                f"{frame_count} rows"
            )
    # images is [T, cameras, RGB, H, W]; every sample therefore carries every view.

    states = np.stack(table["observation.state"].to_numpy()).astype(np.float32)
    actions = np.stack(table["action"].to_numpy()).astype(np.float32)
    timestamps = table["timestamp"].to_numpy(dtype=np.float32).copy()
    episode_indices = table["episode_index"].to_numpy(dtype=np.int64)
    image_tensor = torch.from_numpy(images)
    state_tensor = torch.from_numpy(states)
    action_tensor = torch.from_numpy(actions)
    timestamp_tensor = torch.from_numpy(timestamps)
    episodes = []
    unique_episodes = list(dict.fromkeys(episode_indices.tolist()))
    for number, episode_index in enumerate(unique_episodes):
        _check_cancel(cancel_event)
        row_indices = np.flatnonzero(episode_indices == episode_index)
        if row_indices.size == 0:
            continue
        start, stop = int(row_indices[0]), int(row_indices[-1]) + 1
        if not np.array_equal(row_indices, np.arange(start, stop)):
            raise ValueError(f"Episode {episode_index} is not contiguous in parquet data")
        episode: Episode = {
            "observation.image": image_tensor[start:stop],
            "observation.state": state_tensor[start:stop],
            "action": action_tensor[start:stop],
            "timestamp": timestamp_tensor[start:stop],
            "time_offset": torch.zeros(stop - start, dtype=torch.float32),
            "image_time_offset": torch.zeros(stop - start, len(video_keys), dtype=torch.float32),
            "state_time_offset": torch.zeros(stop - start, dtype=torch.float32),
            "action_label_time_offset": torch.zeros(stop - start, dtype=torch.float32),
            "image_quality": torch.ones(stop - start, len(video_keys), dtype=torch.float32),
            "state_quality": torch.ones(stop - start, dtype=torch.float32),
            "action_label_quality": torch.ones(stop - start, dtype=torch.float32),
            "quality_score": torch.ones(stop - start, dtype=torch.float32),
            "missing_mask": torch.zeros(stop - start, dtype=torch.bool),
            "image_missing_mask": torch.zeros(stop - start, len(video_keys), dtype=torch.bool),
            "state_missing_mask": torch.zeros(stop - start, dtype=torch.bool),
            "action_label_missing_mask": torch.zeros(stop - start, dtype=torch.bool),
            "provenance": {
                "source": "lerobot",
                "root": str(root),
                "episode_index": int(episode_index),
                "image_keys": list(video_keys),
                "quality_metadata": "initialized_clean",
            },
        }
        validate_episode(episode)
        episodes.append(episode)
        _emit(
            progress,
            90 + int(10 * (number + 1) / len(unique_episodes)),
            f"Building episodes {number + 1}/{len(unique_episodes)}",
        )
    metadata = {
        "root": str(root),
        "codebase_version": info.get("codebase_version", "unknown"),
        "fps": info.get("fps"),
        "image_key": video_keys[0],
        "image_keys": video_keys,
        "camera_count": len(video_keys),
        "camera_original_shapes": image_shapes,
        "max_image_size": max_image_size,
        "source_episodes": len(set(source_episode_indices.tolist())),
        "source_frames": source_frame_count,
        "episode_limit": episode_limit,
        "frames_per_episode_limit": frames_per_episode_limit,
        "state_dim": int(states.shape[1]),
        "action_dim": int(actions.shape[1]),
        "image_shape": [len(video_keys), channels, target_height, target_width],
        "episodes": len(episodes),
        "frames": frame_count,
    }
    _emit(progress, 100, "Dataset loaded")
    return episodes, metadata


class LeRobotEpisodeAdapter:
    """Optional boundary for LeRobot integration; core Sync2Act never imports LeRobot."""

    def __init__(self, dataset: Any):
        self.dataset = dataset

    @classmethod
    def from_repo_id(cls, repo_id: str, **kwargs: Any) -> LeRobotEpisodeAdapter:
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as exc:
            raise RuntimeError(
                "Install Sync2Act with the 'lerobot' extra to use this adapter"
            ) from exc
        return cls(LeRobotDataset(repo_id, **kwargs))

    @classmethod
    def from_local_dir(cls, root: str | Path, **kwargs: Any) -> LeRobotEpisodeAdapter:
        episodes, metadata = load_local_lerobot_dataset(root, **kwargs)
        adapter = cls(episodes)
        adapter.metadata = metadata
        return adapter
