from __future__ import annotations

import json

import av
import numpy as np
import pandas as pd
import torch

from sync2act.data.lerobot import load_local_lerobot_dataset


def test_load_local_lerobot_dataset_builds_episode_contract(tmp_path):
    (tmp_path / "meta").mkdir()
    (tmp_path / "data" / "chunk-000").mkdir(parents=True)
    image_keys = ["observation.images.overhead", "observation.images.wrist"]
    video_dirs = [tmp_path / "videos" / key / "chunk-000" for key in image_keys]
    for video_dir in video_dirs:
        video_dir.mkdir(parents=True)
    info = {
        "codebase_version": "v3.0",
        "fps": 10,
        "features": {
            image_keys[0]: {"dtype": "video", "shape": [8, 8, 3]},
            image_keys[1]: {"dtype": "video", "shape": [6, 10, 3]},
            "observation.state": {"dtype": "float32", "shape": [2]},
            "action": {"dtype": "float32", "shape": [2]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "timestamp": {"dtype": "float32", "shape": [1]},
        },
    }
    (tmp_path / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    table = pd.DataFrame(
        {
            "observation.state": [np.array([i, i + 1], dtype=np.float32) for i in range(4)],
            "action": [np.array([i / 10, i / 20], dtype=np.float32) for i in range(4)],
            "episode_index": [0, 0, 1, 1],
            "timestamp": [0.0, 0.1, 0.0, 0.1],
        }
    )
    table.to_parquet(tmp_path / "data" / "chunk-000" / "file-000.parquet")
    for camera_index, (video_dir, shape) in enumerate(
        zip(video_dirs, [(8, 8), (6, 10)], strict=True)
    ):
        height, width = shape
        output = av.open(str(video_dir / "file-000.mp4"), "w")
        stream = output.add_stream("mpeg4", rate=10)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        for value in range(4):
            rgb = np.full(
                (height, width, 3), camera_index * 100 + value * 30, dtype=np.uint8
            )
            for packet in stream.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
        output.close()

    events = []
    episodes, metadata = load_local_lerobot_dataset(tmp_path, events.append, max_image_size=4)

    assert len(episodes) == 2
    assert episodes[0]["observation.image"].shape == (2, 2, 3, 4, 4)
    assert episodes[0]["observation.image"].dtype == torch.uint8
    assert not torch.equal(
        episodes[0]["observation.image"][:, 0], episodes[0]["observation.image"][:, 1]
    )
    assert episodes[0]["observation.state"].shape == (2, 2)
    assert episodes[0]["action"].shape == (2, 2)
    assert episodes[0]["quality_score"].eq(1).all()
    assert not episodes[0]["missing_mask"].any()
    assert metadata["episodes"] == 2
    assert metadata["frames"] == 4
    assert metadata["camera_count"] == 2
    assert metadata["image_keys"] == image_keys
    assert metadata["image_shape"] == [2, 3, 4, 4]
    assert events[-1]["percent"] == 100
