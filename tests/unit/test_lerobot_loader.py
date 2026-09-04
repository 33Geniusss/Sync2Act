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
    video_dir = tmp_path / "videos" / "observation.image" / "chunk-000"
    video_dir.mkdir(parents=True)
    info = {
        "codebase_version": "v3.0",
        "fps": 10,
        "features": {
            "observation.image": {"dtype": "video", "shape": [8, 8, 3]},
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
    output = av.open(str(video_dir / "file-000.mp4"), "w")
    stream = output.add_stream("mpeg4", rate=10)
    stream.width = 8
    stream.height = 8
    stream.pix_fmt = "yuv420p"
    for value in range(4):
        rgb = np.full((8, 8, 3), value * 30, dtype=np.uint8)
        for packet in stream.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")):
            output.mux(packet)
    for packet in stream.encode():
        output.mux(packet)
    output.close()

    events = []
    episodes, metadata = load_local_lerobot_dataset(tmp_path, events.append)

    assert len(episodes) == 2
    assert episodes[0]["observation.image"].shape == (2, 3, 8, 8)
    assert episodes[0]["observation.image"].dtype == torch.uint8
    assert episodes[0]["observation.state"].shape == (2, 2)
    assert episodes[0]["action"].shape == (2, 2)
    assert episodes[0]["quality_score"].eq(1).all()
    assert not episodes[0]["missing_mask"].any()
    assert metadata["episodes"] == 2
    assert metadata["frames"] == 4
    assert events[-1]["percent"] == 100
