"""Opt-in checks executed inside the frozen application, with JSON results."""

from __future__ import annotations

import json
import os
import platform
import sys
import traceback
from pathlib import Path


def run_release_check(directory: str) -> int:
    target = Path(directory).resolve()
    target.mkdir(parents=True, exist_ok=True)
    result = {"passed": False, "frozen": bool(getattr(sys, "frozen", False))}
    try:
        # Set before the first CUDA operation; this process is only a diagnostic.
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        import av
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq
        import torch
        from PySide6.QtWidgets import QApplication

        from sync2act import __version__
        from sync2act.config import resolve_training_config
        from sync2act.data.dataset import NormalizationStats
        from sync2act.data.episode import clone_episode
        from sync2act.data.synthetic import generate_demo_episodes
        from sync2act.evaluation.test_corruption import evaluate_paired, preset_config
        from sync2act.gui import MainWindow
        from sync2act.reporting.report import generate_report
        from sync2act.training import train_policy
        from sync2act.training.setup import build_training_policy

        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        result.update(
            version=__version__,
            python=platform.python_version(),
            torch=torch.__version__,
            cuda_available=torch.cuda.is_available(),
            av=av.__version__,
            pyarrow=pa.__version__,
        )
        table = pa.table({"value": [1, 2, 3]})
        pq.write_table(table, target / "roundtrip.parquet")
        assert pq.read_table(target / "roundtrip.parquet").equals(table)
        # Exercise the native FFmpeg binding without downloading a dataset.
        frame = av.VideoFrame(8, 8, "rgb24")
        assert frame.to_ndarray().shape == (8, 8, 3)

        app = QApplication.instance() or QApplication([])
        window = MainWindow()
        window.show()
        app.processEvents()
        assert not window.episodes
        assert window.resume_schedule_combo.currentText() == "strict"
        assert window.grab().save(str(target / "startup.png"))
        window.tabs.setCurrentIndex(3)
        app.processEvents()
        assert window.grab().save(str(target / "training.png"))
        window.tabs.setCurrentIndex(4)
        window.eval_condition.setCurrentIndex(window.eval_condition.findData("mixed"))
        assert window._test_corruption_config().condition == "mixed"
        app.processEvents()
        assert window.grab().save(str(target / "evaluation.png"))
        window.close()

        episodes = generate_demo_episodes(num_episodes=3, length=7, image_size=8)
        result["devices"] = {}
        for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
            config = resolve_training_config(
                {
                    "model": {
                        "name": "act_lite",
                        "hidden_dim": 16,
                        "horizon": 3,
                        "dropout": 0.25,
                        "num_layers": 1,
                    },
                    "training": {
                        "epochs": 2,
                        "batch_size": 4,
                        "device": device,
                        "seed": 19,
                        "checkpoint_interval_steps": 2,
                    },
                }
            )

            def train(name, *, pause=False, resume=None, config=config, device=device):
                settings = {**config["training"], "max_steps": 1 if pause else None}
                model = build_training_policy(config["model"], settings)
                return train_policy(
                    model, episodes, settings, target / device / name, resume_from=resume
                )

            full = train("full")
            partial = train("partial", pause=True)
            state = torch.load(partial.checkpoint, map_location="cpu", weights_only=False)
            assert state["resume_state"]["next_batch"] == 1 and not state["history"]
            resumed = train("resumed", resume=partial.checkpoint)
            a, b = [
                torch.load(r.checkpoint, map_location="cpu", weights_only=False)
                for r in (full, resumed)
            ]
            assert a["history"] == b["history"]
            assert a["step"] == b["step"]
            assert all(torch.equal(v, b["model"][k]) for k, v in a["model"].items())
            best = torch.load(resumed.evaluation_checkpoint, map_location="cpu", weights_only=False)
            assert best["checkpoint_kind"] == "best"
            assert best["selection_value"] == min(r["validation_loss"] for r in b["history"])
            model = build_training_policy(config["model"], config["training"])
            model.load_state_dict(best["model"])
            stats = NormalizationStats.from_dict(best["stats"])
            test_episodes = generate_demo_episodes(
                num_episodes=2, length=64, image_size=8, seed=81
            )
            originals = [clone_episode(episode) for episode in test_episodes]
            paired_checks = {}
            for quality_mode in ("oracle", "unknown"):
                pair_dir = target / device / quality_mode
                pair = evaluate_paired(
                    model, test_episodes, preset_config("mixed", quality_mode=quality_mode),
                    stats=stats, output_dir=pair_dir, device=device, batch_size=16,
                )
                assert pair["test_manifest"]["affected_frames"] > 0
                clean = pd.read_csv(pair_dir / "clean" / "predictions.csv")
                damaged = pd.read_csv(pair_dir / "test" / "predictions.csv")
                columns = ["episode", "step", *[c for c in clean if c.startswith("target_")]]
                pd.testing.assert_frame_equal(clean[columns], damaged[columns])
                predictions = [c for c in clean if c.startswith("prediction_")]
                assert not clean[predictions].equals(damaged[predictions])
                report = generate_report([
                    {"name": condition, "metrics": pair[condition], "paired_evaluation": pair}
                    for condition in ("clean", "test")
                ], pair_dir / "report.html")
                exported = json.loads(report.with_suffix(".json").read_text(encoding="utf-8"))
                assert len(exported["runs"]) == 2
                assert exported["runs"][1]["metrics"] == pair["test"]
                paired_checks[quality_mode] = {
                    "paired_evaluation": True, "targets_preserved": True,
                    "predictions_changed": True, "html_json_export": True,
                    "affected_frames": pair["test_manifest"]["affected_frames"],
                }
            for before, after in zip(originals, test_episodes, strict=True):
                for key, value in before.items():
                    if torch.is_tensor(value):
                        assert torch.equal(value, after[key])
            result["devices"][device] = {
                "exact_resume": True,
                "best_selection": True,
                "steps": b["step"],
                "test_corruption": paired_checks,
                "source_episodes_unchanged": True,
            }
        result["passed"] = True
    except Exception:
        result["error"] = traceback.format_exc()
    (target / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0 if result["passed"] else 1
