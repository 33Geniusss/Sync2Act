import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytestmark = pytest.mark.skipif(
    os.environ.get("SYNC2ACT_RUN_GUI_TESTS") != "1",
    reason="Qt event-loop tests require an interactive or CI-supported desktop",
)

from PySide6.QtCore import QPoint, Qt  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QStyle,
    QStyleOptionSpinBox,
    QWidget,
)

from sync2act.data.episode import clone_episode  # noqa: E402
from sync2act.data.synthetic import generate_demo_episodes  # noqa: E402
from sync2act.gui import MainWindow  # noqa: E402


def test_main_window_starts_without_demo_data(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    assert window.tabs.count() == 6
    assert not window.windowIcon().isNull()
    assert len(window.episodes) == 0
    assert window.overview_values["episodes"].text() == "0"
    assert window.overview_values["frames"].text() == "0"
    assert window.overview_values["shape"].text() == "not loaded"
    assert window.overview_values["status"].text() == "No data"
    assert not window.start_button.isEnabled()
    assert window.device_combo.currentText() in {"cpu", "cuda"}
    assert window.epochs_spin.value() == 3
    assert window.batch_size_spin.value() == 32
    assert window.learning_rate_spin.value() == pytest.approx(0.003)
    assert window.weight_decay_spin.value() == pytest.approx(0.0001)
    assert window.validation_split_spin.value() == pytest.approx(0.15)
    assert window.test_split_spin.value() == pytest.approx(0.15)
    assert window.loss_combo.currentText() == "mse"
    assert window.seed_spin.value() == 7
    assert window.grad_clip_spin.value() == 0.0
    assert window.metrics_table.columnWidth(0) >= 280
    expected_root = Path.home() / "Sync2Act" / "datasets"
    assert Path(window.hf_target.text()) == expected_root
    window.hf_repo_id.setText("lerobot/pusht")
    assert Path(window.hf_target.text()) == expected_root / "pusht"
    window.close()


def test_spin_box_arrow_hit_areas_are_large_and_clickable(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    spin_box = window.epochs_spin
    option = QStyleOptionSpinBox()
    spin_box.initStyleOption(option)
    up = spin_box.style().subControlRect(
        QStyle.ComplexControl.CC_SpinBox,
        option,
        QStyle.SubControl.SC_SpinBoxUp,
        spin_box,
    )
    down = spin_box.style().subControlRect(
        QStyle.ComplexControl.CC_SpinBox,
        option,
        QStyle.SubControl.SC_SpinBoxDown,
        spin_box,
    )
    assert up.width() >= 28
    assert down.width() >= 28
    assert up.height() >= 16
    assert down.height() >= 16

    initial = spin_box.value()
    qtbot.mouseClick(
        spin_box,
        Qt.MouseButton.LeftButton,
        pos=QPoint(up.left() + 3, up.top() + 3),
    )
    assert spin_box.value() == initial + spin_box.singleStep()
    qtbot.mouseClick(
        spin_box,
        Qt.MouseButton.LeftButton,
        pos=QPoint(down.right() - 3, down.bottom() - 3),
    )
    assert spin_box.value() == initial
    window.close()


def test_training_controls_feed_the_training_configuration(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    window.original_episodes = generate_demo_episodes()
    window.episodes = [clone_episode(episode) for episode in window.original_episodes]
    window.refresh_all()
    window.device_combo.setCurrentText("cpu")
    window.epochs_spin.setValue(9)
    window.batch_size_spin.setValue(12)
    window.learning_rate_spin.setValue(0.0007)
    window.weight_decay_spin.setValue(0.002)
    window.validation_split_spin.setValue(0.2)
    window.test_split_spin.setValue(0.25)
    window.loss_combo.setCurrentText("smooth_l1")
    window.seed_spin.setValue(42)
    window.grad_clip_spin.setValue(1.0)
    training_frames = sum(len(episode["timestamp"]) for episode in window.training_episodes)
    dataset_frames = sum(len(episode["timestamp"]) for episode in window.original_episodes)
    expected_steps_per_epoch = int(np.ceil(training_frames / 12))
    expected_steps = 9 * expected_steps_per_epoch
    estimate = window.training_step_estimate_label.text()
    assert f"Estimated optimizer steps: {expected_steps:,}" in estimate
    assert f"ceil({training_frames:,} training frames / batch 12)" in estimate
    assert f"{dataset_frames:,} total dataset frames" in estimate
    _, config = window._training_configs()
    assert config == {
        **config,
        "device": "cpu",
        "epochs": 9,
        "batch_size": 12,
        "learning_rate": pytest.approx(0.0007),
        "weight_decay": pytest.approx(0.002),
        "validation_split": pytest.approx(0.2),
        "test_split": pytest.approx(0.25),
        "loss": "smooth_l1",
        "seed": 42,
        "grad_clip": pytest.approx(1.0),
    }
    window.close()


def test_evaluation_metrics_have_question_mark_help(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    metrics = {
        "evaluation_scope": "offline-only",
        "action_mse": 0.1,
        "action_mae": 0.2,
        "trajectory_smoothness": 0.3,
        "jerk": 0.4,
        "latency_p50_ms": 1.0,
        "latency_p95_ms": 2.0,
        "parameter_count": 123,
        "per_episode": [],
    }

    window._show_comparisons({"latest": metrics})

    assert window.metrics_table.rowCount() == 8
    for row, key in enumerate(key for key in metrics if key != "per_episode"):
        metric_widget = window.metrics_table.cellWidget(row, 0)
        help_label = metric_widget.findChild(QWidget, f"metric_help_{key}")
        assert help_label is not None
        assert help_label.text() == "?"
        assert help_label.toolTip()
    window.close()


def test_corruption_is_applied_only_to_training_episodes(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    window.original_episodes = generate_demo_episodes(num_episodes=6, length=12)
    window.episodes = list(window.original_episodes)
    window.validation_split_spin.setValue(0.2)
    window.test_split_spin.setValue(0.2)
    window._rebuild_episode_partitions()
    window.corruption_type.setCurrentText("frame_drop")
    window.corruption_value.setValue(1.0)
    window.apply_preview()

    assert len(window.training_episodes) == 4
    assert len(window.validation_episodes) == 1
    assert len(window.test_episodes) == 1
    for index in window.episode_split["train"]:
        assert window.episodes[index]["missing_mask"].all()
    for partition in ("validation", "test"):
        for index in window.episode_split[partition]:
            assert not window.episodes[index]["missing_mask"].any()
            assert window.episodes[index] is window.original_episodes[index]
    assert "train 4 (frame_drop)" in window.split_summary_label.text()
    assert "validation 1 (clean)" in window.split_summary_label.text()
    assert "test 1 (clean)" in window.split_summary_label.text()
    window.close()


def test_inspector_dimension_selection_quality_plot_and_step_cursor(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    window.original_episodes = generate_demo_episodes()
    window.episodes = [clone_episode(episode) for episode in window.original_episodes]
    window.refresh_all()
    episode = window.episodes[0]

    dimension = min(
        1,
        episode["observation.state"].shape[1] - 1,
        episode["action"].shape[1] - 1,
    )
    window.signal_dimension_combo.setCurrentIndex(dimension)
    window.step_spin.setValue(2)

    signal_curves = window.signal_plot.listDataItems()
    quality_curves = window.quality_plot.listDataItems()
    assert window.signal_dimension_combo.count() == min(
        episode["observation.state"].shape[1], episode["action"].shape[1]
    )
    assert window.episode_label.text() == f"Episode ({len(window.episodes)} total)"
    assert window.step_label.text() == f"Step ({len(episode['timestamp'])} total)"
    assert len(signal_curves) == 2
    camera_count = episode["image_quality"].shape[1]
    assert len(quality_curves) == camera_count + 2
    np.testing.assert_allclose(signal_curves[0].xData, episode["timestamp"].numpy())
    np.testing.assert_allclose(
        signal_curves[0].yData,
        episode["observation.state"][:, dimension].numpy(),
    )
    np.testing.assert_allclose(
        signal_curves[1].yData,
        episode["action"][:, dimension].numpy(),
    )
    for camera_index in range(camera_count):
        np.testing.assert_allclose(
            quality_curves[camera_index].yData,
            episode["image_quality"][:, camera_index].numpy(),
        )
    np.testing.assert_allclose(quality_curves[-2].yData, episode["state_quality"].numpy())
    np.testing.assert_allclose(quality_curves[-1].yData, episode["action_label_quality"].numpy())
    assert window.signal_plot.getAxis("bottom").labelText == "Time"
    assert window.quality_plot.getAxis("bottom").labelText == "Time"
    expected_timestamp = float(episode["timestamp"][2])
    assert window.signal_cursor.value() == pytest.approx(expected_timestamp)
    assert window.quality_cursor.value() == pytest.approx(expected_timestamp)
    window.close()


def test_corruption_targets_and_plot_visibility_follow_type(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    window.original_episodes = generate_demo_episodes()
    window.episodes = [clone_episode(episode) for episode in window.original_episodes]
    window.refresh_all()

    expected = {
        "temporal_shift": (["image", "state", "action"], True, False, False, False),
        "frame_drop": (["image"], False, True, False, False),
        "action_noise": (["action"], False, False, True, False),
        "state_anomaly": (["state"], False, False, True, True),
    }
    for kind, visibility in expected.items():
        targets, show_diagram, show_frame_plot, show_signal_plot, show_table = visibility
        window.corruption_type.setCurrentText(kind)
        actual = [
            window.corruption_mode.itemText(index)
            for index in range(window.corruption_mode.count())
        ]
        assert actual == targets
        assert window.corruption_mode.isEnabled() == (len(targets) > 1)
        assert window.corruption_plot.isHidden() != show_signal_plot
        assert window.frame_drop_plot.isHidden() != show_frame_plot
        assert window.state_change_table.isHidden() != show_table
        assert window.corruption_seed.isEnabled() == (kind != "temporal_shift")
        assert window.temporal_shift_diagram.isHidden() != show_diagram

    window.corruption_type.setCurrentText("frame_drop")
    window.corruption_value.setValue(0.5)
    assert len(window.frame_drop_plot.listDataItems()) == 2, window.corruption_info.toPlainText()
    assert window.corruption_plot.isHidden()
    assert "Dropped" in window.corruption_visual_summary.text()

    window.corruption_type.setCurrentText("action_noise")
    assert "action[" in window.corruption_plot.getPlotItem().titleLabel.text
    assert window.frame_drop_plot.isHidden()

    window.corruption_type.setCurrentText("state_anomaly")
    state_curves = window.corruption_plot.listDataItems()
    assert "state[" in window.corruption_plot.getPlotItem().titleLabel.text
    assert len(state_curves) == 3
    assert not np.array_equal(state_curves[0].yData, state_curves[1].yData)
    assert "Changed" in window.corruption_visual_summary.text()
    affected = window._preview_episode["provenance"]["affected_indices"]
    assert window.state_change_table.rowCount() == len(affected)
    first_frame = affected[0]
    original_value = float(window.original_episodes[0]["observation.state"][first_frame, 0])
    new_value = float(window._preview_episode["observation.state"][first_frame, 0])
    assert window.state_change_table.item(0, 0).text() == str(first_frame)
    assert float(window.state_change_table.item(0, 2).text()) == pytest.approx(
        original_value, abs=1e-6
    )
    assert float(window.state_change_table.item(0, 3).text()) == pytest.approx(new_value, abs=1e-6)

    window.corruption_type.setCurrentText("temporal_shift")
    window.corruption_mode.setCurrentText("action")
    window.corruption_value.setValue(-2)
    assert window.temporal_shift_diagram.rows == (
        ("state", "S"),
        ("image", "I"),
        ("action", "A"),
    )
    assert window.temporal_shift_diagram.target == "action"
    assert window.temporal_shift_diagram.shift == -2
    assert not window.temporal_shift_diagram.grab().isNull()
    window.close()


def test_background_training_keeps_event_loop_responsive(qtbot, tmp_path, monkeypatch):
    checkpoint = tmp_path / "bc_mlp_clean_test.pt"
    monkeypatch.setattr("sync2act.gui.app.new_model_checkpoint_path", lambda *_: checkpoint)
    window = MainWindow()
    qtbot.addWidget(window)
    window.original_episodes = generate_demo_episodes()
    window.episodes = [clone_episode(episode) for episode in window.original_episodes]
    window.refresh_all()
    window.epochs_spin.setValue(1)
    window.start_training()
    qtbot.waitUntil(lambda: window.thread is not None and window.thread.isRunning(), timeout=3000)
    marker = {"called": False}
    from PySide6.QtCore import QTimer

    QTimer.singleShot(0, lambda: marker.update(called=True))
    qtbot.waitUntil(lambda: marker["called"], timeout=1000)
    qtbot.waitUntil(lambda: not window.thread.isRunning(), timeout=30000)
    assert window.last_checkpoint.exists()
    window.close()
