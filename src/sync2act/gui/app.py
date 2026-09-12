from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pyqtgraph as pg
import torch
from PySide6.QtCore import QRectF, Qt, QThread, Signal
from PySide6.QtGui import QColor, QIcon, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from sync2act.config import save_config
from sync2act.corruptions import apply_corruption
from sync2act.data.dataset import NormalizationStats
from sync2act.data.episode import ensure_modal_quality
from sync2act.data.split import split_episode_indices
from sync2act.evaluation import evaluate_policy
from sync2act.paths import default_dataset_path, models_root, new_model_checkpoint_path
from sync2act.policies import build_policy
from sync2act.reporting import generate_report
from sync2act.training import load_checkpoint

from .worker import DatasetDownloadWorker, DatasetLoadWorker, TrainingWorker

METRIC_HELP = {
    "evaluation_scope": (
        "Evaluation scope. 'offline-only' means predictions are compared with recorded "
        "test-set actions without running a robot or simulator."
    ),
    "action_mse": (
        "Mean squared error between predicted and target actions over every test frame "
        "and action dimension. Lower is better; large errors receive extra weight."
    ),
    "action_mae": (
        "Mean absolute error between predicted and target actions over every test frame "
        "and action dimension. Lower is better."
    ),
    "trajectory_smoothness": (
        "Mean squared change between consecutive predicted actions, computed within each "
        "episode only. Lower means a smoother predicted trajectory."
    ),
    "jerk": (
        "Mean absolute second difference of predicted actions, computed within each "
        "episode only. Lower means fewer abrupt trajectory changes."
    ),
    "latency_p50_ms": (
        "Median per-sample model inference latency in milliseconds. Lower is faster."
    ),
    "latency_p95_ms": (
        "95th-percentile per-sample model inference latency in milliseconds. About 95% "
        "of measured inferences are no slower than this value."
    ),
    "parameter_count": (
        "Total number of trainable and non-trainable model parameters. This describes "
        "model size and should be considered together with accuracy and latency."
    ),
}


def _asset_path(name: str) -> Path:
    root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[3]))
    return root / "assets" / name


def _image_pixmap(tensor: torch.Tensor, width: int = 250) -> QPixmap:
    if tensor.ndim == 4:
        # Display all synchronized camera views side by side in the inspector.
        tensor = torch.cat(list(tensor), dim=2)
    if tensor.dtype == torch.uint8:
        tensor = tensor.float().div(255.0)
    array = np.ascontiguousarray(
        (tensor.detach().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    )
    height, source_width, _ = array.shape
    image = QImage(
        array.data, source_width, height, 3 * source_width, QImage.Format.Format_RGB888
    ).copy()
    return QPixmap.fromImage(image).scaledToWidth(width, Qt.TransformationMode.SmoothTransformation)


class TemporalShiftDiagram(QWidget):
    """Compact timeline diagram showing which stream is temporally displaced."""

    rows = (("state", "S"), ("image", "I"), ("action", "A"))

    def __init__(self):
        super().__init__()
        self.target = "image"
        self.shift = 0
        self.setMinimumHeight(225)
        self.setToolTip("MASK marks boundary samples introduced by the temporal shift.")

    def set_configuration(self, target: str, shift: int):
        self.target = target
        self.shift = int(shift)
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#f8faff"))

        direction = (
            "no displacement"
            if self.shift == 0
            else (
                f"lags by {self.shift} frame(s)"
                if self.shift > 0
                else f"leads by {-self.shift} frame(s)"
            )
        )
        painter.setPen(QColor("#172033"))
        painter.drawText(
            QRectF(14, 8, self.width() - 28, 22),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            f"Temporal alignment preview — {self.target} {direction}",
        )

        left = 82.0
        right = 16.0
        cell_count = 8
        cell_width = max(34.0, (self.width() - left - right) / cell_count)
        top = 40.0
        row_height = 42.0
        colors = {
            "state": QColor("#dce5ff"),
            "image": QColor("#d8f2ea"),
            "action": QColor("#ffe3dc"),
        }

        for row_index, (name, prefix) in enumerate(self.rows):
            y = top + row_index * row_height
            painter.setPen(QColor("#172033"))
            painter.drawText(
                QRectF(10, y, left - 18, 30),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                name,
            )
            for column in range(cell_count):
                source_index = column - self.shift if name == self.target else column
                masked = not 0 <= source_index < cell_count
                rectangle = QRectF(
                    left + column * cell_width + 2,
                    y,
                    cell_width - 4,
                    30,
                )
                painter.setBrush(QColor("#f8d7da") if masked else colors[name])
                border = QColor("#3157d5") if name == self.target else QColor("#aeb8cb")
                painter.setPen(QPen(border, 2 if name == self.target else 1))
                painter.drawRoundedRect(rectangle, 4, 4)
                painter.setPen(QColor("#8b2331") if masked else QColor("#172033"))
                text = "MASK" if masked else f"{prefix}{source_index}"
                painter.drawText(rectangle, Qt.AlignmentFlag.AlignCenter, text)

        axis_y = top + len(self.rows) * row_height + 2
        painter.setPen(QPen(QColor("#6d7890"), 1))
        painter.drawLine(int(left), int(axis_y), int(self.width() - right), int(axis_y))
        painter.drawText(
            QRectF(left, axis_y + 2, self.width() - left - right, 20),
            Qt.AlignmentFlag.AlignCenter,
            "Dataset time / frame index  t0 → t7",
        )
        painter.end()


class MainWindow(QMainWindow):
    training_requested = Signal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sync2Act — Robot Data Quality Lab")
        icon_path = _asset_path("sync2act.ico")
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))
        self.resize(1360, 860)
        self.setMinimumSize(1080, 700)
        self.original_episodes = []
        self.episodes = []
        self.current_model = None
        self.last_checkpoint: Path | None = None
        self.last_metrics: dict | None = None
        self.current_stats: NormalizationStats | None = None
        self.thread: QThread | None = None
        self.worker: TrainingWorker | None = None
        self.download_thread: QThread | None = None
        self.download_worker: DatasetDownloadWorker | None = None
        self.load_thread: QThread | None = None
        self.load_worker: DatasetLoadWorker | None = None
        self.dataset_metadata: dict = {}
        self.active_corruption_config: dict | None = None
        self.episode_split: dict[str, list[int]] = {
            "train": [],
            "validation": [],
            "test": [],
        }
        self.episode_partitions: dict[int, str] = {}
        self.training_episodes = []
        self.validation_episodes = []
        self.test_episodes = []
        self.model_test_episodes = []
        self._download_target_is_default = True
        shell = QWidget()
        shell.setObjectName("appShell")
        shell_layout = QVBoxLayout(shell)
        shell_layout.setContentsMargins(0, 0, 0, 0)
        shell_layout.setSpacing(0)
        shell_layout.addWidget(self._build_app_header())
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        shell_layout.addWidget(self.tabs, 1)
        self.setCentralWidget(shell)
        self._build_overview()
        self._build_inspector()
        self._build_corruption()
        self._build_training()
        self._build_evaluation()
        self._build_report()
        self._configure_spin_boxes()
        self._apply_plot_theme()
        self.setStyleSheet("""
            * { font-size:13px; color:#26324a; }
            QMainWindow, QWidget#appShell { background:#f3f6fb; }
            QFrame#appHeader { background:#17233f; border-bottom:1px solid #263653; }
            QLabel#brandMark { background:#5b7cff; color:white; border-radius:10px;
                font-size:16px; font-weight:800; padding:9px 8px; }
            QLabel#brandName { color:white; font-size:20px; font-weight:750; }
            QLabel#brandTagline { color:#9eabc3; font-size:11px; }
            QLabel#labBadge { color:#cbd5ff; background:#25365e; border:1px solid #3a5182;
                border-radius:12px; padding:5px 11px; font-size:10px; font-weight:700; }
            QTabWidget::pane { border:0; background:#f3f6fb; }
            QTabBar { background:#ffffff; qproperty-drawBase:0; }
            QTabBar::tab { min-width:120px; padding:13px 15px; color:#68758c;
                background:#ffffff; border:0; border-bottom:3px solid transparent; }
            QTabBar::tab:hover { color:#3157d5; background:#f7f9ff; }
            QTabBar::tab:selected { color:#3157d5; background:#f7f9ff;
                border-bottom:3px solid #5b7cff; font-weight:700; }
            QWidget#pageSurface { background:#f3f6fb; }
            QFrame#pageHeader { background:#ffffff; border:1px solid #e3e9f3;
                border-radius:12px; }
            QFrame#pageAccent { background:#5b7cff; border-radius:2px; }
            QLabel#hero { font-size:24px; font-weight:750; color:#17233f; }
            QLabel#pageSubtitle { color:#6c7890; font-size:12px; }
            QGroupBox { background:#ffffff; font-weight:700; color:#34415a;
                border:1px solid #dde5f0; border-radius:11px; margin-top:13px;
                padding:17px 14px 13px 14px; }
            QGroupBox::title { subcontrol-origin:margin; left:14px; padding:0 7px;
                color:#53627c; background:#ffffff; }
            QGroupBox#metricCard { border-top:3px solid #6b87ff; padding:14px 16px; }
            QLabel#overviewValue { font-size:22px; font-weight:800; color:#3157d5; }
            QLineEdit, QComboBox, QTextEdit { background:#fbfcff;
                border:1px solid #d7dfec; border-radius:7px; padding:6px 8px;
                selection-background-color:#5b7cff; }
            QSpinBox, QDoubleSpinBox { background:#fbfcff; border:1px solid #d7dfec;
                border-radius:7px; padding:5px 32px 5px 8px;
                selection-background-color:#5b7cff; }
            QSpinBox::up-button, QDoubleSpinBox::up-button {
                subcontrol-origin:border; subcontrol-position:top right; width:28px;
                background:#edf2fa; border-left:1px solid #d2dbea;
                border-bottom:1px solid #d2dbea; border-top-right-radius:6px; }
            QSpinBox::down-button, QDoubleSpinBox::down-button {
                subcontrol-origin:border; subcontrol-position:bottom right; width:28px;
                background:#edf2fa; border-left:1px solid #d2dbea;
                border-bottom-right-radius:6px; }
            QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
            QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
                background:#dfe7fb; }
            QSpinBox::up-button:pressed, QDoubleSpinBox::up-button:pressed,
            QSpinBox::down-button:pressed, QDoubleSpinBox::down-button:pressed {
                background:#cbd8f8; }
            QSpinBox::up-arrow, QDoubleSpinBox::up-arrow,
            QSpinBox::down-arrow, QDoubleSpinBox::down-arrow { width:9px; height:9px; }
            QLineEdit:hover, QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover,
            QTextEdit:hover { border-color:#a9b8d4; }
            QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus,
            QTextEdit:focus { border:2px solid #6b87ff; background:white; }
            QComboBox::drop-down { border:0; width:26px; }
            QPushButton { min-height:18px; padding:8px 16px; border-radius:7px;
                border:1px solid #3157d5; background:#3157d5; color:white; font-weight:650; }
            QPushButton:hover { background:#2649bd; border-color:#2649bd; }
            QPushButton:pressed { background:#1f3b98; }
            QPushButton[variant="secondary"] { color:#42516c; background:#ffffff;
                border:1px solid #ccd6e5; }
            QPushButton[variant="secondary"]:hover { color:#3157d5; background:#f3f6ff;
                border-color:#90a5d4; }
            QPushButton[variant="danger"] { color:#b73346; background:#fff5f6;
                border:1px solid #efc3ca; }
            QPushButton[variant="danger"]:hover { background:#ffe8eb; border-color:#df8795; }
            QPushButton:disabled { color:#919db1; background:#e6eaf1; border-color:#e0e5ed; }
            QProgressBar { min-height:10px; border:0; border-radius:5px;
                background:#e5eaf2; text-align:center; color:#42516c; }
            QProgressBar::chunk { border-radius:5px; background:#5b7cff; }
            QTableWidget { background:white; alternate-background-color:#f7f9fc;
                border:1px solid #dfe6f0; border-radius:8px; gridline-color:#e8edf5;
                selection-background-color:#e8edff; selection-color:#24304a; }
            QHeaderView::section { background:#edf2fa; color:#43516b; border:0;
                border-right:1px solid #dde5f0; border-bottom:1px solid #d8e1ed;
                padding:8px; font-weight:700; }
            QToolTip { color:#eef2ff; background:#1d2942; border:1px solid #445475;
                border-radius:5px; padding:6px; }
            QStatusBar { background:#17233f; color:#dce4f5; border-top:1px solid #263653; }
            QStatusBar QLabel { color:#dce4f5; }
            QScrollBar:vertical { background:#eef2f7; width:10px; margin:0; }
            QScrollBar::handle:vertical { background:#b8c3d4; min-height:24px; border-radius:5px; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height:0; }
        """)
        self.statusBar().showMessage("No dataset loaded")
        self.refresh_all()

    def _configure_spin_boxes(self):
        for spin_box in self.findChildren(QAbstractSpinBox):
            spin_box.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.UpDownArrows)
            spin_box.setAccelerated(True)
            spin_box.setMinimumHeight(36)

    def _build_app_header(self):
        header = QFrame()
        header.setObjectName("appHeader")
        header.setFixedHeight(72)
        layout = QHBoxLayout(header)
        layout.setContentsMargins(24, 10, 24, 10)
        mark = QLabel("S2A")
        mark.setObjectName("brandMark")
        mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mark.setFixedSize(48, 46)
        identity = QVBoxLayout()
        identity.setSpacing(0)
        name = QLabel("Sync2Act")
        name.setObjectName("brandName")
        tagline = QLabel("From dataset quality to reliable robot policies")
        tagline.setObjectName("brandTagline")
        identity.addWidget(name)
        identity.addWidget(tagline)
        badge = QLabel("ROBOT DATA QUALITY LAB")
        badge.setObjectName("labBadge")
        layout.addWidget(mark)
        layout.addSpacing(4)
        layout.addLayout(identity)
        layout.addStretch()
        layout.addWidget(badge)
        return header

    def _page(self, title: str, subtitle: str):
        page = QWidget()
        page.setObjectName("pageSurface")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(22, 18, 22, 20)
        layout.setSpacing(13)
        header = QFrame()
        header.setObjectName("pageHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(17, 13, 18, 13)
        accent = QFrame()
        accent.setObjectName("pageAccent")
        accent.setFixedWidth(5)
        accent.setMinimumHeight(43)
        titles = QVBoxLayout()
        titles.setSpacing(2)
        heading = QLabel(title)
        heading.setObjectName("hero")
        description = QLabel(subtitle)
        description.setObjectName("pageSubtitle")
        titles.addWidget(heading)
        titles.addWidget(description)
        header_layout.addWidget(accent)
        header_layout.addSpacing(5)
        header_layout.addLayout(titles)
        header_layout.addStretch()
        layout.addWidget(header)
        return page, layout

    def _apply_plot_theme(self):
        for plot in (
            self.signal_plot,
            self.quality_plot,
            self.frame_drop_plot,
            self.corruption_plot,
            self.training_plot,
            self.compare_plot,
        ):
            plot.setBackground("#ffffff")
            plot.getPlotItem().setContentsMargins(10, 10, 10, 8)
            for axis_name in ("left", "bottom"):
                axis = plot.getAxis(axis_name)
                axis.setPen(pg.mkPen("#aab6c8"))
                axis.setTextPen(pg.mkPen("#5f6d85"))

    def _build_overview(self):
        page, layout = self._page(
            "Overview", "A reproducible path from data damage to policy evidence."
        )
        cards = QGridLayout()
        self.overview_values = {}
        for index, (key, label) in enumerate(
            [
                ("episodes", "EPISODES"),
                ("frames", "FRAMES"),
                ("shape", "IMAGE SHAPE"),
                ("device", "DEVICE"),
                ("status", "TRAINING"),
                ("best", "BEST METRIC"),
            ]
        ):
            box = QGroupBox(label)
            box.setObjectName("metricCard")
            inner = QVBoxLayout(box)
            value = QLabel("—")
            value.setObjectName("overviewValue")
            inner.addWidget(value)
            self.overview_values[key] = value
            cards.addWidget(box, index // 3, index % 3)
        layout.addLayout(cards)
        download_box = QGroupBox("Download a Hugging Face dataset")
        download_form = QGridLayout(download_box)
        self.hf_repo_id = QLineEdit()
        self.hf_repo_id.setPlaceholderText("owner/dataset (for example: lerobot/pusht)")
        self.hf_revision = QLineEdit("main")
        self.hf_target = QLineEdit(str(default_dataset_path("")))
        self.hf_repo_id.textChanged.connect(self._update_default_download_target)
        self.hf_target.textEdited.connect(self._mark_download_target_custom)
        browse = QPushButton("Browse...")
        browse.setProperty("variant", "secondary")
        browse.clicked.connect(self.choose_download_target)
        self.download_button = QPushButton("Download")
        self.download_button.clicked.connect(self.start_dataset_download)
        self.cancel_download_button = QPushButton("Cancel")
        self.cancel_download_button.setProperty("variant", "danger")
        self.cancel_download_button.setEnabled(False)
        self.cancel_download_button.clicked.connect(self.cancel_dataset_download)
        self.load_dataset_button = QPushButton("Load local dataset")
        self.load_dataset_button.setProperty("variant", "secondary")
        self.load_dataset_button.clicked.connect(self.start_dataset_load)
        self.download_progress = QProgressBar()
        self.download_progress.setRange(0, 100)
        self.download_status = QLabel(
            "Ready. Public datasets need no token; private datasets use your local Hugging Face login."
        )
        download_form.addWidget(QLabel("Repository ID"), 0, 0)
        download_form.addWidget(self.hf_repo_id, 0, 1, 1, 3)
        download_form.addWidget(QLabel("Revision"), 1, 0)
        download_form.addWidget(self.hf_revision, 1, 1)
        download_form.addWidget(QLabel("Target folder"), 2, 0)
        download_form.addWidget(self.hf_target, 2, 1, 1, 2)
        download_form.addWidget(browse, 2, 3)
        download_form.addWidget(self.download_button, 3, 1)
        download_form.addWidget(self.cancel_download_button, 3, 2)
        download_form.addWidget(self.load_dataset_button, 3, 3)
        download_form.addWidget(self.download_progress, 4, 0, 1, 4)
        download_form.addWidget(self.download_status, 5, 0, 1, 4)
        layout.addWidget(download_box)
        layout.addStretch()
        self.tabs.addTab(page, "Overview")

    def _build_inspector(self):
        page, layout = self._page(
            "Dataset Inspector",
            "Original and corrupted data stay side by side; the source is never overwritten.",
        )
        controls = QHBoxLayout()
        self.episode_spin = QSpinBox()
        self.step_spin = QSpinBox()
        self.episode_spin.valueChanged.connect(self.refresh_inspector)
        self.step_spin.valueChanged.connect(self.refresh_inspector)
        self.signal_dimension_combo = QComboBox()
        self.signal_dimension_combo.currentIndexChanged.connect(self.refresh_inspector)
        self.signal_dimension_combo.currentIndexChanged.connect(self.preview_corruption)
        self.episode_label = QLabel("Episode (0 total)")
        self.step_label = QLabel("Step (0 total)")
        controls.addWidget(self.episode_label)
        controls.addWidget(self.episode_spin)
        controls.addWidget(self.step_label)
        controls.addWidget(self.step_spin)
        controls.addWidget(QLabel("State / action dimension"))
        controls.addWidget(self.signal_dimension_combo)
        controls.addStretch()
        self.timestamp_label = QLabel()
        controls.addWidget(self.timestamp_label)
        layout.addLayout(controls)
        images = QHBoxLayout()
        original_box = QGroupBox("Original RGB")
        original_layout = QVBoxLayout(original_box)
        self.original_image = QLabel()
        original_layout.addWidget(self.original_image)
        corrupt_box = QGroupBox("Working copy")
        corrupt_layout = QVBoxLayout(corrupt_box)
        self.corrupt_image = QLabel()
        corrupt_layout.addWidget(self.corrupt_image)
        images.addWidget(original_box)
        images.addWidget(corrupt_box)
        plots = QVBoxLayout()
        self.signal_plot = pg.PlotWidget(title="State and action over time")
        self.signal_plot.addLegend()
        self.signal_plot.setLabel("bottom", "Time", units="s")
        self.signal_plot.setLabel("left", "Value")
        self.signal_plot.showGrid(x=True, y=True, alpha=0.2)
        self.quality_plot = pg.PlotWidget(title="Modality quality over time")
        self.quality_plot.addLegend()
        self.quality_plot.setLabel("bottom", "Time", units="s")
        self.quality_plot.setLabel("left", "Quality", units="0-1")
        self.quality_plot.setYRange(0.0, 1.05, padding=0)
        self.quality_plot.showGrid(x=True, y=True, alpha=0.2)
        self.quality_plot.setXLink(self.signal_plot)
        self.quality_plot.setMaximumHeight(210)
        plots.addWidget(self.signal_plot, 2)
        plots.addWidget(self.quality_plot, 1)
        images.addLayout(plots, 1)
        layout.addLayout(images, 1)
        self.inspector_detail = QLabel()
        layout.addWidget(self.inspector_detail)
        self.tabs.addTab(page, "Dataset Inspector")

    def _build_corruption(self):
        page, layout = self._page(
            "Corruption Studio",
            "Configure a seeded transformation and preview it without touching the original episode.",
        )
        form_box = QGroupBox("Corruption controls")
        form = QFormLayout(form_box)
        self.corruption_type = QComboBox()
        self.corruption_type.addItems(
            ["temporal_shift", "frame_drop", "action_noise", "state_anomaly"]
        )
        self.corruption_mode = QComboBox()
        self.corruption_value = QDoubleSpinBox()
        self.corruption_value.setRange(-10, 10)
        self.corruption_value.setValue(2)
        self.corruption_value.setDecimals(2)
        self.corruption_seed = QSpinBox()
        self.corruption_seed.setRange(0, 999999)
        self.corruption_seed.setValue(7)
        self.corruption_seed.setToolTip("Used only by stochastic corruption types.")
        form.addRow("Type", self.corruption_type)
        form.addRow("Target", self.corruption_mode)
        form.addRow("Shift / probability / sigma", self.corruption_value)
        form.addRow("Random seed", self.corruption_seed)
        self.corruption_type.currentIndexChanged.connect(self._update_corruption_controls)
        self.corruption_mode.currentIndexChanged.connect(self.preview_corruption)
        self.corruption_value.valueChanged.connect(self.preview_corruption)
        self.corruption_seed.valueChanged.connect(self.preview_corruption)
        buttons = QHBoxLayout()
        for label, callback, variant in [
            ("Apply", self.apply_preview, "primary"),
            ("Reset", self.reset_corruption, "danger"),
            ("Save Config", self.save_corruption_config, "secondary"),
        ]:
            button = QPushButton(label)
            button.setProperty("variant", variant)
            button.clicked.connect(callback)
            buttons.addWidget(button)
        buttons.addStretch()
        layout.addWidget(form_box)
        layout.addLayout(buttons)
        self.temporal_shift_diagram = TemporalShiftDiagram()
        layout.addWidget(self.temporal_shift_diagram)
        self.corruption_visual_summary = QLabel()
        self.corruption_visual_summary.setStyleSheet("color:#536079;font-weight:600")
        layout.addWidget(self.corruption_visual_summary)
        self.frame_drop_plot = pg.PlotWidget(title="Dropped frame locations")
        self.frame_drop_plot.setLabel("bottom", "Time", units="s")
        self.frame_drop_plot.setLabel("left", "Frame status")
        self.frame_drop_plot.getAxis("left").setTicks([[(0, "kept"), (1, "dropped")]])
        self.frame_drop_plot.setYRange(-0.25, 1.25, padding=0)
        self.frame_drop_plot.showGrid(x=True, y=True, alpha=0.2)
        layout.addWidget(self.frame_drop_plot, 1)
        self.corruption_plot = pg.PlotWidget(title="Original vs corrupted action[0]")
        self.corruption_plot.addLegend()
        self.corruption_plot.setLabel("bottom", "Time", units="s")
        self.corruption_plot.showGrid(x=True, y=True, alpha=0.2)
        layout.addWidget(self.corruption_plot, 1)
        self.state_change_table = QTableWidget(0, 5)
        self.state_change_table.setHorizontalHeaderLabels(
            ["Frame", "Time (s)", "Original", "New", "Delta"]
        )
        self.state_change_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.state_change_table.setAlternatingRowColors(True)
        self.state_change_table.setMaximumHeight(220)
        self.state_change_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        layout.addWidget(self.state_change_table)
        self.corruption_info = QTextEdit()
        self.corruption_info.setReadOnly(True)
        self.corruption_info.setMaximumHeight(140)
        layout.addWidget(self.corruption_info)
        self.tabs.addTab(page, "Corruption Studio")
        self._update_corruption_controls()

    def _build_training(self):
        page, layout = self._page(
            "Training Monitor",
            "Training runs in a worker thread; stopping saves a resumable checkpoint safely.",
        )
        parameter_box = QGroupBox("Training parameters")
        controls = QGridLayout(parameter_box)
        self.model_combo = QComboBox()
        self.model_combo.addItems(["bc_mlp", "act_lite", "quality_act"])
        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cpu", "cuda"])
        self.device_combo.setCurrentText("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(3)
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 4096)
        self.batch_size_spin.setValue(32)
        self.learning_rate_spin = QDoubleSpinBox()
        self.learning_rate_spin.setRange(0.000001, 1.0)
        self.learning_rate_spin.setDecimals(6)
        self.learning_rate_spin.setSingleStep(0.0001)
        self.learning_rate_spin.setValue(0.003)
        self.weight_decay_spin = QDoubleSpinBox()
        self.weight_decay_spin.setRange(0.0, 1.0)
        self.weight_decay_spin.setDecimals(6)
        self.weight_decay_spin.setSingleStep(0.0001)
        self.weight_decay_spin.setValue(0.0001)
        self.validation_split_spin = QDoubleSpinBox()
        self.validation_split_spin.setRange(0.01, 0.5)
        self.validation_split_spin.setDecimals(2)
        self.validation_split_spin.setSingleStep(0.05)
        self.validation_split_spin.setValue(0.15)
        self.test_split_spin = QDoubleSpinBox()
        self.test_split_spin.setRange(0.01, 0.5)
        self.test_split_spin.setDecimals(2)
        self.test_split_spin.setSingleStep(0.05)
        self.test_split_spin.setValue(0.15)
        self.loss_combo = QComboBox()
        self.loss_combo.addItems(["mse", "smooth_l1"])
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2_147_483_647)
        self.seed_spin.setValue(7)
        self.grad_clip_spin = QDoubleSpinBox()
        self.grad_clip_spin.setRange(0.0, 100.0)
        self.grad_clip_spin.setDecimals(2)
        self.grad_clip_spin.setValue(0.0)
        self.grad_clip_spin.setToolTip("0 disables gradient clipping")

        parameters = [
            ("Model", self.model_combo, "Recommended: compare all three"),
            ("Device", self.device_combo, "Recommended: auto, or CUDA when available"),
            ("Epochs", self.epochs_spin, "Recommended: start with 5; tune on validation"),
            ("Batch size", self.batch_size_spin, "Recommended: BC 32; ACT 16"),
            ("Learning rate", self.learning_rate_spin, "Recommended: BC 0.003; ACT 0.001"),
            ("Weight decay", self.weight_decay_spin, "Recommended: 0.0001"),
            ("Validation split", self.validation_split_spin, "Recommended: 0.15-0.20"),
            ("Test split", self.test_split_spin, "Recommended: 0.15-0.20"),
            ("Loss", self.loss_combo, "Recommended: BC MSE; ACT Smooth L1"),
            ("Seed", self.seed_spin, "Recommended: 7; report at least 3 seeds"),
            ("Gradient clip", self.grad_clip_spin, "Recommended: ACT 1.0; 0 disables"),
        ]
        for index, (label, widget, recommendation) in enumerate(parameters):
            row, group = divmod(index, 2)
            column = group * 3
            hint = QLabel(recommendation)
            hint.setStyleSheet("color:#6d7890;font-size:11px")
            controls.addWidget(QLabel(label), row, column)
            controls.addWidget(widget, row, column + 1)
            controls.addWidget(hint, row, column + 2)
        controls.setColumnStretch(2, 1)
        controls.setColumnStretch(5, 1)
        self.validation_split_spin.valueChanged.connect(self._on_split_settings_changed)
        self.test_split_spin.valueChanged.connect(self._on_split_settings_changed)
        self.seed_spin.valueChanged.connect(self._on_split_settings_changed)
        self.split_summary_label = QLabel("No dataset loaded")
        self.split_summary_label.setStyleSheet("color:#536079;font-weight:600")
        self.training_step_estimate_label = QLabel(
            "Estimated optimizer steps: load a dataset first"
        )
        self.training_step_estimate_label.setObjectName("trainingStepEstimate")
        self.training_step_estimate_label.setStyleSheet(
            "color:#3157d5;background:#eef2ff;border:1px solid #d7e0ff;"
            "border-radius:7px;padding:8px 10px;font-weight:650"
        )
        self.training_step_estimate_label.setToolTip(
            "Optimizer steps only. Validation batches do not update the model and are not included."
        )
        self.epochs_spin.valueChanged.connect(self._update_training_step_estimate)
        self.batch_size_spin.valueChanged.connect(self._update_training_step_estimate)

        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Pause / Stop")
        self.resume_button = QPushButton("Resume")
        self.stop_button.setProperty("variant", "danger")
        self.resume_button.setProperty("variant", "secondary")
        self.stop_button.setEnabled(False)
        self.resume_button.setEnabled(False)
        self.start_button.clicked.connect(self.start_training)
        self.stop_button.clicked.connect(self.stop_training)
        self.resume_button.clicked.connect(lambda: self.start_training(resume=True))
        buttons = QHBoxLayout()
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.stop_button)
        buttons.addWidget(self.resume_button)
        buttons.addStretch()
        layout.addWidget(parameter_box)
        layout.addWidget(self.split_summary_label)
        layout.addWidget(self.training_step_estimate_label)
        layout.addLayout(buttons)
        self.train_progress = QProgressBar()
        layout.addWidget(self.train_progress)
        self.training_plot = pg.PlotWidget(title="Live training loss")
        self.training_curve = self.training_plot.plot(pen=pg.mkPen("#3157d5", width=2))
        layout.addWidget(self.training_plot, 1)
        self.training_status = QTextEdit()
        self.training_status.setReadOnly(True)
        self.training_status.setMaximumHeight(150)
        layout.addWidget(self.training_status)
        self.tabs.addTab(page, "Training Monitor")

    def _build_evaluation(self):
        page, layout = self._page(
            "Evaluation & Compare",
            "Compare offline prediction quality, smoothness, latency, and model size on the clean test set.",
        )
        buttons = QHBoxLayout()
        evaluate = QPushButton("Evaluate latest checkpoint")
        evaluate.clicked.connect(self.evaluate_latest)
        choose = QPushButton("Choose checkpoint…")
        choose.setProperty("variant", "secondary")
        choose.clicked.connect(self.choose_checkpoint)
        buttons.addWidget(evaluate)
        buttons.addWidget(choose)
        buttons.addStretch()
        layout.addLayout(buttons)
        self.metrics_table = QTableWidget(0, 2)
        self.metrics_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self._configure_metrics_table()
        layout.addWidget(self.metrics_table, 1)
        self.compare_plot = pg.PlotWidget(title="Prediction vs target (action[0])")
        self.compare_plot.addLegend()
        layout.addWidget(self.compare_plot, 1)
        self.tabs.addTab(page, "Evaluation & Compare")

    def _build_report(self):
        page, layout = self._page(
            "Report",
            "Export measured evidence, experiment settings, and reproducibility metadata.",
        )
        export = QPushButton("Export HTML + JSON report")
        export.clicked.connect(self.export_report)
        layout.addWidget(export, alignment=Qt.AlignmentFlag.AlignLeft)
        self.report_preview = QTextEdit()
        self.report_preview.setReadOnly(True)
        self.report_preview.setPlainText(
            "No evaluation has been run yet. Train and evaluate a model to populate this report."
        )
        layout.addWidget(self.report_preview, 1)
        self.tabs.addTab(page, "Report")

    def refresh_all(self):
        if not self.episodes:
            self.episode_spin.setRange(0, 0)
            self.step_spin.setRange(0, 0)
            self.overview_values["episodes"].setText("0")
            self.overview_values["frames"].setText("0")
            self.overview_values["shape"].setText("not loaded")
            self.overview_values["device"].setText("CUDA" if torch.cuda.is_available() else "CPU")
            self.overview_values["status"].setText("No data")
            self.overview_values["best"].setText("not run")
            self.episode_label.setText("Episode (0 total)")
            self.step_label.setText("Step (0 total)")
            self.original_image.clear()
            self.corrupt_image.clear()
            self.timestamp_label.setText("No timestamp")
            self.inspector_detail.setText("No dataset loaded. Download and import a dataset first.")
            self.signal_plot.clear()
            self.quality_plot.clear()
            self.signal_dimension_combo.clear()
            self.corruption_plot.clear()
            self.frame_drop_plot.clear()
            self.state_change_table.setRowCount(0)
            self.corruption_info.setPlainText("No dataset loaded.")
            self.split_summary_label.setText("No dataset loaded")
            self._update_training_step_estimate()
            self.start_button.setEnabled(False)
            return
        episode = self.episodes[0]
        self.episode_spin.setRange(0, len(self.episodes) - 1)
        self.step_spin.setRange(0, len(episode["timestamp"]) - 1)
        self.overview_values["episodes"].setText(str(len(self.episodes)))
        self.overview_values["frames"].setText(
            str(sum(len(item["timestamp"]) for item in self.episodes))
        )
        image_shape = episode["observation.image"].shape
        shape_text = (
            f"{image_shape[1]} cameras · " + " × ".join(map(str, image_shape[2:])) + " each"
            if len(image_shape) == 5
            else "1 camera · " + " × ".join(map(str, image_shape[1:]))
        )
        self.overview_values["shape"].setText(shape_text)
        self.overview_values["device"].setText("CUDA" if torch.cuda.is_available() else "CPU")
        self.overview_values["status"].setText("Ready")
        self.overview_values["best"].setText(
            "not run" if not self.last_metrics else f"MSE {self.last_metrics['action_mse']:.4f}"
        )
        self.start_button.setEnabled(True)
        self.episode_label.setText(f"Episode ({len(self.episodes)} total)")
        self._update_inspector_dimensions(episode)
        self.refresh_inspector()
        self.preview_corruption()

    def _update_default_download_target(self, repo_id: str):
        if self._download_target_is_default:
            self.hf_target.setText(str(default_dataset_path(repo_id)))

    def _mark_download_target_custom(self, _text: str):
        self._download_target_is_default = False

    def choose_download_target(self):
        initial = str(Path(self.hf_target.text() or ".").expanduser().resolve())
        path = QFileDialog.getExistingDirectory(self, "Choose dataset target folder", initial)
        if path:
            self._download_target_is_default = False
            self.hf_target.setText(path)

    def start_dataset_download(self):
        if self.download_thread and self.download_thread.isRunning():
            return
        repo_id = self.hf_repo_id.text().strip()
        if not repo_id or "/" not in repo_id:
            QMessageBox.information(
                self, "Repository ID required", "Enter a repository ID such as owner/dataset."
            )
            return
        if self._download_target_is_default:
            self.hf_target.setText(str(default_dataset_path(repo_id)))
        target = self.hf_target.text().strip()
        if not target:
            QMessageBox.information(self, "Target required", "Choose a target folder.")
            return
        self.download_thread = QThread(self)
        self.download_worker = DatasetDownloadWorker(
            repo_id, Path(target), self.hf_revision.text().strip() or "main"
        )
        self.download_worker.moveToThread(self.download_thread)
        self.download_thread.started.connect(self.download_worker.run)
        self.download_worker.progress.connect(self.on_download_progress)
        self.download_worker.finished.connect(self.on_download_finished)
        self.download_worker.failed.connect(self.on_download_failed)
        self.download_worker.cancelled.connect(self.on_download_cancelled)
        self.download_worker.finished.connect(self.download_thread.quit)
        self.download_worker.failed.connect(self.download_thread.quit)
        self.download_worker.cancelled.connect(self.download_thread.quit)
        self.download_button.setEnabled(False)
        self.load_dataset_button.setEnabled(False)
        self.cancel_download_button.setEnabled(True)
        self.download_progress.setRange(0, 0)
        self.download_status.setText(f"Reading {repo_id}...")
        self.download_thread.start()

    def cancel_dataset_download(self):
        if self.download_worker:
            self.download_worker.stop()
        if self.load_worker:
            self.load_worker.stop()
            self.download_status.setText(
                "Cancellation requested; waiting for the current transfer..."
            )

    def on_download_progress(self, event):
        total = int(event.get("total", 0))
        if total:
            self.download_progress.setRange(0, 100)
            self.download_progress.setValue(int(event.get("percent", 0)))
            self.download_status.setText(
                f"{event.get('description', 'Downloading')}: "
                f"{event.get('current', 0)}/{total} files"
            )
        else:
            self.download_progress.setRange(0, 0)
            self.download_status.setText(event.get("description", "Preparing download"))

    def on_download_finished(self, target):
        self.download_progress.setRange(0, 100)
        self.download_progress.setValue(100)
        self.download_button.setEnabled(True)
        self.load_dataset_button.setEnabled(True)
        self.cancel_download_button.setEnabled(False)
        self.download_status.setText(f"Downloaded to {target}")
        self.statusBar().showMessage(f"Hugging Face dataset downloaded to {target}")

    def on_download_cancelled(self):
        self.download_progress.setRange(0, 100)
        self.download_button.setEnabled(True)
        self.load_dataset_button.setEnabled(True)
        self.cancel_download_button.setEnabled(False)
        self.download_status.setText(
            "Download cancelled. Already downloaded files remain reusable."
        )

    def on_download_failed(self, details):
        self.download_progress.setRange(0, 100)
        self.download_button.setEnabled(True)
        self.load_dataset_button.setEnabled(True)
        self.cancel_download_button.setEnabled(False)
        self.download_status.setText("Download failed; see the error message.")
        QMessageBox.critical(self, "Dataset download failed", details.splitlines()[-1])

    def start_dataset_load(self):
        if self.load_thread and self.load_thread.isRunning():
            return
        target = self.hf_target.text().strip()
        if not target:
            QMessageBox.information(self, "Dataset folder required", "Choose a dataset folder.")
            return
        self.load_thread = QThread(self)
        self.load_worker = DatasetLoadWorker(Path(target))
        self.load_worker.moveToThread(self.load_thread)
        self.load_thread.started.connect(self.load_worker.run)
        self.load_worker.progress.connect(self.on_dataset_load_progress)
        self.load_worker.finished.connect(self.on_dataset_loaded)
        self.load_worker.failed.connect(self.on_dataset_load_failed)
        self.load_worker.cancelled.connect(self.on_dataset_load_cancelled)
        self.load_worker.finished.connect(self.load_thread.quit)
        self.load_worker.failed.connect(self.load_thread.quit)
        self.load_worker.cancelled.connect(self.load_thread.quit)
        self.download_button.setEnabled(False)
        self.load_dataset_button.setEnabled(False)
        self.cancel_download_button.setEnabled(True)
        self.download_progress.setRange(0, 100)
        self.download_progress.setValue(0)
        self.download_status.setText(f"Loading LeRobot dataset from {target}...")
        self.load_thread.start()

    def on_dataset_load_progress(self, event):
        self.download_progress.setRange(0, 100)
        self.download_progress.setValue(int(event.get("percent", 0)))
        self.download_status.setText(event.get("description", "Loading dataset"))

    def on_dataset_loaded(self, episodes, metadata):
        self.original_episodes = episodes
        self.active_corruption_config = None
        self._rebuild_episode_partitions()
        self.dataset_metadata = metadata
        self.current_model = None
        self.current_stats = None
        self.last_checkpoint = None
        self.last_metrics = None
        self.model_test_episodes = []
        self.download_button.setEnabled(True)
        self.load_dataset_button.setEnabled(True)
        self.cancel_download_button.setEnabled(False)
        self.download_progress.setValue(100)
        self.download_status.setText(
            f"Loaded {metadata['episodes']} episodes / {metadata['frames']} frames / "
            f"{metadata.get('camera_count', 1)} cameras"
        )
        self.statusBar().showMessage(f"LeRobot dataset loaded from {metadata['root']}")
        self.refresh_all()

    def on_dataset_load_cancelled(self):
        self.download_button.setEnabled(True)
        self.load_dataset_button.setEnabled(True)
        self.cancel_download_button.setEnabled(False)
        self.download_status.setText("Dataset loading cancelled")

    def on_dataset_load_failed(self, details):
        self.download_button.setEnabled(True)
        self.load_dataset_button.setEnabled(True)
        self.cancel_download_button.setEnabled(False)
        self.download_status.setText("Dataset loading failed; see the error message.")
        QMessageBox.critical(self, "Dataset loading failed", details.splitlines()[-1])

    def _rebuild_episode_partitions(self):
        if not self.original_episodes:
            return
        self.episode_split = split_episode_indices(
            len(self.original_episodes),
            self.validation_split_spin.value(),
            self.test_split_spin.value(),
            self.seed_spin.value(),
        )
        self.episode_partitions = {
            episode_index: partition
            for partition, indices in self.episode_split.items()
            for episode_index in indices
        }
        working_episodes = list(self.original_episodes)
        if self.active_corruption_config:
            config = self.active_corruption_config
            for episode_index in self.episode_split["train"]:
                working_episodes[episode_index] = apply_corruption(
                    self.original_episodes[episode_index],
                    {**config, "seed": config["seed"] + episode_index},
                )
        self.episodes = working_episodes
        self.training_episodes = [self.episodes[index] for index in self.episode_split["train"]]
        self.validation_episodes = [
            self.original_episodes[index] for index in self.episode_split["validation"]
        ]
        self.test_episodes = [self.original_episodes[index] for index in self.episode_split["test"]]
        corruption = (
            self.active_corruption_config["type"] if self.active_corruption_config else "clean"
        )
        self.split_summary_label.setText(
            f"Episode split · train {len(self.training_episodes)} ({corruption})"
            f" · validation {len(self.validation_episodes)} (clean)"
            f" · test {len(self.test_episodes)} (clean)"
        )
        self._update_training_step_estimate()

    def _update_training_step_estimate(self, *_args):
        if not self.original_episodes or not self.training_episodes:
            self.training_step_estimate_label.setText(
                "Estimated optimizer steps: load a dataset first"
            )
            return
        dataset_frames = sum(len(episode["timestamp"]) for episode in self.original_episodes)
        training_frames = sum(len(episode["timestamp"]) for episode in self.training_episodes)
        batch_size = self.batch_size_spin.value()
        epochs = self.epochs_spin.value()
        steps_per_epoch = math.ceil(training_frames / batch_size)
        total_steps = epochs * steps_per_epoch
        self.training_step_estimate_label.setText(
            f"Estimated optimizer steps: {total_steps:,} = {epochs} epoch(s) × "
            f"ceil({training_frames:,} training frames / batch {batch_size:,}) "
            f"· {steps_per_epoch:,} steps/epoch · {dataset_frames:,} total dataset frames"
        )

    def _on_split_settings_changed(self, *_args):
        if not self.original_episodes:
            return
        try:
            self._rebuild_episode_partitions()
            self.refresh_all()
        except ValueError as exc:
            self.split_summary_label.setText(str(exc))

    def _update_inspector_dimensions(self, episode):
        count = min(episode["observation.state"].shape[1], episode["action"].shape[1])
        selected = min(max(self.signal_dimension_combo.currentIndex(), 0), count - 1)
        expected = [f"dimension[{index}]" for index in range(count)]
        current = [
            self.signal_dimension_combo.itemText(index)
            for index in range(self.signal_dimension_combo.count())
        ]
        if current != expected:
            self.signal_dimension_combo.blockSignals(True)
            self.signal_dimension_combo.clear()
            self.signal_dimension_combo.addItems(expected)
            self.signal_dimension_combo.setCurrentIndex(selected)
            self.signal_dimension_combo.blockSignals(False)

    def refresh_inspector(self, *_args):
        if not self.episodes:
            return
        episode_index = min(self.episode_spin.value(), len(self.episodes) - 1)
        original, current = self.original_episodes[episode_index], self.episodes[episode_index]
        ensure_modal_quality(current)
        frame_count = len(current["timestamp"])
        self.step_spin.setMaximum(frame_count - 1)
        self.step_label.setText(f"Step ({frame_count} total)")
        step = min(self.step_spin.value(), frame_count - 1)
        self._update_inspector_dimensions(current)
        dimension = max(self.signal_dimension_combo.currentIndex(), 0)
        self.original_image.setPixmap(_image_pixmap(original["observation.image"][step]))
        self.corrupt_image.setPixmap(_image_pixmap(current["observation.image"][step]))
        timestamp = float(current["timestamp"][step])
        interval = (
            float(current["timestamp"][step] - current["timestamp"][step - 1]) if step else 0.0
        )
        self.timestamp_label.setText(f"t={timestamp:.3f}s · Δt={interval:.3f}s")
        partition = self.episode_partitions.get(episode_index, "not split")
        image_quality = current["image_quality"][step].numpy().round(2).tolist()
        image_missing = current["image_missing_mask"][step].numpy().tolist()
        image_offset = current["image_time_offset"][step].numpy().round(4).tolist()
        self.inspector_detail.setText(
            f"partition={partition}  image quality={image_quality}  "
            f"state quality={float(current['state_quality'][step]):.2f}  "
            f"action-label quality={float(current['action_label_quality'][step]):.2f}  "
            f"image missing={image_missing}  "
            f"state missing={bool(current['state_missing_mask'][step])}  "
            f"action-label missing={bool(current['action_label_missing_mask'][step])}  "
            f"image offset={image_offset}s  "
            f"state offset={float(current['state_time_offset'][step]):.4f}s  "
            f"action-label offset={float(current['action_label_time_offset'][step]):.4f}s  "
            f"cameras={1 if current['observation.image'].ndim == 4 else current['observation.image'].shape[1]}  "
            f"state={current['observation.state'][step].numpy().round(3)}"
        )
        self.signal_plot.clear()
        timestamps = current["timestamp"].numpy()
        self.signal_plot.plot(
            timestamps,
            current["observation.state"][:, dimension].numpy(),
            pen="#3157d5",
            name=f"state[{dimension}]",
        )
        self.signal_plot.plot(
            timestamps,
            current["action"][:, dimension].numpy(),
            pen="#eb6a49",
            name=f"action[{dimension}]",
        )
        cursor_pen = pg.mkPen("#172033", width=2, style=Qt.PenStyle.DashLine)
        self.signal_cursor = pg.InfiniteLine(
            pos=timestamp, angle=90, pen=cursor_pen, movable=False, label=f"Step {step}"
        )
        self.signal_plot.addItem(self.signal_cursor)
        self.quality_plot.clear()
        image_colors = ["#13a67a", "#3157d5", "#8b5cf6", "#0891b2"]
        for camera_index in range(current["image_quality"].shape[1]):
            self.quality_plot.plot(
                timestamps,
                current["image_quality"][:, camera_index].numpy(),
                pen=pg.mkPen(image_colors[camera_index % len(image_colors)], width=2),
                name=f"image[{camera_index}]",
            )
        self.quality_plot.plot(
            timestamps,
            current["state_quality"].numpy(),
            pen=pg.mkPen("#eb6a49", width=2),
            name="state",
        )
        self.quality_plot.plot(
            timestamps,
            current["action_label_quality"].numpy(),
            pen=pg.mkPen("#d19a00", width=2),
            name="action label",
        )
        self.quality_cursor = pg.InfiniteLine(
            pos=timestamp, angle=90, pen=cursor_pen, movable=False
        )
        self.quality_plot.addItem(self.quality_cursor)

    def _update_corruption_controls(self, *_args):
        kind = self.corruption_type.currentText()
        targets = {
            "temporal_shift": ["image", "state", "action"],
            "frame_drop": ["image"],
            "action_noise": ["action"],
            "state_anomaly": ["state"],
        }[kind]
        selected = self.corruption_mode.currentText()
        self.corruption_mode.blockSignals(True)
        self.corruption_mode.clear()
        self.corruption_mode.addItems(targets)
        if selected in targets:
            self.corruption_mode.setCurrentText(selected)
        self.corruption_mode.setEnabled(len(targets) > 1)
        self.corruption_mode.blockSignals(False)
        is_temporal_shift = kind == "temporal_shift"
        self.corruption_seed.setEnabled(not is_temporal_shift)
        self._set_corruption_visual_visibility(kind)
        if is_temporal_shift:
            self.temporal_shift_diagram.set_configuration(
                self.corruption_mode.currentText(), int(self.corruption_value.value())
            )
        self.preview_corruption()

    def _set_corruption_visual_visibility(self, kind: str):
        self.temporal_shift_diagram.setVisible(kind == "temporal_shift")
        self.frame_drop_plot.setVisible(kind == "frame_drop")
        self.corruption_plot.setVisible(kind in {"action_noise", "state_anomaly"})
        self.state_change_table.setVisible(kind == "state_anomaly")
        self.corruption_visual_summary.setVisible(kind != "temporal_shift")

    def _corruption_config(self):
        kind, value, seed = (
            self.corruption_type.currentText(),
            self.corruption_value.value(),
            self.corruption_seed.value(),
        )
        if kind == "temporal_shift":
            return {
                "type": kind,
                "target": self.corruption_mode.currentText(),
                "shift": int(value),
                "boundary": "mask",
                "seed": seed,
            }
        if kind == "frame_drop":
            return {
                "type": kind,
                "probability": min(1.0, max(0.0, abs(value))),
                "replacement": "previous",
                "seed": seed,
            }
        if kind == "action_noise":
            return {"type": kind, "mode": "gaussian", "sigma": abs(value), "seed": seed}
        return {
            "type": kind,
            "mode": "spike",
            "probability": min(1.0, max(0.0, abs(value))),
            "seed": seed,
        }

    def preview_corruption(self):
        kind = self.corruption_type.currentText()
        self._set_corruption_visual_visibility(kind)
        dimension = max(self.signal_dimension_combo.currentIndex(), 0)
        if kind == "action_noise":
            self.corruption_plot.setTitle(f"Original vs corrupted action[{dimension}]")
        elif kind == "state_anomaly":
            self.corruption_plot.setTitle(f"Original vs corrupted state[{dimension}]")
        if kind == "temporal_shift":
            self.temporal_shift_diagram.set_configuration(
                self.corruption_mode.currentText(), int(self.corruption_value.value())
            )
        if not self.original_episodes:
            self.corruption_plot.clear()
            self.frame_drop_plot.clear()
            self.state_change_table.setRowCount(0)
            self.corruption_visual_summary.setText("Load a dataset to generate this preview.")
            return
        try:
            config = self._corruption_config()
            preview_index = self.episode_split["train"][0] if self.episode_split["train"] else 0
            original = self.original_episodes[preview_index]
            preview = apply_corruption(original, config)
            self.corruption_plot.clear()
            self.frame_drop_plot.clear()
            self.state_change_table.setRowCount(0)
            timestamps = original["timestamp"].numpy()
            affected = np.asarray(preview["provenance"]["affected_indices"], dtype=int)
            if kind == "action_noise":
                self.corruption_plot.setTitle(f"Original vs corrupted action[{dimension}]")
                self.corruption_plot.plot(
                    timestamps,
                    original["action"][:, dimension].numpy(),
                    pen="#9da8c2",
                    name="original",
                )
                self.corruption_plot.plot(
                    timestamps,
                    preview["action"][:, dimension].numpy(),
                    pen="#eb6a49",
                    name="preview",
                )
                self.corruption_visual_summary.setText(
                    f"Training Episode {preview_index} · action noise changes "
                    f"{len(affected)} / {len(timestamps)} frames."
                )
            elif kind == "state_anomaly":
                dimension = min(
                    max(self.signal_dimension_combo.currentIndex(), 0),
                    original["observation.state"].shape[1] - 1,
                )
                original_state = original["observation.state"][:, dimension].numpy()
                changed_state = preview["observation.state"][:, dimension].numpy()
                self.corruption_plot.setTitle(f"Original vs corrupted state[{dimension}]")
                self.corruption_plot.plot(
                    timestamps, original_state, pen="#9da8c2", name="original"
                )
                self.corruption_plot.plot(
                    timestamps, changed_state, pen="#3157d5", name="corrupted"
                )
                if affected.size:
                    self.corruption_plot.plot(
                        timestamps[affected],
                        changed_state[affected],
                        pen=None,
                        symbol="o",
                        symbolSize=7,
                        symbolBrush="#d83b4c",
                        name="changed frames",
                    )
                preview_indices = ", ".join(map(str, affected[:12]))
                suffix = "..." if len(affected) > 12 else ""
                self.corruption_visual_summary.setText(
                    f"Training Episode {preview_index} · Changed "
                    f"{len(affected)} / {len(timestamps)} state frames"
                    f" · indices: {preview_indices}{suffix}"
                )
                self.state_change_table.setRowCount(len(affected))
                for row, frame_index in enumerate(affected):
                    original_value = float(original_state[frame_index])
                    new_value = float(changed_state[frame_index])
                    values = (
                        str(frame_index),
                        f"{float(timestamps[frame_index]):.4f}",
                        f"{original_value:.6f}",
                        f"{new_value:.6f}",
                        f"{new_value - original_value:+.6f}",
                    )
                    for column, value in enumerate(values):
                        self.state_change_table.setItem(row, column, QTableWidgetItem(value))
            elif kind == "frame_drop":
                dropped = np.zeros(len(timestamps), dtype=float)
                dropped[affected] = 1.0
                if len(timestamps) > 1:
                    time_edges = np.concatenate(
                        (
                            [timestamps[0] - (timestamps[1] - timestamps[0]) / 2],
                            (timestamps[:-1] + timestamps[1:]) / 2,
                            [timestamps[-1] + (timestamps[-1] - timestamps[-2]) / 2],
                        )
                    )
                else:
                    time_edges = np.asarray([timestamps[0] - 0.5, timestamps[0] + 0.5])
                self.frame_drop_plot.plot(
                    time_edges,
                    dropped,
                    pen=pg.mkPen("#9da8c2", width=1),
                    stepMode="center",
                    fillLevel=0,
                    brush=pg.mkBrush(216, 59, 76, 80),
                )
                if affected.size:
                    self.frame_drop_plot.plot(
                        timestamps[affected],
                        np.ones(len(affected)),
                        pen=None,
                        symbol="o",
                        symbolSize=8,
                        symbolBrush="#d83b4c",
                    )
                preview_indices = ", ".join(map(str, affected[:16]))
                suffix = "..." if len(affected) > 16 else ""
                self.corruption_visual_summary.setText(
                    f"Training Episode {preview_index} · Dropped "
                    f"{len(affected)} / {len(timestamps)} image frames"
                    f" · indices: {preview_indices}{suffix}"
                )
            self.corruption_info.setPlainText(
                json.dumps(
                    {
                        "latest_event": preview["provenance"],
                        "corruption_events": preview.get("corruption_events", []),
                    },
                    indent=2,
                )
            )
            self._preview_episode = preview
        except Exception as exc:
            self.corruption_info.setPlainText(str(exc))

    def apply_preview(self):
        if not self.original_episodes:
            QMessageBox.information(
                self, "No dataset", "Load a dataset before applying corruption."
            )
            return
        config = self._corruption_config()
        self.active_corruption_config = config
        try:
            self._rebuild_episode_partitions()
        except ValueError as exc:
            QMessageBox.critical(self, "Invalid episode split", str(exc))
            return
        self.statusBar().showMessage(
            f"Applied {config['type']} to {len(self.training_episodes)} training episodes only"
        )
        self.refresh_all()

    def reset_corruption(self):
        self.active_corruption_config = None
        self._rebuild_episode_partitions()
        self.refresh_all()
        self.statusBar().showMessage("Working copy reset")

    def save_corruption_config(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save corruption config", "configs/corruption/custom.yaml", "YAML (*.yaml)"
        )
        if path:
            save_config(self._corruption_config(), path)

    def _training_configs(self):
        name = self.model_combo.currentText()
        horizon = 1 if name == "bc_mlp" else 8
        state_dim = int(self.episodes[0]["observation.state"].shape[1])
        action_dim = int(self.episodes[0]["action"].shape[1])
        model = {
            "name": name,
            "state_dim": state_dim,
            "action_dim": action_dim,
            "input_mode": "image_state",
            "horizon": horizon,
            "hidden_dim": 64,
            "num_layers": 1,
            "num_heads": 4,
        }
        training = {
            "epochs": self.epochs_spin.value(),
            "batch_size": self.batch_size_spin.value(),
            "learning_rate": self.learning_rate_spin.value(),
            "weight_decay": self.weight_decay_spin.value(),
            "validation_split": self.validation_split_spin.value(),
            "test_split": self.test_split_spin.value(),
            "loss": self.loss_combo.currentText(),
            "seed": self.seed_spin.value(),
            "grad_clip": self.grad_clip_spin.value(),
            "device": (
                ("cuda" if torch.cuda.is_available() else "cpu")
                if self.device_combo.currentText() == "auto"
                else self.device_combo.currentText()
            ),
            "horizon": horizon,
            "quality_weighted_loss": name == "quality_act",
            "lambda_smooth": 0.01,
            "episode_split": self.episode_split,
        }
        return model, training

    def start_training(self, checked=False, resume=False):
        if not self.episodes:
            QMessageBox.information(
                self,
                "No training dataset",
                "Download and import a compatible dataset before starting training.",
            )
            return
        if self.thread and self.thread.isRunning():
            return
        try:
            self._rebuild_episode_partitions()
        except ValueError as exc:
            QMessageBox.critical(self, "Invalid episode split", str(exc))
            return
        model_config, training_config = self._training_configs()
        resume_path = self.last_checkpoint if resume else None
        if resume_path is not None:
            output = resume_path.parent
            training_config["checkpoint_name"] = resume_path.name
        else:
            data_condition = (self.active_corruption_config or {"type": "clean"}).get(
                "type", "clean"
            )
            checkpoint = new_model_checkpoint_path(model_config["name"], data_condition)
            output = checkpoint.parent
            training_config["checkpoint_name"] = checkpoint.name
        self.thread = QThread(self)
        self.worker = TrainingWorker(
            self.training_episodes,
            model_config,
            training_config,
            output,
            resume_path,
            validation_episodes=self.validation_episodes,
        )
        self.model_test_episodes = list(self.test_episodes)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.on_training_progress)
        self.worker.finished.connect(self.on_training_finished)
        self.worker.failed.connect(self.on_training_failed)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.training_requested.connect(self.worker.stop)
        self._losses = []
        self.training_status.setPlainText(
            f"Checkpoint will be saved to {output / training_config['checkpoint_name']}"
        )
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.resume_button.setEnabled(False)
        training_frames = sum(len(episode["timestamp"]) for episode in self.training_episodes)
        estimated_steps = self.epochs_spin.value() * math.ceil(
            training_frames / self.batch_size_spin.value()
        )
        self.train_progress.setRange(0, estimated_steps)
        self.train_progress.setValue(0)
        self.overview_values["status"].setText("Training")
        self.thread.start()

    def stop_training(self):
        if self.worker:
            self.worker.stop_event.set()
            self.training_status.append(
                "Stop requested; saving after the current batch/validation pass…"
            )

    def on_training_progress(self, event):
        if event.get("epoch_complete"):
            self.train_progress.setValue(int(event["step"]))
            self.training_status.append(
                f"Epoch {event['epoch'] + 1}: train={event['train_loss']:.5f}, val={event['validation_loss']:.5f}"
            )
        elif "train_loss" in event:
            self.train_progress.setValue(int(event["step"]))
            self._losses.append(event["train_loss"])
            self.training_curve.setData(self._losses)
            self.training_status.setPlainText(
                f"step {event['step']} · loss {event['train_loss']:.6f} · lr {event['learning_rate']:.2e} · ETA {event['eta_seconds']:.1f}s"
            )

    def on_training_finished(self, model, result):
        self.current_model, self.last_checkpoint = model, result.checkpoint
        payload = torch.load(result.checkpoint, map_location="cpu", weights_only=False)
        self.current_stats = NormalizationStats.from_dict(payload["stats"])
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.resume_button.setEnabled(True)
        self.overview_values["status"].setText("Stopped" if result.stopped else "Complete")
        self.training_status.append(
            f"Checkpoint: {result.checkpoint} ({result.training_seconds:.2f}s)"
        )

    def on_training_failed(self, details):
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.overview_values["status"].setText("Error")
        self.training_status.setPlainText(details)
        QMessageBox.critical(self, "Training failed", details.splitlines()[-1])

    def choose_checkpoint(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Choose checkpoint", str(models_root()), "PyTorch checkpoint (*.pt)"
        )
        if not paths:
            return
        try:
            comparisons = {}
            for path in paths:
                payload = torch.load(path, map_location="cpu", weights_only=False)
                model_config = payload.get("config", {}).get("model")
                if not model_config:
                    raise ValueError(f"{path} has no model configuration")
                model = build_policy(model_config)
                load_checkpoint(path, model)
                stats = NormalizationStats.from_dict(payload["stats"])
                saved_split = payload.get("config", {}).get("episode_split", {})
                test_indices = saved_split.get("test", self.episode_split["test"])
                evaluation_episodes = [self.original_episodes[index] for index in test_indices]
                metrics = evaluate_policy(
                    model,
                    evaluation_episodes,
                    stats=stats,
                    output_dir=Path("runs/gui/comparisons") / Path(path).stem,
                )
                comparisons[Path(path).name] = metrics
                self.current_model, self.current_stats = model, stats
                self.last_checkpoint = Path(path)
                self.model_test_episodes = evaluation_episodes
                index = self.model_combo.findText(model_config["name"])
                if index >= 0:
                    self.model_combo.setCurrentIndex(index)
            self.last_metrics = next(iter(comparisons.values()))
            self._show_comparisons(comparisons)
            self.tabs.setCurrentIndex(4)
        except Exception as exc:
            QMessageBox.critical(self, "Checkpoint load failed", str(exc))

    def evaluate_latest(self):
        if self.current_model is None:
            QMessageBox.information(
                self,
                "No model",
                "Train a model in this session before evaluation, or choose its checkpoint.",
            )
            return
        try:
            self.last_metrics = evaluate_policy(
                self.current_model,
                self.model_test_episodes or self.test_episodes,
                stats=self.current_stats,
                output_dir="runs/gui/evaluation",
            )
            self._show_comparisons({"latest": self.last_metrics})
            self.report_preview.setPlainText(json.dumps(self.last_metrics, indent=2))
            self.overview_values["best"].setText(f"MSE {self.last_metrics['action_mse']:.4f}")
            csv_path = Path("runs/gui/evaluation/predictions.csv")
            if csv_path.exists():
                import pandas as pd

                frame = pd.read_csv(csv_path)
                self.compare_plot.clear()
                self.compare_plot.addLegend()
                self.compare_plot.plot(frame["target_0"].to_numpy(), pen="#9da8c2", name="target")
                self.compare_plot.plot(
                    frame["prediction_0"].to_numpy(), pen="#3157d5", name="prediction"
                )
            self.tabs.setCurrentIndex(4)
        except Exception as exc:
            QMessageBox.critical(self, "Evaluation failed", str(exc))

    def _show_comparisons(self, comparisons):
        names = list(comparisons)
        keys = [key for key in next(iter(comparisons.values())) if key != "per_episode"]
        self.metrics_table.setColumnCount(len(names) + 1)
        self.metrics_table.setHorizontalHeaderLabels(["Metric", *names])
        self._configure_metrics_table()
        self.metrics_table.setRowCount(len(keys))
        for row, key in enumerate(keys):
            metric_widget = QWidget()
            metric_layout = QHBoxLayout(metric_widget)
            metric_layout.setContentsMargins(6, 0, 6, 0)
            metric_layout.setSpacing(7)
            metric_layout.addWidget(QLabel(key))
            help_label = QLabel("?")
            help_label.setObjectName(f"metric_help_{key}")
            help_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            help_label.setFixedSize(17, 17)
            help_label.setStyleSheet(
                "QLabel { color:#3157d5; border:1px solid #3157d5; "
                "border-radius:8px; font-weight:700; }"
            )
            help_label.setToolTip(METRIC_HELP.get(key, "No description available."))
            metric_layout.addWidget(help_label)
            metric_layout.addStretch()
            self.metrics_table.setCellWidget(row, 0, metric_widget)
            for column, name in enumerate(names, start=1):
                value = comparisons[name].get(key)
                self.metrics_table.setItem(
                    row,
                    column,
                    QTableWidgetItem("not run" if value is None else str(value)),
                )

    def _configure_metrics_table(self):
        header = self.metrics_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self.metrics_table.setColumnWidth(0, 280)
        for column in range(1, self.metrics_table.columnCount()):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Stretch)

    def export_report(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export report", "reports/gui_report.html", "HTML (*.html)"
        )
        if not path:
            return
        run = {
            "name": "gui-run",
            "model": self.model_combo.currentText(),
            "corruption": self.active_corruption_config or {"type": "clean"},
            "seed": self.seed_spin.value(),
            "validation_split": self.validation_split_spin.value(),
            "test_split": self.test_split_spin.value(),
            "episode_split": self.episode_split,
            "metrics": self.last_metrics or {"evaluation_scope": "not run"},
        }
        target = generate_report([run], path)
        self.statusBar().showMessage(f"Report exported to {target}")

    def closeEvent(self, event):
        if self.worker:
            self.worker.stop_event.set()
        if self.thread and self.thread.isRunning():
            self.thread.quit()
            self.thread.wait(5000)
        if self.download_worker:
            self.download_worker.stop_event.set()
        if self.download_thread and self.download_thread.isRunning():
            self.download_thread.quit()
            self.download_thread.wait(5000)
        if self.load_worker:
            self.load_worker.stop_event.set()
        if self.load_thread and self.load_thread.isRunning():
            self.load_thread.quit()
            self.load_thread.wait(5000)
        event.accept()


def main() -> int:
    if sys.platform == "win32":
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "Sync2Act.RobotDataQualityLab"
        )
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("Sync2Act")
    icon_path = _asset_path("sync2act.ico")
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))
    window = MainWindow()
    window.show()
    return app.exec()
