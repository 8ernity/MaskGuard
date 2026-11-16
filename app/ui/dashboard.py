"""
Dashboard view for MaskGuard desktop application.
Displays system readiness, real-time statistics cards, CNN specifications, and quick actions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.utils.config import AppConfig
from app.utils.resource_paths import (
    get_default_model_path,
    get_model_metadata_path,
    get_results_dir,
)
from app.core.camera_worker import CameraWorker


class DashboardView(QWidget):
    """
    Main overview and monitoring dashboard.
    """

    navigate_to = Signal(str)  # Emits destination page key e.g. "live", "image", "video"

    def __init__(self, config: AppConfig, parent: QWidget | None = None):
        super().__init__(parent)
        self.config = config
        self._setup_ui()
        self.refresh_dashboard()

    def _setup_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(28, 24, 28, 24)
        main_layout.setSpacing(20)

        # Header Title
        title_box = QVBoxLayout()
        page_title = QLabel("System Dashboard")
        page_title.setStyleSheet("font-size: 24px; font-weight: 700; color: #F0F6FC;")
        page_desc = QLabel("Real-time telemetry, model readiness, and quick detection actions.")
        page_desc.setStyleSheet("font-size: 13px; color: #8B949E; margin-top: 2px;")
        title_box.addWidget(page_title)
        title_box.addWidget(page_desc)
        main_layout.addLayout(title_box)

        # Status Strip (Model status + Camera status)
        status_strip = QHBoxLayout()
        status_strip.setSpacing(16)

        # Model Status Card
        self.model_status_card = QFrame()
        self.model_status_card.setProperty("class", "Card")
        m_layout = QHBoxLayout(self.model_status_card)
        m_layout.setContentsMargins(16, 12, 16, 12)

        m_info = QVBoxLayout()
        m_label = QLabel("NEURAL NETWORK STATUS")
        m_label.setProperty("class", "CardTitle")
        self.model_file_label = QLabel("mask_detector.h5")
        self.model_file_label.setStyleSheet("font-weight: 600; color: #F0F6FC; font-size: 14px;")
        m_info.addWidget(m_label)
        m_info.addWidget(self.model_file_label)
        m_layout.addLayout(m_info)
        m_layout.addStretch()

        self.model_badge = QLabel("READY")
        self.model_badge.setProperty("class", "BadgeOnline")
        m_layout.addWidget(self.model_badge)
        status_strip.addWidget(self.model_status_card)

        # Camera Status Card
        self.camera_status_card = QFrame()
        self.camera_status_card.setProperty("class", "Card")
        c_layout = QHBoxLayout(self.camera_status_card)
        c_layout.setContentsMargins(16, 12, 16, 12)

        c_info = QVBoxLayout()
        c_label = QLabel("OPTICAL SENSOR STATUS")
        c_label.setProperty("class", "CardTitle")
        self.camera_info_label = QLabel(f"Camera Device {self.config.camera_index}")
        self.camera_info_label.setStyleSheet("font-weight: 600; color: #F0F6FC; font-size: 14px;")
        c_info.addWidget(c_label)
        c_info.addWidget(self.camera_info_label)
        c_layout.addLayout(c_info)
        c_layout.addStretch()

        self.camera_badge = QLabel("AVAILABLE")
        self.camera_badge.setProperty("class", "BadgeOnline")
        c_layout.addWidget(self.camera_badge)
        status_strip.addWidget(self.camera_status_card)

        main_layout.addLayout(status_strip)

        # Statistics Cards Grid
        stats_layout = QGridLayout()
        stats_layout.setSpacing(16)

        # Stat Card 1: Unique Faces
        self.stat_faces_val = QLabel("0")
        self.stat_faces_val.setProperty("class", "StatValue")
        stats_layout.addWidget(self._create_stat_card("TOTAL FACES DETECTED", self.stat_faces_val, "#38BDF8"), 0, 0)

        # Stat Card 2: Masks Count
        self.stat_masks_val = QLabel("0")
        self.stat_masks_val.setProperty("class", "StatValue")
        stats_layout.addWidget(self._create_stat_card("MASK COMPLIANCE", self.stat_masks_val, "#10B981"), 0, 1)

        # Stat Card 3: No-Mask Count
        self.stat_nomask_val = QLabel("0")
        self.stat_nomask_val.setProperty("class", "StatValue")
        stats_layout.addWidget(self._create_stat_card("NO-MASK VIOLATIONS", self.stat_nomask_val, "#EF4444"), 0, 2)

        # Stat Card 4: Model Test Accuracy
        self.stat_acc_val = QLabel("98.5%")
        self.stat_acc_val.setProperty("class", "StatValue")
        stats_layout.addWidget(self._create_stat_card("MODEL TEST ACCURACY", self.stat_acc_val, "#A855F7"), 0, 3)

        main_layout.addLayout(stats_layout)

        # Middle Content Row: Model Architecture Specs & Quick Actions
        mid_row = QHBoxLayout()
        mid_row.setSpacing(16)

        # CNN Architecture Summary Card
        arch_card = QFrame()
        arch_card.setProperty("class", "Card")
        arch_layout = QVBoxLayout(arch_card)
        arch_layout.setSpacing(12)

        arch_title = QLabel("Custom CNN Architecture Specifications")
        arch_title.setProperty("class", "SectionHeader")
        arch_layout.addWidget(arch_title)

        specs_grid = QGridLayout()
        specs_grid.setSpacing(10)
        specs_grid.addWidget(QLabel("Input Dimensions:"), 0, 0)
        specs_grid.addWidget(self._create_code_label("128 x 128 x 3 (RGB)"), 0, 1)

        specs_grid.addWidget(QLabel("Convolutional Backbone:"), 1, 0)
        specs_grid.addWidget(self._create_code_label("3 Blocks (32, 64, 128 filters + BN + ReLU + MaxPool)"), 1, 1)

        specs_grid.addWidget(QLabel("Global Pooling:"), 2, 0)
        specs_grid.addWidget(self._create_code_label("GlobalAveragePooling2D"), 2, 1)

        specs_grid.addWidget(QLabel("Classifier Head:"), 3, 0)
        specs_grid.addWidget(self._create_code_label("Dense(128, ReLU) -> Dropout(0.5) -> Dense(1, Sigmoid)"), 3, 1)

        specs_grid.addWidget(QLabel("Face Detection Engine:"), 4, 0)
        specs_grid.addWidget(self._create_code_label("OpenCV Frontal Face Cascade (Haar) with Padding"), 4, 1)

        arch_layout.addLayout(specs_grid)
        arch_layout.addStretch()
        mid_row.addWidget(arch_card, 3)

        # Quick Actions Card
        actions_card = QFrame()
        actions_card.setProperty("class", "Card")
        actions_layout = QVBoxLayout(actions_card)
        actions_layout.setSpacing(12)

        actions_title = QLabel("Quick Actions")
        actions_title.setProperty("class", "SectionHeader")
        actions_layout.addWidget(actions_title)

        btn_live = QPushButton("Launch Live Webcam Detection")
        btn_live.setProperty("class", "PrimaryButton")
        btn_live.setCursor(Qt.PointingHandCursor)
        btn_live.clicked.connect(lambda: self.navigate_to.emit("live"))
        actions_layout.addWidget(btn_live)

        btn_image = QPushButton("Analyze Static Photo")
        btn_image.setProperty("class", "SecondaryButton")
        btn_image.setCursor(Qt.PointingHandCursor)
        btn_image.clicked.connect(lambda: self.navigate_to.emit("image"))
        actions_layout.addWidget(btn_image)

        btn_video = QPushButton("Batch Video Analysis")
        btn_video.setProperty("class", "SecondaryButton")
        btn_video.setCursor(Qt.PointingHandCursor)
        btn_video.clicked.connect(lambda: self.navigate_to.emit("video"))
        actions_layout.addWidget(btn_video)

        btn_analytics = QPushButton("Inspect Model Analytics")
        btn_analytics.setProperty("class", "SecondaryButton")
        btn_analytics.setCursor(Qt.PointingHandCursor)
        btn_analytics.clicked.connect(lambda: self.navigate_to.emit("analytics"))
        actions_layout.addWidget(btn_analytics)

        actions_layout.addStretch()
        mid_row.addWidget(actions_card, 2)

        main_layout.addLayout(mid_row)
        main_layout.addStretch()

    def _create_stat_card(self, title: str, value_label: QLabel, accent_color: str) -> QFrame:
        card = QFrame()
        card.setProperty("class", "StatCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 14, 16, 14)

        t = QLabel(title)
        t.setProperty("class", "CardTitle")
        value_label.setStyleSheet(f"font-size: 26px; font-weight: 700; color: {accent_color}; margin: 4px 0px;")

        layout.addWidget(t)
        layout.addWidget(value_label)
        return card

    def _create_code_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(
            "background-color: #21262D; color: #10B981; font-family: Consolas, monospace; "
            "padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: 600;"
        )
        return lbl

    def refresh_dashboard(self) -> None:
        """Query disk assets and update badges & telemetry."""
        # 1. Model Status
        model_path = Path(self.config.model_path)
        if model_path.exists():
            self.model_badge.setText("READY")
            self.model_badge.setProperty("class", "BadgeOnline")
            self.model_file_label.setText(model_path.name)
        else:
            self.model_badge.setText("MODEL NOT FOUND")
            self.model_badge.setProperty("class", "BadgeOffline")
            self.model_file_label.setText("Train model via training/train.py")

        self.model_badge.style().unpolish(self.model_badge)
        self.model_badge.style().polish(self.model_badge)

        # 2. Camera Availability
        cams = CameraWorker.get_available_cameras(max_to_test=2)
        if cams and self.config.camera_index in cams:
            self.camera_badge.setText(f"DEVICE {self.config.camera_index} READY")
            self.camera_badge.setProperty("class", "BadgeOnline")
        elif cams:
            self.camera_badge.setText(f"DEVICE {cams[0]} DETECTED")
            self.camera_badge.setProperty("class", "BadgeOnline")
        else:
            self.camera_badge.setText("CAMERA OFFLINE")
            self.camera_badge.setProperty("class", "BadgeOffline")

        self.camera_badge.style().unpolish(self.camera_badge)
        self.camera_badge.style().polish(self.camera_badge)

        # 3. Model Metrics
        metrics_file = get_results_dir() / "evaluation_metrics.json"
        if metrics_file.exists():
            try:
                with open(metrics_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                acc = data.get("test_accuracy", 0.0)
                self.stat_acc_val.setText(f"{acc * 100:.1f}%")
            except Exception:
                self.stat_acc_val.setText("N/A")
        else:
            self.stat_acc_val.setText("--")

    def update_session_counts(self, unique_faces: int, masks: int, nomasks: int) -> None:
        """Update live telemetry counters from active sessions."""
        self.stat_faces_val.setText(str(unique_faces))
        self.stat_masks_val.setText(str(masks))
        self.stat_nomask_val.setText(str(nomasks))
