"""
Main Window for the MaskGuard desktop application.
Orchestrates sidebar navigation, view switching, and application lifecycle.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from app.core.inference_pipeline import InferencePipeline
from app.ui.dashboard import DashboardView
from app.ui.image_analysis import ImageAnalysisView
from app.ui.live_detection import LiveDetectionView
from app.ui.model_analytics import ModelAnalyticsView
from app.ui.settings import SettingsView
from app.ui.styles import get_stylesheet
from app.ui.video_analysis import VideoAnalysisView
from app.utils.config import AppConfig


class MainWindow(QMainWindow):
    """
    Top-level application window.
    """

    def __init__(self, config: AppConfig | None = None):
        super().__init__()
        self.config = config or AppConfig.load()
        self.pipeline = InferencePipeline(config=self.config)

        self.setWindowTitle("MaskGuard — AI Face Mask Detection System")
        self.setMinimumSize(1100, 720)
        self.resize(1280, 800)

        self._setup_ui()
        self.apply_theme(self.config.theme)

    def _setup_ui(self) -> None:
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        root_layout = QHBoxLayout(central_widget)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # 1. Left Sidebar Navigation
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 16)
        sidebar_layout.setSpacing(4)

        # Title / Brand Logo Box
        title_label = QLabel("MaskGuard")
        title_label.setObjectName("AppTitle")
        subtitle_label = QLabel("AI VISION DASHBOARD")
        subtitle_label.setObjectName("AppSubtitle")
        sidebar_layout.addWidget(title_label)
        sidebar_layout.addWidget(subtitle_label)

        # Navigation Buttons
        self.nav_buttons: dict[str, QPushButton] = {}
        nav_items = [
            ("dashboard", "Dashboard"),
            ("live", "Live Detection"),
            ("image", "Image Analysis"),
            ("video", "Video Analysis"),
            ("analytics", "Model Analytics"),
            ("settings", "Settings"),
        ]

        for key, text in nav_items:
            btn = QPushButton(text)
            btn.setProperty("class", "NavButton")
            btn.setCheckable(True)
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda checked, k=key: self.switch_view(k))
            sidebar_layout.addWidget(btn)
            self.nav_buttons[key] = btn

        sidebar_layout.addStretch()

        # Footer Info in sidebar
        version_lbl = QLabel("v1.0.0 | Custom CNN & OpenCV")
        version_lbl.setStyleSheet("color: #484F58; font-size: 11px; padding: 0 16px;")
        sidebar_layout.addWidget(version_lbl)

        root_layout.addWidget(sidebar)

        # 2. Main Views (QStackedWidget)
        self.stack = QStackedWidget()

        self.view_dashboard = DashboardView(self.config)
        self.view_live = LiveDetectionView(self.config, pipeline=self.pipeline)
        self.view_image = ImageAnalysisView(self.config, pipeline=self.pipeline)
        self.view_video = VideoAnalysisView(self.config, pipeline=self.pipeline)
        self.view_analytics = ModelAnalyticsView(self.config)
        self.view_settings = SettingsView(self.config)

        # Connect signals
        self.view_dashboard.navigate_to.connect(self.switch_view)
        self.view_live.session_stats_updated.connect(self.view_dashboard.update_session_counts)
        self.view_settings.theme_changed.connect(self.apply_theme)
        self.view_settings.config_saved.connect(self._on_config_saved)

        self.stack.addWidget(self.view_dashboard)   # 0
        self.stack.addWidget(self.view_live)        # 1
        self.stack.addWidget(self.view_image)       # 2
        self.stack.addWidget(self.view_video)       # 3
        self.stack.addWidget(self.view_analytics)   # 4
        self.stack.addWidget(self.view_settings)    # 5

        self.view_indices = {
            "dashboard": 0,
            "live": 1,
            "image": 2,
            "video": 3,
            "analytics": 4,
            "settings": 5,
        }

        root_layout.addWidget(self.stack, 1)

        # Initial view selection
        self.switch_view("dashboard")

        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("MaskGuard AI System initialized and ready.")

    def switch_view(self, key: str) -> None:
        """Switch stacked view and update active nav button."""
        if key not in self.view_indices:
            return

        idx = self.view_indices[key]

        # Stop camera if leaving live detection
        if self.stack.currentIndex() == self.view_indices["live"] and key != "live":
            self.view_live.stop_camera()

        self.stack.setCurrentIndex(idx)

        # Update button active states
        for btn_key, btn in self.nav_buttons.items():
            is_active = (btn_key == key)
            btn.setChecked(is_active)
            btn.setProperty("active", "true" if is_active else "false")
            btn.style().unpolish(btn)
            btn.style().polish(btn)

        # Refresh dashboard telemetry when switching to dashboard
        if key == "dashboard":
            self.view_dashboard.refresh_dashboard()
        elif key == "analytics":
            self.view_analytics.reload_metrics()

    def apply_theme(self, theme: str) -> None:
        """Apply dark or light theme QSS across application."""
        qss = get_stylesheet(theme)
        self.setStyleSheet(qss)
        self.config.theme = theme

    def _on_config_saved(self, updated_config: AppConfig) -> None:
        """Sync updated configuration across subviews."""
        self.config = updated_config
        self.pipeline.config = updated_config
        self.pipeline.detector.update_parameters(
            scale_factor=updated_config.face_scale_factor,
            min_neighbors=updated_config.face_min_neighbors,
            padding=updated_config.face_padding,
        )
        self.pipeline.classifier.confidence_threshold = updated_config.confidence_threshold
        if Path(updated_config.model_path).exists():
            self.pipeline.classifier.load_model(updated_config.model_path)
        self.status_bar.showMessage("Configuration preferences updated and applied.", 3000)

    def closeEvent(self, event: QCloseEvent) -> None:
        """Ensure clean shutdown of background workers and camera resources."""
        self.view_live.stop_camera()
        if self.view_video.processor:
            self.view_video.cancel_processing()
        event.accept()
