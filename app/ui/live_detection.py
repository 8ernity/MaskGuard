"""
Live webcam face mask detection view.
Displays real-time video feed, bounding boxes, confidence tags, FPS HUD, and telemetry.
"""

from __future__ import annotations

from datetime import datetime

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from app.core.camera_worker import CameraWorker
from app.core.inference_pipeline import InferencePipeline
from app.utils.config import AppConfig
from app.utils.image_utils import save_image_bgr
from app.utils.resource_paths import get_screenshots_dir


class LiveDetectionView(QWidget):
    """
    Real-time webcam inference view.
    """

    session_stats_updated = Signal(int, int, int)  # total_unique, masks, nomasks

    def __init__(
        self,
        config: AppConfig,
        pipeline: InferencePipeline | None = None,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.config = config
        self.pipeline = pipeline or InferencePipeline(config=config)
        self.worker: CameraWorker | None = None
        self.current_frame_bgr: np.ndarray | None = None

        self._setup_ui()

    def _setup_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(24, 20, 24, 20)
        main_layout.setSpacing(16)

        # Header and Control Bar
        header_layout = QHBoxLayout()

        title_box = QVBoxLayout()
        title = QLabel("Real-Time Camera Detection")
        title.setStyleSheet("font-size: 22px; font-weight: 700; color: #F0F6FC;")
        desc = QLabel("Live face detection and mask classification with dynamic confidence thresholds.")
        desc.setStyleSheet("font-size: 13px; color: #8B949E;")
        title_box.addWidget(title)
        title_box.addWidget(desc)
        header_layout.addLayout(title_box)
        header_layout.addStretch()

        # Camera Selector
        header_layout.addWidget(QLabel("Camera:"))
        self.camera_combo = QComboBox()
        self.camera_combo.setMinimumWidth(110)
        self._populate_cameras()
        header_layout.addWidget(self.camera_combo)

        # Start / Stop Toggle Button
        self.btn_toggle_camera = QPushButton("Start Camera")
        self.btn_toggle_camera.setProperty("class", "PrimaryButton")
        self.btn_toggle_camera.setCursor(Qt.PointingHandCursor)
        self.btn_toggle_camera.clicked.connect(self.toggle_camera)
        header_layout.addWidget(self.btn_toggle_camera)

        # Screenshot Button
        self.btn_screenshot = QPushButton("Capture Snapshot")
        self.btn_screenshot.setProperty("class", "SecondaryButton")
        self.btn_screenshot.setCursor(Qt.PointingHandCursor)
        self.btn_screenshot.setEnabled(False)
        self.btn_screenshot.clicked.connect(self.capture_snapshot)
        header_layout.addWidget(self.btn_screenshot)

        main_layout.addLayout(header_layout)

        # Center Video Viewport
        self.video_label = QLabel("Camera feed stopped. Press 'Start Camera' to begin detection.")
        self.video_label.setObjectName("VideoViewport")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 420)
        self.video_label.setStyleSheet(
            "background-color: #010409; border: 1px solid #30363D; border-radius: 8px; color: #8B949E; font-size: 14px;"
        )
        main_layout.addWidget(self.video_label, 1)

        # Bottom Bar: Live Metrics & Sliders
        bottom_layout = QHBoxLayout()
        bottom_layout.setSpacing(16)

        # Live telemetry pills
        stats_frame = QFrame()
        stats_frame.setProperty("class", "Card")
        sf_layout = QHBoxLayout(stats_frame)
        sf_layout.setContentsMargins(14, 8, 14, 8)
        sf_layout.setSpacing(20)

        self.lbl_fps = QLabel("FPS: --")
        self.lbl_fps.setStyleSheet("color: #10B981; font-weight: 700; font-size: 13px;")

        self.lbl_faces = QLabel("In Frame: 0")
        self.lbl_faces.setStyleSheet("color: #F0F6FC; font-weight: 600; font-size: 13px;")

        self.lbl_masks = QLabel("Masks: 0")
        self.lbl_masks.setStyleSheet("color: #10B981; font-weight: 600; font-size: 13px;")

        self.lbl_nomasks = QLabel("No-Masks: 0")
        self.lbl_nomasks.setStyleSheet("color: #EF4444; font-weight: 600; font-size: 13px;")

        self.lbl_uncertain = QLabel("Uncertain: 0")
        self.lbl_uncertain.setStyleSheet("color: #F59E0B; font-weight: 600; font-size: 13px;")

        sf_layout.addWidget(self.lbl_fps)
        sf_layout.addWidget(self.lbl_faces)
        sf_layout.addWidget(self.lbl_masks)
        sf_layout.addWidget(self.lbl_nomasks)
        sf_layout.addWidget(self.lbl_uncertain)
        bottom_layout.addWidget(stats_frame, 3)

        # Confidence Slider Frame
        slider_frame = QFrame()
        slider_frame.setProperty("class", "Card")
        sl_layout = QHBoxLayout(slider_frame)
        sl_layout.setContentsMargins(14, 8, 14, 8)
        sl_layout.setSpacing(12)

        sl_layout.addWidget(QLabel("Confidence Threshold:"))
        self.slider_conf = QSlider(Qt.Horizontal)
        self.slider_conf.setRange(20, 95)
        self.slider_conf.setValue(int(self.config.confidence_threshold * 100))
        self.slider_conf.valueChanged.connect(self._on_conf_slider_changed)
        sl_layout.addWidget(self.slider_conf)

        self.lbl_conf_val = QLabel(f"{int(self.config.confidence_threshold * 100)}%")
        self.lbl_conf_val.setStyleSheet("font-weight: 700; color: #10B981; min-width: 35px;")
        sl_layout.addWidget(self.lbl_conf_val)

        bottom_layout.addWidget(slider_frame, 2)
        main_layout.addLayout(bottom_layout)

    def _populate_cameras(self) -> None:
        self.camera_combo.clear()
        cameras = CameraWorker.get_available_cameras(max_to_test=3)
        for idx in cameras:
            self.camera_combo.addItem(f"Camera {idx}", idx)

        # Select saved index if present
        idx = self.camera_combo.findData(self.config.camera_index)
        if idx >= 0:
            self.camera_combo.setCurrentIndex(idx)

    def _on_conf_slider_changed(self, value: int) -> None:
        float_val = value / 100.0
        self.config.confidence_threshold = float_val
        self.pipeline.classifier.confidence_threshold = float_val
        self.lbl_conf_val.setText(f"{value}%")

    def toggle_camera(self) -> None:
        """Start or stop active webcam feed."""
        if self.worker is None:
            self.start_camera()
        else:
            self.stop_camera()

    def start_camera(self) -> None:
        """Initialize camera worker thread."""
        cam_idx = self.camera_combo.currentData()
        if cam_idx is None:
            cam_idx = 0

        self.config.camera_index = cam_idx

        # Create worker
        self.worker = CameraWorker(
            camera_index=cam_idx,
            pipeline=self.pipeline,
            target_fps=30,
        )

        from PySide6.QtCore import QThread
        self.camera_thread = QThread(self)
        self.worker.moveToThread(self.camera_thread)

        self.camera_thread.started.connect(self.worker.start_capture)
        self.worker.frame_ready.connect(self.on_frame_ready)
        self.worker.camera_error.connect(self.on_camera_error)
        self.worker.stopped.connect(self.on_camera_stopped)
        self.worker.stopped.connect(self.camera_thread.quit)

        self.camera_thread.start()

        self.btn_toggle_camera.setText("Stop Camera")
        self.btn_toggle_camera.setProperty("class", "DangerButton")
        self.btn_toggle_camera.style().unpolish(self.btn_toggle_camera)
        self.btn_toggle_camera.style().polish(self.btn_toggle_camera)

        self.btn_screenshot.setEnabled(True)
        self.camera_combo.setEnabled(False)

    def stop_camera(self) -> None:
        """Signal camera worker to stop."""
        if self.worker:
            self.worker.stop()
            self.worker = None

        if hasattr(self, "camera_thread") and self.camera_thread.isRunning():
            self.camera_thread.quit()
            self.camera_thread.wait(1500)

        self.btn_toggle_camera.setText("Start Camera")
        self.btn_toggle_camera.setProperty("class", "PrimaryButton")
        self.btn_toggle_camera.style().unpolish(self.btn_toggle_camera)
        self.btn_toggle_camera.style().polish(self.btn_toggle_camera)

        self.btn_screenshot.setEnabled(False)
        self.camera_combo.setEnabled(True)
        self.video_label.setText("Camera feed stopped.")

    @Slot(np.ndarray, list, dict)
    def on_frame_ready(self, frame_bgr: np.ndarray, detections: list, stats: dict) -> None:
        """Receive and render annotated frame from camera worker."""
        self.current_frame_bgr = frame_bgr

        # Convert BGR to QImage
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Scale pixmap smoothly to fit viewport while preserving aspect ratio
        scaled_pixmap = QPixmap.fromImage(q_img).scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.video_label.setPixmap(scaled_pixmap)

        # Update HUD stats
        self.lbl_fps.setText(f"FPS: {stats.get('fps', 0.0):.1f}")
        self.lbl_faces.setText(f"In Frame: {stats.get('faces_in_frame', 0)}")
        self.lbl_masks.setText(f"Masks: {stats.get('mask_count', 0)}")
        self.lbl_nomasks.setText(f"No-Masks: {stats.get('nomask_count', 0)}")
        self.lbl_uncertain.setText(f"Uncertain: {stats.get('uncertain_count', 0)}")

        self.session_stats_updated.emit(
            stats.get("total_unique_faces", 0),
            stats.get("mask_count", 0),
            stats.get("nomask_count", 0),
        )

    @Slot(str)
    def on_camera_error(self, err_msg: str) -> None:
        self.video_label.setText(f"Camera Error: {err_msg}")
        self.stop_camera()

    @Slot()
    def on_camera_stopped(self) -> None:
        self.btn_toggle_camera.setText("Start Camera")
        self.btn_toggle_camera.setProperty("class", "PrimaryButton")
        self.btn_screenshot.setEnabled(False)
        self.camera_combo.setEnabled(True)

    def capture_snapshot(self) -> None:
        """Capture and save current annotated frame."""
        if self.current_frame_bgr is not None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            target_path = get_screenshots_dir() / f"snapshot_{ts}.jpg"
            if save_image_bgr(target_path, self.current_frame_bgr):
                self.video_label.setToolTip(f"Snapshot saved: {target_path.name}")
