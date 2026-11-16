"""
Offline video processing view for MaskGuard.
Allows selecting MP4/AVI files, runs sequential frame inference, renders previews, and exports annotated videos.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.core.inference_pipeline import InferencePipeline
from app.core.video_processor import VideoProcessor
from app.utils.config import AppConfig
from app.utils.resource_paths import get_videos_dir


class VideoAnalysisView(QWidget):
    """
    Video upload and batch processing view.
    """

    def __init__(
        self,
        config: AppConfig,
        pipeline: InferencePipeline | None = None,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.config = config
        self.pipeline = pipeline or InferencePipeline(config=config)
        self.processor: VideoProcessor | None = None
        self.processor_thread: QThread | None = None
        self.selected_video_path: Path | None = None

        self._setup_ui()

    def _setup_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(24, 20, 24, 20)
        main_layout.setSpacing(16)

        # Header Title and Action Buttons
        header_layout = QHBoxLayout()

        title_box = QVBoxLayout()
        title = QLabel("Video Batch Analysis")
        title.setStyleSheet("font-size: 22px; font-weight: 700; color: #F0F6FC;")
        desc = QLabel("Process MP4 and AVI video files frame-by-frame, track compliance, and export annotated output.")
        desc.setStyleSheet("font-size: 13px; color: #8B949E;")
        title_box.addWidget(title)
        title_box.addWidget(desc)
        header_layout.addLayout(title_box)
        header_layout.addStretch()

        self.btn_select_video = QPushButton("Select Video File")
        self.btn_select_video.setProperty("class", "PrimaryButton")
        self.btn_select_video.setCursor(Qt.PointingHandCursor)
        self.btn_select_video.clicked.connect(self.browse_video)
        header_layout.addWidget(self.btn_select_video)

        self.btn_start_processing = QPushButton("Start Processing")
        self.btn_start_processing.setProperty("class", "PrimaryButton")
        self.btn_start_processing.setCursor(Qt.PointingHandCursor)
        self.btn_start_processing.setEnabled(False)
        self.btn_start_processing.clicked.connect(self.start_processing)
        header_layout.addWidget(self.btn_start_processing)

        self.btn_cancel_processing = QPushButton("Cancel")
        self.btn_cancel_processing.setProperty("class", "DangerButton")
        self.btn_cancel_processing.setCursor(Qt.PointingHandCursor)
        self.btn_cancel_processing.setEnabled(False)
        self.btn_cancel_processing.clicked.connect(self.cancel_processing)
        header_layout.addWidget(self.btn_cancel_processing)

        main_layout.addLayout(header_layout)

        # Center Video Preview Viewport
        self.preview_label = QLabel("No video loaded. Click 'Select Video File' to begin.")
        self.preview_label.setObjectName("VideoViewport")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(640, 380)
        self.preview_label.setStyleSheet(
            "background-color: #010409; border: 1px solid #30363D; border-radius: 8px; color: #8B949E; font-size: 14px;"
        )
        main_layout.addWidget(self.preview_label, 1)

        # Progress and Status Card
        progress_card = QFrame()
        progress_card.setProperty("class", "Card")
        p_layout = QVBoxLayout(progress_card)
        p_layout.setContentsMargins(16, 12, 16, 12)
        p_layout.setSpacing(8)

        status_row = QHBoxLayout()
        self.lbl_status = QLabel("Status: Standby")
        self.lbl_status.setStyleSheet("color: #F0F6FC; font-weight: 600; font-size: 13px;")
        self.lbl_frames_info = QLabel("Frame: 0 / 0 (0%)")
        self.lbl_frames_info.setStyleSheet("color: #8B949E; font-size: 12px;")
        status_row.addWidget(self.lbl_status)
        status_row.addStretch()
        status_row.addWidget(self.lbl_frames_info)
        p_layout.addLayout(status_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        p_layout.addWidget(self.progress_bar)

        self.lbl_summary_stats = QLabel("")
        self.lbl_summary_stats.setStyleSheet("color: #10B981; font-weight: 600; font-size: 12px;")
        p_layout.addWidget(self.lbl_summary_stats)

        main_layout.addWidget(progress_card)

    def browse_video(self) -> None:
        """Open file dialog for MP4 / AVI files."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File for Processing",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv)",
        )
        if file_path:
            self.selected_video_path = Path(file_path)
            self.lbl_status.setText(f"Loaded: {self.selected_video_path.name}")
            self.btn_start_processing.setEnabled(True)
            self.progress_bar.setValue(0)
            self.lbl_summary_stats.setText("")

            # Display first frame
            cap = cv2.VideoCapture(str(self.selected_video_path))
            ret, frame = cap.read()
            if ret and frame is not None:
                self._render_frame(frame)
            cap.release()

    def start_processing(self) -> None:
        """Launch background video processing thread."""
        if not self.selected_video_path or not self.selected_video_path.exists():
            return

        out_name = f"annotated_{self.selected_video_path.stem}.mp4"
        out_path = get_videos_dir() / out_name

        self.processor = VideoProcessor(
            input_path=self.selected_video_path,
            output_path=out_path,
            pipeline=self.pipeline,
        )

        self.processor_thread = QThread(self)
        self.processor.moveToThread(self.processor_thread)

        self.processor_thread.started.connect(self.processor.process)
        self.processor.progress_updated.connect(self.on_progress)
        self.processor.frame_processed.connect(self.on_frame_preview)
        self.processor.status_message.connect(self.lbl_status.setText)
        self.processor.finished.connect(self.on_finished)
        self.processor.cancelled.connect(self.on_cancelled)
        self.processor.error_occurred.connect(self.on_error)

        self.processor_thread.start()

        self.btn_start_processing.setEnabled(False)
        self.btn_select_video.setEnabled(False)
        self.btn_cancel_processing.setEnabled(True)

    def cancel_processing(self) -> None:
        """Cancel active video encoding."""
        if self.processor:
            self.processor.cancel()
        self.btn_cancel_processing.setEnabled(False)

    @Slot(int, int, float)
    def on_progress(self, current: int, total: int, pct: float) -> None:
        self.progress_bar.setValue(int(pct))
        self.lbl_frames_info.setText(f"Frame: {current} / {total} ({pct:.1f}%)")

    @Slot(np.ndarray)
    def on_frame_preview(self, frame_bgr: np.ndarray) -> None:
        self._render_frame(frame_bgr)

    def _render_frame(self, frame_bgr: np.ndarray) -> None:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        q_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img).scaled(
            self.preview_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.preview_label.setPixmap(pixmap)

    @Slot(str, dict)
    def on_finished(self, out_path: str, summary: Dict[str, Any]) -> None:
        self._cleanup_thread()
        self.lbl_status.setText(f"Video Processing Complete! Exported to: {Path(out_path).name}")
        self.lbl_summary_stats.setText(
            f"Processed {summary['total_frames_processed']} frames in {summary['elapsed_seconds']}s "
            f"({summary['processing_fps']} FPS) | Unique Faces: {summary['total_unique_faces']} "
            f"| Masks: {summary['mask_detections']} | Violations: {summary['nomask_detections']}"
        )
        self.btn_start_processing.setEnabled(True)
        self.btn_select_video.setEnabled(True)
        self.btn_cancel_processing.setEnabled(False)

    @Slot()
    def on_cancelled(self) -> None:
        self._cleanup_thread()
        self.lbl_status.setText("Processing cancelled by user.")
        self.btn_start_processing.setEnabled(True)
        self.btn_select_video.setEnabled(True)
        self.btn_cancel_processing.setEnabled(False)

    @Slot(str)
    def on_error(self, err_msg: str) -> None:
        self._cleanup_thread()
        self.lbl_status.setText(f"Error: {err_msg}")
        self.btn_start_processing.setEnabled(True)
        self.btn_select_video.setEnabled(True)
        self.btn_cancel_processing.setEnabled(False)

    def _cleanup_thread(self) -> None:
        if self.processor_thread and self.processor_thread.isRunning():
            self.processor_thread.quit()
            self.processor_thread.wait(1000)
        self.processor = None
