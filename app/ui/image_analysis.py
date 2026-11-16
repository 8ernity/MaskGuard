"""
Static image analysis view.
Supports file browsing, drag-and-drop, multi-face detection, bounding box annotation, and image export.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
import numpy as np
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QImage, QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.core.inference_pipeline import InferencePipeline
from app.core.mask_classifier import FaceDetection
from app.utils.config import AppConfig
from app.utils.image_utils import read_image_bgr, save_image_bgr
from app.utils.resource_paths import get_results_dir


class ImageAnalysisView(QWidget):
    """
    Image upload, multi-face detection, and analysis view.
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
        self.current_annotated_bgr: np.ndarray | None = None
        self.current_image_path: Path | None = None

        self.setAcceptDrops(True)
        self._setup_ui()

    def _setup_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(24, 20, 24, 20)
        main_layout.setSpacing(16)

        # Header Title and Actions
        header_layout = QHBoxLayout()

        title_box = QVBoxLayout()
        title = QLabel("Static Image Analysis")
        title.setStyleSheet("font-size: 22px; font-weight: 700; color: #F0F6FC;")
        desc = QLabel("Analyze photographs for face mask compliance. Drag & drop or select an image file.")
        desc.setStyleSheet("font-size: 13px; color: #8B949E;")
        title_box.addWidget(title)
        title_box.addWidget(desc)
        header_layout.addLayout(title_box)
        header_layout.addStretch()

        self.btn_browse = QPushButton("Select Image")
        self.btn_browse.setProperty("class", "PrimaryButton")
        self.btn_browse.setCursor(Qt.PointingHandCursor)
        self.btn_browse.clicked.connect(self.browse_image)
        header_layout.addWidget(self.btn_browse)

        self.btn_save_image = QPushButton("Save Annotated Image")
        self.btn_save_image.setProperty("class", "SecondaryButton")
        self.btn_save_image.setCursor(Qt.PointingHandCursor)
        self.btn_save_image.setEnabled(False)
        self.btn_save_image.clicked.connect(self.save_annotated_image)
        header_layout.addWidget(self.btn_save_image)

        main_layout.addLayout(header_layout)

        # Content Split: Left Image Preview, Right Detections Table
        content_layout = QHBoxLayout()
        content_layout.setSpacing(16)

        # Image Viewport
        self.image_viewport = QLabel("Drag & drop an image here or click 'Select Image'")
        self.image_viewport.setObjectName("VideoViewport")
        self.image_viewport.setAlignment(Qt.AlignCenter)
        self.image_viewport.setMinimumSize(480, 400)
        self.image_viewport.setStyleSheet(
            "background-color: #010409; border: 2px dashed #30363D; border-radius: 8px; color: #8B949E; font-size: 14px;"
        )
        content_layout.addWidget(self.image_viewport, 3)

        # Right Side: Detections Table Card
        table_card = QFrame()
        table_card.setProperty("class", "Card")
        tc_layout = QVBoxLayout(table_card)
        tc_layout.setContentsMargins(14, 14, 14, 14)
        tc_layout.setSpacing(10)

        tc_title = QLabel("Detected Faces Breakdown")
        tc_title.setProperty("class", "SectionHeader")
        tc_layout.addWidget(tc_title)

        self.summary_label = QLabel("No image loaded.")
        self.summary_label.setStyleSheet("color: #8B949E; font-size: 12px;")
        tc_layout.addWidget(self.summary_label)

        self.results_table = QTableWidget(0, 4)
        self.results_table.setHorizontalHeaderLabels(["Face #", "Box (X, Y, W, H)", "Status", "Confidence"])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.results_table.setAlternatingRowColors(True)
        self.results_table.setEditTriggers(QTableWidget.NoEditTriggers)
        tc_layout.addWidget(self.results_table)

        content_layout.addWidget(table_card, 2)
        main_layout.addLayout(content_layout, 1)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        urls = event.mimeData().urls()
        if urls:
            file_path = Path(urls[0].toLocalFile())
            if file_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                self.process_image_file(file_path)

    def browse_image(self) -> None:
        """Open file dialog for image selection."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Photo for Mask Detection",
            "",
            "Images (*.jpg *.jpeg *.png *.bmp *.webp)",
        )
        if file_path:
            self.process_image_file(Path(file_path))

    def process_image_file(self, file_path: Path) -> None:
        """Read image, run inference pipeline, display annotated image and table."""
        img_bgr = read_image_bgr(file_path)
        if img_bgr is None:
            self.image_viewport.setText(f"Failed to read image: {file_path.name}")
            return

        self.current_image_path = file_path

        # Run inference
        annotated_bgr, detections, stats = self.pipeline.process_frame(img_bgr, track_session=False)
        self.current_annotated_bgr = annotated_bgr

        # Render annotated image
        rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        q_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img).scaled(
            self.image_viewport.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.image_viewport.setPixmap(pixmap)
        self.image_viewport.setStyleSheet(
            "background-color: #010409; border: 1px solid #30363D; border-radius: 8px;"
        )

        # Update Table and Summary
        self.btn_save_image.setEnabled(True)
        self._populate_results_table(detections)

    def _populate_results_table(self, detections: List[FaceDetection]) -> None:
        self.results_table.setRowCount(0)

        if not detections:
            self.summary_label.setText("No human faces detected in the image.")
            return

        mask_count = sum(1 for d in detections if d.label == "MASK" and not d.is_uncertain)
        nomask_count = sum(1 for d in detections if d.label == "NO MASK" and not d.is_uncertain)
        uncertain_count = sum(1 for d in detections if d.is_uncertain)

        self.summary_label.setText(
            f"Detected {len(detections)} face(s): {mask_count} Masked, {nomask_count} Unmasked, {uncertain_count} Uncertain"
        )

        for i, det in enumerate(detections):
            row = self.results_table.rowCount()
            self.results_table.insertRow(row)

            # Face #
            item_num = QTableWidgetItem(f"Face {i + 1}")
            item_num.setTextAlignment(Qt.AlignCenter)
            self.results_table.setItem(row, 0, item_num)

            # Coordinates
            x, y, w, h = det.box
            item_box = QTableWidgetItem(f"({x}, {y}, {w}, {h})")
            item_box.setTextAlignment(Qt.AlignCenter)
            self.results_table.setItem(row, 1, item_box)

            # Status Badge
            item_status = QTableWidgetItem(det.label)
            item_status.setTextAlignment(Qt.AlignCenter)
            if det.is_uncertain:
                item_status.setForeground(Qt.yellow)
            elif det.label == "MASK":
                item_status.setForeground(Qt.green)
            else:
                item_status.setForeground(Qt.red)
            self.results_table.setItem(row, 2, item_status)

            # Confidence
            item_conf = QTableWidgetItem(f"{det.confidence * 100:.1f}%")
            item_conf.setTextAlignment(Qt.AlignCenter)
            self.results_table.setItem(row, 3, item_conf)

    def save_annotated_image(self) -> None:
        """Export current annotated image."""
        if self.current_annotated_bgr is None:
            return

        default_name = f"annotated_{self.current_image_path.stem}.jpg" if self.current_image_path else "annotated_detection.jpg"
        save_dest, _ = QFileDialog.getSaveFileName(
            self,
            "Save Annotated Detection",
            str(get_results_dir() / default_name),
            "JPEG Image (*.jpg);;PNG Image (*.png)",
        )
        if save_dest:
            save_image_bgr(save_dest, self.current_annotated_bgr)
