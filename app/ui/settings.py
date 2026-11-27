"""
Application settings and hardware configuration view.
Controls camera selection, hyperparameters, model paths, theme, and persistence.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.core.camera_worker import CameraWorker
from app.utils.config import AppConfig


class SettingsView(QWidget):
    """
    Settings configuration view.
    """

    theme_changed = Signal(str)        # "dark" | "light"
    config_saved = Signal(AppConfig)

    def __init__(self, config: AppConfig, parent: QWidget | None = None):
        super().__init__(parent)
        self.config = config
        self._setup_ui()
        self.load_from_config()

    def _setup_ui(self) -> None:
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(20)

        # Header Title and Save Button
        header_layout = QHBoxLayout()

        title_box = QVBoxLayout()
        title = QLabel("System Settings & Preferences")
        title.setStyleSheet("font-size: 22px; font-weight: 700; color: #F0F6FC;")
        desc = QLabel("Configure camera interfaces, detection thresholds, display overlays, and file paths.")
        desc.setStyleSheet("font-size: 13px; color: #8B949E;")
        title_box.addWidget(title)
        title_box.addWidget(desc)
        header_layout.addLayout(title_box)
        header_layout.addStretch()

        self.btn_reset_defaults = QPushButton("Reset Defaults")
        self.btn_reset_defaults.setProperty("class", "SecondaryButton")
        self.btn_reset_defaults.setCursor(Qt.PointingHandCursor)
        self.btn_reset_defaults.clicked.connect(self.reset_defaults)
        header_layout.addWidget(self.btn_reset_defaults)

        self.btn_save_settings = QPushButton("Save Settings")
        self.btn_save_settings.setProperty("class", "PrimaryButton")
        self.btn_save_settings.setCursor(Qt.PointingHandCursor)
        self.btn_save_settings.clicked.connect(self.save_settings)
        header_layout.addWidget(self.btn_save_settings)

        layout.addLayout(header_layout)

        # 1. Appearance & Theme Card
        theme_card = QFrame()
        theme_card.setProperty("class", "Card")
        tc_layout = QVBoxLayout(theme_card)
        tc_title = QLabel("Appearance & Interface Theme")
        tc_title.setProperty("class", "SectionHeader")
        tc_layout.addWidget(tc_title)

        theme_row = QHBoxLayout()
        theme_row.addWidget(QLabel("Application Theme:"))
        self.combo_theme = QComboBox()
        self.combo_theme.addItem("Dark Charcoal & Emerald (Default)", "dark")
        self.combo_theme.addItem("Light Mode", "light")
        self.combo_theme.currentIndexChanged.connect(self._on_theme_selection_changed)
        theme_row.addWidget(self.combo_theme)
        theme_row.addStretch()
        tc_layout.addLayout(theme_row)
        layout.addWidget(theme_card)

        # 2. Camera Sensor & Device Card
        cam_card = QFrame()
        cam_card.setProperty("class", "Card")
        cc_layout = QVBoxLayout(cam_card)
        cc_title = QLabel("Optical Camera Hardware")
        cc_title.setProperty("class", "SectionHeader")
        cc_layout.addWidget(cc_title)

        cam_row = QHBoxLayout()
        cam_row.addWidget(QLabel("Primary Video Capture Device:"))
        self.combo_cam = QComboBox()
        self._refresh_camera_list()
        cam_row.addWidget(self.combo_cam)

        self.btn_refresh_cams = QPushButton("Rescan Hardware")
        self.btn_refresh_cams.setProperty("class", "SecondaryButton")
        self.btn_refresh_cams.setCursor(Qt.PointingHandCursor)
        self.btn_refresh_cams.clicked.connect(self._refresh_camera_list)
        cam_row.addWidget(self.btn_refresh_cams)
        cam_row.addStretch()
        cc_layout.addLayout(cam_row)
        layout.addWidget(cam_card)

        # 3. Detection Hyperparameters Card
        params_card = QFrame()
        params_card.setProperty("class", "Card")
        pc_layout = QVBoxLayout(params_card)
        pc_title = QLabel("Inference & Detection Hyperparameters")
        pc_title.setProperty("class", "SectionHeader")
        pc_layout.addWidget(pc_title)

        grid = QGridLayout()
        grid.setSpacing(14)

        # Confidence Threshold Slider
        grid.addWidget(QLabel("Classification Confidence Threshold:"), 0, 0)
        self.slider_conf = QSlider(Qt.Horizontal)
        self.slider_conf.setRange(20, 95)
        self.slider_conf.setValue(65)
        self.lbl_conf_val = QLabel("65%")
        self.lbl_conf_val.setStyleSheet("font-weight: 700; color: #10B981; min-width: 40px;")
        self.slider_conf.valueChanged.connect(lambda v: self.lbl_conf_val.setText(f"{v}%"))
        grid.addWidget(self.slider_conf, 0, 1)
        grid.addWidget(self.lbl_conf_val, 0, 2)

        # Face Padding Slider
        grid.addWidget(QLabel("Face Crop Padding / Margin:"), 1, 0)
        self.slider_pad = QSlider(Qt.Horizontal)
        self.slider_pad.setRange(0, 30)
        self.slider_pad.setValue(12)
        self.lbl_pad_val = QLabel("12%")
        self.lbl_pad_val.setStyleSheet("font-weight: 700; color: #10B981; min-width: 40px;")
        self.slider_pad.valueChanged.connect(lambda v: self.lbl_pad_val.setText(f"{v}%"))
        grid.addWidget(self.slider_pad, 1, 1)
        grid.addWidget(self.lbl_pad_val, 1, 2)

        # Face scale factor
        grid.addWidget(QLabel("Haar Scale Factor:"), 2, 0)
        self.spin_scale = QComboBox()
        for sf in ["1.05", "1.10", "1.15", "1.20", "1.25", "1.30"]:
            self.spin_scale.addItem(sf, float(sf))
        grid.addWidget(self.spin_scale, 2, 1)

        # Min Neighbors
        grid.addWidget(QLabel("Haar Min Neighbors:"), 3, 0)
        self.spin_neighbors = QSpinBox()
        self.spin_neighbors.setRange(2, 10)
        self.spin_neighbors.setValue(5)
        grid.addWidget(self.spin_neighbors, 3, 1)

        pc_layout.addLayout(grid)
        layout.addWidget(params_card)

        # 4. Display Overlays Card
        disp_card = QFrame()
        disp_card.setProperty("class", "Card")
        dc_layout = QVBoxLayout(disp_card)
        dc_title = QLabel("Video & Image HUD Overlays")
        dc_title.setProperty("class", "SectionHeader")
        dc_layout.addWidget(dc_title)

        dgrid = QGridLayout()
        self.chk_boxes = QCheckBox("Draw Bounding Boxes on Faces")
        self.chk_labels = QCheckBox("Display Classification Labels (MASK / NO MASK / UNCERTAIN)")
        self.chk_conf = QCheckBox("Display Confidence Percentage Badge")
        self.chk_fps = QCheckBox("Show Real-Time FPS Counter HUD")

        dgrid.addWidget(self.chk_boxes, 0, 0)
        dgrid.addWidget(self.chk_labels, 0, 1)
        dgrid.addWidget(self.chk_conf, 1, 0)
        dgrid.addWidget(self.chk_fps, 1, 1)
        dc_layout.addLayout(dgrid)
        layout.addWidget(disp_card)

        # 5. Paths & Models Card
        path_card = QFrame()
        path_card.setProperty("class", "Card")
        pfc_layout = QVBoxLayout(path_card)
        pfc_title = QLabel("Storage Directories & Model Artifacts")
        pfc_title.setProperty("class", "SectionHeader")
        pfc_layout.addWidget(pfc_title)

        pgrid = QGridLayout()
        pgrid.setSpacing(10)

        # Model file path
        pgrid.addWidget(QLabel("Trained Model (.h5):"), 0, 0)
        self.txt_model_path = QLineEdit()
        pgrid.addWidget(self.txt_model_path, 0, 1)
        self.btn_browse_model = QPushButton("Browse")
        self.btn_browse_model.setProperty("class", "SecondaryButton")
        self.btn_browse_model.clicked.connect(self._browse_model)
        pgrid.addWidget(self.btn_browse_model, 0, 2)

        # Output directory
        pgrid.addWidget(QLabel("Export & Results Directory:"), 1, 0)
        self.txt_output_dir = QLineEdit()
        pgrid.addWidget(self.txt_output_dir, 1, 1)
        self.btn_browse_output = QPushButton("Browse")
        self.btn_browse_output.setProperty("class", "SecondaryButton")
        self.btn_browse_output.clicked.connect(self._browse_output)
        pgrid.addWidget(self.btn_browse_output, 1, 2)

        pfc_layout.addLayout(pgrid)
        layout.addWidget(path_card)

        # Status Label
        self.lbl_status = QLabel("")
        self.lbl_status.setStyleSheet("color: #10B981; font-weight: 600; font-size: 13px;")
        layout.addWidget(self.lbl_status)

        layout.addStretch()

        self.container = container
        scroll.setWidget(container)

        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.addWidget(scroll)

    def _refresh_camera_list(self) -> None:
        self.combo_cam.clear()
        cams = CameraWorker.get_available_cameras(max_to_test=3)
        for idx in cams:
            self.combo_cam.addItem(f"Camera Device {idx}", idx)

    def _on_theme_selection_changed(self) -> None:
        theme = self.combo_theme.currentData()
        if theme:
            self.theme_changed.emit(theme)

    def _browse_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Trained Keras Model",
            self.txt_model_path.text(),
            "Keras Models (*.h5 *.keras)",
        )
        if path:
            self.txt_model_path.setText(path)

    def _browse_output(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            self.txt_output_dir.text(),
        )
        if folder:
            self.txt_output_dir.setText(folder)

    def load_from_config(self) -> None:
        """Load values from AppConfig into widgets."""
        # Theme
        idx = self.combo_theme.findData(self.config.theme)
        if idx >= 0:
            self.combo_theme.setCurrentIndex(idx)

        # Camera
        c_idx = self.combo_cam.findData(self.config.camera_index)
        if c_idx >= 0:
            self.combo_cam.setCurrentIndex(c_idx)

        # Confidence
        conf_int = int(self.config.confidence_threshold * 100)
        self.slider_conf.setValue(conf_int)
        self.lbl_conf_val.setText(f"{conf_int}%")

        # Padding
        pad_int = int(self.config.face_padding * 100)
        self.slider_pad.setValue(pad_int)
        self.lbl_pad_val.setText(f"{pad_int}%")

        # Scale Factor
        sf_str = f"{self.config.face_scale_factor:.2f}"
        s_idx = self.spin_scale.findText(sf_str)
        if s_idx >= 0:
            self.spin_scale.setCurrentIndex(s_idx)

        # Min Neighbors
        self.spin_neighbors.setValue(self.config.face_min_neighbors)

        # Checks
        self.chk_boxes.setChecked(self.config.show_boxes)
        self.chk_labels.setChecked(self.config.show_labels)
        self.chk_conf.setChecked(self.config.show_confidence)
        self.chk_fps.setChecked(self.config.show_fps)

        # Paths
        self.txt_model_path.setText(str(self.config.model_path))
        self.txt_output_dir.setText(str(self.config.output_dir))

    def save_settings(self) -> None:
        """Write form values back to AppConfig and persist to disk."""
        self.config.theme = self.combo_theme.currentData() or "dark"
        self.config.camera_index = self.combo_cam.currentData() or 0
        self.config.confidence_threshold = self.slider_conf.value() / 100.0
        self.config.face_padding = self.slider_pad.value() / 100.0
        self.config.face_scale_factor = float(self.spin_scale.currentText())
        self.config.face_min_neighbors = self.spin_neighbors.value()
        self.config.show_boxes = self.chk_boxes.isChecked()
        self.config.show_labels = self.chk_labels.isChecked()
        self.config.show_confidence = self.chk_conf.isChecked()
        self.config.show_fps = self.chk_fps.isChecked()
        self.config.model_path = self.txt_model_path.text()
        self.config.output_dir = self.txt_output_dir.text()

        self.config.save()
        self.config_saved.emit(self.config)
        self.lbl_status.setText("Preferences saved successfully.")

    def reset_defaults(self) -> None:
        """Reset config to default factory settings."""
        self.config = AppConfig()
        self.load_from_config()
        self.save_settings()
        self.lbl_status.setText("Reset to default settings.")
