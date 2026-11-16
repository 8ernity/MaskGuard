"""
Model Analytics view for MaskGuard.
Presents test accuracy, precision, recall, F1, loss/accuracy curves, confusion matrix, and layer specifications.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QHeaderView,
)

from app.utils.config import AppConfig
from app.utils.resource_paths import (
    get_dataset_dir,
    get_default_model_path,
    get_model_metadata_path,
    get_results_dir,
)
from training.evaluate import evaluate_model
from training.train import train_model


class EvaluationWorker(QThread):
    finished_eval = Signal(dict)
    error_eval = Signal(str)

    def __init__(self, model_path: str, dataset_dir: str, results_dir: str):
        super().__init__()
        self.model_path = model_path
        self.dataset_dir = dataset_dir
        self.results_dir = results_dir

    def run(self) -> None:
        try:
            metrics = evaluate_model(self.model_path, self.dataset_dir, self.results_dir)
            self.finished_eval.emit(metrics)
        except Exception as e:
            self.error_eval.emit(str(e))


class TrainingWorker(QThread):
    finished_train = Signal(dict)
    error_train = Signal(str)

    def __init__(self, dataset_dir: str, output_model: str):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.output_model = output_model

    def run(self) -> None:
        try:
            summary = train_model(dataset_dir=self.dataset_dir, output_model_path=self.output_model, epochs=20)
            self.finished_train.emit(summary)
        except Exception as e:
            self.error_train.emit(str(e))


class ModelAnalyticsView(QWidget):
    """
    Comprehensive deep learning telemetry and evaluation metrics view.
    """

    def __init__(self, config: AppConfig, parent: QWidget | None = None):
        super().__init__(parent)
        self.config = config
        self._setup_ui()
        self.reload_metrics()

    def _setup_ui(self) -> None:
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(20)

        # Header Title and Action Buttons
        header_layout = QHBoxLayout()

        title_box = QVBoxLayout()
        title = QLabel("Model Evaluation & Analytics")
        title.setStyleSheet("font-size: 22px; font-weight: 700; color: #F0F6FC;")
        desc = QLabel("Detailed test metrics, confusion matrix, loss/accuracy curves, and neural architecture.")
        desc.setStyleSheet("font-size: 13px; color: #8B949E;")
        title_box.addWidget(title)
        title_box.addWidget(desc)
        header_layout.addLayout(title_box)
        header_layout.addStretch()

        self.btn_run_eval = QPushButton("Run Test Evaluation")
        self.btn_run_eval.setProperty("class", "SecondaryButton")
        self.btn_run_eval.setCursor(Qt.PointingHandCursor)
        self.btn_run_eval.clicked.connect(self.run_evaluation)
        header_layout.addWidget(self.btn_run_eval)

        self.btn_train_model = QPushButton("Train / Retrain Model")
        self.btn_train_model.setProperty("class", "PrimaryButton")
        self.btn_train_model.setCursor(Qt.PointingHandCursor)
        self.btn_train_model.clicked.connect(self.trigger_training)
        header_layout.addWidget(self.btn_train_model)

        layout.addLayout(header_layout)

        # Key Metrics Strip (Accuracy, Precision, Recall, F1)
        metrics_grid = QGridLayout()
        metrics_grid.setSpacing(14)

        self.card_acc = self._create_metric_card("TEST ACCURACY", "--", "#10B981")
        self.card_prec = self._create_metric_card("PRECISION", "--", "#38BDF8")
        self.card_rec = self._create_metric_card("RECALL", "--", "#F59E0B")
        self.card_f1 = self._create_metric_card("F1-SCORE", "--", "#A855F7")

        metrics_grid.addWidget(self.card_acc["frame"], 0, 0)
        metrics_grid.addWidget(self.card_prec["frame"], 0, 1)
        metrics_grid.addWidget(self.card_rec["frame"], 0, 2)
        metrics_grid.addWidget(self.card_f1["frame"], 0, 3)
        layout.addLayout(metrics_grid)

        # Visual Analytics: Training Curves & Confusion Matrix Plots
        plots_layout = QHBoxLayout()
        plots_layout.setSpacing(16)

        # Training Curves Card
        curves_card = QFrame()
        curves_card.setProperty("class", "Card")
        cc_layout = QVBoxLayout(curves_card)
        cc_layout.setContentsMargins(14, 14, 14, 14)
        cc_title = QLabel("Learning Dynamics (Loss & Accuracy)")
        cc_title.setProperty("class", "SectionHeader")
        cc_layout.addWidget(cc_title)

        self.curves_image_label = QLabel("No training history plot available. Train model to generate.")
        self.curves_image_label.setAlignment(Qt.AlignCenter)
        self.curves_image_label.setMinimumHeight(240)
        self.curves_image_label.setStyleSheet("color: #8B949E; background-color: #0D1117; border-radius: 6px;")
        cc_layout.addWidget(self.curves_image_label)
        plots_layout.addWidget(curves_card, 3)

        # Confusion Matrix Card
        cm_card = QFrame()
        cm_card.setProperty("class", "Card")
        cm_layout = QVBoxLayout(cm_card)
        cm_layout.setContentsMargins(14, 14, 14, 14)
        cm_title = QLabel("Confusion Matrix (Held-out Test)")
        cm_title.setProperty("class", "SectionHeader")
        cm_layout.addWidget(cm_title)

        self.cm_image_label = QLabel("No confusion matrix plot available. Run evaluation to generate.")
        self.cm_image_label.setAlignment(Qt.AlignCenter)
        self.cm_image_label.setMinimumHeight(240)
        self.cm_image_label.setStyleSheet("color: #8B949E; background-color: #0D1117; border-radius: 6px;")
        cm_layout.addWidget(self.cm_image_label)
        plots_layout.addWidget(cm_card, 2)

        layout.addLayout(plots_layout)

        # Layer-by-Layer Architecture Table Card
        arch_card = QFrame()
        arch_card.setProperty("class", "Card")
        ac_layout = QVBoxLayout(arch_card)
        ac_layout.setContentsMargins(14, 14, 14, 14)

        ac_title = QLabel("Custom CNN Layer Specification")
        ac_title.setProperty("class", "SectionHeader")
        ac_layout.addWidget(ac_title)

        self.arch_table = QTableWidget(7, 4)
        self.arch_table.setHorizontalHeaderLabels(["Layer Name", "Layer Type", "Output Shape", "Key Hyperparameters"])
        self.arch_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.arch_table.setEditTriggers(QTableWidget.NoEditTriggers)

        layers_info = [
            ("input_image", "InputLayer", "(None, 128, 128, 3)", "Input image normalized [0, 1]"),
            ("conv1 + bn1 + pool1", "Conv2D + BN + MaxPool", "(None, 64, 64, 32)", "32 filters, 3x3 kernel, ReLU, 2x2 pool"),
            ("conv2 + bn2 + pool2", "Conv2D + BN + MaxPool", "(None, 32, 32, 64)", "64 filters, 3x3 kernel, ReLU, 2x2 pool"),
            ("conv3 + bn3 + pool3", "Conv2D + BN + MaxPool", "(None, 16, 16, 128)", "128 filters, 3x3 kernel, ReLU, 2x2 pool"),
            ("gap", "GlobalAveragePooling2D", "(None, 128)", "Spatial averaging over feature maps"),
            ("dense128 + dropout", "Dense + Dropout", "(None, 128)", "128 units, ReLU, dropout rate 0.50"),
            ("output_classification", "Dense (Sigmoid)", "(None, 1)", "Binary output: 0=with_mask, 1=without_mask"),
        ]

        for row, (name, ltype, shape, desc_text) in enumerate(layers_info):
            self.arch_table.setItem(row, 0, QTableWidgetItem(name))
            self.arch_table.setItem(row, 1, QTableWidgetItem(ltype))
            self.arch_table.setItem(row, 2, QTableWidgetItem(shape))
            self.arch_table.setItem(row, 3, QTableWidgetItem(desc_text))

        ac_layout.addWidget(self.arch_table)
        layout.addWidget(arch_card)

        # Status output line
        self.lbl_task_status = QLabel("")
        self.lbl_task_status.setStyleSheet("color: #10B981; font-weight: 600; font-size: 13px;")
        layout.addWidget(self.lbl_task_status)

        layout.addStretch()

        self.container = container
        scroll.setWidget(container)

        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.addWidget(scroll)

    def _create_metric_card(self, title: str, default_val: str, color: str) -> dict:
        card = QFrame()
        card.setProperty("class", "StatCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 14, 16, 14)

        t = QLabel(title)
        t.setProperty("class", "CardTitle")

        v = QLabel(default_val)
        v.setStyleSheet(f"font-size: 26px; font-weight: 700; color: {color}; margin: 4px 0px;")

        layout.addWidget(t)
        layout.addWidget(v)
        return {"frame": card, "val_label": v}

    def reload_metrics(self) -> None:
        """Read saved evaluation results from disk and display."""
        metrics_file = get_results_dir() / "evaluation_metrics.json"
        if metrics_file.exists():
            try:
                with open(metrics_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                acc = data.get("test_accuracy", 0.0)
                prec = data.get("precision", 0.0)
                rec = data.get("recall", 0.0)
                f1 = data.get("f1_score", 0.0)

                self.card_acc["val_label"].setText(f"{acc * 100:.1f}%")
                self.card_prec["val_label"].setText(f"{prec * 100:.1f}%")
                self.card_rec["val_label"].setText(f"{rec * 100:.1f}%")
                self.card_f1["val_label"].setText(f"{f1 * 100:.1f}%")
            except Exception:
                pass

        # Load curves plot if present
        curves_path = get_results_dir() / "training_curves.png"
        if curves_path.exists():
            pix = QPixmap(str(curves_path))
            w = max(480, self.curves_image_label.width())
            h = max(240, self.curves_image_label.height())
            scaled = pix.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.curves_image_label.setPixmap(scaled)

        # Load CM plot if present
        cm_path = get_results_dir() / "confusion_matrix.png"
        if cm_path.exists():
            pix = QPixmap(str(cm_path))
            w = max(320, self.cm_image_label.width())
            h = max(240, self.cm_image_label.height())
            scaled = pix.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.cm_image_label.setPixmap(scaled)

    def run_evaluation(self) -> None:
        """Trigger asynchronous model evaluation on test set."""
        self.btn_run_eval.setEnabled(False)
        self.lbl_task_status.setText("Evaluating model on held-out test split...")

        self.eval_worker = EvaluationWorker(
            model_path=str(self.config.model_path),
            dataset_dir=str(get_dataset_dir()),
            results_dir=str(get_results_dir()),
        )
        self.eval_worker.finished_eval.connect(self._on_eval_finished)
        self.eval_worker.error_eval.connect(self._on_eval_error)
        self.eval_worker.start()

    @Slot(dict)
    def _on_eval_finished(self, metrics: dict) -> None:
        self.btn_run_eval.setEnabled(True)
        self.lbl_task_status.setText("Evaluation completed successfully.")
        self.reload_metrics()

    @Slot(str)
    def _on_eval_error(self, err: str) -> None:
        self.btn_run_eval.setEnabled(True)
        self.lbl_task_status.setText(f"Evaluation error: {err}")

    def trigger_training(self) -> None:
        """Trigger background training session."""
        self.btn_train_model.setEnabled(False)
        self.lbl_task_status.setText("Training custom CNN model in background...")

        self.train_worker = TrainingWorker(
            dataset_dir=str(get_dataset_dir()),
            output_model=str(self.config.model_path),
        )
        self.train_worker.finished_train.connect(self._on_train_finished)
        self.train_worker.error_train.connect(self._on_train_error)
        self.train_worker.start()

    @Slot(dict)
    def _on_train_finished(self, summary: dict) -> None:
        self.btn_train_model.setEnabled(True)
        self.lbl_task_status.setText(f"Model training complete! Saved to {Path(summary['model_path']).name}")
        self.run_evaluation()

    @Slot(str)
    def _on_train_error(self, err: str) -> None:
        self.btn_train_model.setEnabled(True)
        self.lbl_task_status.setText(f"Training error: {err}")
