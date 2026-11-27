"""
Dual AI Model Manager for MaskGuard.
Handles registration, lazy loading, metadata validation, caching, thread-safe routing,
and analytics lookup for Custom CNN (`custom_cnn`) and MobileNetV2 (`mobilenetv2`).
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np

from app.core.mask_classifier import CustomCNNClassifier, FaceDetection
from app.core.mobilenet_classifier import MobileNetV2Classifier
from app.utils.resource_paths import (
    get_default_model_path,
    get_mobilenetv2_dir,
    get_mobilenetv2_evaluation_path,
    get_mobilenetv2_metadata_path,
    get_mobilenetv2_model_path,
    get_model_metadata_path,
    get_results_dir,
)


class ModelSelectionError(Exception):
    """Raised when an unknown, unavailable, or failed model is requested."""


class DualModelManager:
    """
    Central coordinator managing independent classifiers (`custom_cnn`, `mobilenetv2`).
    - Default selection is `custom_cnn`.
    - Lazy-loads `mobilenetv2` only when selected or explicitly queried.
    - Never silently falls back to another model on failure.
    """

    available_models: Dict[str, Type[Any]] = {
        "custom_cnn": CustomCNNClassifier,
        "mobilenetv2": MobileNetV2Classifier,
    }

    MODEL_DESCRIPTIONS: Dict[str, Dict[str, Any]] = {
        "custom_cnn": {
            "model_id": "custom_cnn",
            "display_name": "Custom CNN",
            "subtitle": "Original MaskGuard Model",
            "input_dimensions": [128, 128, 3],
            "preprocessing": "RGB 128x128 normalized to [0.0, 1.0]",
        },
        "mobilenetv2": {
            "model_id": "mobilenetv2",
            "display_name": "MobileNetV2",
            "subtitle": "Transfer Learning Model",
            "input_dimensions": [224, 224, 3],
            "preprocessing": "RGB 224x224 MobileNetV2 preprocess_input [-1.0, 1.0]",
        },
    }

    def __init__(
        self,
        active_model_id: str = "custom_cnn",
        confidence_threshold: float = 0.65,
        uncertain_band: float = 0.15,
        custom_cnn_path: Path | str | None = None,
        mobilenetv2_path: Path | str | None = None,
        lazy_load_mobilenet: bool = True,
    ):
        self._lock = threading.RLock()
        self.confidence_threshold = confidence_threshold
        self.uncertain_band = uncertain_band
        self._custom_cnn_path = Path(custom_cnn_path) if custom_cnn_path else get_default_model_path()
        self._mobilenetv2_path = Path(mobilenetv2_path) if mobilenetv2_path else get_mobilenetv2_model_path()
        self._instances: Dict[str, Any] = {}

        # Always initialize Custom CNN first so primary baseline works independently
        self._get_or_init_instance("custom_cnn", auto_load=True)

        # Resolve initial active model safely
        requested = active_model_id if active_model_id in self.available_models else "custom_cnn"
        if requested == "mobilenetv2":
            if self._artifact_exists("mobilenetv2") and not lazy_load_mobilenet:
                ok = self._get_or_init_instance("mobilenetv2", auto_load=True).is_ready()
                self.active_model_id = "mobilenetv2" if ok else "custom_cnn"
            elif self._artifact_exists("mobilenetv2"):
                # Even if persisted as mobilenetv2, load if requested cleanly; fall back to custom_cnn on startup only if artifact missing/broken
                inst = self._get_or_init_instance("mobilenetv2", auto_load=True)
                self.active_model_id = "mobilenetv2" if inst.is_ready() else "custom_cnn"
            else:
                self.active_model_id = "custom_cnn"
        else:
            self.active_model_id = "custom_cnn"

    def _artifact_path(self, model_id: str) -> Path:
        if model_id == "custom_cnn":
            return self._custom_cnn_path
        if model_id == "mobilenetv2":
            return self._mobilenetv2_path
        raise ModelSelectionError(f"Unknown model_id: '{model_id}'")

    def _artifact_exists(self, model_id: str) -> bool:
        p = self._artifact_path(model_id)
        if p.is_file():
            return True
        if model_id == "custom_cnn":
            return get_default_model_path().is_file()
        return False

    def _get_or_init_instance(self, model_id: str, auto_load: bool = True) -> Any:
        with self._lock:
            if model_id in self._instances:
                inst = self._instances[model_id]
                if auto_load and not inst.is_ready():
                    inst.load_model(self._artifact_path(model_id))
                return inst

            if model_id not in self.available_models:
                raise ModelSelectionError(f"Unknown model_id: '{model_id}'. Valid options: {list(self.available_models.keys())}")

            cls = self.available_models[model_id]
            if model_id == "custom_cnn":
                inst = cls(
                    model_path=self._custom_cnn_path,
                    confidence_threshold=self.confidence_threshold,
                    uncertain_band=self.uncertain_band,
                )
            else:
                inst = cls(
                    model_path=self._mobilenetv2_path,
                    confidence_threshold=self.confidence_threshold,
                    uncertain_band=self.uncertain_band,
                    auto_load=auto_load,
                )
            self._instances[model_id] = inst
            return inst

    def update_thresholds(self, confidence_threshold: float, uncertain_band: float) -> None:
        """Synchronize confidence and uncertainty thresholds across all cached models."""
        with self._lock:
            self.confidence_threshold = confidence_threshold
            self.uncertain_band = uncertain_band
            for inst in self._instances.values():
                inst.confidence_threshold = confidence_threshold
                inst.uncertain_band = uncertain_band

    def load_metadata(self, model_id: str) -> Dict[str, Any]:
        """Load persisted model metadata JSON if available."""
        meta_path = get_model_metadata_path() if model_id == "custom_cnn" else get_mobilenetv2_metadata_path()
        if meta_path.is_file():
            try:
                return json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def load_evaluation_metrics(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Load independent verified evaluation metrics JSON for the specified model."""
        if model_id == "custom_cnn":
            eval_path = get_results_dir() / "evaluation_metrics.json"
        elif model_id == "mobilenetv2":
            eval_path = get_mobilenetv2_evaluation_path()
        else:
            return None

        if eval_path.is_file():
            try:
                data = json.loads(eval_path.read_text(encoding="utf-8"))
                if isinstance(data, dict) and "accuracy" in data and "test_accuracy" not in data:
                    data["test_accuracy"] = data["accuracy"]
                return data
            except Exception:
                return None
        return None

    def get_model_analytics(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        """Return model-specific analytics, curves URL, confusion matrix URL, and metadata."""
        mid = model_id or self.active_model_id
        if mid not in self.available_models:
            raise ModelSelectionError(f"Unknown model_id: '{mid}'")

        metrics = self.load_evaluation_metrics(mid)
        metadata = self.load_metadata(mid)

        if mid == "custom_cnn":
            report_path = get_results_dir() / "classification_report.txt"
            cm_url = "/results_files/confusion_matrix.png"
            curves_url = "/results_files/training_curves.png"
            training_metrics = metadata.get("training_metrics", {})
        else:
            mn_dir = get_mobilenetv2_dir()
            report_path = mn_dir / "classification_report.txt"
            cm_url = "/api/results/mobilenetv2/confusion_matrix.png"
            curves_url = "/api/results/mobilenetv2/training_curves.png"
            training_metrics = metadata.get("dataset_information", {})

        report_text = ""
        if report_path.is_file():
            try:
                report_text = report_path.read_text(encoding="utf-8")
            except Exception:
                pass

        return {
            "model_id": mid,
            "display_name": self.MODEL_DESCRIPTIONS[mid]["display_name"],
            "status": "verified" if metrics is not None else "not_evaluated",
            "metrics": metrics,
            "metadata": metadata,
            "classification_report": report_text,
            "confusion_matrix_url": cm_url,
            "training_curves_url": curves_url,
            "dataset_summary": {
                "train_samples": training_metrics.get("train_samples", 9828),
                "val_samples": training_metrics.get("val_samples", 800),
                "test_samples": training_metrics.get("test_samples", 992),
                "classes": ["with_mask", "without_mask"],
            },
        }

    def list_models(self) -> List[Dict[str, Any]]:
        """Describe all registered models with availability, loading state, metadata, and metrics."""
        with self._lock:
            items: List[Dict[str, Any]] = []
            for mid, desc in self.MODEL_DESCRIPTIONS.items():
                inst = self._instances.get(mid)
                loaded = bool(inst is not None and inst.is_ready())
                available = self._artifact_exists(mid)
                meta = self.load_metadata(mid)
                eval_metrics = self.load_evaluation_metrics(mid)
                artifact_path = self._artifact_path(mid)
                size_mb = round(artifact_path.stat().st_size / (1024 * 1024), 2) if artifact_path.is_file() else None
                items.append({
                    **desc,
                    "available": available,
                    "loaded": loaded,
                    "active": (mid == self.active_model_id),
                    "model_file": artifact_path.name,
                    "model_size_mb": meta.get("model_size_bytes", 0) and round(meta["model_size_bytes"] / (1024 * 1024), 2) or size_mb,
                    "total_parameters": meta.get("total_parameters"),
                    "load_error": getattr(inst, "load_error", None) if inst else None,
                    "metrics": eval_metrics,
                })
            return items

    def select_model(self, model_id: str) -> Dict[str, Any]:
        """
        Validate, lazy-load, and activate `model_id`.
        Raises ModelSelectionError on invalid/missing/corrupted models without changing active_model_id.
        """
        with self._lock:
            if model_id not in self.available_models:
                raise ModelSelectionError(
                    f"Invalid model_id '{model_id}'. Supported models: {', '.join(self.available_models.keys())}."
                )

            if not self._artifact_exists(model_id):
                raise ModelSelectionError(
                    f"Model '{model_id}' is not available because its weight file was not found at {self._artifact_path(model_id)}."
                )

            inst = self._get_or_init_instance(model_id, auto_load=True)
            if not inst.is_ready():
                err = getattr(inst, "load_error", None) or f"Failed to load model '{model_id}'."
                raise ModelSelectionError(err)

            self.active_model_id = model_id
            return self.get_active_model_info()

    def get_active_model_info(self) -> Dict[str, Any]:
        """Return metadata and readiness for the currently active model."""
        with self._lock:
            for m in self.list_models():
                if m["model_id"] == self.active_model_id:
                    return m
            raise ModelSelectionError(f"Active model '{self.active_model_id}' is not registered.")

    def get_classifier(self, model_id: Optional[str] = None) -> Any:
        """
        Retrieve a ready classifier for `model_id` (or `self.active_model_id`).
        Never falls back silently to another model when `model_id` is explicitly requested.
        """
        with self._lock:
            mid = model_id or self.active_model_id
            if mid not in self.available_models:
                raise ModelSelectionError(f"Unknown model_id: '{mid}'")

            inst = self._get_or_init_instance(mid, auto_load=True)
            if not inst.is_ready():
                err = getattr(inst, "load_error", None) or f"Requested model '{mid}' is not loaded or available."
                raise ModelSelectionError(err)
            return inst

    def is_ready(self, model_id: Optional[str] = None) -> bool:
        """Check if specified model (or active model) is loaded and ready."""
        try:
            return self.get_classifier(model_id).is_ready()
        except Exception:
            return False

    def classify_crops(
        self,
        face_crops: List[np.ndarray],
        boxes: List[Tuple[int, int, int, int]],
        model_id: Optional[str] = None,
    ) -> List[FaceDetection]:
        """Route classification through the requested/active model."""
        classifier = self.get_classifier(model_id)
        return classifier.classify_crops(face_crops, boxes)
