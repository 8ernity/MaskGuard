"""
MobileNetV2 transfer-learning inference engine for MaskGuard.
Loads models/mobilenetv2/mask_detector_mobilenetv2.h5 independently of the Custom CNN,
applies dedicated 224x224 [-1.0, 1.0] preprocessing, and never falls back to Custom CNN.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import List, Tuple

import numpy as np
import tensorflow as tf

from app.core.mask_classifier import FaceDetection
from app.utils.image_utils import preprocess_face_mobilenetv2
from app.utils.resource_paths import (
    get_mobilenetv2_class_mapping_path,
    get_mobilenetv2_metadata_path,
    get_mobilenetv2_model_path,
)
from training.prepare_dataset import load_class_mapping


class MobileNetV2Classifier:
    """
    Independent MobileNetV2 CNN inference engine for cropped face regions.
    """

    model_id: str = "mobilenetv2"
    display_name: str = "MobileNetV2"

    def __init__(
        self,
        model_path: Path | str | None = None,
        confidence_threshold: float = 0.65,
        uncertain_band: float = 0.15,
        target_size: Tuple[int, int] = (224, 224),
        auto_load: bool = True,
    ):
        self.model_path = Path(model_path) if model_path else get_mobilenetv2_model_path()
        self.confidence_threshold = confidence_threshold
        self.uncertain_band = uncertain_band
        self.target_size = target_size
        self.model: tf.keras.Model | None = None
        self._fast_predict = None
        self._lock = threading.Lock()
        self.load_error: str | None = None

        mapping_path = get_mobilenetv2_class_mapping_path()
        if mapping_path.is_file():
            self.class_map = load_class_mapping(mapping_path)
        else:
            self.class_map = {0: "with_mask", 1: "without_mask"}

        if auto_load:
            self.load_model(self.model_path)

    def _validate_metadata(self) -> None:
        """Verify MobileNetV2 metadata consistency if present."""
        meta_path = get_mobilenetv2_metadata_path()
        if not meta_path.is_file():
            return
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            dims = meta.get("input_dimensions")
            if dims and list(dims[:2]) != [self.target_size[0], self.target_size[1]]:
                raise ValueError(
                    f"MobileNetV2 metadata input_dimensions={dims} incompatible with expected {self.target_size}."
                )
            cmap = meta.get("class_mapping")
            if cmap and (cmap.get("0") != "with_mask" or cmap.get("1") != "without_mask"):
                raise ValueError(f"Unexpected MobileNetV2 class_mapping in metadata: {cmap}")
        except ValueError:
            raise
        except Exception as exc:
            print(f"[MobileNetV2Classifier] Metadata check warning: {exc}")

    def load_model(self, path: Path | str | None = None) -> bool:
        """Load MobileNetV2 weights strictly from MobileNetV2 artifact path (no fallback)."""
        target_path = Path(path) if path else self.model_path
        if not target_path.is_file():
            self.load_error = f"MobileNetV2 model file not found at {target_path}. Train MobileNetV2 first."
            print(f"[MobileNetV2Classifier] {self.load_error}")
            self.model = None
            self._fast_predict = None
            return False

        try:
            self._validate_metadata()
            with self._lock:
                print(f"[MobileNetV2Classifier] Loading model from {target_path}...")
                self.model = tf.keras.models.load_model(str(target_path))

                @tf.function(reduce_retracing=True)
                def _forward(x):
                    return self.model(x, training=False)

                self._fast_predict = _forward
                dummy = tf.zeros((1, *self.target_size, 3), dtype=np.float32)
                _ = self._fast_predict(dummy)
                self.model_path = target_path
                self.load_error = None
                print("[MobileNetV2Classifier] Model loaded, compiled, and warmed up.")
                return True
        except Exception as exc:
            self.load_error = f"Failed to load MobileNetV2 model from {target_path}: {exc}"
            print(f"[MobileNetV2Classifier] {self.load_error}")
            self.model = None
            self._fast_predict = None
            return False

    def is_ready(self) -> bool:
        """Check if MobileNetV2 model is loaded and ready."""
        return self.model is not None

    def preprocess_face(self, face_bgr: np.ndarray) -> np.ndarray:
        """Apply dedicated 224x224 [-1.0, 1.0] preprocessing for MobileNetV2."""
        return preprocess_face_mobilenetv2(face_bgr, target_size=self.target_size)

    def classify_crop(
        self,
        face_bgr: np.ndarray,
        box: Tuple[int, int, int, int],
    ) -> FaceDetection:
        """Classify a single face crop."""
        results = self.classify_crops([face_bgr], [box])
        return results[0]

    def classify_crops(
        self,
        face_crops: List[np.ndarray],
        boxes: List[Tuple[int, int, int, int]],
    ) -> List[FaceDetection]:
        """Batch inference across detected face crops using MobileNetV2."""
        if not self.is_ready():
            raise RuntimeError(self.load_error or "MobileNetV2 model is not loaded.")

        if not face_crops:
            return []

        preprocessed = []
        valid_indices = []
        for i, crop in enumerate(face_crops):
            try:
                proc = self.preprocess_face(crop)
                preprocessed.append(proc)
                valid_indices.append(i)
            except Exception:
                continue

        if not preprocessed:
            return []

        batch_tensor = np.array(preprocessed, dtype=np.float32)

        with self._lock:
            if self._fast_predict is not None:
                raw_predictions = self._fast_predict(tf.constant(batch_tensor)).numpy().flatten()
            else:
                raw_predictions = self.model(batch_tensor, training=False).numpy().flatten()

        detections: List[FaceDetection] = []
        for idx, orig_idx in enumerate(valid_indices):
            raw_val = float(raw_predictions[idx])
            box = boxes[orig_idx]

            # 0 = with_mask, 1 = without_mask
            mask_prob = 1.0 - raw_val
            nomask_prob = raw_val

            if raw_val < 0.5:
                pred_label = "MASK"
                conf = mask_prob
            else:
                pred_label = "NO MASK"
                conf = nomask_prob

            effective_band = min(
                self.uncertain_band,
                max(0.0, self.confidence_threshold - 0.5),
            )
            is_uncertain = (
                abs(raw_val - 0.5) < effective_band
                or conf < self.confidence_threshold
            )
            if is_uncertain:
                pred_label = "UNCERTAIN"

            detections.append(
                FaceDetection(
                    box=box,
                    label=pred_label,
                    confidence=conf,
                    raw_score=raw_val,
                    is_uncertain=is_uncertain,
                    model_id=self.model_id,
                )
            )

        return detections
