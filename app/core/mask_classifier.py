"""
Face mask classification engine powered by the trained custom CNN.
Performs exact preprocessing, forward inference, probability calibration, and uncertainty detection.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import tensorflow as tf

from app.utils.resource_paths import (
    get_default_model_path,
    get_class_mapping_path,
)
from training.prepare_dataset import load_class_mapping


@dataclass
class FaceDetection:
    """Dataclass holding inference results for a single face."""
    box: Tuple[int, int, int, int]  # (x, y, w, h)
    label: str                      # "MASK" | "NO MASK" | "UNCERTAIN"
    confidence: float              # 0.0 to 1.0
    raw_score: float               # Raw sigmoid output [0.0, 1.0]
    is_uncertain: bool             # Flag indicating low confidence or ambiguous boundary


class MaskClassifier:
    """
    CNN inference engine for cropped face regions.
    Reuses a single compiled Keras model instance.
    """

    def __init__(
        self,
        model_path: Path | str | None = None,
        confidence_threshold: float = 0.65,
        uncertain_band: float = 0.15,
        target_size: Tuple[int, int] = (128, 128),
    ):
        self.model_path = Path(model_path) if model_path else get_default_model_path()
        self.confidence_threshold = confidence_threshold
        self.uncertain_band = uncertain_band
        self.target_size = target_size
        self.model: tf.keras.Model | None = None
        self.class_map = load_class_mapping(get_class_mapping_path())

        self.load_model(self.model_path)

    def load_model(self, path: Path | str) -> bool:
        """Load or reload model weights from disk."""
        target_path = Path(path)
        if not target_path.exists():
            print(f"[MaskClassifier] Model file not found at {target_path}. Engine initialized in standby mode.")
            self.model = None
            return False

        try:
            print(f"[MaskClassifier] Loading model from {target_path}...")
            self.model = tf.keras.models.load_model(str(target_path))
            # Warm up graph execution with dummy tensor
            dummy = np.zeros((1, *self.target_size, 3), dtype=np.float32)
            _ = self.model.predict(dummy, verbose=0)
            self.model_path = target_path
            print("[MaskClassifier] Model successfully loaded and warmed up.")
            return True
        except Exception as e:
            print(f"[MaskClassifier] Error loading model from {target_path}: {e}")
            self.model = None
            return False

    def is_ready(self) -> bool:
        """Check if model is loaded and ready for inference."""
        return self.model is not None

    def preprocess_face(self, face_bgr: np.ndarray) -> np.ndarray:
        """
        Preprocess face crop to match training pipeline:
        1. BGR -> RGB
        2. Resize to 128x128
        3. Normalize pixel values to [0.0, 1.0]
        """
        if face_bgr is None or face_bgr.size == 0:
            raise ValueError("Empty face crop provided for preprocessing.")

        # Ensure 3 channels
        if len(face_bgr.shape) == 2:
            face_bgr = cv2.cvtColor(face_bgr, cv2.COLOR_GRAY2BGR)

        # Convert to RGB
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)

        # Bilinear / area interpolation resize
        resized = cv2.resize(face_rgb, self.target_size, interpolation=cv2.INTER_AREA)

        # Normalize float32 [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        return normalized

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
        """
        Batch inference across multiple detected face crops.
        """
        if not face_crops or not self.is_ready():
            # Return fallback empty detections
            detections = []
            for b in boxes:
                detections.append(
                    FaceDetection(
                        box=b,
                        label="UNCERTAIN",
                        confidence=0.0,
                        raw_score=0.5,
                        is_uncertain=True,
                    )
                )
            return detections

        # Preprocess batch
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

        # Forward pass
        raw_predictions = self.model.predict(batch_tensor, verbose=0).flatten()

        detections: List[FaceDetection] = []
        for idx, orig_idx in enumerate(valid_indices):
            raw_val = float(raw_predictions[idx])
            box = boxes[orig_idx]

            # In training: 0 = with_mask, 1 = without_mask
            # Therefore:
            # P(with_mask) = 1.0 - raw_val
            # P(without_mask) = raw_val
            mask_prob = 1.0 - raw_val
            nomask_prob = raw_val

            # Determine dominant class and nominal confidence
            if raw_val < 0.5:
                pred_label = "MASK"
                conf = mask_prob
            else:
                pred_label = "NO MASK"
                conf = nomask_prob

            # Check uncertainty conditions:
            # 1. Close to decision boundary 0.5 within uncertainty band
            # 2. Or confidence fails user-defined confidence threshold
            is_uncertain = (
                abs(raw_val - 0.5) < self.uncertain_band
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
                )
            )

        return detections
