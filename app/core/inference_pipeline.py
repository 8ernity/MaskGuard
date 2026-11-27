"""
Unified inference pipeline connecting OpenCV face detection and CNN mask classification.
Shared across webcam live detection, static image analysis, and video processing.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from app.core.face_detector import FaceDetector
from app.core.mask_classifier import FaceDetection, MaskClassifier
from app.core.model_manager import DualModelManager
from app.utils.config import AppConfig
from app.utils.image_utils import draw_bounding_box


class SimpleFaceTracker:
    """
    Lightweight centroid tracker to prevent counting the same person
    across consecutive video frames as a brand-new unique face.
    """

    def __init__(self, max_distance: float = 65.0, max_disappeared: int = 15):
        self.next_object_id = 1
        self.objects: Dict[int, Tuple[int, int]] = {}  # id -> centroid (cx, cy)
        self.disappeared: Dict[int, int] = {}
        self.total_unique_faces = 0
        self.max_distance = max_distance
        self.max_disappeared = max_disappeared

    def reset(self) -> None:
        self.next_object_id = 1
        self.objects.clear()
        self.disappeared.clear()
        self.total_unique_faces = 0

    def update(self, boxes: List[Tuple[int, int, int, int]]) -> int:
        """
        Update tracker with current frame bounding boxes.
        Returns total cumulative unique faces detected in this session.
        """
        if len(boxes) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    del self.objects[object_id]
                    del self.disappeared[object_id]
            return self.total_unique_faces

        input_centroids = []
        for (x, y, w, h) in boxes:
            input_centroids.append((int(x + w / 2.0), int(y + h / 2.0)))

        if len(self.objects) == 0:
            for centroid in input_centroids:
                self.objects[self.next_object_id] = centroid
                self.disappeared[self.next_object_id] = 0
                self.next_object_id += 1
                self.total_unique_faces += 1
            return self.total_unique_faces

        # Match existing centroids to new centroids
        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())

        # Distance matrix
        dists = []
        for (ox, oy) in object_centroids:
            row = []
            for (ix, iy) in input_centroids:
                d = np.hypot(ox - ix, oy - iy)
                row.append(d)
            dists.append(row)

        dists = np.array(dists)
        rows = dists.min(axis=1).argsort()
        cols = dists.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        for (row, col) in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if dists[row, col] > self.max_distance:
                continue

            object_id = object_ids[row]
            self.objects[object_id] = input_centroids[col]
            self.disappeared[object_id] = 0

            used_rows.add(row)
            used_cols.add(col)

        unused_rows = set(range(len(object_ids))).difference(used_rows)
        unused_cols = set(range(len(input_centroids))).difference(used_cols)

        for row in unused_rows:
            object_id = object_ids[row]
            self.disappeared[object_id] += 1
            if self.disappeared[object_id] > self.max_disappeared:
                del self.objects[object_id]
                del self.disappeared[object_id]

        for col in unused_cols:
            self.objects[self.next_object_id] = input_centroids[col]
            self.disappeared[self.next_object_id] = 0
            self.next_object_id += 1
            self.total_unique_faces += 1

        return self.total_unique_faces


class InferencePipeline:
    """
    Complete inference pipeline integrating face detector and mask classifier.
    """

    def __init__(
        self,
        config: AppConfig | None = None,
        face_detector: FaceDetector | None = None,
        mask_classifier: MaskClassifier | None = None,
        model_manager: DualModelManager | None = None,
    ):
        self.config = config or AppConfig.load()
        self.detector = face_detector or FaceDetector(
            scale_factor=self.config.face_scale_factor,
            min_neighbors=self.config.face_min_neighbors,
            min_size=(self.config.face_min_size, self.config.face_min_size),
            padding=self.config.face_padding,
        )
        if mask_classifier is not None:
            self.classifier = mask_classifier
            self.model_manager = model_manager
        else:
            self.model_manager = model_manager or DualModelManager(
                active_model_id=getattr(self.config, "active_model", "custom_cnn"),
                confidence_threshold=self.config.confidence_threshold,
                uncertain_band=self.config.uncertain_band,
                custom_cnn_path=self.config.model_path,
            )
            self.classifier = self.model_manager.get_classifier(self.model_manager.active_model_id)
        self.tracker = SimpleFaceTracker()
        self.last_frame_time = time.time()
        self.fps_smoothed = 0.0

    @property
    def active_model_id(self) -> str:
        """Return currently active model identifier."""
        if self.model_manager is not None:
            return self.model_manager.active_model_id
        return getattr(self.classifier, "model_id", "custom_cnn")

    def select_model(self, model_id: str) -> Dict[str, Any]:
        """Switch active model cleanly through DualModelManager."""
        if self.model_manager is None:
            raise RuntimeError("DualModelManager is not configured for this pipeline.")
        info = self.model_manager.select_model(model_id)
        self.classifier = self.model_manager.get_classifier(model_id)
        return info

    def update_thresholds(self, confidence_threshold: float, uncertain_band: float) -> None:
        """Synchronize confidence and uncertainty thresholds across classifier(s)."""
        if self.model_manager is not None:
            self.model_manager.update_thresholds(confidence_threshold, uncertain_band)
        self.classifier.confidence_threshold = confidence_threshold
        self.classifier.uncertain_band = uncertain_band

    def reset_session_stats(self) -> None:
        """Reset tracker and counters for a new detection session."""
        self.tracker.reset()

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        track_session: bool = True,
        model_id: str | None = None,
    ) -> Tuple[np.ndarray, List[FaceDetection], Dict[str, Any]]:
        """
        Process a single BGR frame:
        1. Detect faces.
        2. Extract crops and classify.
        3. Draw bounding boxes and status badges.
        4. Track statistics.

        Returns:
            (annotated_frame, list_of_detections, stats_dict)
        """
        now = time.time()
        dt = max(1e-5, now - self.last_frame_time)
        self.last_frame_time = now
        inst_fps = 1.0 / dt
        self.fps_smoothed = 0.85 * self.fps_smoothed + 0.15 * inst_fps if self.fps_smoothed > 0 else inst_fps

        used_model_id = model_id or self.active_model_id

        if frame_bgr is None or frame_bgr.size == 0:
            return frame_bgr, [], {
                "faces_in_frame": 0,
                "mask_count": 0,
                "nomask_count": 0,
                "uncertain_count": 0,
                "total_unique_faces": self.tracker.total_unique_faces,
                "fps": round(self.fps_smoothed, 1),
                "model_id": used_model_id,
            }

        annotated = frame_bgr.copy()
        boxes = self.detector.detect(frame_bgr)

        # Update centroid tracker
        if track_session:
            unique_faces = self.tracker.update(boxes)
        else:
            unique_faces = len(boxes)

        detections: List[FaceDetection] = []
        if boxes:
            face_crops = []
            valid_boxes = []
            for b in boxes:
                crop = self.detector.crop_face(frame_bgr, b, apply_padding=True)
                if crop is not None:
                    face_crops.append(crop)
                    valid_boxes.append(b)

            if face_crops:
                if self.model_manager is not None:
                    detections = self.model_manager.classify_crops(
                        face_crops,
                        valid_boxes,
                        model_id=used_model_id,
                    )
                else:
                    detections = self.classifier.classify_crops(face_crops, valid_boxes)

        # Count classifications
        mask_count = sum(1 for d in detections if d.label == "MASK" and not d.is_uncertain)
        nomask_count = sum(1 for d in detections if d.label == "NO MASK" and not d.is_uncertain)
        uncertain_count = sum(1 for d in detections if d.is_uncertain)

        # Annotate bounding boxes
        for det in detections:
            draw_bounding_box(
                annotated,
                box=det.box,
                label=det.label,
                confidence=det.confidence,
                is_uncertain=det.is_uncertain,
                show_box=self.config.show_boxes,
                show_label=self.config.show_labels,
                show_conf=self.config.show_confidence,
            )

        # Draw FPS indicator if enabled
        if self.config.show_fps:
            fps_text = f"FPS: {self.fps_smoothed:.1f}"
            cv2.putText(
                annotated,
                fps_text,
                (14, 28),
                cv2.FONT_HERSHEY_DUPLEX,
                0.65,
                (16, 185, 129),  # Emerald Green
                2,
                cv2.LINE_AA,
            )

        stats = {
            "faces_in_frame": len(detections),
            "mask_count": mask_count,
            "nomask_count": nomask_count,
            "uncertain_count": uncertain_count,
            "total_unique_faces": unique_faces,
            "fps": round(self.fps_smoothed, 1),
            "model_id": used_model_id,
        }

        return annotated, detections, stats
