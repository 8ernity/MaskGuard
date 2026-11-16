"""
Face detection module powered by OpenCV.
Detects human faces in real-time camera frames and static images using OpenCV Haar Cascade.
Includes safe bounding box expansion to ensure full mask coverage (mouth to chin).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


class FaceDetector:
    """
    OpenCV-based face detection engine.
    Locates faces and extracts crops with boundary-safe padding.
    """

    def __init__(
        self,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: Tuple[int, int] = (30, 30),
        padding: float = 0.12,
    ):
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size
        self.padding = padding

        # Locate OpenCV cascade file: check bundled assets first, then cv2.data
        bundled_cascade = (
            Path(__file__).resolve().parent / "assets" / "haarcascade_frontalface_default.xml"
        )
        if bundled_cascade.is_file():
            cascade_path = str(bundled_cascade)
        elif hasattr(cv2, "data") and os.path.isfile(
            os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        ):
            cascade_path = os.path.join(
                cv2.data.haarcascades, "haarcascade_frontalface_default.xml"
            )
        else:
            cascade_path = cv2.samples.findFile("haarcascade_frontalface_default.xml")

        self.classifier = cv2.CascadeClassifier(cascade_path)
        if self.classifier.empty():
            raise RuntimeError(f"Failed to load OpenCV face cascade from {cascade_path}")

    def update_parameters(
        self,
        scale_factor: float | None = None,
        min_neighbors: int | None = None,
        min_size: int | None = None,
        padding: float | None = None,
    ) -> None:
        """Update detection hyperparameters dynamically."""
        if scale_factor is not None:
            self.scale_factor = max(1.01, float(scale_factor))
        if min_neighbors is not None:
            self.min_neighbors = max(1, int(min_neighbors))
        if min_size is not None:
            self.min_size = (int(min_size), int(min_size))
        if padding is not None:
            self.padding = max(0.0, min(0.4, float(padding)))

    def detect(self, frame_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect face bounding boxes in a BGR frame.
        Returns:
            List of (x, y, w, h) bounding boxes.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        # Convert to grayscale for Haar detection
        if len(frame_bgr.shape) == 3:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame_bgr

        # Equalize histogram for illumination invariance
        gray = cv2.equalizeHist(gray)

        faces = self.classifier.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        if len(faces) == 0:
            return []

        # Convert numpy array to list of tuples
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]

    def expand_box(
        self,
        box: Tuple[int, int, int, int],
        frame_shape: Tuple[int, int, ...],
        padding: float | None = None,
    ) -> Tuple[int, int, int, int]:
        """
        Expand bounding box outwards with extra downward bias to capture the entire chin/mask.
        Clamps to image boundaries.
        """
        pad = self.padding if padding is None else padding
        if pad <= 0:
            return box

        x, y, w, h = box
        img_h, img_w = frame_shape[:2]

        # Horizontal padding
        pad_w = int(w * pad)
        # Vertical padding: 60% of padding downwards to capture lower mask / chin
        pad_top = int(h * pad * 0.7)
        pad_bottom = int(h * pad * 1.3)

        new_x = max(0, x - pad_w)
        new_y = max(0, y - pad_top)
        new_w = min(img_w - new_x, w + 2 * pad_w)
        new_h = min(img_h - new_y, h + pad_top + pad_bottom)

        return (new_x, new_y, new_w, new_h)

    def crop_face(
        self,
        frame_bgr: np.ndarray,
        box: Tuple[int, int, int, int],
        apply_padding: bool = True,
    ) -> np.ndarray | None:
        """
        Extract the cropped face region from the image with boundary checks.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return None

        if apply_padding:
            x, y, w, h = self.expand_box(box, frame_bgr.shape)
        else:
            x, y, w, h = box

        if w <= 0 or h <= 0:
            return None

        crop = frame_bgr[y : y + h, x : x + w]
        if crop.size == 0:
            return None
        return crop
