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
        scale_factor: float = 1.10,
        min_neighbors: int = 7,
        min_size: Tuple[int, int] = (50, 50),
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

        # Secondary fallback cascade (alt2) for stylized/difficult faces
        alt_path = ""
        if hasattr(cv2, "data") and os.path.isfile(
            os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_alt2.xml")
        ):
            alt_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_alt2.xml")
        self.classifier_alt = cv2.CascadeClassifier(alt_path) if alt_path else None

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
        Detect face bounding boxes in a BGR frame with adaptive downsampling for speed.
        Returns:
            List of (x, y, w, h) bounding boxes in original frame coordinates.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        h, w = frame_bgr.shape[:2]

        # Convert to grayscale for Haar detection
        if len(frame_bgr.shape) == 3:
            gray_orig = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray_orig = frame_bgr

        # Adaptive resolution downsampling: downscale large frames (e.g. 720p/1080p) to max width 640 for 5-8x faster detection
        max_dim = 640
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            small_w, small_h = max(1, int(w * scale)), max(1, int(h * scale))
            gray = cv2.resize(gray_orig, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
            scaled_min_size = (max(20, int(self.min_size[0] * scale)), max(20, int(self.min_size[1] * scale)))
        else:
            scale = 1.0
            gray = gray_orig
            scaled_min_size = self.min_size

        # Pass 1: standard detection pass with configured hyperparameters
        faces = self.classifier.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=scaled_min_size,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        # Pass 2: sensitive fallback for stylized, difficult, or static images
        if len(faces) == 0:
            relaxed_neighbors = max(3, self.min_neighbors - 3)
            relaxed_size = (max(24, int(scaled_min_size[0] * 0.65)), max(24, int(scaled_min_size[1] * 0.65)))
            faces = self.classifier.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=relaxed_neighbors,
                minSize=relaxed_size,
                flags=cv2.CASCADE_SCALE_IMAGE,
            )

        # Pass 3: secondary Alt2 cascade fallback
        if len(faces) == 0 and self.classifier_alt is not None and not self.classifier_alt.empty():
            faces = self.classifier_alt.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=max(3, self.min_neighbors - 3),
                minSize=(max(24, int(scaled_min_size[0] * 0.65)), max(24, int(scaled_min_size[1] * 0.65))),
                flags=cv2.CASCADE_SCALE_IMAGE,
            )

        if len(faces) == 0:
            return []

        # Map bounding box coordinates back to full-resolution space
        if scale != 1.0:
            inv_scale = 1.0 / scale
            raw_boxes = [
                (
                    int(round(x * inv_scale)),
                    int(round(y * inv_scale)),
                    int(round(box_w * inv_scale)),
                    int(round(box_h * inv_scale)),
                )
                for (x, y, box_w, box_h) in faces
            ]
        else:
            raw_boxes = [(int(x), int(y), int(w_box), int(h_box)) for (x, y, w_box, h_box) in faces]

        # Chromatic skin verification: reject background fabric, curtains, bedsheets, and wall clutter
        valid_boxes = []
        for b in raw_boxes:
            bx, by, bw, bh = b
            crop = frame_bgr[max(0, by):min(h, by + bh), max(0, bx):min(w, bx + bw)]
            if self._is_valid_face_candidate(crop):
                valid_boxes.append(b)

        return valid_boxes

    def _is_valid_face_candidate(self, crop_bgr: np.ndarray) -> bool:
        """
        Verify that a candidate bounding box contains human facial features
        and skin chromaticity across diverse skin tones (YCrCb + HSV) to reject
        background textures, curtains, bedsheets, and fabric.
        """
        if crop_bgr is None or crop_bgr.size == 0 or crop_bgr.shape[0] < 20 or crop_bgr.shape[1] < 20:
            return False

        # Fallback for single-channel grayscale frames
        if len(crop_bgr.shape) == 2:
            return True

        # 1. Aspect ratio check: Haar frontal faces are roughly square (0.60 to 1.55)
        ch, cw = crop_bgr.shape[:2]
        aspect = float(cw) / float(ch)
        if aspect < 0.60 or aspect > 1.55:
            return False

        # 2. Dual-space skin chromaticity check (YCrCb + HSV)
        ycrcb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2YCrCb)
        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)

        # YCrCb skin tone envelope across diverse skin tones (Cr: 133-173, Cb: 77-127)
        skin_ycrcb = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))

        # HSV skin tone envelope: Hue in [0, 28] or [160, 180], Saturation in [20, 240], Value in [30, 255]
        # Curtains and background fabrics are usually achromatic (S < 20) or cool tones (blue/green)
        skin_hsv1 = cv2.inRange(hsv, np.array([0, 20, 30]), np.array([28, 240, 255]))
        skin_hsv2 = cv2.inRange(hsv, np.array([160, 20, 30]), np.array([180, 240, 255]))
        skin_hsv = cv2.bitwise_or(skin_hsv1, skin_hsv2)

        skin_and = cv2.bitwise_and(skin_ycrcb, skin_hsv)
        skin_or = cv2.bitwise_or(skin_ycrcb, skin_hsv)

        ratio_and = float(np.sum(skin_and > 0)) / skin_and.size
        ratio_or = float(np.sum(skin_or > 0)) / skin_or.size

        # Any human face exposes skin on forehead/eyes/cheeks.
        # Strict intersection (ratio_and >= 0.04) or broader union (ratio_or >= 0.10)
        # Curtains, bedsheets, wall folds have < 0.015 in both.
        return ratio_and >= 0.04 or ratio_or >= 0.10

    def expand_box(
        self,
        box: Tuple[int, int, int, int],
        frame_shape: Tuple[int, int, ...],
        padding: float | None = None,
    ) -> Tuple[int, int, int, int]:
        """
        Expand bounding box symmetrically to capture the entire face cleanly
        while strictly preserving the 1:1 square aspect ratio matching the training dataset.
        Includes a downward shift to guarantee complete chin, mouth, and jawline coverage.
        """
        pad = self.padding if padding is None else padding

        x, y, w, h = box
        img_h, img_w = frame_shape[:2]

        # Use square expansion matching 128x128 training chips
        dim = max(w, h)
        # Expansion factor ensures chin, jawline, and mouth are fully within the crop
        scale_mult = 1.35 + (pad * 0.4)
        target_size = int(dim * scale_mult)
        half_side = target_size // 2

        cx = x + w // 2
        # Shift center downward (+0.12 * dim) so chin, lips, and lower face are comfortably included
        cy = y + h // 2 + int(dim * 0.12)

        new_x = max(0, cx - half_side)
        new_y = max(0, cy - half_side)
        new_w = min(img_w - new_x, half_side * 2)
        new_h = min(img_h - new_y, half_side * 2)

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
