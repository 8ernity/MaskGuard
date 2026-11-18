"""
Image processing utilities, format conversions, and computer vision rendering helpers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


# Palette in BGR for OpenCV drawing
COLOR_MASK_BGR = (94, 197, 34)       # Emerald Green #22C55E -> BGR(34, 197, 94)
COLOR_NOMASK_BGR = (68, 68, 239)     # Crimson Red #EF4444 -> BGR(68, 68, 239)
COLOR_UNCERTAIN_BGR = (11, 158, 245) # Amber #F59E0B -> BGR(11, 158, 245)
COLOR_TEXT_WHITE = (255, 255, 255)
COLOR_TEXT_DARK = (18, 20, 23)


def validate_image_file(path: Path | str) -> bool:
    """Validate that an image exists, is non-empty, and decodable."""
    p = Path(path)
    if not p.is_file() or p.stat().st_size == 0:
        return False
    try:
        data = np.fromfile(str(p), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img is not None and img.size > 0
    except Exception:
        return False


def read_image_bgr(path: Path | str) -> np.ndarray | None:
    """
    Safely read image using imdecode to support Unicode / Windows paths.
    Returns BGR numpy array or None on failure.
    """
    p = Path(path)
    if not p.is_file():
        return None
    try:
        data = np.fromfile(str(p), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def save_image_bgr(path: Path | str, image: np.ndarray) -> bool:
    """
    Safely write image using imencode to support Unicode / Windows paths.
    """
    p = Path(path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        ext = p.suffix.lower() if p.suffix else ".jpg"
        success, buffer = cv2.imencode(ext, image)
        if success:
            with open(p, "wb") as f:
                f.write(buffer)
            return True
        return False
    except Exception:
        return False


def bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    """Convert OpenCV BGR image to RGB."""
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(img_rgb: np.ndarray) -> np.ndarray:
    """Convert RGB image to OpenCV BGR."""
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)


def preprocess_face_tensor(
    face_bgr: np.ndarray,
    target_size: Tuple[int, int] = (128, 128),
) -> np.ndarray:
    """
    Canonical, unified preprocessing function shared identically across:
    - Training data generation
    - Validation and evaluation
    - Live webcam inference
    - Diagnostic inspection

    Pipeline:
    1. Validate input dimensions.
    2. Convert 1-channel / 2D grayscale to 3-channel BGR.
    3. Convert OpenCV BGR to RGB.
    4. Resize to target_size (128x128) using INTER_AREA.
    5. Cast to float32 and normalize strictly to [0.0, 1.0] by dividing by 255.0.

    Ensures zero distribution shift between training and production inference.
    """
    if face_bgr is None or face_bgr.size == 0:
        raise ValueError("Cannot preprocess an empty face image.")

    if len(face_bgr.shape) == 2:
        face_bgr = cv2.cvtColor(face_bgr, cv2.COLOR_GRAY2BGR)

    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(face_rgb, target_size, interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    return normalized


def get_detection_color(label: str, is_uncertain: bool = False) -> Tuple[int, int, int]:
    """Return BGR color tuple based on detection state."""
    if is_uncertain:
        return COLOR_UNCERTAIN_BGR
    if label.upper() in ("MASK", "WITH_MASK"):
        return COLOR_MASK_BGR
    return COLOR_NOMASK_BGR


def draw_bounding_box(
    image: np.ndarray,
    box: Tuple[int, int, int, int],
    label: str,
    confidence: float,
    is_uncertain: bool = False,
    show_box: bool = True,
    show_label: bool = True,
    show_conf: bool = True,
) -> np.ndarray:
    """
    Draw a polished bounding box with high-contrast text badge.
    Modifies image in-place and returns it.
    """
    x, y, w, h = box
    color = get_detection_color(label, is_uncertain)

    # Clean display label
    if is_uncertain:
        display_label = "UNCERTAIN"
    elif label.upper() in ("MASK", "WITH_MASK"):
        display_label = "MASK"
    else:
        display_label = "NO MASK"

    if show_conf:
        tag_text = f"{display_label} {confidence * 100:.1f}%"
    else:
        tag_text = display_label

    # Draw box if enabled
    if show_box:
        thickness = 2
        # Main rectangle
        cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness, cv2.LINE_AA)

        # Subtle corner accents for a sleek modern HUD look
        corner_len = max(10, min(w, h) // 5)
        accent_thickness = 3
        # Top-left
        cv2.line(image, (x, y), (x + corner_len, y), color, accent_thickness, cv2.LINE_AA)
        cv2.line(image, (x, y), (x, y + corner_len), color, accent_thickness, cv2.LINE_AA)
        # Top-right
        cv2.line(image, (x + w, y), (x + w - corner_len, y), color, accent_thickness, cv2.LINE_AA)
        cv2.line(image, (x + w, y), (x + w, y + corner_len), color, accent_thickness, cv2.LINE_AA)
        # Bottom-left
        cv2.line(image, (x, y + h), (x + corner_len, y + h), color, accent_thickness, cv2.LINE_AA)
        cv2.line(image, (x, y + h), (x, y + h - corner_len), color, accent_thickness, cv2.LINE_AA)
        # Bottom-right
        cv2.line(image, (x + w, y + h), (x + w - corner_len, y + h), color, accent_thickness, cv2.LINE_AA)
        cv2.line(image, (x + w, y + h), (x + w, y + corner_len), color, accent_thickness, cv2.LINE_AA)

    # Draw filled badge and text if enabled
    if show_label:
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.55
        font_thickness = 1

        (text_w, text_h), baseline = cv2.getTextSize(tag_text, font, font_scale, font_thickness)
        badge_h = text_h + 10
        badge_w = text_w + 12

        # Position badge above box if space permits, else inside top of box
        badge_y1 = max(0, y - badge_h)
        badge_y2 = badge_y1 + badge_h
        badge_x1 = max(0, x)
        badge_x2 = min(image.shape[1], badge_x1 + badge_w)

        # Draw solid badge background
        cv2.rectangle(image, (badge_x1, badge_y1), (badge_x2, badge_y2), color, cv2.FILLED)

        # Text inside badge
        text_origin = (badge_x1 + 6, badge_y2 - 6)
        text_color = COLOR_TEXT_WHITE if not is_uncertain else COLOR_TEXT_DARK
        cv2.putText(image, tag_text, text_origin, font, font_scale, text_color, font_thickness, cv2.LINE_AA)

    return image
