"""
Diagnostic utility for MaskGuard prediction flow inspection.
Traces every stage: Image -> Face Detection -> Face Crop -> Preprocessing -> CNN Raw Output -> Classification -> UI Representation.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Suppress TF verbose logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import tensorflow as tf

from app.core.face_detector import FaceDetector
from app.utils.resource_paths import (
    get_class_mapping_path,
    get_default_model_path,
    get_model_metadata_path,
    get_results_dir,
)


def load_model_and_metadata(model_path: Optional[Path | str] = None):
    m_path = Path(model_path) if model_path else get_default_model_path()
    if not m_path.is_file():
        raise FileNotFoundError(f"Model file not found at {m_path}")
    model = tf.keras.models.load_model(str(m_path))

    mapping_path = get_class_mapping_path()
    with open(mapping_path, "r", encoding="utf-8") as f:
        class_mapping = json.load(f)

    meta_path = get_model_metadata_path()
    meta = {}
    if meta_path.is_file():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    return model, class_mapping, meta


def shared_preprocess_face(face_bgr: np.ndarray, target_size: Tuple[int, int] = (128, 128)) -> np.ndarray:
    """
    The canonical, shared, single source of truth preprocessing pipeline:
    1. Verify valid 3-channel BGR.
    2. Convert BGR to RGB.
    3. Resize to 128x128 with cv2.INTER_AREA.
    4. Cast to float32 and normalize strictly to [0.0, 1.0] by dividing by 255.0.
    """
    if face_bgr is None or face_bgr.size == 0:
        raise ValueError("Cannot preprocess empty face image.")

    if len(face_bgr.shape) == 2:
        face_bgr = cv2.cvtColor(face_bgr, cv2.COLOR_GRAY2BGR)

    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(face_rgb, target_size, interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    return normalized


def diagnose_image(
    image_path: Path | str,
    output_dir: Optional[Path | str] = None,
    save_artifacts: bool = True,
    model_path: Optional[Path | str] = None,
) -> List[Dict[str, Any]]:
    """
    Run diagnostic inspection on a single image and export full traceable artifacts.
    """
    img_path = Path(image_path)
    if not img_path.is_file():
        raise FileNotFoundError(f"Image not found: {img_path}")

    # Read image using cv2.imdecode to support Unicode paths
    data = np.fromfile(str(img_path), dtype=np.uint8)
    frame_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        raise ValueError(f"Could not decode image from {img_path}")

    model, class_mapping, meta = load_model_and_metadata(model_path)
    detector = FaceDetector(scale_factor=1.10, min_neighbors=5, min_size=(40, 40), padding=0.12)

    boxes = detector.detect(frame_bgr)
    results = []

    diag_dir = Path(output_dir) if output_dir else (get_results_dir() / "diagnostics")
    if save_artifacts:
        diag_dir.mkdir(parents=True, exist_ok=True)
        # Save original frame copy
        cv2.imwrite(str(diag_dir / f"{img_path.stem}_01_orig.jpg"), frame_bgr)

    annotated = frame_bgr.copy()

    for idx, box in enumerate(boxes):
        x, y, w, h = box
        # Extract crop using standard face detector expansion
        crop_bgr = detector.crop_face(frame_bgr, box, apply_padding=True)
        if crop_bgr is None or crop_bgr.size == 0:
            continue

        expanded_box = detector.expand_box(box, frame_bgr.shape, frame_bgr=frame_bgr)
        ex, ey, ew, eh = expanded_box

        # Standard preprocessing
        tensor_preproc = shared_preprocess_face(crop_bgr, target_size=(128, 128))
        tensor_batch = np.expand_dims(tensor_preproc, axis=0)

        # CNN forward pass
        raw_prob = float(model(tensor_batch, training=False).numpy()[0, 0])

        # Interpretation:
        # 0 = with_mask (raw -> 0)
        # 1 = without_mask (raw -> 1)
        prob_with_mask = 1.0 - raw_prob
        prob_without_mask = raw_prob

        if raw_prob < 0.5:
            pred_class = "with_mask"
            ui_label = "MASK"
            confidence = prob_with_mask
            ui_color = "#10B981"  # Emerald Green
        else:
            pred_class = "without_mask"
            ui_label = "NO MASK"
            confidence = prob_without_mask
            ui_color = "#EF4444"  # Rose Red

        # Check uncertainty band [0.4, 0.6]
        is_uncertain = (0.40 <= raw_prob <= 0.60) or (confidence < 0.65)
        displayed_label = "UNCERTAIN" if is_uncertain else ui_label
        displayed_color = "#F59E0B" if is_uncertain else ui_color

        report = {
            "image_name": img_path.name,
            "face_index": idx,
            "raw_box": [int(x), int(y), int(w), int(h)],
            "expanded_box": [int(ex), int(ey), int(ew), int(eh)],
            "crop_dimensions": list(crop_bgr.shape),
            "preprocessed_tensor_shape": list(tensor_preproc.shape),
            "preprocessed_min": float(tensor_preproc.min()),
            "preprocessed_max": float(tensor_preproc.max()),
            "preprocessed_mean": float(tensor_preproc.mean()),
            "raw_cnn_sigmoid_output": round(raw_prob, 6),
            "prob_with_mask": round(prob_with_mask, 6),
            "prob_without_mask": round(prob_without_mask, 6),
            "predicted_class": pred_class,
            "canonical_label": ui_label,
            "confidence": round(confidence, 4),
            "confidence_pct": round(confidence * 100, 1),
            "is_uncertain": is_uncertain,
            "frontend_displayed_label": displayed_label,
            "frontend_color_hex": displayed_color,
        }
        results.append(report)

        if save_artifacts:
            # Save extracted crop
            cv2.imwrite(str(diag_dir / f"{img_path.stem}_face_{idx}_crop.jpg"), crop_bgr)
            # Save preprocessed representation
            preproc_vis = (tensor_preproc * 255.0).astype(np.uint8)
            preproc_bgr = cv2.cvtColor(preproc_vis, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(diag_dir / f"{img_path.stem}_face_{idx}_preproc128.jpg"), preproc_bgr)

            # Draw on annotated
            color_bgr = (16, 185, 129) if displayed_label == "MASK" else (68, 68, 239) if displayed_label == "NO MASK" else (11, 158, 245)
            cv2.rectangle(annotated, (ex, ey), (ex + ew, ey + eh), color_bgr, 2)
            cv2.putText(
                annotated,
                f"{displayed_label} {confidence*100:.1f}% (raw={raw_prob:.3f})",
                (ex, max(20, ey - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color_bgr,
                2,
            )

    if save_artifacts:
        cv2.imwrite(str(diag_dir / f"{img_path.stem}_02_annotated.jpg"), annotated)
        with open(diag_dir / f"{img_path.stem}_diagnostic_report.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MaskGuard diagnostic flow tracer.")
    parser.add_argument("image_path", type=str, help="Path to input image.")
    parser.add_argument("--save", action="store_true", default=True, help="Save intermediate visual artifacts.")
    args = parser.parse_args()

    res = diagnose_image(args.image_path, save_artifacts=args.save)
    print(json.dumps(res, indent=2))
