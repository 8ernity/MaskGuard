"""
Standalone OpenCV webcam demonstration script for MaskGuard.
Executes real-time face mask detection in an OpenCV window without requiring the PySide6 desktop GUI.

Controls:
  'q' or ESC - Exit demo
  's'        - Save screenshot to results/screenshots/
  'b'        - Toggle bounding box display
  'l'        - Toggle label display
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to sys.path so app modules can be imported
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
from app.core.face_detector import FaceDetector
from app.core.inference_pipeline import InferencePipeline
from app.core.mask_classifier import MaskClassifier
from app.utils.config import AppConfig
from app.utils.image_utils import save_image_bgr
from app.utils.resource_paths import get_default_model_path, get_screenshots_dir


def run_webcam_demo(
    camera_index: int = 0,
    model_path: str | None = None,
    confidence_threshold: float = 0.50,
    scale_factor: float = 1.10,
    min_neighbors: int = 7,
) -> None:
    """Run standalone OpenCV camera loop."""
    config = AppConfig.load()
    config.camera_index = camera_index
    config.confidence_threshold = confidence_threshold
    config.face_scale_factor = scale_factor
    config.face_min_neighbors = min_neighbors
    if model_path:
        config.model_path = model_path

    print("==================================================")
    print("      MaskGuard - Standalone Webcam Demo          ")
    print("==================================================")
    print(f"Model Path:            {config.model_path}")
    print(f"Camera Device Index:   {config.camera_index}")
    print(f"Confidence Threshold:  {config.confidence_threshold}")
    print("Keys: [q / ESC] Quit  |  [s] Save Snapshot  |  [b] Toggle Box  |  [l] Toggle Labels")
    print("==================================================")

    # Initialize components
    detector = FaceDetector(
        scale_factor=config.face_scale_factor,
        min_neighbors=config.face_min_neighbors,
        min_size=(config.face_min_size, config.face_min_size),
        padding=config.face_padding,
    )
    classifier = MaskClassifier(
        model_path=config.model_path,
        confidence_threshold=config.confidence_threshold,
        uncertain_band=config.uncertain_band,
    )

    if not classifier.is_ready():
        print(f"\n[WARNING] Model file at '{config.model_path}' could not be loaded.")
        print("Please train the model first by running: python training/train.py\n")

    pipeline = InferencePipeline(
        config=config,
        face_detector=detector,
        mask_classifier=classifier,
    )

    # Open camera with DirectShow backend on Windows
    backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else cv2.CAP_ANY
    cap = cv2.VideoCapture(camera_index, backend)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"\n[ERROR] Failed to open camera with index {camera_index}.")
        print("Check if your webcam is connected or if another program is using it.")
        return

    # Set frame dimensions
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    window_name = "MaskGuard - Real-time Face Mask Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)

    print("\nWebcam feed started. Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[WARNING] Empty frame received from camera. Retrying...")
                time.sleep(0.05)
                continue

            # Process frame through pipeline
            annotated_frame, detections, stats = pipeline.process_frame(frame, track_session=True)

            # Draw HUD bottom bar with statistics
            h, w = annotated_frame.shape[:2]
            hud_bg = annotated_frame.copy()
            cv2.rectangle(hud_bg, (0, h - 36), (w, h), (18, 20, 23), cv2.FILLED)
            annotated_frame = cv2.addWeighted(annotated_frame, 0.4, hud_bg, 0.6, 0)

            hud_text = (
                f"Faces: {stats['faces_in_frame']}  |  "
                f"Masks: {stats['mask_count']}  |  "
                f"No-Masks: {stats['nomask_count']}  |  "
                f"Uncertain: {stats['uncertain_count']}  |  "
                f"Unique Total: {stats['total_unique_faces']}"
            )
            cv2.putText(
                annotated_frame,
                hud_text,
                (12, h - 12),
                cv2.FONT_HERSHEY_DUPLEX,
                0.48,
                (240, 246, 252),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # 'q' or ESC
                print("\nExiting webcam demo...")
                break
            elif key == ord("s"):  # 's' Snapshot
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                snap_path = get_screenshots_dir() / f"snapshot_{ts}.jpg"
                if save_image_bgr(snap_path, annotated_frame):
                    print(f"Snapshot saved: {snap_path}")
            elif key == ord("b"):  # 'b' Toggle box
                config.show_boxes = not config.show_boxes
                print(f"Bounding boxes: {'ON' if config.show_boxes else 'OFF'}")
            elif key == ord("l"):  # 'l' Toggle label
                config.show_labels = not config.show_labels
                print(f"Labels: {'ON' if config.show_labels else 'OFF'}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released. Goodbye!")


def main():
    parser = argparse.ArgumentParser(description="MaskGuard Standalone Webcam Demo.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0).")
    parser.add_argument("--model", type=str, default=str(get_default_model_path()), help="Path to .h5 model.")
    parser.add_argument("--conf", type=float, default=0.65, help="Confidence threshold (0.0 - 1.0).")
    parser.add_argument("--scale-factor", type=float, default=1.1, help="Face detector scale factor.")
    parser.add_argument("--min-neighbors", type=int, default=5, help="Face detector min neighbors.")
    args = parser.parse_args()

    run_webcam_demo(
        camera_index=args.camera,
        model_path=args.model,
        confidence_threshold=args.conf,
        scale_factor=args.scale_factor,
        min_neighbors=args.min_neighbors,
    )


if __name__ == "__main__":
    main()
