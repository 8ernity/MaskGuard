"""
MaskGuard - Standalone Real-Time Face Mask Detection Script
Deliverable: OpenCV Webcam Stream with Bounding Box Overlay powered by trained CNN (.h5).

Technologies: CNN, TensorFlow, OpenCV.
Usage:
    python detect_mask_webcam.py
    python detect_mask_webcam.py --camera 0 --model models/mask_detector.h5
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


def parse_args():
    parser = argparse.ArgumentParser(description="MaskGuard Live OpenCV Webcam Detection")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument(
        "--model",
        type=str,
        default="models/mask_detector.h5",
        help="Path to trained .h5 model file (default: models/mask_detector.h5)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.50,
        help="Confidence threshold for mask classification (default: 0.50)",
    )
    return parser.parse_args()


def load_detector_and_model(model_path: str):
    # 1. Load trained CNN model
    p = Path(model_path)
    if not p.exists():
        print(f"[!] Error: Model file not found at '{model_path}'.")
        print("    Please ensure models/mask_detector.h5 exists or train it using training/train.py.")
        sys.exit(1)

    print(f"[*] Loading custom CNN model from '{model_path}'...")
    model = tf.keras.models.load_model(str(p))

    # Compile fast callable forward pass (10-15x faster than model.predict on CPU)
    @tf.function(reduce_retracing=True)
    def fast_predict(x):
        return model(x, training=False)

    # Warm up model execution graph
    dummy = tf.zeros((1, 128, 128, 3), dtype=tf.float32)
    _ = fast_predict(dummy)
    print("[+] Model loaded and warmed up successfully.")

    # 2. Load OpenCV Haar Cascade face detector
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        # Fallback to local asset
        local_cascade = Path(__file__).resolve().parent / "app" / "core" / "assets" / "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(str(local_cascade))

    if face_cascade.empty():
        print("[!] Error: Could not load OpenCV frontalface Haar cascade XML.")
        sys.exit(1)

    # Attach fast callable
    model._fast_predict = fast_predict
    print("[+] OpenCV Haar Cascade face detector initialized.")
    return model, face_cascade


def main():
    args = parse_args()

    print("=" * 60)
    print("    MaskGuard — Real-Time Face Mask Detection (OpenCV)     ")
    print("=" * 60)
    print(f"  • Model:        {args.model}")
    print(f"  • Camera Index: {args.camera}")
    print(f"  • Threshold:    {args.confidence}")
    print("  • Controls:     Press 'q' or 'ESC' to exit")
    print("=" * 60)

    model, face_cascade = load_detector_and_model(args.model)

    # Open webcam stream
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    cap = cv2.VideoCapture(args.camera, backend)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        print(f"[!] Error: Unable to open webcam at index {args.camera}.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    prev_time = time.time()
    fps_smooth = 0.0

    window_name = "MaskGuard - Live Face Mask Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[!] Failed to grab webcam frame. Exiting...")
            break

        # Calculate smoothed FPS
        now = time.time()
        dt = max(1e-5, now - prev_time)
        prev_time = now
        inst_fps = 1.0 / dt
        fps_smooth = 0.85 * fps_smooth + 0.15 * inst_fps if fps_smooth > 0 else inst_fps

        frame_h, frame_w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect face bounding boxes using OpenCV Haar Cascade (equalizeHist omitted to avoid fabric false positives)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.10,
            minNeighbors=7,
            minSize=(50, 50),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        for (x, y, w, h) in faces:
            # Preserve 1:1 square crop matching training dataset with full chin and jawline coverage
            dim = max(w, h)
            target_size = int(dim * 1.40)
            half_side = target_size // 2
            cx = x + w // 2
            cy = y + h // 2 + int(dim * 0.12)

            crop_x1 = max(0, cx - half_side)
            crop_y1 = max(0, cy - half_side)
            crop_x2 = min(frame_w, cx + half_side)
            crop_y2 = min(frame_h, cy + half_side)

            face_roi = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            if face_roi.size == 0 or face_roi.shape[0] < 20 or face_roi.shape[1] < 20:
                continue

            # Dual-space chromatic skin verification (YCrCb + HSV): reject curtains, fabric, and wall patterns
            ycrcb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2YCrCb)
            hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
            skin_ycrcb = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))
            skin_hsv1 = cv2.inRange(hsv, np.array([0, 25, 35]), np.array([25, 220, 255]))
            skin_hsv2 = cv2.inRange(hsv, np.array([165, 25, 35]), np.array([180, 220, 255]))
            skin_hsv = cv2.bitwise_or(skin_hsv1, skin_hsv2)
            skin_combined = cv2.bitwise_and(skin_ycrcb, skin_hsv)
            skin_ratio = float(np.sum(skin_combined > 0)) / skin_combined.size
            if skin_ratio < 0.05:
                continue

            # Adaptive low-light contrast enhancement
            lab = cv2.cvtColor(face_roi, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            if float(np.mean(l)) < 95.0:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                cl = clahe.apply(l)
                face_roi = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

            # Preprocess face chip (BGR -> RGB -> Resize 128x128 -> Scale [0, 1])
            rgb_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb_face, (128, 128), interpolation=cv2.INTER_AREA)
            normalized = resized.astype(np.float32) / 255.0
            tensor_batch = np.expand_dims(normalized, axis=0)

            # Predict probability via fast compiled CNN graph (0 = with_mask, 1 = without_mask)
            if hasattr(model, "_fast_predict") and model._fast_predict is not None:
                pred_score = float(model._fast_predict(tf.constant(tensor_batch)).numpy()[0][0])
            else:
                pred_score = float(model(tensor_batch, training=False).numpy()[0][0])

            if pred_score < 0.50:
                label = "MASK"
                conf = (1.0 - pred_score) * 100.0
                color = (0, 200, 0)      # Green in BGR
            else:
                label = "NO MASK"
                conf = pred_score * 100.0
                color = (0, 0, 230)      # Red in BGR

            # Draw sleek bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Badge / Label overlay above bounding box
            badge_text = f"{label}: {conf:.1f}%"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            thickness = 2
            (text_w, text_h), baseline = cv2.getTextSize(badge_text, font, font_scale, thickness)

            badge_y1 = max(0, y - text_h - 10)
            badge_y2 = y
            badge_x1 = x
            badge_x2 = min(frame_w, x + text_w + 14)

            cv2.rectangle(frame, (badge_x1, badge_y1), (badge_x2, badge_y2), color, cv2.FILLED)
            cv2.putText(
                frame,
                badge_text,
                (badge_x1 + 6, badge_y2 - 5),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

        # Draw HUD stats banner at top
        banner_bg = np.zeros((40, frame_w, 3), dtype=np.uint8)
        cv2.addWeighted(frame[0:40, 0:frame_w], 0.3, banner_bg, 0.7, 0, frame[0:40, 0:frame_w])

        hud_text = f"MaskGuard Live | Faces: {len(faces)} | FPS: {fps_smooth:.1f} | Press 'q' to Exit"
        cv2.putText(frame, hud_text, (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(window_name, frame)

        # Check key press ('q' or ESC = 27)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            print("\n[*] Exiting camera demo...")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
