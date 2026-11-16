# AI Face Mask Detection System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete, production-grade AI Face Mask Detection desktop application featuring a custom TensorFlow/Keras CNN, OpenCV face detection and video processing, and a modern emerald-and-charcoal PySide6 dashboard.

**Architecture:** A decoupled computer-vision architecture where OpenCV handles face localization across webcam feeds, images, and video files, while a custom 3-block Convolutional Neural Network (128x128x3 input) classifies cropped face patches as `MASK`, `NO MASK`, or `UNCERTAIN`. A multi-threaded PySide6 application manages asynchronous camera/video workers to deliver smooth 30+ FPS UI rendering with real-time statistics, analytics, and configurable settings.

**Tech Stack:** Python 3.11, TensorFlow 2.x / Keras, OpenCV (cv2), PySide6 (Qt6), NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, pytest.

**Spec:** MASTER PROMPT — AI FACE MASK DETECTION SYSTEM

## Global Constraints

- Language: Python 3.11
- Strict Exclusion: Do NOT use YOLO, Ultralytics, or pretrained object-detection frameworks.
- Deep Learning: Custom CNN built from scratch with TensorFlow/Keras; input size 128x128x3.
- Model Deliverable: Genuine trained `.h5` model file saved to `models/mask_detector.h5` with `class_mapping.json` and `model_metadata.json`.
- Computer Vision: OpenCV used for face localization, webcam streaming, frame preprocessing, and video I/O.
- Desktop UI: PySide6 (Qt for Python), dark theme by default (charcoal #0D1117/#161B22, emerald green #10B981).
- Multi-face support: Detect and classify multiple faces in a single frame.
- Bounding Box Spec: Green for MASK, Red for NO MASK, Amber for UNCERTAIN (< threshold).
- Standalone CLI: `scripts/webcam_demo.py` runnable without GUI.
- Evaluation: Genuine metrics (accuracy, precision, recall, F1, confusion matrix, loss/accuracy curves) saved to `results/`.
- Offline Capability: Self-contained, offline execution with local file storage.

## Review Focus

1. **Face Detector & Crop Hygiene:** OpenCV Haar Cascade face bounding box must be validated against image boundaries with safe padding to avoid empty crops or index-out-of-bound errors.
2. **Channel Order & Normalization Consistency:** RGB vs. BGR conversion must match precisely across training, image analysis, video analysis, and real-time webcam inference (OpenCV produces BGR; model expects RGB normalized to [0, 1]).
3. **Threading & Qt Stability:** Camera capture and video encoding must run inside dedicated `QThread` workers so the PySide6 UI never stutters, blocks, or crashes on camera disconnects or app closure.
4. **Model Serialization & Keras 3 Compatibility:** Ensure `.h5` model format export and reload functions work reliably across TensorFlow/Keras versions with proper custom objects handling if needed.
5. **Session Counting vs. Re-detection:** Distinguish between current-frame face counts and cumulative session detections without inflating counts on consecutive video frames.

---

## Tasks

### Task 1: Environment & Project Foundation
- [ ] Create repository directory structure matching specification (`app/`, `app/ui/`, `app/core/`, `app/utils/`, `training/`, `scripts/`, `models/`, `dataset/`, `results/`, `tests/`).
- [ ] Generate comprehensive `requirements.txt` and `.gitignore`.
- [ ] Implement `app/utils/resource_paths.py` for cross-platform, deterministic path resolution.
- [ ] Implement `app/utils/config.py` for persistent user configuration (`settings.json`).
- [ ] Implement `app/utils/image_utils.py` for RGB/BGR conversions, Qt QImage conversions, and bounding-box drawing.

### Task 2: Custom CNN Architecture & Dataset Pipeline
- [ ] Implement `training/model_architecture.py` creating the custom CNN (128x128x3 -> 3 Conv/BN/ReLU/MaxPool blocks -> GAP -> Dense(128) -> Dropout(0.5) -> Dense(1, Sigmoid)).
- [ ] Implement `training/prepare_dataset.py` with image validation, reproducible train/val/test splits, data augmentation on train split only, class imbalance weighting, and dataset statistics reporting.
- [ ] Provide a dataset bootstrap/verification tool within `prepare_dataset.py` that can synthesize or verify benchmark face samples so the pipeline can execute end-to-end.
- [ ] Implement `training/train.py` with Adam optimizer, binary crossentropy, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, genuine `.h5` export, `class_mapping.json`, and `model_metadata.json`.
- [ ] Implement `training/evaluate.py` to evaluate held-out test data, generating accuracy, precision, recall, F1, classification report, confusion matrix heatmap, and loss/accuracy curves in `results/`.

### Task 3: Core Computer Vision & Inference Engine
- [ ] Implement `app/core/face_detector.py` wrapping OpenCV face detection (Haar Cascade with dynamic boundary padding and parameter tuning: scaleFactor, minNeighbors, minSize).
- [ ] Implement `app/core/mask_classifier.py` for model loading (`.h5`), face patch preprocessing (128x128, RGB, normalized [0, 1]), batch inference, confidence calculation, and classification (`MASK`, `NO MASK`, `UNCERTAIN`).
- [ ] Implement `app/core/inference_pipeline.py` orchestrating detection + classification, annotation with colored bounding boxes and high-contrast badges, and statistics aggregation.
- [ ] Implement `app/core/camera_worker.py` with `QThread` for asynchronous webcam capture, error recovery, FPS calculation, and graceful shutdown.
- [ ] Implement `app/core/video_processor.py` for batch video processing (MP4/AVI) with frame progress signaling, preview frames, and cancellation support.

### Task 4: Standalone Scripts & Automated Tests
- [ ] Implement `scripts/webcam_demo.py` for standalone OpenCV CLI webcam detection without GUI.
- [ ] Implement `tests/test_preprocessing.py` for validation, resizing, normalization, and shape assertions.
- [ ] Implement `tests/test_model.py` for model creation, compilation, output shapes, `.h5` export, and reload verification.
- [ ] Implement `tests/test_inference.py` for face detection, mask classification, multi-face processing, empty frame handling, and thresholding logic.
- [ ] Execute `pytest` and verify 100% test passing rate.

### Task 5: PySide6 Desktop User Interface
- [ ] Implement custom styles, theme palettes (Dark Emerald Charcoal & Light Mode), and reusable UI card/button components in `app/ui/`.
- [ ] Implement `app/ui/dashboard.py` with model readiness badge, camera status, session statistics cards, CNN specs, and quick-action navigation.
- [ ] Implement `app/ui/live_detection.py` with real-time video viewport, start/stop controls, camera selection, live FPS/counts HUD, confidence slider, and screenshot tool.
- [ ] Implement `app/ui/image_analysis.py` supporting drag-and-drop / file dialog for images, multi-face annotation, results detail table, and image export.
- [ ] Implement `app/ui/video_analysis.py` for video upload, sequential frame processing preview, progress bar, cancellation, and annotated video export.
- [ ] Implement `app/ui/model_analytics.py` displaying model accuracy/precision/recall/F1 cards, confusion matrix plot, training curves plot, and architecture summary.
- [ ] Implement `app/ui/settings.py` for camera settings, confidence thresholds, face detection parameters, bounding box toggles, theme switcher, model path selection, and output directory selection.
- [ ] Implement `app/ui/main_window.py` and `app/main.py` integrating all navigation views, status bar, and clean shutdown handling.

### Task 6: Execution, Verification, and Documentation
- [ ] Execute model training and evaluation pipeline to produce genuine `mask_detector.h5`, `class_mapping.json`, `model_metadata.json`, and evaluation plots in `results/`.
- [ ] Verify standalone webcam script and GUI startup in headless/windowed mode.
- [ ] Create detailed `README.md` with complete architecture breakdown, academic viva presentation guide, installation and usage instructions, and screenshot placeholders.
- [ ] Verify all 16 acceptance criteria and report final status.
