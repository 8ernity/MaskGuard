"""
Integration tests for FaceDetector, MaskClassifier, and InferencePipeline.
Verifies multi-face processing, boundary padding, uncertainty thresholding, and tracker counting.
"""

from pathlib import Path
import numpy as np
import pytest
import cv2

from app.core.face_detector import FaceDetector
from app.core.mask_classifier import MaskClassifier, FaceDetection
from app.core.inference_pipeline import InferencePipeline, SimpleFaceTracker
from app.utils.config import AppConfig
from training.model_architecture import build_mask_detector_cnn


def test_face_detector_initialization():
    """Test detector instantiation and parameter updating."""
    detector = FaceDetector(scale_factor=1.1, min_neighbors=5, min_size=(30, 30), padding=0.15)
    assert detector.classifier is not None

    detector.update_parameters(scale_factor=1.2, min_neighbors=4, min_size=40, padding=0.20)
    assert detector.scale_factor == 1.2
    assert detector.min_neighbors == 4
    assert detector.min_size == (40, 40)
    assert detector.padding == 0.20


def test_face_detector_expand_box():
    """Test bounding box expansion logic and boundary clamping."""
    detector = FaceDetector(padding=0.10)
    frame_shape = (480, 640, 3)

    # Centered box
    box = (100, 100, 50, 50)
    expanded = detector.expand_box(box, frame_shape)
    ex, ey, ew, eh = expanded
    assert ex < 100
    assert ey < 100
    assert ew > 50
    assert eh > 50

    # Near top-left boundary
    boundary_box = (2, 2, 40, 40)
    exp_boundary = detector.expand_box(boundary_box, frame_shape)
    assert exp_boundary[0] >= 0
    assert exp_boundary[1] >= 0

    # Near bottom-right boundary
    bottom_box = (620, 460, 40, 40)
    exp_bottom = detector.expand_box(bottom_box, frame_shape)
    assert exp_bottom[0] + exp_bottom[2] <= 640
    assert exp_bottom[1] + exp_bottom[3] <= 480


def test_face_detector_empty_frame():
    """Test detector behavior on empty or solid frames."""
    detector = FaceDetector()
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    assert detector.detect(empty) == []

    # Solid black frame without faces
    solid = np.zeros((200, 200, 3), dtype=np.uint8)
    faces = detector.detect(solid)
    assert isinstance(faces, list)
    assert len(faces) == 0


def test_mask_classifier_calibration(tmp_path: Path):
    """Test prediction calibration and uncertainty band logic."""
    model = build_mask_detector_cnn()
    model_path = tmp_path / "calib_model.h5"
    model.save(str(model_path))

    classifier = MaskClassifier(
        model_path=model_path,
        confidence_threshold=0.70,
        uncertain_band=0.15,
    )
    assert classifier.is_ready() is True

    # Test dummy crops
    crop1 = np.full((100, 100, 3), 50, dtype=np.uint8)
    crop2 = np.full((100, 100, 3), 200, dtype=np.uint8)

    detections = classifier.classify_crops([crop1, crop2], [(10, 10, 50, 50), (70, 70, 50, 50)])
    assert len(detections) == 2
    for det in detections:
        assert isinstance(det, FaceDetection)
        assert det.label in ("MASK", "NO MASK", "UNCERTAIN")
        assert 0.0 <= det.confidence <= 1.0


def test_simple_face_tracker():
    """Test that face tracker tracks unique persons without duplicate frame counting."""
    tracker = SimpleFaceTracker(max_distance=50.0, max_disappeared=5)

    # Frame 1: 1 person at (100, 100)
    count1 = tracker.update([(100, 100, 40, 40)])
    assert count1 == 1

    # Frame 2: same person moved slightly to (103, 102)
    count2 = tracker.update([(103, 102, 40, 40)])
    assert count2 == 1  # Should NOT increment unique face count

    # Frame 3: second person enters at (300, 300)
    count3 = tracker.update([(104, 103, 40, 40), (300, 300, 40, 40)])
    assert count3 == 2

    # Frame 4: empty frame (people temporarily obscured)
    count4 = tracker.update([])
    assert count4 == 2


def test_inference_pipeline_execution(tmp_path: Path):
    """Test full inference pipeline execution with frame annotation and statistics."""
    model = build_mask_detector_cnn()
    model_path = tmp_path / "pipeline_model.h5"
    model.save(str(model_path))

    config = AppConfig(model_path=str(model_path))
    pipeline = InferencePipeline(config=config)

    # Frame without faces
    blank_frame = np.zeros((300, 300, 3), dtype=np.uint8)
    annotated, detections, stats = pipeline.process_frame(blank_frame)

    assert annotated.shape == (300, 300, 3)
    assert len(detections) == 0
    assert stats["faces_in_frame"] == 0
    assert stats["fps"] > 0
