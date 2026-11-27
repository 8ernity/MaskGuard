"""
Unit tests for image preprocessing, validation, normalization, and augmentation.
"""

from pathlib import Path
import numpy as np
import pytest

from app.utils.image_utils import (
    validate_image_file,
    read_image_bgr,
    save_image_bgr,
    bgr_to_rgb,
    rgb_to_bgr,
    get_detection_color,
    draw_bounding_box,
    COLOR_MASK_BGR,
    COLOR_NOMASK_BGR,
    COLOR_UNCERTAIN_BGR,
)
from training.prepare_dataset import (
    augment_image,
    load_and_preprocess_image,
)
from app.core.mask_classifier import MaskClassifier


def test_bgr_rgb_conversions():
    """Verify BGR to RGB and RGB to BGR conversions preserve channels accurately."""
    original = np.zeros((10, 10, 3), dtype=np.uint8)
    original[:, :] = (255, 128, 0)  # BGR: Blue=255, Green=128, Red=0

    rgb = bgr_to_rgb(original)
    assert rgb[0, 0, 0] == 0    # Red
    assert rgb[0, 0, 1] == 128  # Green
    assert rgb[0, 0, 2] == 255  # Blue

    restored = rgb_to_bgr(rgb)
    np.testing.assert_array_equal(original, restored)


def test_validate_and_save_image(tmp_path: Path):
    """Test safe image writing, validation, and reading with unicode paths."""
    test_file = tmp_path / "valid_image.png"
    canvas = np.zeros((64, 64, 3), dtype=np.uint8)
    canvas[:] = (50, 100, 150)

    # Save
    success = save_image_bgr(test_file, canvas)
    assert success is True
    assert test_file.exists()

    # Validate
    assert validate_image_file(test_file) is True

    # Read
    loaded = read_image_bgr(test_file)
    assert loaded is not None
    assert loaded.shape == (64, 64, 3)

    # Validate corrupted file
    corrupt_file = tmp_path / "corrupt.jpg"
    corrupt_file.write_bytes(b"not a valid image format")
    assert validate_image_file(corrupt_file) is False
    assert read_image_bgr(corrupt_file) is None

    # Validate non-existent file
    assert validate_image_file(tmp_path / "missing.jpg") is False


def test_augment_image_properties():
    """Test that training augmentation maintains dimension and uint8 range."""
    img = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
    aug = augment_image(img)

    assert aug.shape == (128, 128, 3)
    assert aug.dtype == np.uint8
    assert np.all(aug >= 0) and np.all(aug <= 255)


def test_load_and_preprocess_image(tmp_path: Path):
    """Test full image preprocessing pipeline produces normalized float32 tensor."""
    test_img_path = tmp_path / "sample.jpg"
    canvas = np.full((200, 200, 3), 128, dtype=np.uint8)
    save_image_bgr(test_img_path, canvas)

    processed = load_and_preprocess_image(test_img_path, target_size=(128, 128), augment=False)
    assert processed.shape == (128, 128, 3)
    assert processed.dtype == np.float32
    assert processed.min() >= 0.0
    assert processed.max() <= 1.0


def test_mask_classifier_preprocessing():
    """Test MaskClassifier internal crop preprocessing logic."""
    classifier = MaskClassifier(model_path=Path("non_existent_model.h5"))
    crop = np.full((80, 80, 3), 255, dtype=np.uint8)

    proc = classifier.preprocess_face(crop)
    assert proc.shape == (128, 128, 3)
    assert proc.dtype == np.float32
    assert np.isclose(proc.max(), 1.0)

    # Empty crop should raise ValueError
    empty_crop = np.array([], dtype=np.uint8)
    with pytest.raises(ValueError):
        classifier.preprocess_face(empty_crop)


def test_drawing_and_colors():
    """Test bounding box drawing and correct color mappings."""
    assert get_detection_color("MASK") == COLOR_MASK_BGR
    assert get_detection_color("NO MASK") == COLOR_NOMASK_BGR
    assert get_detection_color("MASK", is_uncertain=True) == COLOR_UNCERTAIN_BGR

    canvas = np.zeros((200, 200, 3), dtype=np.uint8)
    annotated = draw_bounding_box(
        canvas,
        box=(20, 20, 60, 60),
        label="MASK",
        confidence=0.95,
        is_uncertain=False,
    )
    assert annotated.shape == (200, 200, 3)
    # Ensure drawing altered pixels inside canvas
    assert np.any(annotated > 0)
