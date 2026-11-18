"""
Unit tests for custom CNN architecture creation, forward inference, .h5 serialization, and reload.
"""

from pathlib import Path
import numpy as np
import pytest
import tensorflow as tf

from training.model_architecture import (
    build_mask_detector_cnn,
    get_model_summary_string,
    export_model_metadata,
)


def test_build_cnn_architecture():
    """Verify custom CNN structural layers, tensor shapes, and compilation."""
    model = build_mask_detector_cnn(input_shape=(128, 128, 3), learning_rate=0.001)

    assert model is not None
    assert model.name == "MaskGuard_Custom_CNN"
    assert model.input_shape == (None, 128, 128, 3)
    assert model.output_shape == (None, 1)

    # Check key layer types
    layer_types = [layer.__class__.__name__ for layer in model.layers]
    assert "Conv2D" in layer_types
    assert "BatchNormalization" in layer_types
    assert "MaxPooling2D" in layer_types
    assert "Flatten" in layer_types
    assert "Dense" in layer_types
    assert "Dropout" in layer_types

    # Check summary generation
    summary_str = get_model_summary_string(model)
    assert "MaskGuard_Custom_CNN" in summary_str
    assert "conv1" in summary_str


def test_cnn_forward_pass():
    """Test forward execution with batch of synthetic image tensors."""
    model = build_mask_detector_cnn(input_shape=(128, 128, 3))
    batch_size = 4
    dummy_input = np.random.uniform(0.0, 1.0, (batch_size, 128, 128, 3)).astype(np.float32)

    predictions = model.predict(dummy_input, verbose=0)
    assert predictions.shape == (batch_size, 1)
    # Output activation is sigmoid so values must be strictly within [0.0, 1.0]
    assert np.all(predictions >= 0.0)
    assert np.all(predictions <= 1.0)


def test_model_serialization_and_reload(tmp_path: Path):
    """Test saving model genuinely as .h5 and reloading it successfully."""
    model = build_mask_detector_cnn(input_shape=(128, 128, 3))
    model_path = tmp_path / "test_model.h5"

    # Save to .h5
    model.save(str(model_path))
    assert model_path.exists()
    assert model_path.stat().st_size > 0

    # Reload from .h5
    reloaded_model = tf.keras.models.load_model(str(model_path))
    assert reloaded_model is not None
    assert reloaded_model.output_shape == (None, 1)

    # Compare predictions
    test_tensor = np.random.uniform(0.0, 1.0, (2, 128, 128, 3)).astype(np.float32)
    pred_orig = model.predict(test_tensor, verbose=0)
    pred_reload = reloaded_model.predict(test_tensor, verbose=0)

    np.testing.assert_allclose(pred_orig, pred_reload, rtol=1e-5, atol=1e-5)


def test_export_model_metadata(tmp_path: Path):
    """Test metadata JSON export contains required fields."""
    meta_path = tmp_path / "model_metadata.json"
    export_model_metadata(
        meta_path,
        input_shape=(128, 128, 3),
        training_metrics={"final_train_accuracy": 0.98},
    )

    assert meta_path.exists()
    import json
    with open(meta_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["model_name"] == "MaskGuard_Custom_CNN"
    assert data["input_dimensions"] == [128, 128, 3]
    assert data["color_space"] == "RGB"
    assert "class_mapping" in data
