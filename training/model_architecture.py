"""
Custom Convolutional Neural Network (CNN) architecture for face mask classification.
Constructed using TensorFlow / Keras with strict adherence to architectural specifications.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, metrics


def build_mask_detector_cnn(
    input_shape: Tuple[int, int, int] = (128, 128, 3),
    learning_rate: float = 0.001,
) -> tf.keras.Model:
    """
    Construct and compile the custom 3-block CNN for face mask detection.

    Architecture:
      Input (128, 128, 3)
      Block 1: Conv2D(32, 3x3) -> BatchNormalization -> ReLU -> MaxPool2D(2x2)
      Block 2: Conv2D(64, 3x3) -> BatchNormalization -> ReLU -> MaxPool2D(2x2)
      Block 3: Conv2D(128, 3x3) -> BatchNormalization -> ReLU -> MaxPool2D(2x2)
      GlobalAveragePooling2D
      Dense(128, ReLU) -> Dropout(0.5)
      Dense(1, Sigmoid) [Output: 0 = with_mask, 1 = without_mask]
    """
    model = models.Sequential(
        [
            layers.Input(shape=input_shape, name="input_image"),

            # Conv Block 1
            layers.Conv2D(32, (3, 3), padding="same", use_bias=False, name="conv1"),
            layers.BatchNormalization(name="bn1"),
            layers.Activation("relu", name="relu1"),
            layers.MaxPooling2D(pool_size=(2, 2), name="pool1"),

            # Conv Block 2
            layers.Conv2D(64, (3, 3), padding="same", use_bias=False, name="conv2"),
            layers.BatchNormalization(name="bn2"),
            layers.Activation("relu", name="relu2"),
            layers.MaxPooling2D(pool_size=(2, 2), name="pool2"),

            # Conv Block 3
            layers.Conv2D(128, (3, 3), padding="same", use_bias=False, name="conv3"),
            layers.BatchNormalization(name="bn3"),
            layers.Activation("relu", name="relu3"),
            layers.MaxPooling2D(pool_size=(2, 2), name="pool3"),

            # Classifier Head
            layers.GlobalAveragePooling2D(name="gap"),
            layers.Dense(128, activation="relu", name="dense128"),
            layers.Dropout(0.5, name="dropout0.5"),
            layers.Dense(1, activation="sigmoid", name="output_classification"),
        ],
        name="MaskGuard_Custom_CNN",
    )

    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            metrics.Precision(name="precision"),
            metrics.Recall(name="recall"),
        ],
    )
    return model


def get_model_summary_string(model: tf.keras.Model) -> str:
    """Capture model.summary() into a formatted string."""
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + "\n"))
    return stream.getvalue()


def export_model_metadata(
    output_path: Path | str,
    input_shape: Tuple[int, int, int] = (128, 128, 3),
    class_mapping: dict | None = None,
    training_metrics: dict | None = None,
) -> None:
    """Save model metadata JSON for inference and documentation."""
    if class_mapping is None:
        class_mapping = {"0": "with_mask", "1": "without_mask"}

    metadata = {
        "model_name": "MaskGuard_Custom_CNN",
        "input_dimensions": list(input_shape),
        "color_space": "RGB",
        "normalization": "pixel_val / 255.0 (range [0.0, 1.0])",
        "output_activation": "sigmoid",
        "output_interpretation": {
            "0": "with_mask (probability close to 0)",
            "1": "without_mask (probability close to 1)",
        },
        "class_mapping": class_mapping,
        "default_threshold": 0.5,
        "uncertainty_band": [0.40, 0.60],
        "training_metrics": training_metrics or {},
    }

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
