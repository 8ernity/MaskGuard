"""
MobileNetV2 transfer learning architecture for MaskGuard.
Uses ImageNet-pretrained MobileNetV2 backbone with a custom binary classification head
for 224x224 RGB face crops scaled to [-1.0, 1.0].
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import tensorflow as tf
from tensorflow.keras import layers, metrics, models, optimizers


def build_mobilenetv2_model(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    learning_rate: float = 1e-4,
    freeze_backbone: bool = True,
    weights: str | None = "imagenet",
) -> tf.keras.Model:
    """
    Construct MobileNetV2 transfer-learning binary classifier.

    Expects pre-scaled float32 RGB inputs in [-1.0, 1.0] via
    `preprocess_face_mobilenetv2` / `mobilenet_v2.preprocess_input`.
    """
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights=weights,
    )
    base_model.trainable = not freeze_backbone

    inputs = layers.Input(shape=input_shape, name="input_image")
    # Pass training=False so BatchNormalization inside MobileNetV2 remains in inference mode
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="mobilenet_gap")(x)
    x = layers.Dense(128, activation="relu", name="mobilenet_head_dense")(x)
    x = layers.Dropout(0.3, name="mobilenet_head_dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output_classification")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="MaskGuard_MobileNetV2")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            metrics.Precision(name="precision"),
            metrics.Recall(name="recall"),
        ],
    )
    return model


def unfreeze_mobilenetv2_for_finetuning(
    model: tf.keras.Model,
    unfreeze_last_n: int = 30,
    learning_rate: float = 1e-5,
) -> tf.keras.Model:
    """
    Stage 2 fine-tuning helper:
    Unfreezes only the top `unfreeze_last_n` layers of the MobileNetV2 backbone
    while forcing all BatchNormalization layers to remain non-trainable (inference mode).
    """
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and "mobilenet" in layer.name.lower():
            base_model = layer
            break

    if base_model is None:
        raise ValueError("Could not locate MobileNetV2 backbone inside model.")

    base_model.trainable = True
    total_layers = len(base_model.layers)
    freeze_until = max(0, total_layers - unfreeze_last_n)

    for idx, layer in enumerate(base_model.layers):
        if idx < freeze_until or isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            metrics.Precision(name="precision"),
            metrics.Recall(name="recall"),
        ],
    )
    return model


def compute_file_sha256(file_path: Path | str) -> str:
    """Compute SHA256 digest of a serialized model artifact."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def export_mobilenetv2_metadata(
    output_path: Path | str,
    model_path: Path | str | None = None,
    total_parameters: int | None = None,
    trainable_parameters: int | None = None,
    training_config: Dict[str, Any] | None = None,
    dataset_info: Dict[str, Any] | None = None,
    evaluation_provenance: Dict[str, Any] | None = None,
) -> None:
    """Export comprehensive MobileNetV2 metadata JSON."""
    sha256 = ""
    size_bytes = 0
    if model_path and Path(model_path).is_file():
        p = Path(model_path)
        sha256 = compute_file_sha256(p)
        size_bytes = p.stat().st_size

    metadata = {
        "model_id": "mobilenetv2",
        "display_name": "MobileNetV2 (Transfer Learning)",
        "model_name": "MaskGuard_MobileNetV2",
        "architecture": "MobileNetV2 (ImageNet Pretrained) + GAP + Dense(128) + Dropout(0.3) + Sigmoid",
        "backbone": "MobileNetV2 (alpha=1.0, weights=imagenet, include_top=False)",
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "model_sha256": sha256,
        "model_size_bytes": size_bytes,
        "input_dimensions": [224, 224, 3],
        "input_shape": [224, 224, 3],
        "color_space": "RGB",
        "preprocessing_method": "tf.keras.applications.mobilenet_v2.preprocess_input: (pixel_val / 127.5) - 1.0 -> [-1.0, 1.0]",
        "output_activation": "sigmoid",
        "class_mapping": {
            "0": "with_mask",
            "1": "without_mask",
        },
        "output_interpretation": {
            "0": "with_mask (probability close to 0)",
            "1": "without_mask (probability close to 1)",
        },
        "default_threshold": 0.5,
        "training_configuration": training_config or {},
        "dataset_information": dataset_info or {},
        "evaluation_provenance": evaluation_provenance or {},
    }

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
