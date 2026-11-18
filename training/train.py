"""
Training script for the improved custom CNN Face Mask Detector.
Implements strict split isolation, in-memory caching, vectorized data augmentation,
reproducible training loops, and genuine .h5 export.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Configure environment for CPU execution
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import random
import time
from typing import Dict, Tuple

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from app.utils.resource_paths import (
    get_class_mapping_path,
    get_default_model_path,
    get_dataset_dir,
    get_model_dir,
    get_model_metadata_path,
    get_results_dir,
)
from training.model_architecture import (
    build_mask_detector_cnn,
    export_model_metadata,
)
from training.prepare_dataset import (
    load_dataset_splits,
    save_class_mapping,
)


def set_seed(seed: int = 42) -> None:
    """Ensure reproducibility across random, numpy, and tensorflow."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_or_create_cache(
    dataset_dir: Path | str,
    cache_path: Path | str,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[int, float]]:
    """
    Load preprocessed 128x128 uint8 images from disk cache if present,
    or build cache once from clean splits and persist to disk.
    Ensures zero dataset leakage.
    """
    c_path = Path(cache_path)
    if c_path.is_file():
        print(f"[*] Loading clean cached splits from {c_path}...")
        data = np.load(str(c_path), allow_pickle=True)
        x_train = data["x_train"]
        y_train = data["y_train"]
        x_val = data["x_val"]
        y_val = data["y_val"]
        x_test = data["x_test"]
        y_test = data["y_test"]
        class_weights = data["class_weights"].item()
        print(f"[*] Cache loaded: Train={len(x_train)}, Val={len(x_val)}, Test={len(x_test)}")
        return x_train, y_train, x_val, y_val, x_test, y_test, class_weights

    print("[*] Cache not found. Generating clean dataset splits from source images...")
    (
        train_p, train_l,
        val_p, val_l,
        test_p, test_l,
        class_weights,
    ) = load_dataset_splits(dataset_dir, purge_leakage=True, random_state=seed)

    def _load_batch(paths):
        arr = np.empty((len(paths), 128, 128, 3), dtype=np.uint8)
        for i, p in enumerate(paths):
            data = np.fromfile(str(p), dtype=np.uint8)
            img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img_bgr is None:
                arr[i] = 0
                continue
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            arr[i] = cv2.resize(rgb, (128, 128), interpolation=cv2.INTER_AREA)
        return arr

    print(f"[*] Caching {len(train_p)} train images...")
    x_train = _load_batch(train_p)
    y_train = np.array(train_l, dtype=np.float32)

    print(f"[*] Caching {len(val_p)} val images...")
    x_val = _load_batch(val_p)
    y_val = np.array(val_l, dtype=np.float32)

    print(f"[*] Caching {len(test_p)} test images...")
    x_test = _load_batch(test_p)
    y_test = np.array(test_l, dtype=np.float32)

    c_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(c_path),
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        class_weights=class_weights,
    )
    print(f"[*] Saved clean dataset cache to {c_path} ({c_path.stat().st_size / 1e6:.1f} MB)")
    return x_train, y_train, x_val, y_val, x_test, y_test, class_weights


def build_augmented_training_dataset(
    x_train: np.ndarray,
    y_train: np.ndarray,
    batch_size: int = 64,
    seed: int = 42,
) -> tf.data.Dataset:
    """
    Build high-throughput tf.data pipeline with fast vectorized in-graph augmentations:
    - Normalization from uint8 [0, 255] to float32 [0.0, 1.0]
    - Random horizontal flip (50% chance)
    - Random brightness variation (+/- 10%)
    """
    ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    ds = ds.shuffle(buffer_size=min(4000, len(x_train)), seed=seed, reshuffle_each_iteration=True)

    data_augmentation = tf.keras.Sequential(
        [
            layers.Rescaling(1.0 / 255.0),
            layers.RandomFlip("horizontal", seed=seed),
            layers.RandomBrightness(factor=0.10, value_range=(0.0, 1.0), seed=seed),
        ],
        name="data_augmentation",
    )

    def _augment(x, y):
        return data_augmentation(x, training=True), y

    ds = ds.batch(batch_size)
    ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def build_evaluation_dataset(
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    batch_size: int = 64,
) -> tf.data.Dataset:
    """Deterministic evaluation dataset: rescaling uint8 -> float32 [0, 1]."""
    ds = tf.data.Dataset.from_tensor_slices((x_eval, y_eval))
    ds = ds.batch(batch_size)
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def train_model(
    dataset_dir: Path | str,
    epochs: int = 6,
    batch_size: int = 64,
    learning_rate: float = 0.0005,
    seed: int = 42,
    output_model_path: Path | str | None = None,
    full_dataset: bool = False,
) -> Dict[str, any]:
    """
    Train custom CNN model, export .h5 deliverable and metadata, and return training summary.
    """
    set_seed(seed)
    ds_path = Path(dataset_dir)
    model_save_path = Path(output_model_path) if output_model_path else get_default_model_path()
    model_save_path.parent.mkdir(parents=True, exist_ok=True)

    cache_file = get_results_dir() / "clean_splits_cache.npz"
    (
        x_train, y_train,
        x_val, y_val,
        x_test, y_test,
        class_weights,
    ) = get_or_create_cache(ds_path, cache_file, seed=seed)

    print(f"\n[*] Training configuration:")
    print(f"    - Train samples: {len(x_train)}")
    print(f"    - Val samples:   {len(x_val)}")
    print(f"    - Test samples:  {len(x_test)} (STRICTLY HELD OUT)")
    print(f"    - Batch size:    {batch_size}")
    print(f"    - Learning rate: {learning_rate}")
    print(f"    - Epochs:        {epochs}")
    print(f"    - Class weights: {class_weights}")

    train_ds = build_augmented_training_dataset(x_train, y_train, batch_size=batch_size, seed=seed)
    val_ds = build_evaluation_dataset(x_val, y_val, batch_size=batch_size)
    test_ds = build_evaluation_dataset(x_test, y_test, batch_size=batch_size)

    # Build improved CNN architecture
    model = build_mask_detector_cnn(input_shape=(128, 128, 3), learning_rate=learning_rate)
    print("\n--- Model Architecture Summary ---")
    model.summary()

    checkpoint_temp = get_model_dir() / "best_weights.keras"
    callbacks = [
        EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=str(checkpoint_temp),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    print("\nStarting CNN training...")
    t_start = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )
    train_duration = time.time() - t_start
    print(f"\nTraining completed in {train_duration:.1f}s ({train_duration/60:.2f} min).")

    if checkpoint_temp.exists():
        try:
            print(f"Loading best checkpoint from {checkpoint_temp}...")
            model = tf.keras.models.load_model(str(checkpoint_temp))
        except Exception as e:
            print(f"Note loading checkpoint: {e}")

    # Evaluate on held-out test set
    print("\n--- Held-out Test Set Evaluation (Zero Leakage) ---")
    test_results = model.evaluate(test_ds, verbose=1)
    metric_names = model.metrics_names
    test_metrics = dict(zip(metric_names, [float(v) for v in test_results]))
    for k, v in test_metrics.items():
        print(f"  Test {k}: {v:.4f}")

    # Save genuinely trained model as .h5
    print(f"\nExporting trained model to {model_save_path}...")
    try:
        model.save(str(model_save_path))
    except Exception as e:
        model.save(str(model_save_path), save_format="h5")

    # Verify reload of exported .h5 model
    print("Verifying saved .h5 model reload...")
    reloaded_model = tf.keras.models.load_model(str(model_save_path))
    dummy_input = np.zeros((1, 128, 128, 3), dtype=np.float32)
    dummy_pred = reloaded_model(dummy_input, training=False).numpy()
    print(f"Verification successful! Output shape: {dummy_pred.shape}")

    # Save class mapping
    save_class_mapping(get_class_mapping_path())

    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    final_train_acc = float(history.history["accuracy"][-1])
    final_val_acc = float(history.history.get("val_accuracy", [0])[-1])

    export_model_metadata(
        get_model_metadata_path(),
        input_shape=(128, 128, 3),
        training_metrics={
            "final_train_accuracy": round(final_train_acc, 4),
            "final_val_accuracy": round(final_val_acc, 4),
            "test_accuracy": round(test_metrics.get("accuracy", 0.0), 4),
            "test_precision": round(test_metrics.get("precision", 0.0), 4),
            "test_recall": round(test_metrics.get("recall", 0.0), 4),
            "epochs_trained": len(history.history["loss"]),
            "batch_size": batch_size,
            "train_samples": len(x_train),
            "val_samples": len(x_val),
            "test_samples": len(x_test),
            "train_duration_sec": round(train_duration, 1),
        },
    )

    history_file = get_results_dir() / "training_history.json"
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(history_dict, f, indent=4)

    return {
        "status": "success",
        "model_path": str(model_save_path),
        "history": history_dict,
        "test_metrics": test_metrics,
        "train_samples": len(x_train),
        "val_samples": len(x_val),
        "test_samples": len(x_test),
        "final_val_accuracy": final_val_acc,
    }


def main():
    parser = argparse.ArgumentParser(description="Train custom CNN for Face Mask Detection.")
    parser.add_argument("--epochs", type=int, default=6, help="Maximum training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.0005, help="Initial learning rate.")
    parser.add_argument("--dataset-dir", type=str, default=str(get_dataset_dir()), help="Dataset directory.")
    parser.add_argument("--output-model", type=str, default=str(get_default_model_path()), help="Output .h5 model path.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    train_model(
        dataset_dir=args.dataset_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
        output_model_path=args.output_model,
    )


if __name__ == "__main__":
    main()
