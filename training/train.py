"""
Training script for the custom CNN Face Mask Detector.
Implements reproducible data loading, callbacks, training loops, evaluation, and genuine .h5 export.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Configure environment for optimal multi-threaded CPU execution
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"
# Let TensorFlow and NumPy use multi-core threading
os.environ.pop("OPENBLAS_NUM_THREADS", None)
os.environ.pop("OMP_NUM_THREADS", None)

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import random
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from app.utils.resource_paths import (
    get_dataset_dir,
    get_default_model_path,
    get_model_dir,
    get_results_dir,
    get_class_mapping_path,
    get_model_metadata_path,
)
from training.model_architecture import (
    build_mask_detector_cnn,
    export_model_metadata,
)
from training.prepare_dataset import (
    load_dataset_splits,
    create_dataset_splits,
    load_and_preprocess_image,
    save_class_mapping,
    generate_synthetic_starter_dataset,
    validate_and_collect_images,
)


def set_seed(seed: int = 42) -> None:
    """Ensure reproducibility across random, numpy, and tensorflow."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


class DataSequence(tf.keras.utils.Sequence):
    """
    Keras Sequence for memory-efficient batch data generation with training augmentations.
    """

    def __init__(
        self,
        image_paths: List[Path],
        labels: List[int],
        batch_size: int = 32,
        target_size: Tuple[int, int] = (128, 128),
        augment: bool = False,
        shuffle: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_paths = list(image_paths)
        self.labels = np.array(labels, dtype=np.float32)
        self.batch_size = batch_size
        self.target_size = target_size
        self.augment = augment
        self.shuffle = shuffle
        self.indices = np.arange(len(self.image_paths))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self) -> int:
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        batch_indices = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_images = []
        batch_labels = self.labels[batch_indices]

        for i in batch_indices:
            img = load_and_preprocess_image(
                self.image_paths[i],
                target_size=self.target_size,
                augment=self.augment,
            )
            batch_images.append(img)

        return np.array(batch_images, dtype=np.float32), batch_labels

    def on_epoch_end(self) -> None:
        if self.shuffle:
            np.random.shuffle(self.indices)
        import gc
        gc.collect()


def train_model(
    dataset_dir: Path | str,
    epochs: int = 25,
    batch_size: int = 32,
    learning_rate: float = 0.001,
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

    # Check if dataset is populated; if empty, create starter dataset
    paths, labels, counts = validate_and_collect_images(ds_path)
    if len(paths) < 10:
        print(f"Dataset in {ds_path} contains fewer than 10 valid images. Synthesizing starter dataset...")
        generate_synthetic_starter_dataset(ds_path, samples_per_class=100, seed=seed)

    # Load splits (official splits if available, otherwise stratified split)
    (
        train_paths, train_labels,
        val_paths, val_labels,
        test_paths, test_labels,
        class_weights,
    ) = load_dataset_splits(
        ds_path, test_size=0.15, val_size=0.15, random_state=seed, full_dataset=full_dataset
    )

    print(f"Dataset split: Train={len(train_paths)}, Val={len(val_paths)}, Test={len(test_paths)}")
    print(f"Computed class weights: {class_weights}")

    # Build sequences
    train_gen = DataSequence(
        train_paths, train_labels, batch_size=batch_size, augment=True, shuffle=True
    )
    val_gen = DataSequence(
        val_paths, val_labels, batch_size=batch_size, augment=False, shuffle=False
    )

    # Build CNN
    model = build_mask_detector_cnn(input_shape=(128, 128, 3), learning_rate=learning_rate)
    print("\n--- Model Architecture Summary ---")
    model.summary()

    # Callbacks
    checkpoint_temp = get_model_dir() / "best_weights.keras"
    callbacks = [
        EarlyStopping(
            monitor="val_accuracy",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="loss",
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
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    if checkpoint_temp.exists():
        try:
            print(f"Loading best checkpoint from {checkpoint_temp}...")
            model = tf.keras.models.load_model(str(checkpoint_temp))
        except Exception as e:
            print(f"Note loading checkpoint: {e}")

    # Save genuinely trained model as .h5
    print(f"\nExporting trained model to {model_save_path}...")
    try:
        model.save(str(model_save_path))
    except Exception as e:
        # Fallback to keras format or save_weights if legacy h5 issues occur
        print(f"Direct .h5 save note ({e}), saving via export format...")
        model.save(str(model_save_path), save_format="h5")

    # Verify reload of exported .h5 model
    print("Verifying saved .h5 model reload...")
    reloaded_model = tf.keras.models.load_model(str(model_save_path))
    dummy_input = np.zeros((1, 128, 128, 3), dtype=np.float32)
    dummy_pred = reloaded_model.predict(dummy_input, verbose=0)
    print(f"Verification successful! Output shape: {dummy_pred.shape}")

    # Save class mapping and metadata
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
            "epochs_trained": len(history.history["loss"]),
            "batch_size": batch_size,
            "train_samples": len(train_paths),
            "val_samples": len(val_paths),
            "test_samples": len(test_paths),
        },
    )

    # Persist training history JSON for evaluation and UI analytics
    history_file = get_results_dir() / "training_history.json"
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(history_dict, f, indent=4)

    # Clean up temp checkpoint if desired or keep it as backup
    return {
        "status": "success",
        "model_path": str(model_save_path),
        "history": history_dict,
        "train_samples": len(train_paths),
        "val_samples": len(val_paths),
        "test_samples": len(test_paths),
        "final_val_accuracy": final_val_acc,
    }


def main():
    parser = argparse.ArgumentParser(description="Train custom CNN for Face Mask Detection.")
    parser.add_argument("--epochs", type=int, default=20, help="Maximum training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate.")
    parser.add_argument("--dataset-dir", type=str, default=str(get_dataset_dir()), help="Dataset directory.")
    parser.add_argument("--output-model", type=str, default=str(get_default_model_path()), help="Output .h5 model path.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--full-dataset", action="store_true", help="Train on all 11,792 images across Train, Val, and Test.")
    args = parser.parse_args()

    train_model(
        dataset_dir=args.dataset_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
        output_model_path=args.output_model,
        full_dataset=args.full_dataset,
    )


if __name__ == "__main__":
    main()
