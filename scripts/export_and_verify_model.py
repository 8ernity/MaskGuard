"""
Export and Verification Utility for Phase 7 & Phase 8.
Exports best_weights.keras -> models/mask_detector.h5,
verifies serialization, updates model_metadata.json,
and generates full test evaluation reports.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from app.utils.resource_paths import (
    get_class_mapping_path,
    get_default_model_path,
    get_model_dir,
    get_model_metadata_path,
    get_results_dir,
)
from training.evaluate import plot_confusion_matrix
from training.prepare_dataset import save_class_mapping


def export_and_verify():
    checkpoint_path = get_model_dir() / "best_weights.keras"
    output_h5_path = get_default_model_path()
    
    print(f"[*] Loading best checkpoint from {checkpoint_path}...")
    model = tf.keras.models.load_model(str(checkpoint_path))
    
    print(f"[*] Exporting deliverable model to {output_h5_path}...")
    try:
        model.save(str(output_h5_path))
    except Exception as e:
        model.save(str(output_h5_path), save_format="h5")
        
    print("[*] Verifying reload of exported .h5 model...")
    reloaded_model = tf.keras.models.load_model(str(output_h5_path))
    dummy_input = np.zeros((1, 128, 128, 3), dtype=np.float32)
    dummy_pred = reloaded_model(dummy_input, training=False).numpy()
    assert dummy_pred.shape == (1, 1), f"Unexpected shape {dummy_pred.shape}"
    print(f"[*] Verification SUCCESS! Output shape: {dummy_pred.shape}")
    
    # Save class mapping
    save_class_mapping(get_class_mapping_path())
    
    # Evaluate on clean held-out test split (992 images) from cache
    cache_path = get_results_dir() / "clean_splits_cache.npz"
    assert cache_path.is_file(), "Cache not found!"
    cache = np.load(str(cache_path))
    x_test = cache["x_test"]
    y_test = cache["y_test"]
    
    print(f"\n[*] Evaluating on full held-out test set ({len(x_test)} samples)...")
    # Batch predict
    test_ds = tf.data.Dataset.from_tensor_slices(x_test).batch(64).map(lambda x: tf.cast(x, tf.float32) / 255.0)
    raw_preds = reloaded_model.predict(test_ds, verbose=0).flatten()
    
    y_pred_binary = (raw_preds >= 0.5).astype(int)
    
    acc = accuracy_score(y_test, y_pred_binary)
    prec = precision_score(y_test, y_pred_binary, pos_label=1)
    rec = recall_score(y_test, y_pred_binary, pos_label=1)
    f1 = f1_score(y_test, y_pred_binary, pos_label=1)
    cm = confusion_matrix(y_test, y_pred_binary)
    cr = classification_report(y_test, y_pred_binary, target_names=["with_mask", "without_mask"])
    
    print(f"\n--- Held-Out Test Evaluation Results ---")
    print(f"Accuracy:  {acc*100:.2f}%")
    print(f"Precision: {prec*100:.2f}%")
    print(f"Recall:    {rec*100:.2f}%")
    print(f"F1-Score:  {f1*100:.2f}%")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(cr)
    
    # Export reports
    results_dir = get_results_dir()
    with open(results_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(cr)
        
    metrics_data = {
        "model_name": "MaskGuard_Custom_CNN_v2",
        "total_test_samples": len(y_test),
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "confusion_matrix": {
            "true_positive_mask": int(cm[0, 0]),
            "false_positive_nomask": int(cm[0, 1]),
            "false_negative_mask": int(cm[1, 0]),
            "true_negative_nomask": int(cm[1, 1]),
        },
    }
    with open(results_dir / "evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=4)
        
    # Generate confusion matrix plot
    plot_confusion_matrix(cm, ["with_mask", "without_mask"], results_dir / "confusion_matrix.png")
    
    # Update model metadata
    metadata = {
        "model_name": "MaskGuard_Custom_CNN_v2",
        "architecture": "Custom 4-Block Spatial CNN with Topology Preservation",
        "total_parameters": 508417,
        "input_dimensions": [128, 128, 3],
        "input_shape": [128, 128, 3],
        "color_space": "RGB",
        "normalization": "pixel_val / 255.0 (range [0.0, 1.0])",
        "output_activation": "sigmoid",
        "output_interpretation": {
            "0": "with_mask (probability close to 0)",
            "1": "without_mask (probability close to 1)",
        },
        "class_mapping": {
            "0": "with_mask",
            "1": "without_mask",
        },
        "default_threshold": 0.5,
        "uncertainty_band": [0.40, 0.60],
        "training_metrics": {
            "final_train_accuracy": 0.9879,
            "final_val_accuracy": 0.9688,
            "test_accuracy": round(float(acc), 4),
            "test_precision": round(float(prec), 4),
            "test_recall": round(float(rec), 4),
            "test_f1": round(float(f1), 4),
            "train_samples": len(cache["x_train"]),
            "val_samples": len(cache["x_val"]),
            "test_samples": len(x_test),
        },
    }
    with open(get_model_metadata_path(), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
        
    print("[*] Successfully exported all deliverable artifacts!")


if __name__ == "__main__":
    export_and_verify()
