"""
Two-Stage Transfer Learning Training Pipeline for MobileNetV2 in MaskGuard.
- Stage 1: Feature Extraction (ImageNet backbone frozen, train custom classification head)
- Stage 2: Controlled Fine-Tuning (unfreeze upper backbone layers, BatchNorm strictly in inference mode)

Uses memory-efficient tf.data streaming from verified file paths (zero large RAM allocation)
and saves all artifacts independently under models/mobilenetv2/.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import gc
import json
import random
import time
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from app.utils.resource_paths import (
    get_dataset_dir,
    get_mobilenetv2_class_mapping_path,
    get_mobilenetv2_dir,
    get_mobilenetv2_evaluation_path,
    get_mobilenetv2_metadata_path,
    get_mobilenetv2_model_path,
)
from training.evaluate import plot_confusion_matrix, plot_training_history
from training.mobilenetv2_architecture import (
    build_mobilenetv2_model,
    export_mobilenetv2_metadata,
    unfreeze_mobilenetv2_for_finetuning,
)
from training.prepare_dataset import load_dataset_splits, save_class_mapping


def set_seed(seed: int = 42) -> None:
    """Set deterministic seeds across Python, NumPy, and TensorFlow."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _decode_and_resize_tf(path_tensor: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Read image file, decode RGB, and resize to 224x224 in [0.0, 255.0] float32."""
    raw = tf.io.read_file(path_tensor)
    img = tf.image.decode_image(raw, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    resized = tf.image.resize(img, [224, 224], method=tf.image.ResizeMethod.AREA)
    return tf.cast(resized, tf.float32), tf.cast(label, tf.float32)


def build_mobilenetv2_train_dataset(
    train_paths: List[Path],
    train_labels: List[int],
    batch_size: int = 32,
    seed: int = 42,
) -> tf.data.Dataset:
    """
    Build memory-efficient training tf.data pipeline with realistic training-only augmentations
    followed by exact MobileNetV2 preprocessing ([-1.0, 1.0]).
    Never normalizes twice.
    """
    str_paths = [str(p) for p in train_paths]
    ds = tf.data.Dataset.from_tensor_slices((str_paths, train_labels))
    ds = ds.shuffle(buffer_size=min(2000, len(str_paths)), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(_decode_and_resize_tf, num_parallel_calls=2)

    augmenter = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal", seed=seed),
            layers.RandomBrightness(factor=0.10, value_range=(0.0, 255.0), seed=seed),
            layers.RandomContrast(factor=0.10, seed=seed),
        ],
        name="mobilenetv2_train_augmentation",
    )

    def _prep_train(x_float_255, y):
        x_aug = augmenter(x_float_255, training=True)
        x_norm = tf.keras.applications.mobilenet_v2.preprocess_input(x_aug)
        return x_norm, y

    ds = ds.batch(batch_size)
    ds = ds.map(_prep_train, num_parallel_calls=2)
    ds = ds.prefetch(1)
    return ds


def build_mobilenetv2_eval_dataset(
    eval_paths: List[Path],
    eval_labels: List[int],
    batch_size: int = 16,
) -> tf.data.Dataset:
    """Deterministic validation/test dataset with canonical MobileNetV2 scaling [-1.0, 1.0]."""
    str_paths = [str(p) for p in eval_paths]
    ds = tf.data.Dataset.from_tensor_slices((str_paths, eval_labels))
    ds = ds.map(_decode_and_resize_tf, num_parallel_calls=2)
    ds = ds.batch(batch_size)
    ds = ds.map(
        lambda x, y: (tf.keras.applications.mobilenet_v2.preprocess_input(x), y),
        num_parallel_calls=2,
    )
    ds = ds.prefetch(1)
    return ds


def train_mobilenetv2(
    dataset_dir: Path | str | None = None,
    stage1_epochs: int = 4,
    stage2_epochs: int = 4,
    batch_size: int = 32,
    stage1_lr: float = 1e-4,
    stage2_lr: float = 1e-5,
    unfreeze_last_n: int = 30,
    seed: int = 42,
    output_model_path: Path | str | None = None,
) -> Dict[str, Any]:
    """
    Execute Stage 1 (feature extraction) + Stage 2 (controlled fine-tuning) for MobileNetV2,
    export verified .h5 deliverable, evaluate on independent held-out test split, and save artifacts.
    """
    set_seed(seed)
    ds_path = Path(dataset_dir) if dataset_dir else get_dataset_dir()
    out_dir = get_mobilenetv2_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(output_model_path) if output_model_path else get_mobilenetv2_model_path()
    model_path.parent.mkdir(parents=True, exist_ok=True)

    if not (ds_path / "Train").exists() and not (ds_path / "with_mask").exists():
        raise FileNotFoundError(
            f"Kaggle dataset not found at {ds_path}. Download it explicitly with: "
            "python training/prepare_dataset.py --download"
        )

    print("[*] Loading leakage-audited Kaggle 12K dataset splits (streaming 224x224 RGB for MobileNetV2)...")
    (
        train_p, train_l,
        val_p, val_l,
        test_p, test_l,
        class_weights,
    ) = load_dataset_splits(ds_path, purge_leakage=True, random_state=seed)

    print(f"[*] Splits verified: Train={len(train_p)}, Val={len(val_p)}, Test={len(test_p)}")

    train_ds = build_mobilenetv2_train_dataset(train_p, train_l, batch_size=batch_size, seed=seed)
    val_ds = build_mobilenetv2_eval_dataset(val_p, val_l, batch_size=batch_size)

    # --- STAGE 1: FEATURE EXTRACTION ---
    print("\n============================================================")
    print(" STAGE 1: MobileNetV2 Feature Extraction (Backbone Frozen)")
    print("============================================================")
    model = build_mobilenetv2_model(
        input_shape=(224, 224, 3),
        learning_rate=stage1_lr,
        freeze_backbone=True,
        weights="imagenet",
    )

    ckpt_stage1 = out_dir / "_tmp_stage1_best.keras"
    callbacks_s1 = [
        EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1),
        ModelCheckpoint(filepath=str(ckpt_stage1), monitor="val_accuracy", save_best_only=True, verbose=1),
    ]

    t_s1 = time.time()
    hist_s1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=stage1_epochs,
        class_weight=class_weights,
        callbacks=callbacks_s1,
        verbose=1,
    )
    s1_time = time.time() - t_s1
    if ckpt_stage1.is_file():
        model = tf.keras.models.load_model(str(ckpt_stage1))
    gc.collect()
    s1_best_val_acc = float(max(hist_s1.history.get("val_accuracy", [0.0])))
    print(f"[*] Stage 1 complete in {s1_time:.1f}s | Best Val Acc: {s1_best_val_acc:.4f}")

    # --- STAGE 2: CONTROLLED FINE-TUNING ---
    print("\n============================================================")
    print(f" STAGE 2: Fine-Tuning Top {unfreeze_last_n} Layers (BN Inference Mode)")
    print("============================================================")
    model = unfreeze_mobilenetv2_for_finetuning(
        model,
        unfreeze_last_n=unfreeze_last_n,
        learning_rate=stage2_lr,
    )

    ckpt_stage2 = out_dir / "_tmp_stage2_best.keras"
    callbacks_s2 = [
        EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1),
        ModelCheckpoint(filepath=str(ckpt_stage2), monitor="val_accuracy", save_best_only=True, verbose=1),
    ]

    t_s2 = time.time()
    hist_s2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=stage2_epochs,
        class_weight=class_weights,
        callbacks=callbacks_s2,
        verbose=1,
    )
    s2_time = time.time() - t_s2
    s2_best_val_acc = float(max(hist_s2.history.get("val_accuracy", [0.0])))
    print(f"[*] Stage 2 complete in {s2_time:.1f}s | Best Val Acc: {s2_best_val_acc:.4f}")

    if s2_best_val_acc >= s1_best_val_acc and ckpt_stage2.is_file():
        model = tf.keras.models.load_model(str(ckpt_stage2))
        selected_stage = "stage2_finetuned"
    elif ckpt_stage1.is_file():
        model = tf.keras.models.load_model(str(ckpt_stage1))
        selected_stage = "stage1_feature_extraction"
    else:
        selected_stage = "final_weights"
    print(f"[*] Selected best checkpoint ({selected_stage}) for .h5 export.")

    try:
        model.save(str(model_path))
    except Exception:
        model.save(str(model_path), save_format="h5")

    # Reload verification
    reloaded = tf.keras.models.load_model(str(model_path))
    sample_batch = np.zeros((4, 224, 224, 3), dtype=np.float32)
    for idx, sp in enumerate(test_p[:4]):
        data = np.fromfile(str(sp), dtype=np.uint8)
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        sample_batch[idx] = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32)
    sample_x = tf.keras.applications.mobilenet_v2.preprocess_input(sample_batch)
    orig_pred = model(sample_x, training=False).numpy()
    reload_pred = reloaded(sample_x, training=False).numpy()
    np.testing.assert_allclose(orig_pred, reload_pred, rtol=1e-5, atol=1e-5)
    print(f"[*] Verified .h5 reload numerical fidelity at {model_path}")

    ckpt_stage1.unlink(missing_ok=True)
    ckpt_stage2.unlink(missing_ok=True)

    save_class_mapping(get_mobilenetv2_class_mapping_path())

    combined_history: Dict[str, List[float]] = {}
    for k in hist_s1.history.keys():
        combined_history[k] = [float(v) for v in hist_s1.history[k]] + [
            float(v) for v in hist_s2.history.get(k, [])
        ]

    hist_json_path = out_dir / "training_history.json"
    with open(hist_json_path, "w", encoding="utf-8") as f:
        json.dump(combined_history, f, indent=4)
    plot_training_history(hist_json_path, out_dir / "training_curves.png")

    # --- INDEPENDENT HELD-OUT TEST EVALUATION ---
    print("\n--- Independent Held-Out Test Evaluation (Zero Leakage, n=992) ---")
    test_ds = build_mobilenetv2_eval_dataset(test_p, test_l, batch_size=batch_size)
    raw_preds = reloaded.predict(test_ds, verbose=0).flatten()
    y_test = np.array(test_l, dtype=np.int32)
    y_pred = (raw_preds >= 0.5).astype(np.int32)

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    cm = confusion_matrix(y_test, y_pred)
    tn_withmask, fp_nomask_pred, fn_mask_pred, tp_nomask = (
        int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    )
    false_mask_rate_on_unmasked = fn_mask_pred / max(1, int(np.sum(y_test == 1)))

    # Measure single-crop CPU inference latency (after warmup)
    _ = reloaded(sample_x[:1], training=False).numpy()
    latencies_ms = []
    for i in range(50):
        t0 = time.perf_counter()
        _ = reloaded(sample_x[i % 4 : (i % 4) + 1], training=False).numpy()
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
    avg_latency_ms = float(np.mean(latencies_ms))
    observed_fps = 1000.0 / max(0.001, avg_latency_ms)

    class_names = ["with_mask", "without_mask"]
    rep_text = classification_report(y_test, y_pred, target_names=class_names, digits=4)
    print(f"  Accuracy:  {acc * 100:.2f}%")
    print(f"  Precision: {prec * 100:.2f}%")
    print(f"  Recall:    {rec * 100:.2f}%")
    print(f"  F1-Score:  {f1 * 100:.2f}%")
    print(f"  False MASK predictions on unmasked faces: {fn_mask_pred} / {int(np.sum(y_test == 1))} ({false_mask_rate_on_unmasked * 100:.2f}%)")
    print(f"  Avg single-face latency: {avg_latency_ms:.2f} ms ({observed_fps:.1f} FPS)")

    with open(out_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(rep_text)
    plot_confusion_matrix(cm, class_names, out_dir / "confusion_matrix.png")

    eval_summary = {
        "model_id": "mobilenetv2",
        "model_name": "MaskGuard_MobileNetV2",
        "total_test_samples": int(len(y_test)),
        "accuracy": round(acc, 4),
        "test_accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "avg_inference_latency_ms": round(avg_latency_ms, 2),
        "observed_fps": round(observed_fps, 1),
        "model_size_mb": round(model_path.stat().st_size / (1024 * 1024), 2),
        "false_mask_on_unmasked_count": fn_mask_pred,
        "unmasked_test_total": int(np.sum(y_test == 1)),
        "false_mask_on_unmasked_rate": round(false_mask_rate_on_unmasked, 4),
        "confusion_matrix": {
            "true_with_mask": tn_withmask,
            "false_without_mask": fp_nomask_pred,
            "false_with_mask": fn_mask_pred,
            "true_without_mask": tp_nomask,
        },
        "evaluation_dataset": "Kaggle Face Mask 12K Official Held-out Test Split (MD5 audited, zero overlap)",
    }
    with open(get_mobilenetv2_evaluation_path(), "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=4)

    total_params = int(reloaded.count_params())
    trainable_params = int(sum(int(np.prod(v.shape)) for v in reloaded.trainable_variables))
    export_mobilenetv2_metadata(
        get_mobilenetv2_metadata_path(),
        model_path=model_path,
        total_parameters=total_params,
        trainable_parameters=trainable_params,
        training_config={
            "stage1_epochs": len(hist_s1.history["loss"]),
            "stage2_epochs": len(hist_s2.history["loss"]),
            "selected_stage": selected_stage,
            "batch_size": batch_size,
            "stage1_lr": stage1_lr,
            "stage2_lr": stage2_lr,
            "unfreeze_last_n": unfreeze_last_n,
            "seed": seed,
            "total_train_duration_sec": round(s1_time + s2_time, 1),
            "stage1_best_val_accuracy": round(s1_best_val_acc, 4),
            "stage2_best_val_accuracy": round(s2_best_val_acc, 4),
        },
        dataset_info={
            "source": "Kaggle ashishjangra27/face-mask-12k-images-dataset",
            "train_samples": int(len(train_p)),
            "val_samples": int(len(val_p)),
            "test_samples": int(len(test_p)),
            "leakage_audit": "MD5 cross-split deduplication applied",
        },
        evaluation_provenance=eval_summary,
    )

    return eval_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MobileNetV2 Transfer Learning Model for MaskGuard.")
    parser.add_argument("--stage1-epochs", type=int, default=2)
    parser.add_argument("--stage2-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--stage1-lr", type=float, default=1e-4)
    parser.add_argument("--stage2-lr", type=float, default=1e-5)
    parser.add_argument("--unfreeze-last-n", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-dir", type=str, default=str(get_dataset_dir()))
    args = parser.parse_args()

    train_mobilenetv2(
        dataset_dir=args.dataset_dir,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        batch_size=args.batch_size,
        stage1_lr=args.stage1_lr,
        stage2_lr=args.stage2_lr,
        unfreeze_last_n=args.unfreeze_last_n,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
