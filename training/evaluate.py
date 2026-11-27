"""
Evaluation pipeline for MaskGuard.
Computes test metrics, classification report, confusion matrix, and plots loss/accuracy curves.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Configure environment before heavy library imports
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import tensorflow as tf

from app.utils.resource_paths import (
    get_dataset_dir,
    get_default_model_path,
    get_results_dir,
)
from training.prepare_dataset import (
    load_dataset_splits,
    load_and_preprocess_image,
)


def plot_training_history(history_file: Path | str, output_image: Path | str) -> bool:
    """
    Generate side-by-side plots for Training & Validation Loss and Accuracy.
    """
    h_path = Path(history_file)
    if not h_path.exists():
        print(f"History file {h_path} not found. Skipping history plot.")
        return False

    with open(h_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    epochs = range(1, len(history.get("loss", [])) + 1)
    if not epochs:
        return False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    fig.patch.set_facecolor("#161B22")

    # Set dark styling
    for ax in (ax1, ax2):
        ax.set_facecolor("#0D1117")
        ax.tick_params(colors="#C9D1D9")
        ax.xaxis.label.set_color("#C9D1D9")
        ax.yaxis.label.set_color("#C9D1D9")
        ax.title.set_color("#F0F6FC")
        for spine in ax.spines.values():
            spine.set_color("#30363D")
        ax.grid(True, linestyle="--", alpha=0.3, color="#8B949E")

    # Accuracy Plot
    train_acc = history.get("accuracy", [])
    val_acc = history.get("val_accuracy", [])
    ax1.plot(epochs, train_acc, label="Train Accuracy", color="#10B981", linewidth=2.5)
    if val_acc:
        ax1.plot(epochs, val_acc, label="Val Accuracy", color="#38BDF8", linewidth=2.5, linestyle="--")
    ax1.set_title("Training & Validation Accuracy", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Accuracy", fontsize=11)
    ax1.legend(facecolor="#161B22", edgecolor="#30363D", labelcolor="#F0F6FC")

    # Loss Plot
    train_loss = history.get("loss", [])
    val_loss = history.get("val_loss", [])
    ax2.plot(epochs, train_loss, label="Train Loss", color="#EF4444", linewidth=2.5)
    if val_loss:
        ax2.plot(epochs, val_loss, label="Val Loss", color="#F59E0B", linewidth=2.5, linestyle="--")
    ax2.set_title("Training & Validation Loss", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Loss", fontsize=11)
    ax2.legend(facecolor="#161B22", edgecolor="#30363D", labelcolor="#F0F6FC")

    plt.tight_layout()
    out_path = Path(output_image)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return True


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    output_image: Path | str,
) -> None:
    """
    Render and save a high-contrast Confusion Matrix heatmap.
    """
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    fig.patch.set_facecolor("#161B22")
    ax.set_facecolor("#0D1117")

    cmap = sns.color_palette("mako", as_cmap=True)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        ax=ax,
        annot_kws={"size": 14, "weight": "bold", "color": "#FFFFFF"},
    )

    ax.tick_params(colors="#C9D1D9")
    ax.xaxis.label.set_color("#C9D1D9")
    ax.yaxis.label.set_color("#C9D1D9")
    ax.title.set_color("#F0F6FC")
    ax.set_title("Confusion Matrix (Held-out Test Set)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Class", fontsize=11)
    ax.set_ylabel("Actual Class", fontsize=11)

    plt.tight_layout()
    out_path = Path(output_image)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def evaluate_model(
    model_path: Path | str,
    dataset_dir: Path | str,
    results_dir: Path | str,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run evaluation on the held-out test split, generating all metrics and plots.
    """
    m_path = Path(model_path)
    ds_path = Path(dataset_dir)
    res_dir = Path(results_dir)
    res_dir.mkdir(parents=True, exist_ok=True)

    if not m_path.exists():
        raise FileNotFoundError(f"Model file not found at {m_path}. Train the model first.")

    print(f"Loading trained model from {m_path}...")
    model = tf.keras.models.load_model(str(m_path))

    # Load held-out test split
    (
        _, _,
        _, _,
        test_paths, test_labels,
        _,
    ) = load_dataset_splits(ds_path, test_size=0.15, val_size=0.15, random_state=seed)

    print(f"Evaluating model on {len(test_paths)} held-out test samples...")

    # Batched inference for minimal memory footprint
    batch_size = 64
    pred_list = []
    for i in range(0, len(test_paths), batch_size):
        batch_paths = test_paths[i : i + batch_size]
        batch_imgs = [
            load_and_preprocess_image(p, target_size=(128, 128), augment=False)
            for p in batch_paths
        ]
        X_batch = np.array(batch_imgs, dtype=np.float32)
        batch_preds = model.predict(X_batch, verbose=0).flatten()
        pred_list.extend(batch_preds)

    raw_preds = np.array(pred_list, dtype=np.float32)
    y_test = np.array(test_labels, dtype=np.int32)
    y_pred = (raw_preds >= 0.5).astype(np.int32)

    # Metrics
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    cm = confusion_matrix(y_test, y_pred)

    class_names = ["with_mask", "without_mask"]
    rep_text = classification_report(y_test, y_pred, target_names=class_names, digits=4)
    rep_dict = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)

    print("\n--- Test Evaluation Results ---")
    print(f"Accuracy:  {acc * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}%")
    print(f"Recall:    {rec * 100:.2f}%")
    print(f"F1-Score:  {f1 * 100:.2f}%")
    print("\nClassification Report:\n", rep_text)
    print("Confusion Matrix:\n", cm)

    # Save classification report text
    with open(res_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(rep_text)

    # Save metrics JSON
    metrics_summary = {
        "test_accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "test_samples": len(y_test),
        "confusion_matrix": cm.tolist(),
        "detailed_report": rep_dict,
    }
    with open(res_dir / "evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=4)

    # Render confusion matrix heatmap
    cm_plot_path = res_dir / "confusion_matrix.png"
    plot_confusion_matrix(cm, class_names, cm_plot_path)
    print(f"Saved confusion matrix plot to {cm_plot_path}")

    # Render training loss & accuracy curves
    history_file = res_dir / "training_history.json"
    curves_plot_path = res_dir / "training_curves.png"
    if history_file.exists():
        plot_training_history(history_file, curves_plot_path)
        print(f"Saved training curves plot to {curves_plot_path}")

    return metrics_summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained MaskGuard CNN model.")
    parser.add_argument("--model-path", type=str, default=str(get_default_model_path()), help="Path to .h5 model.")
    parser.add_argument("--dataset-dir", type=str, default=str(get_dataset_dir()), help="Dataset directory.")
    parser.add_argument("--results-dir", type=str, default=str(get_results_dir()), help="Results directory.")
    parser.add_argument("--seed", type=int, default=42, help="Reproducible seed for test split.")
    args = parser.parse_args()

    evaluate_model(
        model_path=args.model_path,
        dataset_dir=args.dataset_dir,
        results_dir=args.results_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
