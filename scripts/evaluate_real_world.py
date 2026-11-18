"""
Real-World Accuracy Evaluation & Benchmark Suite for MaskGuard.
Evaluates model on:
1. 50 real held-out masked faces from Test split
2. 50 real held-out unmasked faces from Test split
3. Real webcam user faces from user uploads in artifacts directory
4. Impact of face detector crop padding factor (1.12 vs 1.20 vs 1.35 vs 1.53)
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

import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score

from app.utils.image_utils import preprocess_face_tensor
from app.utils.resource_paths import get_default_model_path, get_results_dir, get_model_dir


def load_best_eval_model() -> tf.keras.Model:
    """Load best available trained model."""
    best_weights = get_model_dir() / "best_weights.keras"
    default_h5 = get_default_model_path()
    
    if best_weights.is_file():
        print(f"[*] Loading model from checkpoint: {best_weights}")
        return tf.keras.models.load_model(str(best_weights))
    elif default_h5.is_file():
        print(f"[*] Loading model from default path: {default_h5}")
        return tf.keras.models.load_model(str(default_h5))
    else:
        raise FileNotFoundError("No trained model found.")


def run_real_world_evaluation():
    model = load_best_eval_model()
    
    # 1. Evaluate 50 Masked and 50 Unmasked from Held-out Test Set
    test_mask_dir = Path("dataset/Test/WithMask")
    test_nomask_dir = Path("dataset/Test/WithoutMask")
    
    mask_files = sorted(test_mask_dir.glob("*.png"))[:50]
    nomask_files = sorted(test_nomask_dir.glob("*.png"))[:50]
    
    y_true = []
    y_pred = []
    raw_scores = []
    details = []
    
    print("\n" + "="*60)
    print("PHASE 8.1: HELD-OUT TEST BENCHMARK (50 MASKED, 50 UNMASKED)")
    print("="*60)
    
    for p in mask_files:
        data = np.fromfile(str(p), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            continue
        tensor = preprocess_face_tensor(img, target_size=(128, 128))
        raw = float(model(np.expand_dims(tensor, 0), training=False).numpy()[0, 0])
        # 0 = with_mask, 1 = without_mask
        pred = 0 if raw < 0.5 else 1
        y_true.append(0)
        y_pred.append(pred)
        raw_scores.append(raw)
        details.append({"file": p.name, "expected": "with_mask", "raw": raw, "pred": "with_mask" if pred==0 else "without_mask", "correct": pred==0})
        
    for p in nomask_files:
        data = np.fromfile(str(p), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            continue
        tensor = preprocess_face_tensor(img, target_size=(128, 128))
        raw = float(model(np.expand_dims(tensor, 0), training=False).numpy()[0, 0])
        pred = 0 if raw < 0.5 else 1
        y_true.append(1)
        y_pred.append(pred)
        raw_scores.append(raw)
        details.append({"file": p.name, "expected": "without_mask", "raw": raw, "pred": "with_mask" if pred==0 else "without_mask", "correct": pred==1})
        
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=1)
    rec = recall_score(y_true, y_pred, pos_label=1)
    f1 = f1_score(y_true, y_pred, pos_label=1)
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"Overall Accuracy:  {acc*100:.2f}%")
    print(f"Precision (NoMask):{prec*100:.2f}%")
    print(f"Recall (NoMask):   {rec*100:.2f}%")
    print(f"F1-Score:          {f1*100:.2f}%")
    print("\nConfusion Matrix:")
    print("                Pred Mask    Pred NoMask")
    print(f"Actual Mask        {cm[0, 0]:<12}{cm[0, 1]}")
    print(f"Actual NoMask      {cm[1, 0]:<12}{cm[1, 1]}")
    
    # 2. Evaluate Real User Uploaded Faces (The core problem case)
    print("\n" + "="*60)
    print("PHASE 8.2: REAL WEBCAM USER FACE DIAGNOSTIC EVALUATION")
    print("="*60)
    
    user_test_image = Path(r"C:\Users\arpan\.gemini\antigravity-cli\brain\0aa93e9e-91e3-423b-aaba-426af83f9cff\.user_uploaded\uploaded_media_1794931884474.png")
    user_results = {}
    if user_test_image.is_file():
        img = cv2.imread(str(user_test_image))
        # User face is around [35, 88, 104, 104]
        # Compare tight crop vs padded crops
        fx, fy, fw, fh = 35, 88, 104, 104
        dim = max(fw, fh)
        
        scales = [1.00, 1.12, 1.20, 1.25, 1.35, 1.53]
        for s in scales:
            target_size = int(round(dim * s))
            half_side = target_size // 2
            cx = fx + fw // 2
            cy = fy + fh // 2 + int(dim * 0.05)
            
            nx = max(0, cx - half_side)
            ny = max(0, cy - half_side)
            nw = min(img.shape[1] - nx, target_size)
            nh = min(img.shape[0] - ny, target_size)
            
            crop = img[ny:ny+nh, nx:nx+nw]
            tensor = preprocess_face_tensor(crop, target_size=(128, 128))
            raw = float(model(np.expand_dims(tensor, 0), training=False).numpy()[0, 0])
            pred_lbl = "without_mask (NO MASK)" if raw >= 0.5 else "with_mask (MASK)"
            conf = raw if raw >= 0.5 else (1.0 - raw)
            is_correct = raw >= 0.5 # User has NO mask
            
            user_results[f"scale_{s:.2f}"] = {
                "scale_factor": s,
                "crop_w_h": [nw, nh],
                "raw_sigmoid": round(raw, 6),
                "predicted": pred_lbl,
                "confidence_pct": round(conf * 100, 2),
                "correct": is_correct,
            }
            status_str = "PASS [CORRECT]" if is_correct else "FAIL [FALSE MASK]"
            print(f"Scale {s:.2f} ({nw}x{nh}px): raw={raw:.6f} -> {pred_lbl} ({conf*100:.1f}%) | {status_str}")
            
    # Save benchmark report
    report = {
        "held_out_test_metrics": {
            "total_samples": len(y_true),
            "accuracy": round(acc, 4),
            "precision_nomask": round(prec, 4),
            "recall_nomask": round(rec, 4),
            "f1_score": round(f1, 4),
            "confusion_matrix": {
                "true_positive_mask": int(cm[0, 0]),
                "false_positive_nomask": int(cm[0, 1]),
                "false_negative_mask": int(cm[1, 0]),
                "true_negative_nomask": int(cm[1, 1]),
            },
        },
        "user_webcam_face_benchmark": user_results,
    }
    
    out_file = get_results_dir() / "real_world_evaluation_report.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
    print(f"\nSaved benchmark report to {out_file}")
    return report


if __name__ == "__main__":
    run_real_world_evaluation()
