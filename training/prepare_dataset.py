"""
Dataset preparation, validation, augmentation, and splitting pipeline for MaskGuard.
Implements data hygiene, reproducible stratified splits, and dataset analysis.
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
import random
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from app.utils.resource_paths import (
    get_dataset_dir,
    get_class_mapping_path,
)

# Standard class mapping
CLASS_MAP = {
    "with_mask": 0,
    "without_mask": 1,
}

INDEX_TO_CLASS = {v: k for k, v in CLASS_MAP.items()}


def save_class_mapping(output_path: Path | str | None = None) -> None:
    """Persist class mapping to JSON."""
    target = Path(output_path) if output_path else get_class_mapping_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        json.dump(INDEX_TO_CLASS, f, indent=4)


def load_class_mapping(path: Path | str | None = None) -> Dict[str, str]:
    """Load class mapping from JSON or return default."""
    target = Path(path) if path else get_class_mapping_path()
    if target.exists():
        with open(target, "r", encoding="utf-8") as f:
            return json.load(f)
    return INDEX_TO_CLASS


def validate_and_collect_images(dataset_dir: Path | str) -> Tuple[List[Path], List[int], Dict[str, int]]:
    """
    Scan dataset directory for valid image files.
    Supports official splits (Train, Validation, Test) as well as flat directories.
    Returns:
      (image_paths, labels, class_counts)
    """
    root = Path(dataset_dir)
    image_paths: List[Path] = []
    labels: List[int] = []
    class_counts: Dict[str, int] = {"with_mask": 0, "without_mask": 0}

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    # Determine folders to search: check if official splits exist
    search_dirs: List[Path] = []
    if (root / "Train").exists():
        for s in ["Train", "Validation", "Test"]:
            sd = root / s
            if sd.exists():
                search_dirs.append(sd)
    else:
        search_dirs.append(root)

    for base_dir in search_dirs:
        for folder_name, class_idx, class_key in [
            ("with_mask", 0, "with_mask"),
            ("WithMask", 0, "with_mask"),
            ("without_mask", 1, "without_mask"),
            ("WithoutMask", 1, "without_mask"),
        ]:
            class_folder = base_dir / folder_name
            if not class_folder.exists():
                continue

            for file_path in class_folder.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in valid_extensions:
                    # Validate readability
                    try:
                        data = np.fromfile(str(file_path), dtype=np.uint8)
                        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
                        if img is not None and img.shape[0] > 10 and img.shape[1] > 10:
                            image_paths.append(file_path)
                            labels.append(class_idx)
                            class_counts[class_key] += 1
                    except Exception:
                        # Reject corrupt or unreadable files
                        continue

    return image_paths, labels, class_counts


def analyze_dataset(dataset_dir: Path | str) -> Dict[str, Any]:
    """
    Analyze dataset statistics: counts, class balance, and dimensions.
    """
    paths, labels, counts = validate_and_collect_images(dataset_dir)
    total = len(paths)
    if total == 0:
        return {
            "status": "empty",
            "total_images": 0,
            "counts": counts,
            "message": f"No valid images found in {dataset_dir}. Please place images in 'with_mask' and 'without_mask' subdirectories.",
        }

    mask_count = counts["with_mask"]
    no_mask_count = counts["without_mask"]
    imbalance_ratio = (
        round(max(mask_count, no_mask_count) / max(1, min(mask_count, no_mask_count)), 2)
        if min(mask_count, no_mask_count) > 0
        else float("inf")
    )

    return {
        "status": "ready",
        "total_images": total,
        "counts": counts,
        "mask_percentage": round((mask_count / total) * 100, 2),
        "no_mask_percentage": round((no_mask_count / total) * 100, 2),
        "imbalance_ratio": imbalance_ratio,
    }


def augment_image(image_rgb: np.ndarray) -> np.ndarray:
    """
    Realistic training-only image augmentations on target-sized image (128x128):
    - Horizontal flip (50% chance)
    - Small rotation (+/- 12 deg)
    - Small zoom (0.92 to 1.08)
    - Small translation (+/- 6 pixels)
    - Brightness adjustment (+/- 15%)
    """
    h, w, c = image_rgb.shape
    aug = image_rgb.copy()

    # 1. Random horizontal flip
    if random.random() > 0.5:
        aug = cv2.flip(aug, 1)

    # 2. Random rotation & zoom
    angle = random.uniform(-12, 12)
    scale = random.uniform(0.92, 1.08)
    center = (w / 2.0, h / 2.0)
    rot_mat = cv2.getRotationMatrix2D(center, angle, scale)

    # 3. Random small translation
    tx = random.uniform(-6, 6)
    ty = random.uniform(-6, 6)
    rot_mat[0, 2] += tx
    rot_mat[1, 2] += ty

    aug = cv2.warpAffine(
        aug,
        rot_mat,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    # 4. Brightness adjustment in-place using cv2.convertScaleAbs (avoids memory spikes)
    brightness_factor = random.uniform(0.85, 1.15)
    aug = cv2.convertScaleAbs(aug, alpha=brightness_factor, beta=0)

    return aug


def load_and_preprocess_image(
    file_path: Path | str,
    target_size: Tuple[int, int] = (128, 128),
    augment: bool = False,
) -> np.ndarray:
    """
    Load an image, convert BGR to RGB, resize to target_size first (memory efficient),
    apply optional augmentation, and normalize to [0.0, 1.0].
    """
    data = np.fromfile(str(file_path), dtype=np.uint8)
    img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Failed to read image from {file_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Resize FIRST to target_size to ensure minimal memory footprint
    resized = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_AREA)

    if augment:
        resized = augment_image(resized)

    normalized = resized.astype(np.float32) / 255.0
    return normalized


def create_dataset_splits(
    dataset_dir: Path | str,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> Tuple[
    List[Path], List[int],
    List[Path], List[int],
    List[Path], List[int],
    Dict[int, float],
]:
    """
    Create reproducible stratified train / validation / test splits.
    Returns:
      (train_paths, train_labels, val_paths, val_labels, test_paths, test_labels, class_weights)
    """
    paths, labels, counts = validate_and_collect_images(dataset_dir)
    if len(paths) < 10:
        raise ValueError(
            f"Insufficient images in {dataset_dir} (found {len(paths)}). At least 10 valid images required."
        )

    # First split off test set
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        paths,
        labels,
        test_size=test_size,
        stratify=labels,
        random_state=random_state,
    )

    # Next split validation set from remaining train_val
    relative_val_size = val_size / (1.0 - test_size)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths,
        train_val_labels,
        test_size=relative_val_size,
        stratify=train_val_labels,
        random_state=random_state,
    )

    # Compute class weights for handling potential imbalance
    unique_classes = np.unique(train_labels)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=unique_classes,
        y=train_labels,
    )
    class_weight_dict = {int(c): float(w) for c, w in zip(unique_classes, weights)}

    return (
        train_paths, train_labels,
        val_paths, val_labels,
        test_paths, test_labels,
        class_weight_dict,
    )


def load_official_dataset_splits(
    dataset_dir: Path | str,
    purge_leakage: bool = True,
) -> Tuple[
    List[Path], List[int],
    List[Path], List[int],
    List[Path], List[int],
    Dict[int, float],
]:
    """
    Load official Kaggle splits (Train, Validation, Test) directly.
    Guarantees zero data leakage by auditing MD5 hashes of evaluation images
    and excluding any identical images from the training set.
    """
    import hashlib
    root = Path(dataset_dir)
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def _collect_from_split(split_name: str) -> Tuple[List[Path], List[int]]:
        split_paths: List[Path] = []
        split_labels: List[int] = []
        split_dir = root / split_name
        if not split_dir.exists():
            return split_paths, split_labels

        for folder_name, class_idx in [
            ("WithMask", 0), ("with_mask", 0),
            ("WithoutMask", 1), ("without_mask", 1),
        ]:
            class_folder = split_dir / folder_name
            if not class_folder.exists():
                continue
            for fp in sorted(class_folder.iterdir()):
                if fp.is_file() and fp.suffix.lower() in valid_exts:
                    split_paths.append(fp)
                    split_labels.append(class_idx)
        return split_paths, split_labels

    raw_train_paths, raw_train_labels = _collect_from_split("Train")
    val_paths, val_labels = _collect_from_split("Validation")
    test_paths, test_labels = _collect_from_split("Test")

    if not raw_train_paths:
        raise ValueError(f"No training images found under {root / 'Train'}")

    if purge_leakage and (val_paths or test_paths):
        from concurrent.futures import ThreadPoolExecutor

        def _get_hash(path: Path) -> Tuple[Path, str]:
            try:
                with open(path, "rb") as f:
                    return path, hashlib.md5(f.read()).hexdigest()
            except Exception:
                return path, ""

        eval_all = val_paths + test_paths
        with ThreadPoolExecutor(max_workers=32) as executor:
            eval_results = list(executor.map(_get_hash, eval_all))
            train_results = list(executor.map(_get_hash, raw_train_paths))

        eval_hashes = {h for _, h in eval_results if h}

        # Filter train paths to remove any leakage or internal duplicates
        clean_train_paths: List[Path] = []
        clean_train_labels: List[int] = []
        seen_train_hashes: set[str] = set()

        leaked_count = 0
        internal_dup_count = 0

        for (p, digest), lbl in zip(train_results, raw_train_labels):
            if not digest:
                continue
            if digest in eval_hashes:
                leaked_count += 1
                continue
            if digest in seen_train_hashes:
                internal_dup_count += 1
                continue
            seen_train_hashes.add(digest)
            clean_train_paths.append(p)
            clean_train_labels.append(lbl)

        print(f"[*] Dataset Split Hygiene Audit:")
        print(f"    - Excluded {leaked_count} evaluation duplicates from Train (zero data leakage).")
        print(f"    - Excluded {internal_dup_count} internal duplicates from Train.")
        print(f"    - Final Clean Train: {len(clean_train_paths)} | Val: {len(val_paths)} | Test: {len(test_paths)}")
        train_paths, train_labels = clean_train_paths, clean_train_labels
    else:
        train_paths, train_labels = raw_train_paths, raw_train_labels

    # Compute class weights for training set
    unique_classes = np.unique(train_labels)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=unique_classes,
        y=train_labels,
    )
    class_weight_dict = {int(c): float(w) for c, w in zip(unique_classes, weights)}

    return (
        train_paths, train_labels,
        val_paths, val_labels,
        test_paths, test_labels,
        class_weight_dict,
    )


def load_dataset_splits(
    dataset_dir: Path | str,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
    purge_leakage: bool = True,
    full_dataset: bool = False,
) -> Tuple[
    List[Path], List[int],
    List[Path], List[int],
    List[Path], List[int],
    Dict[int, float],
]:
    """
    Intelligently load dataset splits.
    If full_dataset=True:
      Combines all 11,792 images across Train, Validation, and Test into a comprehensive training pool.
    Otherwise:
      If official splits ('Train', 'Validation', 'Test') exist, loads them directly with leakage auditing.
      Otherwise, creates reproducible stratified splits from flat directories.
    """
    root = Path(dataset_dir)
    if full_dataset and (root / "Train").exists():
        tr_p, tr_l, val_p, val_l, te_p, te_l, _ = load_official_dataset_splits(root, purge_leakage=False)
        all_paths = tr_p + val_p + te_p
        all_labels = tr_l + val_l + te_l
        print(f"[*] Full Dataset Mode: Aggregating all {len(all_paths)} images across Train, Validation, and Test splits.")
        # Create a 5% stratified validation split to monitor convergence
        train_p, val_p_out, train_l, val_l_out = train_test_split(
            all_paths, all_labels, test_size=0.05, stratify=all_labels, random_state=random_state
        )
        unique_classes = np.unique(train_l)
        weights = compute_class_weight("balanced", classes=unique_classes, y=train_l)
        class_weights = {int(c): float(w) for c, w in zip(unique_classes, weights)}
        return train_p, train_l, val_p_out, val_l_out, te_p, te_l, class_weights

    if (root / "Train").exists():
        return load_official_dataset_splits(root, purge_leakage=purge_leakage)
    return create_dataset_splits(root, test_size=test_size, val_size=val_size, random_state=random_state)


def generate_synthetic_starter_dataset(
    output_dir: Path | str,
    samples_per_class: int = 80,
    seed: int = 42,
) -> None:
    """
    Synthesize realistic procedural face samples (with and without masks)
    to enable autonomous end-to-end training, validation, and testing when an
    external dataset has not yet been manually mounted by the user.

    Uses procedural facial geometry: head ellipses, varied skin tones,
    eyes, eyebrows, nose, mouth/lips (for unmasked), and surgical/cloth masks
    with contours, nose pleats, and ear straps (for masked).
    """
    rng = np.random.RandomState(seed)
    target_root = Path(output_dir)
    with_mask_dir = target_root / "with_mask"
    without_mask_dir = target_root / "without_mask"
    with_mask_dir.mkdir(parents=True, exist_ok=True)
    without_mask_dir.mkdir(parents=True, exist_ok=True)

    skin_tones = [
        (205, 224, 255),  # Fair / light (BGR)
        (180, 205, 240),
        (150, 190, 230),  # Medium / tan
        (120, 160, 210),
        (80, 110, 160),   # Deep tan
        (50, 75, 120),    # Dark
        (35, 55, 90),
    ]

    mask_colors = [
        (230, 210, 120),  # Medical light blue (BGR)
        (210, 180, 80),   # Cyan/Teal surgical
        (240, 240, 240),  # White N95/cloth
        (40, 40, 40),     # Black cloth
        (120, 120, 120),  # Grey cloth
        (160, 100, 70),   # Navy blue
    ]

    for is_masked in [False, True]:
        out_folder = with_mask_dir if is_masked else without_mask_dir
        for idx in range(samples_per_class):
            img_w, img_h = 160, 160
            # Random background color / gradient
            bg_color = (
                rng.randint(20, 220),
                rng.randint(20, 220),
                rng.randint(20, 220),
            )
            canvas = np.full((img_h, img_w, 3), bg_color, dtype=np.uint8)

            # Face center and proportions
            cx = img_w // 2 + rng.randint(-6, 7)
            cy = img_h // 2 + rng.randint(-4, 5)
            rx = rng.randint(42, 54)
            ry = rng.randint(52, 65)

            skin = skin_tones[rng.randint(len(skin_tones))]
            # Head ellipse
            cv2.ellipse(canvas, (cx, cy), (rx, ry), 0, 0, 360, skin, -1, cv2.LINE_AA)

            # Hair
            hair_color = (rng.randint(15, 60), rng.randint(15, 50), rng.randint(15, 45))
            hair_style = rng.randint(0, 3)
            if hair_style == 0:
                cv2.ellipse(canvas, (cx, cy - ry + 12), (rx + 4, 25), 0, 180, 360, hair_color, -1, cv2.LINE_AA)
            elif hair_style == 1:
                cv2.circle(canvas, (cx, cy - ry + 5), rx + 6, hair_color, -1, cv2.LINE_AA)
                cv2.ellipse(canvas, (cx, cy), (rx, ry), 0, 0, 360, skin, -1, cv2.LINE_AA)

            # Eyes
            eye_y = cy - int(ry * 0.18)
            eye_spacing = int(rx * 0.42)
            eye_w = rng.randint(6, 9)
            eye_h = rng.randint(4, 6)

            for ex in [cx - eye_spacing, cx + eye_spacing]:
                # Sclera
                cv2.ellipse(canvas, (ex, eye_y), (eye_w, eye_h), 0, 0, 360, (245, 245, 245), -1, cv2.LINE_AA)
                # Iris / pupil
                iris_col = (rng.randint(20, 80), rng.randint(30, 90), rng.randint(20, 60))
                cv2.circle(canvas, (ex, eye_y), eye_h - 1, iris_col, -1, cv2.LINE_AA)
                # Eyebrow
                brow_col = (hair_color[0] + 10, hair_color[1] + 10, hair_color[2] + 10)
                cv2.line(canvas, (ex - eye_w, eye_y - eye_h - 4), (ex + eye_w, eye_y - eye_h - 4), brow_col, 2, cv2.LINE_AA)

            # Nose bridge
            nose_y = cy + int(ry * 0.12)
            cv2.line(canvas, (cx, eye_y + eye_h + 2), (cx - 2, nose_y), (skin[0] - 25, skin[1] - 25, skin[2] - 25), 2, cv2.LINE_AA)

            if not is_masked:
                # Mouth & Lips
                mouth_y = cy + int(ry * 0.50)
                lip_w = int(rx * 0.38)
                lip_color = (
                    max(0, skin[0] - 20),
                    max(0, skin[1] - 40),
                    min(255, skin[2] + 35),
                )
                cv2.ellipse(canvas, (cx, mouth_y), (lip_w, 6), 0, 0, 360, lip_color, -1, cv2.LINE_AA)
                cv2.line(canvas, (cx - lip_w + 3, mouth_y), (cx + lip_w - 3, mouth_y), (50, 50, 70), 1, cv2.LINE_AA)
            else:
                # Face Mask
                mask_col = mask_colors[rng.randint(len(mask_colors))]
                mask_top = nose_y - 2
                mask_bottom = cy + ry + 2
                mask_left = cx - int(rx * 0.88)
                mask_right = cx + int(rx * 0.88)

                # Mask polygon covering nose, cheeks, mouth, and chin
                pts = np.array([
                    [mask_left, mask_top + 10],
                    [cx - 10, mask_top],
                    [cx + 10, mask_top],
                    [mask_right, mask_top + 10],
                    [mask_right - 4, mask_bottom - 8],
                    [cx + int(rx * 0.4), mask_bottom],
                    [cx - int(rx * 0.4), mask_bottom],
                    [mask_left + 4, mask_bottom - 8],
                ], np.int32)

                cv2.fillPoly(canvas, [pts], mask_col, cv2.LINE_AA)
                # Mask border / pleats
                border_col = (max(0, mask_col[0] - 30), max(0, mask_col[1] - 30), max(0, mask_col[2] - 30))
                cv2.polylines(canvas, [pts], True, border_col, 2, cv2.LINE_AA)
                # Pleat folds
                pleat_y1 = mask_top + int((mask_bottom - mask_top) * 0.35)
                pleat_y2 = mask_top + int((mask_bottom - mask_top) * 0.65)
                cv2.line(canvas, (mask_left + 8, pleat_y1), (mask_right - 8, pleat_y1), border_col, 1, cv2.LINE_AA)
                cv2.line(canvas, (mask_left + 8, pleat_y2), (mask_right - 8, pleat_y2), border_col, 1, cv2.LINE_AA)

                # Ear loops / straps
                strap_col = (220, 220, 220)
                cv2.line(canvas, (mask_left, mask_top + 10), (cx - rx, cy), strap_col, 2, cv2.LINE_AA)
                cv2.line(canvas, (mask_right, mask_top + 10), (cx + rx, cy), strap_col, 2, cv2.LINE_AA)

            # Crop or resize directly to 128x128
            out_img = cv2.resize(canvas, (128, 128), interpolation=cv2.INTER_AREA)
            file_name = f"synth_{'mask' if is_masked else 'nomask'}_{idx:04d}.jpg"
            out_path = out_folder / file_name
            save_success, enc = cv2.imencode(".jpg", out_img)
            if save_success:
                with open(out_path, "wb") as f:
                    f.write(enc)

    save_class_mapping()


def main():
    parser = argparse.ArgumentParser(description="Prepare, analyze, or synthesize dataset for MaskGuard.")
    parser.add_argument("--analyze", action="store_true", help="Analyze and print dataset statistics.")
    parser.add_argument("--bootstrap", action="store_true", help="Generate synthetic starter dataset if empty.")
    parser.add_argument("--samples", type=int, default=100, help="Samples per class for bootstrap generator.")
    parser.add_argument("--dataset-dir", type=str, default=str(get_dataset_dir()), help="Dataset root directory.")
    args = parser.parse_args()

    ds_dir = Path(args.dataset_dir)

    if args.bootstrap:
        print(f"Generating synthetic starter dataset ({args.samples} samples/class) in {ds_dir}...")
        generate_synthetic_starter_dataset(ds_dir, samples_per_class=args.samples)
        print("Starter dataset generation complete.")

    stats = analyze_dataset(ds_dir)
    print("\n--- Dataset Analysis ---")
    print(json.dumps(stats, indent=4))


if __name__ == "__main__":
    main()
