"""
Dataset Download and Ingestion Module for MaskGuard.
Downloads the official Face Mask 12K Images Dataset from Kaggle via kagglehub,
validates integrity, checks for duplicates & cross-split data leakage,
and cleans up any legacy synthetic/cartoon images.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure Kaggle cache stays on project drive to avoid C: drive out-of-space errors
CACHE_DIR = PROJECT_ROOT / ".cache" / "kagglehub"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["KAGGLEHUB_CACHE"] = str(CACHE_DIR)

DATASET_NAME = "ashishjangra27/face-mask-12k-images-dataset"


def compute_file_hash(filepath: Path) -> Tuple[Path, str]:
    """Compute MD5 hash of an image file to detect duplicate content."""
    hasher = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return filepath, hasher.hexdigest()
    except Exception:
        return filepath, ""


def download_kaggle_dataset(force: bool = False) -> Path:
    """
    Download Face Mask 12K Images Dataset via kagglehub.
    Provides clear troubleshooting instructions if download or auth fails.
    """
    print("=====================================================================")
    print("      MaskGuard — Kaggle Face Mask 12K Dataset Ingestion")
    print("=====================================================================")
    print(f"[*] Target dataset: '{DATASET_NAME}'")
    print(f"[*] Cache location: {CACHE_DIR}")

    # Check if already cached
    versions_dir = CACHE_DIR / "datasets" / "ashishjangra27" / "face-mask-12k-images-dataset" / "versions"
    if not force and versions_dir.exists():
        for candidate in sorted(versions_dir.glob("*"), reverse=True):
            if candidate.is_dir() and (candidate / "Face Mask Dataset").exists():
                print(f"[+] Found existing cached dataset at: {candidate}")
                return candidate

    print("[*] Downloading dataset via kagglehub...")
    try:
        import kagglehub
        downloaded_path = kagglehub.dataset_download(DATASET_NAME, force_download=force)
        path = Path(downloaded_path)
        print(f"[+] Download complete! Raw files at: {path}")
        return path
    except Exception as e:
        print("\n[ERROR] Failed to download dataset via kagglehub:")
        print(f"        {e}\n")
        print("Troubleshooting Steps:")
        print("1. If Kaggle authentication is required, obtain an API token from:")
        print("   https://www.kaggle.com/settings -> 'Create New Token'")
        print("   Place 'kaggle.json' into ~/.kaggle/kaggle.json")
        print("   Or set environment variables: KAGGLE_USERNAME and KAGGLE_KEY")
        print("2. Ensure sufficient disk space on drive D:")
        print("3. Alternatively, manually download the 329 MB zip from:")
        print("   https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset")
        print("   and extract its 'Face Mask Dataset' folder into: dataset/")
        raise


def find_face_mask_dataset_root(extracted_dir: Path) -> Path:
    """Locate the root directory containing Train, Validation, and Test folders."""
    candidate = extracted_dir / "Face Mask Dataset"
    if candidate.exists() and (candidate / "Train").exists():
        return candidate
    if (extracted_dir / "Train").exists():
        return extracted_dir
    # Recursive search
    for p in extracted_dir.rglob("Train"):
        if p.is_dir() and (p / "WithMask").exists():
            return p.parent
    raise FileNotFoundError(f"Could not find 'Face Mask Dataset' structure in {extracted_dir}")


def clean_synthetic_cartoon_images(dataset_dir: Path) -> int:
    """Delete all legacy synthetic/cartoon images from dataset."""
    removed = 0
    for synth_file in dataset_dir.rglob("synth_*.*"):
        if synth_file.is_file():
            synth_file.unlink()
            removed += 1
    if removed > 0:
        print(f"[+] Successfully purged {removed} legacy synthetic cartoon images from dataset.")
    return removed


def _copy_worker(item: Tuple[Path, Path]) -> bool:
    src, dst = item
    if not dst.exists():
        shutil.copy2(src, dst)
        return True
    return False


def setup_dataset(force_download: bool = False) -> Dict[str, Any]:
    """
    Main ingestion workflow:
    1. Downloads dataset via kagglehub.
    2. Deletes legacy synthetic images.
    3. Populates dataset/ with official Train, Validation, Test splits.
    4. Audits duplicates and data leakage across splits.
    """
    raw_root = download_kaggle_dataset(force=force_download)
    source_root = find_face_mask_dataset_root(raw_root)
    dest_root = PROJECT_ROOT / "dataset"
    dest_root.mkdir(parents=True, exist_ok=True)

    # 1. Clean old cartoon/synthetic images
    clean_synthetic_cartoon_images(dest_root)

    # 2. Multithreaded copy of official splits into dataset/
    splits = ["Train", "Validation", "Test"]
    copy_tasks: List[Tuple[Path, Path]] = []
    split_counts: Dict[str, Dict[str, int]] = {}

    for split in splits:
        src_split = source_root / split
        dst_split = dest_root / split
        dst_split.mkdir(parents=True, exist_ok=True)
        split_counts[split] = {"WithMask": 0, "WithoutMask": 0}

        for class_name in ["WithMask", "WithoutMask"]:
            src_class = src_split / class_name
            dst_class = dst_split / class_name
            dst_class.mkdir(parents=True, exist_ok=True)

            if src_class.exists():
                for img_file in src_class.iterdir():
                    if img_file.is_file() and img_file.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                        target_file = dst_class / img_file.name
                        copy_tasks.append((img_file, target_file))
                        split_counts[split][class_name] += 1

    print(f"[*] Synchronizing {len(copy_tasks)} real dataset images into dataset/ ...")
    with ThreadPoolExecutor(max_workers=32) as executor:
        list(executor.map(_copy_worker, copy_tasks))

    # 3. Multithreaded audit for duplicates and cross-split leakage
    print("\n[*] Auditing data integrity: checking duplicates and data leakage across splits...")
    hashes_by_split: Dict[str, Set[str]] = {"Train": set(), "Validation": set(), "Test": set()}
    duplicates_within_split: Dict[str, int] = {"Train": 0, "Validation": 0, "Test": 0}

    for split in splits:
        split_files = [
            p for p in (dest_root / split).rglob("*.*")
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
        with ThreadPoolExecutor(max_workers=32) as executor:
            hashed_results = list(executor.map(compute_file_hash, split_files))

        for _, h in hashed_results:
            if not h:
                continue
            if h in hashes_by_split[split]:
                duplicates_within_split[split] += 1
            else:
                hashes_by_split[split].add(h)

    # Check cross-split leakage
    train_hashes = hashes_by_split["Train"]
    val_leakage = hashes_by_split["Validation"].intersection(train_hashes)
    test_leakage = hashes_by_split["Test"].intersection(train_hashes)
    val_test_leakage = hashes_by_split["Validation"].intersection(hashes_by_split["Test"])

    print("\n--- Dataset Distribution Summary ---")
    for s, c in split_counts.items():
        total_s = c["WithMask"] + c["WithoutMask"]
        print(f"  {s:<12}: WithMask = {c['WithMask']:>5}, WithoutMask = {c['WithoutMask']:>5} | Total = {total_s:>5}")

    print("\n--- Data Hygiene & Leakage Report ---")
    print(f"  Internal duplicates in Train:      {duplicates_within_split['Train']}")
    print(f"  Internal duplicates in Validation: {duplicates_within_split['Validation']}")
    print(f"  Internal duplicates in Test:       {duplicates_within_split['Test']}")
    print(f"  Overlap between Train & Val:       {len(val_leakage)} (handled automatically in training pipeline)")
    print(f"  Overlap between Train & Test:      {len(test_leakage)} (handled automatically in training pipeline)")
    print(f"  Overlap between Val & Test:        {len(val_test_leakage)} ({'PASS - Zero leakage' if len(val_test_leakage) == 0 else 'WARNING'})")

    return {
        "status": "success",
        "counts": split_counts,
        "leakage": {
            "val_leakage": len(val_leakage),
            "test_leakage": len(test_leakage),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Download and prepare Kaggle Face Mask dataset on demand.")
    parser.add_argument("--force", action="store_true", help="Force redownload even if cached.")
    args = parser.parse_args()
    setup_dataset(force_download=args.force)


if __name__ == "__main__":
    main()
