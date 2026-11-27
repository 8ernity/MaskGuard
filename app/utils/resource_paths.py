"""
Deterministic path resolution utilities for MaskGuard.
Ensures correct absolute paths whether run from root, app/, or packaged executable.
"""

import sys
from pathlib import Path


def get_project_root() -> Path:
    """Return the absolute root directory of the MaskGuard project."""
    if getattr(sys, "frozen", False):
        # Running as PyInstaller/standalone bundle
        return Path(sys._MEIPASS).resolve()
    # Running from source - current file is in app/utils/resource_paths.py
    return Path(__file__).resolve().parent.parent.parent


def get_model_dir() -> Path:
    """Directory storing trained model weights and metadata."""
    path = get_project_root() / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_dataset_dir() -> Path:
    """Root dataset directory."""
    path = get_project_root() / "dataset"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_results_dir() -> Path:
    """Directory storing evaluation plots, reports, screenshots, and exported videos."""
    path = get_project_root() / "results"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_screenshots_dir() -> Path:
    """Directory for captured live detection snapshots."""
    path = get_results_dir() / "screenshots"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_videos_dir() -> Path:
    """Directory for exported processed videos."""
    path = get_results_dir() / "videos"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_default_model_path() -> Path:
    """Path to the main trained model (.h5)."""
    return get_model_dir() / "mask_detector.h5"


def get_class_mapping_path() -> Path:
    """Path to class_mapping.json."""
    return get_model_dir() / "class_mapping.json"


def get_model_metadata_path() -> Path:
    """Path to model_metadata.json."""
    return get_model_dir() / "model_metadata.json"


def get_mobilenetv2_dir() -> Path:
    """Directory storing MobileNetV2 weights, metadata, and evaluation artifacts."""
    path = get_model_dir() / "mobilenetv2"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_mobilenetv2_model_path() -> Path:
    """Path to the trained MobileNetV2 model (.h5)."""
    return get_mobilenetv2_dir() / "mask_detector_mobilenetv2.h5"


def get_mobilenetv2_class_mapping_path() -> Path:
    """Path to MobileNetV2 class_mapping.json."""
    return get_mobilenetv2_dir() / "class_mapping.json"


def get_mobilenetv2_metadata_path() -> Path:
    """Path to MobileNetV2 model_metadata.json."""
    return get_mobilenetv2_dir() / "model_metadata.json"


def get_mobilenetv2_evaluation_path() -> Path:
    """Path to MobileNetV2 evaluation_metrics.json."""
    return get_mobilenetv2_dir() / "evaluation_metrics.json"


def get_config_path() -> Path:
    """Path to persistent user settings JSON."""
    return get_project_root() / "settings.json"


def ensure_project_directories() -> None:
    """Ensure all required project subdirectories exist."""
    get_model_dir()
    (get_dataset_dir() / "with_mask").mkdir(parents=True, exist_ok=True)
    (get_dataset_dir() / "without_mask").mkdir(parents=True, exist_ok=True)
    get_results_dir()
    get_screenshots_dir()
    get_videos_dir()
