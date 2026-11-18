"""
Application configuration management.
Handles persistent user preferences and runtime detection hyperparameters.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict

from app.utils.resource_paths import (
    get_config_path,
    get_default_model_path,
    get_results_dir,
)


@dataclass
class AppConfig:
    # UI Appearance
    theme: str = "dark"  # "dark" | "light"

    # Camera & Detection Settings
    camera_index: int = 0
    confidence_threshold: float = 0.65  # Confidence required to declare confident MASK / NO MASK
    uncertain_band: float = 0.15  # +/- band around 0.5 where prediction is flagged as UNCERTAIN

    # Face Detection Hyperparameters (OpenCV Haar Cascade)
    face_scale_factor: float = 1.1
    face_min_neighbors: int = 5
    face_min_size: int = 30
    face_padding: float = 0.12  # 12% margin around detected face box

    # Display Toggles
    show_boxes: bool = True
    show_labels: bool = True
    show_confidence: bool = True
    show_fps: bool = True

    # Performance
    frame_skip: int = 1  # Process every Nth frame (1 = every frame)
    max_inference_width: int = 640  # Resize frame width for faster inference if necessary

    # File Paths
    model_path: str = field(default_factory=lambda: str(get_default_model_path()))
    output_dir: str = field(default_factory=lambda: str(get_results_dir()))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AppConfig:
        valid_keys = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in data.items() if k in valid_keys}

        # Self-healing cross-platform path resolution:
        # If model_path from settings.json points to a different machine/OS (e.g. Windows D:\ on Linux),
        # automatically fallback to the local project's model directory.
        if "model_path" in filtered:
            p = Path(filtered["model_path"])
            if not p.is_file():
                filtered["model_path"] = str(get_default_model_path())

        if "output_dir" in filtered:
            p = Path(filtered["output_dir"])
            if not p.is_dir():
                filtered["output_dir"] = str(get_results_dir())

        return cls(**filtered)

    def save(self, filepath: Path | str | None = None) -> None:
        """Persist current configuration to JSON atomically."""
        import os
        import tempfile
        target = Path(filepath) if filepath else get_config_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write to temporary file on same filesystem prevents partial file corruption
        with tempfile.NamedTemporaryFile("w", dir=target.parent, delete=False, encoding="utf-8") as tf:
            json.dump(self.to_dict(), tf, indent=4)
            temp_name = tf.name
        os.replace(temp_name, target)

    @classmethod
    def load(cls, filepath: Path | str | None = None) -> AppConfig:
        """Load configuration from JSON or return defaults if missing/corrupt."""
        target = Path(filepath) if filepath else get_config_path()
        if not target.exists():
            cfg = cls()
            cfg.save(target)
            return cfg

        try:
            with open(target, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls.from_dict(data)
        except Exception:
            # Fallback to defaults on corrupt JSON
            cfg = cls()
            cfg.save(target)
            return cfg
