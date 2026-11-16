"""
Main entry point for the MaskGuard desktop application.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure root directory is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from app.ui.main_window import MainWindow
from app.utils.config import AppConfig
from app.utils.resource_paths import ensure_project_directories


def run_app() -> int:
    """Initialize and run the PySide6 application."""
    ensure_project_directories()

    # Create Qt Application
    app = QApplication(sys.argv)
    app.setApplicationName("MaskGuard")
    app.setApplicationDisplayName("MaskGuard — AI Face Mask Detection")
    app.setOrganizationName("MaskGuardAI")

    config = AppConfig.load()
    window = MainWindow(config=config)
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(run_app())
