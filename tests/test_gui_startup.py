"""
Automated smoke test for PySide6 GUI initialization and view instantiation.
Runs in offscreen mode to test UI components without requiring a physical monitor.
"""

import os
import sys
from pathlib import Path
import pytest

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from PySide6.QtWidgets import QApplication
from app.ui.main_window import MainWindow
from app.utils.config import AppConfig


@pytest.fixture(scope="session")
def qapp():
    """Ensure single persistent QApplication instance."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


def test_main_window_instantiation(qapp):
    """Test that MainWindow loads all views, buttons, and switches pages without crash."""
    config = AppConfig.load()
    window = MainWindow(config=config)
    assert window is not None
    assert window.windowTitle().startswith("MaskGuard")

    # Test navigation switches
    for page_key in ["dashboard", "live", "image", "video", "analytics", "settings"]:
        window.switch_view(page_key)
        assert window.nav_buttons[page_key].isChecked() is True

    # Test theme toggle
    window.apply_theme("light")
    assert window.config.theme == "light"
    window.apply_theme("dark")
    assert window.config.theme == "dark"

    window.close()
