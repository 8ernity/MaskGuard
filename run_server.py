"""
MaskGuard Web Application Server Launcher.
Initializes the environment, starts the FastAPI ASGI server on 127.0.0.1:8000,
and automatically opens the default web browser.
"""

from __future__ import annotations

import os
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Windows OpenBLAS and oneDNN stability settings
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


def _launch_browser_when_ready(url: str = "http://localhost:8000", max_wait: int = 40) -> None:
    """Polls server readiness in background and launches the default browser."""
    poll_url = "http://127.0.0.1:8000/api/status"
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        try:
            with urllib.request.urlopen(poll_url, timeout=1.5) as resp:
                if resp.status == 200:
                    time.sleep(0.3)
                    print(f"\n[+] MaskGuard is ready! Opening browser to {url} ...\n")
                    webbrowser.open(url)
                    return
        except Exception:
            time.sleep(0.5)

    # Fallback open if poll took too long
    print(f"\n[*] Opening browser to {url} ...\n")
    webbrowser.open(url)


def main() -> None:
    # Ensure model and directories exist
    from app.utils.resource_paths import ensure_project_directories, get_default_model_path
    ensure_project_directories()

    model_path = get_default_model_path()
    if not model_path.exists():
        print("[!] Pre-trained model not found. Generating initial dataset and training...")
        from training.prepare_dataset import bootstrap_synthetic_dataset
        from training.train import train_pipeline
        bootstrap_synthetic_dataset(samples_per_class=150)
        train_pipeline(epochs=15, batch_size=32)

    # Launch browser opener in daemon thread
    browser_thread = threading.Thread(
        target=_launch_browser_when_ready,
        args=("http://localhost:8000", 35),
        daemon=True,
    )
    browser_thread.start()

    print("=====================================================================")
    print("       MaskGuard — AI-Powered Face Mask Detection System")
    print("=====================================================================")
    print("  -> Web Dashboard:  http://localhost:8000")
    print("  -> API Health:     http://localhost:8000/api/status")
    print("  -> API Docs:       http://localhost:8000/docs")
    print("  -> Press CTRL+C in this terminal to stop the server.")
    print("=====================================================================\n")

    import uvicorn
    uvicorn.run("app.web.server:app", host="127.0.0.1", port=8000, log_level="info")


if __name__ == "__main__":
    main()
