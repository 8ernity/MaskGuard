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
        print(f"[!] Pre-trained model not found at: {model_path}")
        print("[!] To train the custom CNN with the official Kaggle dataset, run:")
        print("      python training/train.py")
        print("[!] MaskGuard will start in standby mode (inference endpoints will return 503 until model is present).\n")

    import socket
    port = 8000
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(("127.0.0.1", 8000)) == 0:
            port = 8001
            print(f"[*] Port 8000 in use by another service. Switching to port {port}...")

    target_url = f"http://localhost:{port}"

    # Launch browser opener in daemon thread
    browser_thread = threading.Thread(
        target=_launch_browser_when_ready,
        args=(target_url, 35),
        daemon=True,
    )
    browser_thread.start()

    print("=====================================================================")
    print("       MaskGuard — AI-Powered Face Mask Detection System")
    print("=====================================================================")
    print(f"  -> Web Dashboard:  {target_url}")
    print(f"  -> API Health:     {target_url}/api/status")
    print(f"  -> API Docs:       {target_url}/docs")
    print("  -> Press CTRL+C in this terminal to stop the server.")
    print("=====================================================================\n")

    import uvicorn
    uvicorn.run("app.web.server:app", host="127.0.0.1", port=port, log_level="info")


if __name__ == "__main__":
    main()
