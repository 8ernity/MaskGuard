"""
Asynchronous camera worker running on a dedicated PySide6 QThread.
Captures webcam frames, routes them through the inference pipeline, and emits Qt signals.
"""

from __future__ import annotations

import time
from typing import List, Tuple

import cv2
import numpy as np
from PySide6.QtCore import QObject, QThread, Signal, Slot

from app.core.inference_pipeline import InferencePipeline


class CameraWorker(QObject):
    """
    QObject worker running inside a QThread to handle webcam streaming without blocking the UI.
    """

    frame_ready = Signal(np.ndarray, list, dict)  # (frame_bgr, detections, stats)
    status_changed = Signal(str)
    camera_error = Signal(str)
    stopped = Signal()

    def __init__(
        self,
        camera_index: int = 0,
        pipeline: InferencePipeline | None = None,
        target_fps: int = 30,
    ):
        super().__init__()
        self.camera_index = camera_index
        self.pipeline = pipeline or InferencePipeline()
        self.target_fps = target_fps
        self._is_running = False
        self._cap: cv2.VideoCapture | None = None

    @staticmethod
    def get_available_cameras(max_to_test: int = 3) -> List[int]:
        """
        Probe for available camera device indices.
        Uses quick non-blocking probes.
        """
        available = []
        for idx in range(max_to_test):
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW) if hasattr(cv2, "CAP_DSHOW") else cv2.VideoCapture(idx)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    available.append(idx)
                cap.release()
        return available if available else [0]

    def set_camera_index(self, index: int) -> None:
        """Switch camera device index."""
        if self._is_running:
            self.stop()
            self.camera_index = index
            self.start_capture()
        else:
            self.camera_index = index

    @Slot()
    def start_capture(self) -> None:
        """Main loop executed inside worker thread."""
        self._is_running = True
        self.status_changed.emit(f"Connecting to camera {self.camera_index}...")

        # Initialize VideoCapture with DirectShow on Windows for fast init
        backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else cv2.CAP_ANY
        self._cap = cv2.VideoCapture(self.camera_index, backend)

        if not self._cap or not self._cap.isOpened():
            # Retry without CAP_DSHOW
            self._cap = cv2.VideoCapture(self.camera_index)

        if not self._cap or not self._cap.isOpened():
            self._is_running = False
            self.camera_error.emit(
                f"Unable to open camera {self.camera_index}. Please verify that the webcam is connected and not in use by another app."
            )
            self.status_changed.emit("Camera offline")
            self.stopped.emit()
            return

        # Attempt to set reasonable resolution for real-time inference
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)

        self.status_changed.emit(f"Camera {self.camera_index} live")
        self.pipeline.reset_session_stats()

        frame_duration = 1.0 / max(1, self.target_fps)

        while self._is_running:
            start_time = time.time()

            if not self._cap or not self._cap.isOpened():
                self.camera_error.emit("Camera connection lost.")
                break

            ret, frame = self._cap.read()
            if not ret or frame is None or frame.size == 0:
                time.sleep(0.01)
                continue

            try:
                annotated_frame, detections, stats = self.pipeline.process_frame(
                    frame, track_session=True
                )
                self.frame_ready.emit(annotated_frame, detections, stats)
            except Exception as e:
                print(f"[CameraWorker] Inference error: {e}")

            # Pace frame rate to prevent unnecessary CPU churn
            elapsed = time.time() - start_time
            sleep_time = max(0.001, frame_duration - elapsed)
            time.sleep(sleep_time)

        # Cleanup
        if self._cap:
            self._cap.release()
            self._cap = None

        self.status_changed.emit("Camera stopped")
        self.stopped.emit()

    @Slot()
    def stop(self) -> None:
        """Signal the capture loop to stop."""
        self._is_running = False


class CameraThread(QThread):
    """Helper wrapper holding CameraWorker in its own thread."""

    def __init__(
        self,
        camera_index: int = 0,
        pipeline: InferencePipeline | None = None,
        parent: QObject | None = None,
    ):
        super().__init__(parent)
        self.worker = CameraWorker(camera_index=camera_index, pipeline=pipeline)
        self.worker.moveToThread(self)
        self.started.connect(self.worker.start_capture)
        self.worker.stopped.connect(self.quit)

    def stop_camera(self) -> None:
        """Gracefully stop camera worker and wait for thread to terminate."""
        self.worker.stop()
        self.quit()
        self.wait(2000)
