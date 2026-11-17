"""
Offline video processing engine for MaskGuard.
Processes MP4 and AVI files frame-by-frame, renders annotations, and exports annotated video.
"""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Callable, Dict, Optional

import cv2
import numpy as np

try:
    from PySide6.QtCore import QObject, Signal, Slot
except ImportError:
    # Allow running in headless environments without PySide6
    class QObject:  # type: ignore
        pass
    def Signal(*args, **kwargs):  # type: ignore
        return None
    def Slot(*args, **kwargs):  # type: ignore
        def decorator(f):
            return f
        return decorator

from app.core.inference_pipeline import InferencePipeline
from app.utils.resource_paths import get_videos_dir


def process_video_file(
    input_path: Path | str,
    output_path: Path | str,
    pipeline: InferencePipeline,
    progress_callback: Optional[Callable[[int, int, float], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    """
    Pure Python headless video processor for web API and background tasks.
    Reads input video frame-by-frame, annotates faces, and exports to output_path.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input video file not found: {input_path}")

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {input_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = orig_fps if orig_fps > 0 else 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if total_frames <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise ValueError("Corrupted or empty video stream detected.")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        fallback_path = output_path.with_suffix(".avi")
        writer = cv2.VideoWriter(str(fallback_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError("Failed to initialize video encoder. Verify codec availability.")
        output_path = fallback_path

    current_frame = 0
    start_time = time.time()
    last_callback_time = 0.0

    try:
        while True:
            if cancel_check and cancel_check():
                break

            ret, frame = cap.read()
            if not ret or frame is None:
                break

            current_frame += 1
            annotated, detections, stats = pipeline.process_frame(frame, track_session=False)
            writer.write(annotated)

            now = time.time()
            if progress_callback and (now - last_callback_time >= 0.15 or current_frame == total_frames):
                elapsed = max(0.001, now - start_time)
                proc_fps = current_frame / elapsed
                progress_callback(current_frame, total_frames, proc_fps)
                last_callback_time = now
    finally:
        cap.release()
        writer.release()

    return {
        "total_frames": current_frame,
        "output_path": str(output_path),
        "fps": fps,
        "width": width,
        "height": height,
    }


class VideoProcessor(QObject):
    """
    QObject worker running in a background thread to process video files.
    """

    progress_updated = Signal(int, int, float)  # current_frame, total_frames, percentage
    frame_processed = Signal(np.ndarray)        # latest annotated frame for preview
    status_message = Signal(str)
    finished = Signal(str, dict)                # output_path, summary_stats
    error_occurred = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        input_path: Path | str,
        output_path: Path | str | None = None,
        pipeline: InferencePipeline | None = None,
    ):
        super().__init__()
        self.input_path = Path(input_path)
        self.output_path = (
            Path(output_path)
            if output_path
            else get_videos_dir() / f"annotated_{self.input_path.stem}.mp4"
        )
        self.pipeline = pipeline or InferencePipeline()
        self._is_cancelled = False

    @Slot()
    def cancel(self) -> None:
        """Request cooperative cancellation."""
        self._is_cancelled = True
        self.status_message.emit("Cancelling video processing...")

    @Slot()
    def process(self) -> None:
        """Run sequential frame inference and video export."""
        self._is_cancelled = False

        if not self.input_path.exists():
            self.error_occurred.emit(f"Input video file not found: {self.input_path}")
            return

        cap = cv2.VideoCapture(str(self.input_path))
        if not cap.isOpened():
            self.error_occurred.emit(f"Failed to open video file: {self.input_path}")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        fps = orig_fps if orig_fps > 0 else 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if total_frames <= 0 or width <= 0 or height <= 0:
            cap.release()
            self.error_occurred.emit("Corrupted or empty video stream detected.")
            return

        # Prepare VideoWriter
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        # Use mp4v or XVID codec
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(self.output_path), fourcc, fps, (width, height))

        if not writer.isOpened():
            # Fallback codec
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            fallback_path = self.output_path.with_suffix(".avi")
            writer = cv2.VideoWriter(str(fallback_path), fourcc, fps, (width, height))
            if not writer.isOpened():
                cap.release()
                self.error_occurred.emit("Failed to initialize video encoder. Verify codec availability.")
                return
            self.output_path = fallback_path

        self.status_message.emit(f"Processing {total_frames} frames ({width}x{height} @ {fps:.1f} FPS)...")
        self.pipeline.reset_session_stats()

        current_frame = 0
        total_mask_detections = 0
        total_nomask_detections = 0
        total_uncertain_detections = 0
        start_time = time.time()

        while True:
            if self._is_cancelled:
                cap.release()
                writer.release()
                if self.output_path.exists():
                    try:
                        self.output_path.unlink()
                    except Exception:
                        pass
                self.cancelled.emit()
                return

            ret, frame = cap.read()
            if not ret or frame is None:
                break

            current_frame += 1

            try:
                annotated, detections, stats = self.pipeline.process_frame(frame, track_session=True)
            except Exception as e:
                print(f"[VideoProcessor] Frame {current_frame} processing warning: {e}")
                annotated = frame
                detections = []

            for d in detections:
                if d.label == "MASK" and not d.is_uncertain:
                    total_mask_detections += 1
                elif d.label == "NO MASK" and not d.is_uncertain:
                    total_nomask_detections += 1
                elif d.is_uncertain:
                    total_uncertain_detections += 1

            writer.write(annotated)

            # Emit preview periodically (e.g. every 2 frames)
            if current_frame % 2 == 0 or current_frame == total_frames:
                pct = (current_frame / total_frames) * 100.0 if total_frames > 0 else 0.0
                self.progress_updated.emit(current_frame, total_frames, pct)
                self.frame_processed.emit(annotated)

        cap.release()
        writer.release()

        elapsed = time.time() - start_time
        summary = {
            "total_frames_processed": current_frame,
            "elapsed_seconds": round(elapsed, 2),
            "processing_fps": round(current_frame / max(0.1, elapsed), 2),
            "total_unique_faces": self.pipeline.tracker.total_unique_faces,
            "mask_detections": total_mask_detections,
            "nomask_detections": total_nomask_detections,
            "uncertain_detections": total_uncertain_detections,
            "output_path": str(self.output_path),
        }

        self.status_message.emit("Video processing complete!")
        self.finished.emit(str(self.output_path), summary)
