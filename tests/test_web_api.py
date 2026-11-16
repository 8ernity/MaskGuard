"""
Automated tests for MaskGuard Web API server endpoints.
"""

from __future__ import annotations

import base64
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.web.server import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_serve_index(client: TestClient) -> None:
    """Verify root index HTML page loads with MaskGuard dashboard."""
    response = client.get("/")
    assert response.status_code == 200
    assert "MaskGuard" in response.text
    assert "Live Detection" in response.text


def test_api_status(client: TestClient) -> None:
    """Verify /api/status returns model and system readiness."""
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["model_loaded"] is True
    assert "mask_detector.h5" in data["model_file"]


def test_api_analytics(client: TestClient) -> None:
    """Verify /api/analytics returns valid evaluation statistics."""
    response = client.get("/api/analytics")
    assert response.status_code == 200
    data = response.json()
    assert "metrics" in data
    assert "test_accuracy" in data["metrics"]
    assert "confusion_matrix_url" in data


def test_api_settings_get_and_post(client: TestClient) -> None:
    """Verify retrieving and updating settings."""
    res_get = client.get("/api/settings")
    assert res_get.status_code == 200
    orig_cfg = res_get.json()

    # Update confidence threshold
    res_post = client.post("/api/settings", json={"confidence_threshold": 0.72})
    assert res_post.status_code == 200
    updated = res_post.json()["config"]
    assert updated["confidence_threshold"] == 0.72

    # Restore
    client.post("/api/settings", json={"confidence_threshold": orig_cfg.get("confidence_threshold", 0.65)})


def test_api_detect_frame(client: TestClient) -> None:
    """Verify base64 frame inference endpoint."""
    # Create synthetic test frame
    dummy = np.full((200, 200, 3), 128, dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", dummy)
    b64 = base64.b64encode(buf).decode("utf-8")
    data_uri = f"data:image/jpeg;base64,{b64}"

    response = client.post("/api/detect/frame", json={"image": data_uri})
    assert response.status_code == 200
    data = response.json()
    assert "faces_count" in data
    assert "annotated_image" in data
    assert "fps" in data
