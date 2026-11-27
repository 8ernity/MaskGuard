"""
Automated Security and Hardening Test Suite for MaskGuard.
Tests defensive limits, file-signature validation, path containment,
CORS restrictions, security headers, and typed settings bounds.
"""

from __future__ import annotations

import io
import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.web.server import app
from app.utils.config import AppConfig


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_security_headers_present(client: TestClient) -> None:
    """Verify defense-in-depth security headers on HTTP responses."""
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.headers.get("x-content-type-options") == "nosniff"
    assert resp.headers.get("x-frame-options") == "DENY"
    assert resp.headers.get("referrer-policy") == "strict-origin-when-cross-origin"


def test_cors_restriction_rejects_arbitrary_origins(client: TestClient) -> None:
    """Verify CORS headers do not grant access to untrusted external domains."""
    resp = client.options(
        "/api/status",
        headers={
            "Origin": "http://malicious-attacker-website.com",
            "Access-Control-Request-Method": "GET",
        },
    )
    # Origin should not be allowed
    assert resp.headers.get("access-control-allow-origin") != "http://malicious-attacker-website.com"
    assert resp.headers.get("access-control-allow-origin") != "*"


def test_image_upload_magic_bytes_valid_jpeg(client: TestClient) -> None:
    """Verify valid JPEG file passes magic bytes validation."""
    # Create simple 100x100 dummy image
    img = np.full((100, 100, 3), 120, dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    file_bytes = buf.tobytes()

    resp = client.post(
        "/api/detect/image",
        files={"file": ("test_face.jpg", io.BytesIO(file_bytes), "image/jpeg")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "faces_count" in data
    assert "download_url" in data
    assert data["download_url"].startswith("/results_files/screenshots/analyzed_")


def test_image_upload_rejects_empty_file(client: TestClient) -> None:
    """Verify empty image upload is rejected with HTTP 400."""
    resp = client.post(
        "/api/detect/image",
        files={"file": ("empty.jpg", io.BytesIO(b""), "image/jpeg")},
    )
    assert resp.status_code == 400
    assert "empty" in resp.json()["detail"].lower()


def test_image_upload_rejects_fake_extension(client: TestClient) -> None:
    """Verify file with .jpg extension but text/executable content is rejected."""
    fake_content = b"This is a plaintext script disguised as an image."
    resp = client.post(
        "/api/detect/image",
        files={"file": ("malicious.jpg", io.BytesIO(fake_content), "image/jpeg")},
    )
    assert resp.status_code == 400
    assert "signature" in resp.json()["detail"].lower()


def test_image_upload_sanitizes_path_traversal_filename(client: TestClient) -> None:
    """
    Verify malicious directory traversal in user filename is completely sanitized
    and saved as a clean UUID inside results/screenshots.
    """
    img = np.full((60, 60, 3), 150, dtype=np.uint8)
    _, buf = cv2.imencode(".png", img)

    malicious_filename = "../../../../../etc/cron.d/exploit.png"
    resp = client.post(
        "/api/detect/image",
        files={"file": (malicious_filename, io.BytesIO(buf.tobytes()), "image/png")},
    )
    assert resp.status_code == 200
    download_url = resp.json()["download_url"]
    # Path must remain strictly within screenshots directory
    assert ".." not in download_url
    assert "/results_files/screenshots/analyzed_" in download_url


def test_video_upload_rejects_invalid_container(client: TestClient) -> None:
    """Verify video upload rejects non-video streams."""
    fake_video = b"NOT_A_REAL_VIDEO_HEADER_123456789"
    resp = client.post(
        "/api/detect/video",
        files={"file": ("fake.mp4", io.BytesIO(fake_video), "video/mp4")},
    )
    assert resp.status_code == 400
    assert "invalid video" in resp.json()["detail"].lower()


def test_settings_validation_rejects_out_of_bounds(client: TestClient) -> None:
    """Verify Pydantic validation rejects out-of-bounds parameter values with 422."""
    # Confidence threshold > 1.0
    resp1 = client.post("/api/settings", json={"confidence_threshold": 1.5})
    assert resp1.status_code == 422

    # Confidence threshold < 0.0
    resp2 = client.post("/api/settings", json={"confidence_threshold": -0.1})
    assert resp2.status_code == 422

    # Face scale factor < 1.01
    resp3 = client.post("/api/settings", json={"face_scale_factor": 0.5})
    assert resp3.status_code == 422

    # Face min neighbors > 20
    resp4 = client.post("/api/settings", json={"face_min_neighbors": 50})
    assert resp4.status_code == 422


def test_settings_valid_update_and_atomic_persistence(client: TestClient) -> None:
    """Verify valid settings update succeeds and persists atomically."""
    orig_res = client.get("/api/settings")
    assert orig_res.status_code == 200
    orig_val = orig_res.json().get("confidence_threshold", 0.65)

    try:
        # Valid update
        update_res = client.post("/api/settings", json={"confidence_threshold": 0.60})
        assert update_res.status_code == 200
        assert update_res.json()["config"]["confidence_threshold"] == 0.60

        # Reload from fresh AppConfig to ensure file persistence
        reloaded = AppConfig.load()
        assert reloaded.confidence_threshold == 0.60
    finally:
        # Restore original value
        client.post("/api/settings", json={"confidence_threshold": orig_val})


def test_secure_results_rejects_path_traversal(client: TestClient) -> None:
    """Verify /api/results endpoint rejects path traversal sequences."""
    resp = client.get("/api/results/screenshots/..%2F..%2Frun_server.py")
    assert resp.status_code in {400, 404}
