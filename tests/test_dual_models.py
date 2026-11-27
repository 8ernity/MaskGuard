"""
Automated test suite for Dual AI Model Integration (Custom CNN + MobileNetV2):
- Independent preprocessing verification (128x128 [0, 1] vs 224x224 [-1, 1])
- Architecture build, forward shapes, and .h5 reload parity
- DualModelManager lazy loading, missing artifact safety, and zero silent fallback
- FastAPI model selection, persistence, per-request routing, generation safety,
  image inference, locked video job model_id, and model-scoped analytics
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import tensorflow as tf
from fastapi.testclient import TestClient

from app.core.mask_classifier import MaskClassifier
from app.core.mobilenet_classifier import MobileNetV2Classifier
from app.core.model_manager import DualModelManager, ModelSelectionError
from app.utils.config import AppConfig
from app.utils.image_utils import preprocess_face_mobilenetv2, preprocess_face_tensor
from app.utils.resource_paths import (
    get_class_mapping_path,
    get_dataset_dir,
    get_default_model_path,
    get_mobilenetv2_model_path,
)
from app.web.server import app
from training.mobilenetv2_architecture import build_mobilenetv2_model


def _make_synthetic_video_bytes(num_frames: int = 4, width: int = 160, height: int = 120) -> bytes:
    """Create a tiny valid MP4 byte payload using OpenCV."""
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf_file:
        tmp_path = Path(tf_file.name)
    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(tmp_path), fourcc, 10.0, (width, height))
        for _ in range(num_frames):
            frame = np.full((height, width, 3), 120, dtype=np.uint8)
            writer.write(frame)
        writer.release()
        return tmp_path.read_bytes()
    finally:
        tmp_path.unlink(missing_ok=True)


def test_independent_preprocessing_pipelines():
    """Verify Custom CNN ([0, 1] 128x128) and MobileNetV2 ([-1, 1] 224x224) preprocessing."""
    sample_bgr = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

    cnn_tensor = preprocess_face_tensor(sample_bgr, target_size=(128, 128))
    mn_tensor = preprocess_face_mobilenetv2(sample_bgr, target_size=(224, 224))

    assert cnn_tensor.shape == (128, 128, 3)
    assert cnn_tensor.dtype == np.float32
    assert float(np.min(cnn_tensor)) >= 0.0
    assert float(np.max(cnn_tensor)) <= 1.0

    assert mn_tensor.shape == (224, 224, 3)
    assert mn_tensor.dtype == np.float32
    assert float(np.min(mn_tensor)) >= -1.0
    assert float(np.max(mn_tensor)) <= 1.0

    # Verify MobileNetV2 preprocessing matches tf.keras.applications.mobilenet_v2.preprocess_input
    rgb = cv2.cvtColor(sample_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32)
    keras_ref = tf.keras.applications.mobilenet_v2.preprocess_input(resized)
    np.testing.assert_allclose(mn_tensor, keras_ref, rtol=1e-5, atol=1e-5)


def test_mobilenetv2_architecture_and_h5_reload(tmp_path: Path):
    """Verify MobileNetV2 build, valid output shape, .h5 serialization, and reload fidelity."""
    model = build_mobilenetv2_model(
        input_shape=(224, 224, 3),
        learning_rate=1e-4,
        freeze_backbone=True,
        weights=None,
    )
    assert model.input_shape == (None, 224, 224, 3)
    assert model.output_shape == (None, 1)

    dummy_x = np.random.uniform(-1.0, 1.0, (2, 224, 224, 3)).astype(np.float32)
    orig_pred = model(dummy_x, training=False).numpy()
    assert orig_pred.shape == (2, 1)
    assert np.all(orig_pred >= 0.0) and np.all(orig_pred <= 1.0)

    out_h5 = tmp_path / "test_mobilenetv2.h5"
    model.save(str(out_h5))
    reloaded = tf.keras.models.load_model(str(out_h5))
    reload_pred = reloaded(dummy_x, training=False).numpy()
    np.testing.assert_allclose(orig_pred, reload_pred, rtol=1e-5, atol=1e-5)


def test_class_mapping_consistency():
    """Verify canonical class mapping matches across Custom CNN and MobileNetV2."""
    cnn_map = json.loads(get_class_mapping_path().read_text(encoding="utf-8"))
    assert cnn_map == {"0": "with_mask", "1": "without_mask"}


def test_model_manager_missing_mobilenet_graceful_and_no_fallback(tmp_path: Path):
    """
    Missing MobileNetV2 artifact must NOT crash startup and must NOT silently fall back to Custom CNN.
    """
    missing_mn = tmp_path / "nonexistent_mobilenetv2.h5"
    mgr = DualModelManager(
        active_model_id="custom_cnn",
        custom_cnn_path=get_default_model_path(),
        mobilenetv2_path=missing_mn,
    )
    assert mgr.active_model_id == "custom_cnn"
    assert mgr.is_ready("custom_cnn") is True
    assert mgr.is_ready("mobilenetv2") is False

    with pytest.raises(ModelSelectionError):
        mgr.select_model("mobilenetv2")

    # Selection must remain custom_cnn
    assert mgr.active_model_id == "custom_cnn"

    with pytest.raises(ModelSelectionError):
        mgr.select_model("invalid_model_name")


def test_dual_model_manager_both_models(tmp_path: Path):
    """Test switching cleanly between Custom CNN and a MobileNetV2 artifact without mixing outputs."""
    # Use trained MobileNetV2 if present, else save a lightweight MobileNetV2 .h5 in tmp_path
    mn_path = get_mobilenetv2_model_path()
    if not mn_path.is_file():
        mn_path = tmp_path / "mock_mobilenetv2.h5"
        m = build_mobilenetv2_model(weights=None)
        m.save(str(mn_path))

    mgr = DualModelManager(
        active_model_id="custom_cnn",
        custom_cnn_path=get_default_model_path(),
        mobilenetv2_path=mn_path,
    )
    assert mgr.active_model_id == "custom_cnn"
    crop = np.full((120, 120, 3), 140, dtype=np.uint8)

    det_cnn = mgr.classify_crops([crop], [(10, 10, 50, 50)], model_id="custom_cnn")
    assert len(det_cnn) == 1
    assert det_cnn[0].model_id == "custom_cnn"

    info = mgr.select_model("mobilenetv2")
    assert info["model_id"] == "mobilenetv2"
    assert mgr.active_model_id == "mobilenetv2"

    det_mn = mgr.classify_crops([crop], [(10, 10, 50, 50)])
    assert len(det_mn) == 1
    assert det_mn[0].model_id == "mobilenetv2"


def test_backend_model_apis_and_routing():
    """Test /api/models, /api/models/active, /api/models/select, persistence, and per-request routing."""
    with TestClient(app) as client:
        orig_settings = client.get("/api/settings").json()
        orig_model = orig_settings.get("active_model", "custom_cnn")
        try:
            r_list = client.get("/api/models")
            assert r_list.status_code == 200
            data = r_list.json()
            ids = [m["model_id"] for m in data["models"]]
            assert "custom_cnn" in ids
            assert "mobilenetv2" in ids

            r_active = client.get("/api/models/active")
            assert r_active.status_code == 200
            assert r_active.json()["model_id"] in ("custom_cnn", "mobilenetv2")

            # Reject unknown model IDs
            r_bad = client.post("/api/models/select", json={"model_id": "yolo_v8"})
            assert r_bad.status_code == 400

            # Switch to custom_cnn explicitly
            r_sel = client.post("/api/models/select", json={"model_id": "custom_cnn"})
            assert r_sel.status_code == 200
            assert r_sel.json()["active_model"] == "custom_cnn"
            assert AppConfig.load().active_model == "custom_cnn"

            # Verify frame detection echoes requested model_id and generation_id
            dummy = np.full((180, 180, 3), 120, dtype=np.uint8)
            _, buf = cv2.imencode(".jpg", dummy)
            b64 = base64.b64encode(buf).decode("utf-8")
            r_frame = client.post(
                "/api/detect/frame",
                json={
                    "image": f"data:image/jpeg;base64,{b64}",
                    "model_id": "custom_cnn",
                    "session_id": "test_sess",
                    "generation_id": 7,
                },
            )
            assert r_frame.status_code == 200
            fjson = r_frame.json()
            assert fjson["model_id"] == "custom_cnn"
            assert fjson["generation_id"] == 7
            assert fjson["session_id"] == "test_sess"

            # Image detection respects model_id
            r_img = client.post(
                "/api/detect/image?model_id=custom_cnn",
                files={"file": ("face.jpg", io.BytesIO(buf.tobytes()), "image/jpeg")},
            )
            assert r_img.status_code == 200
            assert r_img.json()["model_id"] == "custom_cnn"

            # Analytics returns model-scoped metrics & comparison table
            r_ana = client.get("/api/analytics?model_id=custom_cnn")
            assert r_ana.status_code == 200
            ana_json = r_ana.json()
            assert ana_json["model_id"] == "custom_cnn"
            assert "comparison" in ana_json
            assert "custom_cnn" in ana_json["comparison"]
            assert "mobilenetv2" in ana_json["comparison"]

            # Video upload locks starting model_id
            vid_bytes = _make_synthetic_video_bytes()
            r_vid = client.post(
                "/api/detect/video?model_id=custom_cnn",
                files={"file": ("clip.mp4", io.BytesIO(vid_bytes), "video/mp4")},
            )
            assert r_vid.status_code == 200
            assert r_vid.json()["model_id"] == "custom_cnn"

            # If trained MobileNetV2 exists, test live switching & frame inference with it too
            if get_mobilenetv2_model_path().is_file():
                r_mn = client.post("/api/models/select", json={"model_id": "mobilenetv2"})
                assert r_mn.status_code == 200
                assert r_mn.json()["active_model"] == "mobilenetv2"
                assert AppConfig.load().active_model == "mobilenetv2"

                r_frame_mn = client.post(
                    "/api/detect/frame",
                    json={
                        "image": f"data:image/jpeg;base64,{b64}",
                        "model_id": "mobilenetv2",
                        "generation_id": 8,
                    },
                )
                assert r_frame_mn.status_code == 200
                assert r_frame_mn.json()["model_id"] == "mobilenetv2"
                assert r_frame_mn.json()["generation_id"] == 8
        finally:
            client.post("/api/models/select", json={"model_id": orig_model})


def test_real_labeled_images_both_models():
    """Verify real labeled test images classify properly with Custom CNN (and MobileNetV2 if trained)."""
    test_dir = get_dataset_dir() / "Test"
    with_mask_imgs = sorted((test_dir / "WithMask").glob("*.png"))[:20]
    without_mask_imgs = sorted((test_dir / "WithoutMask").glob("*.png"))[:20]
    if not with_mask_imgs or not without_mask_imgs:
        pytest.skip("Local Kaggle Test images not available.")

    cnn = MaskClassifier(confidence_threshold=0.50, uncertain_band=0.0)
    mask_crops = [cv2.imread(str(p)) for p in with_mask_imgs]
    nomask_crops = [cv2.imread(str(p)) for p in without_mask_imgs]

    cnn_mask_preds = cnn.classify_crops(mask_crops, [(0, 0, 50, 50)] * len(mask_crops))
    cnn_nomask_preds = cnn.classify_crops(nomask_crops, [(0, 0, 50, 50)] * len(nomask_crops))
    assert sum(1 for d in cnn_mask_preds if d.label == "MASK") >= 16
    assert sum(1 for d in cnn_nomask_preds if d.label == "NO MASK") >= 18

    mn_path = get_mobilenetv2_model_path()
    if mn_path.is_file():
        mn = MobileNetV2Classifier(model_path=mn_path, confidence_threshold=0.50, uncertain_band=0.0)
        mn_mask_preds = mn.classify_crops(mask_crops, [(0, 0, 50, 50)] * len(mask_crops))
        mn_nomask_preds = mn.classify_crops(nomask_crops, [(0, 0, 50, 50)] * len(nomask_crops))
        assert sum(1 for d in mn_mask_preds if d.label == "MASK") >= 16
        assert sum(1 for d in mn_nomask_preds if d.label == "NO MASK") >= 18
