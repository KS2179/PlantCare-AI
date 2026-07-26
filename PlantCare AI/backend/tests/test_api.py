"""
Smoke tests that work even without a trained model checkpoint present --
they check routing, validation, and the disease-info lookup, and only
skip the parts that need a real model.

Run with: pytest tests/ -v
"""
import io

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.models.db import init_db
from app.services import disease_info as disease_info_service

# Explicitly ensure tables exist -- don't rely on TestClient triggering
# the app's lifespan startup event, since that behavior has changed
# across FastAPI/Starlette versions.
init_db()

client = TestClient(app)


def test_health_check():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_predict_rejects_bad_content_type():
    resp = client.post(
        "/api/predict",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert resp.status_code == 400


def test_predict_accepts_image_but_may_503_without_model():
    buf = io.BytesIO()
    Image.new("RGB", (100, 100), color=(0, 128, 0)).save(buf, format="JPEG")
    buf.seek(0)

    resp = client.post(
        "/api/predict",
        files={"file": ("leaf.jpg", buf, "image/jpeg")},
    )
    # 200 if a trained model is present, 503 with a clear message if not --
    # either is "correct" behavior for this test environment.
    assert resp.status_code in (200, 503)


def test_history_list_empty_ok():
    resp = client.get("/api/history")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_disease_info_known_class():
    info = disease_info_service.get_disease_info("Tomato___Late_blight")
    assert info["plant"] == "Tomato"
    assert info["is_healthy"] is False
    assert "Phytophthora" in info["causes"]


def test_disease_info_unknown_class_falls_back_gracefully():
    info = disease_info_service.get_disease_info("Mystery_Plant___Weird_disease")
    assert info["plant"] == "Mystery Plant"
    assert info["treatment"] is not None
