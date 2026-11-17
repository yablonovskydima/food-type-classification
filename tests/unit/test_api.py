import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.api.main import app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_image_bytes() -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", (64, 64), color=(100, 150, 200))
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_health_check_endpoint(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data


def test_predict_valid_image(client: TestClient, sample_image_bytes: bytes) -> None:
    response = client.post(
        "/predict",
        files={"file": ("food.jpg", sample_image_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "food.jpg"
    assert len(data["predictions"]) == 3
    for item in data["predictions"]:
        assert "class" in item
        assert "probability" in item
        assert "index" in item
        assert 0.0 <= item["probability"] <= 1.0


def test_predict_with_custom_top_k(client: TestClient, sample_image_bytes: bytes) -> None:
    response = client.post(
        "/predict?top_k=5",
        files={"file": ("food.jpg", sample_image_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["predictions"]) == 5


def test_predict_invalid_image_file(client: TestClient) -> None:
    response = client.post(
        "/predict",
        files={"file": ("not_an_image.txt", b"plain text content", "text/plain")},
    )
    assert response.status_code == 400
    assert "Invalid image" in response.json()["detail"]


def test_predict_empty_file(client: TestClient) -> None:
    response = client.post(
        "/predict",
        files={"file": ("empty.jpg", b"", "image/jpeg")},
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"]


def test_predict_top_k_bounds(client: TestClient, sample_image_bytes: bytes) -> None:
    response = client.post(
        "/predict?top_k=20",
        files={"file": ("food.jpg", sample_image_bytes, "image/jpeg")},
    )
    assert response.status_code == 422


def test_predict_without_loaded_model_returns_unavailable(
    client: TestClient, sample_image_bytes: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("app.api.main.predictor", None)
    response = client.post(
        "/predict", files={"file": ("food.jpg", sample_image_bytes, "image/jpeg")}
    )
    assert response.status_code == 503
    assert response.json()["detail"] == "Model not loaded"
