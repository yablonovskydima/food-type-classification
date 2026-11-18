import io
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from app.api.main import app
from src.models import build_model


@pytest.fixture
def checkpoint(tmp_path: Path) -> Path:
    path = tmp_path / "best.pt"
    torch.save({"model_state_dict": build_model().state_dict()}, path)
    return path


@pytest.fixture
def client(checkpoint: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CHECKPOINT_PATH", str(checkpoint))
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def client_without_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CHECKPOINT_PATH", str(tmp_path / "missing.pt"))
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_image_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color=(100, 150, 200)).save(buf, format="JPEG")
    return buf.getvalue()


def test_health_check_endpoint(client: TestClient, checkpoint: Path) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["checkpoint"] == str(checkpoint)
    assert data["error"] is None


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
        assert {"class", "probability", "index"} <= item.keys()
        assert 0.0 <= item["probability"] <= 1.0


def test_predict_with_custom_top_k(client: TestClient, sample_image_bytes: bytes) -> None:
    response = client.post(
        "/predict?top_k=5",
        files={"file": ("food.jpg", sample_image_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 5


def test_predict_invalid_image_file(client: TestClient) -> None:
    response = client.post(
        "/predict",
        files={"file": ("not_an_image.txt", b"plain text content", "text/plain")},
    )
    assert response.status_code == 400
    assert "could not be read as an image" in response.json()["detail"]


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


def test_without_checkpoint_api_refuses_to_predict(
    client_without_model: TestClient, sample_image_bytes: bytes
) -> None:
    health = client_without_model.get("/health").json()
    assert health["model_loaded"] is False
    assert "not found" in health["error"]

    response = client_without_model.post(
        "/predict",
        files={"file": ("food.jpg", sample_image_bytes, "image/jpeg")},
    )
    assert response.status_code == 503
