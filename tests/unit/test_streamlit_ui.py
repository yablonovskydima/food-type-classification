"""Exercise upload, prediction, and rerun behavior without real weights or a GPU."""

import io
from pathlib import Path

import pytest
import streamlit as st
from PIL import Image
from streamlit.testing.v1 import AppTest

from src.data.constants import CLASSES
from src.inference.images import decode_uploaded_image
from src.inference.nutrition import NUTRITION_REFERENCES

APP_FILE = Path(__file__).resolve().parents[2] / "app" / "ui" / "streamlit_app.py"


def image_bytes(format: str = "PNG", color: str = "orange") -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (64, 48), color).save(stream, format=format)
    return stream.getvalue()


@pytest.mark.parametrize("format", ["JPEG", "PNG"])
def test_image_decoding(format: str) -> None:
    image = decode_uploaded_image(image_bytes(format))
    assert image.size == (64, 48)
    assert image.mode == "RGB"


@pytest.mark.parametrize("content", [b"", b"not an image", image_bytes("GIF")])
def test_invalid_uploads_are_rejected(content: bytes) -> None:
    with pytest.raises(ValueError):
        decode_uploaded_image(content)


def test_upload_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    content = image_bytes()
    monkeypatch.setattr("src.inference.images.MAX_UPLOAD_BYTES", 1)
    with pytest.raises(ValueError, match="10 MB"):
        decode_uploaded_image(content)
    monkeypatch.setattr("src.inference.images.MAX_UPLOAD_BYTES", len(content))
    monkeypatch.setattr("src.inference.images.MAX_IMAGE_PIXELS", 1)
    with pytest.raises(ValueError, match="20 megapixels"):
        decode_uploaded_image(content)


def test_reference_foods_cover_every_category() -> None:
    assert set(NUTRITION_REFERENCES) == set(CLASSES)
    rice = NUTRITION_REFERENCES["rice"]
    assert rice.kcal == 130
    assert rice.carbohydrate_g == 28.2
    assert str(rice.fdc_id) in rice.source_url


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("CHECKPOINT_PATH", raising=False)
    st.cache_resource.clear()
    yield AppTest.from_file(str(APP_FILE), default_timeout=20).run()
    st.cache_resource.clear()


def test_ui_requires_trained_checkpoint(app: AppTest) -> None:
    assert not app.exception
    assert "trained checkpoint" in app.info[0].value
    app.file_uploader[0].upload("dish.png", image_bytes(), "image/png").run()
    assert not app.exception
    assert app.button[0].disabled
    assert not app.metric


def test_ui_rejects_invalid_image(app: AppTest) -> None:
    app.file_uploader[0].upload("dish.jpg", b"broken", "image/jpeg").run()
    assert not app.exception
    assert "valid JPG or PNG" in app.error[0].value
    assert not app.metric


def test_prediction_persists_topk_reruns_and_invalidates_on_input_changes(
    app: AppTest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    class PredictorStub:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def predict_image(self, image, top_k):
            assert image.mode == "RGB"
            assert top_k == 5
            return [
                {"class": category, "probability": probability, "index": CLASSES.index(category)}
                for category, probability in [
                    ("rice", 0.6),
                    ("noodles_pasta", 0.2),
                    ("bread", 0.1),
                    ("soup", 0.06),
                    ("dessert", 0.04),
                ]
            ]

    monkeypatch.setattr("src.inference.predictor.FoodPredictor", PredictorStub)
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"stub checkpoint")
    app.text_input[0].set_value(str(checkpoint)).run()
    app.file_uploader[0].upload("dish.png", image_bytes(), "image/png").run()
    app.button[0].click().run()
    assert not app.exception
    assert app.metric[0].value == "Rice"
    assert app.metric[1].value == "130 kcal"
    assert any("60.0%" in c.value for c in app.caption)
    assert any("not the food or portion" in c.value for c in app.caption)
    assert len(calls) == 1
    app.radio[0].set_value(5).run()
    assert not app.exception
    assert app.metric[0].value == "Rice"
    assert len(calls) == 1
    app.file_uploader[0].upload("other.png", image_bytes(color="green"), "image/png").run()
    assert not app.metric  # A new photo must never show stale predictions.
    app.button[0].click().run()
    assert app.metric[0].value == "Rice"
    assert len(calls) == 1  # Model is reused; photos are not globally cached.
    checkpoint.write_bytes(b"new stub checkpoint")
    app.run()
    assert not app.metric
    app.button[0].click().run()
    assert len(calls) == 2


def test_ui_handles_checkpoint_load_failure(
    app: AppTest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken_predictor(**kwargs):
        raise RuntimeError("incompatible state dict")

    monkeypatch.setattr("src.inference.predictor.FoodPredictor", broken_predictor)
    checkpoint = tmp_path / "broken.pt"
    checkpoint.write_bytes(b"stub")
    app.text_input[0].set_value(str(checkpoint)).run()
    app.file_uploader[0].upload("dish.png", image_bytes(), "image/png").run()
    app.button[0].click().run()
    assert not app.exception
    assert "could not make a prediction" in app.error[0].value
    assert not app.metric
