import io
from collections.abc import Callable
from pathlib import Path

import pytest
import streamlit as st
from PIL import Image
from streamlit.testing.v1 import AppTest

from src.data.constants import CLASSES

APP_FILE = Path(__file__).resolve().parents[2] / "app" / "ui" / "streamlit_app.py"

AppFactory = Callable[[], AppTest]


def image_bytes(color: str = "orange") -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (64, 48), color).save(stream, format="PNG")
    return stream.getvalue()


@pytest.fixture
def checkpoint_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    directory = tmp_path / "checkpoints"
    directory.mkdir()
    monkeypatch.setenv("CHECKPOINT_DIR", str(directory))
    monkeypatch.delenv("CHECKPOINT_PATH", raising=False)
    return directory


@pytest.fixture
def make_app(checkpoint_dir: Path):
    st.cache_resource.clear()
    yield lambda: AppTest.from_file(str(APP_FILE), default_timeout=20).run()
    st.cache_resource.clear()


class PredictorStub:
    calls: list[dict] = []

    def __init__(self, **kwargs):
        PredictorStub.calls.append(kwargs)

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


@pytest.fixture
def predictor_stub(monkeypatch: pytest.MonkeyPatch) -> type[PredictorStub]:
    PredictorStub.calls = []
    monkeypatch.setattr("src.inference.predictor.FoodPredictor", PredictorStub)
    return PredictorStub


def test_ui_requires_trained_checkpoint(make_app: AppFactory) -> None:
    app = make_app()
    assert not app.exception
    assert "No trained checkpoint found" in app.info[0].value
    app.file_uploader[0].upload("dish.png", image_bytes(), "image/png").run()
    assert not app.exception
    assert app.button[0].disabled
    assert not app.metric


def test_checkpoints_are_discovered(
    make_app: AppFactory, checkpoint_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (checkpoint_dir / "scratch_best.pt").write_bytes(b"stub")
    (checkpoint_dir / "runs" / "pretrained_best.pth").parent.mkdir()
    (checkpoint_dir / "runs" / "pretrained_best.pth").write_bytes(b"stub")
    (checkpoint_dir / "notes.txt").write_text("not a checkpoint")
    env_checkpoint = tmp_path / "elsewhere.pt"
    env_checkpoint.write_bytes(b"stub")
    monkeypatch.setenv("CHECKPOINT_PATH", str(env_checkpoint))

    app = make_app()

    options = app.selectbox[0].options
    assert len(options) == 3
    assert options[0].endswith("elsewhere.pt")  # CHECKPOINT_PATH is preselected
    assert any(option.endswith("scratch_best.pt") for option in options)
    assert any(option.endswith("pretrained_best.pth") for option in options)
    assert not app.info


def test_input_size_is_limited_to_training_sizes(make_app: AppFactory) -> None:
    app = make_app()
    assert app.selectbox[1].options == ["128", "224"]
    assert app.selectbox[1].value == 224


def test_ui_rejects_invalid_image(make_app: AppFactory) -> None:
    app = make_app()
    app.file_uploader[0].upload("dish.jpg", b"broken", "image/jpeg").run()
    assert not app.exception
    assert "valid JPG or PNG" in app.error[0].value
    assert not app.metric


def test_prediction_persists_topk_reruns_and_invalidates_on_input_changes(
    make_app: AppFactory, checkpoint_dir: Path, predictor_stub: type[PredictorStub]
) -> None:
    checkpoint = checkpoint_dir / "model.pt"
    checkpoint.write_bytes(b"stub checkpoint")
    app = make_app()
    app.file_uploader[0].upload("dish.png", image_bytes(), "image/png").run()
    app.button[0].click().run()
    assert not app.exception
    assert app.metric[0].value == "Rice"
    assert any("60.0%" in c.value for c in app.caption)
    assert len(predictor_stub.calls) == 1

    app.radio[0].set_value(5).run()
    assert not app.exception
    assert app.metric[0].value == "Rice"
    assert len(predictor_stub.calls) == 1

    app.file_uploader[0].upload("other.png", image_bytes(color="green"), "image/png").run()
    assert not app.metric
    app.button[0].click().run()
    assert app.metric[0].value == "Rice"
    assert len(predictor_stub.calls) == 1

    checkpoint.write_bytes(b"new stub checkpoint")
    app.run()
    assert not app.metric
    app.button[0].click().run()
    assert len(predictor_stub.calls) == 2


def test_ui_shows_checkpoint_load_failure_reason(
    make_app: AppFactory, checkpoint_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken_predictor(**kwargs):
        raise RuntimeError("incompatible state dict")

    monkeypatch.setattr("src.inference.predictor.FoodPredictor", broken_predictor)
    (checkpoint_dir / "broken.pt").write_bytes(b"stub")
    app = make_app()
    app.file_uploader[0].upload("dish.png", image_bytes(), "image/png").run()
    app.button[0].click().run()
    assert not app.exception
    assert "could not make a prediction" in app.error[0].value
    assert "incompatible state dict" in app.error[0].value
    assert not app.metric
