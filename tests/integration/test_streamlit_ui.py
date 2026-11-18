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


def add_checkpoint(checkpoint_dir: Path, experiment: str = "resnet18_scratch") -> Path:
    path = checkpoint_dir / experiment / "best.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"stub checkpoint")
    return path


@pytest.fixture
def make_app(checkpoint_dir: Path):
    st.cache_resource.clear()
    yield lambda: AppTest.from_file(str(APP_FILE), default_timeout=20).run()
    st.cache_resource.clear()


class PredictorStub:
    calls: list[dict] = []
    image_size = 224
    normalization = "food11"
    metadata = {"epoch": 42, "metrics": {"macro_f1": 0.8123}}

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
    assert "No trained model found" in app.info[0].value
    app.file_uploader[0].upload("dish.png", image_bytes(), "image/png").run()
    assert not app.exception
    assert app.button[0].disabled
    assert not app.metric


def test_only_best_checkpoints_are_listed(
    make_app: AppFactory,
    checkpoint_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    predictor_stub: type[PredictorStub],
) -> None:
    add_checkpoint(checkpoint_dir, "resnet18_scratch")
    add_checkpoint(checkpoint_dir, "resnet18_pretrained")
    (checkpoint_dir / "resnet18_scratch" / "last.pt").write_bytes(b"training state")
    env_checkpoint = tmp_path / "elsewhere.pt"
    env_checkpoint.write_bytes(b"stub")
    monkeypatch.setenv("CHECKPOINT_PATH", str(env_checkpoint))

    app = make_app()

    options = app.selectbox[0].options
    assert len(options) == 3
    assert options[0].endswith("elsewhere.pt")  # CHECKPOINT_PATH is preselected
    assert not any(option.endswith("last.pt") for option in options)
    assert not app.info


def test_preprocessing_comes_from_the_checkpoint(
    make_app: AppFactory, checkpoint_dir: Path, predictor_stub: type[PredictorStub]
) -> None:
    add_checkpoint(checkpoint_dir)
    app = make_app()
    # Only model and device are selectable; size and normalisation are shown, not chosen.
    assert [box.label for box in app.selectbox] == ["Trained model", "Device"]
    description = " ".join(caption.value for caption in app.sidebar.caption)
    assert "224 px" in description and "food11" in description
    assert "epoch 42" in description and "0.812" in description


def test_ui_rejects_invalid_image(make_app: AppFactory) -> None:
    app = make_app()
    app.file_uploader[0].upload("dish.jpg", b"broken", "image/jpeg").run()
    assert not app.exception
    assert "valid JPG or PNG" in app.error[0].value
    assert not app.metric


def test_prediction_persists_topk_reruns_and_invalidates_on_input_changes(
    make_app: AppFactory, checkpoint_dir: Path, predictor_stub: type[PredictorStub]
) -> None:
    checkpoint = add_checkpoint(checkpoint_dir)
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
    assert not app.metric  # A new photo must never show stale predictions.
    app.button[0].click().run()
    assert app.metric[0].value == "Rice"
    assert len(predictor_stub.calls) == 1  # The model is cached, not reloaded per photo.

    checkpoint.write_bytes(b"retrained checkpoint")
    app.run()
    assert not app.metric
    assert len(predictor_stub.calls) == 2  # A retrained best.pt is loaded again.


def test_ui_shows_checkpoint_load_failure_reason(
    make_app: AppFactory, checkpoint_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken_predictor(**kwargs):
        raise RuntimeError("incompatible state dict")

    monkeypatch.setattr("src.inference.predictor.FoodPredictor", broken_predictor)
    add_checkpoint(checkpoint_dir)
    app = make_app()
    assert not app.exception
    assert "could not be loaded" in app.error[0].value
    assert "incompatible state dict" in app.error[0].value
    app.file_uploader[0].upload("dish.png", image_bytes(), "image/png").run()
    assert app.button[0].disabled
    assert not app.metric
