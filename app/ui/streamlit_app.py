import hashlib
import os
import pickle
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.images import (  # noqa: E402
    MAX_IMAGE_MEGAPIXELS,
    MAX_UPLOAD_MB,
    decode_uploaded_image,
)
from src.inference.predictor import FoodPredictor  # noqa: E402

LOAD_ERRORS = (OSError, ValueError, RuntimeError, pickle.UnpicklingError)


@st.cache_resource(show_spinner=False, max_entries=2)
def load_predictor(
    checkpoint_path: str,
    checkpoint_version: tuple[int, int],
    device: str,
) -> FoodPredictor:
    return FoodPredictor(
        checkpoint_path=checkpoint_path, device=None if device == "auto" else device
    )


def display_name(category: str) -> str:
    return category.replace("_", " ").title()


def find_checkpoints() -> dict[str, Path]:
    checkpoint_dir = Path(os.getenv("CHECKPOINT_DIR", PROJECT_ROOT / "checkpoints")).expanduser()
    paths = []
    if env_checkpoint := os.getenv("CHECKPOINT_PATH"):
        paths.append(Path(env_checkpoint).expanduser())
    if checkpoint_dir.is_dir():
        paths.extend(sorted(checkpoint_dir.rglob("best.pt")))

    checkpoints: dict[str, Path] = {}
    for path in paths:
        resolved = path.resolve()
        if not resolved.is_file() or resolved in checkpoints.values():
            continue
        try:
            label = str(resolved.relative_to(PROJECT_ROOT))
        except ValueError:
            label = str(resolved)
        checkpoints[label] = resolved
    return checkpoints


def checkpoint_version(path: Path) -> tuple[int, int] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return stat.st_mtime_ns, stat.st_size


def describe_model(predictor: FoodPredictor) -> str:
    metadata = getattr(predictor, "metadata", {}) or {}
    parts = [
        f"{getattr(predictor, 'image_size', '?')} px",
        f"{getattr(predictor, 'normalization', '?')} normalisation",
    ]
    if "epoch" in metadata:
        parts.append(f"epoch {metadata['epoch']}")
    macro_f1 = (metadata.get("metrics") or {}).get("macro_f1")
    if macro_f1 is not None:
        parts.append(f"val macro-F1 {macro_f1:.3f}")
    return " · ".join(parts)


def main() -> None:
    st.set_page_config(page_title="Food Lens · Food-11", page_icon="🍽️", layout="wide")
    st.title("Food Classifier")
    st.write("Upload a photo to explore the food categories the model recognises.")

    checkpoints = find_checkpoints()
    predictor = None
    with st.sidebar:
        st.header("Food classifier demo")
        st.caption("ResNet-18 · 11 food categories")
        checkpoint_label = st.selectbox(
            "Trained model", list(checkpoints), placeholder="No trained model found"
        )
        device = st.selectbox("Device", ["auto", "cpu", "cuda"])
        checkpoint = checkpoints.get(checkpoint_label) if checkpoint_label else None
        version = checkpoint_version(checkpoint) if checkpoint is not None else None
        if version is not None:
            try:
                with st.spinner("Loading the model…"):
                    predictor = load_predictor(str(checkpoint), version, device)
                st.caption(describe_model(predictor))
            except LOAD_ERRORS as error:
                st.error(f"The model could not be loaded.\n\nDetails: {error}")
        st.divider()
        st.caption("Photos stay in memory on the demo server and are not saved to disk.")

    if version is None:
        st.info(
            "No trained model found. Train one with `uv run scripts/train.py` "
            "(it saves `checkpoints/<experiment>/best.pt`) or set `CHECKPOINT_PATH`."
        )

    photo_column, result_column = st.columns(2, gap="large")
    with photo_column:
        st.subheader("Your photo")
        uploaded = st.file_uploader(
            "Choose a food photo",
            type=["jpg", "jpeg", "png"],
            max_upload_size=MAX_UPLOAD_MB,
            help=f"JPG or PNG, up to {MAX_UPLOAD_MB} MB and {MAX_IMAGE_MEGAPIXELS} megapixels.",
        )
        if uploaded is None:
            st.caption("Try a clear photo of one dish with the food filling the frame.")
            with result_column:
                st.subheader("Results")
                st.write("Your predictions will appear here.")
            return
        content = uploaded.getvalue()
        try:
            image = decode_uploaded_image(content)
        except ValueError as error:
            st.error(str(error))
            return
        st.image(image, width="stretch", alt="Uploaded food photo")

    with result_column:
        st.subheader("Results")
        top_k = st.radio(
            "Show predictions", [3, 5], horizontal=True, format_func=lambda k: f"Top {k}"
        )
        result_key = (
            hashlib.sha256(content).hexdigest(),
            str(checkpoint) if checkpoint is not None else "",
            version,
            device,
        )
        if st.button("Identify food", type="primary", disabled=predictor is None):
            st.session_state.pop("food_prediction", None)
            try:
                with st.spinner("Looking at your photo…"):
                    predictions = predictor.predict_image(image, top_k=5)
                st.session_state.food_prediction = {"key": result_key, "predictions": predictions}
            except LOAD_ERRORS as error:
                st.error(f"The model could not make a prediction.\n\nDetails: {error}")
                return
        result = st.session_state.get("food_prediction")
        if not result or result["key"] != result_key:
            st.caption("Choose Identify food to see the model's predictions.")
            return
        predictions = result["predictions"]
        winner = predictions[0]
        st.metric("Most likely category", display_name(winner["class"]))
        st.caption(f"Model confidence: {winner['probability']:.1%}")
        st.bar_chart(
            {
                "Category": [display_name(item["class"]) for item in predictions[:top_k]],
                "Confidence (%)": [item["probability"] * 100 for item in predictions[:top_k]],
            },
            x="Category",
            y="Confidence (%)",
            horizontal=True,
            sort="-Confidence (%)",
            color="#2e7d68",
            height=280,
        )
        st.caption(
            "Scores show the model's choices among 11 categories, even for unfamiliar foods."
        )


if __name__ == "__main__":
    main()
