import hashlib
import os
import pickle
from pathlib import Path

import streamlit as st

from src.data.constants import DEFAULT_IMAGE_SIZE
from src.inference.images import MAX_IMAGE_MEGAPIXELS, MAX_UPLOAD_MB, decode_uploaded_image
from src.inference.predictor import FoodPredictor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_EXTENSIONS = {".pt", ".pth"}
IMAGE_SIZES = (128, 224)


@st.cache_resource(show_spinner=False, max_entries=2)
def load_predictor(
    checkpoint_path: str,
    checkpoint_version: tuple[int, int],
    image_size: int,
    use_imagenet_stats: bool,
    device: str,
) -> FoodPredictor:
    return FoodPredictor(
        checkpoint_path=checkpoint_path,
        image_size=image_size,
        use_imagenet_stats=use_imagenet_stats,
        device=None if device == "auto" else device,
    )


def display_name(category: str) -> str:
    return category.replace("_", " ").title()


def find_checkpoints() -> dict[str, Path]:
    checkpoint_dir = Path(os.getenv("CHECKPOINT_DIR", PROJECT_ROOT / "checkpoints")).expanduser()
    paths = []
    if env_checkpoint := os.getenv("CHECKPOINT_PATH"):
        paths.append(Path(env_checkpoint).expanduser())
    if checkpoint_dir.is_dir():
        paths.extend(
            sorted(p for p in checkpoint_dir.rglob("*") if p.suffix in CHECKPOINT_EXTENSIONS)
        )

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


def main() -> None:
    st.set_page_config(page_title="Food Lens · Food-11", page_icon="🍽️", layout="wide")
    st.title("Food Lens")
    st.markdown("### A food photo. Eleven possibilities.")
    st.write("Upload a photo to explore the food categories the model recognises.")

    checkpoints = find_checkpoints()
    with st.sidebar:
        st.header("Food-11 demo")
        st.caption("ResNet-18 · 11 food categories")
        with st.expander("Model settings", expanded=True):
            checkpoint_label = st.selectbox(
                "Trained checkpoint",
                list(checkpoints),
                placeholder="No checkpoints found",
            )
            image_size = st.selectbox(
                "Input size (px)", IMAGE_SIZES, index=IMAGE_SIZES.index(DEFAULT_IMAGE_SIZE)
            )
            normalization = st.selectbox("Normalization", ["Food-11", "ImageNet"])
            device = st.selectbox("Device", ["auto", "cpu", "cuda"])
            st.caption(
                "Input size and normalization must match the checkpoint's training settings: "
                "Food-11 for the from-scratch model, ImageNet for the pretrained reference."
            )
        st.divider()
        st.caption("Photos stay in memory on the demo server and are not saved to disk.")

    checkpoint = checkpoints.get(checkpoint_label) if checkpoint_label else None
    version = checkpoint_version(checkpoint) if checkpoint is not None else None
    if version is None:
        st.info(
            "No trained checkpoint found. Put a `.pt` file into `checkpoints/` "
            "or set `CHECKPOINT_PATH` to enable predictions."
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
                st.subheader("What could it be?")
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
        st.subheader("What could it be?")
        top_k = st.radio(
            "Show predictions", [3, 5], horizontal=True, format_func=lambda k: f"Top {k}"
        )
        result_key = (
            hashlib.sha256(content).hexdigest(),
            str(checkpoint) if checkpoint is not None else "",
            version,
            image_size,
            normalization,
            device,
        )
        if st.button("Identify food", type="primary", disabled=version is None):
            st.session_state.pop("food_prediction", None)
            try:
                with st.spinner("Looking at your photo…"):
                    predictor = load_predictor(
                        str(checkpoint),
                        version,
                        image_size,
                        normalization == "ImageNet",
                        device,
                    )
                    predictions = predictor.predict_image(image, top_k=5)
                st.session_state.food_prediction = {"key": result_key, "predictions": predictions}
            except (OSError, ValueError, RuntimeError, pickle.UnpicklingError) as error:
                st.error(
                    "The model could not make a prediction. Check the trained checkpoint, "
                    f"preprocessing settings and selected device.\n\nDetails: {error}"
                )
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
