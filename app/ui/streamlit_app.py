"""Run with: uv run --extra app python -m streamlit run app/ui/streamlit_app.py."""

import hashlib
import os
import pickle
from pathlib import Path

import streamlit as st

from src.data.constants import DEFAULT_IMAGE_SIZE
from src.inference.images import decode_uploaded_image
from src.inference.nutrition import NUTRITION_REFERENCES
from src.inference.predictor import FoodPredictor


@st.cache_resource(show_spinner=False, max_entries=2)
def load_predictor(
    checkpoint_path: str,
    checkpoint_version: tuple[int, int],
    image_size: int,
    use_imagenet_stats: bool,
    device: str,
) -> FoodPredictor:
    """Invalidate cached weights when a checkpoint changes at the same path."""
    return FoodPredictor(
        checkpoint_path=checkpoint_path,
        image_size=image_size,
        use_imagenet_stats=use_imagenet_stats,
        device=None if device == "auto" else device,
    )


def display_name(category: str) -> str:
    return category.replace("_", " ").title()


def main() -> None:
    st.set_page_config(page_title="Food Lens · Food-11", page_icon="🍽️", layout="wide")
    st.title("Food Lens")
    st.markdown("### A food photo. Eleven possibilities.")
    st.write("Upload a photo to explore the food categories the model recognises.")

    with st.sidebar:
        st.header("Food-11 demo")
        st.caption("ResNet-18 · 11 food categories")
        with st.expander("Model settings", expanded=not bool(os.getenv("CHECKPOINT_PATH"))):
            checkpoint_text = st.text_input(
                "Trained checkpoint",
                value=os.getenv("CHECKPOINT_PATH", ""),
                placeholder="checkpoints/scratch_best.pt",
            )
            image_size = int(
                st.number_input(
                    "Input size (px)",
                    min_value=32,
                    max_value=1024,
                    value=DEFAULT_IMAGE_SIZE,
                    step=32,
                )
            )
            normalization = st.selectbox("Normalization", ["Food-11", "ImageNet"])
            device = st.selectbox("Device", ["auto", "cpu", "cuda"])
            st.caption(
                "Input size and normalization must match the checkpoint's training settings."
            )
        st.divider()
        st.caption("Photos stay in memory on the demo server and are not saved to disk.")

    checkpoint = Path(checkpoint_text).expanduser() if checkpoint_text.strip() else None
    checkpoint_version = None
    if checkpoint is not None:
        try:
            if checkpoint.is_file():
                stat = checkpoint.stat()
                checkpoint_version = (stat.st_mtime_ns, stat.st_size)
        except OSError:
            pass
    if checkpoint_version is None:
        st.info("Choose a trained checkpoint in Model settings to enable predictions.")

    photo_column, result_column = st.columns(2, gap="large")
    with photo_column:
        st.subheader("Your photo")
        uploaded = st.file_uploader(
            "Choose a food photo",
            type=["jpg", "jpeg", "png"],
            max_upload_size=10,
            help="JPG or PNG, up to 10 MB and 20 megapixels.",
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
            str(checkpoint.resolve()) if checkpoint is not None else "",
            checkpoint_version,
            image_size,
            normalization,
            device,
        )
        if st.button("Identify food", type="primary", disabled=checkpoint_version is None):
            # Clear prior results before a failed retry so stale predictions cannot appear.
            st.session_state.pop("food_prediction", None)
            try:
                with st.spinner("Looking at your photo…"):
                    predictor = load_predictor(
                        str(checkpoint.resolve()),
                        checkpoint_version,
                        image_size,
                        normalization == "ImageNet",
                        device,
                    )
                    predictions = predictor.predict_image(image, top_k=5)
                st.session_state.food_prediction = {"key": result_key, "predictions": predictions}
            except (OSError, ValueError, RuntimeError, pickle.UnpicklingError):
                st.error(
                    "The model could not make a prediction. Check the trained checkpoint, "
                    "preprocessing settings and selected device."
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

    st.divider()
    st.subheader("Inside this category")
    reference = NUTRITION_REFERENCES[winner["class"]]
    st.write(f"**Reference example: {reference.example} · per 100 g**")
    st.caption(
        "Reference values describe this example food, not the food or portion in your photo."
    )
    columns = st.columns(4)
    for column, label, value in zip(
        columns,
        ("Energy", "Protein", "Carbohydrate", "Fat"),
        (
            f"{reference.kcal:g} kcal",
            f"{reference.protein_g:g} g",
            f"{reference.carbohydrate_g:g} g",
            f"{reference.fat_g:g} g",
        ),
        strict=True,
    ):
        column.metric(label, value, border=True)
    st.write(reference.culinary_note)
    st.caption(f"Source: [USDA FoodData Central]({reference.source_url}) · SR Legacy")


if __name__ == "__main__":
    main()
