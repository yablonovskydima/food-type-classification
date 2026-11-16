"""Decode uploaded food photos without writing them to disk."""

import io

from PIL import Image, ImageOps, UnidentifiedImageError

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000


def decode_uploaded_image(content: bytes) -> Image.Image:
    """Validate JPEG/PNG contents and return an EXIF-oriented RGB image."""
    if not content:
        raise ValueError("The uploaded file is empty. Choose a JPG or PNG photo.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError("The photo exceeds 10 MB. Choose a smaller image.")
    try:
        with Image.open(io.BytesIO(content)) as source:
            if source.format not in {"JPEG", "PNG"}:
                raise ValueError("Choose a JPG or PNG photo.")
            if source.width * source.height > MAX_IMAGE_PIXELS:
                raise ValueError("The photo exceeds 20 megapixels. Resize it and try again.")
            source.verify()
        with Image.open(io.BytesIO(content)) as source:
            return ImageOps.exif_transpose(source).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ValueError(
            "This file could not be read as an image. Choose a valid JPG or PNG."
        ) from error
