import io

from PIL import Image, ImageOps, UnidentifiedImageError

MAX_UPLOAD_MB = 10
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_IMAGE_MEGAPIXELS = 20
MAX_IMAGE_PIXELS = MAX_IMAGE_MEGAPIXELS * 1_000_000


def decode_uploaded_image(content: bytes) -> Image.Image:
    if not content:
        raise ValueError("The uploaded file is empty. Choose a JPG or PNG photo.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError(f"The photo exceeds {MAX_UPLOAD_MB} MB. Choose a smaller image.")
    try:
        with Image.open(io.BytesIO(content)) as source:
            if source.format not in {"JPEG", "PNG"}:
                raise ValueError("Choose a JPG or PNG photo.")
            if source.width * source.height > MAX_IMAGE_PIXELS:
                raise ValueError(
                    f"The photo exceeds {MAX_IMAGE_MEGAPIXELS} megapixels. Resize it and try again."
                )
            source.verify()
        with Image.open(io.BytesIO(content)) as source:
            return ImageOps.exif_transpose(source).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ValueError(
            "This file could not be read as an image. Choose a valid JPG or PNG."
        ) from error
