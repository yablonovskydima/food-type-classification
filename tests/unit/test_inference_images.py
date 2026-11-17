"""Unit tests for decoding uploaded photos."""

import io

import pytest
from PIL import Image

from src.inference.images import decode_uploaded_image


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


def test_exif_orientation_is_applied() -> None:
    stream = io.BytesIO()
    exif = Image.Exif()
    exif[0x0112] = 6  # Orientation: rotate 90 degrees clockwise to display
    Image.new("RGB", (64, 48), "orange").save(stream, format="JPEG", exif=exif)

    image = decode_uploaded_image(stream.getvalue())

    assert image.size == (48, 64)
