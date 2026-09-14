import io
from PIL import Image
from app.core.config import settings
from app.services.image_service import ImageService
from app.core.errors import InvalidImageError, UnsupportedImageTypeError, FileTooLargeError
import pytest


def test_corrupt_image_bytes_rejected():
    """Corrupted bytes must raise InvalidImageError."""
    corrupt_bytes = b"NOT_A_REAL_IMAGE_FILE_CONTENT_JUST_TEXT"
    with pytest.raises(InvalidImageError) as exc:
        ImageService.validate_and_decode(corrupt_bytes, "image/jpeg")
    assert exc.value.code == "INVALID_IMAGE"


def test_empty_file_rejected():
    """Empty payload must raise InvalidImageError."""
    with pytest.raises(InvalidImageError) as exc:
        ImageService.validate_and_decode(b"", "image/jpeg")
    assert exc.value.code == "INVALID_IMAGE"


def test_unsupported_image_format_rejected():
    """BMP or TIFF format must raise UnsupportedImageTypeError."""
    img = Image.new("RGB", (64, 64), color=(100, 100, 100))
    buf = io.BytesIO()
    img.save(buf, format="BMP")
    bmp_bytes = buf.getvalue()

    with pytest.raises(UnsupportedImageTypeError) as exc:
        ImageService.validate_and_decode(bmp_bytes, "image/bmp")
    assert exc.value.code == "UNSUPPORTED_IMAGE_TYPE"


def test_oversized_file_rejected(monkeypatch):
    """File exceeding MAX_UPLOAD_SIZE_BYTES must raise FileTooLargeError."""
    # Temporarily set max size to 500 bytes for fast testing
    monkeypatch.setattr(settings, "MAX_UPLOAD_SIZE_BYTES", 500)
    fake_large_bytes = b"X" * 600

    with pytest.raises(FileTooLargeError) as exc:
        ImageService.validate_and_decode(fake_large_bytes, "image/jpeg")
    assert exc.value.code == "FILE_TOO_LARGE"


def test_subminimum_dimension_rejected():
    """Image with dimensions below MIN_IMAGE_DIMENSION must raise InvalidImageError."""
    tiny_img = Image.new("RGB", (10, 10), color=(255, 255, 255))
    buf = io.BytesIO()
    tiny_img.save(buf, format="JPEG")
    tiny_bytes = buf.getvalue()

    with pytest.raises(InvalidImageError) as exc:
        ImageService.validate_and_decode(tiny_bytes, "image/jpeg")
    assert exc.value.code == "INVALID_IMAGE"


def test_valid_jpeg_png_webp_accepted(sample_jpeg_bytes, sample_png_bytes, sample_webp_bytes):
    """Supported image formats must decode successfully and standardize to RGB."""
    for img_bytes, expected_format in [
        (sample_jpeg_bytes, "JPEG"),
        (sample_png_bytes, "PNG"),
        (sample_webp_bytes, "WEBP"),
    ]:
        image, meta = ImageService.validate_and_decode(img_bytes)
        assert image.mode == "RGB"
        assert meta["format"] == expected_format
        assert meta["width"] > 0
        assert meta["height"] > 0

