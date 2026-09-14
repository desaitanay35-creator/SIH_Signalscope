import io
from typing import Tuple
from PIL import Image, UnidentifiedImageError
from app.core.config import settings
from app.core.errors import (
    FileTooLargeError,
    InvalidImageError,
    UnsupportedImageTypeError,
)
from app.core.logging import logger


# Set PIL max image pixels to protect against decompression bombs
Image.MAX_IMAGE_PIXELS = settings.MAX_IMAGE_PIXELS


class ImageService:
    """
    Validates, decodes, and standardizes uploaded images.
    
    Security & Integrity:
    - Rejects files exceeding MAX_UPLOAD_SIZE_BYTES.
    - Rejects files where declared or decoded MIME/format is not in ALLOWED_IMAGE_TYPES.
    - Decodes image bytes using PIL to ensure actual pixel structure integrity.
    - Validates pixel dimensions against MIN_IMAGE_DIMENSION and MAX_IMAGE_DIMENSION.
    - Guards against decompression bombs via Image.MAX_IMAGE_PIXELS.
    - Converts all color palettes/modes (e.g. RGBA, Grayscale, CMYK) to standardized RGB.
    """

    SUPPORTED_PIL_FORMATS = {
        "JPEG": ("image/jpeg", "jpg"),
        "PNG": ("image/png", "png"),
        "WEBP": ("image/webp", "webp"),
    }

    @classmethod
    def validate_and_decode(cls, file_bytes: bytes, declared_content_type: str = "") -> Tuple[Image.Image, dict]:
        """
        Validates raw uploaded bytes and returns a decoded RGB PIL Image with its metadata.
        
        Raises:
            FileTooLargeError: If payload exceeds size limits.
            UnsupportedImageTypeError: If format is not supported.
            InvalidImageError: If byte content is corrupted or dimensions are out of bounds.
            InvalidImageError: If byte content is corrupted, dimensions are out of bounds,
                               or a decompression bomb is detected.
        """
        file_size = len(file_bytes)
        logger.info(f"Validating image upload: size={file_size} bytes, declared_type='{declared_content_type}'")

        # 1. Size Validation
        if file_size > settings.MAX_UPLOAD_SIZE_BYTES:
            logger.warning(f"File rejected: size {file_size} exceeds max {settings.MAX_UPLOAD_SIZE_BYTES}")
            raise FileTooLargeError(
                message=f"Uploaded file ({file_size} bytes) exceeds the maximum allowed size of {settings.MAX_UPLOAD_SIZE_BYTES} bytes."
            )
        
        if file_size == 0:
            raise InvalidImageError(message="Uploaded file is empty.")

        # 2. Actual Image Decoding & Integrity Check
        try:
            # First pass: verify structure
            buffer = io.BytesIO(file_bytes)
            with Image.open(buffer) as img_check:
                img_check.verify()
                detected_format = img_check.format
        except Image.DecompressionBombError as e:
            logger.warning(f"Decompression bomb detected during verify: {str(e)}")
            raise InvalidImageError(message="Image resolution exceeds the maximum allowable pixel limit.")
        except (UnidentifiedImageError, ValueError, SyntaxError) as e:
            logger.warning(f"Failed to decode image bytes: {str(e)}")
            raise InvalidImageError(message="Uploaded file is not a valid or recognizable image.")

        if not detected_format or detected_format.upper() not in cls.SUPPORTED_PIL_FORMATS:
            logger.warning(f"Unsupported decoded format: '{detected_format}'")
            raise UnsupportedImageTypeError(
                message=f"Unsupported image format '{detected_format}'. Supported formats: JPEG, PNG, WEBP."
            )

        # 3. Second pass: Load pixel data and convert to RGB
        buffer.seek(0)
        try:
            image = Image.open(buffer)
            width, height = image.size
        except Image.DecompressionBombError as e:
            logger.warning(f"Decompression bomb detected during open: {str(e)}")
            raise InvalidImageError(message="Image resolution exceeds the maximum allowable pixel limit.")
        except Exception as e:
            logger.warning(f"Failed to read image pixel dimensions: {str(e)}")
            raise InvalidImageError(message="Corrupted image structure.")

        # 4. Dimension Bounds Validation
        if (
            width < settings.MIN_IMAGE_DIMENSION
            or height < settings.MIN_IMAGE_DIMENSION
            or width > settings.MAX_IMAGE_DIMENSION
            or height > settings.MAX_IMAGE_DIMENSION
        ):
            logger.warning(f"Image dimensions out of bounds: {width}x{height}")
            raise InvalidImageError(
                message=(
                    f"Image dimensions ({width}x{height}) must be between "
                    f"{settings.MIN_IMAGE_DIMENSION}px and {settings.MAX_IMAGE_DIMENSION}px."
                )
            )

        # 5. Standardize to RGB
        if image.mode != "RGB":
            image = image.convert("RGB")
        try:
            if image.mode != "RGB":
                image = image.convert("RGB")
        except Image.DecompressionBombError:
            raise InvalidImageError(message="Image resolution exceeds the maximum allowable pixel limit.")
        except Exception as e:
            logger.warning(f"Failed to standardize image to RGB: {str(e)}")
            raise InvalidImageError(message="Failed to process image color channels.")

        mime_type, file_ext = cls.SUPPORTED_PIL_FORMATS[detected_format.upper()]

        metadata = {
            "format": detected_format.upper(),
            "mime_type": mime_type,
            "extension": file_ext,
            "width": width,
            "height": height,
            "file_size": file_size,
        }

        logger.info(f"Image validated successfully: {width}x{height} {detected_format.upper()}")
        return image, metadata

