from typing import Any, Dict
from PIL import Image
from PIL.ExifTags import TAGS
from app.core.logging import logger
from app.schemas.analysis import ImageMetadataResponse


class MetadataService:
    """
    Forensic Image Metadata & Provenance Extraction Service.
    
    Principles & Ethical Disclosures:
    - Metadata is supporting evidence only.
    - The presence or absence of EXIF/C2PA data NEVER determines authenticity alone.
    - C2PA byte marker detection indicates the presence of manifest structures,
      distinguished from cryptographic signature and trust chain verification.
    """

    # Common EXIF tags of interest
    RELEVANT_EXIF_TAGS = {
        "Make": "camera_make",
        "Model": "camera_model",
        "Software": "software",
        "DateTime": "datetime",
        "Flash": "flash",
        "FocalLength": "focal_length",
        "ISOSpeedRatings": "iso_speed",
    }

    @classmethod
    def extract_metadata(
        cls,
        image: Image.Image,
        raw_bytes: bytes,
        basic_meta: Dict[str, Any]
    ) -> ImageMetadataResponse:
        """
        Extracts structural, EXIF, and provenance indicators from image data.
        """
        has_exif = False
        exif_summary: Dict[str, Any] = {}

        # 1. EXIF Extraction
        try:
            exif_data = image.getexif()
            if exif_data:
                for tag_id, value in exif_data.items():
                    tag_name = TAGS.get(tag_id, tag_id)
                    if tag_name in cls.RELEVANT_EXIF_TAGS:
                        # Clean values to avoid binary leakage or complex objects
                        summary_key = cls.RELEVANT_EXIF_TAGS[tag_name]
                        exif_summary[summary_key] = str(value)
                if exif_summary:
                    has_exif = True
        except Exception as e:
            logger.debug(f"EXIF parsing skipped: {str(e)}")

        # 2. C2PA / Content Credentials Detection
        # Searches for JUMBF / C2PA byte signatures (e.g. 'c2pa' or 'jumb') in the raw header
        c2pa_detected = False
        if b"c2pa" in raw_bytes or b"jumb" in raw_bytes:
            c2pa_detected = True
            logger.info("C2PA / Content Credentials manifest signature marker detected.")

        # Cryptographic verification requires trust-root public key anchoring, not evaluated in MVP
        c2pa_verified = False

        return ImageMetadataResponse(
            format=basic_meta.get("format", "UNKNOWN"),
            width=basic_meta.get("width", image.width),
            height=basic_meta.get("height", image.height),
            file_size=basic_meta.get("file_size", len(raw_bytes)),
            has_exif=has_exif,
            exif_summary=exif_summary if has_exif else None,
            c2pa_detected=c2pa_detected,
            c2pa_verified=c2pa_verified,
            has_c2pa=c2pa_detected,
            note="Metadata provides supporting context only and does not determine authenticity."
        )

