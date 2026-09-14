import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Float, Integer, Text, DateTime, JSON
from app.db.database import Base


def utc_now():
    """Timezone-aware UTC timestamp generator."""
    return datetime.now(timezone.utc)


class AnalysisRecord(Base):
    """
    Database model representing a processed image analysis record.
    """
    __tablename__ = "analyses"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    filename = Column(String(255), nullable=False)
    stored_path = Column(String(512), nullable=False)
    verdict = Column(String(50), nullable=False)
    confidence = Column(Float, nullable=False)
    threshold = Column(Float, nullable=False)
    model_version = Column(String(50), nullable=False)
    processing_time_ms = Column(Integer, nullable=False)
    caption = Column(Text, nullable=True)
    
    # Calibration information
    is_calibrated = Column(Integer, default=0, nullable=False)  # 0 or 1 for SQLite
    calibration_method = Column(String(50), nullable=True)

    # Explainability & Evidence
    explanation_summary = Column(Text, nullable=True)
    explanation_cues = Column(JSON, nullable=True)
    heatmap_path = Column(String(512), nullable=True)

    # Image metadata & Provenance signals
    image_metadata = Column(JSON, nullable=True)

    created_at = Column(DateTime, default=utc_now, nullable=False, index=True)

