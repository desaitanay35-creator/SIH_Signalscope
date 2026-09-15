from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


def utc_now():
    return datetime.now(timezone.utc)


class HealthResponse(BaseModel):
    """Health check endpoint response schema."""
    status: str = Field(..., json_schema_extra={"example": "ok"})
    model_loaded: bool = Field(..., description="Indicates whether the ML detector weights are loaded and operational")


class ModelInfoResponse(BaseModel):
    """Detector metadata and state response schema."""
    name: str = Field(..., json_schema_extra={"example": "SignalScope Detector"})
    version: str = Field(..., json_schema_extra={"example": "signalscope-v1"})
    task: str = Field(..., json_schema_extra={"example": "real-vs-ai-generated"})
    loaded: bool = Field(..., description="Whether model weights are actively loaded in memory")
    device: Optional[str] = Field(None, json_schema_extra={"example": "cpu"})
    architecture: Optional[str] = Field(None, description="Registered model architecture id", json_schema_extra={"example": "rgb_frequency_fusion"})
    checkpoint_path: Optional[str] = Field(None, description="Filesystem path of the loaded checkpoint")


class CueItem(BaseModel):
    """Visual or statistical forensic cue item."""
    type: str = Field(..., description="Forensic cue classification, e.g. texture, frequency, boundary")
    description: str = Field(..., description="Human-readable description of detected artifact")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in cue detection")


class ExplanationResponse(BaseModel):
    """Grounded explanation and heatmap availability details."""
    summary: Optional[str] = Field(
        None,
        description="Grounded explanation summary based on model activations or explainability module"
    )
    cues: List[CueItem] = Field(default_factory=list, description="Forensic cue detections")
    heatmap_url: Optional[str] = Field(
        None,
        description="Relative API URL to download Grad-CAM heatmap if available"
    )
    heatmap_available: bool = Field(False, description="Whether a visual heatmap was generated")


class ImageMetadataResponse(BaseModel):
    """Extracted image structure and provenance signals."""
    format: str = Field(..., description="Detected image format (JPEG, PNG, WEBP)", json_schema_extra={"example": "JPEG"})
    width: int = Field(..., description="Pixel width", json_schema_extra={"example": 1024})
    height: int = Field(..., description="Pixel height", json_schema_extra={"example": 768})
    file_size: int = Field(..., description="File size in bytes", json_schema_extra={"example": 245890})
    has_exif: bool = Field(False, description="Presence of EXIF data")
    exif_summary: Optional[Dict[str, Any]] = Field(None, description="Sanitized basic EXIF attributes")
    
    # C2PA / Content Credentials Distinction
    c2pa_detected: bool = Field(
        False,
        description="Presence of raw C2PA / JUMBF metadata markers in file headers (supporting evidence only)"
    )
    c2pa_verified: bool = Field(
        False,
        description="Cryptographic validation of Content Credentials trust chain (requires public-key trust root; unsupported in MVP)"
    )
    has_c2pa: bool = Field(
        False,
        description="Convenience flag for raw C2PA manifest marker detection"
    )
    note: str = Field(
        "Metadata provides supporting context only and does not determine authenticity.",
        description="Responsible-use disclosure"
    )


class AnalysisResponse(BaseModel):
    """Primary analysis verdict and forensic report."""
    analysis_id: str = Field(..., description="Unique UUID identifier for this analysis")
    verdict: str = Field(
        ...,
        description="Likelihood assessment ('likely_ai_generated' or 'likely_real')",
        json_schema_extra={"example": "likely_ai_generated"}
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model confidence score for the stated verdict (e.g. 0.88 = 88% confidence)",
        json_schema_extra={"example": 0.88}
    )
    threshold: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Configured decision threshold used to partition probabilities",
        json_schema_extra={"example": 0.50}
    )
    model_version: str = Field(..., json_schema_extra={"example": "signalscope-v1"})
    processing_time_ms: int = Field(..., description="Total pipeline execution latency in milliseconds")
    
    # Calibration details
    is_calibrated: bool = Field(
        False,
        description="Indicates whether calibrated probabilities were produced via empirical calibration parameters"
    )
    calibration_method: Optional[str] = Field(None, description="Calibration method applied (or none)")

    explanation: ExplanationResponse
    metadata: ImageMetadataResponse
    created_at: datetime = Field(
        default_factory=utc_now,
        description="ISO-8601 UTC timestamp of analysis record creation"
    )


class AnalysisListItemResponse(BaseModel):
    """Brief summary item for analysis history listing."""
    analysis_id: str
    filename: str
    verdict: str
    confidence: float
    model_version: str
    processing_time_ms: int
    created_at: datetime


class AnalysisListResponse(BaseModel):
    """Paginated list of historical analyses."""
    items: List[AnalysisListItemResponse]
    total: int
    page: int
    page_size: int
