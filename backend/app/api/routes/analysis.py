from typing import Optional
from fastapi import APIRouter, Depends, File, Form, Query, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.schemas.analysis import AnalysisResponse, AnalysisListResponse
from app.schemas.common import ErrorResponse
from app.services.analysis_service import AnalysisService

router = APIRouter()
analysis_service = AnalysisService()


@router.post(
    "/analyze",
    response_model=AnalysisResponse,
    status_code=200,
    tags=["Analysis"],
    summary="Analyze image for synthetic/AI-generated indicators",
    responses={
        200: {
            "model": AnalysisResponse,
            "description": "Successful media forensics likelihood analysis.",
        },
        400: {
            "model": ErrorResponse,
            "description": "Invalid image binary structure (INVALID_IMAGE) or unsupported format (UNSUPPORTED_IMAGE_TYPE).",
        },
        413: {
            "model": ErrorResponse,
            "description": "Upload size exceeds configured maximum limit of 10MB (FILE_TOO_LARGE).",
        },
        500: {
            "model": ErrorResponse,
            "description": "Detector returned non-numeric/unbounded probability (INVALID_MODEL_OUTPUT) or inference failed (INFERENCE_ERROR).",
        },
        503: {
            "model": ErrorResponse,
            "description": "ML detector is unavailable or model weights are not loaded (MODEL_UNAVAILABLE).",
        },
    },
)
async def analyze_image(
    image: UploadFile = File(
        ...,
        description="Uploaded image file. Accepted formats: JPEG, PNG, WEBP. Max size: 10MB."
    ),
    caption: Optional[str] = Form(
        None,
        description="Optional textual context or caption accompanying the image."
    ),
    db: Session = Depends(get_db)
):
    """
    Forensic analysis endpoint for determining whether media is Likely AI-Generated or Likely Real.
    
    ### Request:
    - **image** (*required*): Multipart binary file stream. Supported formats: JPEG, PNG, WEBP.
    - **caption** (*optional*): Accompanying text string.
    
    ### Pipeline Processing:
    1. Validates and decodes image bytes (PIL integrity check & dimension constraints: 32px to 8192px).
    2. Extracts image format, dimensions, EXIF data, and C2PA provenance markers.
    3. Verifies detector availability (`MODEL_UNAVAILABLE` 503 if weights are absent).
    4. Saves original image using a randomized UUID path.
    5. Normalizes image into configurable tensor dimensions.
    6. Executes detector forward pass and strictly validates output probabilities.
    7. Computes calibrated likelihood assessment (`likely_ai_generated` or `likely_real`).
    8. Attaches grounded explainability cues / heatmap if generated.
    9. Persists analysis in SQLite and commits transaction.
    
    ### Responsible-Use Guarantee:
    Verdicts are probabilistic assessments, never definitive proofs.
    """
    file_bytes = await image.read()
    filename = image.filename or "unknown.jpg"
    content_type = image.content_type or ""

    return analysis_service.process_analysis(
        file_bytes=file_bytes,
        filename=filename,
        declared_content_type=content_type,
        caption=caption,
        db=db
    )


@router.get(
    "/analyses",
    response_model=AnalysisListResponse,
    tags=["Analysis"],
    summary="List historical analyses with pagination",
    responses={
        200: {
            "model": AnalysisListResponse,
            "description": "Paginated historical list of analyses.",
        },
        400: {
            "model": ErrorResponse,
            "description": "Invalid page or page_size query parameters.",
        },
    },
)
async def list_analyses(
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page (max: 100)"),
    db: Session = Depends(get_db)
):
    """
    Retrieves a paginated list of past forensic analyses ordered chronologically descending.
    Efficiently executes SQL `offset` and `limit` without loading full database tables into memory.
    """
    return analysis_service.list_analyses(page=page, page_size=page_size, db=db)


@router.get(
    "/analyses/{analysis_id:path}/heatmap",
    tags=["Analysis"],
    summary="Download forensic Grad-CAM heatmap image",
    responses={
        200: {
            "content": {"image/jpeg": {}},
            "description": "Forensic Grad-CAM visual heatmap JPEG file stream.",
        },
        400: {
            "model": ErrorResponse,
            "description": "Analysis ID is not a valid UUID format (INVALID_ANALYSIS_ID).",
        },
        404: {
            "model": ErrorResponse,
            "description": "Analysis not found or visual heatmap was not generated (HEATMAP_NOT_FOUND).",
        },
    },
)
async def get_analysis_heatmap(
    analysis_id: str,
    db: Session = Depends(get_db)
):
    """
    Streams the visual Grad-CAM heatmap corresponding to the analysis ID.
    Returns 404 with standard error envelope if no heatmap exists.
    Enforces strict path-containment validation to prevent directory traversal.
    """
    heatmap_path = analysis_service.get_heatmap_file(analysis_id, db)
    return FileResponse(
        path=str(heatmap_path),
        media_type="image/jpeg",
        filename=f"heatmap_{analysis_id}.jpg"
    )


@router.get(
    "/analyses/{analysis_id:path}",
    response_model=AnalysisResponse,
    tags=["Analysis"],
    summary="Retrieve previously stored analysis by ID",
    responses={
        200: {
            "model": AnalysisResponse,
            "description": "Retrieved historical analysis record.",
        },
        400: {
            "model": ErrorResponse,
            "description": "Analysis ID is not a valid UUID format (INVALID_ANALYSIS_ID).",
        },
        404: {
            "model": ErrorResponse,
            "description": "Analysis record with specified UUID was not found (ANALYSIS_NOT_FOUND).",
        },
    },
)
async def get_analysis(
    analysis_id: str,
    db: Session = Depends(get_db)
):
    """
    Retrieves the complete forensic analysis record for a given UUID identifier.
    Returns 400 if UUID format is invalid, or 404 if record does not exist.
    """
    return analysis_service.get_analysis_by_id(analysis_id, db)

