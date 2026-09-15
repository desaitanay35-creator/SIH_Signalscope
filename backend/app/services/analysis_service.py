import math
import os
import time
import uuid
from typing import Any, Optional
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from app.core.config import settings
from app.core.errors import (
    AnalysisFailedError,
    AnalysisNotFoundError,
    AppException,
    DatabaseError,
    InferenceError,
    InvalidAnalysisIdError,
    InvalidModelOutputError,
    ModelUnavailableError,
)
from app.core.logging import logger
from app.db.models import AnalysisRecord
from app.ml.calibration import Calibrator
from app.ml.model_loader import ModelLoader
from app.schemas.analysis import (
    AnalysisResponse,
    AnalysisListResponse,
    AnalysisListItemResponse,
    ExplanationResponse,
    ImageMetadataResponse,
)
from app.services.image_service import ImageService
from app.services.inference_service import InferenceService
from app.services.metadata_service import MetadataService
from app.services.explanation_service import ExplanationService
from app.services.storage_service import StorageService


class AnalysisService:
    """
    Main Analysis Service orchestrating the complete media-forensics pipeline.
    
    Hardened Logical Pipeline Order:
    1. Receive uploaded bytes
    2. Validate and decode image (ImageService)
    3. Extract basic metadata (MetadataService)
    4. Obtain detector (ModelLoader)
    5. Verify detector is actually loaded
    6. If unavailable:
         raise MODEL_UNAVAILABLE (HTTP 503)
         DO NOT save uploaded image
         DO NOT create DB record
    7. Save original using safe UUID path (StorageService)
    8. Preprocess through InferenceService
    9. Call detector.predict()
    10. Validate detector output (finite float in [0.0, 1.0])
    11. Apply calibration only if genuinely configured
    12. Determine responsible verdict
    13. Request explanation through ExplanationService
    14. Persist successful analysis (commit DB transaction)
    15. Return API schema
    
    Transactional Safety:
    - If model availability check fails: zero files or DB records created.
    - If a failure occurs after saving original file: newly stored file is deleted
      and DB transaction rolled back.
    """

    def __init__(self, storage_service: Optional[StorageService] = None):
        self.storage_service = storage_service or StorageService()

    @staticmethod
    def _validate_detector_output(raw_output: Any) -> float:
        """
        Validates detector prediction payload against the minimum contract.
        
        Minimum Contract:
        {
            "raw_ai_probability": float,
            "model_version": str
        }
        
        Validation rules for raw_ai_probability:
        - Must exist and not be null/None.
        - Must be numeric (int or float, not boolean).
        - Must be finite (reject NaN, +Infinity, -Infinity).
        - Must satisfy 0.0 <= probability <= 1.0 (reject -0.1, 1.1, etc.).
        - Never silently clamped.
        
        Raises:
            InvalidModelOutputError (HTTP 500) if validation fails.
        """
        if not isinstance(raw_output, dict):
            logger.error("Detector output is not a dictionary.")
            raise InvalidModelOutputError(
                message="The detector returned an invalid output format."
            )

        if "raw_ai_probability" not in raw_output or raw_output["raw_ai_probability"] is None:
            logger.error("Detector output missing 'raw_ai_probability'.")
            raise InvalidModelOutputError(
                message="The detector output is missing the required 'raw_ai_probability' field."
            )

        val = raw_output["raw_ai_probability"]

        # Reject booleans (in Python, isinstance(True, int) is True) and non-numerics
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            logger.error(f"Detector output 'raw_ai_probability' is non-numeric: {type(val)}")
            raise InvalidModelOutputError(
                message="The detector returned a non-numeric probability value."
            )

        # Reject NaN, Infinity, -Infinity
        if math.isnan(val) or math.isinf(val):
            logger.error(f"Detector output 'raw_ai_probability' is not finite: {val}")
            raise InvalidModelOutputError(
                message="The detector returned an infinite or non-finite probability."
            )

        # Reject out-of-range probabilities
        if val < 0.0 or val > 1.0:
            logger.error(f"Detector output 'raw_ai_probability' is out of bounds: {val}")
            raise InvalidModelOutputError(
                message=f"The detector returned a probability outside the allowed range [0.0, 1.0]: {val}."
            )

        return float(val)

    @staticmethod
    def _validate_uuid(analysis_id: str) -> str:
        """Validates that analysis_id conforms to a standard UUID string format."""
        try:
            clean_id = str(uuid.UUID(str(analysis_id).strip()))
            return clean_id
        except (ValueError, AttributeError, TypeError):
            logger.warning(f"Invalid UUID format requested: '{analysis_id}'")
            raise InvalidAnalysisIdError(
                message="The provided analysis ID is not a valid UUID format."
            )

    def process_analysis(
        self,
        file_bytes: bytes,
        filename: str,
        declared_content_type: str,
        caption: Optional[str],
        db: Session
    ) -> AnalysisResponse:
        start_time = time.perf_counter()
        analysis_id = str(uuid.uuid4())
        logger.info(f"Starting analysis {analysis_id} for file '{filename}'")
        
        # Sanitize client-provided filename: strip directory traversal paths, control chars, and truncate to 255 chars
        safe_filename = os.path.basename(str(filename or "").replace("\\", "/")).strip()
        if not safe_filename:
            safe_filename = "unnamed_image"
        if len(safe_filename) > 255:
            safe_filename = safe_filename[:255]

        logger.info(f"Starting analysis {analysis_id} for sanitized file '{safe_filename}'")

        # 1. Receive uploaded bytes & 2. Validate and decode image
        image, basic_meta = ImageService.validate_and_decode(
            file_bytes=file_bytes,
            declared_content_type=declared_content_type
        )

        # 3. Extract basic metadata
        metadata_res = MetadataService.extract_metadata(
            image=image,
            raw_bytes=file_bytes,
            basic_meta=basic_meta
        )

        # 4. Obtain detector & 5. Verify detector is actually loaded
        detector = ModelLoader.get_detector()
        if not detector.is_loaded():
            logger.warning(f"Analysis {analysis_id} rejected: ML detector is not loaded.")
            # 6. If unavailable: raise MODEL_UNAVAILABLE (503), NO file saved, NO DB record created
            raise ModelUnavailableError(
                message="ML detector is unavailable or model weights are not loaded."
            )

        stored_path: Optional[str] = None
        heatmap_rel_path: Optional[str] = None

        try:
            # 7. Save original using safe UUID path
            _, stored_path = self.storage_service.save_original(
                file_bytes=file_bytes,
                extension=basic_meta.get("extension", "jpg")
            )

            # 8. Preprocess through InferenceService
            try:
                tensor = InferenceService.preprocess_image(image)
            except Exception as e:
                logger.error(f"Preprocessing failed for analysis {analysis_id}: {str(e)}")
                raise InferenceError(message="Image preprocessing failed prior to model inference.")

            # 9. Call detector.predict()
            try:
                raw_output = detector.predict(tensor)
            except AppException:
                raise
            except Exception as e:
                logger.error(f"Detector predict() threw unexpected error for analysis {analysis_id}: {str(e)}")
                raise InferenceError(message="An unexpected error occurred during model inference.")

            # 10. Validate detector output
            raw_ai_prob = self._validate_detector_output(raw_output)
            model_version = str(raw_output.get("model_version", settings.MODEL_VERSION))

            # 11. Apply calibration only if genuinely configured - prefer the
            # raw-logit path (sigmoid(raw_logit / T)) whenever the detector
            # supplies a finite logit, falling back to the probability-based
            # path for detectors that only report a probability (e.g.
            # StubImageDetector or a test mock detector).
            raw_logit = raw_output.get("logits") if isinstance(raw_output, dict) else None
            if (
                isinstance(raw_logit, (int, float))
                and not isinstance(raw_logit, bool)
                and math.isfinite(raw_logit)
            ):
                calibrated_prob, is_calibrated, cal_method = Calibrator.apply_calibration_from_logit(raw_logit)
            else:
                calibrated_prob, is_calibrated, cal_method = Calibrator.apply_calibration(raw_ai_prob)

            # 12. Determine responsible verdict
            threshold = settings.DECISION_THRESHOLD
            verdict, confidence = Calibrator.compute_verdict(calibrated_prob, threshold)

            # 13. Request explanation through ExplanationService
            try:
                explanation_res, heatmap_rel_path = ExplanationService.generate_explanation(
                    detector=detector,
                    tensor=tensor,
                    analysis_id=analysis_id,
                    storage_service=self.storage_service
                )
            except Exception as e:
                logger.warning(f"Explanation generation encountered non-fatal error: {str(e)}")
                explanation_res = ExplanationResponse(
                    summary="Grounded visual explanation was unavailable.",
                    cues=[],
                    heatmap_url=None,
                    heatmap_available=False
                )
                heatmap_rel_path = None

            elapsed_ms = int((time.perf_counter() - start_time) * 1000)

            # 14. Persist successful analysis
            record = AnalysisRecord(
                id=analysis_id,
                filename=safe_filename,
                stored_path=stored_path,
                verdict=verdict,
                confidence=confidence,
                threshold=threshold,
                model_version=model_version,
                processing_time_ms=elapsed_ms,
                caption=caption,
                is_calibrated=1 if is_calibrated else 0,
                calibration_method=cal_method,
                explanation_summary=explanation_res.summary,
                explanation_cues=[cue.model_dump() for cue in explanation_res.cues],
                heatmap_path=heatmap_rel_path,
                image_metadata=metadata_res.model_dump(),
            )
            db.add(record)
            db.commit()
            db.refresh(record)

            logger.info(f"Analysis {analysis_id} successfully persisted in {elapsed_ms}ms.")

            # 15. Return API schema
            return AnalysisResponse(
                analysis_id=record.id,
                verdict=record.verdict,
                confidence=record.confidence,
                threshold=record.threshold,
                model_version=record.model_version,
                processing_time_ms=record.processing_time_ms,
                is_calibrated=bool(record.is_calibrated),
                calibration_method=record.calibration_method,
                explanation=explanation_res,
                metadata=metadata_res,
                created_at=record.created_at
            )

        except Exception as e:
            # Transaction Safety: Rollback storage of newly created file
            if stored_path:
                logger.info(f"Transaction cleanup: removing stored original '{stored_path}'")
                self.storage_service.delete_original(stored_path)
            if heatmap_rel_path:
                logger.info(f"Transaction cleanup: removing stored heatmap for analysis '{analysis_id}'")
                self.storage_service.delete_heatmap(analysis_id)

            # Transaction Safety: Rollback database session
            db.rollback()
            try:
                db.rollback()
            except Exception as rollback_err:
                logger.error(f"Error during db.rollback(): {str(rollback_err)}")

            if isinstance(e, AppException):
                raise e

            if isinstance(e, SQLAlchemyError):
                logger.exception(f"Database error during analysis transaction: {str(e)}")
                raise DatabaseError(message="A database error occurred while saving the analysis record.")

            logger.exception(f"Unhandled error in analysis transaction: {str(e)}")
            raise AnalysisFailedError(message="Analysis pipeline failed to process the request.")

    def get_analysis_by_id(self, analysis_id: str, db: Session) -> AnalysisResponse:
        """Retrieves a previously stored analysis by its ID."""
        record = db.query(AnalysisRecord).filter(AnalysisRecord.id == analysis_id).first()
        clean_id = self._validate_uuid(analysis_id)
        try:
            record = db.query(AnalysisRecord).filter(AnalysisRecord.id == clean_id).first()
        except SQLAlchemyError as e:
            logger.exception(f"Database query error in get_analysis_by_id: {str(e)}")
            raise DatabaseError(message="A database error occurred while retrieving the analysis record.")

        if not record:
            logger.warning(f"Analysis '{analysis_id}' not found.")
            raise AnalysisNotFoundError(message=f"Analysis with ID '{analysis_id}' not found.")
            logger.warning(f"Analysis '{clean_id}' not found.")
            raise AnalysisNotFoundError(message=f"Analysis with ID '{clean_id}' not found.")

        meta_dict = record.image_metadata or {}
        metadata_res = ImageMetadataResponse(**meta_dict)

        heatmap_url = f"/api/v1/analyses/{record.id}/heatmap" if record.heatmap_path else None
        cues = record.explanation_cues or []
        explanation_res = ExplanationResponse(
            summary=record.explanation_summary,
            cues=cues,
            heatmap_url=heatmap_url,
            heatmap_available=bool(record.heatmap_path)
        )

        return AnalysisResponse(
            analysis_id=record.id,
            verdict=record.verdict,
            confidence=record.confidence,
            threshold=record.threshold,
            model_version=record.model_version,
            processing_time_ms=record.processing_time_ms,
            is_calibrated=bool(record.is_calibrated),
            calibration_method=record.calibration_method,
            explanation=explanation_res,
            metadata=metadata_res,
            created_at=record.created_at
        )

    def list_analyses(self, page: int, page_size: int, db: Session) -> AnalysisListResponse:
        """Returns paginated analysis history."""
        page = max(1, page)
        page_size = max(1, min(100, page_size))
        offset = (page - 1) * page_size

        total = db.query(AnalysisRecord).count()
        records = (
            db.query(AnalysisRecord)
            .order_by(AnalysisRecord.created_at.desc())
            .offset(offset)
            .limit(page_size)
            .all()
        )
        try:
            total = db.query(AnalysisRecord).count()
            records = (
                db.query(AnalysisRecord)
                .order_by(AnalysisRecord.created_at.desc())
                .offset(offset)
                .limit(page_size)
                .all()
            )
        except SQLAlchemyError as e:
            logger.exception(f"Database query error in list_analyses: {str(e)}")
            raise DatabaseError(message="A database error occurred while listing analyses.")

        items = [
            AnalysisListItemResponse(
                analysis_id=r.id,
                filename=r.filename,
                verdict=r.verdict,
                confidence=r.confidence,
                model_version=r.model_version,
                processing_time_ms=r.processing_time_ms,
                created_at=r.created_at
            )
            for r in records
        ]

        return AnalysisListResponse(
            items=items,
            total=total,
            page=page,
            page_size=page_size
        )

    def get_heatmap_file(self, analysis_id: str, db: Session):
        """Resolves heatmap image path or raises 404."""
        record = db.query(AnalysisRecord).filter(AnalysisRecord.id == analysis_id).first()
        clean_id = self._validate_uuid(analysis_id)
        try:
            record = db.query(AnalysisRecord).filter(AnalysisRecord.id == clean_id).first()
        except SQLAlchemyError as e:
            logger.exception(f"Database query error in get_heatmap_file: {str(e)}")
            raise DatabaseError(message="A database error occurred while retrieving the heatmap record.")

        if not record:
            raise AnalysisNotFoundError(message=f"Analysis with ID '{analysis_id}' not found.")
            raise AnalysisNotFoundError(message=f"Analysis with ID '{clean_id}' not found.")

        if not record.heatmap_path:
            raise AppException(
                code="HEATMAP_NOT_FOUND",
                message="No visual heatmap was generated for this analysis.",
                status_code=404
            )

        heatmap_path = self.storage_service.get_heatmap_path(analysis_id)
        heatmap_path = self.storage_service.get_heatmap_path(clean_id)
        if not heatmap_path or not heatmap_path.exists():
            raise AppException(
                code="HEATMAP_NOT_FOUND",
                message="Heatmap file is not available on storage disk.",
                status_code=404
            )

        return heatmap_path
