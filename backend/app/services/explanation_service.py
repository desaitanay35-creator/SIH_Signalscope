from typing import Any, Optional, Tuple
from app.core.logging import logger
from app.ml.detector import ImageDetector
from app.schemas.analysis import ExplanationResponse
from app.services.storage_service import StorageService


class ExplanationService:
    """
    Forensic Explainability & Visual Evidence Service.
    
    Principles:
    - Never invent ungrounded explanations or arbitrary LLM text.
    - Saliency maps (Grad-CAM) must originate from actual model layer gradients/activations.
    - If Grad-CAM or explainability modules are not loaded, return a clearly
      indicated unavailable state.
    """

    @classmethod
    def generate_explanation(
        cls,
        detector: ImageDetector,
        tensor: Any,
        analysis_id: str,
        storage_service: StorageService
    ) -> Tuple[ExplanationResponse, Optional[str]]:
        """
        Attempts to generate grounded visual evidence from the detector.
        
        Returns:
            Tuple of (ExplanationResponse, heatmap_storage_path)
        """
        try:
            explanation_data = detector.explain(tensor)
        except Exception as e:
            logger.warning(f"Detector explain() raised error: {str(e)}")
            explanation_data = None

        if explanation_data is None:
            # Clearly documented unavailable state
            return ExplanationResponse(
                summary="Grounded visual explanation is unavailable because explainability module is not loaded.",
                cues=[],
                heatmap_url=None,
                heatmap_available=False
            ), None

        # When explainability data is returned (e.g. heatmap bytes or numpy matrix)
        # Note: If detector provides heatmap image bytes:
        if isinstance(explanation_data, bytes):
            rel_path = storage_service.save_heatmap(analysis_id, explanation_data)
            heatmap_url = f"/api/v1/analyses/{analysis_id}/heatmap"
            return ExplanationResponse(
                summary=(
                    "Highlighted regions show where the model's RGB (visual) analysis "
                    "concentrated when evaluating this image's likelihood of being "
                    "likely AI-generated. The frequency-domain branch also contributes "
                    "to the prediction but is not directly visualized by this heatmap. "
                    "This shows where the model focused, not proof of any specific "
                    "forensic artifact."
                ),
                cues=[],
                heatmap_url=heatmap_url,
                heatmap_available=True
            ), rel_path

        return ExplanationResponse(
            summary="Explainability module did not yield visual activation maps.",
            cues=[],
            heatmap_url=None,
            heatmap_available=False
        ), None

