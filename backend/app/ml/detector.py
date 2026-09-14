from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from app.core.errors import ModelUnavailableError
from app.core.logging import logger


class ImageDetector(ABC):
    """
    Abstract Base Class defining the inference contract for ML detection models.
    
    Loose-coupling guarantee:
    The backend services and FastAPI routes depend exclusively on this interface.
    The ML team can implement and swap architectures (ResNet, EfficientNet, ViT,
    ensemble, etc.) without altering any backend routes or contracts.
    """

    @abstractmethod
    def is_loaded(self) -> bool:
        """Returns True if model weights are loaded and ready for inference."""
        pass

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """Returns detector metadata (name, version, task, loaded status)."""
        pass

    @abstractmethod
    def predict(self, image_tensor: Any) -> Dict[str, Any]:
        """
        Runs forward pass inference on a preprocessed image tensor.
        
        Minimum Detector Output Contract:
        Must return a dictionary containing at least:
        {
            "raw_ai_probability": float,  # Finite float strictly satisfying 0.0 <= p <= 1.0
            "model_version": str          # Identifier string of the model checkpoint
        }
        
        Optional extra attributes:
        {
            "raw_real_probability": Optional[float],
            "logits": Optional[Any]
        }
        
        Must raise ModelUnavailableError if weights are not loaded.
        """
        pass

    @abstractmethod
    def explain(self, image_tensor: Any) -> Optional[Any]:
        """
        Generates grounded explainability data (e.g. Grad-CAM activation array)
        if supported by the active model architecture.
        
        Returns None or raises an error if explainability is not available.
        """
        pass


class StubImageDetector(ImageDetector):
    """
    Default detector implementation when trained model weights are not present.
    
    Principles:
    1. Does NOT return hardcoded predictions or fake probabilities.
    2. Explicitly reports is_loaded() == False.
    3. Raises ModelUnavailableError on predict() attempts so the API returns HTTP 503.
    """

    def __init__(self, model_version: str = "signalscope-v1", weights_path: Optional[str] = None):
        self.model_version = model_version
        self.weights_path = weights_path
        self._is_loaded = False
        logger.info(
            f"StubImageDetector initialized (version={self.model_version}, "
            f"weights_path={self.weights_path}). Awaiting real model weights."
        )

    def is_loaded(self) -> bool:
        return self._is_loaded

    def get_info(self) -> Dict[str, Any]:
        return {
            "name": "SignalScope Detector",
            "version": self.model_version,
            "task": "real-vs-ai-generated",
            "loaded": self._is_loaded,
            "device": "unassigned"
        }

    def predict(self, image_tensor: Any) -> Dict[str, Any]:
        logger.warning("predict() called on StubImageDetector: model weights are not loaded.")
        raise ModelUnavailableError(
            message="ML detector is unavailable or model weights are not loaded."
        )

    def explain(self, image_tensor: Any) -> Optional[Any]:
        return None
