from pathlib import Path
from typing import Optional
from app.core.config import settings
from app.core.logging import logger
from app.ml.detector import ImageDetector, StubImageDetector


class ModelLoader:
    """
    Manages detector lifecycle and loads weights when available.
    """
    _instance: Optional[ImageDetector] = None

    @classmethod
    def get_detector(cls) -> ImageDetector:
        """Returns the active detector instance (singleton)."""
        if cls._instance is None:
            cls._instance = cls.load_detector()
        return cls._instance

    @classmethod
    def set_detector(cls, detector: ImageDetector) -> None:
        """Sets an active detector instance (useful for testing or switching models)."""
        cls._instance = detector

    @classmethod
    def load_detector(cls) -> ImageDetector:
        """
        Inspects configured weights path.
        If weights are absent, initializes StubImageDetector.
        When weights are present, this serves as the hook for the ML team's model loading.
        """
        weights_path = Path(settings.MODEL_WEIGHTS_PATH)
        if not weights_path.is_absolute():
            weights_path = settings.BASE_DIR / weights_path

        if not weights_path.exists() or weights_path.is_dir():
            logger.info(
                f"Model weights file not found at '{weights_path}'. "
                "Initializing StubImageDetector (model_loaded=False)."
            )
            return StubImageDetector(
                model_version=settings.MODEL_VERSION,
                weights_path=str(weights_path)
            )

        # Integration point for ML model loading once trained weights are placed in model/weights/
        logger.info(f"Discovered weights at '{weights_path}'. Awaiting ML team model class.")
        return StubImageDetector(
            model_version=settings.MODEL_VERSION,
            weights_path=str(weights_path)
        )

