import math
from typing import Any, Dict
import numpy as np
from PIL import Image
from app.core.config import settings
from app.core.logging import logger
from app.ml.calibration import Calibrator
from app.ml.detector import ImageDetector


class InferenceService:
    """
    Image Preprocessing and Model Inference Pipeline.
    
    Principles:
    - Preprocessing parameters (dimensions, mean, standard deviation) are fully
      configurable via application settings to match the ML team's specification.
    - No assumptions are made regarding fixed architecture dimensions.
    """

    @classmethod
    def preprocess_image(cls, image: Image.Image) -> np.ndarray:
        """
        Preprocesses a decoded RGB PIL image into a normalized NCHW floating-point tensor array.
        
        Configurable parameters:
        - target_size: (MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT)
        - normalization: (pixel / 255.0 - mean) / std
        """
        target_size = (settings.MODEL_INPUT_WIDTH, settings.MODEL_INPUT_HEIGHT)
        logger.info(f"Preprocessing image: resizing to {target_size}")

        # Resize with high-quality Lanczos resampling
        resized = image.resize(target_size, resample=Image.Resampling.LANCZOS)
        img_array = np.array(resized, dtype=np.float32) / 255.0  # shape: (H, W, C)

        # Apply configurable channel-wise normalization
        mean = np.array(settings.MODEL_NORM_MEAN, dtype=np.float32)
        std = np.array(settings.MODEL_NORM_STD, dtype=np.float32)
        normalized = (img_array - mean) / std

        # Transpose from (H, W, C) to (C, H, W) and add batch dimension -> (1, C, H, W)
        tensor = np.transpose(normalized, (2, 0, 1))
        batch_tensor = np.expand_dims(tensor, axis=0)

        return batch_tensor

    @classmethod
    def run_inference(
        cls,
        image: Image.Image,
        detector: ImageDetector
    ) -> Dict[str, Any]:
        """
        Executes end-to-end preprocessing, inference, and calibrated verdict calculation.
        
        Raises:
            ModelUnavailableError: if the detector is not loaded.
        """
        # 1. Preprocess
        tensor = cls.preprocess_image(image)

        # 2. Forward pass through detector interface
        logger.info(f"Invoking detector '{detector.get_info().get('name')}'...")
        raw_output = detector.predict(tensor)
        
        raw_ai_prob = float(raw_output.get("raw_ai_probability", 0.0))
        model_version = raw_output.get("model_version", settings.MODEL_VERSION)

        # 3. Calibration - prefer the raw-logit path (operates on raw_logit,
        # never on an already-sigmoided probability) whenever the detector
        # supplies a finite logit; fall back to the probability-based path
        # for detectors that only report a probability (e.g. StubImageDetector
        # or a mock detector), preserving backward compatibility.
        raw_logit = raw_output.get("logits")
        if (
            isinstance(raw_logit, (int, float))
            and not isinstance(raw_logit, bool)
            and math.isfinite(raw_logit)
        ):
            calibrated_prob, is_calibrated, cal_method = Calibrator.apply_calibration_from_logit(raw_logit)
        else:
            calibrated_prob, is_calibrated, cal_method = Calibrator.apply_calibration(raw_ai_prob)

        # 4. Verdict determination
        threshold = settings.DECISION_THRESHOLD
        verdict, confidence = Calibrator.compute_verdict(calibrated_prob, threshold)

        logger.info(
            f"Inference complete: verdict='{verdict}', confidence={confidence:.4f}, "
            f"calibrated={is_calibrated}, threshold={threshold}"
        )

        return {
            "verdict": verdict,
            "confidence": confidence,
            "ai_probability": calibrated_prob,
            "raw_ai_probability": raw_ai_prob,
            "threshold": threshold,
            "model_version": model_version,
            "is_calibrated": is_calibrated,
            "calibration_method": cal_method if is_calibrated else None,
            "tensor": tensor,
        }

