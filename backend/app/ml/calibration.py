import math
from typing import Tuple
from app.core.config import settings
from app.core.logging import logger


class Calibrator:
    """
    Forensic Probability Calibration and Verdict Assignment.
    
    Principles:
    - Do NOT invent calibration parameters.
    - Calibration is only applied when empirical parameters (from held-out validation sets) are supplied.
    - If calibration is disabled or parameters are unconfigured, probabilities are passed as raw
      and explicitly tagged as uncalibrated.
    - Never silently clamp or mutate invalid values; validation precedes calibration.
    """

    @classmethod
    def apply_calibration(cls, raw_ai_probability: float) -> Tuple[float, bool, str]:
        """
        Calibrates raw AI probability if empirical calibration parameters are configured.
        
        Returns:
            Tuple of (calibrated_ai_probability, is_calibrated, calibration_method)
        """
        prob = float(raw_ai_probability)

        if not settings.CALIBRATION_ENABLED or settings.CALIBRATION_METHOD == "none":
            # Uncalibrated passthrough
            return prob, False, "uncalibrated"

        if settings.CALIBRATION_METHOD == "temperature":
            temp = settings.CALIBRATION_TEMPERATURE
            if temp <= 0:
                logger.warning(f"Invalid temperature {temp}; falling back to uncalibrated.")
                return prob, False, "uncalibrated"
            
            # Standard temperature scaling on log-odds
            epsilon = 1e-7
            p_bounded = max(epsilon, min(1.0 - epsilon, prob))
            logit = math.log(p_bounded / (1.0 - p_bounded))
            calibrated_logit = logit / temp
            calibrated_prob = 1.0 / (1.0 + math.exp(-calibrated_logit))
            return calibrated_prob, True, "temperature_scaling"

        # Unknown calibration method: keep raw
        return prob, False, "uncalibrated"

    @classmethod
    def apply_calibration_from_logit(cls, raw_logit: float) -> Tuple[float, bool, str]:
        """
        Calibrates directly from the model's raw pre-sigmoid logit, when available.

        This is the preferred calibration path: calibrated_probability =
        sigmoid(raw_logit / T). Operating on the logit avoids reconstructing
        it via an inverse-sigmoid round trip through an already-rounded
        probability (which `apply_calibration` must do when only a
        probability is available) - see .claude/specs/08a-probability-calibration.md
        ("Rules for implementation": "Calibration must operate on logits,
        never on already-sigmoided probabilities").

        Returns:
            Tuple of (calibrated_ai_probability, is_calibrated, calibration_method)
        """
        logit = float(raw_logit)

        if not settings.CALIBRATION_ENABLED or settings.CALIBRATION_METHOD == "none":
            uncalibrated_prob = 1.0 / (1.0 + math.exp(-logit))
            return uncalibrated_prob, False, "uncalibrated"

        if settings.CALIBRATION_METHOD == "temperature":
            temp = settings.CALIBRATION_TEMPERATURE
            if temp <= 0:
                logger.warning(f"Invalid temperature {temp}; falling back to uncalibrated.")
                uncalibrated_prob = 1.0 / (1.0 + math.exp(-logit))
                return uncalibrated_prob, False, "uncalibrated"

            calibrated_logit = logit / temp
            calibrated_prob = 1.0 / (1.0 + math.exp(-calibrated_logit))
            return calibrated_prob, True, "temperature_scaling"

        # Unknown calibration method: keep raw (sigmoid of the untouched logit)
        uncalibrated_prob = 1.0 / (1.0 + math.exp(-logit))
        return uncalibrated_prob, False, "uncalibrated"

    @classmethod
    def compute_verdict(
        cls,
        ai_probability: float,
        threshold: float
    ) -> Tuple[str, float]:
        """
        Determines the responsible likelihood verdict and corresponding confidence score.
        
        Responsible-use guidelines:
        - Never uses definitive wording ('100% fake', 'definitely AI').
        - Outputs likelihood assessment: 'likely_ai_generated' or 'likely_real'.
        - Confidence reflects the likelihood of the selected verdict.
        """
        if ai_probability >= threshold:
            verdict = "likely_ai_generated"
            confidence = ai_probability
        else:
            verdict = "likely_real"
            confidence = 1.0 - ai_probability

        confidence = round(confidence, 4)
        return verdict, confidence
