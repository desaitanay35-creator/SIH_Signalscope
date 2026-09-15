"""
Integration test for .claude/specs/09-backend-ml-integration.md: a decoded
PIL image -> backend preprocessing -> the real RGBFrequencyFusionDetector
-> a calibrated verdict, plus a preprocessing-parity check against
model/training's own dual-branch preprocessing path.

Uses only the on-the-fly fixture checkpoint from
`tests/fixtures/tiny_fusion_checkpoint.py` - no production checkpoint is
loaded or required.
"""

import numpy as np
import pytest
import torch
from PIL import Image

from app.core.config import settings
from app.ml.model_loader import ModelLoader
from app.ml.rgb_frequency_detector import RGBFrequencyFusionDetector
from app.services.inference_service import InferenceService
from model.training.frequency_features import compute_log_magnitude_spectrum

from tests.fixtures.tiny_fusion_checkpoint import build_tiny_fusion_checkpoint


def _synthetic_rgb_image(size=(224, 224), seed: int = 3) -> Image.Image:
    """A deterministic synthetic RGB image, already at the backend's target
    size. Kept at exactly 224x224 (== settings.MODEL_INPUT_WIDTH/HEIGHT) so
    the backend's LANCZOS resize and model/training's BILINEAR resize are
    both a same-size no-op - isolating the parity check to the
    normalize/frequency-reconstruction logic this step actually adds,
    rather than confounding it with an unrelated resize-algorithm
    difference between the two pipelines."""
    rng = np.random.default_rng(seed)
    array = rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
    return Image.fromarray(array, mode="RGB")


@pytest.fixture
def fixture_checkpoint(tmp_path):
    return build_tiny_fusion_checkpoint(tmp_path / "tiny_fusion.pt")


@pytest.fixture
def real_detector(fixture_checkpoint):
    return RGBFrequencyFusionDetector(
        weights_path=fixture_checkpoint,
        model_version="test-fusion-v1",
    )


class TestEndToEndInference:
    def test_uploaded_image_through_backend_pipeline_yields_calibrated_verdict(
        self, real_detector, monkeypatch
    ):
        monkeypatch.setattr(settings, "CALIBRATION_ENABLED", True)
        monkeypatch.setattr(settings, "CALIBRATION_METHOD", "temperature")
        monkeypatch.setattr(settings, "CALIBRATION_TEMPERATURE", 1.1838611364364624)

        image = _synthetic_rgb_image()
        result = InferenceService.run_inference(image, real_detector)

        assert result["is_calibrated"] is True
        assert result["calibration_method"] == "temperature_scaling"
        assert 0.0 <= result["ai_probability"] <= 1.0
        assert 0.0 <= result["raw_ai_probability"] <= 1.0
        assert result["model_version"] == "test-fusion-v1"

    def test_model_loader_wires_real_detector_when_weights_present(self, fixture_checkpoint, monkeypatch):
        monkeypatch.setattr(settings, "MODEL_WEIGHTS_PATH", str(fixture_checkpoint))
        loaded = ModelLoader.load_detector()
        assert isinstance(loaded, RGBFrequencyFusionDetector)
        assert loaded.is_loaded() is True


class TestPreprocessingParity:
    def test_backend_preprocessing_matches_training_preprocessing_path(self, real_detector):
        """The raw logit produced via the backend's preprocessing path must
        match the raw logit produced via model/training's own dual-branch
        preprocessing (same underlying model, same image) - this is the
        only direct check that the detector's denormalize -> FFT
        reconstruction (see app/ml/rgb_frequency_detector.py) is actually
        equivalent to the training-time frequency input."""
        image = _synthetic_rgb_image()

        # Path 1: backend preprocessing -> detector.predict()
        backend_tensor = InferenceService.preprocess_image(image)
        backend_output = real_detector.predict(backend_tensor)
        backend_logit = backend_output["logits"]

        # Path 2: model/training-equivalent preprocessing, run through the
        # SAME underlying model instance.
        rgb_array_0_1 = np.asarray(image, dtype=np.float32) / 255.0  # (H, W, 3)
        mean = np.asarray(settings.MODEL_NORM_MEAN, dtype=np.float32).reshape(3, 1, 1)
        std = np.asarray(settings.MODEL_NORM_STD, dtype=np.float32).reshape(3, 1, 1)
        rgb_chw = rgb_array_0_1.transpose(2, 0, 1)
        normalized = (rgb_chw - mean) / std
        rgb_tensor = torch.from_numpy(normalized.astype(np.float32)).unsqueeze(0)
        frequency_tensor = compute_log_magnitude_spectrum(rgb_array_0_1).unsqueeze(0)

        with torch.no_grad():
            training_logit = real_detector._model(
                {"rgb": rgb_tensor, "frequency": frequency_tensor}
            ).item()

        assert backend_logit == pytest.approx(training_logit, abs=1e-4)
