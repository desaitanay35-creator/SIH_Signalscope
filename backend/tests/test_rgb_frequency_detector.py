"""
Unit tests for the real backend ML detector adapter
(`app.ml.rgb_frequency_detector.RGBFrequencyFusionDetector`).

Uses only the on-the-fly fixture checkpoint from
`tests/fixtures/tiny_fusion_checkpoint.py` - never the committed production
checkpoint - so this suite requires no network access and no checkpoint
artifact in the repository. See .claude/specs/09-backend-ml-integration.md
("Test plan").
"""

import math

import numpy as np
import pytest
import torch

from app.core.config import settings
from app.core.errors import ModelIncompatibleError
from app.ml.calibration import Calibrator
from app.ml.model_loader import ModelLoader
from app.ml.detector import StubImageDetector
from app.ml.rgb_frequency_detector import RGBFrequencyFusionDetector
from model.architectures.efficientnet_b4 import EfficientNetB4Baseline
from model.training.frequency_features import compute_log_magnitude_spectrum

from tests.fixtures.tiny_fusion_checkpoint import build_tiny_fusion_checkpoint

INPUT_SHAPE = (1, 3, 224, 224)


def _random_input_tensor(batch_size: int = 1, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn((batch_size, 3, 224, 224), generator=generator)


@pytest.fixture
def fixture_checkpoint(tmp_path):
    return build_tiny_fusion_checkpoint(tmp_path / "tiny_fusion.pt")


@pytest.fixture
def detector(fixture_checkpoint):
    return RGBFrequencyFusionDetector(
        weights_path=fixture_checkpoint,
        model_version="test-fusion-v1",
    )


class TestCheckpointLoading:
    def test_checkpoint_loading_succeeds_and_is_loaded(self, detector):
        assert detector.is_loaded() is True

    def test_missing_checkpoint_falls_back_to_stub(self, tmp_path, monkeypatch):
        missing_path = tmp_path / "does_not_exist.pt"
        monkeypatch.setattr(settings, "MODEL_WEIGHTS_PATH", str(missing_path))
        detector = ModelLoader.load_detector()
        assert isinstance(detector, StubImageDetector)
        assert detector.is_loaded() is False

    def test_corrupt_checkpoint_raises_model_incompatible_error(self, tmp_path):
        corrupt_path = tmp_path / "corrupt.pt"
        corrupt_path.write_bytes(b"not a valid torch checkpoint")

        with pytest.raises(ModelIncompatibleError):
            RGBFrequencyFusionDetector(weights_path=corrupt_path, model_version="test-v1")

    def test_incompatible_architecture_raises_model_incompatible_error(self, tmp_path):
        wrong_arch_path = tmp_path / "wrong_arch.pt"
        model = EfficientNetB4Baseline(pretrained=False)
        torch.save(
            {"model_state_dict": model.state_dict(), "model_config": model.get_spec()},
            wrong_arch_path,
        )

        with pytest.raises(ModelIncompatibleError):
            RGBFrequencyFusionDetector(weights_path=wrong_arch_path, model_version="test-v1")

    def test_missing_weights_never_raises_model_incompatible(self, tmp_path, monkeypatch):
        """Missing-weights must remain the StubImageDetector path, never
        ModelIncompatibleError - that error is reserved for a present but
        broken checkpoint."""
        missing_path = tmp_path / "nope.pt"
        monkeypatch.setattr(settings, "MODEL_WEIGHTS_PATH", str(missing_path))
        # Must not raise.
        detector = ModelLoader.load_detector()
        assert isinstance(detector, StubImageDetector)


class TestMetadata:
    def test_get_info_reports_loaded_metadata(self, detector):
        info = detector.get_info()
        assert info["loaded"] is True
        assert info["version"] == "test-fusion-v1"
        assert info["architecture"] == "rgb_frequency_fusion"
        assert info["device"] in ("cpu", "cuda")
        assert "checkpoint_path" in info

    def test_model_version_uses_configured_version_not_checkpoint_metadata(self, fixture_checkpoint):
        """Requirement: model_version must come from settings.MODEL_VERSION,
        never invented from arbitrary checkpoint metadata."""
        detector = RGBFrequencyFusionDetector(
            weights_path=fixture_checkpoint,
            model_version="whatever-version-string",
        )
        assert detector.get_info()["version"] == "whatever-version-string"
        output = detector.predict(_random_input_tensor())
        assert output["model_version"] == "whatever-version-string"


class TestPredict:
    def test_predict_probability_in_valid_range(self, detector):
        output = detector.predict(_random_input_tensor())
        prob = output["raw_ai_probability"]
        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0

    def test_predict_logits_field_matches_sigmoid_of_probability(self, detector):
        output = detector.predict(_random_input_tensor())
        logit = output["logits"]
        prob = output["raw_ai_probability"]
        assert math.isfinite(logit)
        expected_prob = 1.0 / (1.0 + math.exp(-logit))
        assert prob == pytest.approx(expected_prob, abs=1e-6)

    def test_predict_deterministic_in_eval_mode(self, detector):
        tensor = _random_input_tensor(seed=42)
        first = detector.predict(tensor.clone())
        second = detector.predict(tensor.clone())
        assert first["raw_ai_probability"] == pytest.approx(second["raw_ai_probability"], abs=1e-7)
        assert first["logits"] == pytest.approx(second["logits"], abs=1e-7)

    def test_predict_output_changes_with_input(self, detector):
        output_a = detector.predict(_random_input_tensor(seed=1))
        output_b = detector.predict(_random_input_tensor(seed=2))
        assert output_a["logits"] != pytest.approx(output_b["logits"], abs=1e-9)

    def test_predict_supports_batch_n_greater_than_one(self, detector):
        batch_tensor = _random_input_tensor(batch_size=3, seed=7)
        output = detector.predict(batch_tensor)
        assert isinstance(output["raw_ai_probability"], list)
        assert len(output["raw_ai_probability"]) == 3
        assert len(output["logits"]) == 3
        for prob in output["raw_ai_probability"]:
            assert 0.0 <= prob <= 1.0

    def test_cpu_inference_forced(self, fixture_checkpoint, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        cpu_detector = RGBFrequencyFusionDetector(
            weights_path=fixture_checkpoint,
            model_version="test-cpu-v1",
        )
        assert cpu_detector.get_info()["device"] == "cpu"
        output = cpu_detector.predict(_random_input_tensor())
        assert 0.0 <= output["raw_ai_probability"] <= 1.0


class TestFrequencyBranchConstruction:
    def test_build_frequency_input_shape_for_batch(self, detector):
        batch_tensor = _random_input_tensor(batch_size=3, seed=5)
        frequency_tensor = detector._build_frequency_input(batch_tensor)
        assert frequency_tensor.shape == (3, 1, 224, 224)
        assert torch.all(frequency_tensor >= 0.0)
        assert torch.all(frequency_tensor <= 1.0)

    def test_build_frequency_input_matches_manual_reconstruction(self, detector):
        tensor = _random_input_tensor(batch_size=1, seed=9)
        frequency_tensor = detector._build_frequency_input(tensor)

        mean = np.asarray(settings.MODEL_NORM_MEAN, dtype=np.float32).reshape(3, 1, 1)
        std = np.asarray(settings.MODEL_NORM_STD, dtype=np.float32).reshape(3, 1, 1)
        denormalized = (tensor[0].numpy() * std + mean).clip(0.0, 1.0)
        rgb_array = denormalized.transpose(1, 2, 0)
        expected = compute_log_magnitude_spectrum(rgb_array)

        assert torch.allclose(frequency_tensor[0], expected, atol=1e-5)


class TestCalibrationFromLogit:
    TEMPERATURE = 1.1838611364364624

    def test_apply_calibration_from_logit_matches_sigmoid_over_temperature(self, monkeypatch):
        monkeypatch.setattr(settings, "CALIBRATION_ENABLED", True)
        monkeypatch.setattr(settings, "CALIBRATION_METHOD", "temperature")
        monkeypatch.setattr(settings, "CALIBRATION_TEMPERATURE", self.TEMPERATURE)

        raw_logit = 2.3456
        calibrated_prob, is_calibrated, method = Calibrator.apply_calibration_from_logit(raw_logit)

        expected = 1.0 / (1.0 + math.exp(-(raw_logit / self.TEMPERATURE)))
        assert calibrated_prob == pytest.approx(expected, abs=1e-12)
        assert is_calibrated is True
        assert method == "temperature_scaling"

    def test_apply_calibration_from_logit_disabled_returns_raw_sigmoid(self, monkeypatch):
        monkeypatch.setattr(settings, "CALIBRATION_ENABLED", False)
        raw_logit = -1.5
        calibrated_prob, is_calibrated, method = Calibrator.apply_calibration_from_logit(raw_logit)
        expected = 1.0 / (1.0 + math.exp(-raw_logit))
        assert calibrated_prob == pytest.approx(expected, abs=1e-12)
        assert is_calibrated is False
        assert method == "uncalibrated"

    def test_calibration_does_not_reconstruct_logit_via_probability_roundtrip(self, monkeypatch):
        """Regression guard: apply_calibration_from_logit must use the true
        raw logit directly, not an inverse-sigmoid reconstruction from an
        already-rounded probability (which the legacy apply_calibration
        path does, and which this new path exists specifically to avoid)."""
        monkeypatch.setattr(settings, "CALIBRATION_ENABLED", True)
        monkeypatch.setattr(settings, "CALIBRATION_METHOD", "temperature")
        monkeypatch.setattr(settings, "CALIBRATION_TEMPERATURE", self.TEMPERATURE)

        raw_logit = 5.0
        rounded_prob = round(1.0 / (1.0 + math.exp(-raw_logit)), 4)  # simulate precision loss

        from_logit, _, _ = Calibrator.apply_calibration_from_logit(raw_logit)
        from_rounded_prob, _, _ = Calibrator.apply_calibration(rounded_prob)

        # Both should be close (same underlying value), but the logit path
        # must equal the exact-logit computation - not the reconstructed one.
        exact_expected = 1.0 / (1.0 + math.exp(-(raw_logit / self.TEMPERATURE)))
        assert from_logit == pytest.approx(exact_expected, abs=1e-12)


class TestThresholdDecisionConsistency:
    TEMPERATURE = 1.1838611364364624

    def test_threshold_decision_agrees_between_logit_and_probability_paths(self, monkeypatch):
        monkeypatch.setattr(settings, "CALIBRATION_ENABLED", True)
        monkeypatch.setattr(settings, "CALIBRATION_METHOD", "temperature")
        monkeypatch.setattr(settings, "CALIBRATION_TEMPERATURE", self.TEMPERATURE)

        raw_logit = 1.75
        raw_prob = 1.0 / (1.0 + math.exp(-raw_logit))  # exact, no rounding loss

        prob_via_logit, _, _ = Calibrator.apply_calibration_from_logit(raw_logit)
        prob_via_probability, _, _ = Calibrator.apply_calibration(raw_prob)

        assert prob_via_logit == pytest.approx(prob_via_probability, abs=1e-9)

        threshold = 0.5
        verdict_a, confidence_a = Calibrator.compute_verdict(prob_via_logit, threshold)
        verdict_b, confidence_b = Calibrator.compute_verdict(prob_via_probability, threshold)

        assert verdict_a == verdict_b
        assert confidence_a == pytest.approx(confidence_b, abs=1e-9)
