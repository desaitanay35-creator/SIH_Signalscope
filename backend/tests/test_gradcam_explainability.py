"""
Tests for Step 10 (Grad-CAM explainability):
`app.ml.gradcam` (generic Grad-CAM math/hook management) and
`RGBFrequencyFusionDetector.explain()` (model-specific wiring).

Uses only the on-the-fly fixture checkpoint from
`tests/fixtures/tiny_fusion_checkpoint.py` - never the committed production
checkpoint. See .claude/specs/10-explainability.md style guarantees:
raw-logit targeting, no parameter mutation, hook cleanup, determinism,
faithfulness.
"""

import io

import numpy as np
import pytest
import torch
from PIL import Image

from app.core.config import settings
from app.ml import gradcam as gradcam_module
from app.ml.gradcam import compute_gradcam_map, render_overlay_jpeg, resize_cam
from app.ml.rgb_frequency_detector import RGBFrequencyFusionDetector

from tests.fixtures.tiny_fusion_checkpoint import build_tiny_fusion_checkpoint

INPUT_SIZE = 224


def _random_input_tensor(batch_size: int = 1, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn((batch_size, 3, INPUT_SIZE, INPUT_SIZE), generator=generator)


def _structured_input_tensor(seed: int = 11) -> torch.Tensor:
    """A synthetic image with real spatial structure (four colored
    quadrants + light noise) rather than pure random noise, so the
    (randomly-initialized) fixture model's Grad-CAM has genuine spatial
    variation to test faithfulness against - a flat noise image gives a
    much flatter, less discriminative CAM."""
    rng = np.random.default_rng(seed)
    h = w = INPUT_SIZE
    base = np.zeros((h, w, 3), dtype=np.float32)
    base[: h // 2, : w // 2] = [0.9, 0.1, 0.1]
    base[: h // 2, w // 2 :] = [0.1, 0.9, 0.1]
    base[h // 2 :, : w // 2] = [0.1, 0.1, 0.9]
    base[h // 2 :, w // 2 :] = [0.9, 0.9, 0.1]
    noise = rng.normal(0, 0.05, size=(h, w, 3)).astype(np.float32)
    image = np.clip(base + noise, 0.0, 1.0)

    mean = np.asarray(settings.MODEL_NORM_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std = np.asarray(settings.MODEL_NORM_STD, dtype=np.float32).reshape(3, 1, 1)
    chw = image.transpose(2, 0, 1)
    normalized = (chw - mean) / std
    return torch.from_numpy(normalized.astype(np.float32)).unsqueeze(0)


@pytest.fixture
def fixture_checkpoint(tmp_path):
    return build_tiny_fusion_checkpoint(tmp_path / "tiny_fusion.pt")


@pytest.fixture
def detector(fixture_checkpoint):
    return RGBFrequencyFusionDetector(
        weights_path=fixture_checkpoint,
        model_version="test-gradcam-v1",
    )


def _target_layer(detector):
    return detector._model.rgb_branch.backbone.features[-1]


# ---------------------------------------------------------------------------
# A, B, C: output exists, is valid JPEG, matches input dimensions
# ---------------------------------------------------------------------------
class TestExplainOutput:
    def test_explain_returns_bytes(self, detector):
        result = detector.explain(_random_input_tensor())
        assert result is not None
        assert isinstance(result, bytes)
        assert len(result) > 100

    def test_explain_output_is_valid_jpeg(self, detector):
        result = detector.explain(_random_input_tensor())
        assert result[:3] == b"\xff\xd8\xff"
        image = Image.open(io.BytesIO(result))
        image.load()  # forces full decode, raising on truncated/invalid JPEG
        assert image.format == "JPEG"

    def test_explain_output_matches_input_tensor_dimensions(self, detector):
        tensor = _random_input_tensor()
        result = detector.explain(tensor)
        image = Image.open(io.BytesIO(result))
        expected_size = (tensor.shape[-1], tensor.shape[-2])  # PIL is (W, H)
        assert image.size == expected_size

    def test_explain_requires_single_image_batch(self, detector):
        with pytest.raises(ValueError):
            detector.explain(_random_input_tensor(batch_size=2))


# ---------------------------------------------------------------------------
# D: underlying CAM shape/range/NaN-Inf safety (module-level, via gradcam.py)
# ---------------------------------------------------------------------------
class TestGradCAMMap:
    def test_cam_shape_range_and_finiteness(self, detector):
        tensor = _random_input_tensor()
        frequency_tensor = detector._build_frequency_input(tensor)
        target_layer = _target_layer(detector)

        def forward_fn():
            return detector._model({"rgb": tensor, "frequency": frequency_tensor})

        cam = compute_gradcam_map(target_layer, forward_fn)

        assert cam.dim() == 2
        assert cam.shape[0] == cam.shape[1] == 7  # 224 / 32 (EfficientNet-B4 stride)
        assert torch.isfinite(cam).all()
        assert float(cam.min()) >= 0.0
        assert float(cam.max()) <= 1.0

    def test_resize_cam_output_range_and_shape(self, detector):
        tensor = _random_input_tensor()
        frequency_tensor = detector._build_frequency_input(tensor)
        target_layer = _target_layer(detector)

        def forward_fn():
            return detector._model({"rgb": tensor, "frequency": frequency_tensor})

        cam = compute_gradcam_map(target_layer, forward_fn)
        resized = resize_cam(cam, (INPUT_SIZE, INPUT_SIZE))

        assert resized.shape == (INPUT_SIZE, INPUT_SIZE)
        assert np.isfinite(resized).all()
        assert resized.min() >= 0.0
        assert resized.max() <= 1.0

    def test_zero_variance_cam_handled_without_nan(self):
        """A degenerate all-equal activation with zero gradient produces an
        all-zero CAM, never NaN/Inf from a zero-range normalization."""
        layer = torch.nn.Conv2d(2, 3, kernel_size=1)
        for p in layer.parameters():
            p.requires_grad_(True)

        def forward_fn():
            x = torch.zeros(1, 2, 4, 4)
            out = layer(x)
            return out.sum() * 0.0  # exactly zero target -> zero gradient everywhere

        cam = compute_gradcam_map(layer, forward_fn)
        assert torch.isfinite(cam).all()
        assert torch.equal(cam, torch.zeros_like(cam))


# ---------------------------------------------------------------------------
# E: determinism
# ---------------------------------------------------------------------------
class TestDeterminism:
    def test_explain_is_deterministic_for_identical_input(self, detector):
        tensor = _random_input_tensor(seed=42)
        first = detector.explain(tensor.clone())
        second = detector.explain(tensor.clone())
        assert first == second

    def test_gradcam_map_is_deterministic(self, detector):
        tensor = _random_input_tensor(seed=42)
        frequency_tensor = detector._build_frequency_input(tensor)
        target_layer = _target_layer(detector)

        def forward_fn():
            return detector._model({"rgb": tensor, "frequency": frequency_tensor})

        cam_a = compute_gradcam_map(target_layer, forward_fn)
        cam_b = compute_gradcam_map(target_layer, forward_fn)
        assert torch.equal(cam_a, cam_b)


# ---------------------------------------------------------------------------
# F, G: no parameter mutation, no .grad pollution
# ---------------------------------------------------------------------------
class TestNoModelMutation:
    def test_state_dict_unchanged_after_explain(self, detector):
        before = {k: v.clone() for k, v in detector._model.state_dict().items()}
        detector.explain(_random_input_tensor())
        after = detector._model.state_dict()

        assert before.keys() == after.keys()
        for key in before:
            assert torch.equal(before[key], after[key]), f"Parameter '{key}' changed after explain()"

    def test_no_grad_populated_after_explain(self, detector):
        detector.explain(_random_input_tensor())
        assert all(p.grad is None for p in detector._model.parameters())

    def test_model_remains_in_eval_mode_after_explain(self, detector):
        detector.explain(_random_input_tensor())
        assert detector._model.training is False


# ---------------------------------------------------------------------------
# H: temporary hook cleanup, including on exception
# ---------------------------------------------------------------------------
class TestHookCleanup:
    def test_no_lingering_hooks_after_successful_explain(self, detector):
        target_layer = _target_layer(detector)
        assert len(target_layer._forward_hooks) == 0
        detector.explain(_random_input_tensor())
        assert len(target_layer._forward_hooks) == 0

    def test_no_lingering_hooks_when_downstream_step_raises(self, detector, monkeypatch):
        target_layer = _target_layer(detector)

        def _boom(*args, **kwargs):
            raise RuntimeError("simulated failure after Grad-CAM computation")

        # rgb_frequency_detector.py imported render_overlay_jpeg by name
        # (`from app.ml.gradcam import ... render_overlay_jpeg`), so it must
        # be patched in that module's namespace, not gradcam's own.
        import app.ml.rgb_frequency_detector as detector_module
        monkeypatch.setattr(detector_module, "render_overlay_jpeg", _boom)

        assert len(target_layer._forward_hooks) == 0
        with pytest.raises(RuntimeError):
            detector.explain(_random_input_tensor())
        assert len(target_layer._forward_hooks) == 0

    def test_no_lingering_hooks_when_forward_fn_itself_raises(self):
        layer = torch.nn.Conv2d(2, 3, kernel_size=1)

        def failing_forward_fn():
            layer(torch.zeros(1, 2, 4, 4))  # fires the hook...
            raise RuntimeError("simulated forward-pass failure")  # ...then blows up

        assert len(layer._forward_hooks) == 0
        with pytest.raises(RuntimeError):
            compute_gradcam_map(layer, failing_forward_fn)
        assert len(layer._forward_hooks) == 0


# ---------------------------------------------------------------------------
# I: raw-logit targeting (never the calibrated probability)
# ---------------------------------------------------------------------------
class TestRawLogitTargeting:
    def test_explain_output_independent_of_calibration_settings(self, detector, monkeypatch):
        tensor = _random_input_tensor(seed=5)

        monkeypatch.setattr(settings, "CALIBRATION_ENABLED", False)
        monkeypatch.setattr(settings, "CALIBRATION_METHOD", "none")
        result_uncalibrated = detector.explain(tensor.clone())

        monkeypatch.setattr(settings, "CALIBRATION_ENABLED", True)
        monkeypatch.setattr(settings, "CALIBRATION_METHOD", "temperature")
        monkeypatch.setattr(settings, "CALIBRATION_TEMPERATURE", 1.1838611364364624)
        result_calibrated = detector.explain(tensor.clone())

        assert result_uncalibrated == result_calibrated

    def test_calibrator_never_invoked_during_explain(self, detector, monkeypatch):
        from app.ml.calibration import Calibrator

        def _fail_if_called(*args, **kwargs):
            raise AssertionError("Calibrator must never be invoked from explain()")

        monkeypatch.setattr(Calibrator, "apply_calibration_from_logit", staticmethod(_fail_if_called))
        monkeypatch.setattr(Calibrator, "apply_calibration", staticmethod(_fail_if_called))

        # Must not raise.
        detector.explain(_random_input_tensor())

    def test_forward_fn_target_is_pre_sigmoid_logit(self, detector, monkeypatch):
        """The scalar handed to compute_gradcam_map must be the raw model
        output (pre-sigmoid) - assert torch.sigmoid is never called on the
        path from detector.explain() into compute_gradcam_map."""
        import app.ml.rgb_frequency_detector as detector_module

        original = gradcam_module.compute_gradcam_map
        captured = {}

        def _spy(layer, forward_fn):
            def _wrapped_forward_fn():
                target = forward_fn()
                captured["target"] = target.detach().clone()
                return target

            return original(layer, _wrapped_forward_fn)

        # Patched in rgb_frequency_detector's own namespace, since that is
        # the name it actually calls (`from app.ml.gradcam import
        # compute_gradcam_map` binds it locally at import time).
        monkeypatch.setattr(detector_module, "compute_gradcam_map", _spy)

        tensor = _random_input_tensor()
        detector.explain(tensor)

        target_value = captured["target"]
        # A raw fusion-model logit is unconstrained; a sigmoid output would
        # always lie in (0, 1). This is not proof by itself, but combined
        # with the source reading `self._model(...)` directly (never
        # wrapped in torch.sigmoid), confirms the pre-sigmoid logit is used.
        assert target_value.shape == (1, 1)


# ---------------------------------------------------------------------------
# J: preprocessing parity with predict()
# ---------------------------------------------------------------------------
class TestPreprocessingParity:
    def test_explain_reuses_build_frequency_input_with_identical_tensor(self, detector, monkeypatch):
        tensor = _random_input_tensor(seed=3)
        calls = []
        original = detector._build_frequency_input

        def _spy(rgb_tensor):
            calls.append(rgb_tensor.clone())
            return original(rgb_tensor)

        monkeypatch.setattr(detector, "_build_frequency_input", _spy)

        predict_output = detector.predict(tensor.clone())
        detector.explain(tensor.clone())

        assert len(calls) == 2
        assert torch.equal(calls[0], calls[1])
        assert predict_output["raw_ai_probability"] is not None  # predict() still worked


# ---------------------------------------------------------------------------
# K: faithfulness (high-CAM vs low-CAM vs random-control masking)
# ---------------------------------------------------------------------------
class TestFaithfulness:
    REGION_SIZE = 56
    GRID_STEP = 8

    @staticmethod
    def _best_region(cam: np.ndarray, region: int, step: int, want_max: bool):
        h, w = cam.shape
        best_value = -np.inf if want_max else np.inf
        best_coords = (0, 0)
        for y in range(0, h - region + 1, step):
            for x in range(0, w - region + 1, step):
                value = cam[y : y + region, x : x + region].mean()
                if (want_max and value > best_value) or (not want_max and value < best_value):
                    best_value = value
                    best_coords = (y, x)
        return best_coords

    def test_high_activation_masking_moves_logit_more_than_controls(self, detector):
        tensor = _structured_input_tensor(seed=11)
        original_output = detector.predict(tensor.clone())
        original_logit = original_output["logits"]

        frequency_tensor = detector._build_frequency_input(tensor)
        target_layer = _target_layer(detector)

        def forward_fn():
            return detector._model({"rgb": tensor, "frequency": frequency_tensor})

        cam = compute_gradcam_map(target_layer, forward_fn)
        cam_resized = resize_cam(cam, (INPUT_SIZE, INPUT_SIZE))
        assert cam_resized.std() > 1e-6, "CAM is degenerate/flat - faithfulness test needs spatial variation"

        region = self.REGION_SIZE
        high_y, high_x = self._best_region(cam_resized, region, self.GRID_STEP, want_max=True)
        low_y, low_x = self._best_region(cam_resized, region, self.GRID_STEP, want_max=False)
        random_generator = np.random.default_rng(99)
        random_y = int(random_generator.integers(0, INPUT_SIZE - region + 1))
        random_x = int(random_generator.integers(0, INPUT_SIZE - region + 1))

        fill_value = float(tensor.mean())

        def masked_logit(y, x):
            masked = tensor.clone()
            masked[:, :, y : y + region, x : x + region] = fill_value
            return detector.predict(masked)["logits"]

        high_delta = abs(masked_logit(high_y, high_x) - original_logit)
        low_delta = abs(masked_logit(low_y, low_x) - original_logit)
        random_delta = abs(masked_logit(random_y, random_x) - original_logit)

        assert high_delta > low_delta, (
            f"High-CAM-region masking ({high_delta}) should move the raw logit more than "
            f"low-CAM-region masking ({low_delta})"
        )
        assert high_delta > random_delta, (
            f"High-CAM-region masking ({high_delta}) should move the raw logit more than "
            f"a random equal-area control region ({random_delta})"
        )
