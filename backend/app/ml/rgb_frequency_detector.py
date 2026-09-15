"""
Backend adapter wrapping the trained RGB+frequency fusion model
(`model.architectures.fusion_model.RGBFrequencyFusionModel`) behind the
existing `ImageDetector` contract.

This module intentionally contains no model architecture or FFT logic of
its own - it imports the already-trained/evaluated model code from
`model/` (checkpoint loading, architecture dispatch, frequency-domain
feature extraction) rather than duplicating it, per
.claude/specs/09-backend-ml-integration.md ("Model / Architecture
Changes", requirement 4).

The one non-trivial piece of glue this adapter owns is reconstructing the
frequency branch's input from the backend's already-ImageNet-normalized
RGB tensor: `model.training.frequency_features.compute_log_magnitude_spectrum`
requires an un-normalized [0, 1] RGB array (the same precondition training
satisfies via `model.training.dual_branch_dataset`), so this adapter first
inverts the backend's normalization (`x * std + mean`) before computing the
spectrum. See "Model / Architecture Changes" in the spec for the full
rationale - a mismatch here would silently diverge from the training-time
frequency input without raising any error, so it is called out explicitly.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

from app.core.config import settings
from app.core.errors import ModelIncompatibleError
from app.core.logging import logger
from app.ml.detector import ImageDetector

# `model.*` lives at the repository root, one level above `backend/`. When
# the backend is launched with `backend/` as the working directory (e.g.
# `uvicorn app.main:app` run from inside `backend/`), the repo root is not
# automatically on sys.path. Mirrors the same defensive path bootstrap
# `backend/tests/conftest.py` already applies in the opposite direction
# (adding `backend/` so `app.*` resolves under pytest).
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from model.architectures.fusion_model import ARCHITECTURE_ID as FUSION_ARCHITECTURE_ID  # noqa: E402
from model.training.checkpoint import UnknownArchitectureError, load_model_from_checkpoint  # noqa: E402
from model.training.frequency_features import compute_log_magnitude_spectrum  # noqa: E402


def _resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_and_validate_model(path: Union[str, Path], expected_architecture: str) -> torch.nn.Module:
    """Loads and validates a checkpoint against `expected_architecture`.

    Raises `ModelIncompatibleError` (never a raw torch/architecture-registry
    exception, and never a silently-returned model of the wrong type) for a
    corrupt/undeserializable file, an architecture mismatch, or an
    incompatible state_dict. Reuses `model.training.checkpoint`'s existing
    architecture-dispatch registry rather than reimplementing it.
    """
    path = Path(path)

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ModelIncompatibleError(
            message=f"Failed to deserialize model checkpoint at '{path}': {exc}",
            details={"checkpoint_path": str(path)},
        ) from exc

    if not isinstance(checkpoint, dict):
        raise ModelIncompatibleError(
            message=f"Checkpoint at '{path}' does not contain a recognized checkpoint payload.",
            details={"checkpoint_path": str(path)},
        )

    model_config = checkpoint.get("model_config") or {}
    found_architecture = model_config.get("architecture", "efficientnet_b4")
    if found_architecture != expected_architecture:
        raise ModelIncompatibleError(
            message=(
                f"Checkpoint at '{path}' declares architecture '{found_architecture}', "
                f"which is incompatible with the expected architecture '{expected_architecture}'."
            ),
            details={
                "checkpoint_path": str(path),
                "expected_architecture": expected_architecture,
                "found_architecture": found_architecture,
            },
        )

    try:
        model = load_model_from_checkpoint(path, map_location="cpu")
    except UnknownArchitectureError as exc:
        raise ModelIncompatibleError(
            message=f"Checkpoint at '{path}' declares an unrecognized architecture: {exc}",
            details={"checkpoint_path": str(path)},
        ) from exc
    except (RuntimeError, KeyError) as exc:
        raise ModelIncompatibleError(
            message=f"Checkpoint at '{path}' has a state_dict incompatible with '{expected_architecture}': {exc}",
            details={"checkpoint_path": str(path)},
        ) from exc

    return model


class RGBFrequencyFusionDetector(ImageDetector):
    """Real detector adapter for the trained `rgb_frequency_fusion` model.

    Construction either fully succeeds (checkpoint loaded, validated,
    `.eval()`'d, ready for inference) or raises `ModelIncompatibleError` -
    there is no partially-loaded state. `is_loaded()` on a successfully
    constructed instance is therefore always `True`; `is_loaded() == False`
    remains exclusively `StubImageDetector`'s/missing-weights behavior (see
    `backend/app/ml/model_loader.py`).
    """

    def __init__(
        self,
        weights_path: Union[str, Path],
        model_version: str,
        architecture: str = FUSION_ARCHITECTURE_ID,
    ) -> None:
        self._weights_path = str(Path(weights_path))
        self._model_version = model_version
        self._architecture = architecture
        self._device = _resolve_device()

        model = _load_and_validate_model(weights_path, expected_architecture=architecture)
        model.to(self._device)
        model.eval()
        self._model = model

        mean = np.asarray(settings.MODEL_NORM_MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
        std = np.asarray(settings.MODEL_NORM_STD, dtype=np.float32).reshape(1, 3, 1, 1)
        self._mean = torch.from_numpy(mean)
        self._std = torch.from_numpy(std)

        logger.info(
            f"RGBFrequencyFusionDetector loaded (architecture={self._architecture}, "
            f"device={self._device}, checkpoint={self._weights_path})."
        )

    def is_loaded(self) -> bool:
        return True

    def get_info(self) -> Dict[str, Any]:
        return {
            "name": "SignalScope Detector",
            "version": self._model_version,
            "task": "real-vs-ai-generated",
            "loaded": True,
            "device": str(self._device),
            "architecture": self._architecture,
            "checkpoint_path": self._weights_path,
        }

    def _denormalize_to_unit_range(self, rgb_tensor: torch.Tensor) -> torch.Tensor:
        """Inverts the backend's ImageNet mean/std normalization, recovering
        the [0, 1]-range RGB tensor that `compute_log_magnitude_spectrum`
        requires (see module docstring)."""
        mean = self._mean.to(dtype=rgb_tensor.dtype, device=rgb_tensor.device)
        std = self._std.to(dtype=rgb_tensor.dtype, device=rgb_tensor.device)
        return (rgb_tensor * std + mean).clamp(0.0, 1.0)

    def _build_frequency_input(self, rgb_tensor: torch.Tensor) -> torch.Tensor:
        """Builds the (N, 1, H, W) frequency-branch input for every image in
        the batch (N >= 1), by reusing `compute_log_magnitude_spectrum`
        per-image on the recovered [0, 1] RGB array - never duplicating the
        FFT/log-magnitude computation itself."""
        denormalized = self._denormalize_to_unit_range(rgb_tensor)
        batch_size = denormalized.shape[0]

        frequency_tensors = []
        for i in range(batch_size):
            rgb_array = denormalized[i].permute(1, 2, 0).detach().cpu().numpy()
            frequency_tensors.append(compute_log_magnitude_spectrum(rgb_array))

        return torch.stack(frequency_tensors, dim=0)

    def predict(self, image_tensor: Any) -> Dict[str, Any]:
        tensor = image_tensor
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.as_tensor(np.asarray(tensor), dtype=torch.float32)
        tensor = tensor.to(dtype=torch.float32, device=self._device)

        if tensor.dim() != 4:
            raise ValueError(f"Expected an NCHW RGB tensor, got shape {tuple(tensor.shape)!r}.")

        batch_size = tensor.shape[0]
        frequency_tensor = self._build_frequency_input(tensor).to(self._device)

        with torch.no_grad():
            raw_logits = self._model({"rgb": tensor, "frequency": frequency_tensor})

        raw_logits = raw_logits.reshape(batch_size)
        probabilities = torch.sigmoid(raw_logits)

        if batch_size == 1:
            logit_value = float(raw_logits[0].item())
            prob_value = float(probabilities[0].item())
            return {
                "raw_ai_probability": prob_value,
                "raw_real_probability": 1.0 - prob_value,
                "logits": logit_value,
                "model_version": self._model_version,
            }

        logit_values = [float(v) for v in raw_logits.tolist()]
        prob_values = [float(v) for v in probabilities.tolist()]
        return {
            "raw_ai_probability": prob_values,
            "raw_real_probability": [1.0 - p for p in prob_values],
            "logits": logit_values,
            "model_version": self._model_version,
        }

    def explain(self, image_tensor: Any) -> Optional[Any]:
        return None
