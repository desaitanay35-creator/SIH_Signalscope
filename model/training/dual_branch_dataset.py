"""
Dual-branch (RGB + frequency) dataset construction for the SignalScope
training/evaluation pipeline.

Thin wrappers around `data.dataset_loader.SignalScopeDataset`, using the
exact same duck-typed-preprocessor pattern `model/training/dataset.py`
already established for RGB-only augmentation: `SignalScopeDataset`'s
`preprocessor` argument only needs a `.preprocess(image_path)` method - it
returns a dict `{"rgb": tensor, "frequency": tensor}` here instead of a
single tensor. No change to `data/dataset_loader.py` or
`data/preprocessor.py` is required or made.

Critically, both tensors in a dual-branch sample are derived from the SAME
loaded (and, for training, SAME augmented) PIL image - never two
independently-loaded or independently-augmented copies. See
.claude/specs/06-frequency-fusion.md ("Augmentation interaction",
"Input construction for training/evaluation (dataset wiring)").

Responsible Team Member: Member 3 (Data Pipeline & Preprocessing)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import numpy as np
import torch
from PIL import Image

from data.dataset_loader import DatasetRecord, SignalScopeDataset
from data.preprocessor import ImagePreprocessor
from model.training.augmentation import build_train_transform, tensorize
from model.training.frequency_features import compute_log_magnitude_spectrum


def _rgb_array_0_1(image: Image.Image) -> np.ndarray:
    """The same pre-normalization (H, W, 3) [0, 1] array
    `model.training.augmentation.tensorize` derives its RGB tensor from -
    exposed here so the frequency branch's input is computed from the
    IDENTICAL pixel values, never an independently-loaded or
    independently-resized copy."""
    return np.asarray(image, dtype=np.float32) / 255.0


class AugmentedDualBranchPreprocessor:
    """Applies ONE shared augmentation to the loaded image, then derives
    both the RGB tensor (ImageNet-normalized, via the existing
    `tensorize()`) and the frequency tensor (log-magnitude FFT spectrum,
    computed on the pre-normalization pixel values) from that single
    augmented image.

    Exposes the same `.preprocess(image_path)` interface
    `SignalScopeDataset` expects, so dual-branch training can be injected
    without changing `data/dataset_loader.py`.
    """

    def __init__(self, preprocessor: ImagePreprocessor) -> None:
        self._preprocessor = preprocessor
        self._transform = build_train_transform(preprocessor)

    def preprocess(self, image_path) -> Dict[str, torch.Tensor]:
        image = self._preprocessor.load_image(image_path)
        image = self._transform(image)  # the ONE shared augmented image
        rgb_tensor = tensorize(image, self._preprocessor)
        frequency_tensor = compute_log_magnitude_spectrum(_rgb_array_0_1(image))
        return {"rgb": rgb_tensor, "frequency": frequency_tensor}


class DeterministicDualBranchPreprocessor:
    """Val/test equivalent: the same two-derivation logic, with only a
    deterministic resize (no augmentation) - identical to inference-time
    preprocessing for both branches."""

    def __init__(self, preprocessor: ImagePreprocessor) -> None:
        self._preprocessor = preprocessor

    def preprocess(self, image_path) -> Dict[str, torch.Tensor]:
        image = self._preprocessor.load_image(image_path)
        image = image.resize(self._preprocessor.target_size, Image.BILINEAR)
        rgb_tensor = tensorize(image, self._preprocessor)
        frequency_tensor = compute_log_magnitude_spectrum(_rgb_array_0_1(image))
        return {"rgb": rgb_tensor, "frequency": frequency_tensor}


def build_dual_branch_train_dataset(
    records: Sequence[DatasetRecord],
    preprocessor: ImagePreprocessor,
    root_dir: Optional[Union[str, Path]] = None,
) -> SignalScopeDataset:
    """Builds the training dataset for the frequency-only/fusion models:
    one shared augmentation, both branches' tensors derived from it."""
    return SignalScopeDataset(records, preprocessor=AugmentedDualBranchPreprocessor(preprocessor), root_dir=root_dir)


def build_dual_branch_val_dataset(
    records: Sequence[DatasetRecord],
    preprocessor: ImagePreprocessor,
    root_dir: Optional[Union[str, Path]] = None,
) -> SignalScopeDataset:
    """Builds the validation/test dataset for the frequency-only/fusion
    models: deterministic preprocessing only, no augmentation - identical
    to inference-time preprocessing for both branches."""
    return SignalScopeDataset(records, preprocessor=DeterministicDualBranchPreprocessor(preprocessor), root_dir=root_dir)
