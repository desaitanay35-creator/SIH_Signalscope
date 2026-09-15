"""
Deterministic frequency-domain feature extraction for the SignalScope
training/evaluation pipeline.

Implements exactly the representation chosen in
.claude/specs/06-frequency-fusion.md ("Frequency representation" / "Input
construction"): grayscale -> 2D FFT -> fftshift -> magnitude -> log1p ->
per-image min-max normalization. Pure and deterministic - no dependency on
any other image's or the dataset's statistics (per-image normalization,
not a fitted running statistic, specifically to avoid a subtle leakage
surface - see the spec's "Input construction", step 6).

Responsible Team Member: Member 3 (Data Pipeline & Preprocessing)
"""

from __future__ import annotations

import numpy as np
import torch

# ITU-R BT.601 luminance weights - standard, deterministic grayscale conversion.
GRAYSCALE_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float64)

# Prevents a divide-by-zero (NaN) when log_magnitude is constant (e.g. an
# all-zero/all-black input image, whose spectrum is uniformly zero) -
# never used to avoid log(0), since log1p(0) = 0 already handles that.
_MIN_MAX_EPSILON = 1e-8


def compute_log_magnitude_spectrum(rgb_array: np.ndarray) -> torch.Tensor:
    """Computes the log-magnitude FFT spectrum of an RGB image array.

    Args:
        rgb_array: an (H, W, 3) array with values in [0, 1] - the same
            pre-normalization array `model.training.augmentation.tensorize`
            derives its RGB tensor from. Must be called on the SAME
            (already resized, already augmented for training) image the
            RGB branch derives its tensor from, and BEFORE any ImageNet
            mean/std normalization is applied to it - channel-wise
            normalization would distort the spectrum in a way with no
            physical/forensic meaning. See
            .claude/specs/06-frequency-fusion.md ("Augmentation
            interaction").

    Returns:
        A (1, H, W) float32 tensor with values in [0, 1].

    Raises:
        ValueError: if `rgb_array` is not an (H, W, 3) array.
    """
    if rgb_array.ndim != 3 or rgb_array.shape[-1] != 3:
        raise ValueError(f"Expected an (H, W, 3) RGB array, got shape {rgb_array.shape!r}.")

    grayscale = np.tensordot(rgb_array.astype(np.float64), GRAYSCALE_WEIGHTS, axes=([-1], [0]))

    spectrum = np.fft.fft2(grayscale)
    spectrum = np.fft.fftshift(spectrum)
    magnitude = np.abs(spectrum)
    log_magnitude = np.log1p(magnitude)  # log(1 + magnitude); magnitude >= 0 always, so never log(<=0).

    min_value = log_magnitude.min()
    max_value = log_magnitude.max()
    normalized = (log_magnitude - min_value) / (max_value - min_value + _MIN_MAX_EPSILON)

    normalized = np.ascontiguousarray(normalized, dtype=np.float32)
    return torch.from_numpy(normalized).unsqueeze(0)
