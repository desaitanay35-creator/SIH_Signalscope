"""
Generic Grad-CAM computation - deliberately decoupled from any specific
detector/architecture wiring so it is independently unit-testable and
reusable.

This module is exactly the kind of concrete ML-framework code
`test_e2e_model_abstraction_isolation` (backend/tests/test_e2e_integration.py)
is designed to keep out of `app.services`/`app.api`/`app.core`: it is
excluded from that check alongside `app.ml.rgb_frequency_detector`, the
only other module in `app.*` allowed to import torch. See
.claude/specs/09-backend-ml-integration.md ("Model / Architecture
Changes") for the precedent.

Responsibility split:
- This module knows nothing about EfficientNet, the fusion model, or the
  frequency branch - it operates on a caller-supplied `nn.Module` (the
  target layer) and a caller-supplied zero-argument `forward_fn` that runs
  a model forward pass and returns the scalar Grad-CAM target.
- `backend/app/ml/rgb_frequency_detector.py` owns all model-specific
  wiring (which layer, which tensors, which logit).

Guarantees:
- Never mutates model parameters and never populates any parameter's
  `.grad` - gradients are computed only w.r.t. the captured activation
  tensor via `torch.autograd.grad(inputs=[...])`, never `.backward()`.
- The temporary forward hook used to capture the target layer's
  activation is always removed before this module returns control to the
  caller, including when an exception is raised mid-computation.
"""

from __future__ import annotations

import io
from contextlib import contextmanager
from typing import Callable, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn


@contextmanager
def _capture_activation(layer: nn.Module):
    """Registers a temporary forward hook on `layer` that captures its
    output activation tensor into the yielded dict under the key
    "activation". The hook is guaranteed removed on exit, including when
    the caller's code inside the `with` block raises."""
    captured: Dict[str, torch.Tensor] = {}

    def _hook(_module: nn.Module, _inputs, output: torch.Tensor) -> None:
        captured["activation"] = output

    handle = layer.register_forward_hook(_hook)
    try:
        yield captured
    finally:
        handle.remove()


def compute_gradcam_map(layer: nn.Module, forward_fn: Callable[[], torch.Tensor]) -> torch.Tensor:
    """Computes a single-image Grad-CAM map from `layer`'s activation,
    targeting the scalar produced by `forward_fn()`.

    `forward_fn` must run the model (with gradients enabled) and return a
    tensor with exactly one element (e.g. shape `(1, 1)` or `()`) built
    from a forward pass that includes `layer` - the activation is captured
    via a temporary forward hook (see `_capture_activation`), removed
    before this function returns, including on exception.

    Does not mutate model parameters and does not populate any parameter's
    `.grad`: the gradient is computed only w.r.t. the captured activation
    tensor via `torch.autograd.grad(inputs=[...])`.

    Returns a `(H, W)` tensor normalized to `[0, 1]`. An exactly flat/zero
    map (no positive-gradient-weighted activation anywhere - e.g. a
    degenerate constant input) is handled without NaN/Inf: it is returned
    as all-zeros rather than dividing by a zero range.
    """
    with _capture_activation(layer) as captured:
        with torch.enable_grad():
            target = forward_fn()
            activation = captured.get("activation")
            if activation is None:
                raise RuntimeError(
                    "Target layer produced no activation during the forward pass - "
                    "the forward hook never fired."
                )
            if not activation.requires_grad:
                raise RuntimeError(
                    "Captured activation does not require grad; Grad-CAM needs "
                    "gradients enabled for the forward pass (use torch.enable_grad())."
                )

            target_scalar = target.reshape(())
            gradients = torch.autograd.grad(
                outputs=target_scalar,
                inputs=[activation],
                retain_graph=False,
                create_graph=False,
            )[0]

    activation = activation.detach()
    gradients = gradients.detach()

    # activation/gradients: (N, C, H, W) - N == 1 for a single-image
    # forward pass (the only case this module is used for).
    alpha = gradients.mean(dim=(2, 3))  # (N, C) - global-average-pooled gradient per channel
    weighted = (alpha.unsqueeze(-1).unsqueeze(-1) * activation).sum(dim=1)  # (N, H, W)
    cam = torch.relu(weighted)[0]  # (H, W)

    cam_min = cam.min()
    cam_max = cam.max()
    spread = cam_max - cam_min
    if float(spread) <= 0.0:
        return torch.zeros_like(cam)
    return (cam - cam_min) / spread


def resize_cam(cam: torch.Tensor, size: Tuple[int, int]) -> np.ndarray:
    """Upsamples a `(H, W)` CAM tensor (already normalized to `[0, 1]`) to
    `size = (height, width)` via bilinear interpolation, returning a plain
    numpy array re-clamped into `[0, 1]` (bilinear interpolation of an
    in-range signal can overshoot by a negligible float epsilon at sharp
    edges)."""
    resized = F.interpolate(
        cam.unsqueeze(0).unsqueeze(0),
        size=size,
        mode="bilinear",
        align_corners=False,
    )
    resized = resized.squeeze(0).squeeze(0)
    return resized.clamp(0.0, 1.0).cpu().numpy()


def _apply_heat_colormap(cam_hw: np.ndarray) -> np.ndarray:
    """Maps a `(H, W)` array in `[0, 1]` to an `(H, W, 3)` RGB array in
    `[0, 1]` using a simple blue -> green -> red ramp (a standard
    piecewise-linear approximation of a "jet"-style heatmap colormap).
    Pure numpy - no additional plotting dependency."""
    r = np.clip(1.5 - np.abs(4.0 * cam_hw - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * cam_hw - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * cam_hw - 1.0), 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def render_overlay_jpeg(
    base_rgb_0_1: np.ndarray,
    cam_hw: np.ndarray,
    alpha: float = 0.45,
    jpeg_quality: int = 90,
) -> bytes:
    """Alpha-blends a heat-colormapped CAM onto the base RGB image and
    encodes the result as JPEG bytes.

    `base_rgb_0_1`: `(H, W, 3)` array with values in `[0, 1]`.
    `cam_hw`: `(H, W)` array with values in `[0, 1]`, same `(H, W)` as
    `base_rgb_0_1` (already resized via `resize_cam`).

    The blend weight scales with the CAM value itself (stronger tint where
    the model attended more), rather than a uniform overlay opacity, so
    low-attention regions remain visually close to the original image.
    """
    base_uint8 = np.clip(base_rgb_0_1 * 255.0, 0, 255).astype(np.float32)
    heat_rgb = _apply_heat_colormap(cam_hw).astype(np.float32) * 255.0

    blend_weight = np.clip(alpha * cam_hw, 0.0, 1.0)[..., None]  # (H, W, 1)
    blended = base_uint8 * (1.0 - blend_weight) + heat_rgb * blend_weight
    blended_uint8 = np.clip(blended, 0, 255).astype(np.uint8)

    image = Image.fromarray(blended_uint8, mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=jpeg_quality)
    return buffer.getvalue()
