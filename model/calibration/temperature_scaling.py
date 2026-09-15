"""
Temperature scaling for SignalScope (Step 8A).

Post-training, checkpoint-preserving calibration: fits exactly one scalar
parameter `T > 0` on top of a frozen detector's raw (pre-sigmoid) logits.
No detector weights are read or modified by this module - it operates
purely on already-computed `torch.Tensor` logits/labels, with no
dataset/checkpoint/config dependency (matching the pure-function style of
model/evaluation/metrics.py).

Calibration always operates on raw logits, never on already-sigmoided
probabilities: every public function here takes a logit tensor as input.
See .claude/specs/08a-probability-calibration.md ("Model / Architecture
Changes").

Responsible Team Member: Step 8A (Probability Calibration)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Union

import torch
from torch import nn

from model.calibration.calibration_metrics import EmptyCalibrationSetError

DEFAULT_LBFGS_MAX_ITER = 50
DEFAULT_LBFGS_LR = 1.0


class TemperatureFitError(RuntimeError):
    """Raised when a temperature fit does not converge to a finite,
    strictly positive temperature. This must never happen under normal
    operation (T = exp(log_temperature) is positive by construction) - it
    exists as a hard safety net against numerical overflow, not an
    expected code path."""


class TemperatureScaler(nn.Module):
    """A single learned scalar temperature `T`, parameterized as
    `T = exp(log_temperature)` so `T > 0` holds by construction - no
    manual clamp is needed.

    `forward(logits)` returns the *scaled logits* (`logits / T`), not
    probabilities - this is deliberate so the module composes directly
    with `nn.BCEWithLogitsLoss` during fitting (see `fit_temperature`).
    Use the module-level `apply_temperature()` function to get calibrated
    probabilities.
    """

    def __init__(self) -> None:
        super().__init__()
        # log_temperature = 0.0 -> T = 1.0 (identity / no calibration) at
        # initialization - the LBFGS fit starts from "no change" rather
        # than an arbitrary guess.
        self.log_temperature = nn.Parameter(torch.zeros(1))

    @property
    def temperature(self) -> torch.Tensor:
        """T = exp(log_temperature), always > 0 for any finite
        log_temperature."""
        return torch.exp(self.log_temperature)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Returns `logits / T` (still raw scaled logits, no sigmoid)."""
        return logits / self.temperature


def apply_temperature(logits: Union[torch.Tensor, Sequence[float]], temperature: float) -> torch.Tensor:
    """Applies a fixed temperature to raw logits and returns calibrated
    probabilities: `sigmoid(logit / T)`.

    `temperature` must be a finite, strictly positive number - this
    function never accepts or infers a temperature from probabilities;
    the input `logits` must be pre-sigmoid raw scores. Returns a tensor
    with every entry in the open interval (0, 1) (sigmoid's range).
    """
    if not (temperature > 0) or not torch.isfinite(torch.as_tensor(float(temperature))):
        raise ValueError(f"temperature must be a finite, strictly positive number, got {temperature!r}.")
    logits_tensor = logits if isinstance(logits, torch.Tensor) else torch.as_tensor(logits, dtype=torch.float32)
    return torch.sigmoid(logits_tensor / temperature)


@dataclass
class TemperatureFitResult:
    """Everything a reproducibility artifact needs to record about one
    temperature fit - see .claude/specs/08a-probability-calibration.md
    ("Reproducibility")."""

    scaler: TemperatureScaler
    temperature: float
    initial_nll: float
    final_nll: float
    num_fit_samples: int
    lbfgs_max_iter: int
    lbfgs_lr: float
    optimizer_iterations: int

    def to_dict(self) -> dict:
        return {
            "temperature": self.temperature,
            "initial_nll": self.initial_nll,
            "final_nll": self.final_nll,
            "num_fit_samples": self.num_fit_samples,
            "lbfgs_max_iter": self.lbfgs_max_iter,
            "lbfgs_lr": self.lbfgs_lr,
            "optimizer_iterations": self.optimizer_iterations,
        }


def fit_temperature(
    logits: Union[torch.Tensor, Sequence[float]],
    labels: Union[torch.Tensor, Sequence[int]],
    max_iter: int = DEFAULT_LBFGS_MAX_ITER,
    lr: float = DEFAULT_LBFGS_LR,
) -> TemperatureFitResult:
    """Fits a strictly positive temperature parameter by minimizing
    validation negative log-likelihood using LBFGS.

    `logits` must be raw, pre-sigmoid scores (never already-sigmoided
    probabilities) and `labels` must be binary (0/1). The caller is
    responsible for ensuring `logits`/`labels` come only from the
    validation split - this function has no knowledge of splits and will
    fit on whatever it is given (see calibrate.py for the split
    enforcement). Raises EmptyCalibrationSetError on empty input, and
    TemperatureFitError if the fitted temperature is not finite and
    strictly positive (a numerical-safety check; not expected to trigger
    under the `T = exp(log_temperature)` parameterization).
    """
    logits_tensor = logits if isinstance(logits, torch.Tensor) else torch.as_tensor(logits, dtype=torch.float32)
    labels_tensor = labels if isinstance(labels, torch.Tensor) else torch.as_tensor(labels, dtype=torch.float32)
    logits_tensor = logits_tensor.reshape(-1).float()
    labels_tensor = labels_tensor.reshape(-1).float()

    if logits_tensor.numel() == 0:
        raise EmptyCalibrationSetError("Cannot fit a temperature on an empty set of validation logits.")
    if logits_tensor.numel() != labels_tensor.numel():
        raise ValueError(
            f"logits and labels must have the same length "
            f"(got {logits_tensor.numel()} and {labels_tensor.numel()})."
        )

    scaler = TemperatureScaler()
    loss_fn = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        initial_nll = float(loss_fn(scaler(logits_tensor), labels_tensor).item())

    optimizer = torch.optim.LBFGS([scaler.log_temperature], lr=lr, max_iter=max_iter, line_search_fn="strong_wolfe")

    iterations_run = 0

    def closure() -> torch.Tensor:
        nonlocal iterations_run
        optimizer.zero_grad()
        loss = loss_fn(scaler(logits_tensor), labels_tensor)
        loss.backward()
        iterations_run += 1
        return loss

    optimizer.step(closure)

    with torch.no_grad():
        final_nll = float(loss_fn(scaler(logits_tensor), labels_tensor).item())
        fitted_temperature = float(scaler.temperature.item())

    if not (fitted_temperature > 0) or not torch.isfinite(torch.tensor(fitted_temperature)):
        raise TemperatureFitError(
            f"Fitted temperature is not finite and strictly positive: {fitted_temperature!r}. "
            "This indicates a numerical-fitting failure, not a valid calibration result."
        )

    return TemperatureFitResult(
        scaler=scaler,
        temperature=fitted_temperature,
        initial_nll=initial_nll,
        final_nll=final_nll,
        num_fit_samples=int(logits_tensor.numel()),
        lbfgs_max_iter=max_iter,
        lbfgs_lr=lr,
        optimizer_iterations=iterations_run,
    )
