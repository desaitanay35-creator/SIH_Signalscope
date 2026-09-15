"""
Pure batched raw-logit inference for SignalScope calibration (Step 8A).

Deliberately duplicated from model/evaluation/predict.py::run_inference
rather than importing/modifying it - the existing function only exposes
`sigmoid(logit)` probabilities (never the raw logit itself), and
temperature-scaling calibration must operate on raw logits, never on
already-sigmoided probabilities (see
.claude/specs/08a-probability-calibration.md, "Model / Architecture
Changes"). This follows the repository's own established precedent
(model/evaluation/predict.py's own docstring: "Duplicated (not imported)
from model/training/engine.py's identical helper - a two-line utility
does not warrant a cross-package dependency") - it keeps model/evaluation/
completely untouched by this step.

Responsible Team Member: Step 8A (Probability Calibration)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Union

import torch
from torch import nn
from torch.utils.data import DataLoader


def _move_batch_to_device(
    batch: Union[torch.Tensor, Dict[str, torch.Tensor]], device: torch.device
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """Moves a batch to `device`, whether it is a plain tensor (RGB-only
    models) or a dict of tensors (dual-branch models, e.g.
    {"rgb": ..., "frequency": ...}). Duplicated (not imported) from
    model/evaluation/predict.py's identical helper, for the same reason
    that module states for its own duplication from
    model/training/engine.py."""
    if isinstance(batch, dict):
        return {key: value.to(device) for key, value in batch.items()}
    return batch.to(device)


@dataclass(frozen=True)
class LogitPredictionRow:
    """One evaluated image's raw (pre-sigmoid) logit, with enough
    metadata to drive both temperature fitting and split-level/
    per-generator calibration metrics without re-running inference."""

    image_path: str
    true_label: int
    raw_logit: float
    generator: str
    split: str

    def to_dict(self) -> dict:
        return {
            "image_path": self.image_path,
            "true_label": self.true_label,
            "raw_logit": self.raw_logit,
            "generator": self.generator,
            "split": self.split,
        }


@torch.inference_mode()
def run_logit_inference(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    split: str = "test",
) -> List[LogitPredictionRow]:
    """Runs one full batched inference pass and returns one
    LogitPredictionRow per evaluated image, carrying the model's raw
    (pre-sigmoid) logit - never a probability, never a thresholded
    decision. Downstream code (model/calibration/calibrate.py) is
    responsible for applying sigmoid/temperature/threshold.

    Does not call `model.eval()` and does not move `model` to `device` -
    the caller owns that, mirroring model/evaluation/predict.py's
    run_inference contract exactly.
    """
    rows: List[LogitPredictionRow] = []

    for images, labels, meta in dataloader:
        images = _move_batch_to_device(images, device)
        logits = model(images).squeeze(1).cpu().tolist()
        labels_list = labels.tolist()
        image_paths = meta["image_path"]
        generators = meta["generator"]

        for image_path, true_label, raw_logit, generator in zip(image_paths, labels_list, logits, generators):
            rows.append(
                LogitPredictionRow(
                    image_path=image_path,
                    true_label=int(true_label),
                    raw_logit=float(raw_logit),
                    generator=generator,
                    split=split,
                )
            )

    return rows
