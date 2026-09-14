"""
Pure batched inference for the SignalScope evaluation pipeline.

`run_inference` must not load checkpoints and must not construct datasets
or dataloaders - those responsibilities remain in
model/evaluation/evaluate.py (checkpoint loading via
model.training.checkpoint.load_model_from_checkpoint, dataset/dataloader
construction via model.training.dataset.build_val_dataset). This module
receives an already-loaded, already-`.eval()`'d model and an already-built
dataloader, and is responsible only for the forward pass, thresholding,
and row assembly. See .claude/specs/05-evaluation-pipeline.md
("Prediction pipeline", "Implementation sequence" item 4).

Responsible Team Member: Member 1 (Core ML & Model Architecture)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
from torch import nn
from torch.utils.data import DataLoader

REAL_LABEL = 0
AI_GENERATED_LABEL = 1


@dataclass(frozen=True)
class PredictionRow:
    """One evaluated image's prediction, with enough metadata to drive
    both split-level and per-generator metrics without re-running
    inference."""

    image_path: str
    true_label: int
    predicted_probability: float
    predicted_label: int
    generator: str
    split: str

    def to_dict(self) -> dict:
        return {
            "image_path": self.image_path,
            "true_label": self.true_label,
            "predicted_probability": self.predicted_probability,
            "predicted_label": self.predicted_label,
            "generator": self.generator,
            "split": self.split,
        }


@torch.inference_mode()
def run_inference(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float,
    split: str = "test",
) -> List[PredictionRow]:
    """Runs one full batched inference pass and returns one PredictionRow
    per evaluated image.

    `predicted_probability` is `sigmoid(raw_logit)` = P(ai_generated), with
    no calibration applied. `predicted_label` is thresholded at `threshold`
    (>= threshold -> AI_GENERATED_LABEL). `split` is stamped onto every row
    as given - the dataloader's underlying dataset doesn't know which
    logical split ("val"/"test") it represents once built.

    Does not call `model.eval()` and does not move `model` to `device` -
    the caller (evaluate.py) owns that, since this function must not
    assume how the model was constructed or loaded.
    """
    rows: List[PredictionRow] = []

    for images, labels, meta in dataloader:
        images = images.to(device)
        logits = model(images)
        probabilities = torch.sigmoid(logits).squeeze(1).cpu().tolist()
        labels_list = labels.tolist()
        image_paths = meta["image_path"]
        generators = meta["generator"]

        for image_path, true_label, probability, generator in zip(
            image_paths, labels_list, probabilities, generators
        ):
            predicted_label = AI_GENERATED_LABEL if probability >= threshold else REAL_LABEL
            rows.append(
                PredictionRow(
                    image_path=image_path,
                    true_label=int(true_label),
                    predicted_probability=float(probability),
                    predicted_label=predicted_label,
                    generator=generator,
                    split=split,
                )
            )

    return rows
