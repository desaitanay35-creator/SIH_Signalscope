"""
Training and validation loops for the SignalScope EfficientNet-B4
classifier.

Pure functions only - no argparse, no config loading, no CLI. See
model/training/train.py for the entrypoint that wires these together.

Responsible Team Member: Member 1 (Core ML & Model Architecture)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


def _roc_auc_score(labels: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    """Computes binary ROC-AUC via the Mann-Whitney U / rank-sum formula.

    Self-contained (numpy only) rather than a scikit-learn dependency:
    scikit-learn/scipy's compiled extensions fail to import on this
    development machine (a Windows Application Control policy blocks the
    scipy DLL) - see .claude/specs/04-training-pipeline.md. Returns None
    when only one class is present (ROC-AUC is undefined in that case).
    """
    labels_arr = np.asarray(labels)
    scores_arr = np.asarray(scores, dtype=np.float64)
    n_pos = int(np.sum(labels_arr == 1))
    n_neg = int(np.sum(labels_arr == 0))
    if n_pos == 0 or n_neg == 0:
        return None

    order = np.argsort(scores_arr, kind="mergesort")
    sorted_scores = scores_arr[order]
    ranks = np.empty(len(scores_arr), dtype=np.float64)

    i = 0
    rank = 1
    while i < len(sorted_scores):
        j = i
        while j < len(sorted_scores) - 1 and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg_rank = (rank + rank + (j - i)) / 2.0
        ranks[order[i : j + 1]] = avg_rank
        rank += j - i + 1
        i = j + 1

    sum_ranks_pos = ranks[labels_arr == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


# Public name for reuse outside this module (e.g. model/evaluation/metrics.py)
# - same function, `_roc_auc_score` kept as an alias so existing imports
# (tests/test_training.py) are unaffected. See
# .claude/specs/05-evaluation-pipeline.md ("Files to change").
roc_auc_score = _roc_auc_score


@dataclass
class EpochMetrics:
    """Validation-epoch results: aggregate metrics plus the raw
    (probability, label, generator) triples needed by a later, separate
    evaluation step (macro-F1, confusion matrix, FPR, unseen-generator
    ROC-AUC) without retraining."""

    loss: float
    accuracy: float
    roc_auc: Optional[float]
    num_samples: int
    probabilities: List[float] = field(default_factory=list)
    labels: List[int] = field(default_factory=list)
    generators: List[str] = field(default_factory=list)

    def to_log_dict(self) -> Dict[str, Any]:
        return {
            "loss": self.loss,
            "accuracy": self.accuracy,
            "roc_auc": self.roc_auc,
            "num_samples": self.num_samples,
        }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    scaler: "torch.amp.GradScaler",
) -> float:
    """Runs one training epoch (forward, backward, optimizer step) and
    returns the sample-weighted mean training loss."""
    model.train()
    total_loss = 0.0
    total_samples = 0
    use_amp = scaler.is_enabled()

    for images, labels, _meta in loader:
        images = images.to(device)
        labels = labels.to(device).float().unsqueeze(1)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = loss_fn(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
) -> EpochMetrics:
    """Runs one validation pass. Never updates model parameters and must
    never be called on the unseen-generator `test` split."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    correct = 0
    probabilities: List[float] = []
    labels_out: List[int] = []
    generators_out: List[str] = []

    for images, labels, meta in loader:
        images = images.to(device)
        labels_device = labels.to(device).float().unsqueeze(1)

        logits = model(images)
        loss = loss_fn(logits, labels_device)

        probs = torch.sigmoid(logits).squeeze(1)
        preds = (probs >= threshold).long().cpu()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        correct += (preds == labels).sum().item()

        probabilities.extend(probs.cpu().tolist())
        labels_out.extend(labels.tolist())
        generators_out.extend(meta["generator"])

    roc_auc = _roc_auc_score(labels_out, probabilities)

    return EpochMetrics(
        loss=total_loss / max(total_samples, 1),
        accuracy=correct / max(total_samples, 1),
        roc_auc=roc_auc,
        num_samples=total_samples,
        probabilities=probabilities,
        labels=labels_out,
        generators=generators_out,
    )
