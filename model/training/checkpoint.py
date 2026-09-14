"""
Training checkpoint save/load for the SignalScope training pipeline.

Two responsibilities are kept separate:
  - `save_checkpoint` / `load_training_checkpoint` / `restore_training_state`:
    the full training state needed to resume or exactly reproduce a run
    (model, optimizer, scheduler, RNG state, validation history). This is
    distinct from `EfficientNetB4Baseline.save_checkpoint`, which
    intentionally saves architecture-only weights for inference.
  - `load_model_from_checkpoint`: a convenience for downstream
    evaluation/inference code that only needs the trained weights and
    should not need to know the full training-checkpoint schema.

Responsible Team Member: Member 6 (MLOps & Config)
"""

from __future__ import annotations

import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from torch import nn

from model.architectures.efficientnet_b4 import EfficientNetB4Baseline, get_model_spec


def _to_plain_dict(obj: Any) -> Any:
    return asdict(obj) if is_dataclass(obj) else obj


def save_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    best_val_roc_auc: Optional[float],
    val_metrics_history: List[Dict[str, Any]],
    training_config: Any,
    seed: int,
) -> None:
    """Writes a full training checkpoint (model + optimizer + scheduler +
    RNG state + history) - enough to resume training or reproduce
    inference, never test-set metrics (none are computed during training)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "best_val_roc_auc": best_val_roc_auc,
        "val_metrics_history": val_metrics_history,
        "model_config": get_model_spec(pretrained=getattr(model, "pretrained", False)),
        "training_config": _to_plain_dict(training_config),
        "seed": seed,
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }
    torch.save(payload, path)


def load_training_checkpoint(
    path: Union[str, Path],
    map_location: Union[str, torch.device] = "cpu",
) -> Dict[str, Any]:
    """Loads the raw checkpoint dict, for resuming training."""
    return torch.load(Path(path), map_location=map_location, weights_only=False)


def restore_training_state(
    checkpoint: Dict[str, Any],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    restore_rng: bool = True,
) -> None:
    """Applies a loaded checkpoint's state onto live model/optimizer/scheduler
    objects, so a resumed run continues deterministically."""
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if restore_rng:
        torch.set_rng_state(checkpoint["torch_rng_state"])
        np.random.set_state(checkpoint["numpy_rng_state"])
        random.setstate(checkpoint["python_rng_state"])


def load_model_from_checkpoint(
    path: Union[str, Path],
    map_location: Union[str, torch.device] = "cpu",
) -> EfficientNetB4Baseline:
    """Builds the baseline architecture and loads only its trained weights.

    For downstream evaluation/inference code - does not require knowledge
    of the full training-checkpoint schema (optimizer/scheduler/RNG state).
    """
    checkpoint = torch.load(Path(path), map_location=map_location, weights_only=False)
    model = EfficientNetB4Baseline(pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model
