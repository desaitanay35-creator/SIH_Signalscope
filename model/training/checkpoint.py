"""
Training checkpoint save/load for the SignalScope training pipeline.

Two responsibilities are kept separate:
  - `save_checkpoint` / `load_training_checkpoint` / `restore_training_state`:
    the full training state needed to resume or exactly reproduce a run
    (model, optimizer, scheduler, RNG state, validation history). This is
    distinct from each architecture's own `save_checkpoint` instance
    method (e.g. `EfficientNetB4Baseline.save_checkpoint`), which
    intentionally saves architecture-only weights for inference.
  - `load_model_from_checkpoint`: a convenience for downstream
    evaluation/inference code that only needs the trained weights and
    should not need to know the full training-checkpoint schema.

Supports multiple architectures (Step 6: EfficientNet-B4 RGB baseline,
frequency-only, RGB+frequency fusion) via a small, backward-compatible
architecture registry - see `load_model_from_checkpoint` and
.claude/specs/06-frequency-fusion.md ("Model-loading interface").

Responsible Team Member: Member 6 (MLOps & Config)
"""

from __future__ import annotations

import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from torch import nn

from model.architectures.efficientnet_b4 import EfficientNetB4Baseline

# The architecture id every checkpoint saved before this registry existed
# implicitly has (get_model_spec() has always set `architecture:
# "efficientnet_b4"` in model_config) - existing Step 4/5 checkpoints load
# exactly as before, with zero behavior change.
DEFAULT_ARCHITECTURE_ID = "efficientnet_b4"


class UnknownArchitectureError(ValueError):
    """Raised when a checkpoint's recorded architecture id has no known
    constructor registered - e.g. a checkpoint from a newer/unrecognized
    model type, or a corrupted `model_config` field. Never silently
    defaults to a different architecture or fails with an unrelated error
    deeper in model construction."""


def _to_plain_dict(obj: Any) -> Any:
    return asdict(obj) if is_dataclass(obj) else obj


def _build_efficientnet_b4() -> nn.Module:
    return EfficientNetB4Baseline(pretrained=False)


def _build_frequency_branch() -> nn.Module:
    # Imported lazily so this module has no import-time dependency on the
    # Step 6 architecture files for the (still default, still most common)
    # RGB-only training/evaluation path.
    from model.architectures.frequency_branch import FrequencyOnlyModel

    return FrequencyOnlyModel()


def _build_rgb_frequency_fusion() -> nn.Module:
    from model.architectures.fusion_model import RGBFrequencyFusionModel

    return RGBFrequencyFusionModel(pretrained=False)


# Architecture id (from a checkpoint's own model_config["architecture"],
# itself produced by that architecture's own .get_spec()) -> a zero-arg
# constructor building that architecture with pretrained=False (a
# checkpoint's weights fully determine the loaded parameters; pretrained
# only matters for the initial construction before the checkpoint is
# applied).
_ARCHITECTURE_BUILDERS: Dict[str, Callable[[], nn.Module]] = {
    "efficientnet_b4": _build_efficientnet_b4,
    "frequency_branch": _build_frequency_branch,
    "rgb_frequency_fusion": _build_rgb_frequency_fusion,
}


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
    inference, never test-set metrics (none are computed during training).

    `model.get_spec()` is called polymorphically (duck-typed): every
    architecture class (`EfficientNetB4Baseline`, `FrequencyOnlyModel`,
    `RGBFrequencyFusionModel`) implements its own `.get_spec()` returning
    at least `{"architecture": <id>, ...}` - this function has no
    architecture-specific knowledge itself. See
    .claude/specs/06-frequency-fusion.md ("Model-loading interface").
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "best_val_roc_auc": best_val_roc_auc,
        "val_metrics_history": val_metrics_history,
        "model_config": model.get_spec(),
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


def _resolve_architecture_id(model_config: Optional[Dict[str, Any]]) -> str:
    if not model_config:
        return DEFAULT_ARCHITECTURE_ID
    return model_config.get("architecture", DEFAULT_ARCHITECTURE_ID)


def load_model_from_checkpoint(
    path: Union[str, Path],
    map_location: Union[str, torch.device] = "cpu",
) -> nn.Module:
    """Builds the correct architecture - dispatched from the checkpoint's
    own recorded `model_config["architecture"]` - and loads only its
    trained weights.

    Backward compatible: a checkpoint with no `architecture` field (every
    checkpoint saved before this registry existed) resolves to
    `DEFAULT_ARCHITECTURE_ID` ("efficientnet_b4"), so existing Step 4/5
    checkpoints load exactly as they always have, with zero behavior
    change. An architecture id that IS present but not recognized raises
    `UnknownArchitectureError` clearly, rather than defaulting silently or
    failing with an unrelated error deeper in model construction. See
    .claude/specs/06-frequency-fusion.md ("Model-loading interface").
    """
    checkpoint = torch.load(Path(path), map_location=map_location, weights_only=False)
    architecture_id = _resolve_architecture_id(checkpoint.get("model_config"))

    builder = _ARCHITECTURE_BUILDERS.get(architecture_id)
    if builder is None:
        raise UnknownArchitectureError(
            f"Checkpoint {str(path)!r} declares architecture {architecture_id!r}, which has no "
            f"registered constructor. Known architectures: {sorted(_ARCHITECTURE_BUILDERS)}."
        )

    model = builder()
    model.load_state_dict(checkpoint["model_state_dict"])
    return model
