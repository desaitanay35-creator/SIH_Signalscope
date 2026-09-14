"""
Training-run configuration for the SignalScope training pipeline.

Kept separate from config/model_config.yaml (architecture/data, single
source of truth) - this module owns only run-specific hyperparameters
(optimizer, scheduler, batch size, seed, dataloader behavior, checkpoint
paths). See config/training_config.yaml for the default values and
.claude/specs/04-training-pipeline.md for the full rationale.

Responsible Team Member: Member 6 (MLOps & Config)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

DEFAULT_TRAINING_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "training_config.yaml"


@dataclass
class TrainingConfig:
    """Run-specific training hyperparameters, loaded from
    config/training_config.yaml (`training:` section)."""

    seed: int = 42
    epochs: int = 10
    batch_size: int = 16

    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    drop_last_train: bool = True

    optimizer: str = "adamw"
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    scheduler: str = "cosine"

    freeze_backbone_epochs: int = 0
    pos_weight: Optional[float] = None

    mixed_precision: bool = True

    checkpoint_dir: str = "experiments/runs"
    log_dir: str = "experiments/runs"

    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_training_config(config_path: Optional[str] = None) -> TrainingConfig:
    """Loads the `training:` section of a YAML training configuration.

    Falls back to `TrainingConfig()` defaults for any missing keys, so a
    partial or absent config file still produces a usable, fully-typed
    configuration (matches the pattern used by config/settings.py).
    """
    path = Path(config_path) if config_path else DEFAULT_TRAINING_CONFIG_PATH
    if not path.is_file():
        return TrainingConfig()

    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    training_raw = raw.get("training", {}) or {}
    known_fields = {f.name for f in fields(TrainingConfig)}
    overrides = {k: v for k, v in training_raw.items() if k in known_fields}

    defaults = TrainingConfig()
    return TrainingConfig(**{**defaults.to_dict(), **overrides})
