"""
Evaluation-run configuration for the SignalScope evaluation pipeline.

Kept separate from config/model_config.yaml (architecture/data/split,
single source of truth) and from model/training/config.py (training
hyperparameters) - this module owns only evaluation-run settings (batch
size, dataloader behavior, output directory, optional plotting). See
config/evaluation_config.yaml for the default values and
.claude/specs/05-evaluation-pipeline.md for the full rationale.

Responsible Team Member: Member 6 (MLOps & Config)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

DEFAULT_EVALUATION_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "evaluation_config.yaml"


@dataclass
class EvaluationConfig:
    """Run-specific evaluation settings, loaded from
    config/evaluation_config.yaml (`evaluation:` section)."""

    batch_size: int = 16
    num_workers: int = 0
    pin_memory: bool = False
    output_dir: str = "experiments/runs"
    generate_plots: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_evaluation_config(config_path: Optional[str] = None) -> EvaluationConfig:
    """Loads the `evaluation:` section of a YAML evaluation configuration.

    Falls back to `EvaluationConfig()` defaults for any missing keys, so a
    partial or absent config file still produces a usable, fully-typed
    configuration (matches the pattern used by model/training/config.py).
    """
    path = Path(config_path) if config_path else DEFAULT_EVALUATION_CONFIG_PATH
    if not path.is_file():
        return EvaluationConfig()

    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    evaluation_raw = raw.get("evaluation", {}) or {}
    known_fields = {f.name for f in fields(EvaluationConfig)}
    overrides = {k: v for k, v in evaluation_raw.items() if k in known_fields}

    defaults = EvaluationConfig()
    return EvaluationConfig(**{**defaults.to_dict(), **overrides})
