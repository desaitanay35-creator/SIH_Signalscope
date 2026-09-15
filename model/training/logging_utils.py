"""
Per-run training logs for the SignalScope training pipeline.

Writes one `config.json` (the full resolved run configuration) and one
`metrics.jsonl` (one JSON line appended per epoch) per run directory, under
`experiments/runs/<run_id>/` by default. Never overwrites another run's
directory - each run gets its own `run_id`.

Responsible Team Member: Member 6 (MLOps & Config)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Union


class RunLogger:
    """Writes one run's resolved config once and appends one JSON line per
    epoch to that run's metrics log."""

    def __init__(self, run_dir: Union[str, Path]) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.jsonl"

    def write_config(self, config: Dict[str, Any]) -> None:
        config_path = self.run_dir / "config.json"
        with open(config_path, "w", encoding="utf-8") as fh:
            json.dump(config, fh, indent=2, default=str)

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_metrics: Dict[str, Any],
        learning_rate: float,
    ) -> None:
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics.get("loss"),
            "val_accuracy": val_metrics.get("accuracy"),
            "val_roc_auc": val_metrics.get("roc_auc"),
            "learning_rate": learning_rate,
            "timestamp": time.time(),
        }
        with open(self.metrics_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
