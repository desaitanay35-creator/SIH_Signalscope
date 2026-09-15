"""
SignalScope training entrypoint.

Wires manifest loading -> generator-disjoint split -> datasets/dataloaders
-> EfficientNet-B4 -> loss/optimizer/scheduler -> train/validate loop ->
checkpointing -> logging.

Usage:
    python -m model.training.train
    python -m model.training.train --config config/training_config.yaml
    python -m model.training.train --smoke-test

The unseen-generator `test` split produced by data.splitting.split_manifest
is loaded (as part of the same split_manifest call) but never constructed
into a Dataset/DataLoader and never referenced again here - it is reserved
for a later, separate evaluation step. See
.claude/specs/04-training-pipeline.md.

Responsible Team Member: Member 1 (Core ML & Model Architecture)
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from config.settings import load_data_config, load_model_config
from data.dataset_loader import DatasetRecord, load_manifest
from data.preprocessor import ImagePreprocessor
from data.splitting import split_manifest
from model.architectures.efficientnet_b4 import EfficientNetB4Baseline
from model.training.checkpoint import save_checkpoint
from model.training.config import TrainingConfig, load_training_config
from model.training.dataset import assert_no_duplicate_images, build_train_dataset, build_val_dataset
from model.training.dual_branch_dataset import build_dual_branch_train_dataset, build_dual_branch_val_dataset
from model.training.engine import train_one_epoch, validate
from model.training.logging_utils import RunLogger
from model.training.seed import set_seed

DEV_SHARD_MARKER = "genimage_dev"
DEV_SHARD_LABEL = "genimage_dev (development shard, not SIH benchmark)"

DEFAULT_FREQUENCY_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "frequency_config.yaml"
VALID_MODEL_TYPES = ("rgb_only", "frequency_only", "rgb_frequency_fusion")


def _make_run_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _load_model_type(frequency_config_path: Optional[str]) -> str:
    """Reads `model_type` from a frequency config YAML (default:
    config/frequency_config.yaml). Falls back to "rgb_only" - reproducing
    Step 4's exact behavior - when the file is absent or the key is unset
    (this file's own shipped default value is also "rgb_only", so omitting
    `--frequency-config` entirely changes nothing about existing runs).
    Raises clearly for an unrecognized value rather than silently
    defaulting or failing deeper in model construction. See
    .claude/specs/06-frequency-fusion.md ("Configuration").
    """
    path = Path(frequency_config_path) if frequency_config_path else DEFAULT_FREQUENCY_CONFIG_PATH
    model_type = "rgb_only"
    if path.is_file():
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        model_type = raw.get("model_type", "rgb_only")
    if model_type not in VALID_MODEL_TYPES:
        raise ValueError(f"Unsupported model_type {model_type!r}; expected one of {VALID_MODEL_TYPES}.")
    return model_type


def _build_model(model_type: str, model_config) -> nn.Module:
    """Builds the model for the given `model_type`. The two new
    architectures are imported lazily so a plain "rgb_only" run - the
    default and most common case - has no import-time dependency on the
    Step 6 architecture files."""
    if model_type == "rgb_only":
        return EfficientNetB4Baseline(
            pretrained=model_config.pretrained,
            num_output_logits=model_config.num_output_logits,
        )
    if model_type == "frequency_only":
        from model.architectures.frequency_branch import FrequencyOnlyModel

        return FrequencyOnlyModel(num_output_logits=model_config.num_output_logits)
    if model_type == "rgb_frequency_fusion":
        from model.architectures.fusion_model import RGBFrequencyFusionModel

        return RGBFrequencyFusionModel(
            pretrained=model_config.pretrained,
            num_output_logits=model_config.num_output_logits,
        )
    raise ValueError(f"Unsupported model_type {model_type!r}; expected one of {VALID_MODEL_TYPES}.")


def _build_datasets(
    model_type: str,
    train_records: Sequence[DatasetRecord],
    val_records: Sequence[DatasetRecord],
    preprocessor,
):
    """Builds the (train_dataset, val_dataset) pair matching `model_type`'s
    expected input shape - a single RGB tensor for "rgb_only", or a
    {"rgb": ..., "frequency": ...} dict for the two dual-branch model
    types. See .claude/specs/06-frequency-fusion.md ("Input construction
    for training/evaluation (dataset wiring)")."""
    if model_type == "rgb_only":
        return (
            build_train_dataset(train_records, preprocessor),
            build_val_dataset(val_records, preprocessor),
        )
    return (
        build_dual_branch_train_dataset(train_records, preprocessor),
        build_dual_branch_val_dataset(val_records, preprocessor),
    )


def _set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Freezes/unfreezes the RGB backbone's convolutional features, for
    architectures that have one ("rgb_only" -> `model.backbone`,
    "rgb_frequency_fusion" -> `model.rgb_branch.backbone`). No-op for
    "frequency_only", which has no RGB backbone - `freeze_backbone_epochs`
    has no meaning there."""
    if hasattr(model, "backbone"):
        target = model.backbone
    elif hasattr(model, "rgb_branch"):
        target = model.rgb_branch.backbone
    else:
        return
    for param in target.features.parameters():
        param.requires_grad = trainable


def _build_optimizer(model: nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
    if config.optimizer.lower() != "adamw":
        raise ValueError(f"Unsupported optimizer {config.optimizer!r}; only 'adamw' is implemented.")
    return torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)


def _build_scheduler(optimizer: torch.optim.Optimizer, config: TrainingConfig):
    if config.scheduler.lower() != "cosine":
        raise ValueError(f"Unsupported scheduler {config.scheduler!r}; only 'cosine' is implemented.")
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(config.epochs, 1))


def _build_dataloaders(train_dataset, val_dataset, config: TrainingConfig):
    train_generator = torch.Generator().manual_seed(config.seed)
    use_persistent = config.persistent_workers and config.num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=config.drop_last_train,
        persistent_workers=use_persistent,
        generator=train_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False,
        persistent_workers=use_persistent,
    )
    return train_loader, val_loader


def run_training(
    training_config: TrainingConfig,
    model_config_path: Optional[str] = None,
    smoke_test: bool = False,
    run_id: Optional[str] = None,
    frequency_config_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Runs one full training job and returns a summary dict.

    `smoke_test=True` truncates train/val to `max_train_samples`/
    `max_val_samples` and forces a tiny, fast CPU correctness check - not a
    benchmark run. See .claude/specs/04-training-pipeline.md.

    `frequency_config_path` selects which model this run trains (Step 6's
    ablation: "rgb_only" - the default, reproducing this function's
    original, Step-4-only behavior exactly - "frequency_only", or
    "rgb_frequency_fusion"). See .claude/specs/06-frequency-fusion.md.
    """
    set_seed(training_config.seed)

    model_type = _load_model_type(frequency_config_path)

    data_config = load_data_config(model_config_path)
    model_config = load_model_config(model_config_path)

    records = load_manifest(data_config.manifest_path)
    assert_no_duplicate_images(records)

    splits = split_manifest(records, data_config.split)
    train_records = splits["train"]
    val_records = splits["val"]
    # splits["test"] is the unseen-generator evaluation split - intentionally
    # never read past this point (see module docstring).

    if smoke_test:
        # split_manifest returns records sorted by image_path, which can
        # cluster one label together (e.g. real vs. generator filename
        # patterns); shuffle deterministically before truncating so a tiny
        # smoke-test slice still has a realistic chance of containing both
        # classes (needed to exercise the ROC-AUC/best-checkpoint path).
        rng = random.Random(training_config.seed)
        train_records = list(train_records)
        val_records = list(val_records)
        rng.shuffle(train_records)
        rng.shuffle(val_records)
        if training_config.max_train_samples is not None:
            train_records = train_records[: training_config.max_train_samples]
        if training_config.max_val_samples is not None:
            val_records = val_records[: training_config.max_val_samples]

    preprocessor = ImagePreprocessor.from_config(data_config.preprocessing)
    train_dataset, val_dataset = _build_datasets(model_type, train_records, val_records, preprocessor)
    train_loader, val_loader = _build_dataloaders(train_dataset, val_dataset, training_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(model_type, model_config)
    model.to(device)

    if training_config.freeze_backbone_epochs > 0:
        _set_backbone_trainable(model, False)

    pos_weight = None
    if training_config.pos_weight is not None:
        pos_weight = torch.tensor([training_config.pos_weight], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = _build_optimizer(model, training_config)
    scheduler = _build_scheduler(optimizer, training_config)

    use_amp = device.type == "cuda" and training_config.mixed_precision
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    run_id = run_id or _make_run_id()
    run_dir = Path(training_config.checkpoint_dir) / run_id
    checkpoint_dir = run_dir / "checkpoints"
    logger = RunLogger(run_dir)

    dataset_label = DEV_SHARD_LABEL if DEV_SHARD_MARKER in str(data_config.manifest_path) else str(data_config.manifest_path)
    logger.write_config(
        {
            "run_id": run_id,
            "dataset": dataset_label,
            "manifest_path": str(data_config.manifest_path),
            "smoke_test": smoke_test,
            "device": device.type,
            "num_train_samples": len(train_records),
            "num_val_samples": len(val_records),
            "model_type": model_type,
            # model.get_spec() (polymorphic - see model/training/checkpoint.py's
            # architecture registry) rather than a hardcoded EfficientNet-B4
            # dict, so this matches what's actually built for any model_type.
            "model_config": model.get_spec(),
            "training_config": training_config.to_dict(),
        }
    )

    best_val_roc_auc: Optional[float] = None
    val_metrics_history = []

    for epoch in range(1, training_config.epochs + 1):
        if training_config.freeze_backbone_epochs > 0 and epoch == training_config.freeze_backbone_epochs + 1:
            _set_backbone_trainable(model, True)

        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, scaler)
        val_metrics = validate(model, val_loader, loss_fn, device)
        scheduler.step()

        val_metrics_history.append(val_metrics.to_log_dict())
        logger.log_epoch(epoch, train_loss, val_metrics.to_log_dict(), optimizer.param_groups[0]["lr"])

        save_checkpoint(
            checkpoint_dir / "last.pt",
            model,
            optimizer,
            scheduler,
            epoch,
            best_val_roc_auc,
            val_metrics_history,
            training_config,
            training_config.seed,
        )

        if val_metrics.roc_auc is not None and (best_val_roc_auc is None or val_metrics.roc_auc > best_val_roc_auc):
            best_val_roc_auc = val_metrics.roc_auc
            save_checkpoint(
                checkpoint_dir / "best.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                best_val_roc_auc,
                val_metrics_history,
                training_config,
                training_config.seed,
            )

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "epochs_run": training_config.epochs,
        "best_val_roc_auc": best_val_roc_auc,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the SignalScope EfficientNet-B4 baseline classifier."
    )
    parser.add_argument("--config", default=None, help="Path to a training_config.yaml (default: repo default).")
    parser.add_argument("--model-config", default=None, help="Path to model_config.yaml (default: repo default).")
    parser.add_argument(
        "--frequency-config",
        default=None,
        help=(
            "Path to a frequency_config.yaml selecting model_type "
            "('rgb_only' | 'frequency_only' | 'rgb_frequency_fusion'); "
            "default: repo default, whose shipped default is 'rgb_only'."
        ),
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a tiny, fast CPU correctness check (forward/backward/checkpointing) - not a benchmark run.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    training_config = load_training_config(args.config)

    if args.smoke_test:
        training_config.epochs = min(training_config.epochs, 2)
        training_config.batch_size = min(training_config.batch_size, 4)
        training_config.num_workers = 0
        if training_config.max_train_samples is None:
            training_config.max_train_samples = 32
        if training_config.max_val_samples is None:
            training_config.max_val_samples = 16

    summary = run_training(
        training_config,
        model_config_path=args.model_config,
        smoke_test=args.smoke_test,
        frequency_config_path=args.frequency_config,
    )
    print(f"run_id             = {summary['run_id']}")
    print(f"run_dir            = {summary['run_dir']}")
    print(f"epochs_run         = {summary['epochs_run']}")
    print(f"best_val_roc_auc   = {summary['best_val_roc_auc']}")


if __name__ == "__main__":
    main()
