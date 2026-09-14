"""
SignalScope evaluation entrypoint.

Wires manifest loading -> generator-disjoint split -> frozen checkpoint
load -> deterministic dataset/dataloader construction -> batched inference
-> metrics -> report artifacts.

Usage:
    python -m model.evaluation.evaluate \\
        --checkpoint experiments/runs/<run_id>/checkpoints/best.pt \\
        --manifest data/manifests/genimage_dev.csv

This is a pure reporting step: the checkpoint is loaded read-only, the
threshold is fixed before any inference, and the unseen-generator `test`
split is never used to make a decision (checkpoint/threshold/hyperparameter/
architecture/calibration) - only to report a result. See
.claude/specs/05-evaluation-pipeline.md.

Responsible Team Member: Member 1 (Core ML & Model Architecture)
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from config.settings import load_data_config, load_model_config
from data.dataset_loader import DatasetRecord, load_manifest
from data.preprocessor import ImagePreprocessor
from data.splitting import REAL_GENERATOR_ID, split_manifest
from model.evaluation.config import EvaluationConfig, load_evaluation_config
from model.evaluation.metrics import SplitMetrics, compute_split_metrics, per_generator_breakdown
from model.evaluation.predict import PredictionRow, run_inference
from model.evaluation.report import (
    build_metrics_payload,
    is_dev_shard,
    maybe_write_confusion_matrix_plot,
    write_config,
    write_evaluation_report_md,
    write_metrics_json,
    write_predictions_csv,
)
from model.training.checkpoint import load_model_from_checkpoint, load_training_checkpoint
from model.training.dataset import assert_no_duplicate_images, build_val_dataset


REAL_PAIRING_NOTE = (
    "Real images for this unseen-generator result come from the validation "
    "split: data.splitting.split_manifest routes every real image into "
    "train/val and never into test (only non-real generators can be held "
    "out), so the aggregate unseen-generator set pairs test's held-out-"
    "generator fakes with val's real images. See "
    ".claude/specs/05-evaluation-pipeline.md, \"Implementation note: "
    "real-image pairing for the unseen-generator result\"."
)


class EmptyUnseenGeneratorSplitError(ValueError):
    """Raised when the split does not contain the minimum composition
    needed for an unseen-generator result: at least one AI-generated image
    and at least one generator absent from train/val in `test`, and at
    least one real image in `val` (the pool test's fakes are paired
    against - see REAL_PAIRING_NOTE).

    `data.splitting.split_manifest` does not itself guarantee any of this
    (an empty test split, or a val split with no real images, are both
    technically valid output of that function) - this is an
    evaluation-specific hard precondition.
    """


def _resolved_unseen_generators(splits: Dict[str, List[DatasetRecord]]) -> List[str]:
    """The actual, explicit set of generators held out for the test split -
    logged verbatim rather than left implicit in a fraction (see
    .claude/specs/05-evaluation-pipeline.md, "Generator-disjoint evaluation
    design"). By the time `split_manifest` returns successfully,
    `check_generator_leakage` has already guaranteed these are disjoint
    from every train/val generator."""
    return sorted({r.generator for r in splits["test"] if r.generator != REAL_GENERATOR_ID})


def _assert_unseen_generator_split_is_valid(splits: Dict[str, List[DatasetRecord]]) -> None:
    test_records = splits["test"]
    val_real_count = sum(1 for r in splits["val"] if r.generator == REAL_GENERATOR_ID)

    if not test_records:
        raise EmptyUnseenGeneratorSplitError("The test split contains no AI-generated images.")
    if val_real_count == 0:
        raise EmptyUnseenGeneratorSplitError(
            "The validation split contains no real images; there is no real-image pool to "
            "pair with test's held-out-generator fakes (see REAL_PAIRING_NOTE)."
        )
    if not _resolved_unseen_generators(splits):
        raise EmptyUnseenGeneratorSplitError(
            "The test split contains no generator absent from train/val; there is no "
            "unseen generator to evaluate against."
        )


def _make_eval_run_id() -> str:
    return f"eval-{time.strftime('%Y%m%d-%H%M%S')}"


def _resolve_run_dir(checkpoint_path: Path, output_dir_override: Optional[str], eval_run_id: str) -> Path:
    """Nests evaluation output under the checkpoint's own training run
    directory (experiments/runs/<training_run_id>/evaluation/<eval_run_id>/)
    when the checkpoint matches that expected shape; otherwise falls back
    to a standalone experiments/runs/<eval_run_id>/ directory."""
    if output_dir_override:
        return Path(output_dir_override) / eval_run_id

    checkpoints_dir = checkpoint_path.resolve().parent
    if checkpoints_dir.name == "checkpoints":
        training_run_dir = checkpoints_dir.parent
        return training_run_dir / "evaluation" / eval_run_id

    return Path("experiments") / "runs" / eval_run_id


def _infer_training_run_id(checkpoint_path: Path) -> Optional[str]:
    checkpoints_dir = checkpoint_path.resolve().parent
    if checkpoints_dir.name == "checkpoints":
        return checkpoints_dir.parent.name
    return None


def _build_eval_dataloader(dataset, eval_config: EvaluationConfig) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=eval_config.batch_size,
        shuffle=False,
        num_workers=eval_config.num_workers,
        pin_memory=eval_config.pin_memory,
        drop_last=False,
    )


def _labels_probs_generators(rows: List[PredictionRow]):
    labels = [row.true_label for row in rows]
    probabilities = [row.predicted_probability for row in rows]
    generators = [row.generator for row in rows]
    return labels, probabilities, generators


def run_evaluation(
    checkpoint_path: str,
    manifest_path: str,
    model_config_path: Optional[str] = None,
    eval_config_path: Optional[str] = None,
    output_dir_override: Optional[str] = None,
    unseen_generators_override: Optional[List[str]] = None,
    threshold_override: Optional[float] = None,
) -> Dict[str, Any]:
    """Runs one full evaluation job and returns a summary dict.

    The checkpoint is loaded read-only; the threshold is resolved once,
    before any inference, and used identically for both the val (seen-
    generator) and test (unseen-generator) reports.
    """
    checkpoint_path = Path(checkpoint_path)
    eval_config = load_evaluation_config(eval_config_path)
    data_config = load_data_config(model_config_path)
    model_config = load_model_config(model_config_path)

    threshold = threshold_override if threshold_override is not None else model_config.threshold

    split_config = data_config.split
    if unseen_generators_override:
        split_config = replace(split_config, unseen_generators=list(unseen_generators_override))

    records = load_manifest(manifest_path)
    assert_no_duplicate_images(records)

    splits = split_manifest(records, split_config)
    _assert_unseen_generator_split_is_valid(splits)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_from_checkpoint(checkpoint_path, map_location=device)
    model.to(device)
    model.eval()

    training_provenance: Dict[str, Any] = {}
    try:
        training_checkpoint = load_training_checkpoint(checkpoint_path, map_location="cpu")
        training_provenance = {
            "training_config": training_checkpoint.get("training_config"),
            "model_config": training_checkpoint.get("model_config"),
            "training_epoch": training_checkpoint.get("epoch"),
            "training_best_val_roc_auc": training_checkpoint.get("best_val_roc_auc"),
        }
    except Exception:
        # The model already loaded successfully via load_model_from_checkpoint;
        # missing/incompatible full-checkpoint metadata only degrades the
        # reported provenance, it does not block evaluation.
        training_provenance = {"warning": "training checkpoint metadata unavailable"}

    preprocessor = ImagePreprocessor.from_config(data_config.preprocessing)
    val_dataset = build_val_dataset(splits["val"], preprocessor)
    test_dataset = build_val_dataset(splits["test"], preprocessor)

    val_loader = _build_eval_dataloader(val_dataset, eval_config)
    test_loader = _build_eval_dataloader(test_dataset, eval_config)

    val_rows = run_inference(model, val_loader, device, threshold, split="val")
    test_rows = run_inference(model, test_loader, device, threshold, split="test")

    val_labels, val_probs, val_generators = _labels_probs_generators(val_rows)
    val_overall: SplitMetrics = compute_split_metrics(val_labels, val_probs, threshold)
    val_per_generator = per_generator_breakdown(val_labels, val_probs, val_generators, threshold)

    # Unseen-generator evaluation set: test's held-out-generator fakes,
    # paired with val's real images (split_manifest never routes real
    # images into test - see REAL_PAIRING_NOTE / the spec's
    # "Implementation note" section).
    val_real_rows = [row for row in val_rows if row.generator == REAL_GENERATOR_ID]
    unseen_rows = test_rows + val_real_rows
    unseen_labels, unseen_probs, unseen_generators = _labels_probs_generators(unseen_rows)
    test_overall = compute_split_metrics(unseen_labels, unseen_probs, threshold)
    test_overall.notes.append(REAL_PAIRING_NOTE)
    test_per_generator = per_generator_breakdown(unseen_labels, unseen_probs, unseen_generators, threshold)
    for metrics in test_per_generator.values():
        metrics.notes.append(REAL_PAIRING_NOTE)

    metrics_payload = build_metrics_payload(val_overall, val_per_generator, test_overall, test_per_generator)

    eval_run_id = _make_eval_run_id()
    run_dir = _resolve_run_dir(checkpoint_path, output_dir_override, eval_run_id)
    training_run_id = _infer_training_run_id(checkpoint_path)

    dataset_label = (
        "genimage_dev (development shard, not SIH benchmark)"
        if is_dev_shard(manifest_path)
        else str(manifest_path)
    )

    write_config(
        run_dir,
        {
            "eval_run_id": eval_run_id,
            "training_run_id": training_run_id,
            "checkpoint_path": str(checkpoint_path),
            "manifest_path": str(manifest_path),
            "dataset": dataset_label,
            "threshold": threshold,
            "threshold_source": "cli_override" if threshold_override is not None else "model_config",
            "split_config": {
                "seed": split_config.seed,
                "val_fraction": split_config.val_fraction,
                "unseen_generators": _resolved_unseen_generators(splits),
                "held_out_generator_fraction": split_config.held_out_generator_fraction,
            },
            "preprocessing": {
                "image_size": list(data_config.preprocessing.image_size),
                "normalization_mean": list(data_config.preprocessing.normalization.mean),
                "normalization_std": list(data_config.preprocessing.normalization.std),
            },
            "device": device.type,
            "batch_size": eval_config.batch_size,
            "num_workers": eval_config.num_workers,
            "num_val_samples": len(val_rows),
            "num_test_samples": len(test_rows),
            "num_unseen_generator_eval_samples": len(unseen_rows),
            "training_provenance": training_provenance,
            "timestamp": time.time(),
        },
    )

    write_metrics_json(metrics_payload, run_dir / "metrics.json")
    write_predictions_csv(val_rows + test_rows, run_dir / "predictions.csv")
    write_evaluation_report_md(metrics_payload, manifest_path, run_dir / "evaluation_report.md")
    maybe_write_confusion_matrix_plot(
        test_overall.confusion_matrix,
        run_dir / "confusion_matrix.png",
        enabled=eval_config.generate_plots,
    )

    return {
        "eval_run_id": eval_run_id,
        "run_dir": str(run_dir),
        "test_roc_auc": test_overall.roc_auc,
        "val_roc_auc": val_overall.roc_auc,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a frozen SignalScope checkpoint on a generator-disjoint test split."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to a Step 4 training checkpoint (.pt).")
    parser.add_argument("--manifest", required=True, help="Path to the dataset manifest CSV/JSON.")
    parser.add_argument("--model-config", default=None, help="Path to model_config.yaml (default: repo default).")
    parser.add_argument("--eval-config", default=None, help="Path to evaluation_config.yaml (default: repo default).")
    parser.add_argument("--output-dir", default=None, help="Override the inferred run output directory.")
    parser.add_argument(
        "--unseen-generators",
        default=None,
        help="Comma-separated explicit unseen-generator list, overriding config-derived selection.",
    )
    parser.add_argument("--threshold", type=float, default=None, help="Override the classification threshold.")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    unseen_generators_override = (
        [g.strip() for g in args.unseen_generators.split(",") if g.strip()] if args.unseen_generators else None
    )

    summary = run_evaluation(
        checkpoint_path=args.checkpoint,
        manifest_path=args.manifest,
        model_config_path=args.model_config,
        eval_config_path=args.eval_config,
        output_dir_override=args.output_dir,
        unseen_generators_override=unseen_generators_override,
        threshold_override=args.threshold,
    )
    print(f"eval_run_id  = {summary['eval_run_id']}")
    print(f"run_dir      = {summary['run_dir']}")
    print(f"val_roc_auc  = {summary['val_roc_auc']}")
    print(f"test_roc_auc = {summary['test_roc_auc']} (unseen-generator, primary result)")


if __name__ == "__main__":
    main()
