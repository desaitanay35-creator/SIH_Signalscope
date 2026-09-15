"""
SignalScope Step 8A calibration entrypoint.

Wires manifest loading -> generator-disjoint split (reusing
data.splitting.split_manifest unmodified) -> frozen checkpoint load
(read-only) -> raw-logit collection on val/test -> temperature fit on
val-only logits -> before/after metric computation on both val and the
unseen-generator evaluation set -> reproducible artifact bundle.

This is a strictly post-training, checkpoint-preserving operation: the
checkpoint is loaded read-only and its file is checksummed before and
after this module runs to prove it was never modified. No detector
weights are retrained. Temperature is fit ONLY on the validation split's
raw logits - the unseen-generator split is read only for the before/after
evaluation pass, strictly after the temperature is already fixed.

Deliberately does not import from model/evaluation/evaluate.py or
model/evaluation/predict.py, and does not modify either - the small
amount of protocol logic those files own (real-image pairing, the
unseen-generator-split validity check, the dev-shard banner) is
re-derived here instead, following this repository's own established
precedent of duplicating small helpers across modules rather than adding
cross-package coupling. See
.claude/specs/08a-probability-calibration.md ("Rules for implementation",
"Experiment protocol").

Usage:
    python -m model.calibration.calibrate \\
        --checkpoint experiments/runs/<run_id>/checkpoints/best.pt \\
        --manifest data/manifests/genimage_dev.csv \\
        --model-config config/ablation/model_config.yaml

Responsible Team Member: Step 8A (Probability Calibration)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml
from torch.utils.data import DataLoader

from config.settings import load_data_config, load_model_config
from data.dataset_loader import DatasetRecord, load_manifest
from data.splitting import REAL_GENERATOR_ID, split_manifest
from model.calibration.calibration_metrics import brier_score, expected_calibration_error
from model.calibration.logit_inference import LogitPredictionRow, run_logit_inference
from model.calibration.temperature_scaling import TemperatureFitResult, apply_temperature, fit_temperature
from model.evaluation.metrics import SplitMetrics, compute_split_metrics, roc_auc_score, threshold_probabilities
from model.training.checkpoint import (
    DEFAULT_ARCHITECTURE_ID,
    UnknownArchitectureError,
    load_model_from_checkpoint,
    load_training_checkpoint,
)
from model.training.dataset import assert_no_duplicate_images, build_val_dataset
from model.training.dual_branch_dataset import build_dual_branch_val_dataset
from model.training.logging_utils import RunLogger

# Architecture ids whose dataset must supply both RGB and frequency
# tensors - duplicated from model/evaluation/evaluate.py's identical
# tuple (not imported) to keep model/evaluation/ untouched by this step.
_DUAL_BRANCH_ARCHITECTURES = ("frequency_branch", "rgb_frequency_fusion")

# Duplicated verbatim from model/evaluation/evaluate.py::REAL_PAIRING_NOTE
# - see .claude/specs/08a-probability-calibration.md, Risk R2.
REAL_PAIRING_NOTE = (
    "Real images for this unseen-generator result come from the validation "
    "split: data.splitting.split_manifest routes every real image into "
    "train/val and never into test (only non-real generators can be held "
    "out), so the aggregate unseen-generator set pairs test's held-out-"
    "generator fakes with val's real images. See "
    ".claude/specs/05-evaluation-pipeline.md, \"Implementation note: "
    "real-image pairing for the unseen-generator result\"."
)

# Duplicated from model/evaluation/report.py's identical constants.
DEV_SHARD_MARKER = "genimage_dev"
DEV_SHARD_BANNER = "> Tiny-GenImage development shard - NOT the final SIH benchmark."

THRESHOLD_INVARIANCE_GUARANTEED_AT = 0.5
ROC_AUC_TOLERANCE = 1e-6


class EmptyUnseenGeneratorSplitError(ValueError):
    """Raised when the split does not contain the minimum composition
    needed for an unseen-generator result. Duplicated from
    model/evaluation/evaluate.py's identical error/check (see module
    docstring for why this is duplicated rather than imported)."""


class RankInvarianceViolationError(RuntimeError):
    """Raised when a before/after ROC-AUC comparison differs by more than
    ROC_AUC_TOLERANCE. Temperature scaling with T > 0 is provably
    rank-invariant (sigmoid(logit / T) is strictly monotonic in logit for
    any T > 0) - a measured change indicates an implementation bug, never
    a modeling result. See .claude/specs/08a-probability-calibration.md
    ("Model / Architecture Changes")."""


class ThresholdInvarianceViolationError(RuntimeError):
    """Raised when raw- and calibrated-probability thresholded
    predictions differ at threshold 0.5. At exactly threshold 0.5,
    sigmoid(logit) >= 0.5 and sigmoid(logit / T) >= 0.5 (T > 0) are both
    equivalent to logit >= 0, so calibration cannot change any decision -
    a measured difference indicates an implementation bug, never a
    calibration effect."""


class CheckpointModifiedError(RuntimeError):
    """Raised when the checkpoint file's SHA-256 checksum differs before
    vs. after a calibration run. Calibration must never modify the frozen
    detector checkpoint."""


def is_dev_shard(manifest_path: str) -> bool:
    return DEV_SHARD_MARKER in str(manifest_path)


def _resolved_unseen_generators(splits: Dict[str, List[DatasetRecord]]) -> List[str]:
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


def _sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Calibration-run configuration (config/calibration_config.yaml)
# ---------------------------------------------------------------------------


@dataclass
class CalibrationConfig:
    """Run-specific calibration settings, loaded from
    config/calibration_config.yaml (`calibration:` section)."""

    n_bins: int = 15
    lbfgs_max_iter: int = 50
    lbfgs_lr: float = 1.0
    output_dir: str = "experiments/calibration"
    batch_size: int = 16
    num_workers: int = 0
    pin_memory: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


DEFAULT_CALIBRATION_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "calibration_config.yaml"


def load_calibration_config(config_path: Optional[str] = None) -> CalibrationConfig:
    """Loads the `calibration:` section of a YAML calibration
    configuration. Falls back to `CalibrationConfig()` defaults for any
    missing keys - matches the pattern used by
    model/evaluation/config.py::load_evaluation_config."""
    path = Path(config_path) if config_path else DEFAULT_CALIBRATION_CONFIG_PATH
    if not path.is_file():
        return CalibrationConfig()

    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    calibration_raw = raw.get("calibration", {}) or {}
    known_fields = {f.name for f in fields(CalibrationConfig)}
    overrides = {k: v for k, v in calibration_raw.items() if k in known_fields}

    defaults = CalibrationConfig()
    return CalibrationConfig(**{**defaults.to_dict(), **overrides})


# ---------------------------------------------------------------------------
# Metric-block assembly: SplitMetrics (ranking + threshold-policy) plus
# calibration metrics (Brier score, ECE), overall and per-generator.
# ---------------------------------------------------------------------------


def _overall_calibration_block(
    labels: List[int],
    probabilities: List[float],
    ranking_scores: List[float],
    threshold: float,
    n_bins: int,
) -> Dict[str, Any]:
    """`probabilities` (sigmoid outputs) drive the threshold-based metrics
    (accuracy/macro-F1/FPR/confusion matrix) and the calibration metrics
    (Brier/ECE), which are genuinely about the probability values.

    `ranking_scores` - the pre-sigmoid logit for "before", `logit /
    temperature` for "after" - drives `roc_auc` instead of `probabilities`.
    This is required, not cosmetic: `torch.sigmoid` saturates to exactly
    1.0/0.0 in float32 for |logit| gtrsim 17/-100, which can collapse two
    logits that sit on opposite sides of the real/fake boundary into an
    identical probability - a tie that does not exist in the underlying
    (real-valued) ranking. Temperature scaling divides the logit before
    that saturation point, and can reveal the pair's true order, changing
    which one ranks higher - a measurable ROC-AUC difference caused
    entirely by this float32 precision loss, not by any actual change in
    ranking. Dividing a real number by a positive scalar exactly preserves
    its order with no equivalent collapsing behavior, so computing roc_auc
    from `ranking_scores` keeps the before/after comparison mathematically
    rank-invariant, as `_assert_roc_auc_invariant` requires.
    """
    metrics: SplitMetrics = compute_split_metrics(labels, probabilities, threshold)
    metrics.roc_auc = roc_auc_score(labels, ranking_scores)
    block = metrics.to_dict()
    block["brier_score"] = brier_score(probabilities, labels)
    block["ece"] = expected_calibration_error(probabilities, labels, n_bins=n_bins)
    return block


def _per_generator_calibration_blocks(
    labels: List[int],
    probabilities: List[float],
    ranking_scores: List[float],
    generators: List[str],
    threshold: float,
    n_bins: int,
    real_generator_id: str = REAL_GENERATOR_ID,
) -> Dict[str, Any]:
    """Pairs every real example with each non-real generator's own rows -
    the same pairing model/evaluation/metrics.py::per_generator_breakdown
    uses (re-derived here so brier_score/ece can be computed on the exact
    same per-generator subset, not just the ranking/threshold metrics
    that function alone returns)."""
    real_indices = [i for i, g in enumerate(generators) if g == real_generator_id]
    generator_ids = sorted({g for g in generators if g != real_generator_id})

    blocks: Dict[str, Any] = {}
    for generator_id in generator_ids:
        generator_indices = [i for i, g in enumerate(generators) if g == generator_id]
        indices = real_indices + generator_indices
        group_labels = [labels[i] for i in indices]
        group_probabilities = [probabilities[i] for i in indices]
        group_ranking_scores = [ranking_scores[i] for i in indices]
        blocks[generator_id] = _overall_calibration_block(
            group_labels, group_probabilities, group_ranking_scores, threshold, n_bins
        )
    return blocks


def _split_block(
    labels: List[int],
    probabilities: List[float],
    ranking_scores: List[float],
    generators: List[str],
    threshold: float,
    n_bins: int,
) -> Dict[str, Any]:
    return {
        "overall": _overall_calibration_block(labels, probabilities, ranking_scores, threshold, n_bins),
        "per_generator": _per_generator_calibration_blocks(
            labels, probabilities, ranking_scores, generators, threshold, n_bins
        ),
    }


def _append_note(block: Dict[str, Any], note: str) -> None:
    block["overall"]["notes"].append(note)
    for generator_block in block["per_generator"].values():
        generator_block["notes"].append(note)


def _assert_roc_auc_invariant(label: str, before: Optional[float], after: Optional[float]) -> None:
    if before is None and after is None:
        return
    if before is None or after is None:
        raise RankInvarianceViolationError(
            f"{label} ROC-AUC was defined on one side of calibration and undefined on the "
            f"other (before={before!r}, after={after!r}). Temperature scaling cannot change "
            "whether ROC-AUC is defined (it does not change class composition) - this "
            "indicates an implementation bug."
        )
    if abs(before - after) > ROC_AUC_TOLERANCE:
        raise RankInvarianceViolationError(
            f"{label} ROC-AUC changed beyond tolerance under calibration: {before!r} -> "
            f"{after!r} (tolerance={ROC_AUC_TOLERANCE}). Temperature scaling with T > 0 is "
            "provably rank-invariant - this must be treated as an implementation bug, never "
            "as a modeling result."
        )


def _labels_generators_logits(rows: List[LogitPredictionRow]):
    labels = [row.true_label for row in rows]
    generators = [row.generator for row in rows]
    logits = [row.raw_logit for row in rows]
    return labels, generators, logits


def _make_calib_run_id() -> str:
    return f"calib-{time.strftime('%Y%m%d-%H%M%S')}"


# ---------------------------------------------------------------------------
# Primary orchestration
# ---------------------------------------------------------------------------


def run_calibration(
    checkpoint_path: str,
    manifest_path: str,
    model_config_path: Optional[str] = None,
    calibration_config_path: Optional[str] = None,
    output_dir_override: Optional[str] = None,
    unseen_generators_override: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Runs one full calibration job and returns a summary dict.

    The checkpoint is loaded read-only and checksummed before and after
    this function runs (see `CheckpointModifiedError`). Temperature is
    fit exclusively on the validation split's raw logits; the
    unseen-generator set is only read for the before/after evaluation
    pass, after the temperature is already fixed.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checksum_before = _sha256_of_file(checkpoint_path)

    calibration_config = load_calibration_config(calibration_config_path)
    data_config = load_data_config(model_config_path)
    model_config = load_model_config(model_config_path)

    threshold = model_config.threshold

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
    architecture_id = DEFAULT_ARCHITECTURE_ID
    try:
        training_checkpoint = load_training_checkpoint(checkpoint_path, map_location="cpu")
        checkpoint_model_config = training_checkpoint.get("model_config") or {}
        architecture_id = checkpoint_model_config.get("architecture", DEFAULT_ARCHITECTURE_ID)
        training_provenance = {
            "training_config": training_checkpoint.get("training_config"),
            "model_config": training_checkpoint.get("model_config"),
            "training_epoch": training_checkpoint.get("epoch"),
            "training_best_val_roc_auc": training_checkpoint.get("best_val_roc_auc"),
            "training_seed": training_checkpoint.get("seed"),
        }
    except Exception:
        # The model already loaded successfully via load_model_from_checkpoint
        # (which independently resolves architecture from the same field);
        # missing/incompatible full-checkpoint metadata only degrades the
        # reported provenance, matching model/evaluation/evaluate.py's
        # identical fallback - it does not block calibration.
        training_provenance = {"warning": "training checkpoint metadata unavailable"}

    preprocessor_config = data_config.preprocessing
    from data.preprocessor import ImagePreprocessor

    preprocessor = ImagePreprocessor.from_config(preprocessor_config)

    if architecture_id == DEFAULT_ARCHITECTURE_ID:
        val_dataset = build_val_dataset(splits["val"], preprocessor)
        test_dataset = build_val_dataset(splits["test"], preprocessor)
    elif architecture_id in _DUAL_BRANCH_ARCHITECTURES:
        val_dataset = build_dual_branch_val_dataset(splits["val"], preprocessor)
        test_dataset = build_dual_branch_val_dataset(splits["test"], preprocessor)
    else:
        raise UnknownArchitectureError(
            f"Checkpoint {str(checkpoint_path)!r} declares architecture {architecture_id!r}, which "
            "has no registered calibration-dataset construction. Known architectures: "
            f"{(DEFAULT_ARCHITECTURE_ID,) + _DUAL_BRANCH_ARCHITECTURES}."
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=calibration_config.batch_size,
        shuffle=False,
        num_workers=calibration_config.num_workers,
        pin_memory=calibration_config.pin_memory,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=calibration_config.batch_size,
        shuffle=False,
        num_workers=calibration_config.num_workers,
        pin_memory=calibration_config.pin_memory,
        drop_last=False,
    )

    val_rows = run_logit_inference(model, val_loader, device, split="val")
    test_rows = run_logit_inference(model, test_loader, device, split="test")

    # --- Fit: validation logits ONLY. Never the unseen-generator split. ---
    val_labels, val_generators, val_logits_list = _labels_generators_logits(val_rows)
    val_logits_tensor = torch.tensor(val_logits_list, dtype=torch.float32)
    val_labels_tensor = torch.tensor(val_labels, dtype=torch.float32)

    fit_result: TemperatureFitResult = fit_temperature(
        val_logits_tensor,
        val_labels_tensor,
        max_iter=calibration_config.lbfgs_max_iter,
        lr=calibration_config.lbfgs_lr,
    )
    temperature = fit_result.temperature

    # --- Unseen-generator evaluation set: test's held-out-generator rows
    # paired with val's real rows (never val's fakes) - identical
    # composition to model/evaluation/evaluate.py::run_evaluation, but
    # re-derived here (see module docstring). ---
    val_real_rows = [row for row in val_rows if row.generator == REAL_GENERATOR_ID]
    unseen_rows = test_rows + val_real_rows
    unseen_labels, unseen_generators, unseen_logits_list = _labels_generators_logits(unseen_rows)
    unseen_logits_tensor = torch.tensor(unseen_logits_list, dtype=torch.float32)

    raw_val_probs = torch.sigmoid(val_logits_tensor).tolist()
    raw_unseen_probs = torch.sigmoid(unseen_logits_tensor).tolist()
    calibrated_val_probs = apply_temperature(val_logits_tensor, temperature).tolist()
    calibrated_unseen_probs = apply_temperature(unseen_logits_tensor, temperature).tolist()

    # Ranking scores for roc_auc specifically (see _overall_calibration_block):
    # the raw logit for "before", the temperature-scaled logit (pre-sigmoid)
    # for "after" - never the post-sigmoid probability, which can saturate
    # and lose the precision the rank-invariance guarantee depends on.
    #
    # The division itself is done in float64, not float32: two distinct
    # float32 logits can be close enough that dividing them by the same
    # float32 temperature rounds both results to the identical float32
    # value (a real, observed case: 21.444255828857422 and
    # 21.444257736206055 both divide to exactly 36.721885681152344 in
    # float32 at T=0.5839639382636609). That introduces a tie under
    # calibration that did not exist in the raw logits, which can shift
    # roc_auc exactly like the sigmoid-saturation case this ranking-score
    # approach was introduced to fix. float64 has 52 mantissa bits versus
    # float32's 23, so two float32 inputs that are already distinct by at
    # least 1 float32 ULP cannot coincidentally collide after float64
    # division by a common positive scalar - the mathematical ordering is
    # preserved in practice, not just in theory.
    scaled_val_logits = (val_logits_tensor.double() / temperature).tolist()
    scaled_unseen_logits = (unseen_logits_tensor.double() / temperature).tolist()

    n_bins = calibration_config.n_bins

    metrics_before = {
        "val": _split_block(val_labels, raw_val_probs, val_logits_list, val_generators, threshold, n_bins),
        "unseen_generator": _split_block(
            unseen_labels, raw_unseen_probs, unseen_logits_list, unseen_generators, threshold, n_bins
        ),
    }
    metrics_after = {
        "val": _split_block(val_labels, calibrated_val_probs, scaled_val_logits, val_generators, threshold, n_bins),
        "unseen_generator": _split_block(
            unseen_labels, calibrated_unseen_probs, scaled_unseen_logits, unseen_generators, threshold, n_bins
        ),
    }
    _append_note(metrics_before["unseen_generator"], REAL_PAIRING_NOTE)
    _append_note(metrics_after["unseen_generator"], REAL_PAIRING_NOTE)

    # --- Correctness invariants: verified, not assumed. ---
    val_roc_before = metrics_before["val"]["overall"]["roc_auc"]
    val_roc_after = metrics_after["val"]["overall"]["roc_auc"]
    unseen_roc_before = metrics_before["unseen_generator"]["overall"]["roc_auc"]
    unseen_roc_after = metrics_after["unseen_generator"]["overall"]["roc_auc"]
    _assert_roc_auc_invariant("validation", val_roc_before, val_roc_after)
    _assert_roc_auc_invariant("unseen-generator", unseen_roc_before, unseen_roc_after)

    raw_val_preds = threshold_probabilities(raw_val_probs, threshold)
    calibrated_val_preds = threshold_probabilities(calibrated_val_probs, threshold)
    raw_unseen_preds = threshold_probabilities(raw_unseen_probs, threshold)
    calibrated_unseen_preds = threshold_probabilities(calibrated_unseen_probs, threshold)

    val_predictions_identical = raw_val_preds == calibrated_val_preds
    unseen_predictions_identical = raw_unseen_preds == calibrated_unseen_preds

    if abs(threshold - THRESHOLD_INVARIANCE_GUARANTEED_AT) < 1e-12:
        if not val_predictions_identical:
            raise ThresholdInvarianceViolationError(
                "Validation-split thresholded predictions changed under calibration at "
                "threshold 0.5, where this is mathematically impossible for T > 0. This "
                "indicates an implementation bug."
            )
        if not unseen_predictions_identical:
            raise ThresholdInvarianceViolationError(
                "Unseen-generator-set thresholded predictions changed under calibration at "
                "threshold 0.5, where this is mathematically impossible for T > 0. This "
                "indicates an implementation bug."
            )

    checksum_after = _sha256_of_file(checkpoint_path)
    checkpoint_unmodified = checksum_after == checksum_before
    if not checkpoint_unmodified:
        raise CheckpointModifiedError(
            f"Checkpoint file changed during calibration: before={checksum_before!r}, "
            f"after={checksum_after!r}. Calibration must never modify the frozen detector "
            "checkpoint."
        )

    # --- Persist artifacts. ---
    calib_run_id = _make_calib_run_id()
    run_dir = Path(output_dir_override or calibration_config.output_dir) / calib_run_id

    dataset_label = (
        "genimage_dev (development shard, not SIH benchmark)" if is_dev_shard(manifest_path) else str(manifest_path)
    )

    RunLogger(run_dir).write_config(
        {
            "calib_run_id": calib_run_id,
            "calibration_method": "temperature_scaling",
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_architecture": architecture_id,
            "checkpoint_sha256_before": checksum_before,
            "checkpoint_sha256_after": checksum_after,
            "checkpoint_unmodified": checkpoint_unmodified,
            "manifest_path": str(manifest_path),
            "dataset": dataset_label,
            "threshold": threshold,
            "threshold_source": "model_config",
            "split_config": {
                "seed": split_config.seed,
                "val_fraction": split_config.val_fraction,
                "unseen_generators": _resolved_unseen_generators(splits),
                "held_out_generator_fraction": split_config.held_out_generator_fraction,
            },
            "seed": training_provenance.get("training_seed"),
            "calibration_config": calibration_config.to_dict(),
            "device": device.type,
            "num_val_samples": len(val_rows),
            "num_test_samples": len(test_rows),
            "num_unseen_generator_eval_samples": len(unseen_rows),
            "roc_auc_invariance": {
                "val": {"before": val_roc_before, "after": val_roc_after, "tolerance": ROC_AUC_TOLERANCE},
                "unseen_generator": {
                    "before": unseen_roc_before,
                    "after": unseen_roc_after,
                    "tolerance": ROC_AUC_TOLERANCE,
                },
            },
            "threshold_prediction_invariance": {
                "threshold": threshold,
                "val_predictions_identical": val_predictions_identical,
                "unseen_generator_predictions_identical": unseen_predictions_identical,
                "guaranteed_at_threshold": THRESHOLD_INVARIANCE_GUARANTEED_AT,
            },
            "training_provenance": training_provenance,
            "timestamp": time.time(),
        }
    )

    with open(run_dir / "temperature.json", "w", encoding="utf-8") as fh:
        json.dump(fit_result.to_dict(), fh, indent=2, default=str)

    with open(run_dir / "metrics_before.json", "w", encoding="utf-8") as fh:
        json.dump(metrics_before, fh, indent=2, default=str)

    with open(run_dir / "metrics_after.json", "w", encoding="utf-8") as fh:
        json.dump(metrics_after, fh, indent=2, default=str)

    _write_calibration_report(
        run_dir / "calibration_report.md",
        manifest_path=manifest_path,
        architecture_id=architecture_id,
        temperature=temperature,
        threshold=threshold,
        metrics_before=metrics_before,
        metrics_after=metrics_after,
        val_predictions_identical=val_predictions_identical,
        unseen_predictions_identical=unseen_predictions_identical,
    )

    return {
        "calib_run_id": calib_run_id,
        "run_dir": str(run_dir),
        "temperature": temperature,
        "val_roc_auc": {"before": val_roc_before, "after": val_roc_after},
        "unseen_generator_roc_auc": {"before": unseen_roc_before, "after": unseen_roc_after},
        "brier_before": {
            "val": metrics_before["val"]["overall"]["brier_score"],
            "unseen_generator": metrics_before["unseen_generator"]["overall"]["brier_score"],
        },
        "brier_after": {
            "val": metrics_after["val"]["overall"]["brier_score"],
            "unseen_generator": metrics_after["unseen_generator"]["overall"]["brier_score"],
        },
        "ece_before": {
            "val": metrics_before["val"]["overall"]["ece"],
            "unseen_generator": metrics_before["unseen_generator"]["overall"]["ece"],
        },
        "ece_after": {
            "val": metrics_after["val"]["overall"]["ece"],
            "unseen_generator": metrics_after["unseen_generator"]["overall"]["ece"],
        },
        "threshold_predictions_identical": {
            "val": val_predictions_identical,
            "unseen_generator": unseen_predictions_identical,
        },
        "checkpoint_unmodified": checkpoint_unmodified,
    }


def run_secondary_compatibility_check(
    checkpoint_path: str,
    manifest_path: str,
    model_config_path: Optional[str] = None,
    calibration_config_path: Optional[str] = None,
    output_dir_override: Optional[str] = None,
) -> Dict[str, Any]:
    """Non-blocking secondary compatibility check: runs `run_calibration`
    against a second (non-primary) checkpoint architecture and reports
    success/failure without raising. Per
    .claude/specs/08a-probability-calibration.md ("Experiment protocol"),
    failure or unavailability of this check must never block Step 8A's
    primary deliverable (calibration of the Step 7 rgb_frequency_fusion
    checkpoint) - callers should treat this function's return value as a
    report, not a gate.
    """
    try:
        summary = run_calibration(
            checkpoint_path=checkpoint_path,
            manifest_path=manifest_path,
            model_config_path=model_config_path,
            calibration_config_path=calibration_config_path,
            output_dir_override=output_dir_override,
        )
        return {"status": "ok", "summary": summary}
    except Exception as exc:  # noqa: BLE001 - deliberately broad: this check must never raise
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}


def _format_metric(value: Any) -> str:
    if value is None:
        return "undefined"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _report_row(label: str, before: Dict[str, Any], after: Dict[str, Any]) -> List[str]:
    cm_before = before["confusion_matrix"]
    cm_after = after["confusion_matrix"]
    return [
        f"| {label} | ROC-AUC | {_format_metric(before['roc_auc'])} | {_format_metric(after['roc_auc'])} |",
        f"| {label} | Accuracy (threshold={before['threshold']}) | {_format_metric(before['accuracy'])} | "
        f"{_format_metric(after['accuracy'])} |",
        f"| {label} | Macro-F1 (threshold={before['threshold']}) | {_format_metric(before['macro_f1'])} | "
        f"{_format_metric(after['macro_f1'])} |",
        f"| {label} | FPR (threshold={before['threshold']}) | {_format_metric(before['fpr'])} | "
        f"{_format_metric(after['fpr'])} |",
        f"| {label} | Confusion matrix (TP/TN/FP/FN) | "
        f"{cm_before['tp']}/{cm_before['tn']}/{cm_before['fp']}/{cm_before['fn']} | "
        f"{cm_after['tp']}/{cm_after['tn']}/{cm_after['fp']}/{cm_after['fn']} |",
        f"| {label} | Brier score | {_format_metric(before['brier_score'])} | {_format_metric(after['brier_score'])} |",
        f"| {label} | ECE | {_format_metric(before['ece'])} | {_format_metric(after['ece'])} |",
        f"| {label} | num_samples | {before['num_samples']} | {after['num_samples']} |",
    ]


def _write_calibration_report(
    path: Path,
    manifest_path: str,
    architecture_id: str,
    temperature: float,
    threshold: float,
    metrics_before: Dict[str, Any],
    metrics_after: Dict[str, Any],
    val_predictions_identical: bool,
    unseen_predictions_identical: bool,
) -> None:
    lines: List[str] = ["# SignalScope Calibration Report (Step 8A - Temperature Scaling)", ""]
    if is_dev_shard(manifest_path):
        lines.append(DEV_SHARD_BANNER)
        lines.append("")
    lines.append(
        "**Tiny-GenImage is a development shard, not the final SIH benchmark.** No number in "
        "this report may be presented as final SIH benchmark performance."
    )
    lines.append("")
    lines.append(f"**Checkpoint architecture:** `{architecture_id}`")
    lines.append(f"**Fitted temperature T:** {temperature:.6f} (fit on the validation split's raw logits only)")
    lines.append(f"**Frozen decision threshold:** {threshold}")
    lines.append("")
    lines.append(
        "## What this report does and does not claim"
    )
    lines.append("")
    lines.append(
        "- Calibration changes probability/confidence *values* (`sigmoid(logit / T)` vs. "
        f"`sigmoid(logit)`), but at threshold {threshold} it does not change the underlying "
        "binary decisions - `T > 0` makes `sigmoid(logit) >= 0.5` and `sigmoid(logit / T) >= "
        "0.5` both equivalent to `logit >= 0`. This is verified directly below (element-wise "
        "prediction comparison), not merely assumed from the algebra."
    )
    lines.append(
        "- ROC-AUC is expected to remain invariant before vs. after calibration on both the "
        "validation split and the unseen-generator evaluation set, because temperature scaling "
        "with `T > 0` is a strictly monotonic transform of the logit and therefore cannot "
        "change example ranking. This run's measured before/after ROC-AUC values are reported "
        "below; if they ever differed by more than 1e-6 this run would have raised an error "
        "rather than silently reporting the mismatched numbers."
    )
    lines.append(
        "- Any Brier score / ECE improvement reported below reflects the *quality of the "
        "model's confidence values*, not an improvement in AI-generated-image detection "
        "capability. Detection capability is measured by ROC-AUC/accuracy/macro-F1/FPR, which "
        "this calibration step does not and cannot improve."
    )
    lines.append(
        "- The calibrated probabilities reported here are not claimed to be universally "
        "reliable: the temperature was fit on a small development-shard validation split (see "
        "`temperature.json`'s `num_fit_samples`), and a single scalar temperature cannot "
        "correct calibration error that varies by generator or by confidence band."
    )
    lines.append("")
    lines.append(f"**Threshold-{threshold} prediction equality (validation split):** " + ("IDENTICAL" if val_predictions_identical else "MISMATCH - THIS IS A BUG"))
    lines.append(
        f"**Threshold-{threshold} prediction equality (unseen-generator set):** "
        + ("IDENTICAL" if unseen_predictions_identical else "MISMATCH - THIS IS A BUG")
    )
    lines.append("")
    lines.append(REAL_PAIRING_NOTE)
    lines.append("")

    lines.append("## Before vs. after calibration")
    lines.append("")
    lines.append("| Split | Metric | Before (raw) | After (calibrated) |")
    lines.append("|---|---|---|---|")
    lines.extend(_report_row("Validation", metrics_before["val"]["overall"], metrics_after["val"]["overall"]))
    lines.extend(
        _report_row(
            "Unseen-generator (dev eval)",
            metrics_before["unseen_generator"]["overall"],
            metrics_after["unseen_generator"]["overall"],
        )
    )
    lines.append("")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit and evaluate temperature-scaling calibration for a frozen SignalScope checkpoint."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to a Step 4/6/7 training checkpoint (.pt).")
    parser.add_argument("--manifest", required=True, help="Path to the dataset manifest CSV/JSON.")
    parser.add_argument("--model-config", default=None, help="Path to model_config.yaml (default: repo default).")
    parser.add_argument(
        "--calibration-config", default=None, help="Path to calibration_config.yaml (default: repo default)."
    )
    parser.add_argument("--output-dir", default=None, help="Override the calibration output directory.")
    parser.add_argument(
        "--unseen-generators",
        default=None,
        help="Comma-separated explicit unseen-generator list, overriding config-derived selection.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    unseen_generators_override = (
        [g.strip() for g in args.unseen_generators.split(",") if g.strip()] if args.unseen_generators else None
    )

    summary = run_calibration(
        checkpoint_path=args.checkpoint,
        manifest_path=args.manifest,
        model_config_path=args.model_config,
        calibration_config_path=args.calibration_config,
        output_dir_override=args.output_dir,
        unseen_generators_override=unseen_generators_override,
    )
    print(f"calib_run_id = {summary['calib_run_id']}")
    print(f"run_dir      = {summary['run_dir']}")
    print(f"temperature  = {summary['temperature']}")
    print(
        "val_roc_auc (before/after) = "
        f"{summary['val_roc_auc']['before']} / {summary['val_roc_auc']['after']}"
    )
    print(
        "unseen_generator_roc_auc (before/after, primary result) = "
        f"{summary['unseen_generator_roc_auc']['before']} / {summary['unseen_generator_roc_auc']['after']}"
    )
    print(f"checkpoint_unmodified = {summary['checkpoint_unmodified']}")


if __name__ == "__main__":
    main()
