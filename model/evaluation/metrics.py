"""
Pure evaluation metrics for the SignalScope evaluation pipeline.

Operate on plain sequences (labels/probabilities/predictions) with no
torch/dataset/config dependency, so they are trivially unit-testable and
reusable for both the overall split-level result and the per-generator
breakdown. See .claude/specs/05-evaluation-pipeline.md ("Metric
definitions", "Per-generator analysis").

Responsible Team Member: Member 6 (MLOps & Testing)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from model.training.engine import roc_auc_score

REAL_LABEL = 0
AI_GENERATED_LABEL = 1


class EmptyEvaluationSetError(ValueError):
    """Raised when a metric is requested over zero examples.

    Never silently return 0.0/NaN for an empty input - that would be
    indistinguishable from a genuinely bad (but computed) result.
    """


def _validate_non_empty(labels: Sequence[int]) -> None:
    if len(labels) == 0:
        raise EmptyEvaluationSetError("Cannot compute metrics over an empty evaluation set.")


def threshold_probabilities(probabilities: Sequence[float], threshold: float) -> List[int]:
    """Applies a fixed decision threshold to raw P(ai_generated) probabilities."""
    return [AI_GENERATED_LABEL if p >= threshold else REAL_LABEL for p in probabilities]


def accuracy(labels: Sequence[int], predictions: Sequence[int]) -> float:
    """Fraction of predictions matching the true label. Raises
    EmptyEvaluationSetError on an empty input."""
    _validate_non_empty(labels)
    correct = sum(1 for label, pred in zip(labels, predictions) if label == pred)
    return correct / len(labels)


def confusion_matrix(labels: Sequence[int], predictions: Sequence[int]) -> Dict[str, int]:
    """Binary confusion matrix counts; positive class = AI_GENERATED_LABEL (1).

    Raises EmptyEvaluationSetError on an empty input.
    """
    _validate_non_empty(labels)
    tp = tn = fp = fn = 0
    for label, pred in zip(labels, predictions):
        if label == AI_GENERATED_LABEL and pred == AI_GENERATED_LABEL:
            tp += 1
        elif label == REAL_LABEL and pred == REAL_LABEL:
            tn += 1
        elif label == REAL_LABEL and pred == AI_GENERATED_LABEL:
            fp += 1
        else:  # label == AI_GENERATED_LABEL and pred == REAL_LABEL
            fn += 1
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def fpr_at_threshold(confusion: Dict[str, int]) -> Optional[float]:
    """FPR = FP / (FP + TN). Returns None (never 0.0) when FP + TN == 0
    (no real/negative examples in the evaluated subset)."""
    denominator = confusion["fp"] + confusion["tn"]
    if denominator == 0:
        return None
    return confusion["fp"] / denominator


def _binary_f1(confusion: Dict[str, int], positive_label: int) -> float:
    """Per-class F1 for a binary confusion matrix, `positive_label`-relative.

    Undefined precision/recall (zero denominator) is treated as 0.0 rather
    than raised, matching the common "zero_division=0" convention - a
    degenerate single-class evaluation still produces a well-defined (if
    uninformative) number instead of crashing the report.
    """
    if positive_label == AI_GENERATED_LABEL:
        tp, fp, fn = confusion["tp"], confusion["fp"], confusion["fn"]
    else:
        tp, fp, fn = confusion["tn"], confusion["fn"], confusion["fp"]

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def macro_f1(labels: Sequence[int], predictions: Sequence[int]) -> float:
    """Unweighted mean of per-class F1 (real, ai_generated). Raises
    EmptyEvaluationSetError on an empty input."""
    _validate_non_empty(labels)
    confusion = confusion_matrix(labels, predictions)
    f1_real = _binary_f1(confusion, REAL_LABEL)
    f1_ai = _binary_f1(confusion, AI_GENERATED_LABEL)
    return (f1_real + f1_ai) / 2.0


@dataclass
class SplitMetrics:
    """A full metrics block for one evaluated split (or per-generator
    subset): the SIH-required metric set plus the sample count it was
    computed over."""

    num_samples: int
    num_real: int
    num_fake: int
    accuracy: float
    macro_f1: float
    roc_auc: Optional[float]
    confusion_matrix: Dict[str, int]
    fpr: Optional[float]
    threshold: float
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "num_samples": self.num_samples,
            "num_real": self.num_real,
            "num_fake": self.num_fake,
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "roc_auc": self.roc_auc,
            "confusion_matrix": self.confusion_matrix,
            "fpr": self.fpr,
            "threshold": self.threshold,
            "notes": self.notes,
        }


def compute_split_metrics(
    labels: Sequence[int],
    probabilities: Sequence[float],
    threshold: float,
) -> SplitMetrics:
    """Computes the full SIH-required metric set for one group of
    (label, probability) pairs at a fixed threshold.

    Raises EmptyEvaluationSetError on an empty input. ROC-AUC and FPR are
    reported as None (with an explanatory note), never as misleading zero
    values, when mathematically undefined (single-class input).
    """
    _validate_non_empty(labels)
    predictions = threshold_probabilities(probabilities, threshold)

    notes: List[str] = []
    auc = roc_auc_score(labels, probabilities)
    if auc is None:
        notes.append("roc_auc undefined: only one class present in this evaluated group.")

    confusion = confusion_matrix(labels, predictions)
    fpr = fpr_at_threshold(confusion)
    if fpr is None:
        notes.append("fpr undefined: no real (negative) examples in this evaluated group.")

    return SplitMetrics(
        num_samples=len(labels),
        num_real=sum(1 for label in labels if label == REAL_LABEL),
        num_fake=sum(1 for label in labels if label == AI_GENERATED_LABEL),
        accuracy=accuracy(labels, predictions),
        macro_f1=macro_f1(labels, predictions),
        roc_auc=auc,
        confusion_matrix=confusion,
        fpr=fpr,
        threshold=threshold,
        notes=notes,
    )


def per_generator_breakdown(
    labels: Sequence[int],
    probabilities: Sequence[float],
    generators: Sequence[str],
    threshold: float,
    real_generator_id: str = "real",
) -> Dict[str, SplitMetrics]:
    """Computes one SplitMetrics block per non-real generator present, each
    paired with every real example in the same group (see
    .claude/specs/05-evaluation-pipeline.md, "Per-generator analysis").

    Filtering to a single generator's own rows alone would always be
    single-class (every non-real generator's rows are 100% AI_GENERATED,
    "real"'s rows are 100% REAL - see data/dataset_loader.py), making
    ROC-AUC always undefined. Pairing {all real rows} + {this generator's
    rows} answers the actually-meaningful "real vs. generator G" question.
    """
    real_indices = [i for i, g in enumerate(generators) if g == real_generator_id]
    generator_ids = sorted({g for g in generators if g != real_generator_id})

    breakdown: Dict[str, SplitMetrics] = {}
    for generator_id in generator_ids:
        generator_indices = [i for i, g in enumerate(generators) if g == generator_id]
        indices = real_indices + generator_indices
        group_labels = [labels[i] for i in indices]
        group_probabilities = [probabilities[i] for i in indices]
        breakdown[generator_id] = compute_split_metrics(group_labels, group_probabilities, threshold)

    return breakdown
