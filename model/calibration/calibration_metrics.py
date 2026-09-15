"""
Pure calibration-quality metrics for SignalScope temperature scaling
(Step 8A).

Operate on plain sequences (probabilities/labels) with no torch/dataset/
config dependency, mirroring the existing style of
model/evaluation/metrics.py. Never silently return 0.0 for an empty input
- that would be indistinguishable from a genuinely well-calibrated (but
computed) result. See .claude/specs/08a-probability-calibration.md
("Calibration metrics").
"""

from __future__ import annotations

from typing import Sequence


class EmptyCalibrationSetError(ValueError):
    """Raised when a calibration metric or fit is requested over zero
    examples. Mirrors model.evaluation.metrics.EmptyEvaluationSetError's
    convention: never silently return a misleading 0.0/NaN for an empty
    input."""


def _validate_non_empty(labels: Sequence[int]) -> None:
    if len(labels) == 0:
        raise EmptyCalibrationSetError("Cannot compute a calibration metric over an empty evaluation set.")


def _validate_matching_lengths(probabilities: Sequence[float], labels: Sequence[int]) -> None:
    if len(probabilities) != len(labels):
        raise ValueError(
            f"probabilities and labels must have the same length "
            f"(got {len(probabilities)} and {len(labels)})."
        )


def brier_score(probabilities: Sequence[float], labels: Sequence[int]) -> float:
    """Mean squared error between predicted probability and true binary
    label: mean((p_i - y_i)^2), y_i in {0, 1}.

    Lower is better (0.0 = perfect). Unlike ROC-AUC, this is sensitive to
    the actual probability *values*, not just their ranking - it is a
    calibration-quality metric, not a ranking metric. Raises
    EmptyCalibrationSetError on an empty input.
    """
    _validate_non_empty(labels)
    _validate_matching_lengths(probabilities, labels)
    squared_errors = [(float(p) - float(y)) ** 2 for p, y in zip(probabilities, labels)]
    return sum(squared_errors) / len(squared_errors)


def expected_calibration_error(
    probabilities: Sequence[float],
    labels: Sequence[int],
    n_bins: int = 15,
) -> float:
    """Expected Calibration Error (ECE), fixed-width-bin definition.

    Partitions [0, 1] into `n_bins` equal-width bins. For each non-empty
    bin, computes the absolute gap between the bin's mean predicted
    probability (confidence) and its mean true positive rate (accuracy,
    fraction of label == 1), weighted by the bin's share of all examples:

        ECE = sum_b (|bin_b| / N) * |mean(prob in bin_b) - mean(label in bin_b)|

    A probability of exactly 1.0 is assigned to the last bin (closed
    interval on the right edge only for the final bin; every other bin is
    half-open [lo, hi)). Raises EmptyCalibrationSetError on an empty input
    and ValueError for a non-positive `n_bins`.
    """
    _validate_non_empty(labels)
    _validate_matching_lengths(probabilities, labels)
    if n_bins <= 0:
        raise ValueError(f"n_bins must be a positive integer, got {n_bins!r}.")

    num_samples = len(labels)
    bin_prob_sums = [0.0] * n_bins
    bin_label_sums = [0.0] * n_bins
    bin_counts = [0] * n_bins

    for probability, label in zip(probabilities, labels):
        p = float(probability)
        bin_index = int(p * n_bins)
        if bin_index >= n_bins:
            bin_index = n_bins - 1  # p == 1.0 falls into the last bin
        if bin_index < 0:
            bin_index = 0
        bin_prob_sums[bin_index] += p
        bin_label_sums[bin_index] += float(label)
        bin_counts[bin_index] += 1

    ece = 0.0
    for count, prob_sum, label_sum in zip(bin_counts, bin_prob_sums, bin_label_sums):
        if count == 0:
            continue
        mean_confidence = prob_sum / count
        mean_accuracy = label_sum / count
        ece += (count / num_samples) * abs(mean_confidence - mean_accuracy)

    return ece
