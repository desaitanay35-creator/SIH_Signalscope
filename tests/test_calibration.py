"""
Unit tests for SignalScope Step 8A probability calibration.

Covers temperature scaling (forward pass, positivity/finiteness, fitting,
reproducibility, edge cases) and the pure calibration metrics (Brier
score, ECE) against hand-computed values. All synthetic - never touches
the real GenImage shard or any hidden SIH test set.
"""

import math

import pytest

torch = pytest.importorskip("torch")

from model.calibration.calibration_metrics import (
    EmptyCalibrationSetError,
    brier_score,
    expected_calibration_error,
)
from model.calibration.temperature_scaling import (
    TemperatureFitError,
    TemperatureScaler,
    apply_temperature,
    fit_temperature,
)


# ---------------------------------------------------------------------------
# TemperatureScaler forward pass / positivity / finiteness
# ---------------------------------------------------------------------------


def test_temperature_scaler_initializes_at_identity():
    scaler = TemperatureScaler()
    assert scaler.temperature.item() == pytest.approx(1.0)


def test_temperature_scaler_forward_scales_logits():
    scaler = TemperatureScaler()
    with torch.no_grad():
        scaler.log_temperature.fill_(math.log(2.0))
    logits = torch.tensor([2.0, -4.0, 0.0])
    scaled = scaler(logits)
    assert scaled.tolist() == pytest.approx([1.0, -2.0, 0.0])


def test_temperature_scaler_temperature_always_positive_for_any_log_temperature():
    scaler = TemperatureScaler()
    for value in (-50.0, -1.0, 0.0, 1.0, 50.0):
        with torch.no_grad():
            scaler.log_temperature.fill_(value)
        assert scaler.temperature.item() > 0.0
        assert math.isfinite(scaler.temperature.item())


def test_apply_temperature_rejects_non_positive_temperature():
    logits = torch.tensor([0.1, 0.2])
    with pytest.raises(ValueError):
        apply_temperature(logits, 0.0)
    with pytest.raises(ValueError):
        apply_temperature(logits, -1.0)


def test_apply_temperature_probability_bounds():
    # Deliberately avoids logits extreme enough to underflow/overflow
    # sigmoid to an exact 0.0/1.0 in float32 (e.g. +-1000) - that is a
    # floating-point saturation artifact of sigmoid itself, not something
    # apply_temperature's (0, 1) contract is meant to paper over.
    logits = torch.tensor([-15.0, -1.0, 0.0, 1.0, 15.0])
    probs = apply_temperature(logits, 2.0)
    for p in probs.tolist():
        assert 0.0 < p < 1.0


def test_apply_temperature_at_t_equals_one_matches_plain_sigmoid():
    logits = torch.tensor([-3.0, -0.5, 0.0, 0.5, 3.0])
    calibrated = apply_temperature(logits, 1.0)
    raw = torch.sigmoid(logits)
    assert calibrated.tolist() == pytest.approx(raw.tolist(), abs=1e-7)


def test_apply_temperature_never_takes_probabilities_as_input_contract():
    # A raw logit and its sigmoid must produce different calibrated outputs
    # when passed through apply_temperature at T != 1 - this is a proxy for
    # "the function treats its input as a logit, not a probability".
    logit = torch.tensor([2.0])
    probability_of_that_logit = torch.sigmoid(logit)
    assert apply_temperature(logit, 2.0).item() != pytest.approx(apply_temperature(probability_of_that_logit, 2.0).item())


# ---------------------------------------------------------------------------
# fit_temperature: convergence, reproducibility, edge cases
# ---------------------------------------------------------------------------


def _synthetic_separable_logits(n_per_class: int = 50, true_temperature: float = 3.0, seed: int = 0):
    """Deterministically class-separated logits (used only for
    "does the fit run/converge/not crash" checks, NOT for checking a known
    target temperature - see `_synthetic_miscalibrated_logits` for that).
    """
    generator = torch.Generator().manual_seed(seed)
    base_pos = torch.randn(n_per_class, generator=generator) * 0.5 + 2.0
    base_neg = torch.randn(n_per_class, generator=generator) * 0.5 - 2.0
    base_logits = torch.cat([base_pos, base_neg])
    labels = torch.cat([torch.ones(n_per_class), torch.zeros(n_per_class)])
    scaled_logits = base_logits * true_temperature
    return scaled_logits, labels


def _synthetic_miscalibrated_logits(n: int = 800, true_temperature: float = 3.0, seed: int = 0):
    """Constructs logits with a KNOWN, recoverable miscalibration: labels
    are sampled stochastically from `sigmoid(base_logit)` (so `base_logit`
    is, by construction, exactly the well-calibrated logit for this label
    distribution), then over-scaled by `true_temperature` to simulate an
    overconfident model. The NLL-minimizing temperature for the resulting
    (overconfident_logit, label) pairs should recover `true_temperature`
    (dividing back out reproduces the well-calibrated `base_logit`) -
    unlike a deterministically-separable fixture, where NLL is minimized
    by driving T toward 0 (maximum confidence), not toward any particular
    "true" temperature.
    """
    generator = torch.Generator().manual_seed(seed)
    base_logits = torch.randn(n, generator=generator) * 2.0
    probabilities = torch.sigmoid(base_logits)
    labels = torch.bernoulli(probabilities, generator=generator)
    overconfident_logits = base_logits * true_temperature
    return overconfident_logits, labels


def test_fit_temperature_converges_toward_the_known_true_temperature():
    true_temperature = 3.0
    logits, labels = _synthetic_miscalibrated_logits(true_temperature=true_temperature)
    result = fit_temperature(logits, labels, max_iter=100, lr=1.0)
    assert result.temperature == pytest.approx(true_temperature, rel=0.3)
    assert result.temperature > 0.0
    assert math.isfinite(result.temperature)
    assert result.final_nll <= result.initial_nll


def test_fit_temperature_reduces_nll_relative_to_initial():
    logits, labels = _synthetic_miscalibrated_logits(true_temperature=5.0)
    result = fit_temperature(logits, labels)
    assert result.final_nll < result.initial_nll


def test_fit_temperature_records_num_fit_samples_and_iterations():
    logits, labels = _synthetic_separable_logits(n_per_class=10)
    result = fit_temperature(logits, labels, max_iter=20)
    assert result.num_fit_samples == 20
    assert result.lbfgs_max_iter == 20
    assert result.optimizer_iterations >= 1


def test_fit_temperature_is_reproducible_given_identical_inputs():
    logits, labels = _synthetic_miscalibrated_logits(true_temperature=4.0)
    result_a = fit_temperature(logits.clone(), labels.clone(), max_iter=50, lr=1.0)
    result_b = fit_temperature(logits.clone(), labels.clone(), max_iter=50, lr=1.0)
    assert result_a.temperature == pytest.approx(result_b.temperature, rel=1e-9)


def test_fit_temperature_raises_on_empty_input():
    with pytest.raises(EmptyCalibrationSetError):
        fit_temperature(torch.tensor([]), torch.tensor([]))


def test_fit_temperature_raises_on_mismatched_lengths():
    with pytest.raises(ValueError):
        fit_temperature(torch.tensor([1.0, 2.0]), torch.tensor([1.0]))


def test_fit_temperature_handles_single_example_without_crashing():
    result = fit_temperature(torch.tensor([1.5]), torch.tensor([1.0]))
    assert result.temperature > 0.0
    assert math.isfinite(result.temperature)
    assert result.num_fit_samples == 1


def test_fit_temperature_handles_all_one_class_without_crashing():
    logits = torch.tensor([0.5, 1.2, -0.3, 2.0, 0.1])
    labels = torch.ones(5)  # every label is the AI-generated class
    result = fit_temperature(logits, labels, max_iter=50)
    assert result.temperature > 0.0
    assert math.isfinite(result.temperature)

    logits_neg = torch.tensor([0.5, 1.2, -0.3, 2.0, 0.1])
    labels_neg = torch.zeros(5)  # every label is the real class
    result_neg = fit_temperature(logits_neg, labels_neg, max_iter=50)
    assert result_neg.temperature > 0.0
    assert math.isfinite(result_neg.temperature)


# ---------------------------------------------------------------------------
# ROC-AUC / ranking invariance under temperature scaling (logit level)
# ---------------------------------------------------------------------------


def _rank_order(values):
    return sorted(range(len(values)), key=lambda i: values[i])


def test_calibration_never_changes_example_ranking_for_positive_temperature():
    # float64 and a moderate logit/temperature range: large enough to be a
    # meaningful check, small enough that sigmoid does not saturate to an
    # exact 0.0/1.0 (which would produce genuine floating-point ties - a
    # property of sigmoid's finite precision, not a rank-invariance
    # violation in apply_temperature itself).
    generator = torch.Generator().manual_seed(0)
    logits = (torch.rand(40, generator=generator, dtype=torch.float64) - 0.5) * 6.0
    for temperature in (0.5, 1.0, 2.0, 4.0):
        calibrated = apply_temperature(logits, temperature)
        assert _rank_order(logits.tolist()) == _rank_order(calibrated.tolist())


# ---------------------------------------------------------------------------
# Threshold-0.5 decision invariance (logit level) - direct, not assumed
# ---------------------------------------------------------------------------


def test_threshold_0_5_decisions_are_identical_before_and_after_calibration():
    torch.manual_seed(1)
    logits = torch.randn(60) * 4.0
    raw_probs = torch.sigmoid(logits)
    for temperature in (0.2, 0.7, 1.0, 3.0, 8.0):
        calibrated_probs = apply_temperature(logits, temperature)
        raw_decisions = [1 if p >= 0.5 else 0 for p in raw_probs.tolist()]
        calibrated_decisions = [1 if p >= 0.5 else 0 for p in calibrated_probs.tolist()]
        assert raw_decisions == calibrated_decisions


def test_threshold_other_than_0_5_is_not_guaranteed_invariant():
    # Documents the algebraic limitation from the spec: the threshold-0.5
    # guarantee does not generalize to t != 0.5. Construct a logit/temperature
    # pair where a non-0.5 threshold decision does change, to make sure no
    # test in this module accidentally asserts a broader invariant than the
    # spec actually claims.
    logit = torch.tensor([1.0])  # sigmoid(1.0) ~= 0.731
    raw_prob = torch.sigmoid(logit).item()
    calibrated_prob = apply_temperature(logit, 5.0).item()  # sigmoid(0.2) ~= 0.550
    threshold = 0.6
    raw_decision = raw_prob >= threshold
    calibrated_decision = calibrated_prob >= threshold
    assert raw_decision != calibrated_decision


# ---------------------------------------------------------------------------
# Calibration metrics: Brier score, ECE - hand-computed values
# ---------------------------------------------------------------------------


def test_brier_score_hand_computed():
    probabilities = [0.9, 0.1, 0.8, 0.3]
    labels = [1, 0, 1, 1]
    # (0.9-1)^2 + (0.1-0)^2 + (0.8-1)^2 + (0.3-1)^2 = 0.01+0.01+0.04+0.49 = 0.55 -> /4 = 0.1375
    assert brier_score(probabilities, labels) == pytest.approx(0.1375)


def test_brier_score_perfect_predictions_is_zero():
    assert brier_score([1.0, 0.0, 1.0], [1, 0, 1]) == pytest.approx(0.0)


def test_brier_score_raises_on_empty_input():
    with pytest.raises(EmptyCalibrationSetError):
        brier_score([], [])


def test_brier_score_raises_on_mismatched_lengths():
    with pytest.raises(ValueError):
        brier_score([0.5, 0.5], [1])


def test_expected_calibration_error_hand_computed_two_bins():
    # 2 bins: [0, 0.5), [0.5, 1.0]. Bin 1 (2 examples, prob<0.5): probs
    # [0.1, 0.2] mean=0.15, labels [0, 0] mean=0.0 -> gap 0.15, weight 2/4.
    # Bin 2 (2 examples, prob>=0.5): probs [0.8, 0.9] mean=0.85,
    # labels [1, 1] mean=1.0 -> gap 0.15, weight 2/4.
    # ECE = 0.5*0.15 + 0.5*0.15 = 0.15
    probabilities = [0.1, 0.2, 0.8, 0.9]
    labels = [0, 0, 1, 1]
    assert expected_calibration_error(probabilities, labels, n_bins=2) == pytest.approx(0.15)


def test_expected_calibration_error_is_zero_for_perfectly_calibrated_bins():
    # Every example's probability equals its own label exactly (0 or 1) ->
    # each bin's mean confidence equals its mean accuracy exactly.
    probabilities = [0.0, 0.0, 1.0, 1.0]
    labels = [0, 0, 1, 1]
    assert expected_calibration_error(probabilities, labels, n_bins=10) == pytest.approx(0.0)


def test_expected_calibration_error_raises_on_empty_input():
    with pytest.raises(EmptyCalibrationSetError):
        expected_calibration_error([], [])


def test_expected_calibration_error_raises_on_non_positive_n_bins():
    with pytest.raises(ValueError):
        expected_calibration_error([0.5], [1], n_bins=0)


def test_expected_calibration_error_default_n_bins_is_fifteen():
    import inspect

    signature = inspect.signature(expected_calibration_error)
    assert signature.parameters["n_bins"].default == 15


def test_expected_calibration_error_probability_exactly_one_assigned_to_last_bin():
    # Regression guard: p == 1.0 must not overflow into a non-existent bin.
    probabilities = [1.0]
    labels = [1]
    assert expected_calibration_error(probabilities, labels, n_bins=4) == pytest.approx(0.0)
