"""Tests for lesson/eval/stats.py — statistical analysis utilities.

Requires numpy — the whole module is skipped if numpy is not installed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

# Skip entire module if numpy is not available
np = pytest.importorskip("numpy")

from lesson.eval.stats import (
    bootstrap_ci,
    compute_aulc,
    compute_htr,
    compute_rii,
    factorial_decomposition,
    rank_correlation,
    _slope_from_turn_accuracies,
)


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------

def test_bootstrap_ci_known_values():
    # All-same values → CI is a point
    values = [1.0] * 100
    point, lo, hi = bootstrap_ci(values, seed=0)
    assert point == pytest.approx(1.0)
    assert lo == pytest.approx(1.0)
    assert hi == pytest.approx(1.0)


def test_bootstrap_ci_point_estimate_correct():
    values = [0.0, 0.5, 1.0]
    point, lo, hi = bootstrap_ci(values, seed=42)
    assert point == pytest.approx(0.5)


def test_bootstrap_ci_interval_ordered():
    values = [float(i) / 10 for i in range(11)]
    point, lo, hi = bootstrap_ci(values, seed=42)
    assert lo <= point <= hi


def test_bootstrap_ci_empty_list():
    point, lo, hi = bootstrap_ci([], seed=42)
    assert point == pytest.approx(0.0)
    assert lo == pytest.approx(0.0)
    assert hi == pytest.approx(0.0)


def test_bootstrap_ci_single_value():
    point, lo, hi = bootstrap_ci([0.75], seed=42)
    assert point == pytest.approx(0.75)
    assert lo == pytest.approx(0.75)
    assert hi == pytest.approx(0.75)


def test_bootstrap_ci_wider_for_more_variable_data():
    flat = [0.5] * 50
    noisy = [0.0 if i < 25 else 1.0 for i in range(50)]
    _, lo_flat, hi_flat = bootstrap_ci(flat, seed=0)
    _, lo_noisy, hi_noisy = bootstrap_ci(noisy, seed=0)
    width_flat = hi_flat - lo_flat
    width_noisy = hi_noisy - lo_noisy
    assert width_noisy > width_flat


# ---------------------------------------------------------------------------
# compute_aulc
# ---------------------------------------------------------------------------

def test_aulc_two_points_trapezoidal():
    # Trapezoid from n=4 (acc=0.2) to n=8 (acc=0.6):
    # area = (8-4) * (0.2 + 0.6) / 2 = 4 * 0.4 = 1.6; range=4; AULC = 0.4
    result = compute_aulc({4: 0.2, 8: 0.6})
    assert result == pytest.approx(0.4)


def test_aulc_single_point_returns_that_value():
    result = compute_aulc({8: 0.75})
    assert result == pytest.approx(0.75)


def test_aulc_empty_returns_zero():
    result = compute_aulc({})
    assert result == pytest.approx(0.0)


def test_aulc_uniform_accuracy_equals_that_value():
    # All points at 0.6 → AULC should be 0.6
    result = compute_aulc({2: 0.6, 4: 0.6, 8: 0.6, 16: 0.6})
    assert result == pytest.approx(0.6)


def test_aulc_respects_n_values_argument():
    # Provide only a subset of n_values
    result = compute_aulc({4: 0.2, 8: 0.6, 16: 0.9}, n_values=[4, 8])
    # Should only use n=4 and n=8
    expected = (0.2 + 0.6) / 2.0
    assert result == pytest.approx(expected)


# ---------------------------------------------------------------------------
# compute_rii
# ---------------------------------------------------------------------------

def test_compute_rii_normal_case():
    # Type E accuracy 0.6, Type R accuracy 0.8 → RII = 0.75
    result = compute_rii(type_r_accuracy=0.8, type_e_accuracy=0.6)
    assert result == pytest.approx(0.75)


def test_compute_rii_zero_denominator_returns_none():
    result = compute_rii(type_r_accuracy=0.0, type_e_accuracy=0.5)
    assert result is None


def test_compute_rii_equal_accuracies_returns_one():
    result = compute_rii(type_r_accuracy=0.5, type_e_accuracy=0.5)
    assert result == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# compute_htr
# ---------------------------------------------------------------------------

def test_compute_htr_normal_case():
    # Type L accuracy 0.2, Type R accuracy 0.8 → HTR = 1 - 0.25 = 0.75
    result = compute_htr(type_r_accuracy=0.8, type_l_accuracy=0.2)
    assert result == pytest.approx(0.75)


def test_compute_htr_zero_denominator_returns_none():
    result = compute_htr(type_r_accuracy=0.0, type_l_accuracy=0.3)
    assert result is None


def test_compute_htr_equal_accuracies_returns_zero():
    # Model falls for every lure → HTR = 0
    result = compute_htr(type_r_accuracy=0.6, type_l_accuracy=0.6)
    assert result == pytest.approx(0.0)


def test_compute_htr_zero_lure_returns_one():
    # Model resists every lure → HTR = 1
    result = compute_htr(type_r_accuracy=0.8, type_l_accuracy=0.0)
    assert result == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# factorial_decomposition
# ---------------------------------------------------------------------------

def test_factorial_decomposition_known_values():
    # Symmetric baseline: no_feedback=0, practice_only=0.2, error_only=0.2, correction=0.4
    means = {
        "no_feedback": 0.0,
        "practice_only": 0.2,
        "error_only": 0.2,
        "correction": 0.4,
    }
    effects = factorial_decomposition(means)
    # answer_effect = (0.2 + 0.4)/2 - (0.0 + 0.2)/2 = 0.3 - 0.1 = 0.2
    assert effects["answer_effect"] == pytest.approx(0.2)
    # evaluation_effect = (0.2 + 0.4)/2 - (0.0 + 0.2)/2 = 0.3 - 0.1 = 0.2
    assert effects["evaluation_effect"] == pytest.approx(0.2)
    # interaction = 0.4 - 0.2 - 0.2 + 0.0 = 0.0
    assert effects["interaction"] == pytest.approx(0.0)


def test_factorial_decomposition_pure_answer_effect():
    # Only knowing the answer (practice) helps, not error signal
    means = {
        "no_feedback": 0.1,
        "practice_only": 0.5,
        "error_only": 0.1,
        "correction": 0.5,
    }
    effects = factorial_decomposition(means)
    # answer_effect = (0.5 + 0.5)/2 - (0.1 + 0.1)/2 = 0.5 - 0.1 = 0.4
    assert effects["answer_effect"] == pytest.approx(0.4)
    # evaluation_effect = (0.1 + 0.5)/2 - (0.1 + 0.5)/2 = 0.0
    assert effects["evaluation_effect"] == pytest.approx(0.0)


def test_factorial_decomposition_interaction_formula():
    # interaction = correction - practice_only - error_only + no_feedback
    means = {
        "no_feedback": 0.1,
        "practice_only": 0.3,
        "error_only": 0.4,
        "correction": 0.9,
    }
    effects = factorial_decomposition(means)
    expected_interaction = 0.9 - 0.3 - 0.4 + 0.1
    assert effects["interaction"] == pytest.approx(expected_interaction)


def test_factorial_decomposition_all_zeros():
    means = {"no_feedback": 0.0, "practice_only": 0.0, "error_only": 0.0, "correction": 0.0}
    effects = factorial_decomposition(means)
    assert effects["answer_effect"] == pytest.approx(0.0)
    assert effects["evaluation_effect"] == pytest.approx(0.0)
    assert effects["interaction"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _slope_from_turn_accuracies
# ---------------------------------------------------------------------------

def test_slope_positive():
    turn_accs = {0: 0.0, 1: 0.25, 2: 0.5, 3: 0.75, 4: 1.0}
    slope = _slope_from_turn_accuracies(turn_accs)
    assert slope == pytest.approx(0.25)


def test_slope_flat():
    turn_accs = {0: 0.5, 1: 0.5, 2: 0.5}
    slope = _slope_from_turn_accuracies(turn_accs)
    assert slope == pytest.approx(0.0)


def test_slope_single_point_returns_zero():
    slope = _slope_from_turn_accuracies({0: 0.8})
    assert slope == pytest.approx(0.0)


def test_slope_empty_returns_zero():
    slope = _slope_from_turn_accuracies({})
    assert slope == pytest.approx(0.0)


def test_slope_negative():
    turn_accs = {0: 1.0, 1: 0.5, 2: 0.0}
    slope = _slope_from_turn_accuracies(turn_accs)
    assert slope < 0.0


# ---------------------------------------------------------------------------
# rank_correlation
# ---------------------------------------------------------------------------

def test_rank_correlation_perfect_positive():
    model_metrics = {
        f"model_{i}": {"aulc": float(i), "flr": float(i)}
        for i in range(1, 8)
    }
    rho, p = rank_correlation(model_metrics, "aulc", "flr")
    assert rho == pytest.approx(1.0, abs=1e-9)


def test_rank_correlation_perfect_negative():
    n = 7
    model_metrics = {
        f"model_{i}": {"aulc": float(i), "flr": float(n - i)}
        for i in range(1, n + 1)
    }
    rho, p = rank_correlation(model_metrics, "aulc", "flr")
    assert rho == pytest.approx(-1.0, abs=1e-9)


def test_rank_correlation_fewer_than_three_models_returns_zero():
    model_metrics = {
        "m1": {"aulc": 0.5, "flr": 0.3},
        "m2": {"aulc": 0.8, "flr": 0.7},
    }
    rho, p = rank_correlation(model_metrics, "aulc", "flr")
    assert rho == pytest.approx(0.0)
    assert p == pytest.approx(1.0)


def test_rank_correlation_skips_missing_metrics():
    model_metrics = {
        "m1": {"aulc": 0.5, "flr": 0.3},
        "m2": {"aulc": 0.8},        # missing flr
        "m3": {"aulc": 0.2, "flr": 0.1},
        "m4": {"aulc": 0.9, "flr": 0.8},
        "m5": {"aulc": 0.6, "flr": 0.5},
    }
    # Should use m1, m3, m4, m5 (4 models with both metrics)
    rho, p = rank_correlation(model_metrics, "aulc", "flr")
    # All four are positively correlated
    assert rho > 0
