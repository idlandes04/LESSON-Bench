"""Tests for lesson/eval/analysis.py — summary statistics and FLR estimation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import io
import pytest
from unittest.mock import patch

from lesson.eval.analysis import (
    _mean,
    _linreg_slope,
    estimate_flr,
    summarize_sb2_results,
)


# ---------------------------------------------------------------------------
# _mean
# ---------------------------------------------------------------------------

def test_mean_empty_list_returns_zero():
    assert _mean([]) == 0.0


def test_mean_single_value():
    assert _mean([0.75]) == pytest.approx(0.75)


def test_mean_multiple_values():
    assert _mean([0.0, 0.5, 1.0]) == pytest.approx(0.5)


def test_mean_all_ones():
    assert _mean([1.0, 1.0, 1.0]) == pytest.approx(1.0)


def test_mean_all_zeros():
    assert _mean([0.0, 0.0, 0.0]) == pytest.approx(0.0)


def test_mean_floats():
    assert _mean([0.3, 0.6, 0.9]) == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# _linreg_slope
# ---------------------------------------------------------------------------

def test_linreg_slope_perfect_positive():
    # y = x => slope = 1
    xs = [1.0, 2.0, 3.0, 4.0]
    ys = [1.0, 2.0, 3.0, 4.0]
    assert _linreg_slope(xs, ys) == pytest.approx(1.0)


def test_linreg_slope_perfect_negative():
    xs = [1.0, 2.0, 3.0]
    ys = [3.0, 2.0, 1.0]
    assert _linreg_slope(xs, ys) == pytest.approx(-1.0)


def test_linreg_slope_flat_line_returns_zero():
    xs = [1.0, 2.0, 3.0, 4.0]
    ys = [0.5, 0.5, 0.5, 0.5]
    assert _linreg_slope(xs, ys) == pytest.approx(0.0)


def test_linreg_slope_single_point_returns_zero():
    assert _linreg_slope([5.0], [3.0]) == pytest.approx(0.0)


def test_linreg_slope_all_same_x_returns_zero():
    # Zero variance in x → denominator is 0 → returns 0.0
    assert _linreg_slope([2.0, 2.0, 2.0], [1.0, 2.0, 3.0]) == pytest.approx(0.0)


def test_linreg_slope_empty_returns_zero():
    assert _linreg_slope([], []) == pytest.approx(0.0)


def test_linreg_slope_known_value():
    # y = 2x + 1 => slope = 2
    xs = [0.0, 1.0, 2.0, 3.0]
    ys = [1.0, 3.0, 5.0, 7.0]
    assert _linreg_slope(xs, ys) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# estimate_flr
# ---------------------------------------------------------------------------

def _make_results(turns_correct: dict, condition: str) -> list:
    """Build a list of result dicts from {turn: n_correct_out_of_3} mapping."""
    rows = []
    n_subjects = 3
    for turn, n_correct in turns_correct.items():
        for i in range(n_subjects):
            rows.append({
                "condition": condition,
                "turn": turn,
                "correct": i < n_correct,
            })
    return rows


def test_estimate_flr_positive():
    # Correction improves (0% → 100%), practice stays flat (50% → 50%)
    correction = _make_results({1: 0, 2: 1, 3: 2, 4: 3}, "correction")
    practice = _make_results({1: 1, 2: 1, 3: 2, 4: 2}, "practice_only")
    flr = estimate_flr(correction, practice)
    # Correction slope is positive, practice slope is ~0 → FLR > 0
    assert flr > 0.0


def test_estimate_flr_zero_when_both_flat():
    correction = _make_results({1: 2, 2: 2, 3: 2, 4: 2}, "correction")
    practice = _make_results({1: 2, 2: 2, 3: 2, 4: 2}, "practice_only")
    flr = estimate_flr(correction, practice)
    assert flr == pytest.approx(0.0)


def test_estimate_flr_empty_correction():
    # With no correction data the slope for correction is 0.0;
    # FLR = 0 - slope(practice).  Practice has turns 1,2,3 with rising accuracy
    # so FLR should be negative (0 minus a positive slope).
    practice = _make_results({1: 0, 2: 1, 3: 3}, "practice_only")
    flr = estimate_flr([], practice)
    # slope(correction)=0, slope(practice)>0 → FLR < 0
    assert flr <= 0.0


def test_estimate_flr_empty_practice():
    # With no practice data the slope for practice is 0.0;
    # FLR = slope(correction) - 0. Correction is rising so FLR > 0.
    correction = _make_results({1: 0, 2: 1, 3: 3}, "correction")
    flr = estimate_flr(correction, [])
    # slope(practice)=0, slope(correction)>0 → FLR > 0
    assert flr >= 0.0


def test_estimate_flr_single_turn_each_returns_zero():
    # Only one turn in each condition — can't compute a slope
    correction = [{"condition": "correction", "turn": 1, "correct": True}]
    practice = [{"condition": "practice_only", "turn": 1, "correct": False}]
    flr = estimate_flr(correction, practice)
    assert flr == pytest.approx(0.0)


def test_estimate_flr_symmetric():
    # If we swap correction/practice for a perfectly symmetric case, FLR flips sign
    rising = _make_results({1: 0, 2: 1, 3: 2, 4: 3}, "rising")
    flat = _make_results({1: 2, 2: 2, 3: 2, 4: 2}, "flat")
    flr_pos = estimate_flr(rising, flat)
    flr_neg = estimate_flr(flat, rising)
    assert flr_pos == pytest.approx(-flr_neg, abs=1e-9)


# ---------------------------------------------------------------------------
# summarize_sb2_results — bug regression test
# ---------------------------------------------------------------------------

def test_summarize_sb2_results_uses_practice_only_condition():
    """Regression test: summarize_sb2_results() must use 'practice_only' as the
    condition name when filtering rows to compute FLR.

    The current implementation on line 188 of analysis.py filters on the string
    'practice' instead of 'practice_only', so no practice rows are matched and the
    FLR line reads "insufficient data" even when practice_only data is present.
    """
    results = []
    for turn in range(1, 5):
        for i in range(3):
            results.append({
                "condition": "correction",
                "turn": turn,
                "correct": turn > 2,
                "model": "test-model",
            })
            results.append({
                "condition": "practice_only",
                "turn": turn,
                "correct": False,
                "model": "test-model",
            })

    captured = io.StringIO()
    with patch("builtins.print", lambda *args, **kwargs: captured.write(" ".join(str(a) for a in args) + "\n")):
        summarize_sb2_results(results)

    output = captured.getvalue()
    # The bug causes this assertion to fail: the output says "insufficient data"
    # because it filters on "practice" and finds no rows (condition is "practice_only").
    assert "insufficient data" not in output, (
        "FLR estimate was skipped — 'practice_only' condition was not matched. "
        "Fix: change line 188 to filter on 'practice_only' instead of 'practice'."
    )
    assert "FLR estimate:" in output
