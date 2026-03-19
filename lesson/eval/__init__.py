from __future__ import annotations

"""Evaluation pipeline for LESSON benchmarks.

Provides answer extraction utilities and pilot runners for SB1 and SB2.
"""

from .extraction import extract_answer, extract_answer_regex, extract_answer_json
from .pilot import run_sb1_pilot
from .sb2_pilot import run_sb2_pilot, CORE_CONDITIONS, ALL_CONDITIONS
from .runner import CircuitBreaker, retry_with_backoff, get_completed_cells, filter_incomplete
from .stats import (
    bootstrap_ci,
    bootstrap_ci_difference,
    compute_aulc,
    compute_rii,
    compute_htr,
    compute_flr_with_ci,
    compute_condition_trajectory,
    factorial_decomposition,
    factorial_decomposition_with_ci,
    compute_model_profile,
    rank_correlation,
    fmt_ci,
    results_to_instance_accuracies,
)

__all__ = [
    "extract_answer",
    "extract_answer_regex",
    "extract_answer_json",
    "run_sb1_pilot",
    "run_sb2_pilot",
    # Statistical analysis
    "bootstrap_ci",
    "bootstrap_ci_difference",
    "compute_aulc",
    "compute_rii",
    "compute_htr",
    "compute_flr_with_ci",
    "compute_condition_trajectory",
    "factorial_decomposition",
    "factorial_decomposition_with_ci",
    "compute_model_profile",
    "rank_correlation",
    "fmt_ci",
    "results_to_instance_accuracies",
    # Condition constants
    "CORE_CONDITIONS",
    "ALL_CONDITIONS",
    # Resilience
    "CircuitBreaker",
    "retry_with_backoff",
    "get_completed_cells",
    "filter_incomplete",
]
