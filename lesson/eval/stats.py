"""Statistical analysis module for LESSON-Bench production results.

Provides bootstrap confidence intervals, 2x2 factorial decomposition,
FLR estimation, learning curve metrics, and cross-model analysis.

Uses numpy for computation. All CI functions use bootstrap resampling
at the instance level (instances are the independent unit, not individual
turns or items).
"""

from __future__ import annotations

import math
import random as _random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Bootstrap utilities
# ---------------------------------------------------------------------------

def bootstrap_ci(
    values: List[float],
    statistic: str = "mean",
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval.

    Args:
        values: Raw observations (one per independent unit, e.g., per instance).
        statistic: "mean" or "proportion" (same computation, semantic difference).
        n_boot: Number of bootstrap resamples.
        ci: Confidence level (default 0.95).
        seed: Random seed for reproducibility.

    Returns:
        (point_estimate, ci_lower, ci_upper)
    """
    if not values:
        return (0.0, 0.0, 0.0)

    arr = np.asarray(values, dtype=np.float64)
    point = float(np.mean(arr))
    n = len(arr)

    rng = np.random.RandomState(seed)
    boot_means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        sample = arr[rng.randint(0, n, size=n)]
        boot_means[i] = np.mean(sample)

    alpha = 1.0 - ci
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1.0 - alpha / 2)))
    return (point, lo, hi)


def bootstrap_ci_difference(
    values_a: List[float],
    values_b: List[float],
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Bootstrap CI for the difference of means (A - B).

    Resamples paired observations (same indices must correspond to same instances).

    Returns:
        (point_diff, ci_lower, ci_upper)
    """
    if not values_a or not values_b:
        return (0.0, 0.0, 0.0)

    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    n = min(len(a), len(b))
    a = a[:n]
    b = b[:n]
    diff = a - b
    point = float(np.mean(diff))

    rng = np.random.RandomState(seed)
    boot_diffs = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        boot_diffs[i] = np.mean(diff[idx])

    alpha = 1.0 - ci
    lo = float(np.percentile(boot_diffs, 100 * alpha / 2))
    hi = float(np.percentile(boot_diffs, 100 * (1.0 - alpha / 2)))
    return (point, lo, hi)


# ---------------------------------------------------------------------------
# SB1 metrics
# ---------------------------------------------------------------------------

def compute_aulc(
    accuracies_by_n: Dict[int, float],
    n_values: Optional[List[int]] = None,
) -> float:
    """Area Under the Learning Curve (normalized to [0, 1]).

    Args:
        accuracies_by_n: {N: accuracy} mapping (e.g., {4: 0.25, 8: 0.50, 16: 0.65}).
        n_values: If provided, use these N values for integration. Otherwise use all keys.

    Returns:
        Normalized AULC (trapezoidal integration, divided by range).
    """
    if n_values is None:
        n_values = sorted(accuracies_by_n.keys())
    else:
        n_values = sorted(n_values)

    if len(n_values) < 2:
        if len(n_values) == 1 and n_values[0] in accuracies_by_n:
            return accuracies_by_n[n_values[0]]
        return 0.0

    xs = [float(n) for n in n_values]
    ys = [accuracies_by_n.get(n, 0.0) for n in n_values]

    # Trapezoidal integration
    area = 0.0
    for i in range(len(xs) - 1):
        area += (xs[i + 1] - xs[i]) * (ys[i] + ys[i + 1]) / 2.0

    n_range = xs[-1] - xs[0]
    if n_range == 0.0:
        return 0.0
    return area / n_range


def compute_rii(
    type_r_accuracy: float,
    type_e_accuracy: float,
) -> Optional[float]:
    """Rule Induction Index: ratio of Type E to Type R accuracy.

    Returns None if type_r_accuracy is 0 (avoid division by zero).
    RII > 1 means model extrapolates better than it interpolates (unlikely).
    RII ~ 1 means genuine rule induction.
    RII < 1 means model relies on surface patterns.
    """
    if type_r_accuracy == 0.0:
        return None
    return type_e_accuracy / type_r_accuracy


def compute_htr(
    type_r_accuracy: float,
    type_l_accuracy: float,
) -> Optional[float]:
    """Hypothesis Testing Ratio.

    HTR = 1 - (Type L accuracy / Type R accuracy)
    HTR ~ 0 means model falls for lures (follows partial rules).
    HTR ~ 1 means model resists lures (has full rule understanding).
    Returns None if type_r_accuracy is 0.
    """
    if type_r_accuracy == 0.0:
        return None
    return 1.0 - (type_l_accuracy / type_r_accuracy)


# ---------------------------------------------------------------------------
# SB2 metrics
# ---------------------------------------------------------------------------

def _group_by_instance(results: List[Dict]) -> Dict[Any, List[Dict]]:
    """Group result dicts by instance_idx."""
    groups: Dict[Any, List[Dict]] = defaultdict(list)
    for r in results:
        groups[r["instance_idx"]].append(r)
    return dict(groups)


def _slope_from_turn_accuracies(
    turn_accuracies: Dict[int, float],
) -> float:
    """Compute least-squares slope from {turn: accuracy} mapping."""
    if len(turn_accuracies) < 2:
        return 0.0
    turns = sorted(turn_accuracies.keys())
    xs = np.array([float(t) for t in turns])
    ys = np.array([turn_accuracies[t] for t in turns])
    mean_x = np.mean(xs)
    mean_y = np.mean(ys)
    denom = float(np.sum((xs - mean_x) ** 2))
    if denom == 0.0:
        return 0.0
    return float(np.sum((xs - mean_x) * (ys - mean_y)) / denom)


def _compute_per_turn_accuracy(
    results: List[Dict],
) -> Dict[int, float]:
    """Compute per-turn accuracy from a flat list of result dicts."""
    turn_buckets: Dict[int, List[float]] = defaultdict(list)
    for r in results:
        turn_buckets[r["turn_idx"]].append(1.0 if r.get("correct") else 0.0)
    return {t: float(np.mean(vals)) for t, vals in turn_buckets.items()}


def compute_flr_with_ci(
    correction_results: List[Dict],
    practice_results: List[Dict],
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """FLR with bootstrap CI.

    FLR = slope(correction accuracy over turns) - slope(practice accuracy over turns)

    Bootstrap resamples at the instance level: for each resample, draw instances
    with replacement, recompute per-turn accuracies, compute slopes, take difference.

    Args:
        correction_results: List of result dicts with keys: instance_idx, turn_idx, correct
        practice_results: Same structure for practice_only condition.

    Returns:
        (flr_point, ci_lower, ci_upper)
    """
    # Point estimate
    corr_turn_acc = _compute_per_turn_accuracy(correction_results)
    prac_turn_acc = _compute_per_turn_accuracy(practice_results)
    flr_point = _slope_from_turn_accuracies(corr_turn_acc) - _slope_from_turn_accuracies(prac_turn_acc)

    # Group by instance for bootstrap
    corr_by_inst = _group_by_instance(correction_results)
    prac_by_inst = _group_by_instance(practice_results)
    corr_instances = list(corr_by_inst.values())
    prac_instances = list(prac_by_inst.values())

    n_corr = len(corr_instances)
    n_prac = len(prac_instances)
    if n_corr == 0 or n_prac == 0:
        return (flr_point, flr_point, flr_point)

    rng = np.random.RandomState(seed)
    boot_flrs = np.empty(n_boot, dtype=np.float64)

    for b in range(n_boot):
        # Resample instances with replacement
        corr_idx = rng.randint(0, n_corr, size=n_corr)
        prac_idx = rng.randint(0, n_prac, size=n_prac)

        corr_sample: List[Dict] = []
        for i in corr_idx:
            corr_sample.extend(corr_instances[i])
        prac_sample: List[Dict] = []
        for i in prac_idx:
            prac_sample.extend(prac_instances[i])

        corr_ta = _compute_per_turn_accuracy(corr_sample)
        prac_ta = _compute_per_turn_accuracy(prac_sample)
        boot_flrs[b] = _slope_from_turn_accuracies(corr_ta) - _slope_from_turn_accuracies(prac_ta)

    alpha = 1.0 - ci
    lo = float(np.percentile(boot_flrs, 100 * alpha / 2))
    hi = float(np.percentile(boot_flrs, 100 * (1.0 - alpha / 2)))
    return (flr_point, lo, hi)


def compute_condition_trajectory(
    results: List[Dict],
    n_turns: int,
) -> Dict[int, Tuple[float, float, float]]:
    """Per-turn accuracy with bootstrap CIs for a single condition.

    Groups results by turn_idx, computes accuracy per instance per turn,
    then bootstraps across instances.

    Returns:
        {turn_idx: (accuracy, ci_lower, ci_upper)}
    """
    by_instance = _group_by_instance(results)
    instance_ids = list(by_instance.keys())

    trajectory: Dict[int, Tuple[float, float, float]] = {}

    for turn in range(n_turns):
        # Collect one accuracy value per instance for this turn
        instance_accs: List[float] = []
        for inst_id in instance_ids:
            inst_results = by_instance[inst_id]
            turn_items = [r for r in inst_results if r.get("turn_idx") == turn]
            if turn_items:
                acc = sum(1.0 if r.get("correct") else 0.0 for r in turn_items) / len(turn_items)
                instance_accs.append(acc)

        if instance_accs:
            trajectory[turn] = bootstrap_ci(instance_accs)
        else:
            trajectory[turn] = (0.0, 0.0, 0.0)

    return trajectory


# ---------------------------------------------------------------------------
# 2x2 Factorial decomposition
# ---------------------------------------------------------------------------

def factorial_decomposition(
    condition_means: Dict[str, float],
) -> Dict[str, float]:
    """Compute 2x2 factorial effects from condition means.

    Expected keys in condition_means:
        "no_feedback", "practice_only", "error_only", "correction"

    Returns:
        {
            "answer_effect": mean(practice_only, correction) - mean(no_feedback, error_only),
            "evaluation_effect": mean(error_only, correction) - mean(no_feedback, practice_only),
            "interaction": correction - practice_only - error_only + no_feedback,
        }
    """
    nf = condition_means["no_feedback"]
    po = condition_means["practice_only"]
    eo = condition_means["error_only"]
    co = condition_means["correction"]

    answer_effect = (po + co) / 2.0 - (nf + eo) / 2.0
    evaluation_effect = (eo + co) / 2.0 - (nf + po) / 2.0
    interaction = co - po - eo + nf

    return {
        "answer_effect": answer_effect,
        "evaluation_effect": evaluation_effect,
        "interaction": interaction,
    }


def factorial_decomposition_with_ci(
    results_by_condition: Dict[str, List[Dict]],
    n_boot: int = 2000,
    seed: int = 42,
) -> Dict[str, Tuple[float, float, float]]:
    """2x2 factorial effects with bootstrap CIs.

    Bootstrap resamples instances, recomputes all condition means, then derives effects.

    Args:
        results_by_condition: {"correction": [...], "practice_only": [...], ...}
            Each list contains dicts with instance_idx, turn_idx, correct.

    Returns:
        {"answer_effect": (point, lo, hi), "evaluation_effect": (point, lo, hi),
         "interaction": (point, lo, hi)}
    """
    required = ["no_feedback", "practice_only", "error_only", "correction"]
    for key in required:
        if key not in results_by_condition:
            raise ValueError(f"Missing condition: {key}")

    # Point estimate
    condition_means: Dict[str, float] = {}
    for cond in required:
        vals = [1.0 if r.get("correct") else 0.0 for r in results_by_condition[cond]]
        condition_means[cond] = float(np.mean(vals)) if vals else 0.0
    point_effects = factorial_decomposition(condition_means)

    # Group each condition by instance
    cond_instances: Dict[str, List[List[Dict]]] = {}
    for cond in required:
        by_inst = _group_by_instance(results_by_condition[cond])
        cond_instances[cond] = list(by_inst.values())

    rng = np.random.RandomState(seed)
    effect_names = ["answer_effect", "evaluation_effect", "interaction"]
    boot_effects: Dict[str, List[float]] = {e: [] for e in effect_names}

    for _ in range(n_boot):
        boot_means: Dict[str, float] = {}
        for cond in required:
            instances = cond_instances[cond]
            n_inst = len(instances)
            if n_inst == 0:
                boot_means[cond] = 0.0
                continue
            idx = rng.randint(0, n_inst, size=n_inst)
            resampled: List[float] = []
            for i in idx:
                items = instances[i]
                resampled.extend(1.0 if r.get("correct") else 0.0 for r in items)
            boot_means[cond] = float(np.mean(resampled)) if resampled else 0.0

        effects = factorial_decomposition(boot_means)
        for e in effect_names:
            boot_effects[e].append(effects[e])

    result: Dict[str, Tuple[float, float, float]] = {}
    for e in effect_names:
        arr = np.array(boot_effects[e])
        lo = float(np.percentile(arr, 2.5))
        hi = float(np.percentile(arr, 97.5))
        result[e] = (point_effects[e], lo, hi)

    return result


# ---------------------------------------------------------------------------
# Cross-model analysis
# ---------------------------------------------------------------------------

def compute_model_profile(
    sb1_results: List[Dict],
    sb2_results_by_condition: Dict[str, List[Dict]],
    n_turns: int = 12,
) -> Dict[str, Any]:
    """Compute full cognitive profile for a single model.

    Returns dict with:
        - aulc: float (from SB1)
        - rii: float or None (from SB1 Type E vs R)
        - htr: float or None (from SB1 Type L vs R)
        - flr: (point, lo, hi) (from SB2)
        - condition_trajectories: {condition: {turn: (acc, lo, hi)}}
        - factorial: {effect: (point, lo, hi)} (if all 4 core conditions present)
    """
    profile: Dict[str, Any] = {}

    # --- SB1: AULC ---
    # Group by n, compute accuracy at each n
    # SB1 results use "n_examples" key (from pilot.py)
    n_buckets: Dict[int, List[float]] = defaultdict(list)
    for r in sb1_results:
        n_key = r.get("n_examples") or r.get("n")
        if n_key is not None:
            n_buckets[n_key].append(1.0 if r.get("correct") else 0.0)
    accuracies_by_n = {n: float(np.mean(vals)) for n, vals in n_buckets.items()}
    profile["aulc"] = compute_aulc(accuracies_by_n)

    # --- SB1: RII (Type E vs Type R) ---
    type_r = [r for r in sb1_results if r.get("item_type") == "R"]
    type_e = [r for r in sb1_results if r.get("item_type") == "E"]
    type_r_acc = float(np.mean([1.0 if r.get("correct") else 0.0 for r in type_r])) if type_r else 0.0
    type_e_acc = float(np.mean([1.0 if r.get("correct") else 0.0 for r in type_e])) if type_e else 0.0
    profile["rii"] = compute_rii(type_r_acc, type_e_acc)

    # --- SB1: HTR (Type L vs Type R) ---
    type_l = [r for r in sb1_results if r.get("item_type") == "L"]
    type_l_acc = float(np.mean([1.0 if r.get("correct") else 0.0 for r in type_l])) if type_l else 0.0
    profile["htr"] = compute_htr(type_r_acc, type_l_acc)

    # --- SB2: FLR ---
    corr = sb2_results_by_condition.get("correction", [])
    prac = sb2_results_by_condition.get("practice_only", [])
    if corr and prac:
        profile["flr"] = compute_flr_with_ci(corr, prac)
    else:
        profile["flr"] = (0.0, 0.0, 0.0)

    # --- SB2: condition trajectories ---
    trajectories: Dict[str, Dict[int, Tuple[float, float, float]]] = {}
    for cond, cond_results in sb2_results_by_condition.items():
        if cond_results:
            trajectories[cond] = compute_condition_trajectory(cond_results, n_turns)
    profile["condition_trajectories"] = trajectories

    # --- SB2: factorial decomposition (only if all 4 conditions present) ---
    factorial_conditions = {"no_feedback", "practice_only", "error_only", "correction"}
    if factorial_conditions.issubset(sb2_results_by_condition.keys()):
        profile["factorial"] = factorial_decomposition_with_ci(sb2_results_by_condition)
    else:
        profile["factorial"] = None

    return profile


def rank_correlation(
    model_metrics: Dict[str, Dict[str, float]],
    metric_a: str,
    metric_b: str,
) -> Tuple[float, float]:
    """Spearman rank correlation between two metrics across models.

    Args:
        model_metrics: {model_name: {metric_name: value}}
        metric_a, metric_b: Names of metrics to correlate.

    Returns:
        (rho, p_value) -- computed manually without scipy dependency.
    """
    # Collect paired values, skipping models with missing metrics
    pairs: List[Tuple[float, float]] = []
    for model in model_metrics:
        m = model_metrics[model]
        if metric_a in m and metric_b in m:
            va = m[metric_a]
            vb = m[metric_b]
            if va is not None and vb is not None:
                pairs.append((va, vb))

    n = len(pairs)
    if n < 3:
        return (0.0, 1.0)

    def _rank(vals: List[float]) -> List[float]:
        """Assign ranks with tie-averaging."""
        indexed = sorted(enumerate(vals), key=lambda x: x[1])
        ranks = [0.0] * len(vals)
        i = 0
        while i < len(indexed):
            j = i
            while j < len(indexed) and indexed[j][1] == indexed[i][1]:
                j += 1
            avg_rank = (i + 1 + j) / 2.0  # 1-based average rank
            for k in range(i, j):
                ranks[indexed[k][0]] = avg_rank
            i = j
        return ranks

    a_vals = [p[0] for p in pairs]
    b_vals = [p[1] for p in pairs]
    ranks_a = _rank(a_vals)
    ranks_b = _rank(b_vals)

    # Spearman rho as Pearson of ranks
    ra = np.array(ranks_a)
    rb = np.array(ranks_b)
    mean_a = np.mean(ra)
    mean_b = np.mean(rb)
    da = ra - mean_a
    db = rb - mean_b
    num = float(np.sum(da * db))
    denom = math.sqrt(float(np.sum(da ** 2)) * float(np.sum(db ** 2)))
    if denom == 0.0:
        return (0.0, 1.0)
    rho = num / denom

    # p-value approximation via t-distribution (normal approx for large n)
    if abs(rho) >= 1.0:
        p_value = 0.0
    else:
        t_stat = rho * math.sqrt((n - 2) / (1.0 - rho ** 2))
        # Two-tailed p-value using normal approximation (valid for n >= ~10,
        # reasonable approximation for smaller n)
        # CDF of standard normal approximated via error function
        p_value = 2.0 * _normal_sf(abs(t_stat))

    return (float(rho), p_value)


def _normal_sf(x: float) -> float:
    """Survival function (1 - CDF) of the standard normal distribution.

    Uses the complementary error function from math for accuracy.
    """
    return 0.5 * math.erfc(x / math.sqrt(2.0))


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fmt_ci(point: float, lo: float, hi: float, pct: bool = True) -> str:
    """Format a point estimate with CI as string.

    Example: "42.0% [35.1%, 49.2%]" or "0.42 [0.35, 0.49]"
    """
    if pct:
        return f"{point * 100:.1f}% [{lo * 100:.1f}%, {hi * 100:.1f}%]"
    else:
        return f"{point:.2f} [{lo:.2f}, {hi:.2f}]"


def results_to_instance_accuracies(
    results: List[Dict],
    group_by: str = "instance_idx",
) -> List[float]:
    """Convert flat results list to per-instance accuracy values.

    Groups by group_by key, computes accuracy within each group.
    Returns list of accuracies (one per instance).
    """
    groups: Dict[Any, List[float]] = defaultdict(list)
    for r in results:
        key = r.get(group_by)
        if key is not None:
            groups[key].append(1.0 if r.get("correct") else 0.0)

    return [float(np.mean(vals)) for vals in groups.values()]
