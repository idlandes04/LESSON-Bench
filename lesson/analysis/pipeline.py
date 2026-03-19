"""Data loading and computation layer for the LESSON-Bench analysis pipeline.

Reads from the SQLite results DB and returns structured data ready for
plotting by figures.py or export by report.py.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from lesson.eval.stats import (
    bootstrap_ci,
    bootstrap_ci_difference,
    compute_flr_with_ci,
    factorial_decomposition,
    factorial_decomposition_with_ci,
    _slope_from_turn_accuracies,
)
from lesson.results.store import ResultsStore, DEFAULT_DB

CORE_CONDITIONS = ["correction", "practice_only", "error_only", "no_feedback"]

# Model metadata for grouping analyses.
# training_type used for box plots; family for within-family comparisons.
MODEL_META: Dict[str, Dict[str, str]] = {
    "gpt-5.3-codex": {"training_type": "code-tuned", "family": "openai", "label": "GPT-5.3 Codex"},
    "gpt-5.3-chat": {"training_type": "chat-tuned", "family": "openai", "label": "GPT-5.3 Chat"},
    "claude-sonnet-4.6": {"training_type": "chat-tuned", "family": "claude", "label": "Claude Sonnet 4.6"},
    "claude-haiku-4.5": {"training_type": "chat-tuned", "family": "claude", "label": "Claude Haiku 4.5"},
    "deepseek-r1": {"training_type": "reasoning-rl", "family": "deepseek", "label": "DeepSeek-R1"},
    "deepseek-v3.2": {"training_type": "base", "family": "deepseek", "label": "DeepSeek-V3.2"},
    "gemini-flash": {"training_type": "chat-tuned", "family": "google", "label": "Gemini Flash"},
    "arcee-ai__trinity-large-preview_free": {"training_type": "chat-tuned", "family": "arcee", "label": "Arcee Trinity"},
    "glm-5": {"training_type": "chat-tuned", "family": "zhipu", "label": "GLM-5"},
}


def _get_store(db_path: Optional[str] = None) -> ResultsStore:
    path = Path(db_path) if db_path else DEFAULT_DB
    return ResultsStore(path)


def get_model_label(model: str) -> str:
    """Return a human-readable label for a model name."""
    meta = MODEL_META.get(model)
    return meta["label"] if meta else model


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_turn_data(
    db_path: Optional[str] = None,
) -> Dict[str, Dict[str, Dict[int, float]]]:
    """Load per-turn accuracy for all models and conditions.

    Returns {model: {condition: {turn_idx: accuracy}}}.
    Uses the most recent run data for each (model, condition).
    """
    store = _get_store(db_path)
    # Get most recent run_id per (model, condition)
    rows = store.query("""
        SELECT t.model, t.condition, t.turn_idx, t.accuracy
        FROM turns t
        INNER JOIN (
            SELECT model, condition, MAX(run_id) as max_run
            FROM cells WHERE status='complete'
            GROUP BY model, condition
        ) latest ON t.model = latest.model
                AND t.condition = latest.condition
                AND t.run_id = latest.max_run
        ORDER BY t.model, t.condition, t.turn_idx
    """)
    store.close()

    result: Dict[str, Dict[str, Dict[int, float]]] = {}
    for r in rows:
        model = r["model"]
        cond = r["condition"]
        if model not in result:
            result[model] = {}
        if cond not in result[model]:
            result[model][cond] = {}
        result[model][cond][r["turn_idx"]] = r["accuracy"]
    return result


def load_condition_means(
    db_path: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """Load per-condition average accuracy for all models.

    Returns {model: {condition: avg_accuracy}}.
    """
    store = _get_store(db_path)
    matrix = store.status_matrix()
    store.close()

    result: Dict[str, Dict[str, float]] = {}
    for model, conds in matrix.items():
        result[model] = {}
        for cond, data in conds.items():
            if data["status"] == "complete" and data["avg_accuracy"] is not None:
                result[model][cond] = data["avg_accuracy"]
    return result


# ---------------------------------------------------------------------------
# Gap Chart data
# ---------------------------------------------------------------------------

def compute_gap_data(
    db_path: Optional[str] = None,
    late_turns: range = range(7, 12),
) -> List[Dict[str, Any]]:
    """Compute feedback learning gap for each model.

    Gap = mean accuracy on late turns (correction) - mean accuracy on late turns (practice_only).
    Returns list of {model, label, gap, ci_lo, ci_hi} sorted by gap descending.
    """
    turn_data = load_all_turn_data(db_path)
    gaps: List[Dict[str, Any]] = []

    for model, conds in turn_data.items():
        corr = conds.get("correction", {})
        prac = conds.get("practice_only", {})
        if not corr or not prac:
            continue

        corr_late = [corr.get(t, 0.0) for t in late_turns if t in corr]
        prac_late = [prac.get(t, 0.0) for t in late_turns if t in prac]

        if not corr_late or not prac_late:
            # Fall back to all available turns
            corr_late = list(corr.values())
            prac_late = list(prac.values())

        gap = float(np.mean(corr_late)) - float(np.mean(prac_late))

        # Bootstrap CI on the gap
        if len(corr_late) >= 2 and len(prac_late) >= 2:
            _, ci_lo, ci_hi = bootstrap_ci_difference(corr_late, prac_late)
        else:
            ci_lo, ci_hi = gap, gap

        gaps.append({
            "model": model,
            "label": get_model_label(model),
            "gap": gap,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
        })

    gaps.sort(key=lambda x: x["gap"], reverse=True)
    return gaps


# ---------------------------------------------------------------------------
# FLR data
# ---------------------------------------------------------------------------

def compute_flr_data(
    db_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Compute FLR (slope difference) for each model.

    Returns list of {model, label, flr, training_type} sorted by FLR descending.
    """
    turn_data = load_all_turn_data(db_path)
    flrs: List[Dict[str, Any]] = []

    for model, conds in turn_data.items():
        corr = conds.get("correction", {})
        prac = conds.get("practice_only", {})

        corr_slope = _slope_from_turn_accuracies(corr) if len(corr) >= 2 else 0.0
        prac_slope = _slope_from_turn_accuracies(prac) if len(prac) >= 2 else 0.0
        flr = corr_slope - prac_slope

        meta = MODEL_META.get(model, {})
        flrs.append({
            "model": model,
            "label": get_model_label(model),
            "flr": flr,
            "training_type": meta.get("training_type", "unknown"),
        })

    flrs.sort(key=lambda x: x["flr"], reverse=True)
    return flrs


# ---------------------------------------------------------------------------
# Factorial decomposition data
# ---------------------------------------------------------------------------

def compute_factorial_data(
    db_path: Optional[str] = None,
) -> Dict[str, Dict[str, Tuple[float, float, float]]]:
    """Compute 2x2 factorial effects for each model that has all 4 core conditions.

    Returns {model: {"answer_effect": (pt, lo, hi), "evaluation_effect": ..., "interaction": ...}}.
    """
    condition_means = load_condition_means(db_path)
    results: Dict[str, Dict[str, Tuple[float, float, float]]] = {}

    for model, conds in condition_means.items():
        if not all(c in conds for c in CORE_CONDITIONS):
            continue

        # Point estimate via simple means
        effects = factorial_decomposition(conds)

        # For CIs, we need per-turn data to bootstrap (turn-level is our unit)
        turn_data = load_all_turn_data(db_path)
        model_turns = turn_data.get(model, {})

        # Build pseudo-results for bootstrap: one "observation" per turn per condition
        results_by_cond: Dict[str, List[Dict]] = {}
        for cond in CORE_CONDITIONS:
            cond_turns = model_turns.get(cond, {})
            obs = []
            for t_idx, acc in cond_turns.items():
                obs.append({
                    "instance_idx": t_idx,  # treat each turn as an "instance" for bootstrap
                    "turn_idx": 0,
                    "correct": acc > 0.5,  # binary approx for bootstrap
                })
            results_by_cond[cond] = obs

        # If we have enough data, compute with CI
        try:
            effects_ci = factorial_decomposition_with_ci(results_by_cond)
            results[model] = effects_ci
        except (ValueError, ZeroDivisionError):
            results[model] = {
                k: (v, v, v) for k, v in effects.items()
            }

    return results


# ---------------------------------------------------------------------------
# Model grouping data (for box plots)
# ---------------------------------------------------------------------------

def compute_grouping_data(
    db_path: Optional[str] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Group FLR values by training type.

    Returns {training_type: [{model, label, flr}, ...]}.
    """
    flr_data = compute_flr_data(db_path)
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for item in flr_data:
        groups[item["training_type"]].append(item)

    return dict(groups)
