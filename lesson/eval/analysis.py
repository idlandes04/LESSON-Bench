from __future__ import annotations

"""Quick analysis utilities for LESSON pilot results.

Works with the result dicts produced by pilot.py and sb2_pilot.py.
Uses only the standard library — no numpy or pandas required.
"""

from collections import defaultdict
from typing import Sequence


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _mean(values: Sequence[float]) -> float:
    """Return arithmetic mean, or 0.0 for an empty sequence."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def _linreg_slope(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Return the least-squares slope of (xs, ys).

    Returns 0.0 when there are fewer than two points or xs are all identical.
    """
    n = len(xs)
    if n < 2:
        return 0.0
    mean_x = _mean(xs)
    mean_y = _mean(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den = sum((x - mean_x) ** 2 for x in xs)
    if den == 0.0:
        return 0.0
    return num / den


def _fmt_pct(value: float) -> str:
    return f"{value * 100:5.1f}%"


def _hline(widths: list[int], sep: str = "+") -> str:
    return sep + sep.join("-" * w for w in widths) + sep


def _row(cells: list[str], widths: list[int], sep: str = "|") -> str:
    padded = [f" {c:<{w - 2}} " for c, w in zip(cells, widths)]
    return sep + sep.join(padded) + sep


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def summarize_sb1_results(results: list[dict]) -> None:
    """Print a formatted summary table of SB1 pilot results.

    Shows accuracy by tier x N, broken down by item type.

    Expected keys in each result dict (all optional — missing values are
    silently skipped so partially-populated dicts are tolerated):
        tier        (str)   e.g. "T1", "T2", "T3"
        n           (int)   number of transformations in the chain
        item_type   (str)   "A" | "E" | "L"
        correct     (bool)  whether the model answered correctly
        model       (str)   model identifier (used only for the header)
    """
    if not results:
        print("[summarize_sb1_results] No results to display.")
        return

    # ---- collect unique dimensions ----------------------------------------
    tiers = sorted({r["tier"] for r in results if "tier" in r})
    ns = sorted({r["n"] for r in results if "n" in r})
    item_types = sorted({r.get("item_type", "A") for r in results})

    model_name = results[0].get("model", "unknown") if results else "unknown"
    print(f"\n=== SB1 Pilot Results — model: {model_name} ===\n")

    # ---- accuracy by tier x N ---------------------------------------------
    print("Accuracy by Tier x N (all item types):")

    col_w = 10
    header_cells = ["Tier\\N"] + [str(n) for n in ns] + ["Overall"]
    col_widths = [col_w] * len(header_cells)
    print(_hline(col_widths))
    print(_row(header_cells, col_widths))
    print(_hline(col_widths))

    for tier in tiers:
        tier_rows = [r for r in results if r.get("tier") == tier]
        row_cells = [tier]
        for n in ns:
            cell_rows = [r for r in tier_rows if r.get("n") == n]
            if cell_rows:
                acc = _mean([1.0 if r.get("correct") else 0.0 for r in cell_rows])
                row_cells.append(f"{_fmt_pct(acc).strip()} ({len(cell_rows)})")
            else:
                row_cells.append("  —  ")
        overall_acc = _mean([1.0 if r.get("correct") else 0.0 for r in tier_rows])
        row_cells.append(f"{_fmt_pct(overall_acc).strip()} ({len(tier_rows)})")
        print(_row(row_cells, col_widths))

    print(_hline(col_widths))

    # ---- accuracy by item type --------------------------------------------
    if len(item_types) > 1:
        print("\nAccuracy by Item Type:")
        type_col_widths = [col_w, col_w, col_w]
        type_header = ["Item Type", "Accuracy", "Count"]
        print(_hline(type_col_widths))
        print(_row(type_header, type_col_widths))
        print(_hline(type_col_widths))
        for it in item_types:
            type_rows = [r for r in results if r.get("item_type", "A") == it]
            acc = _mean([1.0 if r.get("correct") else 0.0 for r in type_rows])
            print(_row([it, _fmt_pct(acc).strip(), str(len(type_rows))], type_col_widths))
        print(_hline(type_col_widths))

    # ---- overall ----------------------------------------------------------
    overall = _mean([1.0 if r.get("correct") else 0.0 for r in results])
    print(f"\nOverall accuracy: {_fmt_pct(overall)}  (n={len(results)})\n")


def summarize_sb2_results(results: list[dict]) -> None:
    """Print formatted summary of SB2 pilot results.

    Shows per-turn accuracy for each condition, plus an FLR estimate.

    Expected keys in each result dict:
        condition   (str)   "correction" | "practice_only"
        turn        (int)   1-indexed turn number within the session
        correct     (bool)  whether the model answered correctly at this turn
        subject     (str)   optional subject/session identifier
        model       (str)   model identifier (used only for the header)
    """
    if not results:
        print("[summarize_sb2_results] No results to display.")
        return

    model_name = results[0].get("model", "unknown")
    print(f"\n=== SB2 Pilot Results — model: {model_name} ===\n")

    conditions = sorted({r["condition"] for r in results if "condition" in r})
    max_turn = max((r.get("turn", 1) for r in results), default=1)
    turns = list(range(1, max_turn + 1))

    # ---- per-turn accuracy by condition -----------------------------------
    col_w = 14
    header_cells = ["Turn"] + conditions
    col_widths = [8] + [col_w] * len(conditions)
    print("Per-turn accuracy by condition:")
    print(_hline(col_widths))
    print(_row(header_cells, col_widths))
    print(_hline(col_widths))

    cond_turn_acc: dict[str, list[float]] = {c: [] for c in conditions}

    for turn in turns:
        row_cells = [str(turn)]
        for cond in conditions:
            cell_rows = [r for r in results if r.get("condition") == cond and r.get("turn") == turn]
            if cell_rows:
                acc = _mean([1.0 if r.get("correct") else 0.0 for r in cell_rows])
                cond_turn_acc[cond].append(acc)
                row_cells.append(f"{_fmt_pct(acc).strip()} ({len(cell_rows)})")
            else:
                cond_turn_acc[cond].append(float("nan"))
                row_cells.append("  —  ")
        print(_row(row_cells, col_widths))

    print(_hline(col_widths))

    # ---- overall per condition --------------------------------------------
    overall_cells = ["Overall"]
    for cond in conditions:
        cond_rows = [r for r in results if r.get("condition") == cond]
        acc = _mean([1.0 if r.get("correct") else 0.0 for r in cond_rows])
        overall_cells.append(f"{_fmt_pct(acc).strip()} ({len(cond_rows)})")
    print(_row(overall_cells, col_widths))
    print(_hline(col_widths))

    # ---- FLR estimate -----------------------------------------------------
    correction_rows = [r for r in results if r.get("condition") == "correction"]
    practice_rows = [r for r in results if r.get("condition") == "practice_only"]
    if correction_rows and practice_rows:
        flr = estimate_flr(correction_rows, practice_rows)
        print(f"\nFLR estimate: {flr:+.4f} (correction slope minus practice slope)\n")
    else:
        print("\nFLR estimate: insufficient data (need both conditions)\n")


def compute_type_e_feasibility(results: list[dict]) -> dict:
    """Compute Type E feasibility rate per tier.

    A Type E item is considered feasible when the model's answer to the
    diagnostic (extra-step) question is *different* from the answer to the
    paired standard item — indicating the model did not simply guess or apply
    a fixed rule.

    Expected keys in each result dict:
        tier            (str)   e.g. "T1"
        item_type       (str)   "E" for diagnostic items
        feasible        (bool)  True if the item discriminates as expected
        item_id         (str)   optional, used only for counting unique items

    Returns a dict mapping tier -> feasibility_rate (float in [0, 1]).
    """
    type_e = [r for r in results if r.get("item_type") == "E"]
    if not type_e:
        print("[compute_type_e_feasibility] No Type E items found in results.")
        return {}

    tiers = sorted({r["tier"] for r in type_e if "tier" in r})
    feasibility: dict[str, float] = {}

    col_widths = [10, 14, 10]
    print("\nType E Feasibility by Tier:")
    print(_hline(col_widths))
    print(_row(["Tier", "Feasibility", "Count"], col_widths))
    print(_hline(col_widths))

    for tier in tiers:
        tier_rows = [r for r in type_e if r.get("tier") == tier]
        rate = _mean([1.0 if r.get("feasible") else 0.0 for r in tier_rows])
        feasibility[tier] = rate
        print(_row([tier, _fmt_pct(rate).strip(), str(len(tier_rows))], col_widths))

    print(_hline(col_widths))

    overall = _mean(list(feasibility.values()))
    print(f"\nOverall Type E feasibility: {_fmt_pct(overall)}\n")
    return feasibility


def estimate_flr(correction_results: list, practice_results: list) -> float:
    """Estimate Feedback Learning Rate from pilot data.

    FLR = slope(correction accuracy over turns) - slope(practice accuracy over turns)

    A positive FLR means models in the correction condition improve faster
    across turns than those in the practice-only condition, suggesting that
    corrective feedback carries information beyond mere additional practice.

    Parameters
    ----------
    correction_results:
        List of result dicts for the correction condition.  Each dict must
        have a "turn" (int) key and a "correct" (bool) key.
    practice_results:
        Same structure for the practice-only condition.

    Returns
    -------
    float
        FLR estimate.  Returns 0.0 if either list is empty or contains only
        one turn.
    """
    def _slope_from_results(res: list[dict]) -> float:
        # Aggregate by turn, then fit a line through (turn, mean_accuracy).
        turn_buckets: dict[int, list[float]] = defaultdict(list)
        for r in res:
            if "turn" in r:
                turn_buckets[r["turn"]].append(1.0 if r.get("correct") else 0.0)
        if len(turn_buckets) < 2:
            return 0.0
        xs = sorted(turn_buckets.keys())
        ys = [_mean(turn_buckets[t]) for t in xs]
        return _linreg_slope([float(x) for x in xs], ys)

    slope_correction = _slope_from_results(correction_results)
    slope_practice = _slope_from_results(practice_results)
    return slope_correction - slope_practice
