#!/usr/bin/env python3
"""Recompute metrics for free model results, filtering empty responses.

Reads *_complete.json files from a results directory, filters out data points
with empty raw_response or model_answer (thinking models hitting token limits),
recomputes all metrics, and saves updated results.

Usage:
    python scripts/recompute_free_metrics.py results/free_models_20260319_124026
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lesson.eval.stats import compute_model_profile

SB2_TURNS = 12

ALL_CONDITIONS = [
    "correction", "practice_only", "error_only", "no_feedback",
    "explanation", "misleading",
    "clean_context", "prompted_correction", "structured_correction",
    "reformatted_correction",
]


def _has_response(r: Dict) -> bool:
    """Check if a result has a non-empty response."""
    raw = r.get("raw_response", "")
    answer = r.get("model_answer", "")
    return bool(raw and raw.strip()) or bool(answer and answer.strip())


def recompute(results_dir: Path) -> None:
    # Process _complete.json files first, then _progress.json for models
    # that didn't finish, and _sb1.json for models with only SB1 data
    complete_files = {f.stem.replace("_complete", ""): f for f in results_dir.glob("*_complete.json")}
    progress_files = {f.stem.replace("_progress", ""): f for f in results_dir.glob("*_progress.json")}
    sb1_files = {f.stem.replace("_sb1", ""): f for f in results_dir.glob("*_sb1.json")}

    # Merge: prefer complete > progress > sb1
    all_files = {}
    for key, f in sb1_files.items():
        all_files[key] = f
    for key, f in progress_files.items():
        all_files[key] = f
    for key, f in complete_files.items():
        all_files[key] = f

    files = sorted(all_files.values(), key=lambda f: f.name)
    if not files:
        print(f"No *_complete.json files found in {results_dir}")
        return

    print(f"Found {len(files)} completed models in {results_dir}\n")

    all_results = {}

    for f in files:
        with open(f) as fh:
            data = json.load(fh)

        model_id = data.get("model_id", f.stem)

        # Filter SB1
        sb1_raw = data.get("sb1", {}).get("results", [])
        sb1_clean = [r for r in sb1_raw if _has_response(r)]
        sb1_dropped = len(sb1_raw) - len(sb1_clean)

        # Filter SB2
        sb2_by_cond: Dict[str, List[Dict]] = {}
        sb2_dropped = 0
        for cond in ALL_CONDITIONS:
            key = f"sb2_{cond}"
            if key in data and isinstance(data[key], dict):
                raw = data[key].get("results", [])
                clean = [r for r in raw if _has_response(r)]
                sb2_dropped += len(raw) - len(clean)
                if clean:
                    sb2_by_cond[cond] = clean

        total_dropped = sb1_dropped + sb2_dropped
        total_raw = len(sb1_raw) + sum(
            len(data.get(f"sb2_{c}", {}).get("results", []))
            for c in ALL_CONDITIONS
            if f"sb2_{c}" in data and isinstance(data.get(f"sb2_{c}"), dict)
        )

        # Recompute metrics
        if sb1_clean or sb2_by_cond:
            profile = compute_model_profile(
                sb1_results=sb1_clean,
                sb2_results_by_condition=sb2_by_cond,
                n_turns=SB2_TURNS,
            )
            data["metrics_clean"] = profile
            data["metrics_filtered"] = {
                "sb1_empty_discarded": sb1_dropped,
                "sb2_empty_discarded": sb2_dropped,
                "total_discarded": total_dropped,
                "total_raw": total_raw,
                "discard_rate": f"{total_dropped/total_raw:.1%}" if total_raw else "N/A",
            }

        # Recompute SB1 summary excluding empty
        if sb1_clean:
            clean_summary: Dict = {}
            for r in sb1_clean:
                t = r["tier"]
                n = r.get("n_examples", r.get("n"))
                it = r["item_type"]
                clean_summary.setdefault(t, {}).setdefault(n, {}).setdefault(
                    it, {"n_correct": 0, "n_total": 0}
                )
                clean_summary[t][n][it]["n_total"] += 1
                if r["correct"]:
                    clean_summary[t][n][it]["n_correct"] += 1
            data["sb1_clean_summary"] = clean_summary

        # Save updated file
        with open(f, "w") as fh:
            json.dump(data, fh, indent=2, default=str)

        # Print summary
        drop_pct = f"{total_dropped/total_raw:.0%}" if total_raw else "N/A"
        aulc = profile.get("aulc", 0) if 'profile' in dir() else 0
        print(f"  {model_id:<50} dropped {total_dropped}/{total_raw} ({drop_pct})")

        all_results[model_id] = data

    # Save combined clean results
    combined = results_dir / "combined_results_clean.json"
    with open(combined, "w") as fh:
        json.dump(all_results, fh, indent=2, default=str)

    print(f"\nSaved cleaned combined results to {combined}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/recompute_free_metrics.py <results_dir>")
        sys.exit(1)
    recompute(Path(sys.argv[1]))
