#!/usr/bin/env python3
"""Resume an incomplete SB2 pilot run.

Reads results/sb2_results_db.json to find cells that are missing, incomplete,
or rate_limited, then re-runs only those cells, saving results into the same
results directory and updating the DB after each cell completes.

Usage:
    # Show what needs to be run (no API calls)
    python scripts/resume_sb2_pilot.py --dry-run

    # Resume everything missing/rate_limited
    python scripts/resume_sb2_pilot.py

    # Resume only specific models
    python scripts/resume_sb2_pilot.py --models glm-5,gemini-flash

    # Skip rate_limited cells
    python scripts/resume_sb2_pilot.py --skip-rate-limited
"""

import argparse
import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from lesson.models.registry import get_client, get_provider_for
from lesson.eval.runner import run_parallel_by_provider


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DB_PATH = Path("results/sb2_results_db.json")
from lesson.eval.sb2_pilot import CORE_CONDITIONS
REASONING_MODELS = {"deepseek-r1"}
TIMEOUT_NORMAL_S = 60
TIMEOUT_REASONING_S = 120
RERUN_STATUSES = {"missing", "incomplete", "rate_limited"}

_db_lock = threading.Lock()


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def load_db(db_path: Path) -> Dict[str, Any]:
    """Load the results DB from disk."""
    if not db_path.exists():
        print(f"ERROR: Results DB not found at {db_path}")
        sys.exit(1)
    with open(db_path) as f:
        return json.load(f)


def save_db(db: Dict[str, Any], db_path: Path) -> None:
    """Atomically save the results DB (thread-safe via _db_lock)."""
    db["updated_at"] = datetime.now().isoformat()
    tmp = db_path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(db, f, indent=2, default=str)
    tmp.replace(db_path)


def get_cells_to_run(
    db: Dict[str, Any],
    filter_models: Optional[List[str]],
    filter_conditions: Optional[List[str]],
    exclude_models: List[str],
    skip_rate_limited: bool,
) -> List[Tuple[str, str, str]]:
    """Return (provider, model_name, condition) tuples that need running."""
    rerun = set(RERUN_STATUSES)
    if skip_rate_limited:
        rerun.discard("rate_limited")

    cells: List[Tuple[str, str, str]] = []
    for model_name, model_data in db.get("models", {}).items():
        if model_name in exclude_models:
            continue
        if filter_models and model_name not in filter_models:
            continue

        # Auto-detect provider from registry, fallback to DB
        try:
            provider = get_provider_for(model_name)
        except KeyError:
            provider = model_data.get("provider", "openrouter")

        for condition, cell_data in model_data.get("cells", {}).items():
            if filter_conditions and condition not in filter_conditions:
                continue
            status = cell_data.get("status", "missing")
            if status in rerun:
                cells.append((provider, model_name, condition))

    return cells


# ---------------------------------------------------------------------------
# Per-cell evaluation
# ---------------------------------------------------------------------------

def evaluate_cell(
    provider: str,
    model_name: str,
    condition: str,
    print_lock: threading.Lock,
    db: Optional[Dict[str, Any]] = None,
    db_path: Optional[Path] = None,
    results_dir: Optional[Path] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run a single (model, condition) cell and update the DB."""
    from lesson.eval.sb2_pilot import run_sb2_pilot

    config = db.get("config", {}) if db else {}
    n_instances = config.get("n_instances", 3)
    n_turns = config.get("n_turns", 12)
    tier = config.get("tier", 2)

    with print_lock:
        print(f"\n{'─' * 50}")
        print(f"  {provider}:{model_name} — {condition} "
              f"(T{tier}, N=8, {n_turns}t, {n_instances}i)")
        print(f"{'─' * 50}")

    t0 = time.time()
    started_at = datetime.now().isoformat()

    try:
        # Use timeout for OpenRouter to prevent hangs
        timeout_s = TIMEOUT_REASONING_S if model_name in REASONING_MODELS else TIMEOUT_NORMAL_S
        client_kwargs: Dict[str, Any] = {}
        if provider == "openrouter":
            client_kwargs["timeout"] = timeout_s

        client = get_client(provider, model_name, **client_kwargs)
        sb2 = run_sb2_pilot(
            client=client,
            tier=tier,
            n_initial_examples=8,
            n_instances=n_instances,
            n_turns=n_turns,
            conditions=[condition],
            max_parallel=1,
        )
        elapsed = time.time() - t0
        completed_at = datetime.now().isoformat()

        summary = sb2.get("summary", {}).get(condition, {})
        accs = []
        per_turn = {}
        for t_key, t_data in summary.items():
            n_total = t_data.get("n_total", 0)
            n_correct = t_data.get("n_correct", 0)
            if n_total > 0:
                acc = n_correct / n_total
                accs.append(acc)
                per_turn[str(t_key)] = {
                    "n_correct": n_correct,
                    "n_total": n_total,
                    "accuracy": round(acc, 4),
                }

        avg_accuracy = sum(accs) / len(accs) if accs else 0.0

        # Detect rate limiting
        status = "complete"
        if avg_accuracy == 0.0 and len(accs) == n_turns and db:
            model_data = db.get("models", {}).get(model_name, {})
            other_avgs = []
            for other_cond, other_cell in model_data.get("cells", {}).items():
                if other_cond != condition:
                    oa = other_cell.get("avg_accuracy")
                    if oa is not None and oa > 0:
                        other_avgs.append(oa)
            if other_avgs:
                status = "rate_limited"

        cell_result = {
            "status": status,
            "avg_accuracy": round(avg_accuracy, 4),
            "per_turn": per_turn,
            "elapsed_s": round(elapsed, 2),
            "completed_at": completed_at,
        }

        with print_lock:
            flag = " [RATE LIMITED]" if status == "rate_limited" else ""
            print(f"  -> {model_name} {condition}: avg={avg_accuracy:.0%} ({elapsed:.1f}s){flag}")

    except Exception as e:
        elapsed = time.time() - t0
        completed_at = datetime.now().isoformat()
        with print_lock:
            print(f"  -> {model_name} {condition}: FAILED ({elapsed:.1f}s): {e}")
            import traceback; traceback.print_exc()

        cell_result = {
            "status": "incomplete",
            "avg_accuracy": None,
            "per_turn": None,
            "elapsed_s": round(elapsed, 2),
            "completed_at": completed_at,
            "error": str(e),
        }
        sb2 = None

    # Save incremental result file
    if results_dir:
        safe_name = model_name.replace("/", "_").replace(":", "_")
        result_path = results_dir / f"{safe_name}_sb2_{condition}.json"
        existing: Dict[str, Any] = {}
        if result_path.exists():
            try:
                with open(result_path) as f:
                    existing = json.load(f)
            except Exception:
                pass

        existing.update({
            "model": model_name,
            "provider": provider,
            "n_instances": n_instances,
            "n_turns": n_turns,
            "tier": tier,
            "started_at": existing.get("started_at", started_at),
        })
        if sb2 is not None:
            existing[f"sb2_{condition}"] = sb2
        existing[f"sb2_{condition}_elapsed_s"] = round(elapsed, 2)

        with open(result_path, "w") as f:
            json.dump(existing, f, indent=2, default=str)

    # Update JSON DB
    if db is not None and db_path is not None:
        with _db_lock:
            db_model = db.setdefault("models", {}).setdefault(model_name, {})
            db_model["provider"] = provider
            if db_model.get("started_at") is None:
                db_model["started_at"] = started_at
            db_model.setdefault("cells", {})[condition] = cell_result
            save_db(db, db_path)

    return cell_result


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

def print_status_summary(db: Dict[str, Any]) -> None:
    """Print a table of (model, condition) cell statuses from the DB."""
    conditions = db.get("config", {}).get("conditions", CORE_CONDITIONS)
    models_data = db.get("models", {})

    status_symbols = {
        "complete": "OK  ",
        "rate_limited": "RATE",
        "missing": "----",
        "incomplete": "INCL",
    }

    print("\n" + "=" * 80)
    print("SB2 RESULTS DB — Cell Status Matrix")
    print("=" * 80)

    header = f"  {'Model':<25}"
    for cond in conditions:
        header += f"  {cond[:12]:<12}"
    print(header)
    print("  " + "-" * (25 + 14 * len(conditions)))

    for model_name in sorted(models_data):
        model_data = models_data[model_name]
        cells = model_data.get("cells", {})
        row = f"  {model_name:<25}"
        for cond in conditions:
            cell = cells.get(cond, {})
            status = cell.get("status", "missing")
            sym = status_symbols.get(status, "????")
            avg = cell.get("avg_accuracy")
            if avg is not None:
                row += f"  {sym} {avg:4.0%}   "
            else:
                row += f"  {sym} ---    "
        print(row)

    print()
    print("  Legend: OK=complete  RATE=rate_limited  INCL=incomplete  ----=missing")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Resume an incomplete SB2 pilot run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--db", type=str, default=str(DB_PATH),
                        help=f"Path to results DB JSON (default: {DB_PATH}).")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model names to include.")
    parser.add_argument("--exclude-models", type=str, default=None,
                        help="Comma-separated model names to skip.")
    parser.add_argument("--conditions", type=str, default=None,
                        help="Comma-separated conditions to include.")
    parser.add_argument("--skip-rate-limited", action="store_true",
                        help="Skip cells marked rate_limited.")
    parser.add_argument("--or-parallel", type=int, default=8,
                        help="Max parallel OpenRouter cells (default: 8).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be run without making API calls.")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Override results directory.")
    args = parser.parse_args()

    db_path = Path(args.db)
    db = load_db(db_path)

    # Resolve results directory
    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        rel_dir = db.get("config", {}).get("results_dir")
        if rel_dir:
            results_dir = Path(rel_dir)
        else:
            print("ERROR: No results_dir in DB config and --results-dir not specified.")
            sys.exit(1)
    results_dir.mkdir(parents=True, exist_ok=True)

    filter_models = (
        [m.strip() for m in args.models.split(",") if m.strip()]
        if args.models else None
    )
    exclude_models = (
        [m.strip() for m in args.exclude_models.split(",") if m.strip()]
        if args.exclude_models else []
    )
    filter_conditions = (
        [c.strip() for c in args.conditions.split(",") if c.strip()]
        if args.conditions else None
    )

    print_status_summary(db)

    cells = get_cells_to_run(db, filter_models, filter_conditions, exclude_models, args.skip_rate_limited)

    print(f"\n{'=' * 60}")
    print(f"CELLS TO RUN: {len(cells)}")
    print("=" * 60)
    if not cells:
        print("  Nothing to run — all cells are complete.")
        sys.exit(0)
    for provider, model_name, condition in cells:
        print(f"  {provider}:{model_name} — {condition}")

    if args.dry_run:
        print("\n[DRY RUN] No API calls made.")
        sys.exit(0)

    print(f"\nStarting resume: {len(cells)} cells")
    print(f"  Results dir: {results_dir}")
    print(f"  DB path:     {db_path}")
    print(f"  Timeouts:    normal={TIMEOUT_NORMAL_S}s, reasoning={TIMEOUT_REASONING_S}s")

    t_start = time.time()

    # Use run_parallel_by_provider with 3-tuples (provider, model, condition)
    run_parallel_by_provider(
        items=cells,
        eval_fn=evaluate_cell,
        or_parallel=args.or_parallel,
        db=db,
        db_path=db_path,
        results_dir=results_dir,
    )

    total_elapsed = time.time() - t_start

    print(f"\n{'=' * 60}")
    print(f"RESUME COMPLETE")
    print(f"  Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f}m)")
    print(f"  Cells run:  {len(cells)}")
    print(f"  DB updated: {db_path}")
    print(f"{'=' * 60}")

    db_final = load_db(db_path)
    print_status_summary(db_final)


if __name__ == "__main__":
    main()
