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

    # Resume specific conditions for specific models
    python scripts/resume_sb2_pilot.py --models deepseek-r1 --conditions error_only,no_feedback

    # Skip a broken model
    python scripts/resume_sb2_pilot.py --exclude-models glm-5

    # Also re-run rate_limited cells (default: included in resume)
    python scripts/resume_sb2_pilot.py --skip-rate-limited
"""

import argparse
import concurrent.futures
import json
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DB_PATH = Path("results/sb2_results_db.json")
RESULTS_DIR_KEY = "results_dir"
CORE_CONDITIONS = ["correction", "practice_only", "error_only", "no_feedback"]

# Models classified as reasoning models get longer timeouts per API call.
REASONING_MODELS = {"deepseek-r1"}

# Timeout per API call (seconds).  The OpenAI SDK accepts a `timeout` param
# in the constructor; we patch it via a thin wrapper below.
TIMEOUT_NORMAL_S = 60
TIMEOUT_REASONING_S = 120

# Provider for each pilot model (from run_config.json)
MODEL_PROVIDERS: Dict[str, str] = {
    "glm-5": "openrouter",
    "gpt-5.3-codex": "openrouter",
    "gpt-5.3-chat": "openrouter",
    "gemini-flash": "gemini",
    "claude-sonnet-4.6": "openrouter",
    "deepseek-r1": "openrouter",
    "deepseek-v3.2": "openrouter",
    "claude-haiku-4.5": "openrouter",
}

# Statuses that should be re-run
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
    """Return list of (provider, model_name, condition) tuples that need running.

    Filters:
        - model must not be in exclude_models
        - if filter_models given, model must be in filter_models
        - if filter_conditions given, condition must be in filter_conditions
        - cell status must be in RERUN_STATUSES (minus rate_limited if skip_rate_limited)
    """
    rerun = set(RERUN_STATUSES)
    if skip_rate_limited:
        rerun.discard("rate_limited")

    cells: List[Tuple[str, str, str]] = []
    for model_name, model_data in db.get("models", {}).items():
        if model_name in exclude_models:
            continue
        if filter_models and model_name not in filter_models:
            continue

        provider = model_data.get("provider", MODEL_PROVIDERS.get(model_name, "openrouter"))

        for condition, cell_data in model_data.get("cells", {}).items():
            if filter_conditions and condition not in filter_conditions:
                continue
            status = cell_data.get("status", "missing")
            if status in rerun:
                cells.append((provider, model_name, condition))

    return cells


# ---------------------------------------------------------------------------
# Client factory (with timeout support)
# ---------------------------------------------------------------------------

def get_client_with_timeout(provider: str, model_name: str):
    """Create the appropriate LLMClient, injecting a per-call timeout for OR clients."""
    timeout_s = TIMEOUT_REASONING_S if model_name in REASONING_MODELS else TIMEOUT_NORMAL_S

    if provider == "openrouter":
        from lesson.models.openrouter import OPENROUTER_MODEL_CONFIGS, OpenRouterClient
        import openai

        if model_name not in OPENROUTER_MODEL_CONFIGS:
            raise KeyError(f"Unknown OpenRouter model {model_name!r}")

        config = OPENROUTER_MODEL_CONFIGS[model_name]
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not set")

        # Build a client that passes timeout= to the underlying openai.OpenAI constructor
        client = OpenRouterClient.__new__(OpenRouterClient)
        client.name = model_name
        client._model_id = config["model_id"]
        client._max_tokens = config.get("max_tokens", 20_000)
        client._client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            max_retries=5,
            timeout=timeout_s,
            default_headers={
                "HTTP-Referer": "https://github.com/lesson-bench",
                "X-Title": "LESSON-Bench",
            },
        )
        return client

    elif provider == "gemini":
        from lesson.models.registry import get_gemini_client
        return get_gemini_client(model_name)

    elif provider == "lmstudio":
        from lesson.models.registry import get_lmstudio_client
        return get_lmstudio_client(model_name)

    elif provider == "local":
        from lesson.models.registry import get_local_client
        return get_local_client(model_name)

    else:
        raise ValueError(f"Unknown provider: {provider!r}")


# ---------------------------------------------------------------------------
# Per-cell evaluation (single condition for one model)
# ---------------------------------------------------------------------------

def evaluate_cell(
    provider: str,
    model_name: str,
    condition: str,
    db: Dict[str, Any],
    db_path: Path,
    results_dir: Path,
    print_lock: threading.Lock,
) -> Dict[str, Any]:
    """Run a single (model, condition) cell and update the DB.

    Mirrors the per-condition loop in run_sb2_pilot_v12.py's evaluate_model(),
    but operates on a single condition and updates the results DB afterward.
    """
    from lesson.eval.sb2_pilot import run_sb2_pilot

    # Read config from DB
    config = db.get("config", {})
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
        client = get_client_with_timeout(provider, model_name)
        sb2 = run_sb2_pilot(
            client=client,
            tier=tier,
            n_initial_examples=8,
            n_instances=n_instances,
            n_turns=n_turns,
            conditions=[condition],
            max_parallel=1,  # Conservative: one cell at a time per model
        )
        elapsed = time.time() - t0
        completed_at = datetime.now().isoformat()

        # Compute accuracy
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

        # Detect rate limiting: all turns 0% while other conditions have >0%
        status = "complete"
        rate_limit_evidence = None
        if avg_accuracy == 0.0 and len(accs) == n_turns:
            # Check if this model has any other condition with >0%
            model_data = db.get("models", {}).get(model_name, {})
            other_avgs = []
            for other_cond, other_cell in model_data.get("cells", {}).items():
                if other_cond != condition:
                    oa = other_cell.get("avg_accuracy")
                    if oa is not None and oa > 0:
                        other_avgs.append(oa)
            if other_avgs:
                status = "rate_limited"
                rate_limit_evidence = (
                    f"ALL {n_turns} turns 0%; other conditions avg "
                    f"{sum(other_avgs)/len(other_avgs):.1%}"
                )

        cell_result = {
            "status": status,
            "avg_accuracy": round(avg_accuracy, 4),
            "per_turn": per_turn,
            "elapsed_s": round(elapsed, 2),
            "completed_at": completed_at,
        }
        if rate_limit_evidence:
            cell_result["rate_limit_evidence"] = rate_limit_evidence

        with print_lock:
            flag = " [RATE LIMITED]" if status == "rate_limited" else ""
            print(f"  -> {model_name} {condition}: avg={avg_accuracy:.0%} "
                  f"({elapsed:.1f}s){flag}")

    except Exception as e:
        elapsed = time.time() - t0
        completed_at = datetime.now().isoformat()
        with print_lock:
            print(f"  -> {model_name} {condition}: FAILED ({elapsed:.1f}s): {e}")
            import traceback
            traceback.print_exc()

        cell_result = {
            "status": "incomplete",
            "avg_accuracy": None,
            "per_turn": None,
            "elapsed_s": round(elapsed, 2),
            "completed_at": completed_at,
            "error": str(e),
        }
        sb2 = None

    # Save incremental result file (same format as run_sb2_pilot_v12.py)
    safe_name = model_name.replace("/", "_").replace(":", "_")
    result_path = results_dir / f"{safe_name}_sb2_{condition}.json"

    # Load existing model-level file if present (to preserve other conditions)
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

    # Update DB (thread-safe)
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

    # Header
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


def print_run_plan(cells: List[Tuple[str, str, str]]) -> None:
    """Print what would be (or will be) run."""
    print("\n" + "=" * 60)
    print(f"CELLS TO RUN: {len(cells)}")
    print("=" * 60)
    if not cells:
        print("  Nothing to run — all cells are complete.")
        return
    for provider, model_name, condition in cells:
        print(f"  {provider}:<{model_name}> — {condition}")


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_resume(
    cells: List[Tuple[str, str, str]],
    db: Dict[str, Any],
    db_path: Path,
    results_dir: Path,
    or_parallel: int,
) -> None:
    """Run all pending cells using the same parallelism strategy as v12.

    - OpenRouter models: up to or_parallel in parallel
    - Gemini models: sequential (rate-limited)
    - LM Studio / local: sequential
    """
    or_cells = [(p, m, c) for p, m, c in cells if p == "openrouter"]
    gemini_cells = [(p, m, c) for p, m, c in cells if p == "gemini"]
    lm_cells = [(p, m, c) for p, m, c in cells if p == "lmstudio"]
    local_cells = [(p, m, c) for p, m, c in cells if p == "local"]

    print_lock = threading.Lock()

    # --- OpenRouter: parallel across models ---
    if or_cells:
        print(f"\n{'=' * 60}")
        print(f"PHASE: OpenRouter ({len(or_cells)} cells, {or_parallel} parallel)")
        print(f"{'=' * 60}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=or_parallel) as pool:
            futures = {}
            for provider, model_name, condition in or_cells:
                f = pool.submit(
                    evaluate_cell,
                    provider=provider,
                    model_name=model_name,
                    condition=condition,
                    db=db,
                    db_path=db_path,
                    results_dir=results_dir,
                    print_lock=print_lock,
                )
                futures[f] = (model_name, condition)

            for f in concurrent.futures.as_completed(futures):
                model_name, condition = futures[f]
                try:
                    result = f.result()
                    with print_lock:
                        status = result.get("status", "?")
                        avg = result.get("avg_accuracy")
                        avg_str = f"{avg:.0%}" if avg is not None else "N/A"
                        print(f"\n  [done] {model_name}:{condition} -> {status} ({avg_str})")
                except Exception as e:
                    with print_lock:
                        print(f"\n  [fail] {model_name}:{condition} EXCEPTION: {e}")

    # --- Gemini: sequential ---
    if gemini_cells:
        print(f"\n{'=' * 60}")
        print(f"PHASE: Gemini ({len(gemini_cells)} cells, sequential)")
        print(f"{'=' * 60}")

        for provider, model_name, condition in gemini_cells:
            evaluate_cell(
                provider=provider,
                model_name=model_name,
                condition=condition,
                db=db,
                db_path=db_path,
                results_dir=results_dir,
                print_lock=print_lock,
            )

    # --- LM Studio: sequential ---
    if lm_cells:
        print(f"\n{'=' * 60}")
        print(f"PHASE: LM Studio ({len(lm_cells)} cells, sequential)")
        print(f"{'=' * 60}")

        for provider, model_name, condition in lm_cells:
            evaluate_cell(
                provider=provider,
                model_name=model_name,
                condition=condition,
                db=db,
                db_path=db_path,
                results_dir=results_dir,
                print_lock=print_lock,
            )

    # --- Local: sequential ---
    if local_cells:
        print(f"\n{'=' * 60}")
        print(f"PHASE: Local ({len(local_cells)} cells, sequential)")
        print(f"{'=' * 60}")

        for provider, model_name, condition in local_cells:
            evaluate_cell(
                provider=provider,
                model_name=model_name,
                condition=condition,
                db=db,
                db_path=db_path,
                results_dir=results_dir,
                print_lock=print_lock,
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Resume an incomplete SB2 pilot run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--db",
        type=str,
        default=str(DB_PATH),
        help=f"Path to results DB JSON (default: {DB_PATH}).",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model names to include (default: all that need re-running).",
    )
    parser.add_argument(
        "--exclude-models",
        type=str,
        default=None,
        help="Comma-separated model names to skip.",
    )
    parser.add_argument(
        "--conditions",
        type=str,
        default=None,
        help="Comma-separated conditions to include (default: all that need re-running).",
    )
    parser.add_argument(
        "--skip-rate-limited",
        action="store_true",
        help="Skip cells marked rate_limited (do not attempt to re-run them).",
    )
    parser.add_argument(
        "--or-parallel",
        type=int,
        default=8,
        help="Max parallel OpenRouter cells (default: 8).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be run without making any API calls.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Override results directory (default: read from DB config).",
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    db = load_db(db_path)

    # Resolve results directory
    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        rel_dir = db.get("config", {}).get(RESULTS_DIR_KEY)
        if rel_dir:
            results_dir = Path(rel_dir)
        else:
            print("ERROR: No results_dir in DB config and --results-dir not specified.")
            sys.exit(1)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Parse filter lists
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

    # Print current status matrix
    print_status_summary(db)

    # Determine what needs to run
    cells = get_cells_to_run(
        db=db,
        filter_models=filter_models,
        filter_conditions=filter_conditions,
        exclude_models=exclude_models,
        skip_rate_limited=args.skip_rate_limited,
    )

    print_run_plan(cells)

    if not cells:
        print("\nNothing to do. Exiting.")
        sys.exit(0)

    if args.dry_run:
        print("\n[DRY RUN] No API calls made. Use without --dry-run to execute.")
        sys.exit(0)

    # Confirmation
    print(f"\nStarting resume run: {len(cells)} cells")
    print(f"  Results dir: {results_dir}")
    print(f"  DB path:     {db_path}")
    print(f"  OR parallel: {args.or_parallel}")
    print(f"  Timeouts:    normal={TIMEOUT_NORMAL_S}s, reasoning={TIMEOUT_REASONING_S}s")

    t_start = time.time()
    run_resume(
        cells=cells,
        db=db,
        db_path=db_path,
        results_dir=results_dir,
        or_parallel=args.or_parallel,
    )
    total_elapsed = time.time() - t_start

    print(f"\n{'=' * 60}")
    print(f"RESUME COMPLETE")
    print(f"  Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f}m)")
    print(f"  Cells run:  {len(cells)}")
    print(f"  DB updated: {db_path}")
    print(f"{'=' * 60}")

    # Print final status
    db_final = load_db(db_path)
    print_status_summary(db_final)


if __name__ == "__main__":
    main()
