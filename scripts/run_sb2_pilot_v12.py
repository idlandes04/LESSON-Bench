#!/usr/bin/env python3
"""LESSON-Bench v12.0 — SB2 Pilot Runner.

Runs the 8-model SB2 pilot with full parallelism controls:
  - OpenRouter models: 8 parallel streams (one per model)
  - Gemini models: 1 stream (rate-limited to avoid 429s)
  - LM Studio models: 1 model at a time, 4 parallel cells (continuous batching)

Usage:
    # Run the full 8-model pilot (default)
    python scripts/run_sb2_pilot_v12.py

    # Smoke test first
    python scripts/run_sb2_pilot_v12.py --smoke-test

    # Custom model selection
    python scripts/run_sb2_pilot_v12.py --models glm-5,deepseek-r1,gemini-flash

    # Production run (25 instances)
    python scripts/run_sb2_pilot_v12.py --n-instances 25
"""

import argparse
import json
import sys
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from lesson.models.registry import get_client
from lesson.eval.runner import (
    save_incremental,
    parse_model_list,
    smoke_test,
    run_parallel_by_provider,
    print_cross_model_summary,
)


# ---------------------------------------------------------------------------
# Default pilot configuration (v12.0 spec)
# ---------------------------------------------------------------------------

SB2_PILOT_MODELS: List[Tuple[str, str]] = [
    ("openrouter", "glm-5"),
    ("openrouter", "gpt-5.3-codex"),
    ("openrouter", "gpt-5.3-chat"),
    ("gemini",     "gemini-flash"),
    ("openrouter", "claude-sonnet-4.6"),
    ("openrouter", "deepseek-r1"),
    ("openrouter", "deepseek-v3.2"),
    ("openrouter", "claude-haiku-4.5"),
]

from lesson.eval.sb2_pilot import CORE_CONDITIONS, ALL_CONDITIONS


# ---------------------------------------------------------------------------
# Per-model SB2 evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    provider: str,
    model_name: str,
    print_lock: threading.Lock,
    conditions: List[str] = CORE_CONDITIONS,
    n_instances: int = 3,
    n_turns: int = 12,
    tier: int = 2,
    cell_parallel: int = 1,
    results_dir: Optional[Path] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run full SB2 evaluation for a single model across all conditions."""
    from lesson.eval.sb2_pilot import run_sb2_pilot

    model_results: Dict[str, Any] = {
        "model": model_name,
        "provider": provider,
        "conditions": conditions,
        "n_instances": n_instances,
        "n_turns": n_turns,
        "tier": tier,
        "started_at": datetime.now().isoformat(),
    }

    total_start = time.time()

    for condition in conditions:
        with print_lock:
            print(f"\n{'─' * 50}")
            print(f"  {provider}:{model_name} — {condition} "
                  f"(T{tier}, N=8, {n_turns}t, {n_instances}i)")
            print(f"{'─' * 50}")

        t0 = time.time()
        try:
            client = get_client(provider, model_name)
            sb2 = run_sb2_pilot(
                client=client,
                tier=tier,
                n_initial_examples=8,
                n_instances=n_instances,
                n_turns=n_turns,
                conditions=[condition],
                max_parallel=cell_parallel,
            )
            elapsed = time.time() - t0
            model_results[f"sb2_{condition}"] = sb2
            model_results[f"sb2_{condition}_elapsed_s"] = elapsed

            summary = sb2.get("summary", {}).get(condition, {})
            accs = []
            for t_data in summary.values():
                if t_data.get("n_total", 0) > 0:
                    accs.append(t_data["n_correct"] / t_data["n_total"])
            avg_acc = sum(accs) / len(accs) if accs else 0.0

            with print_lock:
                print(f"  -> {model_name} {condition}: avg={avg_acc:.0%} ({elapsed:.1f}s)")

        except Exception as e:
            elapsed = time.time() - t0
            with print_lock:
                print(f"  -> {model_name} {condition}: FAILED ({elapsed:.1f}s): {e}")
                import traceback; traceback.print_exc()
            model_results[f"sb2_{condition}_error"] = str(e)
            model_results[f"sb2_{condition}_elapsed_s"] = elapsed

        if results_dir:
            save_incremental(results_dir, model_name, f"sb2_{condition}", model_results)

    model_results["total_elapsed_s"] = time.time() - total_start
    model_results["finished_at"] = datetime.now().isoformat()

    if results_dir:
        save_incremental(results_dir, model_name, "sb2_all", model_results)

    return model_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LESSON-Bench v12.0 — SB2 Pilot Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model names (default: 8 pilot models).")
    parser.add_argument("--conditions", type=str, default=",".join(CORE_CONDITIONS),
                        help=f"Comma-separated conditions (default: {','.join(CORE_CONDITIONS)}).")
    parser.add_argument("--n-instances", type=int, default=3,
                        help="STS instances per cell (default: 3).")
    parser.add_argument("--n-turns", type=int, default=12, help="Turns per instance (default: 12).")
    parser.add_argument("--tier", type=int, default=2, help="STS difficulty tier (default: 2).")
    parser.add_argument("--or-parallel", type=int, default=8,
                        help="Max parallel OpenRouter models (default: 8).")
    parser.add_argument("--lm-parallel", type=int, default=4,
                        help="Max parallel cells for LM Studio (default: 4).")
    parser.add_argument("--output-dir", type=str, default=None, help="Override results directory.")
    parser.add_argument("--smoke-test", action="store_true", help="Run smoke test first.")
    parser.add_argument("--smoke-test-only", action="store_true", help="Smoke test only, then exit.")
    parser.add_argument("--save-to-db", action="store_true", help="Also save to SQLite DB.")
    args = parser.parse_args()

    # Parse models
    models = parse_model_list(args.models, SB2_PILOT_MODELS)
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]

    for c in conditions:
        if c not in ALL_CONDITIONS:
            print(f"ERROR: Unknown condition {c!r}. Valid: {ALL_CONDITIONS}")
            sys.exit(1)

    # Setup results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(args.output_dir) if args.output_dir else Path("results") / f"sb2_pilot_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "version": "v12.0",
        "models": [(p, m) for p, m in models],
        "conditions": conditions,
        "n_instances": args.n_instances,
        "n_turns": args.n_turns,
        "tier": args.tier,
        "or_parallel": args.or_parallel,
        "lm_parallel": args.lm_parallel,
        "timestamp": timestamp,
    }
    with open(results_dir / "run_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"LESSON-Bench v12.0 — SB2 Pilot Runner")
    print(f"{'=' * 60}")
    print(f"  Models:     {len(models)}")
    for p, m in models:
        print(f"    {p}:{m}")
    print(f"  Conditions: {conditions}")
    print(f"  Instances:  {args.n_instances}")
    print(f"  Turns:      {args.n_turns}")
    print(f"  Tier:       {args.tier}")
    print(f"  Results:    {results_dir}")
    print(f"  Parallelism: OR={args.or_parallel}")

    n_exchanges = len(models) * len(conditions) * args.n_instances * args.n_turns
    print(f"  Est. API calls: ~{n_exchanges * 2}")

    # Smoke test
    if args.smoke_test or args.smoke_test_only:
        models = smoke_test(models)
        if not models:
            print("\nERROR: No models passed smoke test!")
            sys.exit(1)
        if args.smoke_test_only:
            print("\nSmoke test complete. Exiting.")
            sys.exit(0)

    run_start = time.time()

    all_results = run_parallel_by_provider(
        items=models,
        eval_fn=evaluate_model,
        or_parallel=args.or_parallel,
        conditions=conditions,
        n_instances=args.n_instances,
        n_turns=args.n_turns,
        tier=args.tier,
        results_dir=results_dir,
    )

    # Save combined
    combined_path = results_dir / "combined_results.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    total_elapsed = time.time() - run_start

    # Save to DB
    if args.save_to_db:
        from lesson.results.store import ResultsStore
        store = ResultsStore()
        store.save_run(timestamp, version="v12.0", config=config, results_dir=str(results_dir))
        for model_name, model_data in all_results.items():
            provider = model_data.get("provider", "unknown")
            store.save_model_results(timestamp, model_name, provider, model_data, conditions)
        store.close()

    print_cross_model_summary(all_results, conditions)

    print(f"\n{'=' * 60}")
    print(f"PILOT COMPLETE")
    print(f"  Total time:  {total_elapsed:.0f}s ({total_elapsed/60:.1f}m)")
    print(f"  Models:      {len(all_results)}")
    print(f"  Results dir: {results_dir}")
    print(f"  Combined:    {combined_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
