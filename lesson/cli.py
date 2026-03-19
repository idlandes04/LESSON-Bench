"""Unified CLI for LESSON-Bench.

Usage:
    python -m lesson list                         # List all registered models
    python -m lesson smoke --models X,Y           # Smoke test connectivity
    python -m lesson run --models X,Y --conditions correction,practice_only
    python -m lesson resume [--dry-run]           # Re-run missing cells
    python -m lesson status                       # Print status matrix from DB
    python -m lesson results --condition correction  # Leaderboard from DB
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Subcommand: list
# ---------------------------------------------------------------------------

def cmd_list(args: argparse.Namespace) -> None:
    """List all registered models by provider."""
    from lesson.models.registry import (
        OPENROUTER_MODEL_CONFIGS,
        GEMINI_MODELS,
        LMSTUDIO_MODEL_CONFIGS,
        LOCAL_MODELS,
    )

    registries = [
        ("openrouter", OPENROUTER_MODEL_CONFIGS),
        ("gemini", GEMINI_MODELS),
        ("lmstudio", LMSTUDIO_MODEL_CONFIGS),
        ("local", LOCAL_MODELS),
    ]

    total = 0
    for provider, configs in registries:
        names = sorted(configs.keys())
        total += len(names)
        print(f"\n  {provider} ({len(names)} models):")
        for name in names:
            detail = configs[name]
            model_id = detail.get("model_id", "")
            print(f"    {name:<30} {model_id}")

    print(f"\n  Total: {total} models across {len(registries)} providers")


# ---------------------------------------------------------------------------
# Subcommand: smoke
# ---------------------------------------------------------------------------

def cmd_smoke(args: argparse.Namespace) -> None:
    """Run smoke test on specified models."""
    from lesson.eval.runner import smoke_test, parse_model_list

    models = parse_model_list(args.models)
    if not models:
        print("ERROR: No models specified. Use --models X,Y,Z")
        sys.exit(1)

    passed = smoke_test(models)
    print(f"\n{len(passed)}/{len(models)} models passed.")


# ---------------------------------------------------------------------------
# Subcommand: run
# ---------------------------------------------------------------------------

from lesson.eval.sb2_pilot import CORE_CONDITIONS, ALL_CONDITIONS


def cmd_run(args: argparse.Namespace) -> None:
    """Run SB2 evaluation on specified models."""
    from lesson.eval.runner import (
        parse_model_list,
        save_incremental,
        smoke_test,
        run_parallel_by_provider,
        print_cross_model_summary,
    )
    from lesson.models.registry import get_client
    from lesson.eval.sb2_pilot import run_sb2_pilot

    # Parse models
    models = parse_model_list(args.models)
    if not models:
        print("ERROR: No models specified. Use --models X,Y,Z")
        sys.exit(1)

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    for c in conditions:
        if c not in ALL_CONDITIONS:
            print(f"ERROR: Unknown condition {c!r}. Valid: {ALL_CONDITIONS}")
            sys.exit(1)

    # Setup results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(args.output_dir) if args.output_dir else Path("results") / f"sb2_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save run config
    config = {
        "version": "cli",
        "models": [(p, m) for p, m in models],
        "conditions": conditions,
        "n_instances": args.n_instances,
        "n_turns": args.n_turns,
        "tier": args.tier,
        "timestamp": timestamp,
    }
    with open(results_dir / "run_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nLESSON-Bench — SB2 Evaluation")
    print(f"  Models:     {len(models)}")
    for p, m in models:
        print(f"    {p}:{m}")
    print(f"  Conditions: {conditions}")
    print(f"  Instances:  {args.n_instances}")
    print(f"  Turns:      {args.n_turns}")
    print(f"  Tier:       {args.tier}")
    print(f"  Results:    {results_dir}")

    # Smoke test
    if args.smoke_test:
        models = smoke_test(models)
        if not models:
            print("\nERROR: No models passed smoke test!")
            sys.exit(1)

    _save_lock = threading.Lock()

    def evaluate_model(
        provider: str,
        model_name: str,
        print_lock: threading.Lock,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        model_results: Dict[str, Any] = {
            "model": model_name,
            "provider": provider,
            "conditions": conditions,
            "n_instances": args.n_instances,
            "n_turns": args.n_turns,
            "tier": args.tier,
            "started_at": datetime.now().isoformat(),
        }

        for condition in conditions:
            with print_lock:
                print(f"\n  {provider}:{model_name} — {condition}")

            t0 = time.time()
            try:
                client = get_client(provider, model_name)
                sb2 = run_sb2_pilot(
                    client=client,
                    tier=args.tier,
                    n_initial_examples=8,
                    n_instances=args.n_instances,
                    n_turns=args.n_turns,
                    conditions=[condition],
                    max_parallel=1,
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
                model_results[f"sb2_{condition}_error"] = str(e)
                model_results[f"sb2_{condition}_elapsed_s"] = elapsed

            save_incremental(results_dir, model_name, f"sb2_{condition}", model_results, _save_lock)

        model_results["total_elapsed_s"] = time.time() - t0
        model_results["finished_at"] = datetime.now().isoformat()
        save_incremental(results_dir, model_name, "sb2_all", model_results, _save_lock)
        return model_results

    run_start = time.time()
    all_results = run_parallel_by_provider(
        items=models,
        eval_fn=evaluate_model,
        or_parallel=args.or_parallel,
    )
    total_elapsed = time.time() - run_start

    # Save combined
    combined_path = results_dir / "combined_results.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Save to DB if requested
    if args.save_to_db:
        from lesson.results.store import ResultsStore
        store = ResultsStore()
        store.save_run(timestamp, version="cli", config=config, results_dir=str(results_dir))
        for model_name, model_data in all_results.items():
            provider = model_data.get("provider", "unknown")
            store.save_model_results(timestamp, model_name, provider, model_data, conditions)
        store.close()
        print(f"\n  Results saved to DB: {store._db_path}")

    print_cross_model_summary(all_results, conditions)

    print(f"\n{'=' * 60}")
    print(f"COMPLETE — {total_elapsed:.0f}s ({total_elapsed/60:.1f}m)")
    print(f"  Results: {results_dir}")
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# Subcommand: resume
# ---------------------------------------------------------------------------

def cmd_resume(args: argparse.Namespace) -> None:
    """Resume incomplete cells from the SQLite DB."""
    from lesson.eval.runner import (
        run_parallel_by_provider,
        get_completed_cells,
        CircuitBreaker,
    )
    from lesson.results.store import ResultsStore
    from lesson.models.registry import get_client, get_provider_for
    from lesson.eval.sb2_pilot import run_sb2_pilot

    db_path = Path(args.db) if args.db else None
    store = ResultsStore(db_path) if db_path else ResultsStore()

    # Get status matrix
    matrix = store.status_matrix()
    if not matrix:
        print("No results in database to resume.")
        store.close()
        return

    conditions = [c.strip() for c in args.conditions.split(",")] if args.conditions else CORE_CONDITIONS

    # Find incomplete cells
    cells_to_run: List[Tuple[str, str, str]] = []
    for model_name, cond_data in matrix.items():
        if args.models:
            allowed = [m.strip() for m in args.models.split(",")]
            if model_name not in allowed:
                continue
        for cond in conditions:
            cell = cond_data.get(cond, {})
            status = cell.get("status", "missing")
            if status in {"missing", "incomplete", "rate_limited"}:
                if args.skip_rate_limited and status == "rate_limited":
                    continue
                try:
                    provider = get_provider_for(model_name)
                except KeyError:
                    provider = "openrouter"
                cells_to_run.append((provider, model_name, cond))

    print(f"\nCells to resume: {len(cells_to_run)}")
    for provider, model, cond in cells_to_run:
        print(f"  {provider}:{model} — {cond}")

    if args.dry_run:
        print("\n[DRY RUN] No API calls made.")
        store.close()
        return

    if not cells_to_run:
        print("Nothing to resume — all cells complete.")
        store.close()
        return

    # Get run config from the most recent run
    runs = store.query("SELECT * FROM runs ORDER BY started_at DESC LIMIT 1")
    run_config = json.loads(runs[0]["config_json"]) if runs else {}
    n_instances = run_config.get("n_instances", 3)
    n_turns = run_config.get("n_turns", 12)
    tier = run_config.get("tier", 2)
    run_id = f"resume_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    _save_lock = threading.Lock()
    breaker = CircuitBreaker(max_failures=3)

    def evaluate_cell(
        provider: str,
        model_name: str,
        condition: str,
        print_lock: threading.Lock,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        with print_lock:
            print(f"\n  {provider}:{model_name} — {condition}")

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
                max_parallel=1,
            )
            elapsed = time.time() - t0

            summary = sb2.get("summary", {}).get(condition, {})
            per_turn = {}
            accs = []
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

            cell_data = {
                "status": "complete",
                "avg_accuracy": round(avg_accuracy, 4),
                "per_turn": per_turn,
                "elapsed_s": round(elapsed, 2),
                "completed_at": datetime.now().isoformat(),
            }

            with print_lock:
                print(f"  -> {model_name} {condition}: avg={avg_accuracy:.0%} ({elapsed:.1f}s)")

            # Save to DB
            with _save_lock:
                store.save_cell(run_id, model_name, provider, condition, cell_data)

            return {"model": model_name, "provider": provider, **cell_data}

        except Exception as e:
            elapsed = time.time() - t0
            with print_lock:
                print(f"  -> {model_name} {condition}: FAILED ({elapsed:.1f}s): {e}")

            cell_data = {
                "status": "incomplete",
                "avg_accuracy": None,
                "elapsed_s": round(elapsed, 2),
                "error": str(e),
            }
            with _save_lock:
                store.save_cell(run_id, model_name, provider, condition, cell_data)

            return {"model": model_name, "provider": provider, **cell_data}

    store.save_run(run_id, version="resume")
    t_start = time.time()

    run_parallel_by_provider(
        items=cells_to_run,
        eval_fn=evaluate_cell,
        or_parallel=args.or_parallel,
        circuit_breaker=breaker,
    )

    total = time.time() - t_start
    print(f"\nResume complete — {total:.0f}s ({total/60:.1f}m), {len(cells_to_run)} cells")

    store.close()


# ---------------------------------------------------------------------------
# Subcommand: status
# ---------------------------------------------------------------------------

def cmd_status(args: argparse.Namespace) -> None:
    """Print status matrix from the results DB."""
    from lesson.results.store import ResultsStore

    db_path = Path(args.db) if args.db else None
    store = ResultsStore(db_path) if db_path else ResultsStore()

    matrix = store.status_matrix()
    if not matrix:
        print("No results in database.")
        store.close()
        return

    # Collect all conditions
    all_conds = set()
    for model_data in matrix.values():
        all_conds.update(model_data.keys())
    conditions = sorted(all_conds)

    status_symbols = {
        "complete": "OK  ",
        "rate_limited": "RATE",
        "missing": "----",
        "incomplete": "INCL",
    }

    print("\n" + "=" * 80)
    print("RESULTS DB — Status Matrix")
    print("=" * 80)

    header = f"  {'Model':<25}"
    for cond in conditions:
        header += f"  {cond[:12]:<12}"
    print(header)
    print("  " + "-" * (25 + 14 * len(conditions)))

    for model_name in sorted(matrix):
        row = f"  {model_name:<25}"
        for cond in conditions:
            cell = matrix[model_name].get(cond, {})
            status = cell.get("status", "missing")
            sym = status_symbols.get(status, "????")
            avg = cell.get("avg_accuracy")
            if avg is not None:
                row += f"  {sym} {avg:4.0%}   "
            else:
                row += f"  {sym} ---    "
        print(row)

    print()
    store.close()


# ---------------------------------------------------------------------------
# Subcommand: results
# ---------------------------------------------------------------------------

def cmd_results(args: argparse.Namespace) -> None:
    """Query results from the SQLite store."""
    from lesson.results.store import ResultsStore

    db_path = Path(args.db) if args.db else None
    store = ResultsStore(db_path) if db_path else ResultsStore()

    if args.query:
        rows = store.query(args.query)
        for row in rows:
            print(dict(row))
    else:
        condition = args.condition or "correction"
        lb = store.leaderboard(condition, limit=args.limit)

        if not lb:
            print(f"No results for condition '{condition}'.")
            store.close()
            return

        print(f"\nLeaderboard — {condition} (by avg_accuracy)")
        print(f"{'Rank':<6}{'Model':<30}{'Provider':<15}{'Accuracy':>10}{'Status':<12}")
        print("-" * 73)
        for i, row in enumerate(lb, 1):
            acc = row.get("avg_accuracy")
            acc_str = f"{acc:.1%}" if acc is not None else "N/A"
            print(f"{i:<6}{row['model']:<30}{row.get('provider', '?'):<15}{acc_str:>10}{row.get('status', ''):>12}")

    store.close()


# ---------------------------------------------------------------------------
# Subcommand: analyze
# ---------------------------------------------------------------------------

def cmd_analyze(args: argparse.Namespace) -> None:
    """Generate analysis figures and PDF report."""
    from lesson.analysis.report import generate_report, save_individual_figures

    db_path = args.db

    if args.figures_only:
        saved = save_individual_figures(
            output_dir=str(Path(args.output).parent / "figures"),
            db_path=db_path,
        )
        print(f"\n{len(saved)} figures saved.")
    elif args.show:
        # Interactive display
        import matplotlib.pyplot as plt
        from lesson.analysis.pipeline import (
            load_all_turn_data,
            load_condition_means,
            compute_gap_data,
            compute_flr_data,
            compute_factorial_data,
            compute_grouping_data,
        )
        from lesson.analysis.figures import (
            fig_gap_chart,
            fig_factorial_2x2,
            fig_trajectory_grid,
            fig_codex_vs_chat,
            fig_model_grouping_boxplot,
            fig_summary_table,
        )

        turn_data = load_all_turn_data(db_path)
        condition_means = load_condition_means(db_path)
        gap_data = compute_gap_data(db_path)
        flr_data = compute_flr_data(db_path)
        factorial_data = compute_factorial_data(db_path)
        grouping_data = compute_grouping_data(db_path)

        fig_summary_table(condition_means, flr_data)
        if gap_data:
            fig_gap_chart(gap_data)
        if factorial_data:
            fig_factorial_2x2(factorial_data)
        if "gpt-5.3-codex" in turn_data and "gpt-5.3-chat" in turn_data:
            fig_codex_vs_chat(turn_data)
        if turn_data:
            fig_trajectory_grid(turn_data)
        if grouping_data and len(grouping_data) > 1:
            fig_model_grouping_boxplot(grouping_data)

        plt.show()
    else:
        output_path = generate_report(output_path=args.output, db_path=db_path)
        print(f"\nReport: {output_path}")


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="lesson",
        description="LESSON-Bench — Unified CLI",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # list
    sp_list = subparsers.add_parser("list", help="List all registered models")
    sp_list.set_defaults(func=cmd_list)

    # smoke
    sp_smoke = subparsers.add_parser("smoke", help="Smoke test model connectivity")
    sp_smoke.add_argument("--models", type=str, required=True, help="Comma-separated model names")
    sp_smoke.set_defaults(func=cmd_smoke)

    # run
    sp_run = subparsers.add_parser("run", help="Run SB2 evaluation")
    sp_run.add_argument("--models", type=str, required=True, help="Comma-separated model names")
    sp_run.add_argument("--conditions", type=str, default=",".join(CORE_CONDITIONS),
                        help=f"Comma-separated conditions (default: {','.join(CORE_CONDITIONS)})")
    sp_run.add_argument("--n-instances", type=int, default=3, help="STS instances per cell (default: 3)")
    sp_run.add_argument("--n-turns", type=int, default=12, help="Turns per instance (default: 12)")
    sp_run.add_argument("--tier", type=int, default=2, help="STS difficulty tier (default: 2)")
    sp_run.add_argument("--or-parallel", type=int, default=8, help="Max parallel OR models (default: 8)")
    sp_run.add_argument("--output-dir", type=str, default=None, help="Override results directory")
    sp_run.add_argument("--smoke-test", action="store_true", help="Run smoke test first")
    sp_run.add_argument("--save-to-db", action="store_true", help="Also save results to SQLite DB")
    sp_run.set_defaults(func=cmd_run)

    # resume
    sp_resume = subparsers.add_parser("resume", help="Resume incomplete cells from DB")
    sp_resume.add_argument("--models", type=str, default=None, help="Comma-separated model names to resume")
    sp_resume.add_argument("--conditions", type=str, default=None,
                           help=f"Comma-separated conditions (default: {','.join(CORE_CONDITIONS)})")
    sp_resume.add_argument("--or-parallel", type=int, default=8, help="Max parallel OR models (default: 8)")
    sp_resume.add_argument("--skip-rate-limited", action="store_true", help="Skip rate_limited cells")
    sp_resume.add_argument("--dry-run", action="store_true", help="Show what would be run without API calls")
    sp_resume.add_argument("--db", type=str, default=None, help="SQLite DB path")
    sp_resume.set_defaults(func=cmd_resume)

    # status
    sp_status = subparsers.add_parser("status", help="Show status matrix from results DB")
    sp_status.add_argument("--db", type=str, default=None, help="SQLite DB path")
    sp_status.set_defaults(func=cmd_status)

    # analyze
    sp_analyze = subparsers.add_parser("analyze", help="Generate analysis figures and PDF report")
    sp_analyze.add_argument("-o", "--output", type=str, default="results/lesson_bench_report.pdf",
                            help="PDF output path (default: results/lesson_bench_report.pdf)")
    sp_analyze.add_argument("--db", type=str, default=None, help="SQLite DB path")
    sp_analyze.add_argument("--figures-only", action="store_true",
                            help="Save individual PNGs instead of PDF")
    sp_analyze.add_argument("--show", action="store_true", help="Display figures interactively")
    sp_analyze.add_argument("--models", type=str, default=None, help="Comma-separated model filter")
    sp_analyze.set_defaults(func=cmd_analyze)

    # results
    sp_results = subparsers.add_parser("results", help="Query results from DB")
    sp_results.add_argument("--condition", type=str, default=None, help="Condition for leaderboard")
    sp_results.add_argument("--query", type=str, default=None, help="Raw SQL query")
    sp_results.add_argument("--limit", type=int, default=50, help="Max rows (default: 50)")
    sp_results.add_argument("--db", type=str, default=None, help="SQLite DB path")
    sp_results.set_defaults(func=cmd_results)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)
