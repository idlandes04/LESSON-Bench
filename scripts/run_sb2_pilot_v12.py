#!/usr/bin/env python3
"""LESSON-Bench v12.0 — SB2 Pilot Runner.

Runs the 8-model SB2 pilot with full parallelism controls:
  - OpenRouter models: 8 parallel streams (one per model)
  - Gemini models: 1 stream (rate-limited to avoid 429s)
  - LM Studio models: 1 model at a time, 4 parallel cells (continuous batching)

Default configuration matches v12.0 spec:
  - 4 core conditions: correction, practice_only, error_only, no_feedback
  - 3 instances per (model, condition) cell
  - 12 turns per instance
  - Tier 2, N=8

Usage:
    # Run the full 8-model pilot (default)
    python scripts/run_sb2_pilot_v12.py

    # Smoke test first (verify all models respond)
    python scripts/run_sb2_pilot_v12.py --smoke-test

    # Custom model selection
    python scripts/run_sb2_pilot_v12.py --models glm-5,deepseek-r1,gemini-flash

    # Extended conditions
    python scripts/run_sb2_pilot_v12.py --conditions correction,practice_only,error_only,no_feedback,explanation

    # Production run (25 instances)
    python scripts/run_sb2_pilot_v12.py --n-instances 25

    # Custom parallelism
    python scripts/run_sb2_pilot_v12.py --or-parallel 4 --lm-parallel 2
"""

import argparse
import concurrent.futures
import json
import os
import sys
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()


# ---------------------------------------------------------------------------
# Default pilot configuration (v12.0 spec)
# ---------------------------------------------------------------------------

# 8 pilot models, each testing a specific hypothesis
SB2_PILOT_MODELS: List[Tuple[str, str]] = [
    # (provider, model_name) — provider determines client type + parallelism
    ("openrouter", "glm-5"),             # Highest SB1 performer, ceiling reference
    ("openrouter", "gpt-5.3-codex"),     # Code hypothesis H2 (compare to Chat)
    ("openrouter", "gpt-5.3-chat"),      # Code hypothesis H2 (compare to Codex)
    ("gemini",     "gemini-flash"),       # Kaggle SDK model, Google judge appeal
    ("openrouter", "claude-sonnet-4.6"), # Different architecture family
    ("openrouter", "deepseek-r1"),       # Reasoning-trained model hypothesis H3
    ("openrouter", "deepseek-v3.2"),     # Same family, NOT reasoning-trained (control)
    ("openrouter", "claude-haiku-4.5"),  # Scale comparison within Claude family
]

CORE_CONDITIONS = ["correction", "practice_only", "error_only", "no_feedback"]

# All available conditions for reference
ALL_CONDITIONS = [
    "correction", "practice_only", "error_only", "no_feedback",
    "explanation", "misleading",
    "clean_context", "prompted_correction", "structured_correction",
    "reformatted_correction",
]


# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------

def get_client(provider: str, model_name: str):
    """Create the appropriate LLMClient for a provider/model pair."""
    if provider == "gemini":
        from lesson.models.registry import get_gemini_client
        return get_gemini_client(model_name)
    elif provider == "openrouter":
        from lesson.models.registry import get_openrouter_client
        return get_openrouter_client(model_name)
    elif provider == "lmstudio":
        from lesson.models.registry import get_lmstudio_client
        return get_lmstudio_client(model_name)
    elif provider == "local":
        from lesson.models.registry import get_local_client
        return get_local_client(model_name)
    else:
        raise ValueError(f"Unknown provider: {provider!r}")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def run_smoke_test(models: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Verify all models respond before committing to a full pilot run.

    Sends a single STS-like prompt to each model and checks for a non-empty
    response. Returns the list of models that passed.
    """
    test_prompt = (
        "You are learning a symbolic transformation system.\n"
        "Here is 1 example:\n"
        "Example 1: ABC → CBA\n\n"
        "What is the output for: XYZ\n"
        'Respond with ONLY: {"output": "YOUR_ANSWER"}'
    )

    passed: List[Tuple[str, str]] = []
    failed: List[Tuple[str, str, str]] = []

    print("\n" + "=" * 60)
    print("SMOKE TEST — Verifying model connectivity")
    print("=" * 60)

    def _test_one(provider: str, model_name: str) -> Tuple[str, str, bool, str]:
        try:
            client = get_client(provider, model_name)
            t0 = time.time()
            response = client.prompt(test_prompt)
            elapsed = time.time() - t0
            if response.strip():
                return provider, model_name, True, f"OK ({elapsed:.1f}s): {response[:80]!r}"
            else:
                return provider, model_name, False, "EMPTY RESPONSE"
        except Exception as e:
            return provider, model_name, False, f"ERROR: {e}"

    # Run smoke tests in parallel (fast)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(models)) as pool:
        futures = {
            pool.submit(_test_one, p, m): (p, m)
            for p, m in models
        }
        for f in concurrent.futures.as_completed(futures):
            provider, model_name, ok, msg = f.result()
            status = "PASS" if ok else "FAIL"
            print(f"  [{status}] {provider}:{model_name} — {msg}")
            if ok:
                passed.append((provider, model_name))
            else:
                failed.append((provider, model_name, msg))

    print(f"\n  {len(passed)}/{len(models)} models passed smoke test")
    if failed:
        print("  Failed models:")
        for p, m, msg in failed:
            print(f"    {p}:{m} — {msg}")

    return passed


# ---------------------------------------------------------------------------
# Incremental save helper
# ---------------------------------------------------------------------------

def _save(results_dir: Path, model_name: str, label: str, data: Any) -> None:
    """Save results JSON incrementally."""
    safe_name = model_name.replace("/", "_").replace(":", "_")
    path = results_dir / f"{safe_name}_{label}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Per-model SB2 evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    provider: str,
    model_name: str,
    conditions: List[str],
    n_instances: int,
    n_turns: int,
    tier: int,
    cell_parallel: int,
    results_dir: Path,
    print_lock: threading.Lock,
) -> Dict[str, Any]:
    """Run full SB2 evaluation for a single model across all conditions.

    Args:
        provider:       Client provider type.
        model_name:     Model registry name.
        conditions:     List of feedback conditions to evaluate.
        n_instances:    Number of STS instances per condition.
        n_turns:        Number of test turns per instance.
        tier:           STS difficulty tier.
        cell_parallel:  Max parallel (instance, condition) cells.
        results_dir:    Directory for incremental saves.
        print_lock:     Lock for thread-safe console output.

    Returns:
        Dict with per-condition SB2 results and timing.
    """
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

            # Extract average accuracy for quick summary
            summary = sb2.get("summary", {}).get(condition, {})
            accs = []
            for t_data in summary.values():
                if t_data.get("n_total", 0) > 0:
                    accs.append(t_data["n_correct"] / t_data["n_total"])
            avg_acc = sum(accs) / len(accs) if accs else 0.0

            with print_lock:
                print(f"  → {model_name} {condition}: avg={avg_acc:.0%} ({elapsed:.1f}s)")

        except Exception as e:
            elapsed = time.time() - t0
            with print_lock:
                print(f"  → {model_name} {condition}: FAILED ({elapsed:.1f}s): {e}")
                import traceback; traceback.print_exc()
            model_results[f"sb2_{condition}_error"] = str(e)
            model_results[f"sb2_{condition}_elapsed_s"] = elapsed

        # Save incrementally after each condition
        _save(results_dir, model_name, f"sb2_{condition}", model_results)

    model_results["total_elapsed_s"] = time.time() - total_start
    model_results["finished_at"] = datetime.now().isoformat()

    # Save full model result
    _save(results_dir, model_name, "sb2_all", model_results)

    return model_results


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

def print_summary(all_results: Dict[str, Dict], conditions: List[str]) -> None:
    """Print cross-model summary table with FLR estimates."""
    print("\n" + "=" * 80)
    print("SB2 PILOT SUMMARY — Accuracy per Turn (Tier 2, N=8)")
    print("=" * 80)

    for model_name, data in sorted(all_results.items()):
        cond_summaries: Dict[str, Dict] = {}
        for cond in conditions:
            key = f"sb2_{cond}"
            if key in data and "summary" in data[key]:
                cond_summary = data[key]["summary"]
                if cond in cond_summary:
                    cond_summaries[cond] = cond_summary[cond]

        if not cond_summaries:
            print(f"\n  {model_name}: NO RESULTS")
            continue

        print(f"\n  {model_name} ({data.get('provider', '?')}):")

        # Find max turn across all conditions
        max_turn = 0
        for cd in cond_summaries.values():
            for t_key in cd:
                max_turn = max(max_turn, int(t_key) + 1)

        header = f"    {'Condition':<22}"
        for t in range(max_turn):
            header += f" T{t:>3}"
        header += "   Avg"
        print(header)
        print("    " + "─" * (22 + 5 * max_turn + 6))

        for cond in conditions:
            if cond not in cond_summaries:
                continue
            cd = cond_summaries[cond]
            row = f"    {cond:<22}"
            accs = []
            for t in range(max_turn):
                t_key = t if t in cd else str(t)
                if t_key in cd:
                    c = cd[t_key]
                    a = c["n_correct"] / c["n_total"] if c["n_total"] else 0
                    accs.append(a)
                    row += f" {a:>3.0%}"
                else:
                    row += f" {'---':>3}"
            avg = sum(accs) / len(accs) if accs else 0
            row += f"  {avg:>4.0%}"
            print(row)

        # Compute FLR if we have both correction and practice_only
        if "correction" in cond_summaries and "practice_only" in cond_summaries:
            cc = cond_summaries["correction"]
            pp = cond_summaries["practice_only"]
            turns = sorted(
                set(list(cc.keys()) + list(pp.keys())),
                key=lambda x: int(x),
            )
            n = len(turns)
            if n >= 4:
                ca = [
                    cc[t]["n_correct"] / cc[t]["n_total"]
                    if cc.get(t, {}).get("n_total") else 0
                    for t in turns
                ]
                pa = [
                    pp[t]["n_correct"] / pp[t]["n_total"]
                    if pp.get(t, {}).get("n_total") else 0
                    for t in turns
                ]
                c_slope = (
                    sum(ca[n // 2:]) / len(ca[n // 2:])
                    - sum(ca[: n // 2]) / len(ca[: n // 2])
                )
                p_slope = (
                    sum(pa[n // 2:]) / len(pa[n // 2:])
                    - sum(pa[: n // 2]) / len(pa[: n // 2])
                )
                flr = c_slope - p_slope
                print(
                    f"    FLR = {flr:+.3f} "
                    f"(corr slope: {c_slope:+.3f}, "
                    f"prac slope: {p_slope:+.3f})"
                )

    # Cross-model comparison table
    print("\n" + "=" * 80)
    print("CROSS-MODEL COMPARISON — Average accuracy by condition")
    print("=" * 80)

    header = f"{'Model':<25}"
    for cond in conditions:
        header += f" {cond[:12]:>12}"
    header += f" {'FLR':>8}"
    print(header)
    print("─" * len(header))

    for model_name, data in sorted(all_results.items()):
        row = f"{model_name:<25}"
        cond_avgs = {}
        for cond in conditions:
            key = f"sb2_{cond}"
            if key in data and "summary" in data[key]:
                cond_summary = data[key]["summary"].get(cond, {})
                accs = []
                for t_data in cond_summary.values():
                    if t_data.get("n_total", 0) > 0:
                        accs.append(t_data["n_correct"] / t_data["n_total"])
                avg = sum(accs) / len(accs) if accs else 0
                cond_avgs[cond] = avg
                row += f" {avg:>11.0%}"
            else:
                row += f" {'---':>12}"

        # FLR
        if "correction" in cond_avgs and "practice_only" in cond_avgs:
            flr = cond_avgs["correction"] - cond_avgs["practice_only"]
            row += f" {flr:>+7.3f}"
        else:
            row += f" {'---':>8}"

        print(row)


# ---------------------------------------------------------------------------
# Model list parsing
# ---------------------------------------------------------------------------

def parse_models(
    models_arg: Optional[str],
    default_models: List[Tuple[str, str]],
) -> List[Tuple[str, str]]:
    """Parse --models flag into (provider, model_name) pairs.

    Looks up each name in the provider registries to determine the provider.
    """
    if models_arg is None:
        return default_models

    from lesson.models.openrouter import OPENROUTER_MODEL_CONFIGS
    from lesson.models.lmstudio import LMSTUDIO_MODEL_CONFIGS
    from lesson.models.registry import GEMINI_MODELS, LOCAL_MODELS

    requested = [m.strip() for m in models_arg.split(",") if m.strip()]
    result: List[Tuple[str, str]] = []

    for name in requested:
        # Check each registry in priority order
        if name in OPENROUTER_MODEL_CONFIGS:
            result.append(("openrouter", name))
        elif name in GEMINI_MODELS:
            result.append(("gemini", name))
        elif name in LMSTUDIO_MODEL_CONFIGS:
            result.append(("lmstudio", name))
        elif name in LOCAL_MODELS:
            result.append(("local", name))
        else:
            # Try prefix heuristics
            if name.startswith("gemini"):
                result.append(("gemini", name))
            elif name.startswith("or-") or "/" in name:
                result.append(("openrouter", name))
            else:
                print(f"  WARNING: Unknown model {name!r}, assuming openrouter")
                result.append(("openrouter", name))

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LESSON-Bench v12.0 — SB2 Pilot Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model names (default: 8 pilot models from v12.0 spec).",
    )
    parser.add_argument(
        "--conditions",
        type=str,
        default=",".join(CORE_CONDITIONS),
        help=f"Comma-separated conditions (default: {','.join(CORE_CONDITIONS)}).",
    )
    parser.add_argument(
        "--n-instances",
        type=int,
        default=3,
        help="STS instances per (model, condition) cell (default: 3 for pilot, 25 for production).",
    )
    parser.add_argument(
        "--n-turns",
        type=int,
        default=12,
        help="Test turns per instance (default: 12).",
    )
    parser.add_argument(
        "--tier",
        type=int,
        default=2,
        help="STS difficulty tier (default: 2).",
    )
    parser.add_argument(
        "--or-parallel",
        type=int,
        default=8,
        help="Max parallel OpenRouter models (default: 8).",
    )
    parser.add_argument(
        "--lm-parallel",
        type=int,
        default=4,
        help="Max parallel cells for LM Studio models (default: 4, matches continuous batching).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override results directory.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run connectivity check before full evaluation. Skips failing models.",
    )
    parser.add_argument(
        "--smoke-test-only",
        action="store_true",
        help="Run connectivity check only, then exit.",
    )
    args = parser.parse_args()

    # Parse models
    models = parse_models(args.models, SB2_PILOT_MODELS)
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]

    # Validate conditions
    for c in conditions:
        if c not in ALL_CONDITIONS:
            print(f"ERROR: Unknown condition {c!r}. Valid: {ALL_CONDITIONS}")
            sys.exit(1)

    # Setup results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        results_dir = Path(args.output_dir)
    else:
        results_dir = Path("results") / f"sb2_pilot_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save run configuration
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
    print(f"  Parallelism: OR={args.or_parallel}, LM={args.lm_parallel}")

    # Estimate cost
    n_exchanges = len(models) * len(conditions) * args.n_instances * args.n_turns
    print(f"  Est. API calls: ~{n_exchanges * 2} (prompt+fallback)")

    # Smoke test
    if args.smoke_test or args.smoke_test_only:
        models = run_smoke_test(models)
        if not models:
            print("\nERROR: No models passed smoke test!")
            sys.exit(1)
        if args.smoke_test_only:
            print("\nSmoke test complete. Exiting.")
            sys.exit(0)

    # Group models by provider for parallelism strategy
    or_models = [(p, m) for p, m in models if p == "openrouter"]
    gemini_models = [(p, m) for p, m in models if p == "gemini"]
    lm_models = [(p, m) for p, m in models if p == "lmstudio"]
    local_models = [(p, m) for p, m in models if p == "local"]

    all_results: Dict[str, Dict] = {}
    print_lock = threading.Lock()
    run_start = time.time()

    print(f"\n{'=' * 60}")
    print(f"STARTING EVALUATION")
    print(f"  OpenRouter: {len(or_models)} models (parallel={args.or_parallel})")
    print(f"  Gemini:     {len(gemini_models)} models (sequential, rate-limited)")
    print(f"  LM Studio:  {len(lm_models)} models (cell parallel={args.lm_parallel})")
    print(f"  Local:      {len(local_models)} models (sequential)")
    print(f"{'=' * 60}")

    # --- OpenRouter models: run in parallel ---
    if or_models:
        print(f"\n{'━' * 60}")
        print(f"PHASE: OpenRouter ({len(or_models)} models, {args.or_parallel} parallel)")
        print(f"{'━' * 60}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.or_parallel) as pool:
            futures = {}
            for provider, model_name in or_models:
                f = pool.submit(
                    evaluate_model,
                    provider=provider,
                    model_name=model_name,
                    conditions=conditions,
                    n_instances=args.n_instances,
                    n_turns=args.n_turns,
                    tier=args.tier,
                    cell_parallel=1,  # Sequential cells per model (parallelism is across models)
                    results_dir=results_dir,
                    print_lock=print_lock,
                )
                futures[f] = model_name

            for f in concurrent.futures.as_completed(futures):
                model_name = futures[f]
                try:
                    result = f.result()
                    all_results[model_name] = result
                    with print_lock:
                        elapsed = result.get("total_elapsed_s", 0)
                        print(f"\n  ✓ {model_name} complete ({elapsed:.0f}s)")
                except Exception as e:
                    with print_lock:
                        print(f"\n  ✗ {model_name} FAILED: {e}")
                    all_results[model_name] = {
                        "model": model_name,
                        "provider": "openrouter",
                        "error": str(e),
                    }

    # --- Gemini models: sequential, rate-limited ---
    for provider, model_name in gemini_models:
        print(f"\n{'━' * 60}")
        print(f"PHASE: Gemini — {model_name}")
        print(f"{'━' * 60}")

        try:
            result = evaluate_model(
                provider=provider,
                model_name=model_name,
                conditions=conditions,
                n_instances=args.n_instances,
                n_turns=args.n_turns,
                tier=args.tier,
                cell_parallel=1,  # Rate-limited
                results_dir=results_dir,
                print_lock=print_lock,
            )
            all_results[model_name] = result
            elapsed = result.get("total_elapsed_s", 0)
            print(f"\n  ✓ {model_name} complete ({elapsed:.0f}s)")
        except Exception as e:
            print(f"\n  ✗ {model_name} FAILED: {e}")
            all_results[model_name] = {
                "model": model_name,
                "provider": "gemini",
                "error": str(e),
            }

    # --- LM Studio models: sequential models, parallel cells ---
    for provider, model_name in lm_models:
        print(f"\n{'━' * 60}")
        print(f"PHASE: LM Studio — {model_name} (cell parallel={args.lm_parallel})")
        print(f"{'━' * 60}")

        try:
            result = evaluate_model(
                provider=provider,
                model_name=model_name,
                conditions=conditions,
                n_instances=args.n_instances,
                n_turns=args.n_turns,
                tier=args.tier,
                cell_parallel=args.lm_parallel,  # LM Studio supports continuous batching
                results_dir=results_dir,
                print_lock=print_lock,
            )
            all_results[model_name] = result
            elapsed = result.get("total_elapsed_s", 0)
            print(f"\n  ✓ {model_name} complete ({elapsed:.0f}s)")
        except Exception as e:
            print(f"\n  ✗ {model_name} FAILED: {e}")
            all_results[model_name] = {
                "model": model_name,
                "provider": "lmstudio",
                "error": str(e),
            }

    # --- Local llama-server models: sequential ---
    for provider, model_name in local_models:
        print(f"\n{'━' * 60}")
        print(f"PHASE: Local — {model_name}")
        print(f"{'━' * 60}")

        try:
            result = evaluate_model(
                provider=provider,
                model_name=model_name,
                conditions=conditions,
                n_instances=args.n_instances,
                n_turns=args.n_turns,
                tier=args.tier,
                cell_parallel=1,
                results_dir=results_dir,
                print_lock=print_lock,
            )
            all_results[model_name] = result
            elapsed = result.get("total_elapsed_s", 0)
            print(f"\n  ✓ {model_name} complete ({elapsed:.0f}s)")
        except Exception as e:
            print(f"\n  ✗ {model_name} FAILED: {e}")
            all_results[model_name] = {
                "model": model_name,
                "provider": "local",
                "error": str(e),
            }

    # --- Save combined results ---
    combined_path = results_dir / "combined_results.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    total_elapsed = time.time() - run_start

    # --- Print summary ---
    print_summary(all_results, conditions)

    print(f"\n{'=' * 60}")
    print(f"PILOT COMPLETE")
    print(f"  Total time:  {total_elapsed:.0f}s ({total_elapsed/60:.1f}m)")
    print(f"  Models:      {len(all_results)}")
    print(f"  Results dir: {results_dir}")
    print(f"  Combined:    {combined_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
