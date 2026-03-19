#!/usr/bin/env python3
"""Development evaluation runner.

Runs quick SB1 + SB2 evaluations against the 4 development models:
  - qwen3.5-27b-think   (local, reasoning on)
  - qwen3.5-27b-nothink (local, reasoning off)
  - gemini-pro           (API, reasoning on)
  - gemini-pro-nothink   (API, reasoning off)

Usage:
  # Run all dev models (requires local server running + Gemini API key)
  python scripts/run_dev_eval.py

  # Run only local models
  python scripts/run_dev_eval.py --local-only

  # Run only Gemini models
  python scripts/run_dev_eval.py --gemini-only

  # Run only SB1 (baseline accuracy)
  python scripts/run_dev_eval.py --sb1-only

  # Run only SB2 (feedback conditions)
  python scripts/run_dev_eval.py --sb2-only

  # Custom tiers/instances
  python scripts/run_dev_eval.py --tiers 2,3 --n-instances 3
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from lesson.models.registry import get_local_client, get_gemini_client
from lesson.eval.pilot import run_sb1_pilot
from lesson.eval.sb2_pilot import run_sb2_pilot


def check_local_server(port: int = 8080) -> bool:
    """Check if llama-server is reachable."""
    import urllib.request
    try:
        urllib.request.urlopen(f"http://localhost:{port}/health", timeout=3)
        return True
    except Exception:
        return False


def run_eval(args):
    results_dir = Path("results") / datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir.mkdir(parents=True, exist_ok=True)

    tiers = [int(t) for t in args.tiers.split(",")]
    n_values = [int(n) for n in args.n_values.split(",")]
    sb2_tier = int(args.sb2_tier)

    # Determine which models to run
    models = []

    if args.models:
        # Explicit model list: auto-detect type from registry
        from lesson.models.registry import LOCAL_MODELS, GEMINI_MODELS
        for name in args.models.split(","):
            name = name.strip()
            if name in LOCAL_MODELS:
                models.append(("local", name))
            elif name in GEMINI_MODELS:
                models.append(("gemini", name))
            else:
                print(f"WARNING: Unknown model {name!r}, skipping")
    elif args.gemini_only:
        models.append(("gemini", "gemini-pro"))
        models.append(("gemini", "gemini-pro-nothink"))
    elif args.local_only:
        if check_local_server(8080):
            models.append(("local", "qwen3.5-27b-think"))
            models.append(("local", "qwen3.5-27b-nothink"))
        else:
            print("WARNING: llama-server not running on port 8080.")
    else:
        if check_local_server(8080):
            models.append(("local", "qwen3.5-27b-think"))
            models.append(("local", "qwen3.5-27b-nothink"))
        else:
            print("WARNING: llama-server not running on port 8080.")
            print("  Continuing with Gemini models only.\n")
        models.append(("gemini", "gemini-pro"))
        models.append(("gemini", "gemini-pro-nothink"))

    if not models:
        print("ERROR: No models available. Start llama-server or set GEMINI_API_KEY.")
        sys.exit(1)

    print(f"=== Dev Eval: {len(models)} models ===")
    print(f"Models: {[m[1] for m in models]}")
    print(f"SB1 tiers: {tiers}, N-values: {n_values}, instances: {args.n_instances}")
    print(f"SB2 tier: {sb2_tier}, turns: {args.sb2_turns}, instances: {args.sb2_instances}")
    print(f"Results dir: {results_dir}\n")

    all_summaries = {}

    for model_type, model_name in models:
        print(f"\n{'='*60}")
        print(f"MODEL: {model_name}")
        print(f"{'='*60}\n")

        try:
            if model_type == "local":
                client = get_local_client(model_name)
            else:
                client = get_gemini_client(model_name)
        except Exception as e:
            print(f"ERROR: Failed to create client for {model_name}: {e}")
            continue

        model_results = {"model": model_name, "type": model_type}

        # --- SB1: Baseline accuracy ---
        if not args.sb2_only:
            print(f"\n--- SB1: Baseline Accuracy ({model_name}) ---\n")
            t0 = time.time()
            try:
                sb1 = run_sb1_pilot(
                    client=client,
                    tiers=tiers,
                    n_values=n_values,
                    n_instances=args.n_instances,
                    seq_length=5,
                )
                model_results["sb1"] = sb1
                elapsed = time.time() - t0
                print(f"\n  SB1 completed in {elapsed:.1f}s")
            except Exception as e:
                print(f"  SB1 FAILED: {e}")
                model_results["sb1_error"] = str(e)

        # --- SB2: Feedback conditions ---
        if not args.sb1_only:
            print(f"\n--- SB2: Feedback Conditions ({model_name}) ---\n")
            t0 = time.time()
            try:
                sb2 = run_sb2_pilot(
                    client=client,
                    tier=sb2_tier,
                    n_initial_examples=4,
                    n_instances=args.sb2_instances,
                    n_turns=args.sb2_turns,
                    conditions=["correction", "practice_only"],
                )
                model_results["sb2"] = sb2
                elapsed = time.time() - t0
                print(f"\n  SB2 completed in {elapsed:.1f}s")
            except Exception as e:
                print(f"  SB2 FAILED: {e}")
                model_results["sb2_error"] = str(e)

        all_summaries[model_name] = model_results

        # Save per-model results
        out_path = results_dir / f"{model_name}.json"
        with open(out_path, "w") as f:
            json.dump(model_results, f, indent=2, default=str)
        print(f"\n  Results saved: {out_path}")

    # --- Print comparison table ---
    print("\n" + "=" * 70)
    print("COMPARISON: Think vs No-Think")
    print("=" * 70)

    # SB1 comparison
    if not args.sb2_only:
        print("\n--- SB1 Accuracy (Type R, all tiers) ---")
        print(f"{'Model':<25} {'Tier':>5} {'N=2':>8} {'N=4':>8} {'N=8':>8}")
        print("-" * 55)
        for model_name, data in all_summaries.items():
            if "sb1" not in data:
                continue
            summary = data["sb1"]["summary"]
            for tier in sorted(summary, key=int):
                row = f"{model_name:<25} {tier:>5}"
                for n in [2, 4, 8]:
                    n_str = str(n)
                    if n_str in summary[tier] or n in summary[tier]:
                        key = n_str if n_str in summary[tier] else n
                        counts = summary[tier][key].get("R", {"n_correct": 0, "n_total": 0})
                        if counts["n_total"] > 0:
                            acc = counts["n_correct"] / counts["n_total"]
                            row += f" {acc:>7.0%}"
                        else:
                            row += f" {'N/A':>7}"
                    else:
                        row += f" {'N/A':>7}"
                print(row)

    # SB2 comparison
    if not args.sb1_only:
        print("\n--- SB2 Learning Curves (correction condition) ---")
        print(f"{'Model':<25}", end="")
        for t in range(args.sb2_turns):
            print(f" {'T'+str(t):>6}", end="")
        print()
        print("-" * (25 + 7 * args.sb2_turns))
        for model_name, data in all_summaries.items():
            if "sb2" not in data:
                continue
            summary = data["sb2"]["summary"]
            if "correction" in summary:
                row = f"{model_name:<25}"
                for t in range(args.sb2_turns):
                    t_key = str(t) if str(t) in summary["correction"] else t
                    if t_key in summary["correction"]:
                        counts = summary["correction"][t_key]
                        acc = counts["n_correct"] / counts["n_total"] if counts["n_total"] else 0
                        row += f" {acc:>5.0%} "
                    else:
                        row += f" {'N/A':>5} "
                print(row)

    # Save combined summary
    combined_path = results_dir / "combined_summary.json"
    with open(combined_path, "w") as f:
        json.dump(all_summaries, f, indent=2, default=str)
    print(f"\nAll results: {combined_path}")


def main():
    parser = argparse.ArgumentParser(description="Run development evaluation")
    parser.add_argument("--models", default="",
                        help="Comma-separated model names (e.g. gemini-flash,gemini-pro)")
    parser.add_argument("--local-only", action="store_true",
                        help="Only run local Qwen models")
    parser.add_argument("--gemini-only", action="store_true",
                        help="Only run Gemini API models")
    parser.add_argument("--sb1-only", action="store_true",
                        help="Only run SB1 (baseline accuracy)")
    parser.add_argument("--sb2-only", action="store_true",
                        help="Only run SB2 (feedback conditions)")
    parser.add_argument("--tiers", default="2,3",
                        help="Comma-separated SB1 tiers (default: 2,3)")
    parser.add_argument("--n-values", default="2,4,8",
                        help="Comma-separated N values (default: 2,4,8)")
    parser.add_argument("--sb2-tier", default="2",
                        help="SB2 tier (default: 2)")
    parser.add_argument("--n-instances", type=int, default=3,
                        help="SB1 instances per cell (default: 3)")
    parser.add_argument("--sb2-instances", type=int, default=3,
                        help="SB2 instances (default: 3)")
    parser.add_argument("--sb2-turns", type=int, default=6,
                        help="SB2 turns per instance (default: 6)")
    args = parser.parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
