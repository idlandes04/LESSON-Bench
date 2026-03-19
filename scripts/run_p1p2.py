#!/usr/bin/env python3
"""Run Priorities 1 + 2 for available models.

Runs SB1 learning curves and SB2 feedback conditions (correction, practice_only, error_only)
for each available model, saving results incrementally.
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from lesson.models.registry import get_local_client, get_gemini_client
from lesson.eval.pilot import run_sb1_pilot
from lesson.eval.sb2_pilot import run_sb2_pilot


def check_server(port: int) -> bool:
    import urllib.request
    try:
        urllib.request.urlopen(f"http://localhost:{port}/health", timeout=3)
        return True
    except Exception:
        return False


def run_model(model_type, model_name, results_dir):
    """Run SB1 + SB2 (all 3 conditions) for one model."""
    print(f"\n{'=' * 60}")
    print(f"MODEL: {model_name}")
    print(f"{'=' * 60}")

    if model_type == "gemini":
        client = get_gemini_client(model_name)
    else:
        client = get_local_client(model_name)

    model_results = {"model": model_name, "type": model_type}

    # --- SB1: Learning curves across tiers ---
    print(f"\n--- SB1: Learning Curves ---")
    t0 = time.time()
    try:
        sb1 = run_sb1_pilot(
            client=client,
            tiers=[1, 2, 3],
            n_values=[4, 8],
            n_instances=3,
            seq_length=5,
        )
        model_results["sb1"] = sb1
        print(f"\n  SB1 done in {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"  SB1 FAILED: {e}")
        import traceback; traceback.print_exc()
        model_results["sb1_error"] = str(e)

    # Save SB1 results incrementally
    _save(results_dir, model_name, "sb1", model_results)

    # --- SB2: correction + practice_only ---
    print(f"\n--- SB2: correction + practice_only (T2, N=8, 8 turns, 3 instances) ---")
    t0 = time.time()
    try:
        sb2_main = run_sb2_pilot(
            client=client,
            tier=2,
            n_initial_examples=8,
            n_instances=3,
            n_turns=8,
            conditions=["correction", "practice_only"],
        )
        model_results["sb2_main"] = sb2_main
        print(f"\n  SB2 main done in {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"  SB2 main FAILED: {e}")
        import traceback; traceback.print_exc()
        model_results["sb2_main_error"] = str(e)

    _save(results_dir, model_name, "sb2_main", model_results)

    # --- SB2: error_only (Priority 2) ---
    print(f"\n--- SB2: error_only (T2, N=8, 8 turns, 3 instances) ---")
    t0 = time.time()
    try:
        sb2_err = run_sb2_pilot(
            client=client,
            tier=2,
            n_initial_examples=8,
            n_instances=3,
            n_turns=8,
            conditions=["error_only"],
        )
        model_results["sb2_error_only"] = sb2_err
        print(f"\n  SB2 error_only done in {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"  SB2 error_only FAILED: {e}")
        import traceback; traceback.print_exc()
        model_results["sb2_error_only_error"] = str(e)

    _save(results_dir, model_name, "all", model_results)
    return model_results


def _save(results_dir, model_name, label, data):
    path = results_dir / f"{model_name}_{label}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Saved: {path}")


def print_summary(all_results):
    """Print cross-model comparison."""
    print("\n" + "=" * 70)
    print("CROSS-MODEL COMPARISON")
    print("=" * 70)

    # SB1
    print("\n--- SB1: Type R Accuracy ---")
    header = f"{'Model':<22} {'T1N4':>6} {'T1N8':>6} {'T2N4':>6} {'T2N8':>6} {'T3N4':>6} {'T3N8':>6}"
    print(header)
    print("-" * len(header))

    for model_name, data in all_results.items():
        if "sb1" not in data:
            continue
        summary = data["sb1"]["summary"]
        row = f"{model_name:<22}"
        for tier in [1, 2, 3]:
            for n in [4, 8]:
                # Handle both int and str keys
                t_key = tier if tier in summary else str(tier)
                if t_key not in summary:
                    row += f" {'—':>5}"
                    continue
                n_key = n if n in summary[t_key] else str(n)
                if n_key not in summary[t_key]:
                    row += f" {'—':>5}"
                    continue
                r_data = summary[t_key][n_key].get("R", {})
                if r_data.get("n_total", 0) > 0:
                    acc = r_data["n_correct"] / r_data["n_total"]
                    row += f" {acc:>5.0%}"
                else:
                    row += f" {'—':>5}"
        print(row)

    # SB2
    print("\n--- SB2: Accuracy per Turn (T2, N=8) ---")
    for model_name, data in all_results.items():
        conditions = {}
        if "sb2_main" in data:
            conditions.update(data["sb2_main"]["summary"])
        if "sb2_error_only" in data:
            conditions.update(data["sb2_error_only"]["summary"])
        if not conditions:
            continue

        print(f"\n  {model_name}:")
        max_turn = max(max(int(t) for t in c.keys()) for c in conditions.values()) + 1

        header = f"    {'Condition':<16}"
        for t in range(max_turn):
            header += f" T{t:>3}"
        header += "  Avg"
        print(header)
        print("    " + "-" * (16 + 5 * max_turn + 6))

        for cond in ["correction", "practice_only", "error_only"]:
            if cond not in conditions:
                continue
            cd = conditions[cond]
            row = f"    {cond:<16}"
            accs = []
            for t in range(max_turn):
                t_key = t if t in cd else str(t)
                if t_key in cd:
                    c = cd[t_key]
                    a = c["n_correct"] / c["n_total"] if c["n_total"] else 0
                    accs.append(a)
                    row += f" {a:>3.0%}"
                else:
                    row += f" {'—':>3}"
            avg = sum(accs) / len(accs) if accs else 0
            row += f"  {avg:>3.0%}"
            print(row)

        # Compute FLR if we have both correction and practice_only
        if "correction" in conditions and "practice_only" in conditions:
            cc = conditions["correction"]
            pp = conditions["practice_only"]
            turns = sorted(set(list(cc.keys()) + list(pp.keys())), key=lambda x: int(x))
            n = len(turns)
            if n >= 4:
                ca = [cc[t]["n_correct"]/cc[t]["n_total"] if cc.get(t,{}).get("n_total") else 0 for t in turns]
                pa = [pp[t]["n_correct"]/pp[t]["n_total"] if pp.get(t,{}).get("n_total") else 0 for t in turns]
                c_slope = sum(ca[n//2:]) / len(ca[n//2:]) - sum(ca[:n//2]) / len(ca[:n//2])
                p_slope = sum(pa[n//2:]) / len(pa[n//2:]) - sum(pa[:n//2]) / len(pa[:n//2])
                flr = c_slope - p_slope
                print(f"    FLR = {flr:+.3f} (correction slope: {c_slope:+.3f}, practice slope: {p_slope:+.3f})")


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("results") / f"p1p2_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results dir: {results_dir}")

    all_results = {}

    # Determine available models
    models = []
    if os.environ.get("GEMINI_API_KEY"):
        models.append(("gemini", "gemini-flash"))
    if check_server(8082):
        models.append(("local", "qwen3.5-35b-a3b"))
    if check_server(8085):
        models.append(("local", "nemotron-nano"))

    if not models:
        print("ERROR: No models available!")
        sys.exit(1)

    print(f"Available models: {[m[1] for m in models]}")

    for model_type, model_name in models:
        try:
            result = run_model(model_type, model_name, results_dir)
            all_results[model_name] = result
        except Exception as e:
            print(f"\nFATAL error with {model_name}: {e}")
            import traceback; traceback.print_exc()

    # Save combined
    combined_path = results_dir / "combined_summary.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print_summary(all_results)

    print(f"\nAll results: {results_dir}")
    return all_results


if __name__ == "__main__":
    main()
