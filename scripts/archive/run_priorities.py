#!/usr/bin/env python3
"""Master experiment runner for Priorities 1-3.

Priority 1: Run SB2 pilot on Gemini Flash, Qwen3.5-35B-A3B, Nemotron Nano
             (same 3 STS instances, T2 N=8, correction + practice_only)
Priority 2: Run error_only condition on same 3 instances for all models
Priority 3: Generate full 25-instance STS dataset for SB2

Usage:
  python scripts/run_priorities.py --priority 1
  python scripts/run_priorities.py --priority 2
  python scripts/run_priorities.py --priority 3
  python scripts/run_priorities.py --all
"""

import argparse
import json
import os
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from lesson.models.registry import get_local_client, get_gemini_client, LOCAL_MODELS, GEMINI_MODELS
from lesson.eval.pilot import run_sb1_pilot
from lesson.eval.sb2_pilot import run_sb2_pilot
from lesson.sts.generator import (
    generate_sts_instance, generate_training_set, subset_training_set,
    generate_test_items, format_training_examples,
)
from lesson.sts.solver import solve


def check_server(port: int, timeout: int = 3) -> bool:
    """Check if a server is healthy."""
    try:
        urllib.request.urlopen(f"http://localhost:{port}/health", timeout=timeout)
        return True
    except Exception:
        return False


def wait_for_server(port: int, name: str, max_wait: int = 300) -> bool:
    """Wait for a server to come up, checking every 5 seconds."""
    print(f"  Waiting for {name} on port {port}...")
    start = time.time()
    while time.time() - start < max_wait:
        if check_server(port):
            elapsed = time.time() - start
            print(f"  {name} ready after {elapsed:.0f}s")
            return True
        time.sleep(5)
    print(f"  TIMEOUT: {name} not ready after {max_wait}s")
    return False


def get_available_models():
    """Get list of available models (checking server health)."""
    models = []

    # Gemini Flash (always available if API key set)
    if os.environ.get("GEMINI_API_KEY"):
        models.append(("gemini", "gemini-flash"))
    else:
        print("WARNING: GEMINI_API_KEY not set, skipping Gemini models")

    # Local models - check which servers are up
    local_targets = [
        ("qwen3.5-35b-a3b", 8082),
        ("nemotron-nano", 8085),
    ]
    for name, port in local_targets:
        if check_server(port):
            models.append(("local", name))
        else:
            print(f"  Server for {name} (port {port}) not ready, will wait...")

    return models


def make_client(model_type, model_name):
    """Create an LLM client."""
    if model_type == "gemini":
        return get_gemini_client(model_name)
    else:
        return get_local_client(model_name)


def save_results(results_dir: Path, model_name: str, label: str, data: dict):
    """Save results to JSON file."""
    results_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{model_name}_{label}.json"
    path = results_dir / filename
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Saved: {path}")
    return path


# ============================================================
# Priority 1: SB2 pilot on all 3 models
# ============================================================

def run_priority_1(results_dir: Path):
    """Run SB2 pilot with correction + practice_only on all available models."""
    print("\n" + "=" * 70)
    print("PRIORITY 1: SB2 Pilot (correction + practice_only) across models")
    print("=" * 70)

    # SB2 parameters matching existing Flash pilot
    SB2_TIER = 2
    SB2_N_INITIAL = 8
    SB2_INSTANCES = 3
    SB2_TURNS = 8
    SB2_CONDITIONS = ["correction", "practice_only"]

    # Also run SB1 for learning curve comparison
    SB1_TIERS = [1, 2, 3]
    SB1_N_VALUES = [4, 8]
    SB1_INSTANCES = 3

    all_summaries = {}

    # Get available models
    models_to_run = []
    if os.environ.get("GEMINI_API_KEY"):
        models_to_run.append(("gemini", "gemini-flash"))

    # Wait for local models
    for name, port in [("qwen3.5-35b-a3b", 8082), ("nemotron-nano", 8085)]:
        if check_server(port):
            models_to_run.append(("local", name))
        elif wait_for_server(port, name, max_wait=300):
            models_to_run.append(("local", name))
        else:
            print(f"  SKIPPING {name} — server not available")

    if not models_to_run:
        print("ERROR: No models available!")
        return {}

    print(f"\nModels: {[m[1] for m in models_to_run]}")

    for model_type, model_name in models_to_run:
        print(f"\n{'─' * 60}")
        print(f"MODEL: {model_name}")
        print(f"{'─' * 60}")

        try:
            client = make_client(model_type, model_name)
        except Exception as e:
            print(f"  ERROR creating client: {e}")
            continue

        model_results = {"model": model_name, "type": model_type}

        # --- SB1: Learning curves ---
        print(f"\n  --- SB1: Learning Curves ({model_name}) ---")
        t0 = time.time()
        try:
            sb1 = run_sb1_pilot(
                client=client,
                tiers=SB1_TIERS,
                n_values=SB1_N_VALUES,
                n_instances=SB1_INSTANCES,
                seq_length=5,
            )
            model_results["sb1"] = sb1
            elapsed = time.time() - t0
            print(f"  SB1 completed in {elapsed:.1f}s")
        except Exception as e:
            print(f"  SB1 FAILED: {e}")
            import traceback; traceback.print_exc()
            model_results["sb1_error"] = str(e)

        # --- SB2: Feedback conditions ---
        print(f"\n  --- SB2: Feedback Conditions ({model_name}) ---")
        t0 = time.time()
        try:
            sb2 = run_sb2_pilot(
                client=client,
                tier=SB2_TIER,
                n_initial_examples=SB2_N_INITIAL,
                n_instances=SB2_INSTANCES,
                n_turns=SB2_TURNS,
                conditions=SB2_CONDITIONS,
            )
            model_results["sb2"] = sb2
            elapsed = time.time() - t0
            print(f"  SB2 completed in {elapsed:.1f}s")
        except Exception as e:
            print(f"  SB2 FAILED: {e}")
            import traceback; traceback.print_exc()
            model_results["sb2_error"] = str(e)

        all_summaries[model_name] = model_results
        save_results(results_dir, model_name, "p1", model_results)

    return all_summaries


# ============================================================
# Priority 2: error_only condition on same instances
# ============================================================

def run_priority_2(results_dir: Path):
    """Run SB2 with error_only condition on same 3 instances."""
    print("\n" + "=" * 70)
    print("PRIORITY 2: SB2 error_only condition")
    print("=" * 70)

    SB2_TIER = 2
    SB2_N_INITIAL = 8
    SB2_INSTANCES = 3
    SB2_TURNS = 8
    SB2_CONDITIONS = ["error_only"]

    all_summaries = {}

    models_to_run = []
    if os.environ.get("GEMINI_API_KEY"):
        models_to_run.append(("gemini", "gemini-flash"))
    for name, port in [("qwen3.5-35b-a3b", 8082), ("nemotron-nano", 8085)]:
        if check_server(port):
            models_to_run.append(("local", name))

    if not models_to_run:
        print("ERROR: No models available!")
        return {}

    print(f"\nModels: {[m[1] for m in models_to_run]}")

    for model_type, model_name in models_to_run:
        print(f"\n{'─' * 60}")
        print(f"MODEL: {model_name} — error_only condition")
        print(f"{'─' * 60}")

        try:
            client = make_client(model_type, model_name)
        except Exception as e:
            print(f"  ERROR creating client: {e}")
            continue

        t0 = time.time()
        try:
            sb2 = run_sb2_pilot(
                client=client,
                tier=SB2_TIER,
                n_initial_examples=SB2_N_INITIAL,
                n_instances=SB2_INSTANCES,
                n_turns=SB2_TURNS,
                conditions=SB2_CONDITIONS,
            )
            elapsed = time.time() - t0
            print(f"  error_only completed in {elapsed:.1f}s")

            result = {"model": model_name, "type": model_type, "sb2_error_only": sb2}
            all_summaries[model_name] = result
            save_results(results_dir, model_name, "p2_error_only", result)

        except Exception as e:
            print(f"  error_only FAILED: {e}")
            import traceback; traceback.print_exc()

    return all_summaries


# ============================================================
# Priority 3: Generate full 25-instance STS dataset
# ============================================================

def run_priority_3(results_dir: Path):
    """Generate and validate the full 25-instance STS dataset for SB2."""
    print("\n" + "=" * 70)
    print("PRIORITY 3: Generate 25-instance STS dataset for SB2")
    print("=" * 70)

    TIER = 2
    N_INSTANCES = 25
    N_EXAMPLES = 8      # training examples shown
    N_TURNS = 12         # test turns per SB2 session
    SEQ_LENGTH = 5
    N_TEST_ITEMS = (3, 1, 1)  # R, E, L per instance for SB1

    dataset_dir = results_dir / "dataset_t2_25inst"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    all_instances = []
    stats = {
        "total_instances": 0,
        "type_e_count": 0,
        "type_l_count": 0,
        "type_r_count": 0,
        "test_sequence_lengths": [],
        "avg_rules_per_instance": 0,
    }

    import random
    total_rules = 0

    for inst_idx in range(N_INSTANCES):
        seed = TIER * 10_000 + inst_idx
        rng = random.Random(seed + 5000)

        # Generate STS instance
        sts = generate_sts_instance(tier=TIER, seed=seed)
        full_training = generate_training_set(sts, n_max=max(N_EXAMPLES, 32))
        training = subset_training_set(full_training, N_EXAMPLES)

        # Generate SB1 test items
        test_items = generate_test_items(
            sts=sts,
            training_examples=training,
            n_per_type=N_TEST_ITEMS,
            seq_length=SEQ_LENGTH,
            seed=seed,
        )

        # Generate SB2 test sequence (novel inputs for multi-turn)
        from lesson.eval.sb2_pilot import _generate_test_sequence
        test_sequence = _generate_test_sequence(
            sts=sts,
            training_examples=training,
            n_turns=N_TURNS,
            seq_length=SEQ_LENGTH,
            rng=rng,
        )

        # Collect stats
        total_rules += len(sts.rules)
        for item in test_items:
            if item.item_type.value == "E":
                stats["type_e_count"] += 1
            elif item.item_type.value == "L":
                stats["type_l_count"] += 1
            else:
                stats["type_r_count"] += 1
        stats["test_sequence_lengths"].append(len(test_sequence))

        instance_data = {
            "instance_idx": inst_idx,
            "seed": seed,
            "sts_id": sts.id,
            "tier": TIER,
            "alphabet": sts.alphabet,
            "n_rules": len(sts.rules),
            "rules": [{"type": r.rule_type.value, "spec": r.spec} for r in sts.rules],
            "exceptions": [
                {"trigger": e.trigger, "position": e.position, "override": e.output_override}
                for e in sts.exceptions
            ],
            "training_examples": [
                {"input": ex.input_seq, "output": ex.output_seq}
                for ex in training
            ],
            "test_items": [
                {
                    "input": item.input_seq,
                    "correct_output": item.correct_output,
                    "type": item.item_type.value,
                    "partial_rule_answer": item.partial_rule_answer,
                }
                for item in test_items
            ],
            "test_sequence": [
                {"input": inp, "correct_output": out}
                for inp, out in test_sequence
            ],
        }
        all_instances.append(instance_data)

        print(f"  Instance {inst_idx:2d}: {sts.id} | "
              f"{len(sts.rules)} rules | "
              f"{len(test_items)} test items (E={sum(1 for i in test_items if i.item_type.value=='E')}, "
              f"L={sum(1 for i in test_items if i.item_type.value=='L')}) | "
              f"{len(test_sequence)}/{N_TURNS} SB2 turns")

    stats["total_instances"] = N_INSTANCES
    stats["avg_rules_per_instance"] = total_rules / N_INSTANCES
    stats["min_test_sequence"] = min(stats["test_sequence_lengths"])
    stats["max_test_sequence"] = max(stats["test_sequence_lengths"])
    stats["avg_test_sequence"] = sum(stats["test_sequence_lengths"]) / len(stats["test_sequence_lengths"])

    # Save dataset
    dataset = {
        "metadata": {
            "tier": TIER,
            "n_instances": N_INSTANCES,
            "n_training_examples": N_EXAMPLES,
            "n_sb2_turns": N_TURNS,
            "seq_length": SEQ_LENGTH,
            "test_items_per_instance": list(N_TEST_ITEMS),
            "generated_at": datetime.now().isoformat(),
        },
        "stats": stats,
        "instances": all_instances,
    }

    dataset_path = dataset_dir / "dataset.json"
    with open(dataset_path, "w") as f:
        json.dump(dataset, f, indent=2, default=str)
    print(f"\n  Dataset saved: {dataset_path}")

    # Print summary
    print(f"\n  --- Dataset Summary ---")
    print(f"  Instances: {stats['total_instances']}")
    print(f"  Avg rules/instance: {stats['avg_rules_per_instance']:.1f}")
    print(f"  SB1 items: {stats['type_r_count']}R + {stats['type_e_count']}E + {stats['type_l_count']}L = {stats['type_r_count']+stats['type_e_count']+stats['type_l_count']}")
    print(f"  SB2 turns: min={stats['min_test_sequence']}, max={stats['max_test_sequence']}, avg={stats['avg_test_sequence']:.1f}")
    print(f"  Type E feasibility: {stats['type_e_count']}/{N_INSTANCES} ({stats['type_e_count']/N_INSTANCES*100:.0f}%)")

    return dataset


# ============================================================
# Results aggregation and comparison
# ============================================================

def print_comparison(p1_results: dict, p2_results: dict):
    """Print cross-model comparison tables."""
    print("\n" + "=" * 70)
    print("CROSS-MODEL COMPARISON")
    print("=" * 70)

    # SB1 comparison
    print("\n--- SB1: Type R Accuracy by Tier and N ---")
    header = f"{'Model':<22} {'T1 N=4':>8} {'T1 N=8':>8} {'T2 N=4':>8} {'T2 N=8':>8} {'T3 N=4':>8} {'T3 N=8':>8}"
    print(header)
    print("-" * len(header))

    for model_name, data in p1_results.items():
        if "sb1" not in data:
            continue
        summary = data["sb1"]["summary"]
        row = f"{model_name:<22}"
        for tier in [1, 2, 3]:
            for n in [4, 8]:
                t_key = str(tier) if str(tier) in summary else tier
                if t_key in summary:
                    n_key = str(n) if str(n) in summary[t_key] else n
                    if n_key in summary[t_key]:
                        r_data = summary[t_key][n_key].get("R", {})
                        if r_data.get("n_total", 0) > 0:
                            acc = r_data["n_correct"] / r_data["n_total"]
                            row += f" {acc:>7.0%}"
                        else:
                            row += f" {'—':>7}"
                    else:
                        row += f" {'—':>7}"
                else:
                    row += f" {'—':>7}"
        print(row)

    # SB2 comparison: all conditions
    print("\n--- SB2: Accuracy by Turn (all conditions, T2 N=8) ---")
    all_models = set(list(p1_results.keys()) + list(p2_results.keys()))
    for model_name in sorted(all_models):
        print(f"\n  {model_name}:")
        # Collect all conditions for this model
        conditions = {}
        if model_name in p1_results and "sb2" in p1_results[model_name]:
            conditions.update(p1_results[model_name]["sb2"]["summary"])
        if model_name in p2_results and "sb2_error_only" in p2_results[model_name]:
            conditions.update(p2_results[model_name]["sb2_error_only"]["summary"])

        if not conditions:
            print("    (no SB2 data)")
            continue

        # Find max turns
        max_turn = max(
            max(int(t) for t in cond_data.keys())
            for cond_data in conditions.values()
        ) + 1

        header = f"    {'Condition':<16}"
        for t in range(max_turn):
            header += f" {'T'+str(t):>5}"
        print(header)
        print("    " + "-" * (16 + 6 * max_turn))

        for cond_name in ["correction", "practice_only", "error_only"]:
            if cond_name not in conditions:
                continue
            cond_data = conditions[cond_name]
            row = f"    {cond_name:<16}"
            for t in range(max_turn):
                t_key = str(t) if str(t) in cond_data else t
                if t_key in cond_data:
                    counts = cond_data[t_key]
                    acc = counts["n_correct"] / counts["n_total"] if counts["n_total"] else 0
                    row += f" {acc:>4.0%} "
                else:
                    row += f" {'—':>4} "
            print(row)


def generate_observations(p1_results: dict, p2_results: dict, dataset: dict, results_dir: Path):
    """Generate observations markdown from results."""
    lines = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines.append(f"\n### {timestamp}: Priority 1-3 Results (Automated Run)")
    lines.append("")

    # SB1 results
    lines.append("#### SB1: Learning Curves (All Models)")
    lines.append("")
    lines.append("| Model | T1 N=4 | T1 N=8 | T2 N=4 | T2 N=8 | T3 N=4 | T3 N=8 |")
    lines.append("|-------|--------|--------|--------|--------|--------|--------|")

    for model_name, data in p1_results.items():
        if "sb1" not in data:
            continue
        summary = data["sb1"]["summary"]
        row = f"| {model_name}"
        for tier in [1, 2, 3]:
            for n in [4, 8]:
                t_key = str(tier) if str(tier) in summary else tier
                if t_key in summary:
                    n_key = str(n) if str(n) in summary[t_key] else n
                    if n_key in summary[t_key]:
                        r_data = summary[t_key][n_key].get("R", {})
                        if r_data.get("n_total", 0) > 0:
                            acc = r_data["n_correct"] / r_data["n_total"]
                            row += f" | {r_data['n_correct']}/{r_data['n_total']} ({acc:.0%})"
                        else:
                            row += " | —"
                    else:
                        row += " | —"
                else:
                    row += " | —"
        row += " |"
        lines.append(row)

    # SB2 results
    lines.append("")
    lines.append("#### SB2: Feedback Conditions (T2, N=8, 8 turns)")
    lines.append("")

    all_models = set(list(p1_results.keys()) + list(p2_results.keys()))
    for model_name in sorted(all_models):
        lines.append(f"**{model_name}:**")
        lines.append("")

        conditions = {}
        if model_name in p1_results and "sb2" in p1_results[model_name]:
            conditions.update(p1_results[model_name]["sb2"]["summary"])
        if model_name in p2_results and "sb2_error_only" in p2_results[model_name]:
            conditions.update(p2_results[model_name]["sb2_error_only"]["summary"])

        if not conditions:
            lines.append("  (no SB2 data)")
            lines.append("")
            continue

        max_turn = max(
            max(int(t) for t in cond_data.keys())
            for cond_data in conditions.values()
        ) + 1

        header = "| Condition |"
        for t in range(max_turn):
            header += f" T{t} |"
        lines.append(header)
        sep = "|-----------|"
        for t in range(max_turn):
            sep += "-----|"
        lines.append(sep)

        for cond_name in ["correction", "practice_only", "error_only"]:
            if cond_name not in conditions:
                continue
            cond_data = conditions[cond_name]
            row = f"| {cond_name} |"
            for t in range(max_turn):
                t_key = str(t) if str(t) in cond_data else t
                if t_key in cond_data:
                    counts = cond_data[t_key]
                    acc = counts["n_correct"] / counts["n_total"] if counts["n_total"] else 0
                    row += f" {acc:.0%} |"
                else:
                    row += " — |"
            lines.append(row)
        lines.append("")

    # Dataset stats
    if dataset:
        lines.append("#### Priority 3: 25-Instance Dataset Generated")
        lines.append("")
        stats = dataset.get("stats", {})
        meta = dataset.get("metadata", {})
        lines.append(f"- Tier: {meta.get('tier', '?')}")
        lines.append(f"- Instances: {stats.get('total_instances', '?')}")
        lines.append(f"- Avg rules/instance: {stats.get('avg_rules_per_instance', 0):.1f}")
        lines.append(f"- SB1 items: {stats.get('type_r_count', 0)}R + {stats.get('type_e_count', 0)}E + {stats.get('type_l_count', 0)}L")
        lines.append(f"- SB2 turns per instance: min={stats.get('min_test_sequence', '?')}, max={stats.get('max_test_sequence', '?')}, avg={stats.get('avg_test_sequence', 0):.1f}")
        lines.append(f"- Type E feasibility: {stats.get('type_e_count', 0)}/{stats.get('total_instances', 25)} ({stats.get('type_e_count', 0)/max(stats.get('total_instances', 1),1)*100:.0f}%)")
        lines.append("")

    # Key findings
    lines.append("#### Key Findings")
    lines.append("")

    # Compute FLR for each model
    for model_name in sorted(all_models):
        conditions = {}
        if model_name in p1_results and "sb2" in p1_results[model_name]:
            conditions.update(p1_results[model_name]["sb2"]["summary"])

        if "correction" in conditions and "practice_only" in conditions:
            corr_accs = []
            prac_accs = []
            for t_key in sorted(conditions["correction"].keys(), key=int):
                c = conditions["correction"][t_key]
                p = conditions["practice_only"][t_key]
                corr_accs.append(c["n_correct"] / c["n_total"] if c["n_total"] else 0)
                prac_accs.append(p["n_correct"] / p["n_total"] if p["n_total"] else 0)

            # Simple slope comparison (last 3 turns vs first 3 turns)
            n = len(corr_accs)
            if n >= 4:
                corr_late = sum(corr_accs[n//2:]) / len(corr_accs[n//2:])
                corr_early = sum(corr_accs[:n//2]) / len(corr_accs[:n//2])
                prac_late = sum(prac_accs[n//2:]) / len(prac_accs[n//2:])
                prac_early = sum(prac_accs[:n//2]) / len(prac_accs[:n//2])
                corr_slope = corr_late - corr_early
                prac_slope = prac_late - prac_early
                flr = corr_slope - prac_slope
                lines.append(f"- **{model_name}** FLR ≈ {flr:+.3f} (correction slope: {corr_slope:+.3f}, practice slope: {prac_slope:+.3f})")

    lines.append("")

    return "\n".join(lines)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Run Priorities 1-3")
    parser.add_argument("--priority", type=int, choices=[1, 2, 3],
                        help="Run specific priority (default: all)")
    parser.add_argument("--all", action="store_true",
                        help="Run all priorities sequentially")
    parser.add_argument("--obs-file", default="docs/observations.md",
                        help="Path to observations file")
    args = parser.parse_args()

    run_all = args.all or args.priority is None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("results") / f"priorities_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Results directory: {results_dir}")

    p1_results = {}
    p2_results = {}
    dataset = {}

    if run_all or args.priority == 1:
        p1_results = run_priority_1(results_dir)

    if run_all or args.priority == 2:
        p2_results = run_priority_2(results_dir)

    if run_all or args.priority == 3:
        dataset = run_priority_3(results_dir)

    # Print comparison
    if p1_results or p2_results:
        print_comparison(p1_results, p2_results)

    # Generate and append observations
    if p1_results or p2_results or dataset:
        obs_text = generate_observations(p1_results, p2_results, dataset, results_dir)
        obs_path = Path(args.obs_file)

        # Read existing content
        existing = ""
        if obs_path.exists():
            with open(obs_path) as f:
                existing = f.read()

        # Append new observations
        with open(obs_path, "w") as f:
            f.write(existing.rstrip() + "\n" + obs_text + "\n")

        print(f"\nObservations appended to: {obs_path}")
        print(f"All results saved to: {results_dir}")

    # Save combined summary
    combined = {
        "timestamp": timestamp,
        "priority_1": p1_results,
        "priority_2": p2_results,
        "priority_3_stats": dataset.get("stats", {}) if dataset else {},
    }
    combined_path = results_dir / "combined_summary.json"
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2, default=str)
    print(f"Combined summary: {combined_path}")


if __name__ == "__main__":
    main()
