#!/usr/bin/env python3
"""Production evaluation runner for LESSON-Bench v11.0.

Two-phase pipeline:
  Phase 1 (SB1 Scan): Run SB1 on ALL models to map accuracy landscape.
           Filter for models with T2 N=8 accuracy between 15-70%.
  Phase 2 (SB2 Deep): Run full SB2 (4 core conditions, 12 turns) on filtered models.

Supports: LM Studio (local), Gemini (API), OpenRouter (API).

Usage:
    # Phase 1: Scan all models
    python scripts/run_production.py --phase scan

    # Phase 2: Deep evaluation (auto-selects from Phase 1 results)
    python scripts/run_production.py --phase deep --scan-results results/scan_XXXX/

    # Both phases
    python scripts/run_production.py --phase both

    # Extended SB1 with N=16,32 for specific models
    python scripts/run_production.py --phase extended-sb1 --models gemini-flash

    # Run mechanistic probes on specific models
    python scripts/run_production.py --phase probes --models gemini-flash
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------

def discover_models() -> List[Tuple[str, str]]:
    """Return list of (provider, model_name) for all available models.

    Checks:
    - LM Studio: GET http://localhost:1234/v1/models
    - Gemini: GEMINI_API_KEY env var
    - OpenRouter: OPENROUTER_API_KEY env var
    """
    models: List[Tuple[str, str]] = []

    # LM Studio
    try:
        from lesson.models.lmstudio import check_lmstudio_server, LMSTUDIO_MODEL_CONFIGS
        if check_lmstudio_server():
            for name in LMSTUDIO_MODEL_CONFIGS:
                models.append(("lmstudio", name))
            if not LMSTUDIO_MODEL_CONFIGS:
                # No pre-configured models, but server is up — add a generic entry
                models.append(("lmstudio", "lmstudio-default"))
    except ImportError:
        pass

    # Gemini
    if os.environ.get("GEMINI_API_KEY"):
        try:
            from lesson.models.gemini import GEMINI_MODEL_CONFIGS
            for name in GEMINI_MODEL_CONFIGS:
                models.append(("gemini", name))
        except ImportError:
            pass

    # OpenRouter
    if os.environ.get("OPENROUTER_API_KEY"):
        try:
            from lesson.models.openrouter import OPENROUTER_MODEL_CONFIGS
            for name in OPENROUTER_MODEL_CONFIGS:
                models.append(("openrouter", name))
        except ImportError:
            pass

    return models


# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------

def get_client(provider: str, model_name: str):
    """Create the appropriate LLMClient for a provider/model pair."""
    if provider == "gemini":
        from lesson.models.registry import get_gemini_client
        return get_gemini_client(model_name)
    elif provider == "lmstudio":
        from lesson.models.registry import get_lmstudio_client
        return get_lmstudio_client(model_name)
    elif provider == "openrouter":
        from lesson.models.registry import get_openrouter_client
        return get_openrouter_client(model_name)
    elif provider == "local":
        from lesson.models.registry import get_local_client
        return get_local_client(model_name)
    else:
        raise ValueError(f"Unknown provider: {provider!r}")


# ---------------------------------------------------------------------------
# Incremental save helper
# ---------------------------------------------------------------------------

def _save(results_dir: Path, model_name: str, label: str, data: Any) -> None:
    """Save results JSON incrementally — safe model name for filesystem."""
    safe_name = model_name.replace("/", "_").replace(":", "_")
    path = results_dir / f"{safe_name}_{label}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Phase 1: SB1 Scan
# ---------------------------------------------------------------------------

def run_scan(
    models: List[Tuple[str, str]],
    results_dir: Path,
    n_instances: int = 3,
) -> Dict[str, Any]:
    """Phase 1: Run SB1 on all models to map accuracy landscape.

    Args:
        models:      List of (provider, model_name) pairs.
        results_dir: Directory to save results.
        n_instances: Number of STS instances per cell.

    Returns:
        Dict mapping model_name -> result dict.
    """
    from lesson.eval.pilot import run_sb1_pilot

    all_results: Dict[str, Any] = {}

    for idx, (provider, model_name) in enumerate(models, 1):
        print(f"\n{'=' * 60}")
        print(f"SCAN [{idx}/{len(models)}] {provider}:{model_name}")
        print(f"{'=' * 60}")

        t0 = time.time()
        try:
            client = get_client(provider, model_name)
            sb1 = run_sb1_pilot(
                client=client,
                tiers=[1, 2],
                n_values=[4, 8],
                n_instances=n_instances,
                seq_length=5,
            )
            elapsed = time.time() - t0
            result = {
                "model": model_name,
                "provider": provider,
                "sb1": sb1,
                "elapsed_s": elapsed,
            }
            all_results[model_name] = result
            print(f"\n  SB1 scan done in {elapsed:.1f}s")
        except Exception as e:
            elapsed = time.time() - t0
            print(f"\n  SB1 scan FAILED in {elapsed:.1f}s: {e}")
            import traceback; traceback.print_exc()
            all_results[model_name] = {
                "model": model_name,
                "provider": provider,
                "sb1_error": str(e),
                "elapsed_s": elapsed,
            }

        # Save incrementally
        _save(results_dir, model_name, "scan", all_results[model_name])

    # Save combined scan results
    combined_path = results_dir / "scan_combined.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Combined scan results: {combined_path}")

    # Print summary table and filter
    _print_scan_summary(all_results)

    # Filter: keep models where T2 N=8 Type R accuracy is between 15% and 70%
    filtered = _filter_models(all_results)

    # Save filtered model list
    filtered_path = results_dir / "filtered_models.json"
    with open(filtered_path, "w") as f:
        json.dump(filtered, f, indent=2, default=str)
    print(f"\n  Filtered models saved to: {filtered_path}")

    return all_results


def _get_type_r_accuracy(summary: Dict, tier: int, n: int) -> Optional[float]:
    """Extract Type R accuracy from SB1 summary, handling int/str keys."""
    t_key = tier if tier in summary else str(tier)
    if t_key not in summary:
        return None
    n_key = n if n in summary[t_key] else str(n)
    if n_key not in summary[t_key]:
        return None
    r_data = summary[t_key][n_key].get("R", {})
    if r_data.get("n_total", 0) > 0:
        return r_data["n_correct"] / r_data["n_total"]
    return None


def _print_scan_summary(all_results: Dict[str, Any]) -> None:
    """Print Phase 1 summary table with PASS/FAIL filter indicator."""
    print("\n" + "=" * 70)
    print("PHASE 1 SCAN SUMMARY")
    print("=" * 70)

    header = f"{'Model':<30} {'T1N4':>6} {'T1N8':>6} {'T2N4':>6} {'T2N8':>6} {'Filter':>8}"
    print(header)
    print("-" * len(header))

    for model_name, data in all_results.items():
        if "sb1" not in data:
            print(f"{model_name:<30} {'ERROR':>6} {'':>6} {'':>6} {'':>6} {'FAIL':>8}")
            continue
        summary = data["sb1"]["summary"]
        row = f"{model_name:<30}"
        for tier in [1, 2]:
            for n in [4, 8]:
                acc = _get_type_r_accuracy(summary, tier, n)
                if acc is not None:
                    row += f" {acc:>5.0%}"
                else:
                    row += f" {'---':>5}"

        # Filter check: T2 N=8 between 15-70%
        t2n8 = _get_type_r_accuracy(summary, 2, 8)
        if t2n8 is not None and 0.15 <= t2n8 <= 0.70:
            row += f" {'PASS':>8}"
        else:
            row += f" {'FAIL':>8}"

        print(row)


def _filter_models(all_results: Dict[str, Any]) -> List[Dict[str, str]]:
    """Filter models where T2 N=8 Type R accuracy is between 15% and 70%.

    Returns list of {provider, model_name, t2n8_accuracy} dicts.
    """
    filtered = []
    for model_name, data in all_results.items():
        if "sb1" not in data:
            continue
        summary = data["sb1"]["summary"]
        t2n8 = _get_type_r_accuracy(summary, 2, 8)
        if t2n8 is not None and 0.15 <= t2n8 <= 0.70:
            filtered.append({
                "provider": data["provider"],
                "model_name": model_name,
                "t2n8_accuracy": t2n8,
            })
            print(f"  PASS: {model_name} (T2 N=8 = {t2n8:.0%})")
        else:
            acc_str = f"{t2n8:.0%}" if t2n8 is not None else "N/A"
            print(f"  FAIL: {model_name} (T2 N=8 = {acc_str})")

    print(f"\n  {len(filtered)} models passed the filter")
    return filtered


# ---------------------------------------------------------------------------
# Phase 2: SB2 Deep evaluation
# ---------------------------------------------------------------------------

def run_deep(
    models: List[Tuple[str, str]],
    results_dir: Path,
    n_instances: int = 3,
    n_turns: int = 12,
    tier: int = 2,
) -> Dict[str, Any]:
    """Phase 2: Run full SB2 on filtered models.

    Args:
        models:      List of (provider, model_name) pairs.
        results_dir: Directory to save results.
        n_instances: Number of STS instances per condition.
        n_turns:     Number of SB2 turns per instance.
        tier:        STS tier.

    Returns:
        Dict mapping model_name -> result dict.
    """
    from lesson.eval.sb2_pilot import run_sb2_pilot

    conditions = ["correction", "practice_only", "error_only", "no_feedback"]
    all_results: Dict[str, Any] = {}

    for idx, (provider, model_name) in enumerate(models, 1):
        print(f"\n{'=' * 60}")
        print(f"DEEP [{idx}/{len(models)}] {provider}:{model_name}")
        print(f"{'=' * 60}")

        model_results: Dict[str, Any] = {
            "model": model_name,
            "provider": provider,
        }

        for condition in conditions:
            print(f"\n--- SB2: {condition} (T{tier}, N=8, {n_turns} turns, "
                  f"{n_instances} instances) ---")
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
                )
                elapsed = time.time() - t0
                model_results[f"sb2_{condition}"] = sb2
                model_results[f"sb2_{condition}_elapsed_s"] = elapsed
                print(f"\n  SB2 {condition} done in {elapsed:.1f}s")
            except Exception as e:
                elapsed = time.time() - t0
                print(f"\n  SB2 {condition} FAILED in {elapsed:.1f}s: {e}")
                import traceback; traceback.print_exc()
                model_results[f"sb2_{condition}_error"] = str(e)
                model_results[f"sb2_{condition}_elapsed_s"] = elapsed

            # Save incrementally after each condition
            _save(results_dir, model_name, f"deep_{condition}", model_results)

        all_results[model_name] = model_results
        # Save full model result
        _save(results_dir, model_name, "deep_all", model_results)

    # Save combined
    combined_path = results_dir / "deep_combined.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Combined deep results: {combined_path}")

    _print_deep_summary(all_results)
    return all_results


def _print_deep_summary(all_results: Dict[str, Any]) -> None:
    """Print Phase 2 cross-model per-condition per-turn accuracy table."""
    print("\n" + "=" * 70)
    print("PHASE 2 DEEP SUMMARY — Accuracy per Turn (T2, N=8)")
    print("=" * 70)

    conditions_order = ["correction", "practice_only", "error_only", "no_feedback"]

    for model_name, data in all_results.items():
        # Collect all condition summaries
        cond_summaries: Dict[str, Dict] = {}
        for cond in conditions_order:
            key = f"sb2_{cond}"
            if key in data and "summary" in data[key]:
                cond_summary = data[key]["summary"]
                if cond in cond_summary:
                    cond_summaries[cond] = cond_summary[cond]

        if not cond_summaries:
            continue

        print(f"\n  {model_name}:")
        max_turn = 0
        for cond_data in cond_summaries.values():
            for t_key in cond_data:
                max_turn = max(max_turn, int(t_key) + 1)

        header = f"    {'Condition':<22}"
        for t in range(max_turn):
            header += f" T{t:>3}"
        header += "  Avg"
        print(header)
        print("    " + "-" * (22 + 5 * max_turn + 6))

        for cond in conditions_order:
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
            row += f"  {avg:>3.0%}"
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
                    f"(correction slope: {c_slope:+.3f}, "
                    f"practice slope: {p_slope:+.3f})"
                )


# ---------------------------------------------------------------------------
# Phase 3: Extended SB1 (learning curve extension)
# ---------------------------------------------------------------------------

def run_extended_sb1(
    models: List[Tuple[str, str]],
    results_dir: Path,
    n_instances: int = 3,
    tier: int = 2,
) -> Dict[str, Any]:
    """Phase 3: Run SB1 with extended N values [4, 8, 16, 32] on specified models.

    Args:
        models:      List of (provider, model_name) pairs.
        results_dir: Directory to save results.
        n_instances: Number of STS instances per cell.
        tier:        STS tier (default 2).

    Returns:
        Dict mapping model_name -> result dict.
    """
    from lesson.eval.pilot import run_sb1_pilot

    all_results: Dict[str, Any] = {}

    for idx, (provider, model_name) in enumerate(models, 1):
        print(f"\n{'=' * 60}")
        print(f"EXTENDED-SB1 [{idx}/{len(models)}] {provider}:{model_name}")
        print(f"{'=' * 60}")

        t0 = time.time()
        try:
            client = get_client(provider, model_name)
            sb1 = run_sb1_pilot(
                client=client,
                tiers=[tier],
                n_values=[4, 8, 16, 32],
                n_instances=n_instances,
                seq_length=5,
            )
            elapsed = time.time() - t0
            result = {
                "model": model_name,
                "provider": provider,
                "sb1_extended": sb1,
                "elapsed_s": elapsed,
            }
            all_results[model_name] = result
            print(f"\n  Extended SB1 done in {elapsed:.1f}s")
        except Exception as e:
            elapsed = time.time() - t0
            print(f"\n  Extended SB1 FAILED in {elapsed:.1f}s: {e}")
            import traceback; traceback.print_exc()
            all_results[model_name] = {
                "model": model_name,
                "provider": provider,
                "sb1_extended_error": str(e),
                "elapsed_s": elapsed,
            }

        _save(results_dir, model_name, "extended_sb1", all_results[model_name])

    # Save combined
    combined_path = results_dir / "extended_sb1_combined.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Combined extended SB1 results: {combined_path}")

    # Print summary
    _print_extended_sb1_summary(all_results, tier)
    return all_results


def _print_extended_sb1_summary(all_results: Dict[str, Any], tier: int) -> None:
    """Print extended SB1 summary with N=4,8,16,32."""
    print("\n" + "=" * 70)
    print(f"EXTENDED SB1 SUMMARY — Tier {tier}, Type R Accuracy")
    print("=" * 70)

    header = f"{'Model':<30} {'N=4':>6} {'N=8':>6} {'N=16':>6} {'N=32':>6}"
    print(header)
    print("-" * len(header))

    for model_name, data in all_results.items():
        if "sb1_extended" not in data:
            print(f"{model_name:<30} {'ERROR':>6}")
            continue
        summary = data["sb1_extended"]["summary"]
        row = f"{model_name:<30}"
        for n in [4, 8, 16, 32]:
            acc = _get_type_r_accuracy(summary, tier, n)
            if acc is not None:
                row += f" {acc:>5.0%}"
            else:
                row += f" {'---':>5}"
        print(row)


# ---------------------------------------------------------------------------
# Phase 4: Mechanistic probes
# ---------------------------------------------------------------------------

def run_probes(
    models: List[Tuple[str, str]],
    results_dir: Path,
    n_instances: int = 3,
    n_turns: int = 12,
    tier: int = 2,
) -> Dict[str, Any]:
    """Phase 4: Run SB2 with mechanistic probe conditions.

    Conditions: clean_context, prompted_correction, structured_correction,
                reformatted_correction.

    Args:
        models:      List of (provider, model_name) pairs.
        results_dir: Directory to save results.
        n_instances: Number of STS instances per condition.
        n_turns:     Number of SB2 turns per instance.
        tier:        STS tier.

    Returns:
        Dict mapping model_name -> result dict.
    """
    from lesson.eval.sb2_pilot import run_sb2_pilot

    probe_conditions = [
        "clean_context",
        "prompted_correction",
        "structured_correction",
        "reformatted_correction",
    ]
    all_results: Dict[str, Any] = {}

    for idx, (provider, model_name) in enumerate(models, 1):
        print(f"\n{'=' * 60}")
        print(f"PROBES [{idx}/{len(models)}] {provider}:{model_name}")
        print(f"{'=' * 60}")

        model_results: Dict[str, Any] = {
            "model": model_name,
            "provider": provider,
        }

        for condition in probe_conditions:
            print(f"\n--- Probe: {condition} (T{tier}, N=8, {n_turns} turns, "
                  f"{n_instances} instances) ---")
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
                )
                elapsed = time.time() - t0
                model_results[f"probe_{condition}"] = sb2
                model_results[f"probe_{condition}_elapsed_s"] = elapsed
                print(f"\n  Probe {condition} done in {elapsed:.1f}s")
            except Exception as e:
                elapsed = time.time() - t0
                print(f"\n  Probe {condition} FAILED in {elapsed:.1f}s: {e}")
                import traceback; traceback.print_exc()
                model_results[f"probe_{condition}_error"] = str(e)
                model_results[f"probe_{condition}_elapsed_s"] = elapsed

            # Save incrementally after each condition
            _save(results_dir, model_name, f"probe_{condition}", model_results)

        all_results[model_name] = model_results
        _save(results_dir, model_name, "probes_all", model_results)

    # Save combined
    combined_path = results_dir / "probes_combined.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Combined probes results: {combined_path}")

    _print_probes_summary(all_results)
    return all_results


def _print_probes_summary(all_results: Dict[str, Any]) -> None:
    """Print mechanistic probes summary."""
    print("\n" + "=" * 70)
    print("MECHANISTIC PROBES SUMMARY — Accuracy per Turn (T2, N=8)")
    print("=" * 70)

    probe_conditions = [
        "clean_context",
        "prompted_correction",
        "structured_correction",
        "reformatted_correction",
    ]

    for model_name, data in all_results.items():
        cond_summaries: Dict[str, Dict] = {}
        for cond in probe_conditions:
            key = f"probe_{cond}"
            if key in data and "summary" in data[key]:
                cond_summary = data[key]["summary"]
                if cond in cond_summary:
                    cond_summaries[cond] = cond_summary[cond]

        if not cond_summaries:
            continue

        print(f"\n  {model_name}:")
        max_turn = 0
        for cond_data in cond_summaries.values():
            for t_key in cond_data:
                max_turn = max(max_turn, int(t_key) + 1)

        header = f"    {'Condition':<28}"
        for t in range(max_turn):
            header += f" T{t:>3}"
        header += "  Avg"
        print(header)
        print("    " + "-" * (28 + 5 * max_turn + 6))

        for cond in probe_conditions:
            if cond not in cond_summaries:
                continue
            cd = cond_summaries[cond]
            row = f"    {cond:<28}"
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
            row += f"  {avg:>3.0%}"
            print(row)


# ---------------------------------------------------------------------------
# Model list parsing
# ---------------------------------------------------------------------------

def parse_model_list(
    models_arg: Optional[str],
    all_models: List[Tuple[str, str]],
) -> List[Tuple[str, str]]:
    """Parse --models flag into a list of (provider, model_name) pairs.

    If models_arg is None, returns all_models.
    Otherwise, filters all_models to those whose model_name matches any
    of the comma-separated names in models_arg.
    """
    if models_arg is None:
        return all_models

    requested = {m.strip() for m in models_arg.split(",") if m.strip()}
    matched = [(p, m) for p, m in all_models if m in requested]

    # For names not found in discovered models, try to infer provider
    matched_names = {m for _, m in matched}
    for name in requested - matched_names:
        # Heuristic: guess provider from name prefix or config membership
        provider = _guess_provider(name)
        if provider:
            matched.append((provider, name))
            print(f"  Note: {name} not in discovered models, using provider={provider}")
        else:
            print(f"  WARNING: Could not find model {name!r} in any provider config")

    return matched


def _guess_provider(model_name: str) -> Optional[str]:
    """Try to guess the provider for an unknown model name."""
    try:
        from lesson.models.lmstudio import LMSTUDIO_MODEL_CONFIGS
        if model_name in LMSTUDIO_MODEL_CONFIGS:
            return "lmstudio"
    except ImportError:
        pass

    try:
        from lesson.models.gemini import GEMINI_MODEL_CONFIGS
        if model_name in GEMINI_MODEL_CONFIGS:
            return "gemini"
    except ImportError:
        pass

    try:
        from lesson.models.openrouter import OPENROUTER_MODEL_CONFIGS
        if model_name in OPENROUTER_MODEL_CONFIGS:
            return "openrouter"
    except ImportError:
        pass

    # Prefix heuristics
    if model_name.startswith("gemini"):
        return "gemini"
    if model_name.startswith("lmstudio"):
        return "lmstudio"
    if model_name.startswith("or-"):
        return "openrouter"

    return None


def load_filtered_models(scan_results_dir: str) -> List[Tuple[str, str]]:
    """Load filtered model list from a Phase 1 scan results directory."""
    filtered_path = Path(scan_results_dir) / "filtered_models.json"
    if not filtered_path.exists():
        print(f"ERROR: {filtered_path} not found")
        sys.exit(1)

    with open(filtered_path) as f:
        filtered = json.load(f)

    models = [(entry["provider"], entry["model_name"]) for entry in filtered]
    print(f"Loaded {len(models)} filtered models from {filtered_path}")
    for provider, name in models:
        print(f"  {provider}:{name}")
    return models


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Production evaluation runner for LESSON-Bench v11.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--phase",
        required=True,
        choices=["scan", "deep", "both", "extended-sb1", "probes"],
        help="Evaluation phase to run.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model names (overrides auto-discovery).",
    )
    parser.add_argument(
        "--scan-results",
        type=str,
        default=None,
        help="Path to Phase 1 scan results directory (for Phase 2 auto-selection).",
    )
    parser.add_argument(
        "--n-instances",
        type=int,
        default=3,
        help="Number of STS instances per cell (default 3 for pilot, 25 for production).",
    )
    parser.add_argument(
        "--n-turns",
        type=int,
        default=12,
        help="Number of SB2 turns (default 12).",
    )
    parser.add_argument(
        "--tier",
        type=int,
        default=2,
        help="STS tier (default 2).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override results directory.",
    )
    args = parser.parse_args()

    # Set up results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        results_dir = Path(args.output_dir)
    else:
        results_dir = Path("results") / f"production_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results dir: {results_dir}")

    # Save run configuration
    config = {
        "phase": args.phase,
        "models_arg": args.models,
        "scan_results": args.scan_results,
        "n_instances": args.n_instances,
        "n_turns": args.n_turns,
        "tier": args.tier,
        "timestamp": timestamp,
    }
    config_path = results_dir / "run_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # Discover available models
    all_available = discover_models()
    if all_available:
        print(f"\nDiscovered models: {[(p, m) for p, m in all_available]}")
    else:
        print("\nNo models auto-discovered (will need --models or --scan-results)")

    phase_start = time.time()

    # --- Phase dispatch ---
    if args.phase == "scan":
        models = parse_model_list(args.models, all_available)
        if not models:
            print("ERROR: No models available for scan!")
            sys.exit(1)
        run_scan(models, results_dir, n_instances=args.n_instances)

    elif args.phase == "deep":
        # Load models from scan results or --models flag
        if args.scan_results:
            models = load_filtered_models(args.scan_results)
        elif args.models:
            models = parse_model_list(args.models, all_available)
        else:
            print("ERROR: Phase 'deep' requires --scan-results or --models")
            sys.exit(1)

        if not models:
            print("ERROR: No models available for deep evaluation!")
            sys.exit(1)

        run_deep(
            models, results_dir,
            n_instances=args.n_instances,
            n_turns=args.n_turns,
            tier=args.tier,
        )

    elif args.phase == "both":
        # Phase 1: Scan
        scan_models = parse_model_list(args.models, all_available)
        if not scan_models:
            print("ERROR: No models available for scan!")
            sys.exit(1)

        scan_dir = results_dir / "scan"
        scan_dir.mkdir(parents=True, exist_ok=True)
        run_scan(scan_models, scan_dir, n_instances=args.n_instances)

        # Phase 2: Deep — load filtered models from scan
        deep_models = load_filtered_models(str(scan_dir))
        if not deep_models:
            print("WARNING: No models passed the scan filter. Skipping Phase 2.")
        else:
            deep_dir = results_dir / "deep"
            deep_dir.mkdir(parents=True, exist_ok=True)
            run_deep(
                deep_models, deep_dir,
                n_instances=args.n_instances,
                n_turns=args.n_turns,
                tier=args.tier,
            )

    elif args.phase == "extended-sb1":
        models = parse_model_list(args.models, all_available)
        if not models:
            print("ERROR: No models specified for extended-sb1!")
            sys.exit(1)
        run_extended_sb1(
            models, results_dir,
            n_instances=args.n_instances,
            tier=args.tier,
        )

    elif args.phase == "probes":
        if not args.models:
            print("ERROR: Phase 'probes' requires --models flag")
            sys.exit(1)
        models = parse_model_list(args.models, all_available)
        if not models:
            print("ERROR: No models matched for probes!")
            sys.exit(1)
        run_probes(
            models, results_dir,
            n_instances=args.n_instances,
            n_turns=args.n_turns,
            tier=args.tier,
        )

    elapsed = time.time() - phase_start
    print(f"\n{'=' * 60}")
    print(f"Phase '{args.phase}' complete in {elapsed:.1f}s")
    print(f"Results: {results_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
