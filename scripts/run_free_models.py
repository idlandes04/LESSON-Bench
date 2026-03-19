#!/usr/bin/env python3
"""LESSON-Bench — Free Model Comprehensive Evaluation Runner.

Discovers all free models on OpenRouter and runs the full SB1 + SB2 evaluation
suite in parallel with intelligent rate limiting.

Features:
- Auto-discovers free models via OpenRouter /api/v1/models endpoint
- Runs SB1 (T1/T2, N=4/8, 3 instances, 5 items each) for learning curves
- Runs SB2 (all 10 conditions, 3 instances, 12 turns) for feedback analysis
- Parallel execution across models (configurable concurrency)
- Per-model request throttling to stay within free-tier rate limits
- OpenAI SDK max_retries=10 with exponential backoff for 429 handling
- Computes all metrics: AULC, RII, HTR, FLR, factorial decomposition
- Incremental saves per model (survives interruption)
- Smoke test to filter non-responsive models
- Resume support (skip already-completed models)

Usage:
    # Discover free models and run full suite
    python scripts/run_free_models.py

    # List available free models without running
    python scripts/run_free_models.py --dry-run

    # Run specific models only
    python scripts/run_free_models.py --models google/gemma-3-4b-it:free,meta-llama/llama-3.3-70b-instruct:free

    # Custom concurrency and rate limiting
    python scripts/run_free_models.py --max-parallel 20 --min-interval 3.0

    # SB1 scan only (quick)
    python scripts/run_free_models.py --sb1-only

    # Skip weak models for SB2 (require T2N8 > 10%)
    python scripts/run_free_models.py --skip-weak 0.10

    # Resume interrupted run
    python scripts/run_free_models.py --resume --output-dir results/free_models_20260319_230000
"""

import argparse
import concurrent.futures
import json
import os
import sys
import time
import threading
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

import openai
from lesson.models.base import LLMClient, MultiTurnSession
from lesson.models.openrouter import (
    OpenRouterClient,
    OpenRouterMultiTurnSession,
    _json_unsupported,
    _json_unsupported_lock,
)


# ---------------------------------------------------------------------------
# Constants — match existing SB1/SB2 scan protocol
# ---------------------------------------------------------------------------

SB1_TIERS = [1, 2]
SB1_N_VALUES = [4, 8]
SB1_INSTANCES = 3
SB1_SEQ_LENGTH = 5

SB2_TIER = 2
SB2_N_INITIAL = 8
SB2_INSTANCES = 3
SB2_TURNS = 12

CORE_CONDITIONS = ["correction", "practice_only", "error_only", "no_feedback"]

ALL_CONDITIONS = [
    "correction", "practice_only", "error_only", "no_feedback",
    "explanation", "misleading",
    "clean_context", "prompted_correction", "structured_correction",
    "reformatted_correction",
]

MIN_CONTEXT_LENGTH = 4096
DEFAULT_MAX_TOKENS = 2048
DEFAULT_MIN_INTERVAL = 2.0
DEFAULT_MAX_PARALLEL = 25


# ---------------------------------------------------------------------------
# Free Model Discovery
# ---------------------------------------------------------------------------

def discover_free_models(api_key: Optional[str] = None) -> List[Dict[str, Any]]:
    """Fetch all free text-generation models from OpenRouter.

    Queries GET /api/v1/models and filters for models with $0 pricing,
    sufficient context length, and text modality.

    Returns:
        List of dicts with keys: id, name, context_length, description.
    """
    url = "https://openrouter.ai/api/v1/models"
    headers = {"User-Agent": "LESSON-Bench/1.0"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode())

    free_models = []
    for model in data.get("data", []):
        pricing = model.get("pricing", {})
        prompt_price = pricing.get("prompt", "1")
        completion_price = pricing.get("completion", "1")

        # Filter: free pricing only
        try:
            if float(prompt_price) != 0.0 or float(completion_price) != 0.0:
                continue
        except (ValueError, TypeError):
            continue

        # Filter: sufficient context window
        ctx = model.get("context_length", 0)
        if ctx < MIN_CONTEXT_LENGTH:
            continue

        # Filter: must support text generation
        arch = model.get("architecture", {})
        modality = arch.get("modality", "text->text")
        if "text" not in str(modality).lower():
            continue

        model_id = model.get("id", "")
        if not model_id:
            continue

        # Skip meta-routers (not actual models)
        if model_id in ("openrouter/free",):
            continue

        free_models.append({
            "id": model_id,
            "name": model.get("name", model_id),
            "context_length": ctx,
            "description": (model.get("description", "") or "")[:120],
        })

    return sorted(free_models, key=lambda m: m["name"])


# ---------------------------------------------------------------------------
# Throttled Client — per-model rate limiting
# ---------------------------------------------------------------------------

class _ModelThrottle:
    """Enforces minimum interval between API requests for a single model.

    Thread-safe. Used to stay within free-tier rate limits (typically
    10-20 RPM on OpenRouter). The OpenAI SDK's built-in retry handles
    429 responses, but this prevents triggering them in the first place.
    """

    def __init__(self, min_interval: float = 2.0):
        self._min_interval = min_interval
        self._lock = threading.Lock()
        self._last_request = 0.0

    def wait(self):
        with self._lock:
            now = time.time()
            elapsed = now - self._last_request
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            self._last_request = time.time()


class ThrottledSession(MultiTurnSession):
    """Wraps an OpenRouterMultiTurnSession with per-model request spacing.

    The throttle.wait() call before each send/send_json ensures we don't
    exceed the model's rate limit. The __getattr__ fallback forwards
    attribute access (e.g., _messages) to the inner session, so the SB2
    pilot's error-handling code works correctly.
    """

    def __init__(self, inner: MultiTurnSession, throttle: _ModelThrottle):
        object.__setattr__(self, '_inner', inner)
        object.__setattr__(self, '_throttle', throttle)

    def send(self, text: str, role: str = "user") -> str:
        self._throttle.wait()
        return self._inner.send(text, role)

    def send_json(self, text: str, role: str = "user") -> str:
        self._throttle.wait()
        return self._inner.send_json(text, role)

    def inject(self, text: str, role: str = "assistant") -> None:
        self._inner.inject(text, role)

    def reset(self) -> None:
        self._inner.reset()

    def __getattr__(self, name):
        """Forward attribute access to inner session (e.g., _messages)."""
        return getattr(object.__getattribute__(self, '_inner'), name)


class ThrottledClient(LLMClient):
    """OpenRouter client with per-model request throttling.

    Wraps OpenRouterClient with:
    - min_interval delay between requests (avoids triggering rate limits)
    - max_retries=10 in the OpenAI SDK (handles 429 with exponential backoff)
    - Smaller max_tokens (free models often have limited context)
    """

    def __init__(
        self,
        name: str,
        model_id: str,
        api_key: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        min_interval: float = DEFAULT_MIN_INTERVAL,
    ):
        self.name = name
        self._throttle = _ModelThrottle(min_interval)
        self._inner = OpenRouterClient(
            name=name,
            model_id=model_id,
            api_key=api_key,
            max_tokens=max_tokens,
            max_retries=10,
        )

    def prompt(self, text: str) -> str:
        self._throttle.wait()
        return self._inner.prompt(text)

    def prompt_json(self, text: str) -> str:
        self._throttle.wait()
        return self._inner.prompt_json(text)

    def multi_turn(self) -> ThrottledSession:
        return ThrottledSession(self._inner.multi_turn(), self._throttle)


# ---------------------------------------------------------------------------
# Smoke Test
# ---------------------------------------------------------------------------

def run_smoke_test(
    models: List[Dict[str, Any]],
    api_key: Optional[str] = None,
    max_parallel: int = 20,
    min_interval: float = 0.5,
) -> List[Dict[str, Any]]:
    """Quick connectivity test for all models.

    Sends a single STS-like prompt to each model and checks for a non-empty
    response. Returns list of models that passed.
    """
    test_prompt = (
        "You are learning a symbolic transformation system.\n"
        "Here is 1 example:\n"
        "Example 1: ABC \u2192 CBA\n\n"
        "What is the output for: XYZ\n"
        'Respond with ONLY: {"output": "YOUR_ANSWER"}'
    )

    passed: List[Dict[str, Any]] = []
    failed: List[Tuple[Dict, str]] = []
    lock = threading.Lock()

    print(f"\n{'=' * 70}")
    print(f"SMOKE TEST \u2014 Testing {len(models)} free models")
    print(f"{'=' * 70}")

    def _test(model_info: Dict) -> None:
        model_id = model_info["id"]
        try:
            client = ThrottledClient(
                name=model_id,
                model_id=model_id,
                api_key=api_key,
                max_tokens=512,
                min_interval=min_interval,
            )
            t0 = time.time()
            response = client.prompt(test_prompt)
            elapsed = time.time() - t0
            if response.strip():
                with lock:
                    passed.append(model_info)
                    print(f"  [PASS] {model_id} ({elapsed:.1f}s)")
            else:
                with lock:
                    failed.append((model_info, "EMPTY RESPONSE"))
                    print(f"  [FAIL] {model_id} \u2014 empty response")
        except Exception as e:
            with lock:
                reason = str(e)[:100]
                failed.append((model_info, reason))
                print(f"  [FAIL] {model_id} \u2014 {reason}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as pool:
        list(pool.map(_test, models))

    print(f"\n  {len(passed)}/{len(models)} models passed smoke test")
    if failed:
        print(f"  {len(failed)} failed:")
        for info, reason in failed[:15]:
            print(f"    {info['id']} \u2014 {reason}")
        if len(failed) > 15:
            print(f"    ... and {len(failed) - 15} more")

    return passed


# ---------------------------------------------------------------------------
# Atomic Save Helper
# ---------------------------------------------------------------------------

def _safe_save(path: Path, data: Any) -> None:
    """Save JSON atomically (write to temp file, then rename)."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=str)
    tmp.rename(path)


def _safe_name(model_id: str) -> str:
    """Convert model ID to filesystem-safe name."""
    return model_id.replace("/", "__").replace(":", "_").replace(" ", "_")


# ---------------------------------------------------------------------------
# Per-Model Comprehensive Evaluation
# ---------------------------------------------------------------------------

def evaluate_model_comprehensive(
    model_info: Dict[str, Any],
    conditions: List[str],
    results_dir: Path,
    api_key: Optional[str] = None,
    min_interval: float = DEFAULT_MIN_INTERVAL,
    skip_sb2_threshold: float = 0.0,
    run_sb1: bool = True,
    run_sb2: bool = True,
    print_lock: Optional[threading.Lock] = None,
) -> Dict[str, Any]:
    """Run complete SB1 + SB2 + metrics for a single model.

    Evaluation flow:
    1. SB1: Learning curves across tiers (T1, T2) and N-values (4, 8).
       Computes AULC, RII, HTR from Type R/E/L accuracy.
    2. SB2: All feedback conditions with multi-turn sessions.
       Computes FLR, condition trajectories, 2x2 factorial effects.
    3. Combined metrics via compute_model_profile().

    Results are saved incrementally after each condition completes,
    so interrupted runs don't lose progress.

    Args:
        model_info: Dict with id, name, context_length.
        conditions: SB2 conditions to evaluate.
        results_dir: Directory for result JSON files.
        api_key: OpenRouter API key.
        min_interval: Min seconds between requests for this model.
        skip_sb2_threshold: Skip SB2 if SB1 T2N8 accuracy < threshold.
        run_sb1: Whether to run SB1.
        run_sb2: Whether to run SB2.
        print_lock: Lock for thread-safe console output.

    Returns:
        Dict with sb1, sb2_{condition}, metrics, timing info.
    """
    from lesson.eval.pilot import run_sb1_pilot
    from lesson.eval.sb2_pilot import run_sb2_pilot
    from lesson.eval.stats import compute_model_profile

    if print_lock is None:
        print_lock = threading.Lock()

    model_id = model_info["id"]
    safe = _safe_name(model_id)

    result: Dict[str, Any] = {
        "model_id": model_id,
        "model_name": model_info["name"],
        "context_length": model_info["context_length"],
        "started_at": datetime.now().isoformat(),
        "conditions": conditions,
    }

    total_start = time.time()

    # Determine max_tokens based on context window
    ctx = model_info.get("context_length", 8192)
    max_tokens = min(DEFAULT_MAX_TOKENS, ctx // 4)
    max_tokens = max(max_tokens, 512)

    try:
        client = ThrottledClient(
            name=safe,
            model_id=model_id,
            api_key=api_key,
            max_tokens=max_tokens,
            min_interval=min_interval,
        )
    except Exception as e:
        with print_lock:
            print(f"  [ERROR] {model_id}: client creation failed: {e}")
        result["error"] = str(e)
        return result

    # ==================================================================
    # SB1: Learning curves
    # ==================================================================
    t2n8_accuracy = 0.0

    if run_sb1:
        with print_lock:
            print(f"\n  [{model_id}] SB1 starting "
                  f"(T{SB1_TIERS}, N={SB1_N_VALUES}, {SB1_INSTANCES} inst)...")

        sb1_start = time.time()
        try:
            sb1_data = run_sb1_pilot(
                client=client,
                tiers=SB1_TIERS,
                n_values=SB1_N_VALUES,
                n_instances=SB1_INSTANCES,
                seq_length=SB1_SEQ_LENGTH,
            )
            sb1_elapsed = time.time() - sb1_start
            result["sb1"] = sb1_data
            result["sb1_elapsed_s"] = sb1_elapsed

            # Extract key accuracies from summary
            summary = sb1_data.get("summary", {})

            def _get_type_r(tier: int, n: int) -> float:
                rd = summary.get(tier, {}).get(n, {}).get("R", {})
                total = rd.get("n_total", 0)
                return rd.get("n_correct", 0) / total if total else 0.0

            t1n4 = _get_type_r(1, 4)
            t1n8 = _get_type_r(1, 8)
            t2n4 = _get_type_r(2, 4)
            t2n8_accuracy = _get_type_r(2, 8)

            with print_lock:
                print(f"  [{model_id}] SB1 done ({sb1_elapsed:.0f}s): "
                      f"T1N4={t1n4:.0%} T1N8={t1n8:.0%} "
                      f"T2N4={t2n4:.0%} T2N8={t2n8_accuracy:.0%}")

            _safe_save(results_dir / f"{safe}_sb1.json", result)

        except Exception as e:
            sb1_elapsed = time.time() - sb1_start
            with print_lock:
                print(f"  [{model_id}] SB1 FAILED ({sb1_elapsed:.0f}s): {e}")
            result["sb1_error"] = str(e)
            result["sb1_elapsed_s"] = sb1_elapsed
            _safe_save(results_dir / f"{safe}_sb1.json", result)

    # ==================================================================
    # SB2: Feedback conditions
    # ==================================================================
    if run_sb2:
        if skip_sb2_threshold > 0 and t2n8_accuracy < skip_sb2_threshold:
            with print_lock:
                print(f"  [{model_id}] Skipping SB2 "
                      f"(T2N8={t2n8_accuracy:.0%} < {skip_sb2_threshold:.0%} threshold)")
            result["sb2_skipped"] = (
                f"T2N8 accuracy {t2n8_accuracy:.0%} below "
                f"threshold {skip_sb2_threshold:.0%}"
            )
        else:
            for condition in conditions:
                with print_lock:
                    print(f"  [{model_id}] SB2 {condition}...")

                cond_start = time.time()
                try:
                    sb2_data = run_sb2_pilot(
                        client=client,
                        tier=SB2_TIER,
                        n_initial_examples=SB2_N_INITIAL,
                        n_instances=SB2_INSTANCES,
                        n_turns=SB2_TURNS,
                        conditions=[condition],
                        max_parallel=1,  # Sequential within model
                    )
                    cond_elapsed = time.time() - cond_start
                    result[f"sb2_{condition}"] = sb2_data
                    result[f"sb2_{condition}_elapsed_s"] = cond_elapsed

                    # Quick accuracy summary
                    cond_summary = sb2_data.get("summary", {}).get(condition, {})
                    accs = []
                    for t_data in cond_summary.values():
                        if t_data.get("n_total", 0) > 0:
                            accs.append(t_data["n_correct"] / t_data["n_total"])
                    avg_acc = sum(accs) / len(accs) if accs else 0.0

                    with print_lock:
                        print(f"  [{model_id}] SB2 {condition}: "
                              f"avg={avg_acc:.0%} ({cond_elapsed:.0f}s)")

                except Exception as e:
                    cond_elapsed = time.time() - cond_start
                    with print_lock:
                        print(f"  [{model_id}] SB2 {condition} "
                              f"FAILED ({cond_elapsed:.0f}s): {e}")
                    result[f"sb2_{condition}_error"] = str(e)
                    result[f"sb2_{condition}_elapsed_s"] = cond_elapsed

                # Save after each condition (survives interruption)
                _safe_save(results_dir / f"{safe}_progress.json", result)

    # ==================================================================
    # Metrics: compute full cognitive profile
    # ==================================================================
    # Filter out empty responses (thinking models hitting token limit)
    # to avoid inflating error rates with noise.
    def _has_response(r: Dict) -> bool:
        raw = r.get("raw_response", "")
        answer = r.get("model_answer", "")
        return bool(raw and raw.strip()) or bool(answer and answer.strip())

    try:
        sb1_results_raw = result.get("sb1", {}).get("results", [])
        sb1_results = [r for r in sb1_results_raw if _has_response(r)]
        n_sb1_filtered = len(sb1_results_raw) - len(sb1_results)

        sb2_by_condition: Dict[str, List[Dict]] = {}
        n_sb2_filtered = 0
        for cond in conditions:
            key = f"sb2_{cond}"
            if key in result and isinstance(result[key], dict):
                cond_results_raw = result[key].get("results", [])
                cond_results = [r for r in cond_results_raw if _has_response(r)]
                n_sb2_filtered += len(cond_results_raw) - len(cond_results)
                if cond_results:
                    sb2_by_condition[cond] = cond_results

        if n_sb1_filtered or n_sb2_filtered:
            with print_lock:
                print(f"  [{model_id}] Filtered {n_sb1_filtered} SB1 + "
                      f"{n_sb2_filtered} SB2 empty responses")

        if sb1_results or sb2_by_condition:
            profile = compute_model_profile(
                sb1_results=sb1_results,
                sb2_results_by_condition=sb2_by_condition,
                n_turns=SB2_TURNS,
            )
            result["metrics"] = profile
            result["metrics_filtered"] = {
                "sb1_empty_discarded": n_sb1_filtered,
                "sb2_empty_discarded": n_sb2_filtered,
            }
    except Exception as e:
        with print_lock:
            print(f"  [{model_id}] Metrics computation failed: {e}")
        result["metrics_error"] = str(e)

    result["total_elapsed_s"] = time.time() - total_start
    result["finished_at"] = datetime.now().isoformat()

    # Final save
    _safe_save(results_dir / f"{safe}_complete.json", result)

    with print_lock:
        elapsed = result["total_elapsed_s"]
        print(f"\n  \u2713 [{model_id}] COMPLETE ({elapsed:.0f}s / {elapsed/60:.1f}m)")

    return result


# ---------------------------------------------------------------------------
# Cross-Model Summary
# ---------------------------------------------------------------------------

def print_cross_model_summary(
    all_results: Dict[str, Dict],
    conditions: List[str],
) -> None:
    """Print comprehensive cross-model analysis tables."""

    print(f"\n{'=' * 90}")
    print("FREE MODEL COMPREHENSIVE RESULTS")
    print(f"{'=' * 90}")

    # ------------------------------------------------------------------
    # SB1 Summary — Learning Curves (Type R accuracy)
    # ------------------------------------------------------------------
    print(f"\n{'=' * 90}")
    print("SB1 \u2014 Learning Curve Results (Type R accuracy)")
    print(f"{'=' * 90}")

    header = f"{'Model':<50} {'T1N4':>5} {'T1N8':>5} {'T2N4':>5} {'T2N8':>5} {'AULC':>6} {'RII':>5} {'HTR':>5}"
    print(header)
    print("\u2500" * len(header))

    sb1_rows = []
    for model_id, data in all_results.items():
        if "error" in data and "sb1" not in data:
            continue

        summary = data.get("sb1", {}).get("summary", {})
        metrics = data.get("metrics", {})

        def _get_r(tier: int, n: int) -> float:
            rd = summary.get(tier, {}).get(n, {}).get("R", {})
            total = rd.get("n_total", 0)
            return rd.get("n_correct", 0) / total if total else 0.0

        t1n4 = _get_r(1, 4)
        t1n8 = _get_r(1, 8)
        t2n4 = _get_r(2, 4)
        t2n8 = _get_r(2, 8)
        aulc = metrics.get("aulc", 0.0)
        rii = metrics.get("rii")
        htr = metrics.get("htr")

        sb1_rows.append((model_id, t1n4, t1n8, t2n4, t2n8, aulc, rii, htr))

    # Sort by T2N8 descending
    sb1_rows.sort(key=lambda r: r[4], reverse=True)

    for model_id, t1n4, t1n8, t2n4, t2n8, aulc, rii, htr in sb1_rows:
        rii_str = f"{rii:.2f}" if rii is not None else "\u2014"
        htr_str = f"{htr:.2f}" if htr is not None else "\u2014"
        print(f"{model_id[:49]:<50} {t1n4:>4.0%} {t1n8:>4.0%} "
              f"{t2n4:>4.0%} {t2n8:>4.0%} {aulc:>5.2f} {rii_str:>5} {htr_str:>5}")

    # SB1 filter classification
    passing = [r for r in sb1_rows if 0.15 <= r[4] <= 0.70]
    too_high = [r for r in sb1_rows if r[4] > 0.70]
    too_low = [r for r in sb1_rows if r[4] < 0.15]
    print(f"\n  SB2 filter (T2N8 15-70%): {len(passing)} PASS, "
          f"{len(too_high)} above ceiling, {len(too_low)} below floor")

    # ------------------------------------------------------------------
    # SB2 Summary — Condition averages
    # ------------------------------------------------------------------
    _DASH = "\u2014"

    has_sb2 = any(
        f"sb2_{c}" in data and isinstance(data.get(f"sb2_{c}"), dict)
        for data in all_results.values()
        for c in conditions
    )

    if has_sb2:
        print(f"\n{'=' * 90}")
        print("SB2 \u2014 Average Accuracy by Condition")
        print(f"{'=' * 90}")

        # Determine which conditions have data
        active_conds = [
            c for c in conditions
            if any(
                f"sb2_{c}" in d and isinstance(d.get(f"sb2_{c}"), dict)
                for d in all_results.values()
            )
        ]

        header = f"{'Model':<35}"
        for c in active_conds:
            header += f" {c[:11]:>11}"
        header += f" {'FLR':>8}"
        print(header)
        print("\u2500" * len(header))

        for model_id, data in sorted(all_results.items()):
            if data.get("sb2_skipped") or ("error" in data and not any(
                f"sb2_{c}" in data for c in active_conds
            )):
                continue

            row = f"{model_id[:34]:<35}"
            for cond in active_conds:
                key = f"sb2_{cond}"
                if key in data and isinstance(data[key], dict):
                    cond_summary = data[key].get("summary", {}).get(cond, {})
                    accs = []
                    for t_data in cond_summary.values():
                        if t_data.get("n_total", 0) > 0:
                            accs.append(t_data["n_correct"] / t_data["n_total"])
                    avg = sum(accs) / len(accs) if accs else 0.0
                    row += f" {avg:>10.0%}"
                else:
                    row += f" {_DASH:>11}"

            # FLR
            metrics = data.get("metrics", {})
            flr_tuple = metrics.get("flr")
            if flr_tuple and len(flr_tuple) >= 1 and flr_tuple[0] is not None:
                row += f" {flr_tuple[0]:>+7.3f}"
            else:
                row += f" {_DASH:>8}"
            print(row)

    # ------------------------------------------------------------------
    # Factorial Decomposition
    # ------------------------------------------------------------------
    has_factorial = any(
        data.get("metrics", {}).get("factorial") is not None
        for data in all_results.values()
    )

    if has_factorial:
        print(f"\n{'=' * 90}")
        print("2\u00d72 Factorial Decomposition")
        print(f"{'=' * 90}")

        header = (f"{'Model':<35} {'Answer Eff':>10} {'Eval Eff':>10} "
                  f"{'Interaction':>12} {'FLR':>10}")
        print(header)
        print("\u2500" * len(header))

        for model_id, data in sorted(all_results.items()):
            metrics = data.get("metrics", {})
            factorial = metrics.get("factorial")
            if factorial is None:
                continue

            ae = factorial.get("answer_effect", (None,))
            ee = factorial.get("evaluation_effect", (None,))
            inter = factorial.get("interaction", (None,))
            flr = metrics.get("flr", (None,))

            def _fmt(val_tuple):
                if val_tuple and len(val_tuple) >= 3 and val_tuple[0] is not None:
                    return f"{val_tuple[0]:>+.3f}"
                return "\u2014"

            print(f"{model_id[:34]:<35} {_fmt(ae):>10} {_fmt(ee):>10} "
                  f"{_fmt(inter):>12} {_fmt(flr):>10}")

    # ------------------------------------------------------------------
    # Full Metrics Table
    # ------------------------------------------------------------------
    print(f"\n{'=' * 90}")
    print("Comprehensive Metrics Summary")
    print(f"{'=' * 90}")

    header = (f"{'Model':<40} {'AULC':>5} {'RII':>5} {'HTR':>5} "
              f"{'FLR':>8} {'AnsEff':>7} {'EvalEff':>7}")
    print(header)
    print("\u2500" * len(header))

    for model_id, data in sorted(all_results.items()):
        metrics = data.get("metrics", {})
        if not metrics:
            continue

        aulc = metrics.get("aulc", 0.0)
        rii = metrics.get("rii")
        htr = metrics.get("htr")
        flr_t = metrics.get("flr", (None,))
        flr = flr_t[0] if flr_t else None

        factorial = metrics.get("factorial", {}) or {}
        ae_t = factorial.get("answer_effect", (None,))
        ae = ae_t[0] if ae_t and len(ae_t) >= 1 else None
        ee_t = factorial.get("evaluation_effect", (None,))
        ee = ee_t[0] if ee_t and len(ee_t) >= 1 else None

        row = f"{model_id[:39]:<40}"
        row += f" {aulc:>4.2f}" if aulc else f" {_DASH:>5}"
        row += f" {rii:>4.2f}" if rii is not None else f" {_DASH:>5}"
        row += f" {htr:>4.2f}" if htr is not None else f" {_DASH:>5}"
        row += f" {flr:>+7.3f}" if flr is not None else f" {_DASH:>8}"
        row += f" {ae:>+6.3f}" if ae is not None else f" {_DASH:>7}"
        row += f" {ee:>+6.3f}" if ee is not None else f" {_DASH:>7}"
        print(row)

    # ------------------------------------------------------------------
    # Error summary
    # ------------------------------------------------------------------
    errors = [
        (mid, data.get("error", data.get("sb1_error", "")))
        for mid, data in all_results.items()
        if "error" in data or "sb1_error" in data
    ]
    if errors:
        print(f"\n{'=' * 90}")
        print(f"Models with errors ({len(errors)})")
        print(f"{'=' * 90}")
        for mid, err in errors:
            print(f"  {mid}: {str(err)[:80]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LESSON-Bench \u2014 Free Model Comprehensive Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List discovered free models without running evaluation.",
    )
    parser.add_argument(
        "--models", type=str, default=None,
        help="Comma-separated OpenRouter model IDs to evaluate (skip discovery).",
    )
    parser.add_argument(
        "--conditions", type=str, default=",".join(ALL_CONDITIONS),
        help=f"Comma-separated SB2 conditions (default: all {len(ALL_CONDITIONS)}).",
    )
    parser.add_argument(
        "--max-parallel", type=int, default=DEFAULT_MAX_PARALLEL,
        help=f"Max models to evaluate in parallel (default: {DEFAULT_MAX_PARALLEL}).",
    )
    parser.add_argument(
        "--min-interval", type=float, default=DEFAULT_MIN_INTERVAL,
        help=f"Min seconds between requests per model (default: {DEFAULT_MIN_INTERVAL}).",
    )
    parser.add_argument(
        "--sb1-only", action="store_true",
        help="Run SB1 only, skip SB2.",
    )
    parser.add_argument(
        "--sb2-only", action="store_true",
        help="Run SB2 only, skip SB1.",
    )
    parser.add_argument(
        "--skip-weak", type=float, default=0.0, metavar="THRESHOLD",
        help="Skip SB2 for models with SB1 T2N8 < threshold (e.g., 0.10).",
    )
    parser.add_argument(
        "--skip-smoke-test", action="store_true",
        help="Skip smoke test (evaluate all discovered models directly).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip models that already have _complete.json in output dir.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Override results directory (useful with --resume).",
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set.")
        print("  Add it to .env or: export OPENROUTER_API_KEY=sk-or-...")
        sys.exit(1)

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    run_sb1 = not args.sb2_only
    run_sb2 = not args.sb1_only

    # Validate conditions
    for c in conditions:
        if c not in ALL_CONDITIONS:
            print(f"ERROR: Unknown condition {c!r}. Valid: {ALL_CONDITIONS}")
            sys.exit(1)

    # ==================================================================
    # Discover or parse models
    # ==================================================================
    if args.models:
        model_ids = [m.strip() for m in args.models.split(",") if m.strip()]
        models = [
            {"id": mid, "name": mid, "context_length": 32768, "description": ""}
            for mid in model_ids
        ]
        print(f"\nUsing {len(models)} specified model(s)")
    else:
        print("\nDiscovering free models on OpenRouter...")
        try:
            models = discover_free_models(api_key)
        except Exception as e:
            print(f"ERROR: Failed to discover models: {e}")
            sys.exit(1)
        print(f"Found {len(models)} free models")

    if not models:
        print("ERROR: No models to evaluate.")
        sys.exit(1)

    # ==================================================================
    # Dry run: just list models and exit
    # ==================================================================
    if args.dry_run:
        print(f"\n{'Model ID':<55} {'Context':>8}  {'Name'}")
        print("\u2500" * 95)
        for m in models:
            ctx_str = f"{m['context_length']:>7,}"
            print(f"{m['id']:<55} {ctx_str}  {m['name'][:30]}")
        print(f"\nTotal: {len(models)} free models")
        sys.exit(0)

    # ==================================================================
    # Setup results directory
    # ==================================================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        results_dir = Path(args.output_dir)
    else:
        results_dir = Path("results") / f"free_models_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Resume: skip already-completed models
    if args.resume:
        completed = set()
        for f in results_dir.glob("*_complete.json"):
            completed.add(f.stem.replace("_complete", ""))
        before = len(models)
        models = [
            m for m in models
            if _safe_name(m["id"]) not in completed
        ]
        skipped = before - len(models)
        if skipped:
            print(f"Resume: {skipped} models already complete, {len(models)} remaining")

    if not models:
        print("All models already complete. Nothing to do.")
        # Load existing results for summary
        all_results = {}
        for f in results_dir.glob("*_complete.json"):
            with open(f) as fh:
                data = json.load(fh)
                all_results[data.get("model_id", f.stem)] = data
        if all_results:
            print_cross_model_summary(all_results, conditions)
        sys.exit(0)

    # Save run config
    config = {
        "version": "free_models_v1",
        "timestamp": timestamp,
        "n_models": len(models),
        "models": [m["id"] for m in models],
        "conditions": conditions,
        "sb1": {
            "tiers": SB1_TIERS,
            "n_values": SB1_N_VALUES,
            "instances": SB1_INSTANCES,
            "seq_length": SB1_SEQ_LENGTH,
        },
        "sb2": {
            "tier": SB2_TIER,
            "n_initial": SB2_N_INITIAL,
            "instances": SB2_INSTANCES,
            "turns": SB2_TURNS,
        },
        "max_parallel": args.max_parallel,
        "min_interval": args.min_interval,
        "run_sb1": run_sb1,
        "run_sb2": run_sb2,
        "skip_weak": args.skip_weak,
    }
    _safe_save(results_dir / "run_config.json", config)

    # ==================================================================
    # Print run plan
    # ==================================================================
    sb1_calls = (
        len(SB1_TIERS) * len(SB1_N_VALUES) * SB1_INSTANCES * 5
        if run_sb1 else 0
    )
    sb2_calls = (
        len(conditions) * SB2_INSTANCES * SB2_TURNS
        if run_sb2 else 0
    )
    calls_per_model = sb1_calls + sb2_calls
    total_calls = calls_per_model * len(models)
    est_minutes = (calls_per_model * args.min_interval) / 60

    print(f"\n{'=' * 70}")
    print(f"LESSON-Bench \u2014 Free Model Comprehensive Evaluation")
    print(f"{'=' * 70}")
    print(f"  Models:       {len(models)}")
    print(f"  SB1:          {'Yes' if run_sb1 else 'No'}"
          + (f" \u2014 T{SB1_TIERS}, N={SB1_N_VALUES}, "
             f"{SB1_INSTANCES} inst, {sb1_calls} calls" if run_sb1 else ""))
    print(f"  SB2:          {'Yes' if run_sb2 else 'No'}"
          + (f" \u2014 {len(conditions)} conditions, "
             f"{SB2_INSTANCES} inst, {SB2_TURNS}t, "
             f"{sb2_calls} calls" if run_sb2 else ""))
    print(f"  Parallelism:  {args.max_parallel} models concurrent")
    print(f"  Rate limit:   {args.min_interval}s between requests per model")
    print(f"  Results:      {results_dir}")
    print(f"  Est. calls:   ~{calls_per_model} per model, "
          f"~{total_calls:,} total")
    print(f"  Est. time:    ~{est_minutes:.0f}m per model "
          f"(parallel \u2192 same wall clock)")

    # ==================================================================
    # Smoke test
    # ==================================================================
    if not args.skip_smoke_test:
        models = run_smoke_test(
            models, api_key, args.max_parallel, min_interval=1.0,
        )
        if not models:
            print("\nERROR: No models passed smoke test!")
            sys.exit(1)

        # Update config with passing models
        config["models_after_smoke_test"] = [m["id"] for m in models]
        config["n_models_after_smoke_test"] = len(models)
        _safe_save(results_dir / "run_config.json", config)

    # ==================================================================
    # Run evaluation
    # ==================================================================
    all_results: Dict[str, Dict] = {}
    print_lock = threading.Lock()
    run_start = time.time()

    print(f"\n{'=' * 70}")
    print(f"STARTING EVALUATION \u2014 {len(models)} models, "
          f"{args.max_parallel} parallel")
    print(f"{'=' * 70}")

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.max_parallel
    ) as pool:
        futures = {}
        for model_info in models:
            f = pool.submit(
                evaluate_model_comprehensive,
                model_info=model_info,
                conditions=conditions,
                results_dir=results_dir,
                api_key=api_key,
                min_interval=args.min_interval,
                skip_sb2_threshold=args.skip_weak,
                run_sb1=run_sb1,
                run_sb2=run_sb2,
                print_lock=print_lock,
            )
            futures[f] = model_info

        completed_count = 0
        for f in concurrent.futures.as_completed(futures):
            model_info = futures[f]
            completed_count += 1
            try:
                result = f.result()
                all_results[model_info["id"]] = result
                with print_lock:
                    print(f"\n  [{completed_count}/{len(models)}] "
                          f"{model_info['id']} complete")
            except Exception as e:
                with print_lock:
                    print(f"\n  \u2717 [{model_info['id']}] FATAL: {e}")
                all_results[model_info["id"]] = {
                    "model_id": model_info["id"],
                    "error": str(e),
                }

    total_elapsed = time.time() - run_start

    # ==================================================================
    # Load any previously completed results (for resume mode)
    # ==================================================================
    if args.resume:
        for f in results_dir.glob("*_complete.json"):
            with open(f) as fh:
                data = json.load(fh)
                mid = data.get("model_id", f.stem)
                if mid not in all_results:
                    all_results[mid] = data

    # ==================================================================
    # Save combined results
    # ==================================================================
    _safe_save(results_dir / "combined_results.json", all_results)

    # ==================================================================
    # Print summary
    # ==================================================================
    print_cross_model_summary(all_results, conditions)

    n_success = sum(
        1 for d in all_results.values()
        if "error" not in d
    )
    n_errors = len(all_results) - n_success

    print(f"\n{'=' * 70}")
    print(f"EVALUATION COMPLETE")
    print(f"  Total time:   {total_elapsed:.0f}s ({total_elapsed/60:.1f}m)")
    print(f"  Models:       {n_success} succeeded, {n_errors} failed")
    print(f"  Results dir:  {results_dir}")
    print(f"  Combined:     {results_dir / 'combined_results.json'}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
