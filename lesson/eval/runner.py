"""Shared infrastructure for running LESSON-Bench evaluations.

Centralizes the boilerplate that was duplicated across scripts:
- save_incremental(): thread-safe JSON save
- run_parallel_by_provider(): provider-grouped parallelism harness
- parse_model_list(): model name → (provider, name) resolution
- smoke_test(): connectivity verification
- print_cross_model_summary(): cross-model results table with FLR
- CircuitBreaker: stops hammering a model after N consecutive failures
- retry_with_backoff(): smart retry for transient errors (rate limits, timeouts)
- get_completed_cells(): resume from SQLite — skip already-complete cells
"""

from __future__ import annotations

import concurrent.futures
import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Circuit breaker — stops hammering a broken model
# ---------------------------------------------------------------------------

class CircuitBreaker:
    """Track consecutive failures per model. Trip after max_failures.

    Thread-safe. Once tripped, is_tripped() returns True and subsequent
    calls should skip the model instead of wasting API calls.

    Usage::

        breaker = CircuitBreaker(max_failures=3)

        # After a successful API call:
        breaker.record_success("glm-5")

        # After a failure:
        breaker.record_failure("glm-5")

        # Before making a call:
        if breaker.is_tripped("glm-5"):
            print("Skipping glm-5 — circuit breaker tripped")
    """

    def __init__(self, max_failures: int = 3) -> None:
        self.max_failures = max_failures
        self._failures: Dict[str, int] = {}
        self._tripped: Set[str] = set()
        self._lock = threading.Lock()

    def record_failure(self, model: str, error: str = "") -> bool:
        """Record a failure. Returns True if circuit just tripped."""
        with self._lock:
            self._failures[model] = self._failures.get(model, 0) + 1
            if self._failures[model] >= self.max_failures:
                if model not in self._tripped:
                    self._tripped.add(model)
                    return True
        return False

    def record_success(self, model: str) -> None:
        """Record a success. Resets the failure counter."""
        with self._lock:
            self._failures[model] = 0

    def is_tripped(self, model: str) -> bool:
        """Check if a model's circuit breaker is tripped."""
        with self._lock:
            return model in self._tripped

    def get_status(self) -> Dict[str, Any]:
        """Get current status of all tracked models."""
        with self._lock:
            return {
                "failures": dict(self._failures),
                "tripped": list(self._tripped),
            }


# ---------------------------------------------------------------------------
# Smart retry with exponential backoff
# ---------------------------------------------------------------------------

def retry_with_backoff(
    fn: Callable[..., Any],
    max_retries: int = 3,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
    retryable_errors: Optional[tuple] = None,
) -> Any:
    """Call fn() with exponential backoff on retryable errors.

    Args:
        fn: Zero-argument callable to retry.
        max_retries: Maximum number of retry attempts (not counting the first try).
        base_delay: Initial delay in seconds.
        max_delay: Maximum delay in seconds.
        retryable_errors: Tuple of exception types to retry on.
            Default: catches common API errors (rate limit, timeout, server error).

    Returns:
        The return value of fn().

    Raises:
        The last exception if all retries are exhausted.
    """
    if retryable_errors is None:
        # Import here to avoid hard dependency
        try:
            import openai
            retryable_errors = (
                openai.RateLimitError,
                openai.APITimeoutError,
                openai.InternalServerError,
                openai.APIConnectionError,
                ConnectionError,
                TimeoutError,
            )
        except ImportError:
            retryable_errors = (ConnectionError, TimeoutError)

    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except retryable_errors as e:
            last_exc = e
            if attempt < max_retries:
                delay = min(base_delay * (2 ** attempt), max_delay)
                error_type = type(e).__name__
                print(f"    Retry {attempt + 1}/{max_retries} after {delay:.0f}s ({error_type}: {e})")
                time.sleep(delay)
            else:
                raise
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Resume helpers — check what's already complete
# ---------------------------------------------------------------------------

def get_completed_cells(
    db_path: Optional[Path] = None,
) -> Set[Tuple[str, str]]:
    """Get set of (model, condition) pairs that are already complete in the DB.

    Returns empty set if DB doesn't exist or can't be read.

    Returns:
        Set of (model_name, condition) tuples with status='complete'.
    """
    try:
        from lesson.results.store import ResultsStore, DEFAULT_DB
        path = db_path or DEFAULT_DB
        if not Path(path).exists():
            return set()
        store = ResultsStore(Path(path))
        rows = store.query(
            "SELECT model, condition FROM cells WHERE status = 'complete'"
        )
        result = {(r["model"], r["condition"]) for r in rows}
        store.close()
        return result
    except Exception:
        return set()


def filter_incomplete(
    items: List[Tuple[str, str]],
    conditions: List[str],
    db_path: Optional[Path] = None,
) -> List[Tuple[str, str]]:
    """Filter a model list to only those with incomplete work.

    For 2-tuples (provider, model), keeps models that have any incomplete condition.
    Prints a summary of what's being skipped.

    Args:
        items: List of (provider, model_name) tuples.
        conditions: List of condition names to check.
        db_path: Path to SQLite DB. Uses default if None.

    Returns:
        Filtered list of (provider, model_name) tuples.
    """
    completed = get_completed_cells(db_path)
    if not completed:
        return items

    filtered = []
    skipped = []
    for provider, model_name in items:
        all_done = all((model_name, c) in completed for c in conditions)
        if all_done:
            skipped.append(model_name)
        else:
            filtered.append((provider, model_name))

    if skipped:
        print(f"\n  Skipping {len(skipped)} already-complete model(s):")
        for name in skipped:
            print(f"    {name} — all conditions complete")

    return filtered


# ---------------------------------------------------------------------------
# Incremental save
# ---------------------------------------------------------------------------

def save_incremental(
    results_dir: Path,
    model_name: str,
    label: str,
    data: Any,
    lock: Optional[threading.Lock] = None,
) -> Path:
    """Atomically save a JSON result file. Thread-safe when lock provided.

    Returns the path of the saved file.
    """
    safe_name = model_name.replace("/", "_").replace(":", "_")
    path = results_dir / f"{safe_name}_{label}.json"
    content = json.dumps(data, indent=2, default=str)
    if lock:
        with lock:
            path.write_text(content)
    else:
        path.write_text(content)
    return path


# ---------------------------------------------------------------------------
# Model list parsing
# ---------------------------------------------------------------------------

def parse_model_list(
    names_csv: Optional[str],
    default_models: Optional[List[Tuple[str, str]]] = None,
) -> List[Tuple[str, str]]:
    """Parse a comma-separated model list into (provider, model_name) pairs.

    Uses registry.get_provider_for() to auto-detect provider. Falls back to
    heuristics for unknown names (gemini* → gemini, else → openrouter).

    Args:
        names_csv: Comma-separated model names, or None to use defaults.
        default_models: Default model list if names_csv is None.

    Returns:
        List of (provider, model_name) tuples.
    """
    if names_csv is None:
        return default_models or []

    from lesson.models.registry import get_provider_for

    requested = [m.strip() for m in names_csv.split(",") if m.strip()]
    result: List[Tuple[str, str]] = []

    for name in requested:
        try:
            provider = get_provider_for(name)
            result.append((provider, name))
        except KeyError:
            # Fallback heuristics for unknown models
            if name.startswith("gemini"):
                result.append(("gemini", name))
            elif name.startswith("lm-"):
                result.append(("lmstudio", name))
            else:
                print(f"  WARNING: Unknown model {name!r}, assuming openrouter")
                result.append(("openrouter", name))

    return result


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

_SMOKE_PROMPT = (
    "You are learning a symbolic transformation system.\n"
    "Here is 1 example:\n"
    "Example 1: ABC → CBA\n\n"
    "What is the output for: XYZ\n"
    'Respond with ONLY: {"output": "YOUR_ANSWER"}'
)


def smoke_test(
    models: List[Tuple[str, str]],
) -> List[Tuple[str, str]]:
    """Verify all models respond before committing to a full run.

    Sends a single STS-like prompt to each model. Returns passing models.
    """
    from lesson.models.registry import get_client

    passed: List[Tuple[str, str]] = []
    failed: List[Tuple[str, str, str]] = []

    print("\n" + "=" * 60)
    print("SMOKE TEST — Verifying model connectivity")
    print("=" * 60)

    def _test_one(provider: str, model_name: str) -> Tuple[str, str, bool, str]:
        try:
            client = get_client(provider, model_name)
            t0 = time.time()
            response = client.prompt(_SMOKE_PROMPT)
            elapsed = time.time() - t0
            if response.strip():
                return provider, model_name, True, f"OK ({elapsed:.1f}s): {response[:80]!r}"
            else:
                return provider, model_name, False, "EMPTY RESPONSE"
        except Exception as e:
            return provider, model_name, False, f"ERROR: {e}"

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(len(models), 1)) as pool:
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
# Provider-grouped parallel execution
# ---------------------------------------------------------------------------

# Type for the eval callback:
#   fn(provider, model_name, print_lock, **kwargs) -> result_dict
EvalFn = Callable[..., Dict[str, Any]]


def run_parallel_by_provider(
    items: List[Tuple[str, str]],
    eval_fn: EvalFn,
    or_parallel: int = 8,
    lm_parallel: int = 4,
    print_lock: Optional[threading.Lock] = None,
    circuit_breaker: Optional[CircuitBreaker] = None,
    **eval_kwargs: Any,
) -> Dict[str, Dict[str, Any]]:
    """Run eval_fn across all models using provider-appropriate parallelism.

    Parallelism strategy:
        openrouter  — up to or_parallel simultaneously
        gemini      — sequential (rate-limited)
        lmstudio    — sequential (eval_fn can do cell-level parallelism internally)
        local       — sequential

    Circuit breaker: if provided, models that fail max_failures times
    consecutively will be skipped for remaining cells. This prevents
    wasting API calls on broken models (e.g., GLM-5 returning None,
    R1 rate-limited for hours).

    Args:
        items: List of (provider, model_name) or (provider, model_name, condition) tuples.
              Extra elements beyond the first two are passed to eval_fn.
        eval_fn: Called with (provider, model_name, *extra, print_lock=lock, **eval_kwargs).
        or_parallel: Max parallel OpenRouter workers.
        lm_parallel: Passed through in eval_kwargs for LM Studio cell parallelism.
        print_lock: Shared lock for thread-safe console output. Created if not provided.
        circuit_breaker: Optional CircuitBreaker instance. Created with max_failures=3
            if not provided.
        **eval_kwargs: Additional keyword args forwarded to eval_fn.

    Returns:
        {identifier: result_dict} for all items.
        identifier is model_name for 2-tuples, "model_name:condition" for 3-tuples.
    """
    if print_lock is None:
        print_lock = threading.Lock()
    if circuit_breaker is None:
        circuit_breaker = CircuitBreaker(max_failures=3)

    # Normalize items to always have (provider, name, *extra)
    or_items = [i for i in items if i[0] == "openrouter"]
    gemini_items = [i for i in items if i[0] == "gemini"]
    lm_items = [i for i in items if i[0] == "lmstudio"]
    local_items = [i for i in items if i[0] == "local"]

    all_results: Dict[str, Dict[str, Any]] = {}

    def _key(item: tuple) -> str:
        """Generate result key: model_name or model_name:condition."""
        if len(item) > 2:
            return f"{item[1]}:{item[2]}"
        return item[1]

    def _call(item: tuple) -> Dict[str, Any]:
        provider = item[0]
        model_name = item[1]
        extra = item[2:]

        # Check circuit breaker before calling
        if circuit_breaker.is_tripped(model_name):
            with print_lock:
                print(f"  [SKIP] {model_name} — circuit breaker tripped (too many failures)")
            return {
                "model": model_name,
                "provider": provider,
                "status": "circuit_breaker",
                "error": f"Skipped: {circuit_breaker.max_failures} consecutive failures",
            }

        try:
            result = eval_fn(provider, model_name, *extra, print_lock=print_lock, **eval_kwargs)
            # Check if the result indicates failure (e.g., all-zero accuracy with error)
            if result.get("error") or result.get("status") == "incomplete":
                just_tripped = circuit_breaker.record_failure(model_name, str(result.get("error", "")))
                if just_tripped:
                    with print_lock:
                        print(f"\n  *** CIRCUIT BREAKER TRIPPED for {model_name} ***")
                        print(f"      {circuit_breaker.max_failures} consecutive failures — skipping remaining cells")
            else:
                circuit_breaker.record_success(model_name)
            return result
        except Exception as e:
            just_tripped = circuit_breaker.record_failure(model_name, str(e))
            if just_tripped:
                with print_lock:
                    print(f"\n  *** CIRCUIT BREAKER TRIPPED for {model_name} ***")
                    print(f"      {circuit_breaker.max_failures} consecutive failures — skipping remaining cells")
            raise

    def _run_sequential(group_items: List[tuple], label: str) -> None:
        if not group_items:
            return
        print(f"\n{'━' * 60}")
        print(f"PHASE: {label} ({len(group_items)} item(s), sequential)")
        print(f"{'━' * 60}")
        for item in group_items:
            key = _key(item)
            try:
                all_results[key] = _call(item)
            except Exception as e:
                all_results[key] = {
                    "model": item[1],
                    "provider": item[0],
                    "error": str(e),
                }

    # --- OpenRouter: parallel ---
    if or_items:
        print(f"\n{'━' * 60}")
        print(f"PHASE: OpenRouter ({len(or_items)} item(s), {or_parallel} parallel)")
        print(f"{'━' * 60}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=or_parallel) as pool:
            futures = {
                pool.submit(_call, item): item
                for item in or_items
            }
            for f in concurrent.futures.as_completed(futures):
                item = futures[f]
                key = _key(item)
                try:
                    all_results[key] = f.result()
                except Exception as e:
                    all_results[key] = {
                        "model": item[1],
                        "provider": "openrouter",
                        "error": str(e),
                    }

    # --- Sequential providers ---
    _run_sequential(gemini_items, "Gemini")
    _run_sequential(lm_items, f"LM Studio")
    _run_sequential(local_items, "Local")

    # Print circuit breaker summary if any models were tripped
    status = circuit_breaker.get_status()
    if status["tripped"]:
        print(f"\n  Circuit breaker summary:")
        for model in status["tripped"]:
            print(f"    {model}: TRIPPED ({status['failures'].get(model, 0)} failures)")

    return all_results


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

def print_cross_model_summary(
    all_results: Dict[str, Dict[str, Any]],
    conditions: List[str],
) -> None:
    """Print cross-model summary table with per-condition accuracy and FLR.

    Args:
        all_results: {model_name: result_dict} from a completed run.
        conditions: List of condition names to display.
    """
    # Per-model detailed view
    print("\n" + "=" * 80)
    print("SB2 SUMMARY — Accuracy per Turn (Tier 2, N=8)")
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

        # FLR estimate
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

        if "correction" in cond_avgs and "practice_only" in cond_avgs:
            flr = cond_avgs["correction"] - cond_avgs["practice_only"]
            row += f" {flr:>+7.3f}"
        else:
            row += f" {'---':>8}"

        print(row)
