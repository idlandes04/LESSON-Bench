from __future__ import annotations

"""SB1 pilot evaluation runner.

Runs STS items through an LLM and collects accuracy results across tiers and
N-values (learning-curve measurements).

v9.0+ improvements:
- Structured JSON output as primary extraction mode
- Symbol vocabulary included in every prompt
- Raw responses logged for debugging
- Answer normalization (strips whitespace/spaces between symbols)
- Tiered fallback: JSON → symbol-aware → regex
"""

import random
import concurrent.futures
import threading
from typing import Any, Dict, List, Optional

from lesson.sts.generator import generate_dataset, format_training_examples
from lesson.sts.types import STSDataset, TestItem

from .extraction import extract_answer, normalize_answer
from .interaction_log import InteractionLog


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _format_vocabulary(alphabet: list[str]) -> str:
    """Format the STS alphabet as a vocabulary line for the prompt."""
    return " ".join(alphabet)


def _build_prompt(
    n: int,
    examples_text: str,
    test_input: str,
    alphabet: list[str],
) -> str:
    """Construct the evaluation prompt for a single test item.

    Includes the symbol vocabulary so the model knows what symbols are valid,
    and requests JSON output for clean extraction.
    """
    vocab_line = _format_vocabulary(alphabet)
    return (
        f"Below are {n} examples of a symbolic transformation system.\n"
        f"The system uses these symbols: {vocab_line}\n"
        "Study the pattern, then predict the output for the final input.\n"
        f"\n{examples_text}\n"
        f"\nInput: {test_input}\n"
        'Respond with ONLY a JSON object: {"output": "YOUR_ANSWER"}\n'
        "Use only the symbols listed above in your answer."
    )


# ---------------------------------------------------------------------------
# Single-item evaluation (used by both sequential and parallel runners)
# ---------------------------------------------------------------------------

def _eval_single_item(
    client: Any,
    tier: int,
    n: int,
    inst_idx: int,
    item_idx: int,
    item: TestItem,
    examples_text: str,
    alphabet: list,
    sts_id: str,
) -> Dict[str, Any]:
    """Evaluate one test item against the model. Thread-safe."""
    prompt = _build_prompt(n, examples_text, item.input_seq, alphabet)

    try:
        raw_response = client.prompt_json(prompt)
    except Exception:
        try:
            raw_response = client.prompt(prompt)
        except Exception:
            raw_response = ""

    model_answer = extract_answer(raw_response, mode="json", vocabulary=alphabet)
    expected = normalize_answer(item.correct_output)
    correct = model_answer == expected

    return {
        "tier": tier,
        "n_examples": n,
        "instance_idx": inst_idx,
        "sts_id": sts_id,
        "item_idx": item_idx,
        "item_type": item.item_type.value,
        "correct": correct,
        "model_answer": model_answer,
        "expected_answer": expected,
        "input_seq": item.input_seq,
        "raw_response": raw_response,
        "vocabulary": alphabet,
    }


def _build_summary(all_results: List[Dict[str, Any]]) -> Dict:
    """Build summary: tier -> n -> item_type -> {n_correct, n_total}."""
    summary: Dict[int, Dict[int, Dict[str, Dict[str, int]]]] = {}
    for r in all_results:
        t = r["tier"]
        n = r["n_examples"]
        it = r["item_type"]
        summary.setdefault(t, {}).setdefault(n, {}).setdefault(
            it, {"n_correct": 0, "n_total": 0}
        )
        summary[t][n][it]["n_total"] += 1
        if r["correct"]:
            summary[t][n][it]["n_correct"] += 1
    return summary


def _print_summary(summary: Dict) -> None:
    """Print the SB1 summary table."""
    print("\n--- SB1 Pilot Summary ---")
    for tier in sorted(summary):
        for n in sorted(summary[tier]):
            for it, counts in sorted(summary[tier][n].items()):
                acc = counts["n_correct"] / counts["n_total"] if counts["n_total"] else 0.0
                print(
                    f"  tier={tier} N={n:2d} type={it}: "
                    f"{counts['n_correct']}/{counts['n_total']} ({acc:.0%})"
                )


# ---------------------------------------------------------------------------
# Core pilot runner
# ---------------------------------------------------------------------------

def run_sb1_pilot(
    client: Any,  # LLMClient (lesson.models.base.LLMClient)
    tiers: List[int] = [2, 3, 4],
    n_values: List[int] = [2, 4, 8],
    n_instances: int = 5,
    seq_length: int = 5,
) -> Dict[str, Any]:
    """Run SB1 pilot: learning curves across tiers and N values.

    Automatically uses parallel inference when client.max_concurrent > 1
    (e.g. LM Studio with n_parallel slots on a beefy GPU).

    For each (tier, N, instance), generates 5 test items and prompts the model.
    Uses structured JSON output as primary extraction, with symbol-aware and
    regex fallbacks.

    Args:
        client:       An LLMClient instance (must implement .prompt_json(text)).
        tiers:        Difficulty tiers to evaluate.
        n_values:     Training-example counts (N) to sweep.
        n_instances:  Number of independent STS instances per (tier, N) cell.
        seq_length:   Length of input/output symbol sequences.

    Returns:
        {
            "results": [flat list of per-item result dicts, including raw_response],
            "summary": {tier -> {n -> {item_type -> {"n_correct": int, "n_total": int}}}},
            "model": client.name,
        }
    """
    max_concurrent = getattr(client, "max_concurrent", 1)
    if max_concurrent > 1:
        return _run_sb1_parallel(client, tiers, n_values, n_instances, seq_length, max_concurrent)
    return _run_sb1_sequential(client, tiers, n_values, n_instances, seq_length)


def _run_sb1_sequential(
    client: Any,
    tiers: List[int],
    n_values: List[int],
    n_instances: int,
    seq_length: int,
) -> Dict[str, Any]:
    """Original sequential SB1 runner."""
    all_results: List[Dict[str, Any]] = []
    model_name = getattr(client, "name", "unknown")
    log = InteractionLog(f"{model_name}_sb1")

    total_cells = len(tiers) * len(n_values) * n_instances
    cell_idx = 0

    for tier in tiers:
        for n in n_values:
            for inst_idx in range(n_instances):
                cell_idx += 1
                seed = tier * 10_000 + n * 100 + inst_idx

                print(
                    f"[{cell_idx}/{total_cells}] tier={tier}, N={n}, "
                    f"instance={inst_idx} (seed={seed})"
                )

                # Generate dataset with 5 test items (3R + 1E + 1L)
                try:
                    dataset: STSDataset = generate_dataset(
                        tier=tier,
                        seed=seed,
                        n_examples=n,
                        n_test_items=(3, 1, 1),
                        seq_length=seq_length,
                    )
                except Exception as exc:
                    print(f"  WARNING: generate_dataset failed: {exc}")
                    continue

                examples_text = format_training_examples(dataset.training_examples)
                alphabet = dataset.sts.alphabet

                for item_idx, item in enumerate(dataset.test_items):
                    result = _eval_single_item(
                        client, tier, n, inst_idx, item_idx, item,
                        examples_text, alphabet, dataset.sts.id,
                    )
                    all_results.append(result)

                    log.record(
                        prompt=_build_prompt(n, examples_text, item.input_seq, alphabet),
                        raw_response=result["raw_response"],
                        extracted=result["model_answer"],
                        expected=result["expected_answer"],
                        correct=result["correct"],
                        metadata={
                            "tier": tier, "n": n, "instance": inst_idx,
                            "item_type": item.item_type.value,
                            "input_seq": item.input_seq,
                        },
                    )

                    status = "CORRECT" if result["correct"] else f"wrong (got {result['model_answer']!r}, expected {result['expected_answer']!r})"
                    print(f"  item {item_idx} [{item.item_type.value}] {item.input_seq} → {status}")

    summary = _build_summary(all_results)
    _print_summary(summary)
    log.close()

    return {
        "results": all_results,
        "summary": summary,
        "model": getattr(client, "name", "unknown"),
        "log_file": str(log.filepath),
    }


def _run_sb1_parallel(
    client: Any,
    tiers: List[int],
    n_values: List[int],
    n_instances: int,
    seq_length: int,
    max_workers: int,
) -> Dict[str, Any]:
    """Parallel SB1 runner — submits items to a thread pool.

    LM Studio (and llama-server with --parallel) can process multiple
    requests concurrently.  This sends up to *max_workers* prompts at once,
    dramatically speeding up SB1 which has no inter-item dependencies.
    """
    model_name = getattr(client, "name", "unknown")
    log = InteractionLog(f"{model_name}_sb1")
    print_lock = threading.Lock()

    # Pre-generate all work items (datasets are cheap, model calls are not)
    work_items = []
    total_cells = len(tiers) * len(n_values) * n_instances
    cell_idx = 0

    for tier in tiers:
        for n in n_values:
            for inst_idx in range(n_instances):
                cell_idx += 1
                seed = tier * 10_000 + n * 100 + inst_idx

                try:
                    dataset: STSDataset = generate_dataset(
                        tier=tier, seed=seed, n_examples=n,
                        n_test_items=(3, 1, 1), seq_length=seq_length,
                    )
                except Exception as exc:
                    with print_lock:
                        print(f"  WARNING: generate_dataset failed (seed={seed}): {exc}")
                    continue

                examples_text = format_training_examples(dataset.training_examples)
                alphabet = dataset.sts.alphabet

                for item_idx, item in enumerate(dataset.test_items):
                    work_items.append((
                        tier, n, inst_idx, item_idx, item,
                        examples_text, alphabet, dataset.sts.id,
                    ))

    print(f"SB1 parallel: {len(work_items)} items across {total_cells} cells, "
          f"{max_workers} concurrent slots")

    all_results: List[Dict[str, Any]] = []
    completed_count = [0]  # mutable container for closure

    def _do_item(args):
        tier, n, inst_idx, item_idx, item, examples_text, alphabet, sts_id = args
        return _eval_single_item(
            client, tier, n, inst_idx, item_idx, item,
            examples_text, alphabet, sts_id,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_do_item, w): w for w in work_items}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            tier, n, inst_idx, item_idx, item, examples_text, alphabet, sts_id = futures[future]

            # All shared state updates under the lock
            with print_lock:
                all_results.append(result)
                log.record(
                    prompt=_build_prompt(n, examples_text, item.input_seq, alphabet),
                    raw_response=result["raw_response"],
                    extracted=result["model_answer"],
                    expected=result["expected_answer"],
                    correct=result["correct"],
                    metadata={
                        "tier": tier, "n": n, "instance": inst_idx,
                        "item_type": item.item_type.value,
                        "input_seq": item.input_seq,
                    },
                )
                completed_count[0] += 1
                status = "CORRECT" if result["correct"] else f"wrong (got {result['model_answer']!r})"
                print(f"  [{completed_count[0]}/{len(work_items)}] T{tier} N={n} i{inst_idx} "
                      f"[{item.item_type.value}] {item.input_seq} → {status}")

    summary = _build_summary(all_results)
    _print_summary(summary)
    log.close()

    return {
        "results": all_results,
        "summary": summary,
        "model": model_name,
        "log_file": str(log.filepath),
    }


# ---------------------------------------------------------------------------
# Mock client for pipeline testing (no live model required)
# ---------------------------------------------------------------------------

class _MockClient:
    """Dummy LLM client that returns random symbol sequences.

    Used only for testing the evaluation pipeline without a live model.
    """

    name = "mock-random"

    def __init__(self, seed: int = 42) -> None:
        self._rng = random.Random(seed)

    def prompt(self, text: str) -> str:
        symbols = ["◈", "⬡", "⟐", "⧫", "△", "▽", "○", "□"]
        length = self._rng.randint(2, 6)
        return "".join(self._rng.choice(symbols) for _ in range(length))

    def prompt_json(self, text: str) -> str:
        import json as _json
        symbols = ["◈", "⬡", "⟐", "⧫", "△", "▽", "○", "□"]
        length = self._rng.randint(2, 6)
        answer = "".join(self._rng.choice(symbols) for _ in range(length))
        return _json.dumps({"output": answer})

    def multi_turn(self) -> "_MockMultiTurnSession":
        return _MockMultiTurnSession(self._rng)


class _MockMultiTurnSession:
    """Dummy multi-turn session backed by _MockClient."""

    def __init__(self, rng: random.Random) -> None:
        self._rng = rng
        self._history: List[Dict[str, str]] = []

    def _random_answer(self) -> str:
        symbols = ["◈", "⬡", "⟐", "⧫", "△", "▽", "○", "□"]
        length = self._rng.randint(2, 6)
        return "".join(self._rng.choice(symbols) for _ in range(length))

    def send(self, text: str, role: str = "user") -> str:
        self._history.append({"role": role, "content": text})
        response = self._random_answer()
        self._history.append({"role": "assistant", "content": response})
        return response

    def send_json(self, text: str, role: str = "user") -> str:
        self._history.append({"role": role, "content": text})
        import json as _json
        response = _json.dumps({"output": self._random_answer()})
        self._history.append({"role": "assistant", "content": response})
        return response

    def inject(self, text: str, role: str = "assistant") -> None:
        self._history.append({"role": role, "content": text})

    def reset(self) -> None:
        self._history = []


# ---------------------------------------------------------------------------
# Self-test / demo entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running SB1 pilot with mock client (no live model required)...\n")

    mock = _MockClient(seed=0)

    results = run_sb1_pilot(
        client=mock,
        tiers=[2, 3],
        n_values=[2, 4],
        n_instances=2,
        seq_length=5,
    )

    print(f"\nTotal items evaluated: {len(results['results'])}")
    n_correct = sum(1 for r in results["results"] if r["correct"])
    n_total = len(results["results"])
    if n_total:
        print(f"Overall mock accuracy: {n_correct}/{n_total} ({n_correct/n_total:.0%})")
    print("\nPipeline test complete — all components exercised successfully.")
