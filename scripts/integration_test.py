#!/usr/bin/env python3
"""Integration test: 1 SB2 instance per model, all conditions, verify full pipeline.

Tests:
1. Answer extraction pipeline (unit, no API calls)
2. STS generator + solver (unit, no API calls)
3. Per-model: client creation, prompt, prompt_json, multi_turn, mini SB2 run
4. Results structure validation

Remote models run in parallel. Local/LM Studio models run SEQUENTIALLY
to avoid VRAM model-swap thrashing.
"""

import json
import os
import signal
import sys
import time
import traceback
import concurrent.futures
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from lesson.models.base import LLMClient
from lesson.eval.sb2_pilot import run_sb2_pilot
from lesson.eval.extraction import extract_answer, normalize_answer
from lesson.sts.generator import (
    generate_sts_instance, generate_training_set,
    subset_training_set, format_training_examples,
)
from lesson.sts.solver import solve

# Per-model timeout (seconds) — LM Studio models are slow (thinking + JIT load)
MODEL_TIMEOUT = 180


# ---------------------------------------------------------------------------
# Test models
# ---------------------------------------------------------------------------

PILOT_MODELS: List[Tuple[str, str]] = [
    ("openrouter", "glm-5"),
    ("openrouter", "gpt-5.3-codex"),
    ("openrouter", "gpt-5.3-chat"),
    ("gemini",     "gemini-flash"),
    ("openrouter", "claude-sonnet-4.6"),
    ("openrouter", "deepseek-r1"),
    ("openrouter", "deepseek-v3.2"),
    ("openrouter", "claude-haiku-4.5"),
]

# LM Studio models — run sequentially to avoid VRAM swap thrashing
LMSTUDIO_MODELS: List[Tuple[str, str]] = [
    ("lmstudio", "lm-qwen3.5-35b-a3b"),
    ("lmstudio", "lm-qwen3.5-27b"),
    ("lmstudio", "lm-glm-4.7-flash"),
    # nemotron excluded — currently unloaded in LM Studio
]


def get_client(provider: str, model_name: str) -> LLMClient:
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
# Unit tests (no API calls)
# ---------------------------------------------------------------------------

def test_extraction():
    print("\n" + "=" * 60)
    print("TEST: Extraction Pipeline")
    print("=" * 60)

    vocab = ["◈", "⬡", "⟐", "▲", "◆"]
    cases = [
        ('{"output": "◈⬡⟐"}', "◈⬡⟐", "clean JSON"),
        ('{"output": "◈ ⬡ ⟐"}', "◈⬡⟐", "JSON with spaces"),
        ('```json\n{"output": "▲◆"}\n```', "▲◆", "markdown fence"),
        ('I think the answer is ◈⬡⟐', "◈⬡⟐", "symbol-aware"),
        ('Wait ◈◆ no, actually ⬡⟐▲', "⬡⟐▲", "last symbol match"),
        ('Output: ◈⬡', "◈⬡", "Output: prefix"),
        ('', "", "empty response"),
        ('{"output": "▲', "▲", "truncated JSON"),
        ('<think>reasoning ◈◆</think>{"output": "⬡⟐"}', "⬡⟐", "think block + JSON"),
        ('\n\n{"output": "ZYX"}', "ZYX", "leading whitespace + JSON"),
        ('```json\n{"output": "ZYX"}\n```', "ZYX", "markdown + non-symbol"),
    ]

    passed = failed = 0
    for raw, expected, desc in cases:
        result = extract_answer(raw, mode="json", vocabulary=vocab)
        ok = result == expected
        passed += ok
        failed += not ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {desc}: {result!r} (expected {expected!r})")

    print(f"\n  Extraction: {passed}/{passed+failed} passed")
    return failed == 0


def test_sts_pipeline():
    print("\n" + "=" * 60)
    print("TEST: STS Generator + Solver")
    print("=" * 60)

    passed = failed = 0
    for tier in [1, 2, 3]:
        for seed in [42, 99, 200]:
            try:
                sts = generate_sts_instance(tier=tier, seed=seed)
                training = generate_training_set(sts, n_max=8)
                subset = subset_training_set(training, 4)
                for ex in subset:
                    solved = solve(sts, ex.input_seq)
                    if solved != ex.output_seq:
                        print(f"  [FAIL] T{tier} s={seed}: solver({ex.input_seq!r})={solved!r} != {ex.output_seq!r}")
                        failed += 1
                    else:
                        passed += 1
            except Exception as e:
                print(f"  [FAIL] T{tier} s={seed}: {e}")
                failed += 1

    print(f"  STS pipeline: {passed}/{passed+failed} passed")
    return failed == 0


# ---------------------------------------------------------------------------
# Per-model integration test
# ---------------------------------------------------------------------------

def test_model(provider: str, model_name: str) -> Dict[str, Any]:
    """Full integration test for one model. Returns result dict."""
    result = {"model": model_name, "provider": provider, "tests": {}, "errors": []}

    # Generate a fixed STS instance
    sts = generate_sts_instance(tier=2, seed=42)
    training = generate_training_set(sts, n_max=8)
    subset = subset_training_set(training, 4)
    examples_text = format_training_examples(subset)
    vocab = sts.alphabet
    vocab_line = " ".join(vocab)

    test_input = training[0].input_seq
    expected_output = training[0].output_seq
    prompt = (
        f"You are learning a symbolic transformation system.\n"
        f"The system uses these symbols: {vocab_line}\n"
        f"Here are 4 examples:\n\n{examples_text}\n\n"
        f"What is the output for: {test_input}\n"
        f'Respond with ONLY: {{"output": "YOUR_ANSWER"}}'
    )

    # Test 1: Client creation
    try:
        client = get_client(provider, model_name)
        result["tests"]["client_creation"] = "PASS"
    except Exception as e:
        result["tests"]["client_creation"] = f"FAIL: {e}"
        result["errors"].append(f"client_creation: {e}")
        return result

    # Test 2: Single-turn prompt
    try:
        t0 = time.time()
        raw = client.prompt(prompt)
        elapsed = time.time() - t0
        answer = extract_answer(raw, mode="json", vocabulary=vocab)
        expected = normalize_answer(expected_output)
        result["tests"]["prompt"] = {
            "status": "PASS" if raw.strip() else "FAIL: empty",
            "raw": raw[:200], "extracted": answer,
            "correct": answer == expected, "time": round(elapsed, 1),
        }
    except Exception as e:
        result["tests"]["prompt"] = f"FAIL: {e}"
        result["errors"].append(f"prompt: {e}")

    # Test 3: Single-turn prompt_json
    try:
        t0 = time.time()
        raw = client.prompt_json(prompt)
        elapsed = time.time() - t0
        answer = extract_answer(raw, mode="json", vocabulary=vocab)
        result["tests"]["prompt_json"] = {
            "status": "PASS" if raw.strip() else "FAIL: empty",
            "raw": raw[:200], "extracted": answer,
            "correct": answer == normalize_answer(expected_output),
            "time": round(elapsed, 1),
        }
    except Exception as e:
        result["tests"]["prompt_json"] = f"FAIL: {e}"
        result["errors"].append(f"prompt_json: {e}")

    # Test 4: Multi-turn session
    try:
        session = client.multi_turn()
        session.reset()
        session.inject(
            f"You are learning a symbolic transformation system.\n"
            f"Symbols: {vocab_line}\nExamples:\n{examples_text}",
            role="user",
        )
        session.inject('{"output": "understood"}', role="assistant")

        t0 = time.time()
        q = f'What is the output for: {test_input}\nRespond with ONLY: {{"output": "YOUR_ANSWER"}}'
        raw = session.send_json(q, role="user")
        elapsed = time.time() - t0
        answer = extract_answer(raw, mode="json", vocabulary=vocab)

        session.inject("Correct! Good job.", role="user")

        test2 = training[1].input_seq
        expected2 = training[1].output_seq
        q2 = f'What is the output for: {test2}\nRespond with ONLY: {{"output": "YOUR_ANSWER"}}'
        raw2 = session.send_json(q2, role="user")
        answer2 = extract_answer(raw2, mode="json", vocabulary=vocab)

        result["tests"]["multi_turn"] = {
            "status": "PASS",
            "t1_extracted": answer, "t1_correct": answer == normalize_answer(expected_output),
            "t2_extracted": answer2, "t2_correct": answer2 == normalize_answer(expected2),
            "time": round(elapsed, 1),
        }
    except Exception as e:
        result["tests"]["multi_turn"] = f"FAIL: {e}"
        result["errors"].append(f"multi_turn: {e}")

    # Test 5: Mini SB2 run (1 instance, 4 turns, 2 conditions)
    try:
        t0 = time.time()
        sb2 = run_sb2_pilot(
            client=client, tier=2, n_initial_examples=4,
            n_instances=1, n_turns=4,
            conditions=["correction", "no_feedback"],
            max_parallel=1,
        )
        elapsed = time.time() - t0

        assert "results" in sb2 and "summary" in sb2 and "model" in sb2
        n_results = len(sb2["results"])
        n_correct = sum(1 for r in sb2["results"] if r["correct"])

        for cond in ["correction", "no_feedback"]:
            assert cond in sb2["summary"]

        for r in sb2["results"]:
            for field in ["condition", "instance_idx", "sts_id", "tier", "turn_idx",
                          "input_seq", "correct", "model_answer", "expected_answer",
                          "feedback_given", "raw_response", "vocabulary"]:
                assert field in r, f"Missing {field}"

        per_cond = {}
        for cond, cond_data in sb2["summary"].items():
            accs = [d["n_correct"] / d["n_total"] for d in cond_data.values() if d["n_total"]]
            per_cond[cond] = round(sum(accs) / len(accs), 2) if accs else 0

        result["tests"]["sb2_mini"] = {
            "status": "PASS",
            "n_results": n_results, "n_correct": n_correct,
            "accuracy": round(n_correct / n_results, 2) if n_results else 0,
            "per_condition": per_cond, "time": round(elapsed, 1),
        }
    except Exception as e:
        result["tests"]["sb2_mini"] = f"FAIL: {e}"
        result["errors"].append(f"sb2_mini: {e}")
        traceback.print_exc()

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("LESSON-Bench Integration Test")
    print(f"  Time: {datetime.now().isoformat()}")
    print("=" * 70)

    extraction_ok = test_extraction()
    sts_ok = test_sts_pipeline()

    if not extraction_ok or not sts_ok:
        print("\n!! Unit tests failed — fix before running model tests")

    # Discover available models
    models_remote: List[Tuple[str, str]] = []
    models_local: List[Tuple[str, str]] = []

    if os.environ.get("OPENROUTER_API_KEY"):
        models_remote.extend(PILOT_MODELS)
        print(f"\n  OpenRouter + Gemini: {len(PILOT_MODELS)} pilot models")
    else:
        print("\n  OpenRouter: SKIPPED (no API key)")

    from lesson.models.lmstudio import check_lmstudio_server
    if check_lmstudio_server():
        models_local.extend(LMSTUDIO_MODELS)
        print(f"  LM Studio: {len(LMSTUDIO_MODELS)} models (will run SEQUENTIALLY)")
    else:
        print("  LM Studio: SKIPPED (server not running)")

    total = len(models_remote) + len(models_local)
    if not total:
        print("\nERROR: No models available!")
        sys.exit(1)
    print(f"  Total: {total} models")

    all_results: Dict[str, Dict] = {}

    # --- Remote models: parallel ---
    if models_remote:
        print(f"\n{'=' * 60}")
        print(f"REMOTE MODELS ({len(models_remote)}, parallel)")
        print(f"{'=' * 60}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(models_remote), 8)) as pool:
            futures = {pool.submit(test_model, p, m): (p, m) for p, m in models_remote}
            for f in concurrent.futures.as_completed(futures):
                p, m = futures[f]
                try:
                    r = f.result()
                    all_results[f"{p}:{m}"] = r
                    n_pass = sum(1 for t in r["tests"].values()
                                if (isinstance(t, str) and t == "PASS") or
                                   (isinstance(t, dict) and t.get("status", "").startswith("PASS")))
                    print(f"  {p}:{m} — {n_pass}/{len(r['tests'])} passed")
                except Exception as e:
                    print(f"  {p}:{m} — EXCEPTION: {e}")
                    all_results[f"{p}:{m}"] = {"model": m, "provider": p, "error": str(e)}

    # --- LM Studio models: SEQUENTIAL (one at a time to avoid VRAM swap) ---
    if models_local:
        print(f"\n{'=' * 60}")
        print(f"LM STUDIO MODELS ({len(models_local)}, sequential — avoiding VRAM swap)")
        print(f"{'=' * 60}")

        for p, m in models_local:
            print(f"\n  Testing {p}:{m}...")
            try:
                r = test_model(p, m)
                all_results[f"{p}:{m}"] = r
                n_pass = sum(1 for t in r["tests"].values()
                            if (isinstance(t, str) and t == "PASS") or
                               (isinstance(t, dict) and t.get("status", "").startswith("PASS")))
                print(f"  {p}:{m} — {n_pass}/{len(r['tests'])} passed")
            except Exception as e:
                print(f"  {p}:{m} — EXCEPTION: {e}")
                all_results[f"{p}:{m}"] = {"model": m, "provider": p, "error": str(e)}

    # --- Summary ---
    print("\n" + "=" * 70)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 70)

    print(f"\n  {'Model':<35} {'prompt':>8} {'json':>8} {'multi':>8} {'sb2':>8} {'acc':>6}")
    print("  " + "─" * 75)

    for key, r in sorted(all_results.items()):
        if "error" in r and "tests" not in r:
            print(f"  {key:<35} {'ERROR':>8}")
            continue

        tests = r.get("tests", {})
        cols = []
        for test_name in ["prompt", "prompt_json", "multi_turn", "sb2_mini"]:
            t = tests.get(test_name, "SKIP")
            if isinstance(t, str):
                cols.append("PASS" if t == "PASS" else "FAIL" if "FAIL" in t else "SKIP")
            elif isinstance(t, dict):
                s = t.get("status", "?")
                cols.append("PASS" if s.startswith("PASS") else "FAIL")
            else:
                cols.append("?")

        sb2 = tests.get("sb2_mini", {})
        acc = f"{sb2['accuracy']:.0%}" if isinstance(sb2, dict) and "accuracy" in sb2 else "—"

        row = f"  {key:<35}"
        for c in cols:
            row += f" {c:>8}"
        row += f" {acc:>6}"
        print(row)

    # Per-condition breakdown
    print(f"\n  SB2 Per-Condition Accuracy:")
    print(f"  {'Model':<35} {'correction':>12} {'no_feedback':>12}")
    print("  " + "─" * 60)
    for key, r in sorted(all_results.items()):
        sb2 = r.get("tests", {}).get("sb2_mini", {})
        if isinstance(sb2, dict) and "per_condition" in sb2:
            pc = sb2["per_condition"]
            print(f"  {key:<35} {pc.get('correction', 0):>11.0%} {pc.get('no_feedback', 0):>12.0%}")

    # Errors
    errors = []
    for key, r in all_results.items():
        for e in r.get("errors", []):
            errors.append(f"  {key}: {e}")
    if errors:
        print(f"\n  ERRORS ({len(errors)}):")
        for e in errors:
            print(f"    {e}")

    # Save
    results_dir = Path("results") / "integration_test"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results: {out_path}")

    total_tests = total_pass = 0
    for r in all_results.values():
        for t in r.get("tests", {}).values():
            total_tests += 1
            if (isinstance(t, str) and t == "PASS") or \
               (isinstance(t, dict) and t.get("status", "").startswith("PASS")):
                total_pass += 1

    print(f"\n  OVERALL: {total_pass}/{total_tests} tests across {len(all_results)} models")
    if total_pass == total_tests and extraction_ok and sts_ok:
        print("  VERDICT: ALL CLEAR — ready for full pilot run")
    else:
        print("  VERDICT: ISSUES FOUND — review before proceeding")


if __name__ == "__main__":
    main()
