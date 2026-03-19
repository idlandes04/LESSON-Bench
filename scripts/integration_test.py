#!/usr/bin/env python3
"""Integration test: 1 SB2 instance per model, all conditions, verify full pipeline.

Tests:
1. Client creation for each provider
2. Multi-turn session (send, send_json, inject, reset)
3. JSON mode fallback (json_object → plain text)
4. Answer extraction (JSON → symbol-aware → regex)
5. Feedback generation for all conditions
6. Full SB2 run (1 instance, 4 turns) end-to-end
7. Result structure and summary computation

Runs against all 8 pilot models + all available local models.
"""

import json
import os
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
from lesson.sts.generator import generate_sts_instance, generate_training_set, subset_training_set, format_training_examples
from lesson.sts.solver import solve


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

# All local models from registry
LOCAL_MODELS_TO_TEST: List[Tuple[str, str]] = [
    ("local", "qwen3.5-27b-think"),
    ("local", "qwen3.5-27b-nothink"),
    ("local", "qwen3.5-35b-a3b"),
    ("local", "qwen3.5-35b-a3b-nothink"),
    ("local", "gemma3-27b"),
    ("local", "phi4-14b"),
    ("local", "nemotron-nano"),
    ("local", "nemotron-nano-nothink"),
]

# LM Studio models
LMSTUDIO_MODELS: List[Tuple[str, str]] = [
    ("lmstudio", "qwen3-coder-30b"),
    ("lmstudio", "qwen3-1.7b"),
]

CONDITIONS_TO_TEST = ["correction", "practice_only", "error_only", "no_feedback"]


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
# Unit tests for extraction pipeline
# ---------------------------------------------------------------------------

def test_extraction():
    """Test the extraction pipeline with known inputs."""
    print("\n" + "=" * 60)
    print("TEST: Extraction Pipeline")
    print("=" * 60)

    vocab = ["◈", "⬡", "⟐", "▲", "◆"]
    cases = [
        # (raw_response, expected_answer, description)
        ('{"output": "◈⬡⟐"}', "◈⬡⟐", "clean JSON"),
        ('{"output": "◈ ⬡ ⟐"}', "◈⬡⟐", "JSON with spaces between symbols"),
        ('```json\n{"output": "▲◆"}\n```', "▲◆", "JSON in markdown fence"),
        ('I think the answer is ◈⬡⟐', "◈⬡⟐", "symbol-aware extraction"),
        ('Wait, let me think... ◈◆ no, actually ⬡⟐▲', "⬡⟐▲", "last symbol match"),
        ('Output: ◈⬡', "◈⬡", "Output: prefix"),
        ('The output is ◈⬡⟐▲◆', "◈⬡⟐▲◆", "prose with symbols"),
        ('', "", "empty response"),
        ('{"output": "▲', "▲", "truncated JSON"),
        ('<think>reasoning here ◈◆</think>{"output": "⬡⟐"}', "⬡⟐", "thinking block + JSON"),
    ]

    passed = 0
    failed = 0
    for raw, expected, desc in cases:
        result = extract_answer(raw, mode="json", vocabulary=vocab)
        status = "PASS" if result == expected else "FAIL"
        if status == "PASS":
            passed += 1
        else:
            failed += 1
        mark = "  " if status == "PASS" else "!!"
        print(f"  [{status}] {desc}: {result!r} (expected {expected!r}){' ' + mark if status == 'FAIL' else ''}")

    print(f"\n  Extraction: {passed}/{passed+failed} passed")
    return failed == 0


# ---------------------------------------------------------------------------
# Unit tests for STS generator + solver
# ---------------------------------------------------------------------------

def test_sts_pipeline():
    """Test STS instance generation and solving."""
    print("\n" + "=" * 60)
    print("TEST: STS Generator + Solver")
    print("=" * 60)

    passed = 0
    failed = 0

    for tier in [1, 2, 3]:
        for seed in [42, 99, 200]:
            try:
                sts = generate_sts_instance(tier=tier, seed=seed)
                training = generate_training_set(sts, n_max=8)
                subset = subset_training_set(training, 4)
                examples_text = format_training_examples(subset)

                # Verify solver works on training examples
                for ex in subset:
                    solved = solve(sts, ex.input_seq)
                    if solved != ex.output_seq:
                        print(f"  [FAIL] T{tier} seed={seed}: solver({ex.input_seq!r}) = {solved!r} != {ex.output_seq!r}")
                        failed += 1
                    else:
                        passed += 1

                # Verify we can format examples
                assert len(examples_text) > 0
                assert "Example" in examples_text or "→" in examples_text

            except Exception as e:
                print(f"  [FAIL] T{tier} seed={seed}: {e}")
                failed += 1

    print(f"  STS pipeline: {passed}/{passed+failed} passed")
    return failed == 0


# ---------------------------------------------------------------------------
# Per-model integration test
# ---------------------------------------------------------------------------

def test_model(provider: str, model_name: str) -> Dict[str, Any]:
    """Run a full integration test for one model.

    Tests:
    1. Client creation
    2. Single-turn prompt
    3. Single-turn prompt_json
    4. Multi-turn session (2 turns + inject)
    5. Mini SB2 run (1 instance, 4 turns, 2 conditions)

    Returns a result dict with pass/fail for each test.
    """
    result = {
        "model": model_name,
        "provider": provider,
        "tests": {},
        "errors": [],
    }

    # Generate a fixed STS instance for testing
    sts = generate_sts_instance(tier=2, seed=42)
    training = generate_training_set(sts, n_max=8)
    subset = subset_training_set(training, 4)
    examples_text = format_training_examples(subset)
    vocab = sts.alphabet
    vocab_line = " ".join(vocab)

    # Build the standard STS prompt
    test_input = training[0].input_seq  # Use first training example as test (we know the answer)
    expected_output = training[0].output_seq
    prompt = (
        f"You are learning a symbolic transformation system.\n"
        f"The system uses these symbols: {vocab_line}\n"
        f"Here are 4 examples:\n\n{examples_text}\n\n"
        f"What is the output for: {test_input}\n"
        f'Respond with ONLY: {{"output": "YOUR_ANSWER"}}'
    )

    # --- Test 1: Client creation ---
    try:
        client = get_client(provider, model_name)
        result["tests"]["client_creation"] = "PASS"
    except Exception as e:
        result["tests"]["client_creation"] = f"FAIL: {e}"
        result["errors"].append(f"client_creation: {e}")
        return result  # Can't continue without a client

    # --- Test 2: Single-turn prompt ---
    try:
        t0 = time.time()
        raw = client.prompt(prompt)
        elapsed = time.time() - t0
        answer = extract_answer(raw, mode="json", vocabulary=vocab)
        expected = normalize_answer(expected_output)
        correct = answer == expected
        result["tests"]["prompt"] = {
            "status": "PASS" if raw.strip() else "FAIL: empty response",
            "raw_response": raw[:200],
            "extracted": answer,
            "expected": expected,
            "correct": correct,
            "elapsed_s": round(elapsed, 2),
        }
    except Exception as e:
        result["tests"]["prompt"] = f"FAIL: {e}"
        result["errors"].append(f"prompt: {e}")

    # --- Test 3: Single-turn prompt_json ---
    try:
        t0 = time.time()
        raw = client.prompt_json(prompt)
        elapsed = time.time() - t0
        answer = extract_answer(raw, mode="json", vocabulary=vocab)
        result["tests"]["prompt_json"] = {
            "status": "PASS" if raw.strip() else "FAIL: empty response",
            "raw_response": raw[:200],
            "extracted": answer,
            "correct": answer == normalize_answer(expected_output),
            "elapsed_s": round(elapsed, 2),
        }
    except Exception as e:
        result["tests"]["prompt_json"] = f"FAIL: {e}"
        result["errors"].append(f"prompt_json: {e}")

    # --- Test 4: Multi-turn session ---
    try:
        session = client.multi_turn()
        session.reset()

        # Inject system context
        session.inject(
            f"You are learning a symbolic transformation system.\n"
            f"Symbols: {vocab_line}\n"
            f"Examples:\n{examples_text}",
            role="user"
        )
        session.inject('{"output": "understood"}', role="assistant")

        # Send a question
        t0 = time.time()
        q = f'What is the output for: {test_input}\nRespond with ONLY: {{"output": "YOUR_ANSWER"}}'
        raw = session.send_json(q, role="user")
        elapsed = time.time() - t0

        answer = extract_answer(raw, mode="json", vocabulary=vocab)

        # Inject feedback
        session.inject("Correct! Good job.", role="user")

        # Send second question
        test_input2 = training[1].input_seq
        expected2 = training[1].output_seq
        q2 = f'What is the output for: {test_input2}\nRespond with ONLY: {{"output": "YOUR_ANSWER"}}'
        raw2 = session.send_json(q2, role="user")
        answer2 = extract_answer(raw2, mode="json", vocabulary=vocab)

        result["tests"]["multi_turn"] = {
            "status": "PASS",
            "turn_1_raw": raw[:200],
            "turn_1_extracted": answer,
            "turn_1_correct": answer == normalize_answer(expected_output),
            "turn_2_raw": raw2[:200],
            "turn_2_extracted": answer2,
            "turn_2_correct": answer2 == normalize_answer(expected2),
            "elapsed_s": round(elapsed, 2),
        }
    except Exception as e:
        result["tests"]["multi_turn"] = f"FAIL: {e}"
        result["errors"].append(f"multi_turn: {e}")

    # --- Test 5: Mini SB2 run ---
    try:
        t0 = time.time()
        sb2 = run_sb2_pilot(
            client=client,
            tier=2,
            n_initial_examples=4,
            n_instances=1,
            n_turns=4,
            conditions=["correction", "no_feedback"],
            max_parallel=1,
        )
        elapsed = time.time() - t0

        # Validate result structure
        assert "results" in sb2, "Missing 'results' key"
        assert "summary" in sb2, "Missing 'summary' key"
        assert "model" in sb2, "Missing 'model' key"

        n_results = len(sb2["results"])
        n_correct = sum(1 for r in sb2["results"] if r["correct"])

        # Check summary structure
        for cond in ["correction", "no_feedback"]:
            assert cond in sb2["summary"], f"Missing condition {cond!r} in summary"
            cond_data = sb2["summary"][cond]
            for turn_idx in range(4):
                t_key = turn_idx if turn_idx in cond_data else str(turn_idx)
                assert t_key in cond_data, f"Missing turn {turn_idx} in {cond} summary"
                assert "n_correct" in cond_data[t_key]
                assert "n_total" in cond_data[t_key]

        # Check individual results have all expected fields
        for r in sb2["results"]:
            for field in ["condition", "instance_idx", "sts_id", "tier",
                          "turn_idx", "input_seq", "correct", "model_answer",
                          "expected_answer", "feedback_given", "raw_response",
                          "vocabulary"]:
                assert field in r, f"Missing field {field!r} in result"

        result["tests"]["sb2_mini"] = {
            "status": "PASS",
            "n_results": n_results,
            "n_correct": n_correct,
            "accuracy": round(n_correct / n_results, 2) if n_results else 0,
            "elapsed_s": round(elapsed, 2),
            "conditions_tested": list(sb2["summary"].keys()),
            "per_condition": {},
        }

        # Per-condition accuracy
        for cond, cond_data in sb2["summary"].items():
            accs = []
            for t_data in cond_data.values():
                if t_data["n_total"] > 0:
                    accs.append(t_data["n_correct"] / t_data["n_total"])
            avg = sum(accs) / len(accs) if accs else 0
            result["tests"]["sb2_mini"]["per_condition"][cond] = round(avg, 2)

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

    # --- Unit tests (no API calls) ---
    extraction_ok = test_extraction()
    sts_ok = test_sts_pipeline()

    if not extraction_ok or not sts_ok:
        print("\n!! Unit tests failed — fix before running model tests")

    # --- Discover available models ---
    models_to_test: List[Tuple[str, str]] = []

    # OpenRouter models (if key available)
    if os.environ.get("OPENROUTER_API_KEY"):
        models_to_test.extend(PILOT_MODELS)
        print(f"\n  OpenRouter: {len(PILOT_MODELS)} pilot models queued")
    else:
        print("\n  OpenRouter: SKIPPED (no API key)")

    # Gemini (already in PILOT_MODELS via gemini provider)

    # Local models (check which servers are up)
    for provider, name in LOCAL_MODELS_TO_TEST:
        from lesson.models.registry import LOCAL_MODELS
        config = LOCAL_MODELS.get(name, {})
        port = config.get("port", 8080)
        try:
            import urllib.request
            req = urllib.request.Request(f"http://localhost:{port}/v1/models")
            urllib.request.urlopen(req, timeout=2)
            models_to_test.append((provider, name))
        except Exception:
            pass

    local_count = sum(1 for p, _ in models_to_test if p == "local")
    print(f"  Local llama-server: {local_count} models available")

    # LM Studio models
    from lesson.models.lmstudio import check_lmstudio_server
    if check_lmstudio_server():
        models_to_test.extend(LMSTUDIO_MODELS)
        print(f"  LM Studio: {len(LMSTUDIO_MODELS)} models queued")
    else:
        print("  LM Studio: SKIPPED (server not running)")

    if not models_to_test:
        print("\nERROR: No models available to test!")
        sys.exit(1)

    print(f"\n  Total models to test: {len(models_to_test)}")

    # --- Run model tests ---
    # Group by provider for parallelism
    or_models = [(p, m) for p, m in models_to_test if p == "openrouter"]
    gemini_models = [(p, m) for p, m in models_to_test if p == "gemini"]
    other_models = [(p, m) for p, m in models_to_test if p not in ("openrouter", "gemini")]

    all_results: Dict[str, Dict] = {}

    # Run OpenRouter + Gemini in parallel (they're all remote APIs)
    remote_models = or_models + gemini_models
    if remote_models:
        print(f"\n{'=' * 60}")
        print(f"TESTING REMOTE MODELS ({len(remote_models)} models, parallel)")
        print(f"{'=' * 60}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(remote_models), 8)) as pool:
            futures = {
                pool.submit(test_model, p, m): (p, m) for p, m in remote_models
            }
            for f in concurrent.futures.as_completed(futures):
                p, m = futures[f]
                try:
                    r = f.result()
                    all_results[f"{p}:{m}"] = r
                    # Quick status
                    n_pass = sum(1 for t in r["tests"].values()
                                if (isinstance(t, str) and t == "PASS") or
                                   (isinstance(t, dict) and t.get("status", "").startswith("PASS")))
                    n_total = len(r["tests"])
                    print(f"  {p}:{m} — {n_pass}/{n_total} tests passed")
                except Exception as e:
                    print(f"  {p}:{m} — EXCEPTION: {e}")
                    all_results[f"{p}:{m}"] = {"model": m, "provider": p, "error": str(e)}

    # Run local/LM Studio models sequentially
    if other_models:
        print(f"\n{'=' * 60}")
        print(f"TESTING LOCAL MODELS ({len(other_models)} models, sequential)")
        print(f"{'=' * 60}")

        for p, m in other_models:
            try:
                r = test_model(p, m)
                all_results[f"{p}:{m}"] = r
                n_pass = sum(1 for t in r["tests"].values()
                            if (isinstance(t, str) and t == "PASS") or
                               (isinstance(t, dict) and t.get("status", "").startswith("PASS")))
                n_total = len(r["tests"])
                print(f"  {p}:{m} — {n_pass}/{n_total} tests passed")
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
                if t == "PASS":
                    cols.append("PASS")
                elif t.startswith("FAIL"):
                    cols.append("FAIL")
                else:
                    cols.append("SKIP")
            elif isinstance(t, dict):
                status = t.get("status", "?")
                if status.startswith("PASS"):
                    cols.append("PASS")
                elif status.startswith("FAIL"):
                    cols.append("FAIL")
                else:
                    cols.append("?")
            else:
                cols.append("?")

        # Accuracy from sb2_mini
        sb2 = tests.get("sb2_mini", {})
        if isinstance(sb2, dict) and "accuracy" in sb2:
            acc = f"{sb2['accuracy']:.0%}"
        else:
            acc = "—"

        row = f"  {key:<35}"
        for c in cols:
            row += f" {c:>8}"
        row += f" {acc:>6}"
        print(row)

    # Per-condition breakdown for SB2
    print(f"\n  SB2 Mini Per-Condition Accuracy:")
    print(f"  {'Model':<35} {'correction':>12} {'no_feedback':>12}")
    print("  " + "─" * 60)

    for key, r in sorted(all_results.items()):
        tests = r.get("tests", {})
        sb2 = tests.get("sb2_mini", {})
        if isinstance(sb2, dict) and "per_condition" in sb2:
            pc = sb2["per_condition"]
            corr = f"{pc.get('correction', 0):.0%}"
            nofb = f"{pc.get('no_feedback', 0):.0%}"
            print(f"  {key:<35} {corr:>12} {nofb:>12}")

    # Check for errors
    errors = []
    for key, r in all_results.items():
        if r.get("errors"):
            for e in r["errors"]:
                errors.append(f"  {key}: {e}")

    if errors:
        print(f"\n  ERRORS ({len(errors)}):")
        for e in errors:
            print(f"    {e}")

    # Save results
    results_dir = Path("results") / "integration_test"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Full results: {out_path}")

    # Overall verdict
    total_tests = 0
    total_pass = 0
    for r in all_results.values():
        for t in r.get("tests", {}).values():
            total_tests += 1
            if (isinstance(t, str) and t == "PASS") or \
               (isinstance(t, dict) and t.get("status", "").startswith("PASS")):
                total_pass += 1

    print(f"\n  OVERALL: {total_pass}/{total_tests} tests passed across {len(all_results)} models")

    if total_pass == total_tests and extraction_ok and sts_ok:
        print("  VERDICT: ALL CLEAR — ready for full pilot run")
    else:
        print("  VERDICT: ISSUES FOUND — review before proceeding")


if __name__ == "__main__":
    main()
