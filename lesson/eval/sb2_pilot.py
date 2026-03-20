from __future__ import annotations

"""SB2 pilot evaluation runner.

Tests feedback conditions in a multi-turn setting to measure how different
feedback types affect within-session learning.

Core conditions (v11.0): correction, practice_only, error_only, no_feedback
Extended: explanation, misleading
Mechanistic probes: clean_context, prompted_correction, structured_correction,
    reformatted_correction

v9.0+ improvements:
- Structured JSON output via send_json() as primary extraction
- Symbol vocabulary included in system context
- Raw responses logged for debugging
- Answer normalization for robust comparison
- Tiered fallback: JSON → symbol-aware → regex

v11.0 additions:
- no_feedback condition ("Next question." — no evaluation, no answer)
- clean_context condition (single-prompt-with-history, correct pairs only)
- prompted_correction condition (reflection prompt after error)
- structured_correction condition (code-test-style error format)
- reformatted_correction condition (example-formatted feedback)
"""

import concurrent.futures
import random
import threading
from typing import Any, Dict, List, Optional, Tuple

from lesson.sts.generator import (
    generate_sts_instance,
    generate_training_set,
    subset_training_set,
    generate_input_sequence,
    format_training_examples,
)
from lesson.sts.solver import solve, get_partial_rule_answers
from lesson.sts.types import STSInstance, TrainingExample

from .extraction import extract_answer, normalize_answer
from .interaction_log import InteractionLog


# ---------------------------------------------------------------------------
# Canonical condition lists (import from here, not redefined elsewhere)
# ---------------------------------------------------------------------------

CORE_CONDITIONS = ["correction", "practice_only", "error_only", "no_feedback"]

ALL_CONDITIONS = [
    "correction", "practice_only", "error_only", "no_feedback",
    "explanation", "misleading",
    "clean_context", "prompted_correction", "structured_correction",
    "reformatted_correction",
]


# ---------------------------------------------------------------------------
# Feedback generators
# ---------------------------------------------------------------------------

def _feedback_correction(input_seq: str, correct: str, model_answer: str) -> str:
    """Correction feedback: confirms right answers, gives correct answer on errors."""
    if model_answer == correct:
        return f"Correct! The output is {correct}."
    return f"Incorrect. The correct output is {correct}."


def _feedback_practice_only(input_seq: str, correct: str, model_answer: str) -> str:
    """Practice-only feedback: always reveals the answer, phrased as a transition."""
    return f"The output for {input_seq} is {correct}. Next question."


def _feedback_error_only(input_seq: str, correct: str, model_answer: str) -> str:
    """Error-only feedback: says correct/incorrect but never reveals the answer."""
    if model_answer == correct:
        return "Correct!"
    return "Incorrect."


def _feedback_explanation(
    sts: STSInstance, input_seq: str, correct: str, model_answer: str
) -> str:
    """Explanation feedback: includes a brief rule description."""
    if model_answer == correct:
        base = f"Correct! The output is {correct}."
    else:
        base = f"Incorrect. The correct output is {correct}."

    # Append a concise rule description
    rule_parts = []
    for rule in sts.rules:
        rt = rule.rule_type.value
        spec = rule.spec
        if rt == "direct":
            rule_parts.append(
                f"direct replacement: '{spec['pattern']}' → '{spec['replacement']}'"
            )
        elif rt == "positional":
            rule_parts.append(
                f"positional rule at index {spec['position']}"
            )
        elif rt == "conditional":
            rule_parts.append(
                f"conditional rule triggered by '{spec['trigger']}'"
            )
        elif rt == "compositional":
            rule_parts.append("compositional (chained) rule")

    if rule_parts:
        rule_summary = "; ".join(rule_parts)
        base += f" (Rules: {rule_summary}.)"
    return base


def _feedback_misleading(
    sts: STSInstance,
    input_seq: str,
    correct: str,
    model_answer: str,
    rng: random.Random,
    mislead_this_turn: bool,
) -> str:
    """Misleading feedback: provides a wrong answer on designated turns."""
    if not mislead_this_turn:
        return _feedback_correction(input_seq, correct, model_answer)

    # Generate a wrong answer (different from correct)
    partials = get_partial_rule_answers(sts, input_seq)
    if partials:
        wrong = next(iter(partials.values()))
    else:
        # Fallback: random symbol from alphabet
        wrong = rng.choice(sts.alphabet)
        attempts = 0
        while wrong == correct and len(sts.alphabet) > 1 and attempts < 10:
            wrong = rng.choice(sts.alphabet)
            attempts += 1

    return f"Correct! The output is {wrong}."  # Deceptively marks it as correct


def _feedback_no_feedback(input_seq: str, correct: str, model_answer: str) -> str:
    """No-feedback condition: always returns a static transition phrase."""
    return "Next question."


def _feedback_prompted_correction(
    input_seq: str, correct: str, model_answer: str
) -> str:
    """Prompted correction: correction feedback plus a reflection prompt on errors."""
    if model_answer == correct:
        return f"Correct! The output is {correct}."
    return (
        f"Incorrect. The correct output is {correct}.\n"
        "Before answering the next question, briefly state what pattern you "
        "think explains why your previous answer was wrong."
    )


def _feedback_structured_correction(
    input_seq: str, correct: str, model_answer: str
) -> str:
    """Structured correction: code-test-style pass/fail with per-position diff."""
    if model_answer == correct:
        return (
            f"TEST PASSED\n"
            f"Input: {input_seq}\n"
            f"Expected: {correct}\n"
            f"Your output: {model_answer}\n"
            f"Result: PASS"
        )

    # Build per-position diff (only mismatched positions)
    diff_parts = []
    max_len = max(len(correct), len(model_answer))
    for pos in range(max_len):
        expected_char = correct[pos] if pos < len(correct) else "<missing>"
        got_char = model_answer[pos] if pos < len(model_answer) else "<missing>"
        if expected_char != got_char:
            diff_parts.append(
                f"position {pos} (expected {expected_char}, got {got_char})"
            )

    diff_str = ", ".join(diff_parts) if diff_parts else "length mismatch"
    return (
        f"TEST FAILED\n"
        f"Input: {input_seq}\n"
        f"Expected: {correct}\n"
        f"Your output: {model_answer}\n"
        f"Diff: {diff_str}"
    )


def _feedback_reformatted_correction(
    input_seq: str, correct: str, model_answer: str
) -> str:
    """Reformatted correction: presents the correct answer as a training example."""
    if model_answer == correct:
        return f"Correct! Example: {input_seq} \u2192 {correct}"
    return f"Example: {input_seq} \u2192 {correct}"


# ---------------------------------------------------------------------------
# Clean-context prompt builder
# ---------------------------------------------------------------------------

def _build_clean_context_prompt(
    examples_text: str,
    vocab_line: str,
    n_initial_examples: int,
    accumulated_examples: List[Tuple[str, str]],
    input_seq: str,
) -> str:
    """Build a single prompt for the clean_context condition.

    Includes the original training examples, accumulated correct pairs
    formatted as additional examples, and the test question.
    """
    # Format accumulated correct pairs as additional examples
    extra_lines = []
    for i, (inp, out) in enumerate(accumulated_examples, n_initial_examples + 1):
        extra_lines.append(f"Example {i}: {inp} \u2192 {out}")
    extra_text = "\n".join(extra_lines)

    total_examples = n_initial_examples + len(accumulated_examples)
    prompt = (
        f"You are learning a symbolic transformation system.\n"
        f"The system uses these symbols: {vocab_line}\n"
        f"Here are {total_examples} examples:\n\n{examples_text}"
    )
    if extra_text:
        prompt += f"\n{extra_text}"
    prompt += (
        "\n\nStudy the pattern.\n"
        f"What is the output for: {input_seq}\n"
        'Respond with ONLY: {{"output": "YOUR_ANSWER"}}'
    )
    return prompt


# ---------------------------------------------------------------------------
# Test sequence generation
# ---------------------------------------------------------------------------

def _generate_test_sequence(
    sts: STSInstance,
    training_examples: List[TrainingExample],
    n_turns: int,
    seq_length: int,
    rng: random.Random,
) -> List[Tuple[str, str]]:
    """Generate a sequence of (input, correct_output) pairs for SB2 turns.

    Inputs are novel — not present in training_examples.
    """
    training_inputs = {ex.input_seq for ex in training_examples}
    sequence: List[Tuple[str, str]] = []
    attempts = 0
    max_attempts = n_turns * 30

    while len(sequence) < n_turns and attempts < max_attempts:
        attempts += 1
        inp = generate_input_sequence(rng, sts.alphabet, seq_length)
        if inp in training_inputs:
            continue
        # Avoid within-sequence duplicates
        if any(inp == prev_inp for prev_inp, _ in sequence):
            continue
        correct = solve(sts, inp)
        sequence.append((inp, correct))

    return sequence


# ---------------------------------------------------------------------------
# Core SB2 pilot runner
# ---------------------------------------------------------------------------

def run_sb2_pilot(
    client: Any,  # LLMClient (lesson.models.base.LLMClient)
    tier: int = 3,
    n_initial_examples: int = 4,
    n_instances: int = 3,
    n_turns: int = 6,
    conditions: List[str] = ["correction", "practice_only"],
    max_parallel: int = 1,
    completed_instances: Optional[Dict[str, Dict[int, list]]] = None,
) -> Dict[str, Any]:
    """Run minimal SB2 pilot: feedback condition comparisons on a few instances.

    For each (instance, condition):
    1. Present training examples + vocabulary as initial system context.
    2. For each turn, ask for the output via JSON mode, extract answer,
       then inject condition-specific feedback into the session history.
    3. Track accuracy per turn with raw response logging.

    Supported conditions:
        "correction"              — reveals correct answer on errors, confirms on correct
        "practice_only"           — always reveals the answer as a transition phrase
        "error_only"              — correct/incorrect signal only, no answer revealed
        "explanation"             — correction plus brief rule description
        "misleading"              — provides wrong answer on every third turn
        "no_feedback"             — static "Next question." regardless of correctness
        "clean_context"           — single-prompt with growing context of correct pairs only
        "prompted_correction"     — correction plus reflection prompt on errors
        "structured_correction"   — code-test-style pass/fail with per-position diff
        "reformatted_correction"  — correct answer presented as a training example

    Args:
        client:               An LLMClient instance (must implement .multi_turn()).
        tier:                 STS difficulty tier for all instances.
        n_initial_examples:   Number of training examples shown as context.
        n_instances:          Number of independent STS instances.
        n_turns:              Number of test turns per (instance, condition) cell.
        conditions:           List of feedback condition names to evaluate.
        max_parallel:         Maximum number of (instance, condition) cells to run
                              in parallel. Defaults to 1 (sequential).
        completed_instances:  Optional dict mapping condition → {instance_idx → list
                              of turn result dicts} for instances that were already
                              successfully evaluated and should not be re-run.
                              Valid instances are merged with newly-run data before
                              building the summary. Pass the output of
                              ``get_valid_instances()`` (one per condition).

    Returns:
        {
            "results": [flat per-turn result dicts with raw_response],
            "summary": {condition -> {turn_idx -> {"n_correct": int, "n_total": int}}},
            "model": client.name,
        }
    """
    all_results: List[Dict[str, Any]] = []
    model_name = getattr(client, "name", "unknown")
    log = InteractionLog(f"{model_name}_sb2")
    print_lock = threading.Lock()

    # Normalise completed_instances: condition → {inst_idx → [turn_dicts]}
    _completed: Dict[str, Dict[int, list]] = completed_instances or {}

    # Seed all_results with any already-completed turn data so summaries
    # correctly account for all instances (resumed + newly run).
    for cond_turns in _completed.values():
        for turn_list in cond_turns.values():
            all_results.extend(turn_list)

    # ------------------------------------------------------------------
    # Pre-generate all instances and build the list of cells
    # ------------------------------------------------------------------
    cells = []  # Each entry holds everything needed for one (instance, condition) run
    for inst_idx in range(n_instances):
        seed = tier * 10_000 + inst_idx
        rng = random.Random(seed + 5000)

        # Generate STS instance and training set
        sts = generate_sts_instance(tier=tier, seed=seed)
        full_training = generate_training_set(sts, n_max=max(n_initial_examples, 32))
        training_examples = subset_training_set(full_training, n_initial_examples)
        examples_text = format_training_examples(training_examples)
        alphabet = sts.alphabet
        vocab_line = " ".join(alphabet)

        # Generate a shared test sequence for this instance
        # (same inputs across conditions for comparability)
        test_sequence = _generate_test_sequence(
            sts=sts,
            training_examples=training_examples,
            n_turns=n_turns,
            seq_length=5,
            rng=rng,
        )

        if len(test_sequence) < n_turns:
            print(
                f"  WARNING: Only generated {len(test_sequence)}/{n_turns} test turns "
                f"for instance {inst_idx} (tier={tier}, seed={seed})"
            )

        print(f"\nInstance {inst_idx}/{n_instances} — sts_id={sts.id}")
        print(f"  {len(test_sequence)} test turns, {n_initial_examples} training examples")
        print(f"  Vocabulary: {vocab_line}")

        for condition in conditions:
            # Skip instances that are already complete
            if inst_idx in _completed.get(condition, {}):
                with print_lock:
                    print(
                        f"  [SKIP] instance {inst_idx}, condition={condition!r} "
                        f"— already complete (resumed)"
                    )
                continue
            cells.append({
                "inst_idx": inst_idx,
                "condition": condition,
                "sts": sts,
                "examples_text": examples_text,
                "alphabet": alphabet,
                "vocab_line": vocab_line,
                "test_sequence": test_sequence,
                "n_initial_examples": n_initial_examples,
                "tier": tier,
                "n_turns": n_turns,
                "seed": seed,
            })

    total_cells = len(cells)

    # ------------------------------------------------------------------
    # Helper: run a single (instance, condition) cell
    # ------------------------------------------------------------------
    def _run_cell(
        cell_num: int,
        total: int,
        cell: Dict[str, Any],
        log_lock: Optional[threading.Lock],
    ) -> List[Dict[str, Any]]:
        """Execute one (instance, condition) cell and return result dicts."""
        inst_idx = cell["inst_idx"]
        condition = cell["condition"]
        sts = cell["sts"]
        examples_text = cell["examples_text"]
        alphabet = cell["alphabet"]
        vocab_line = cell["vocab_line"]
        test_sequence = cell["test_sequence"]
        n_init = cell["n_initial_examples"]
        c_tier = cell["tier"]
        c_n_turns = cell["n_turns"]
        c_seed = cell["seed"]

        cell_results: List[Dict[str, Any]] = []

        # Each cell gets its own RNG seeded deterministically
        cell_rng = random.Random(c_seed + 5000 + hash(condition))

        with print_lock:
            print(f"  [{cell_num}/{total}] condition={condition!r}")

        # -----------------------------------------------------------
        # clean_context: single-prompt-with-growing-context code path
        # -----------------------------------------------------------
        if condition == "clean_context":
            accumulated_examples: List[Tuple[str, str]] = []

            for turn_idx, (inp, correct) in enumerate(test_sequence):
                prompt = _build_clean_context_prompt(
                    examples_text=examples_text,
                    vocab_line=vocab_line,
                    n_initial_examples=n_init,
                    accumulated_examples=accumulated_examples,
                    input_seq=inp,
                )

                try:
                    raw_response = client.prompt_json(prompt)
                except Exception as exc:
                    with print_lock:
                        print(f"    ERROR on turn {turn_idx}: {exc}")
                    raw_response = ""

                if not raw_response.strip():
                    try:
                        raw_response = client.prompt(prompt)
                    except Exception:
                        raw_response = ""

                model_answer = extract_answer(
                    raw_response,
                    mode="json",
                    vocabulary=alphabet,
                )

                expected = normalize_answer(correct)
                is_correct = model_answer == expected

                # Always accumulate the correct pair (regardless of model accuracy)
                accumulated_examples.append((inp, correct))

                result = {
                    "condition": condition,
                    "instance_idx": inst_idx,
                    "sts_id": sts.id,
                    "tier": c_tier,
                    "turn_idx": turn_idx,
                    "input_seq": inp,
                    "correct": is_correct,
                    "model_answer": model_answer,
                    "expected_answer": expected,
                    "feedback_given": "(clean_context — no feedback)",
                    "raw_response": raw_response,
                    "vocabulary": alphabet,
                }
                cell_results.append(result)

                if log_lock is not None:
                    with log_lock:
                        log.record(
                            prompt=prompt,
                            raw_response=raw_response,
                            extracted=model_answer,
                            expected=expected,
                            correct=is_correct,
                            metadata={
                                "condition": condition, "tier": c_tier,
                                "instance": inst_idx, "turn": turn_idx,
                                "input_seq": inp,
                                "n_accumulated": len(accumulated_examples),
                            },
                        )
                else:
                    log.record(
                        prompt=prompt,
                        raw_response=raw_response,
                        extracted=model_answer,
                        expected=expected,
                        correct=is_correct,
                        metadata={
                            "condition": condition, "tier": c_tier,
                            "instance": inst_idx, "turn": turn_idx,
                            "input_seq": inp,
                            "n_accumulated": len(accumulated_examples),
                        },
                    )

                status = "OK" if is_correct else f"WRONG (got {model_answer!r})"
                with print_lock:
                    print(f"    turn {turn_idx}: {inp} \u2192 {status}")

            return cell_results

        # -----------------------------------------------------------
        # Standard multi-turn code path
        # -----------------------------------------------------------

        # Determine misleading turns (every 3rd turn for "misleading" condition)
        mislead_turns: set = set()
        if condition == "misleading":
            mislead_turns = {i for i in range(c_n_turns) if (i + 1) % 3 == 0}

        session = client.multi_turn()
        session.reset()

        # System context: inject training examples + vocabulary
        context_msg = (
            f"You are learning a symbolic transformation system.\n"
            f"The system uses these symbols: {vocab_line}\n"
            f"Here are {n_init} examples:\n\n{examples_text}\n\n"
            "Study the pattern. You will be tested on it turn by turn.\n"
            'For each question, respond with ONLY a JSON object: {"output": "YOUR_ANSWER"}\n'
            "Use only the symbols listed above in your answer."
        )
        session.inject(context_msg, role="user")
        session.inject('{"output": "understood"}', role="assistant")

        for turn_idx, (inp, correct) in enumerate(test_sequence):
            # Ask the model for the output
            question = (
                f"What is the output for: {inp}\n"
                'Respond with ONLY: {{"output": "YOUR_ANSWER"}}'
            )

            try:
                raw_response = session.send_json(question, role="user")
            except Exception:
                # send_json already appended the user message to history
                # before the API call failed — remove it to avoid
                # double-injection when we fall back to plain send.
                if hasattr(session, '_messages') and session._messages:
                    session._messages.pop()
                raw_response = ""

            # If send_json returned empty (provider silently fails with JSON
            # schema), fall back to plain send
            if not raw_response.strip():
                # Remove empty assistant reply from history if send_json added one
                if hasattr(session, '_messages') and session._messages and \
                   session._messages[-1].get('role') == 'assistant' and \
                   not session._messages[-1].get('content', '').strip():
                    session._messages.pop()
                    # Also remove the user message that send_json added
                    if session._messages and session._messages[-1].get('role') == 'user':
                        session._messages.pop()
                try:
                    raw_response = session.send(question, role="user")
                except Exception as exc:
                    with print_lock:
                        print(f"    ERROR on turn {turn_idx}: {exc}")
                    raw_response = ""

            # GUARD: If we still got an empty response after all retries,
            # abort this instance to avoid recording bad data. Incomplete
            # instances are excluded by get_valid_instances() on resume.
            if not raw_response.strip():
                with print_lock:
                    print(
                        f"    ABORT instance {inst_idx}, condition={condition!r}, "
                        f"turn {turn_idx}: empty response after retries "
                        f"(API error / rate limit / budget exhaustion). "
                        f"Discarding {len(cell_results)} prior turns for this cell."
                    )
                return []  # Return nothing — don't pollute results

            # Tiered extraction: JSON → symbol-aware → regex
            model_answer = extract_answer(
                raw_response,
                mode="json",
                vocabulary=alphabet,
            )

            expected = normalize_answer(correct)
            is_correct = model_answer == expected

            # Generate and inject feedback
            feedback = _generate_feedback(
                condition=condition,
                sts=sts,
                input_seq=inp,
                correct=correct,
                model_answer=model_answer,
                rng=cell_rng,
                mislead_this_turn=(turn_idx in mislead_turns),
            )
            session.inject(feedback, role="user")

            result = {
                "condition": condition,
                "instance_idx": inst_idx,
                "sts_id": sts.id,
                "tier": c_tier,
                "turn_idx": turn_idx,
                "input_seq": inp,
                "correct": is_correct,
                "model_answer": model_answer,
                "expected_answer": expected,
                "feedback_given": feedback,
                "raw_response": raw_response,
                "vocabulary": alphabet,
            }
            cell_results.append(result)

            # Log the interaction for debugging
            if log_lock is not None:
                with log_lock:
                    log.record(
                        prompt=question,
                        raw_response=raw_response,
                        extracted=model_answer,
                        expected=expected,
                        correct=is_correct,
                        metadata={
                            "condition": condition, "tier": c_tier,
                            "instance": inst_idx, "turn": turn_idx,
                            "input_seq": inp, "feedback": feedback,
                        },
                    )
            else:
                log.record(
                    prompt=question,
                    raw_response=raw_response,
                    extracted=model_answer,
                    expected=expected,
                    correct=is_correct,
                    metadata={
                        "condition": condition, "tier": c_tier,
                        "instance": inst_idx, "turn": turn_idx,
                        "input_seq": inp, "feedback": feedback,
                    },
                )

            status = "OK" if is_correct else f"WRONG (got {model_answer!r})"
            with print_lock:
                print(f"    turn {turn_idx}: {inp} \u2192 {status}")

        return cell_results

    # ------------------------------------------------------------------
    # Execute cells: sequential or parallel
    # ------------------------------------------------------------------
    if max_parallel <= 1:
        for i, cell in enumerate(cells):
            results = _run_cell(i + 1, total_cells, cell, None)
            all_results.extend(results)
    else:
        log_lock = threading.Lock()
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as pool:
            futures = {}
            for i, cell in enumerate(cells):
                f = pool.submit(_run_cell, i + 1, total_cells, cell, log_lock)
                futures[f] = cell
            for f in concurrent.futures.as_completed(futures):
                results = f.result()
                all_results.extend(results)

    # Build summary: condition -> turn_idx -> {n_correct, n_total}
    summary: Dict[str, Dict[int, Dict[str, int]]] = {}
    for r in all_results:
        cond = r["condition"]
        t = r["turn_idx"]
        summary.setdefault(cond, {}).setdefault(t, {"n_correct": 0, "n_total": 0})
        summary[cond][t]["n_total"] += 1
        if r["correct"]:
            summary[cond][t]["n_correct"] += 1

    # Print per-condition learning curves
    print("\n--- SB2 Pilot Summary (accuracy per turn) ---")
    for cond in sorted(summary):
        print(f"  Condition: {cond}")
        for turn_idx in sorted(summary[cond]):
            counts = summary[cond][turn_idx]
            acc = counts["n_correct"] / counts["n_total"] if counts["n_total"] else 0.0
            print(
                f"    turn {turn_idx}: {counts['n_correct']}/{counts['n_total']} ({acc:.0%})"
            )

    log.close()

    return {
        "results": all_results,
        "summary": summary,
        "model": model_name,
        "log_file": str(log.filepath),
    }


def _generate_feedback(
    condition: str,
    sts: STSInstance,
    input_seq: str,
    correct: str,
    model_answer: str,
    rng: random.Random,
    mislead_this_turn: bool,
) -> str:
    """Dispatch to the appropriate feedback generator for the given condition."""
    if condition == "correction":
        return _feedback_correction(input_seq, correct, model_answer)
    elif condition == "practice_only":
        return _feedback_practice_only(input_seq, correct, model_answer)
    elif condition == "error_only":
        return _feedback_error_only(input_seq, correct, model_answer)
    elif condition == "explanation":
        return _feedback_explanation(sts, input_seq, correct, model_answer)
    elif condition == "misleading":
        return _feedback_misleading(
            sts, input_seq, correct, model_answer, rng, mislead_this_turn
        )
    elif condition == "no_feedback":
        return _feedback_no_feedback(input_seq, correct, model_answer)
    elif condition == "prompted_correction":
        return _feedback_prompted_correction(input_seq, correct, model_answer)
    elif condition == "structured_correction":
        return _feedback_structured_correction(input_seq, correct, model_answer)
    elif condition == "reformatted_correction":
        return _feedback_reformatted_correction(input_seq, correct, model_answer)
    else:
        raise ValueError(f"Unknown condition: {condition!r}")
