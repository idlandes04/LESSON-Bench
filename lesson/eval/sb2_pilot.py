from __future__ import annotations

"""SB2 pilot evaluation runner.

Tests feedback conditions (correction, practice_only, error_only, explanation,
misleading) in a multi-turn setting to measure how different feedback types
affect within-session learning.

v9.0+ improvements:
- Structured JSON output via send_json() as primary extraction
- Symbol vocabulary included in system context
- Raw responses logged for debugging
- Answer normalization for robust comparison
- Tiered fallback: JSON → symbol-aware → regex
"""

import random
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
) -> Dict[str, Any]:
    """Run minimal SB2 pilot: feedback condition comparisons on a few instances.

    For each (instance, condition):
    1. Present training examples + vocabulary as initial system context.
    2. For each turn, ask for the output via JSON mode, extract answer,
       then inject condition-specific feedback into the session history.
    3. Track accuracy per turn with raw response logging.

    Supported conditions:
        "correction"    — reveals correct answer on errors, confirms on correct
        "practice_only" — always reveals the answer as a transition phrase
        "error_only"    — correct/incorrect signal only, no answer revealed
        "explanation"   — correction plus brief rule description
        "misleading"    — provides wrong answer on every third turn

    Args:
        client:              An LLMClient instance (must implement .multi_turn()).
        tier:                STS difficulty tier for all instances.
        n_initial_examples:  Number of training examples shown as context.
        n_instances:         Number of independent STS instances.
        n_turns:             Number of test turns per (instance, condition) cell.
        conditions:          List of feedback condition names to evaluate.

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

    total_cells = n_instances * len(conditions)
    cell_idx = 0

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
            cell_idx += 1
            print(f"  [{cell_idx}/{total_cells}] condition={condition!r}")

            # Determine misleading turns (every 3rd turn for "misleading" condition)
            mislead_turns = set()  # type: set
            if condition == "misleading":
                mislead_turns = {i for i in range(n_turns) if (i + 1) % 3 == 0}

            session = client.multi_turn()
            session.reset()

            # System context: inject training examples + vocabulary
            context_msg = (
                f"You are learning a symbolic transformation system.\n"
                f"The system uses these symbols: {vocab_line}\n"
                f"Here are {n_initial_examples} examples:\n\n{examples_text}\n\n"
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
                    # Fallback to plain send
                    try:
                        raw_response = session.send(question, role="user")
                    except Exception as exc:
                        print(f"    ERROR on turn {turn_idx}: {exc}")
                        raw_response = ""

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
                    rng=rng,
                    mislead_this_turn=(turn_idx in mislead_turns),
                )
                session.inject(feedback, role="user")

                result = {
                    "condition": condition,
                    "instance_idx": inst_idx,
                    "sts_id": sts.id,
                    "tier": tier,
                    "turn_idx": turn_idx,
                    "input_seq": inp,
                    "correct": is_correct,
                    "model_answer": model_answer,
                    "expected_answer": expected,
                    "feedback_given": feedback,
                    "raw_response": raw_response,
                    "vocabulary": alphabet,
                }
                all_results.append(result)

                # Log the interaction for debugging
                log.record(
                    prompt=question,
                    raw_response=raw_response,
                    extracted=model_answer,
                    expected=expected,
                    correct=is_correct,
                    metadata={
                        "condition": condition, "tier": tier,
                        "instance": inst_idx, "turn": turn_idx,
                        "input_seq": inp, "feedback": feedback,
                    },
                )

                status = "OK" if is_correct else f"WRONG (got {model_answer!r})"
                print(f"    turn {turn_idx}: {inp} → {status}")

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
    else:
        raise ValueError(f"Unknown condition: {condition!r}")
