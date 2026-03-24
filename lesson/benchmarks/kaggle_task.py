"""LESSON-Bench SB2 — Kaggle Benchmarks SDK Task.

Evaluates whether LLMs can learn from corrective feedback within a session,
using procedurally generated symbolic transformation systems (STS).

2x2 factorial design separating two information channels:
- Answer visibility: does the model see the correct output?
- Evaluation signal: does the model know if it was right/wrong?

Four conditions:
- correction:     answer=YES, eval=YES  ("Incorrect. The correct output is X.")
- practice_only:  answer=YES, eval=NO   ("The output for Y is X. Next question.")
- error_only:     answer=NO,  eval=YES  ("Incorrect.")
- no_feedback:    answer=NO,  eval=NO   ("Next question.")

Key metrics:
- FLR: Feedback Learning Rate = late_correction - late_practice (near-zero = blind)
- Answer effect: avg(answer_visible) - avg(answer_hidden)
- Evaluation effect: avg(eval_present) - avg(eval_absent)
- Eval damage: error_only - no_feedback (negative = harmful)

Usage (Kaggle notebook):
    import kaggle_benchmarks as kbench
    from lesson.benchmarks.kaggle_task import lesson_bench_sb2

    # Single model:
    run = lesson_bench_sb2.run(kbench.llm)
    print(run.result)

    # Multi-model:
    lesson_bench_sb2.run(kbench.llms["google/gemini-3-flash-preview"])

Usage (local testing):
    # See notebooks/kaggle_benchmark.py for self-contained version
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import kaggle_benchmarks as kbench
from kaggle_benchmarks import task, chats

from lesson.sts.generator import (
    generate_sts_instance,
    generate_training_set,
    subset_training_set,
    generate_input_sequence,
    format_training_examples,
)
from lesson.sts.solver import solve
from lesson.eval.extraction import normalize_answer


# ---------------------------------------------------------------------------
# Schema — structured output from LLM
# ---------------------------------------------------------------------------

@dataclass
class STSAnswer:
    """Model's response to an STS question. The SDK parses this automatically."""
    output: str


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CORE_CONDITIONS = ["correction", "practice_only", "error_only", "no_feedback"]
DEFAULT_TIER = 2
DEFAULT_N_EXAMPLES = 8
DEFAULT_N_TURNS = 12
DEFAULT_N_INSTANCES = 25
DEFAULT_SEQ_LENGTH = 5


# ---------------------------------------------------------------------------
# Feedback generators (same logic as sb2_pilot.py)
# ---------------------------------------------------------------------------

def _feedback_correction(input_seq: str, correct: str, model_answer: str) -> str:
    if model_answer == correct:
        return f"Correct! The output is {correct}."
    return f"Incorrect. The correct output is {correct}."


def _feedback_practice_only(input_seq: str, correct: str, model_answer: str) -> str:
    return f"The output for {input_seq} is {correct}. Next question."


def _feedback_error_only(input_seq: str, correct: str, model_answer: str) -> str:
    if model_answer == correct:
        return "Correct!"
    return "Incorrect."


def _feedback_no_feedback(input_seq: str, correct: str, model_answer: str) -> str:
    return "Next question."


def _generate_feedback(
    condition: str, input_seq: str, correct: str, model_answer: str,
) -> str:
    if condition == "correction":
        return _feedback_correction(input_seq, correct, model_answer)
    elif condition == "practice_only":
        return _feedback_practice_only(input_seq, correct, model_answer)
    elif condition == "error_only":
        return _feedback_error_only(input_seq, correct, model_answer)
    elif condition == "no_feedback":
        return _feedback_no_feedback(input_seq, correct, model_answer)
    raise ValueError(f"Unknown condition: {condition!r}")


# ---------------------------------------------------------------------------
# Test sequence generation (same logic as sb2_pilot.py)
# ---------------------------------------------------------------------------

def _generate_test_sequence(
    sts: Any,
    training_examples: list,
    n_turns: int,
    seq_length: int,
    rng: random.Random,
) -> List[Tuple[str, str]]:
    """Generate (input, correct_output) pairs for test turns.

    Inputs are novel — not in training set, no duplicates.
    """
    training_inputs = {ex.input_seq for ex in training_examples}
    sequence: List[Tuple[str, str]] = []
    attempts = 0
    max_attempts = n_turns * 30

    while len(sequence) < n_turns and attempts < max_attempts:
        attempts += 1
        inp = generate_input_sequence(rng, sts.alphabet, seq_length)
        if inp in training_inputs or any(inp == prev for prev, _ in sequence):
            continue
        correct = solve(sts, inp)
        sequence.append((inp, correct))

    return sequence


# ---------------------------------------------------------------------------
# Cell-level task: one (instance, condition) evaluation
# ---------------------------------------------------------------------------

@task(name="lesson_sb2_cell", store_task=False, store_run=False)
def lesson_sb2_cell(
    llm,
    instance_idx: int,
    condition: str,
    tier: int = DEFAULT_TIER,
    n_examples: int = DEFAULT_N_EXAMPLES,
    n_turns: int = DEFAULT_N_TURNS,
) -> dict:
    """Evaluate one (instance, condition) cell of the SB2 protocol.

    Runs a multi-turn conversation where the model is shown training examples
    of a symbolic transformation system, then tested on novel inputs across
    multiple turns with condition-specific feedback between turns.

    Returns per-turn accuracy data.
    """
    seed = tier * 10_000 + instance_idx
    rng = random.Random(seed + 5000)

    # Generate STS instance and training/test data
    sts = generate_sts_instance(tier=tier, seed=seed)
    full_training = generate_training_set(sts, n_max=max(n_examples, 32))
    training_examples = subset_training_set(full_training, n_examples)
    examples_text = format_training_examples(training_examples)
    vocab_line = " ".join(sts.alphabet)

    test_sequence = _generate_test_sequence(
        sts, training_examples, n_turns, DEFAULT_SEQ_LENGTH, rng,
    )

    # Build system context
    context = (
        f"You are learning a symbolic transformation system.\n"
        f"The system uses these symbols: {vocab_line}\n"
        f"Here are {n_examples} examples:\n\n{examples_text}\n\n"
        f"Study the pattern. You will be tested on it turn by turn.\n"
        f"Use only the symbols listed above in your answer."
    )

    turn_results: List[Dict[str, Any]] = []

    for turn_idx, (inp, correct) in enumerate(test_sequence):
        question = f"What is the output for: {inp}"

        if turn_idx == 0:
            # First turn: include full context
            prompt = f"{context}\n\n{question}"
        else:
            # Subsequent turns: feedback about previous turn + next question
            prev_inp, prev_correct = test_sequence[turn_idx - 1]
            prev_answer = turn_results[-1]["model_answer"]
            feedback = _generate_feedback(
                condition, prev_inp, prev_correct, prev_answer,
            )
            prompt = f"{feedback}\n\n{question}"

        # Get structured response — SDK handles parsing
        response = llm.prompt(prompt, schema=STSAnswer)
        model_answer = normalize_answer(response.output)
        expected = normalize_answer(correct)
        is_correct = model_answer == expected

        turn_results.append({
            "turn_idx": turn_idx,
            "input_seq": inp,
            "correct": is_correct,
            "model_answer": model_answer,
            "expected_answer": expected,
        })

    n_correct = sum(r["correct"] for r in turn_results)
    n_total = len(turn_results)

    return {
        "instance_idx": instance_idx,
        "condition": condition,
        "sts_id": sts.id,
        "tier": tier,
        "accuracy": n_correct / n_total if n_total > 0 else 0.0,
        "n_correct": n_correct,
        "n_total": n_total,
        "turn_results": turn_results,
    }


# ---------------------------------------------------------------------------
# Main benchmark task
# ---------------------------------------------------------------------------

@task(name="LESSON-Bench SB2")
def lesson_bench_sb2(
    llm,
    n_instances: int = DEFAULT_N_INSTANCES,
    tier: int = DEFAULT_TIER,
    n_turns: int = DEFAULT_N_TURNS,
) -> dict:
    """Full LESSON-Bench SB2 evaluation — Feedback Learning Rate benchmark.

    Runs 4 feedback conditions x n_instances STS instances x n_turns turns.
    Each (instance, condition) cell is an isolated multi-turn conversation.

    Returns a dict with:
    - Per-condition average accuracy
    - FLR (Feedback Learning Rate)
    - Answer effect (2x2 factorial decomposition)
    - Evaluation effect (2x2 factorial decomposition)
    - Eval damage (error_only vs no_feedback)
    - Per-turn accuracy trajectories
    """
    all_results: Dict[str, List[dict]] = {}

    for condition in CORE_CONDITIONS:
        condition_results = []
        for inst_idx in range(n_instances):
            # Each cell gets an isolated conversation context
            with chats.new(f"{condition}_inst{inst_idx}"):
                result = lesson_sb2_cell.run(
                    llm, inst_idx, condition, tier, DEFAULT_N_EXAMPLES, n_turns,
                ).result
                condition_results.append(result)
        all_results[condition] = condition_results

    # -----------------------------------------------------------------------
    # Compute metrics
    # -----------------------------------------------------------------------

    def condition_avg(cond: str) -> float:
        results = all_results[cond]
        return sum(r["accuracy"] for r in results) / len(results) if results else 0.0

    def late_turn_avg(cond: str, start_turn: int = 6) -> float:
        """Average accuracy for turns >= start_turn (late-session performance)."""
        results = all_results[cond]
        late_correct = 0
        late_total = 0
        for r in results:
            for t in r["turn_results"]:
                if t["turn_idx"] >= start_turn:
                    late_total += 1
                    if t["correct"]:
                        late_correct += 1
        return late_correct / late_total if late_total > 0 else 0.0

    def per_turn_accuracy(cond: str) -> Dict[int, float]:
        """Accuracy at each turn index across all instances."""
        turn_correct: Dict[int, int] = {}
        turn_total: Dict[int, int] = {}
        for r in all_results[cond]:
            for t in r["turn_results"]:
                tidx = t["turn_idx"]
                turn_total[tidx] = turn_total.get(tidx, 0) + 1
                if t["correct"]:
                    turn_correct[tidx] = turn_correct.get(tidx, 0) + 1
        return {
            tidx: turn_correct.get(tidx, 0) / turn_total[tidx]
            for tidx in sorted(turn_total)
        }

    # Condition averages
    corr_avg = condition_avg("correction")
    prac_avg = condition_avg("practice_only")
    err_avg = condition_avg("error_only")
    nofb_avg = condition_avg("no_feedback")

    # Late-turn averages for FLR
    corr_late = late_turn_avg("correction")
    prac_late = late_turn_avg("practice_only")

    # Key metrics
    flr = corr_late - prac_late
    answer_effect = (corr_avg + prac_avg) / 2 - (err_avg + nofb_avg) / 2
    eval_effect = (corr_avg + err_avg) / 2 - (prac_avg + nofb_avg) / 2
    eval_damage = err_avg - nofb_avg

    # Per-turn trajectories
    trajectories = {cond: per_turn_accuracy(cond) for cond in CORE_CONDITIONS}

    # Learning slopes (linear regression over turns)
    def compute_slope(trajectory: Dict[int, float]) -> float:
        turns = sorted(trajectory.keys())
        if len(turns) < 2:
            return 0.0
        n = len(turns)
        x_mean = sum(turns) / n
        y_mean = sum(trajectory[t] for t in turns) / n
        num = sum((t - x_mean) * (trajectory[t] - y_mean) for t in turns)
        den = sum((t - x_mean) ** 2 for t in turns)
        return num / den if den > 0 else 0.0

    slopes = {cond: compute_slope(trajectories[cond]) for cond in CORE_CONDITIONS}

    return {
        # Per-condition averages
        "correction_avg": round(corr_avg, 4),
        "practice_avg": round(prac_avg, 4),
        "error_avg": round(err_avg, 4),
        "nofeedback_avg": round(nofb_avg, 4),
        # Key metrics
        "flr": round(flr, 4),
        "answer_effect": round(answer_effect, 4),
        "eval_effect": round(eval_effect, 4),
        "eval_damage": round(eval_damage, 4),
        # Late-turn averages
        "correction_late": round(corr_late, 4),
        "practice_late": round(prac_late, 4),
        # Slopes
        "correction_slope": round(slopes["correction"], 4),
        "practice_slope": round(slopes["practice_only"], 4),
        "error_slope": round(slopes["error_only"], 4),
        "nofeedback_slope": round(slopes["no_feedback"], 4),
        # Trajectories (per-turn accuracy)
        "trajectories": {
            cond: {str(k): round(v, 4) for k, v in traj.items()}
            for cond, traj in trajectories.items()
        },
        # Metadata
        "model": llm.name,
        "n_instances": n_instances,
        "n_turns": n_turns,
        "tier": tier,
    }


# ---------------------------------------------------------------------------
# Quick-run task for smoke testing (3 instances, 6 turns)
# ---------------------------------------------------------------------------

@task(name="LESSON-Bench SB2 Quick", store_task=False, store_run=False)
def lesson_bench_sb2_quick(llm) -> dict:
    """Quick smoke test — 3 instances, 6 turns, 2 conditions."""
    all_results: Dict[str, List[dict]] = {}

    for condition in ["correction", "practice_only"]:
        condition_results = []
        for inst_idx in range(3):
            with chats.new(f"quick_{condition}_{inst_idx}"):
                result = lesson_sb2_cell.run(
                    llm, inst_idx, condition, DEFAULT_TIER, DEFAULT_N_EXAMPLES, 6,
                ).result
                condition_results.append(result)
        all_results[condition] = condition_results

    corr = sum(r["accuracy"] for r in all_results["correction"]) / 3
    prac = sum(r["accuracy"] for r in all_results["practice_only"]) / 3

    return {
        "correction_avg": round(corr, 4),
        "practice_avg": round(prac, 4),
        "gap": round(corr - prac, 4),
        "model": llm.name,
    }
