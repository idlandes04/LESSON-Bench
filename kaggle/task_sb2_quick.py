# %% [markdown]
# # LESSON-Bench SB2 - Smoke Test
#
# Reduced version: 3 instances, 6 turns, 2 conditions. Use this to verify
# connectivity and task structure before running the full benchmark.

# %%
import sys
from pathlib import Path

kaggle_input = Path("/kaggle/input/lesson-bench")
if kaggle_input.exists():
    link_path = Path("/kaggle/working/lesson")
    if not link_path.exists():
        link_path.symlink_to(kaggle_input)
    if str(link_path.parent) not in sys.path:
        sys.path.insert(0, str(link_path.parent))

# %%
import random
from dataclasses import dataclass

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

# %%
@dataclass
class STSAnswer:
    output: str

TIER = 2
N_EXAMPLES = 8
SEQ_LENGTH = 5

# %%
def _feedback_correction(input_seq, correct, model_answer):
    if model_answer == correct:
        return f"Correct! The output is {correct}."
    return f"Incorrect. The correct output is {correct}."


def _feedback_practice_only(input_seq, correct, model_answer):
    return f"The output for {input_seq} is {correct}. Next question."


FEEDBACK_FNS = {
    "correction": _feedback_correction,
    "practice_only": _feedback_practice_only,
}


def generate_test_sequence(sts, training_examples, n_turns, seq_length, rng):
    training_inputs = {ex.input_seq for ex in training_examples}
    sequence = []
    attempts = 0
    while len(sequence) < n_turns and attempts < n_turns * 30:
        attempts += 1
        inp = generate_input_sequence(rng, sts.alphabet, seq_length)
        if inp in training_inputs or any(inp == p for p, _ in sequence):
            continue
        correct = solve(sts, inp)
        sequence.append((inp, correct))
    return sequence

# %%
@task(name="lesson_sb2_cell_quick", store_task=False, store_run=False)
def lesson_sb2_cell(llm, instance_idx: int, condition: str) -> dict:
    seed = TIER * 10_000 + instance_idx
    rng = random.Random(seed + 5000)
    n_turns = 6

    sts = generate_sts_instance(tier=TIER, seed=seed)
    full_training = generate_training_set(sts, n_max=max(N_EXAMPLES, 32))
    training_sub = subset_training_set(full_training, N_EXAMPLES)
    examples_text = format_training_examples(training_sub)
    vocab_line = " ".join(sts.alphabet)
    test_seq = generate_test_sequence(sts, training_sub, n_turns, SEQ_LENGTH, rng)

    context = (
        f"You are learning a symbolic transformation system.\n"
        f"The system uses these symbols: {vocab_line}\n"
        f"Here are {N_EXAMPLES} examples:\n\n{examples_text}\n\n"
        f"Study the pattern. You will be tested on it turn by turn.\n"
        f"Use only the symbols listed above in your answer."
    )

    turn_results = []
    for turn_idx, (inp, correct) in enumerate(test_seq):
        question = f"What is the output for: {inp}"
        if turn_idx == 0:
            prompt = f"{context}\n\n{question}"
        else:
            prev_inp, prev_correct = test_seq[turn_idx - 1]
            prev_answer = turn_results[-1]["model_answer"]
            feedback = FEEDBACK_FNS[condition](prev_inp, prev_correct, prev_answer)
            prompt = f"{feedback}\n\n{question}"

        response = llm.prompt(prompt, schema=STSAnswer)
        model_answer = normalize_answer(response.output)
        expected = normalize_answer(correct)

        turn_results.append({
            "turn_idx": turn_idx,
            "correct": model_answer == expected,
            "model_answer": model_answer,
            "expected_answer": expected,
        })

    n_correct = sum(r["correct"] for r in turn_results)
    return {
        "instance_idx": instance_idx,
        "condition": condition,
        "accuracy": n_correct / len(turn_results) if turn_results else 0.0,
        "turn_results": turn_results,
    }

# %%
@task(
    name="LESSON-Bench SB2 Quick",
    description="Smoke test: 3 instances, 6 turns, 2 conditions.",
)
def lesson_bench_sb2_quick(llm) -> dict:
    all_results = {}
    for condition in ["correction", "practice_only"]:
        cond_results = []
        for inst_idx in range(3):
            with chats.new(f"quick_{condition}_{inst_idx}"):
                result = lesson_sb2_cell.run(llm, inst_idx, condition).result
                cond_results.append(result)
        all_results[condition] = cond_results

    corr = sum(r["accuracy"] for r in all_results["correction"]) / 3
    prac = sum(r["accuracy"] for r in all_results["practice_only"]) / 3

    return {
        "correction_avg": round(corr, 4),
        "practice_avg": round(prac, 4),
        "gap": round(corr - prac, 4),
        "model": llm.name,
    }

# %%
run = lesson_bench_sb2_quick.run(kbench.llm)
result = run.result

print(f"Smoke test - {result['model']}")
print(f"  Correction: {result['correction_avg']:.1%}")
print(f"  Practice:   {result['practice_avg']:.1%}")
print(f"  Gap:        {result['gap']:+.3f}")

# %%
run

# %%
# %choose LESSON-Bench SB2 Quick
