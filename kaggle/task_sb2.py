# %% [markdown]
# # LESSON-Bench: Feedback Learning in Symbolic Operations
#
# Measures whether LLMs can learn from corrective feedback within a single
# session, or whether they only benefit from seeing additional correct examples.
#
# Uses procedurally generated symbolic transformation systems (STS) with a 2x2
# factorial design that separates two information channels in feedback:
# answer visibility (does the model see the correct output?) and evaluation
# signal (does the model know if it was right or wrong?).

# %% [markdown]
# ## Setup
#
# The `lesson` package is attached as a Kaggle dataset (`isaaclandes/lesson-bench`).
# Kaggle extracts dataset contents directly, so we symlink the mount point to
# create a proper Python package path.

# %%
import sys
from pathlib import Path

kaggle_input = Path("/kaggle/input/datasets/isaaclandes/lesson-bench")
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

# %% [markdown]
# ## Constants and Schema

# %%
@dataclass
class STSAnswer:
    """Structured output schema for model responses."""
    output: str


CORE_CONDITIONS = ["correction", "practice_only", "error_only", "no_feedback"]
TIER = 2
N_EXAMPLES = 8
N_TURNS = 12
N_INSTANCES = 25
SEQ_LENGTH = 5

# %% [markdown]
# ## Feedback Generators
#
# The 2x2 factorial design crosses two binary factors:
#
# |                    | Answer visible | Answer hidden |
# |--------------------|----------------|---------------|
# | **Eval present**   | correction     | error_only    |
# | **Eval absent**    | practice_only  | no_feedback   |

# %%
def _feedback_correction(input_seq, correct, model_answer):
    if model_answer == correct:
        return f"Correct! The output is {correct}."
    return f"Incorrect. The correct output is {correct}."


def _feedback_practice_only(input_seq, correct, model_answer):
    return f"The output for {input_seq} is {correct}. Next question."


def _feedback_error_only(input_seq, correct, model_answer):
    if model_answer == correct:
        return "Correct!"
    return "Incorrect."


def _feedback_no_feedback(input_seq, correct, model_answer):
    return "Next question."


FEEDBACK_FNS = {
    "correction": _feedback_correction,
    "practice_only": _feedback_practice_only,
    "error_only": _feedback_error_only,
    "no_feedback": _feedback_no_feedback,
}


def generate_feedback(condition, input_seq, correct, model_answer):
    return FEEDBACK_FNS[condition](input_seq, correct, model_answer)

# %% [markdown]
# ## Test Sequence Generation

# %%
def generate_test_sequence(sts, training_examples, n_turns, seq_length, rng):
    """Generate novel (input, correct_output) pairs not in the training set."""
    training_inputs = {ex.input_seq for ex in training_examples}
    sequence = []
    attempts = 0
    max_attempts = n_turns * 30

    while len(sequence) < n_turns and attempts < max_attempts:
        attempts += 1
        inp = generate_input_sequence(rng, sts.alphabet, seq_length)
        if inp in training_inputs or any(inp == p for p, _ in sequence):
            continue
        correct = solve(sts, inp)
        sequence.append((inp, correct))

    return sequence

# %% [markdown]
# ## Task Definitions

# %%
@task(name="lesson_sb2_cell", store_task=False, store_run=False)
def lesson_sb2_cell(llm, instance_idx: int, condition: str) -> dict:
    """Single evaluation cell: one STS instance under one feedback condition.

    Runs a 12-turn multi-turn conversation where the model is shown 8 training
    examples, then tested on 12 novel inputs with condition-specific feedback.
    """
    seed = TIER * 10_000 + instance_idx
    rng = random.Random(seed + 5000)

    sts = generate_sts_instance(tier=TIER, seed=seed)
    full_training = generate_training_set(sts, n_max=max(N_EXAMPLES, 32))
    training_sub = subset_training_set(full_training, N_EXAMPLES)
    examples_text = format_training_examples(training_sub)
    vocab_line = " ".join(sts.alphabet)

    test_seq = generate_test_sequence(sts, training_sub, N_TURNS, SEQ_LENGTH, rng)

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
            feedback = generate_feedback(condition, prev_inp, prev_correct, prev_answer)
            prompt = f"{feedback}\n\n{question}"

        response = llm.prompt(prompt, schema=STSAnswer)
        model_answer = normalize_answer(response.output)
        expected = normalize_answer(correct)
        is_correct = model_answer == expected

        turn_results.append({
            "turn_idx": turn_idx,
            "correct": is_correct,
            "model_answer": model_answer,
            "expected_answer": expected,
        })

    n_correct = sum(r["correct"] for r in turn_results)
    n_total = len(turn_results)

    return {
        "instance_idx": instance_idx,
        "condition": condition,
        "accuracy": n_correct / n_total if n_total > 0 else 0.0,
        "n_correct": n_correct,
        "n_total": n_total,
        "turn_results": turn_results,
    }

# %%
@task(
    name="lesson_bench_sb2",
    description=(
        "Feedback learning rate benchmark. Tests whether models learn "
        "differentially from corrective feedback vs. neutral answer exposure "
        "using procedurally generated symbolic transformation systems. "
        "2x2 factorial: 4 conditions x 25 instances x 12 turns. "
        "Leaderboard scalar is FLR (feedback learning rate)."
    ),
)
def lesson_bench_sb2(llm) -> float:
    """Full benchmark: 4 feedback conditions x 25 STS instances x 12 turns.

    Returns per-condition accuracy, FLR, factorial effects, and per-turn
    trajectories for the given model.
    """
    all_results = {}

    for condition in CORE_CONDITIONS:
        condition_results = []
        for inst_idx in range(N_INSTANCES):
            with chats.new(f"{condition}_inst{inst_idx}"):
                result = lesson_sb2_cell.run(
                    llm, inst_idx, condition,
                ).result
                condition_results.append(result)
        all_results[condition] = condition_results

    # Compute metrics

    def cond_avg(cond):
        results = all_results[cond]
        return sum(r["accuracy"] for r in results) / len(results) if results else 0.0

    def late_avg(cond, start=6):
        correct = total = 0
        for r in all_results[cond]:
            for t in r["turn_results"]:
                if t["turn_idx"] >= start:
                    total += 1
                    if t["correct"]:
                        correct += 1
        return correct / total if total > 0 else 0.0

    def per_turn(cond):
        tc, tt = {}, {}
        for r in all_results[cond]:
            for t in r["turn_results"]:
                i = t["turn_idx"]
                tt[i] = tt.get(i, 0) + 1
                if t["correct"]:
                    tc[i] = tc.get(i, 0) + 1
        return {i: tc.get(i, 0) / tt[i] for i in sorted(tt)}

    def slope(traj):
        turns = sorted(traj.keys())
        if len(turns) < 2:
            return 0.0
        n = len(turns)
        xm = sum(turns) / n
        ym = sum(traj[t] for t in turns) / n
        num = sum((t - xm) * (traj[t] - ym) for t in turns)
        den = sum((t - xm) ** 2 for t in turns)
        return num / den if den > 0 else 0.0

    ca, pa = cond_avg("correction"), cond_avg("practice_only")
    ea, na = cond_avg("error_only"), cond_avg("no_feedback")

    flr = late_avg("correction") - late_avg("practice_only")
    answer_effect = (ca + pa) / 2 - (ea + na) / 2
    eval_effect = (ca + ea) / 2 - (pa + na) / 2
    eval_damage = ea - na

    trajectories = {c: per_turn(c) for c in CORE_CONDITIONS}
    slopes = {c: slope(trajectories[c]) for c in CORE_CONDITIONS}

    result = {
        "correction_avg": round(ca, 4),
        "practice_avg": round(pa, 4),
        "error_avg": round(ea, 4),
        "nofeedback_avg": round(na, 4),
        "flr": round(flr, 4),
        "answer_effect": round(answer_effect, 4),
        "eval_effect": round(eval_effect, 4),
        "eval_damage": round(eval_damage, 4),
        "correction_late": round(late_avg("correction"), 4),
        "practice_late": round(late_avg("practice_only"), 4),
        "correction_slope": round(slopes["correction"], 4),
        "practice_slope": round(slopes["practice_only"], 4),
        "error_slope": round(slopes["error_only"], 4),
        "nofeedback_slope": round(slopes["no_feedback"], 4),
        "trajectories": {
            c: {str(k): round(v, 4) for k, v in t.items()}
            for c, t in trajectories.items()
        },
        "model": llm.name,
        "n_instances": N_INSTANCES,
        "n_turns": N_TURNS,
        "tier": TIER,
    }
    print(f"Full SB2 results: {result}")
    return result["flr"]

# %% [markdown]
# ## Run

# %%
run = lesson_bench_sb2.run(kbench.llm)
result = run.result
print(f"Leaderboard score (FLR): {result:+.4f}")

# %%
%choose lesson_bench_sb2
