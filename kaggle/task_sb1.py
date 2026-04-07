# %% [markdown]
# # LESSON-Bench SB1: Learning Curves
#
# Measures how sample-efficiently LLMs induce symbolic transformation rules
# from varying numbers of in-context examples. Single-turn, no feedback.
# Sweeps N = 4, 8, 16, 32 training examples across 25 instances at Tier 2.

# %% [markdown]
# ## Setup

# %%
import sys
from pathlib import Path

kaggle_input = Path("/kaggle/input/datasets/isaaclandes/lesson-bench")
if kaggle_input.exists():
    link_path = Path("/kaggle/working/lesson")
    if not link_path.exists():
        link_path.symlink_to(kaggle_input)
    if "/kaggle/working" not in sys.path:
        sys.path.insert(0, "/kaggle/working")

# %%
from dataclasses import dataclass

import kaggle_benchmarks as kbench
from kaggle_benchmarks import task, chats

from lesson.sts.generator import generate_dataset, format_training_examples
from lesson.eval.extraction import normalize_answer

# %% [markdown]
# ## Constants and Schema

# %%
@dataclass
class STSAnswer:
    """Structured output schema for model responses."""
    output: str


TIER = 2
N_VALUES = [4, 8, 16, 32]
N_INSTANCES = 25
SEQ_LENGTH = 5
N_TEST_ITEMS = (3, 1, 1)  # 3 Regular, 1 Extrapolation, 1 Lure = 5 per instance

# %% [markdown]
# ## Task Definitions

# %%
@task(name="lesson_sb1_cell", store_task=False, store_run=False)
def lesson_sb1_cell(llm, n_examples: int, instance_idx: int) -> dict:
    """Single evaluation cell: one STS instance at one N value.

    Generates a dataset with n_examples training examples and 5 test items,
    then prompts the model once per test item (single-turn, no feedback).
    """
    seed = TIER * 10_000 + n_examples * 100 + instance_idx

    dataset = generate_dataset(
        tier=TIER,
        seed=seed,
        n_examples=n_examples,
        n_test_items=N_TEST_ITEMS,
        seq_length=SEQ_LENGTH,
    )

    examples_text = format_training_examples(dataset.training_examples)
    vocab_line = " ".join(dataset.sts.alphabet)

    item_results = []
    for item_idx, item in enumerate(dataset.test_items):
        prompt = (
            f"Below are {n_examples} examples of a symbolic transformation system.\n"
            f"The system uses these symbols: {vocab_line}\n"
            f"Study the pattern, then predict the output for the final input.\n"
            f"\n{examples_text}\n"
            f"\nInput: {item.input_seq}\n"
            f"Use only the symbols listed above in your answer."
        )

        response = llm.prompt(prompt, schema=STSAnswer)
        model_answer = normalize_answer(response.output)
        expected = normalize_answer(item.correct_output)
        is_correct = model_answer == expected

        item_results.append({
            "item_idx": item_idx,
            "item_type": item.item_type.value,
            "correct": is_correct,
            "model_answer": model_answer,
            "expected_answer": expected,
        })

    n_correct = sum(r["correct"] for r in item_results)
    n_total = len(item_results)

    # Per-type accuracy
    type_acc = {}
    for t in ["R", "E", "L"]:
        t_items = [r for r in item_results if r["item_type"] == t]
        if t_items:
            type_acc[t] = sum(r["correct"] for r in t_items) / len(t_items)

    return {
        "n_examples": n_examples,
        "instance_idx": instance_idx,
        "accuracy": n_correct / n_total if n_total > 0 else 0.0,
        "n_correct": n_correct,
        "n_total": n_total,
        "type_accuracy": type_acc,
        "item_results": item_results,
    }

# %%
@task(
    name="LESSON-Bench SB1",
    description=(
        "Learning curve benchmark. Measures how sample-efficiently models "
        "induce symbolic transformation rules from in-context examples. "
        "Sweeps N = 4, 8, 16, 32 training examples across 25 Tier-2 instances "
        "with 5 test items each (3 regular, 1 extrapolation, 1 lure). "
        "Single-turn, no feedback. 500 total prompts per model."
    ),
)
def lesson_bench_sb1(llm) -> dict:
    """Full SB1 benchmark: 4 N-values x 25 instances x 5 test items."""
    all_results = {}

    for n in N_VALUES:
        n_results = []
        for inst_idx in range(N_INSTANCES):
            with chats.new(f"sb1_n{n}_inst{inst_idx}"):
                cell_run = lesson_sb1_cell.run(llm, n, inst_idx)
                cell_result = cell_run.result
                if isinstance(cell_result, dict):
                    n_results.append(cell_result)
                else:
                    print(f"Warning: n={n} inst={inst_idx} failed: {cell_result}")
        all_results[n] = n_results

    # Compute per-N accuracy
    def n_avg(n_val):
        results = all_results[n_val]
        return sum(r["accuracy"] for r in results) / len(results) if results else 0.0

    # Compute per-N, per-type accuracy
    def n_type_avg(n_val, item_type):
        accs = [r["type_accuracy"].get(item_type, 0.0) for r in all_results[n_val]
                if item_type in r["type_accuracy"]]
        return sum(accs) / len(accs) if accs else 0.0

    # Learning curve slope (linear regression over N values)
    accs = [n_avg(n) for n in N_VALUES]
    log_ns = [len(N_VALUES) - i for i in range(len(N_VALUES))]  # rank order
    n_pts = len(N_VALUES)
    xm = sum(range(n_pts)) / n_pts
    ym = sum(accs) / n_pts
    num = sum((i - xm) * (accs[i] - ym) for i in range(n_pts))
    den = sum((i - xm) ** 2 for i in range(n_pts))
    slope = num / den if den > 0 else 0.0

    return {
        "n4_avg": round(n_avg(4), 4),
        "n8_avg": round(n_avg(8), 4),
        "n16_avg": round(n_avg(16), 4),
        "n32_avg": round(n_avg(32), 4),
        "n8_type_R": round(n_type_avg(8, "R"), 4),
        "n8_type_E": round(n_type_avg(8, "E"), 4),
        "n8_type_L": round(n_type_avg(8, "L"), 4),
        "learning_slope": round(slope, 4),
        "model": llm.name,
        "n_instances": N_INSTANCES,
        "n_values": N_VALUES,
        "tier": TIER,
    }

# %% [markdown]
# ## Run

# %%
run = lesson_bench_sb1.run(kbench.llm)
result = run.result
print("run:", run)
print("result type:", type(result))

if isinstance(result, dict):
    print(f"\n{'='*60}")
    print(f"LESSON-Bench SB1 - {result['model']}")
    print(f"{'='*60}")
    print(f"  N=4  avg:  {result['n4_avg']:.1%}")
    print(f"  N=8  avg:  {result['n8_avg']:.1%}")
    print(f"  N=16 avg:  {result['n16_avg']:.1%}")
    print(f"  N=32 avg:  {result['n32_avg']:.1%}")
    print(f"  ---")
    print(f"  N=8 Type R: {result['n8_type_R']:.1%}")
    print(f"  N=8 Type E: {result['n8_type_E']:.1%}")
    print(f"  N=8 Type L: {result['n8_type_L']:.1%}")
    print(f"  ---")
    print(f"  Learning slope: {result['learning_slope']:+.4f}")
    print(f"{'='*60}")
else:
    print("Task did not return a dict result. Check run object above.")

# %%
%choose LESSON-Bench SB1
