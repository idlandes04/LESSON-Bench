# %% [markdown]
# # LESSON-Bench SB1 - Smoke Test
#
# Reduced version: 3 instances, 2 N-values (4, 8), 5 test items each.
# Use this to verify connectivity and task structure before the full run.

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
from dataclasses import dataclass

import kaggle_benchmarks as kbench
from kaggle_benchmarks import task, chats

from lesson.sts.generator import generate_dataset, format_training_examples
from lesson.eval.extraction import normalize_answer

# %%
@dataclass
class STSAnswer:
    output: str

TIER = 2
N_VALUES = [4, 8]
SEQ_LENGTH = 5
N_TEST_ITEMS = (3, 1, 1)

# %%
@task(name="lesson_sb1_cell_quick", store_task=False, store_run=False)
def lesson_sb1_cell(llm, n_examples: int, instance_idx: int) -> dict:
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

        item_results.append({
            "item_idx": item_idx,
            "item_type": item.item_type.value,
            "correct": model_answer == expected,
            "model_answer": model_answer,
            "expected_answer": expected,
        })

    n_correct = sum(r["correct"] for r in item_results)
    n_total = len(item_results)

    return {
        "n_examples": n_examples,
        "instance_idx": instance_idx,
        "accuracy": n_correct / n_total if n_total > 0 else 0.0,
        "item_results": item_results,
    }

# %%
@task(
    name="LESSON-Bench SB1 Quick",
    description="Smoke test: 3 instances, 2 N-values (4, 8), 5 test items each.",
)
def lesson_bench_sb1_quick(llm) -> dict:
    all_results = {}
    for n in N_VALUES:
        n_results = []
        for inst_idx in range(3):
            with chats.new(f"quick_sb1_n{n}_{inst_idx}"):
                result = lesson_sb1_cell.run(llm, n, inst_idx).result
                n_results.append(result)
        all_results[n] = n_results

    def n_avg(n_val):
        results = all_results[n_val]
        return sum(r["accuracy"] for r in results) / len(results) if results else 0.0

    return {
        "n4_avg": round(n_avg(4), 4),
        "n8_avg": round(n_avg(8), 4),
        "model": llm.name,
    }

# %%
run = lesson_bench_sb1_quick.run(kbench.llms["google/gemini-3-flash-preview"])
result = run.result

print(f"SB1 Smoke test - {result['model']}")
print(f"  N=4: {result['n4_avg']:.1%}")
print(f"  N=8: {result['n8_avg']:.1%}")

# %%
run

# %%
# %choose LESSON-Bench SB1 Quick
