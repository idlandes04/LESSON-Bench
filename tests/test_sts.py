"""Tests for the STS generator, solver, and item generation pipeline."""

import sys
sys.path.insert(0, "/Users/isaaclandes/Documents/SRC/Kaggle-2026-AGI")

from lesson.sts.types import RuleType, ItemType
from lesson.sts.solver import solve, get_partial_rule_answers
from lesson.sts.generator import (
    generate_sts_instance,
    generate_training_set,
    subset_training_set,
    generate_test_items,
    generate_dataset,
    format_training_examples,
)


def test_generate_instance_all_tiers():
    """Test that instances can be generated at all tiers."""
    for tier in range(1, 6):
        sts = generate_sts_instance(tier=tier, seed=42)
        assert sts.tier == tier
        assert len(sts.rules) > 0
        assert len(sts.alphabet) > 0
        print(f"  Tier {tier}: {len(sts.rules)} rules, "
              f"{len(sts.alphabet)} symbols, "
              f"{len(sts.exceptions)} exceptions, "
              f"has_non_direct={sts.has_non_direct_rule}")


def test_tier2_plus_has_non_direct():
    """Tiers 2+ must have at least one non-DIRECT rule."""
    for tier in range(2, 6):
        for seed in range(20):
            sts = generate_sts_instance(tier=tier, seed=seed)
            assert sts.has_non_direct_rule, \
                f"Tier {tier} seed {seed} has no non-DIRECT rule"
    print("  All tier 2+ instances have non-DIRECT rules (20 seeds each)")


def test_solver_deterministic():
    """Same input → same output, always."""
    sts = generate_sts_instance(tier=3, seed=42)
    training = generate_training_set(sts, n_max=8)

    for ex in training:
        out1 = solve(sts, ex.input_seq)
        out2 = solve(sts, ex.input_seq)
        assert out1 == out2, f"Non-deterministic: {ex.input_seq} → {out1} vs {out2}"
        assert out1 == ex.output_seq, f"Solver mismatch: {ex.input_seq} → {out1} vs {ex.output_seq}"

    print(f"  Solver deterministic on {len(training)} examples")


def test_nested_training_sets():
    """Verify N=2 ⊂ N=4 ⊂ N=8 ⊂ N=16 ⊂ N=32."""
    sts = generate_sts_instance(tier=3, seed=42)
    full = generate_training_set(sts, n_max=32)

    for n in [2, 4, 8, 16]:
        subset = subset_training_set(full, n)
        superset = subset_training_set(full, n * 2)
        subset_inputs = [ex.input_seq for ex in subset]
        superset_inputs = [ex.input_seq for ex in superset]
        for inp in subset_inputs:
            assert inp in superset_inputs, \
                f"N={n} example {inp} not in N={n*2} superset"

    print("  Nested training sets verified: N=2 ⊂ N=4 ⊂ N=8 ⊂ N=16 ⊂ N=32")


def test_test_item_generation():
    """Test that test items are generated with correct types."""
    sts = generate_sts_instance(tier=3, seed=42)
    training = generate_training_set(sts, n_max=8)
    subset = subset_training_set(training, 8)

    items = generate_test_items(sts, subset, n_per_type=(3, 1, 1))
    assert len(items) == 5, f"Expected 5 items, got {len(items)}"

    # Verify all items have valid outputs
    for item in items:
        verified = solve(sts, item.input_seq)
        assert verified == item.correct_output, \
            f"Item {item.input_seq}: solver says {verified}, item says {item.correct_output}"

    type_counts = {}
    for item in items:
        type_counts[item.item_type] = type_counts.get(item.item_type, 0) + 1

    print(f"  Generated {len(items)} test items: {type_counts}")


def test_type_l_partial_answers():
    """Type L items must have partial-rule answers that differ from correct."""
    found_type_l = False
    for seed in range(50):
        sts = generate_sts_instance(tier=3, seed=seed)
        training = generate_training_set(sts, n_max=8)
        subset = subset_training_set(training, 8)
        items = generate_test_items(sts, subset, n_per_type=(3, 1, 1))

        for item in items:
            if item.item_type == ItemType.L:
                found_type_l = True
                assert item.partial_rule_answer is not None
                assert item.partial_rule_answer != item.correct_output
                break
        if found_type_l:
            break

    if found_type_l:
        print("  Type L items have divergent partial-rule answers")
    else:
        print("  WARNING: No Type L items generated in 50 seeds (may need more complex tiers)")


def test_no_training_test_overlap():
    """Test inputs must not appear in training set."""
    for seed in range(10):
        sts = generate_sts_instance(tier=3, seed=seed)
        training = generate_training_set(sts, n_max=8)
        subset = subset_training_set(training, 8)
        items = generate_test_items(sts, subset, n_per_type=(3, 1, 1))

        training_inputs = {ex.input_seq for ex in subset}
        for item in items:
            assert item.input_seq not in training_inputs, \
                f"Test item {item.input_seq} overlaps with training set"

    print("  No training/test overlap across 10 instances")


def test_full_dataset():
    """Test the full dataset generation pipeline."""
    dataset = generate_dataset(tier=3, seed=42, n_examples=8)
    assert dataset.n_examples == 8
    assert len(dataset.training_examples) == 8
    assert len(dataset.test_items) == 5
    assert dataset.sts.tier == 3

    # Format check
    formatted = format_training_examples(dataset.training_examples)
    assert "Example 1:" in formatted
    assert "→" in formatted

    print(f"  Full dataset: {dataset.n_examples} training, "
          f"{len(dataset.test_items)} test items")
    print(f"  Formatted prompt preview:\n{formatted[:200]}...")


def test_reproducibility():
    """Same seed → identical dataset."""
    d1 = generate_dataset(tier=3, seed=42, n_examples=8)
    d2 = generate_dataset(tier=3, seed=42, n_examples=8)

    assert d1.sts.id == d2.sts.id
    assert len(d1.training_examples) == len(d2.training_examples)
    for e1, e2 in zip(d1.training_examples, d2.training_examples):
        assert e1.input_seq == e2.input_seq
        assert e1.output_seq == e2.output_seq

    for t1, t2 in zip(d1.test_items, d2.test_items):
        assert t1.input_seq == t2.input_seq
        assert t1.correct_output == t2.correct_output
        assert t1.item_type == t2.item_type

    print("  Reproducibility verified: identical datasets from same seed")


def test_type_e_feasibility_rate():
    """Check Type E feasibility across many instances (target: >90% at tiers 2+)."""
    for tier in [2, 3, 4]:
        feasible = 0
        total = 50
        for seed in range(total):
            sts = generate_sts_instance(tier=tier, seed=seed)
            training = generate_training_set(sts, n_max=8)
            subset = subset_training_set(training, 8)
            items = generate_test_items(sts, subset, n_per_type=(0, 1, 0))
            if any(item.item_type == ItemType.E for item in items):
                feasible += 1

        rate = feasible / total
        print(f"  Tier {tier} Type E feasibility: {feasible}/{total} = {rate:.0%}")


if __name__ == "__main__":
    tests = [
        ("Instance generation (all tiers)", test_generate_instance_all_tiers),
        ("Tier 2+ non-DIRECT constraint", test_tier2_plus_has_non_direct),
        ("Solver determinism", test_solver_deterministic),
        ("Nested training sets", test_nested_training_sets),
        ("Test item generation", test_test_item_generation),
        ("Type L partial answers", test_type_l_partial_answers),
        ("No training/test overlap", test_no_training_test_overlap),
        ("Full dataset pipeline", test_full_dataset),
        ("Reproducibility", test_reproducibility),
        ("Type E feasibility rate", test_type_e_feasibility_rate),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        try:
            print(f"\n[TEST] {name}")
            test_fn()
            print(f"  PASSED")
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
