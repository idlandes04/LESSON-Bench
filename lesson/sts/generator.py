"""STS instance generator.

Procedurally generates Symbolic Transformation System instances at specified
difficulty tiers, with contamination defenses and feasibility constraints.
"""

from __future__ import annotations

import hashlib
import random
from itertools import product as iter_product
from typing import Dict, List, Optional, Tuple

from .types import (
    Rule, RuleType, Exception_, STSInstance, TestItem, ItemType,
    TrainingExample, STSDataset,
)
from .solver import solve, get_partial_rule_answers
from .symbols import DEFAULT_SYMBOLS


# --- Tier specifications ---
# tier -> (n_rules, rule_types_allowed, exception_pct)
TIER_SPECS = {
    1: {"n_rules": 2, "types": [RuleType.DIRECT], "exception_pct": 0.0},
    2: {"n_rules": 3, "types": [RuleType.DIRECT, RuleType.POSITIONAL], "exception_pct": 0.0},
    3: {"n_rules": 4, "types": [RuleType.DIRECT, RuleType.POSITIONAL, RuleType.CONDITIONAL], "exception_pct": 0.05},
    4: {"n_rules": 5, "types": [RuleType.DIRECT, RuleType.POSITIONAL, RuleType.CONDITIONAL, RuleType.COMPOSITIONAL], "exception_pct": 0.10},
    5: {"n_rules": 7, "types": [RuleType.DIRECT, RuleType.POSITIONAL, RuleType.CONDITIONAL, RuleType.COMPOSITIONAL], "exception_pct": 0.20},
}


def _make_direct_rule(rng: random.Random, alphabet: list[str]) -> Rule:
    """Generate a DIRECT (substring replacement) rule.

    Prefers 2-symbol patterns to avoid overly destructive single-symbol replacements
    that collapse outputs to a single symbol.
    """
    # Bias toward 2-symbol patterns (less destructive)
    pattern_len = rng.choices([1, 2], weights=[1, 3])[0]
    pattern_syms = [rng.choice(alphabet) for _ in range(pattern_len)]
    pattern = "".join(pattern_syms)

    # Replacement: 1 symbol, must differ from pattern
    replacement = rng.choice(alphabet)
    while replacement == pattern and len(alphabet) > 1:
        replacement = rng.choice(alphabet)

    return Rule(
        rule_type=RuleType.DIRECT,
        spec={"pattern": pattern, "replacement": replacement},
    )


def _make_positional_rule(rng: random.Random, alphabet: list[str], seq_len: int) -> Rule:
    """Generate a POSITIONAL (position-dependent) rule.

    Maps 3-5 symbols so with N=8 training examples on a larger alphabet,
    some mapped symbols are likely unseen at the critical position — enabling Type E items.
    """
    position = rng.randint(0, seq_len - 1)

    # Map 3-5 symbols (with larger alphabets, harder to see all in N=8 examples)
    n_mapped = min(rng.randint(3, 5), len(alphabet) - 1)
    source_syms = rng.sample(alphabet, n_mapped)
    target_syms = []
    for s in source_syms:
        t = rng.choice(alphabet)
        while t == s and len(alphabet) > 1:
            t = rng.choice(alphabet)
        target_syms.append(t)

    mapping = dict(zip(source_syms, target_syms))

    return Rule(
        rule_type=RuleType.POSITIONAL,
        spec={"position": position, "mapping": mapping},
    )


def _make_conditional_rule(rng: random.Random, alphabet: list[str]) -> Rule:
    """Generate a CONDITIONAL (context-dependent) rule."""
    trigger = rng.choice(alphabet)

    # if_present branch: a DIRECT replacement
    pat1 = rng.choice(alphabet)
    rep1 = rng.choice(alphabet)
    while rep1 == pat1 and len(alphabet) > 1:
        rep1 = rng.choice(alphabet)

    # if_absent branch: a different DIRECT replacement
    pat2 = rng.choice(alphabet)
    rep2 = rng.choice(alphabet)
    while rep2 == pat2 and len(alphabet) > 1:
        rep2 = rng.choice(alphabet)

    return Rule(
        rule_type=RuleType.CONDITIONAL,
        spec={
            "trigger": trigger,
            "if_present": {
                "type": RuleType.DIRECT,
                "spec": {"pattern": pat1, "replacement": rep1},
            },
            "if_absent": {
                "type": RuleType.DIRECT,
                "spec": {"pattern": pat2, "replacement": rep2},
            },
        },
    )


def _make_compositional_rule(rng: random.Random, alphabet: list[str], seq_len: int) -> Rule:
    """Generate a COMPOSITIONAL (chained) rule."""
    # Inner: a DIRECT rule
    inner = _make_direct_rule(rng, alphabet)
    # Outer: a POSITIONAL rule (different operation type for max diagnostic value)
    outer = _make_positional_rule(rng, alphabet, seq_len)

    return Rule(
        rule_type=RuleType.COMPOSITIONAL,
        spec={
            "inner": {"type": inner.rule_type, "spec": inner.spec},
            "outer": {"type": outer.rule_type, "spec": outer.spec},
        },
    )


def _generate_rules(rng: random.Random, tier: int, alphabet: list[str], seq_len: int) -> list[Rule]:
    """Generate rules for a given tier.

    Guarantees:
    - Tiers 2+: at least one non-DIRECT rule
    - Tiers 3+: always includes a POSITIONAL rule (most reliable for Type E generation)
    """
    spec = TIER_SPECS[tier]
    n_rules = spec["n_rules"]
    allowed_types = spec["types"]

    makers = {
        RuleType.DIRECT: lambda: _make_direct_rule(rng, alphabet),
        RuleType.POSITIONAL: lambda: _make_positional_rule(rng, alphabet, seq_len),
        RuleType.CONDITIONAL: lambda: _make_conditional_rule(rng, alphabet),
        RuleType.COMPOSITIONAL: lambda: _make_compositional_rule(rng, alphabet, seq_len),
    }

    rules = []

    # For tiers 3+, always include a POSITIONAL rule (most reliable for Type E)
    if tier >= 3 and RuleType.POSITIONAL in allowed_types:
        rules.append(makers[RuleType.POSITIONAL]())

    # For tiers 2+, guarantee at least one non-DIRECT rule
    non_direct_types = [t for t in allowed_types if t != RuleType.DIRECT]
    if non_direct_types and not rules:
        forced_type = rng.choice(non_direct_types)
        rules.append(makers[forced_type]())

    # Fill remaining slots
    while len(rules) < n_rules:
        rule_type = rng.choice(allowed_types)
        rules.append(makers[rule_type]())

    rng.shuffle(rules)
    return rules


def _generate_exceptions(
    rng: random.Random,
    sts_rules: list[Rule],
    alphabet: list[str],
    seq_len: int,
    exception_pct: float,
) -> list[Exception_]:
    """Generate exceptions (alternate rules for specific inputs)."""
    if exception_pct <= 0:
        return []

    # Number of exceptions based on expected input space
    # For a typical evaluation: ~1-3 exceptions
    n_exceptions = max(1, int(exception_pct * 20))  # ~20 possible triggers

    exceptions = []
    for _ in range(n_exceptions):
        trigger = rng.choice(alphabet)
        position = rng.choice([None, rng.randint(0, seq_len - 1)])
        # Override: random output of same length
        override_len = rng.randint(1, seq_len)
        override = "".join(rng.choice(alphabet) for _ in range(override_len))
        exceptions.append(Exception_(trigger=trigger, position=position, output_override=override))

    return exceptions


def generate_input_sequence(rng: random.Random, alphabet: list[str], length: int) -> str:
    """Generate a random input sequence from the alphabet."""
    return "".join(rng.choice(alphabet) for _ in range(length))


# Default alphabet sizes per tier — larger alphabets at higher tiers
# improve Type E feasibility (more symbols = harder to see all in N=8 examples)
DEFAULT_ALPHABET_SIZES = {1: 5, 2: 8, 3: 8, 4: 10, 5: 12}


def generate_sts_instance(
    tier: int,
    seed: int,
    alphabet: list[str] | None = None,
    alphabet_size: int | None = None,
    seq_length: int = 5,
    instance_id: str | None = None,
) -> STSInstance:
    """Generate a single STS instance at the specified tier.

    Args:
        tier: Difficulty tier (1-5)
        seed: Random seed for reproducibility
        alphabet: Specific symbols to use (overrides alphabet_size)
        alphabet_size: Number of symbols to use (defaults to tier-appropriate size)
        seq_length: Length of input/output sequences
        instance_id: Optional custom ID

    Returns:
        A fully specified STSInstance
    """
    if tier not in TIER_SPECS:
        raise ValueError(f"Tier must be 1-5, got {tier}")

    rng = random.Random(seed)

    if alphabet is None:
        if alphabet_size is None:
            alphabet_size = DEFAULT_ALPHABET_SIZES[tier]
        n = min(alphabet_size, len(DEFAULT_SYMBOLS))
        alphabet = DEFAULT_SYMBOLS[:n]
        # Shuffle with seed for variety across instances
        alpha_copy = list(alphabet)
        rng.shuffle(alpha_copy)
        alphabet = alpha_copy

    spec = TIER_SPECS[tier]
    rules = _generate_rules(rng, tier, alphabet, seq_length)
    exceptions = _generate_exceptions(rng, rules, alphabet, seq_length, spec["exception_pct"])

    if instance_id is None:
        instance_id = f"sts_t{tier}_s{seed}"

    return STSInstance(
        id=instance_id,
        alphabet=alphabet,
        rules=rules,
        exceptions=exceptions,
        tier=tier,
        seed=seed,
    )


def _generate_all_inputs(alphabet: list[str], seq_length: int) -> list[str]:
    """Generate all possible input sequences (for small alphabets/lengths)."""
    return ["".join(combo) for combo in iter_product(alphabet, repeat=seq_length)]


def _make_rule_exercising_input(
    rule: Rule, alphabet: list[str], seq_length: int, rng: random.Random
) -> Optional[str]:
    """Generate an input that is likely to exercise a specific rule.

    For DIRECT rules: ensure the pattern appears in the input.
    For POSITIONAL: ensure a mapped symbol appears at the critical position.
    For CONDITIONAL: ensure the trigger is present.
    For COMPOSITIONAL: exercise the inner rule.
    """
    if rule.rule_type == RuleType.DIRECT:
        pattern = rule.spec["pattern"]
        if len(pattern) <= seq_length:
            # Place pattern at a random position, fill rest randomly
            pos = rng.randint(0, seq_length - len(pattern))
            chars = [rng.choice(alphabet) for _ in range(seq_length)]
            for i, ch in enumerate(pattern):
                chars[pos + i] = ch
            return "".join(chars)

    elif rule.rule_type == RuleType.POSITIONAL:
        pos = rule.spec["position"]
        mapping = rule.spec["mapping"]
        if mapping and pos < seq_length:
            chars = [rng.choice(alphabet) for _ in range(seq_length)]
            chars[pos] = rng.choice(list(mapping.keys()))
            return "".join(chars)

    elif rule.rule_type == RuleType.CONDITIONAL:
        trigger = rule.spec["trigger"]
        chars = [rng.choice(alphabet) for _ in range(seq_length)]
        chars[rng.randint(0, seq_length - 1)] = trigger
        return "".join(chars)

    elif rule.rule_type == RuleType.COMPOSITIONAL:
        inner_spec = rule.spec["inner"]
        inner_rule = Rule(rule_type=inner_spec["type"], spec=inner_spec["spec"])
        return _make_rule_exercising_input(inner_rule, alphabet, seq_length, rng)

    return None


def generate_training_set(
    sts: STSInstance,
    n_max: int = 32,
    seq_length: int = 5,
    seed: int | None = None,
) -> list[TrainingExample]:
    """Generate a nested training set where N=2 ⊂ N=4 ⊂ N=8 ⊂ N=16 ⊂ N=32.

    The N=n_max set is a superset containing all smaller sets.
    Deterministic subsampling via seed.

    The first examples are *informative*: they exercise each rule at least once,
    ensuring the model sees actual transformations (not just identity).
    """
    if seed is None:
        seed = hash(sts.id) & 0xFFFFFFFF

    rng = random.Random(seed + 1000)

    seen = set()
    informative = []

    # Phase 1: Generate at least one example exercising each rule
    for rule in sts.rules:
        for _ in range(20):
            inp = _make_rule_exercising_input(rule, sts.alphabet, seq_length, rng)
            if inp and inp not in seen:
                seen.add(inp)
                output = solve(sts, inp)
                # Only keep if output differs from input (rule actually fired)
                if output != inp:
                    informative.append(TrainingExample(input_seq=inp, output_seq=output))
                    break

    # Phase 2: Fill remaining with random examples
    random_examples = []
    max_attempts = n_max * 10
    for _ in range(max_attempts):
        if len(informative) + len(random_examples) >= n_max:
            break
        inp = generate_input_sequence(rng, sts.alphabet, seq_length)
        if inp not in seen:
            seen.add(inp)
            output = solve(sts, inp)
            random_examples.append(TrainingExample(input_seq=inp, output_seq=output))

    # Interleave: put informative examples first (they'll be in every nested subset),
    # then random examples
    rng.shuffle(random_examples)
    examples = informative + random_examples

    return examples[:n_max]


def subset_training_set(full_set: list[TrainingExample], n: int) -> list[TrainingExample]:
    """Get a nested subset: first n examples from the full training set.

    Because the full set is generated deterministically, taking the first n
    elements gives us N=2 ⊂ N=4 ⊂ ... ⊂ N=32.
    """
    return full_set[:n]


# --- Test item generation ---

def _generate_type_r_items(
    sts: STSInstance,
    training_examples: list[TrainingExample],
    n_items: int,
    seq_length: int,
    rng: random.Random,
) -> list[TestItem]:
    """Generate Type R (rule-consistent) test items.

    These are standard items where any reasonable approach gives the correct answer.
    Inputs are novel (not in training set) but rule application is straightforward.
    """
    training_inputs = {ex.input_seq for ex in training_examples}
    items = []
    attempts = 0

    while len(items) < n_items and attempts < n_items * 20:
        attempts += 1
        inp = generate_input_sequence(rng, sts.alphabet, seq_length)
        if inp in training_inputs:
            continue

        correct = solve(sts, inp)
        items.append(TestItem(
            input_seq=inp,
            correct_output=correct,
            item_type=ItemType.R,
            partial_rule_answer=None,
            sts_id=sts.id,
        ))
        training_inputs.add(inp)  # Prevent duplicates within test set too

    return items


def _generate_type_e_items(
    sts: STSInstance,
    training_examples: list[TrainingExample],
    n_items: int,
    seq_length: int,
    rng: random.Random,
) -> list[TestItem]:
    """Generate Type E (extrapolation) test items.

    Items requiring generalization beyond surface similarity to training examples.
    For each non-DIRECT rule, find inputs where the rule acts on symbol values
    NOT seen in training at the rule-relevant positions.
    """
    if not sts.has_non_direct_rule:
        return []  # Type E only meaningful at tiers 2+

    training_inputs = {ex.input_seq for ex in training_examples}
    items = []

    for rule in sts.rules:
        if rule.rule_type == RuleType.DIRECT:
            continue

        if rule.rule_type == RuleType.POSITIONAL:
            pos = rule.spec["position"]
            mapping = rule.spec["mapping"]
            # Find symbols at this position in training data
            seen_at_pos = {
                ex.input_seq[pos]
                for ex in training_examples
                if pos < len(ex.input_seq)
            }
            # Find mapped symbols NOT seen at this position
            unseen = [s for s in mapping.keys() if s not in seen_at_pos]
            for sym in unseen:
                if len(items) >= n_items:
                    break
                # Construct input with unseen symbol at the critical position
                inp_chars = [rng.choice(sts.alphabet) for _ in range(seq_length)]
                inp_chars[pos] = sym
                inp = "".join(inp_chars)
                if inp in training_inputs:
                    continue
                correct = solve(sts, inp)
                items.append(TestItem(
                    input_seq=inp,
                    correct_output=correct,
                    item_type=ItemType.E,
                    partial_rule_answer=None,
                    sts_id=sts.id,
                    metadata={"rule_type": "positional", "position": pos, "unseen_symbol": sym},
                ))
                training_inputs.add(inp)

        elif rule.rule_type == RuleType.CONDITIONAL:
            trigger = rule.spec["trigger"]
            # Check which branches are exercised in training
            has_trigger = any(trigger in ex.input_seq for ex in training_examples)
            no_trigger = any(trigger not in ex.input_seq for ex in training_examples)

            if has_trigger and not no_trigger:
                # Only trigger-present branch seen → create trigger-absent input
                inp = generate_input_sequence(rng, [s for s in sts.alphabet if s != trigger], seq_length)
                if inp not in training_inputs and len(items) < n_items:
                    correct = solve(sts, inp)
                    items.append(TestItem(
                        input_seq=inp,
                        correct_output=correct,
                        item_type=ItemType.E,
                        partial_rule_answer=None,
                        sts_id=sts.id,
                        metadata={"rule_type": "conditional", "unseen_branch": "absent"},
                    ))
                    training_inputs.add(inp)

            elif no_trigger and not has_trigger:
                # Only trigger-absent branch seen → create trigger-present input
                inp_chars = [rng.choice(sts.alphabet) for _ in range(seq_length)]
                inp_chars[rng.randint(0, seq_length - 1)] = trigger
                inp = "".join(inp_chars)
                if inp not in training_inputs and len(items) < n_items:
                    correct = solve(sts, inp)
                    items.append(TestItem(
                        input_seq=inp,
                        correct_output=correct,
                        item_type=ItemType.E,
                        partial_rule_answer=None,
                        sts_id=sts.id,
                        metadata={"rule_type": "conditional", "unseen_branch": "present"},
                    ))
                    training_inputs.add(inp)
            # If both branches seen, this rule isn't testable via Type E at this N

        elif rule.rule_type == RuleType.COMPOSITIONAL:
            # Generate input where the inner rule acts on a novel intermediate
            # This is harder to engineer precisely — generate candidates and check
            for _ in range(20):
                if len(items) >= n_items:
                    break
                inp = generate_input_sequence(rng, sts.alphabet, seq_length)
                if inp in training_inputs:
                    continue
                correct = solve(sts, inp)
                items.append(TestItem(
                    input_seq=inp,
                    correct_output=correct,
                    item_type=ItemType.E,
                    partial_rule_answer=None,
                    sts_id=sts.id,
                    metadata={"rule_type": "compositional"},
                ))
                training_inputs.add(inp)
                break

    return items[:n_items]


def _generate_type_l_items(
    sts: STSInstance,
    training_examples: list[TrainingExample],
    n_items: int,
    seq_length: int,
    rng: random.Random,
) -> list[TestItem]:
    """Generate Type L (lure) test items.

    Items where a plausible partial rule gives a different answer than the full rule.
    Only meaningful at Tiers 3+ where exceptions and complex rules create divergence.
    """
    training_inputs = {ex.input_seq for ex in training_examples}
    items = []
    attempts = 0

    while len(items) < n_items and attempts < n_items * 50:
        attempts += 1
        inp = generate_input_sequence(rng, sts.alphabet, seq_length)
        if inp in training_inputs:
            continue

        correct = solve(sts, inp)
        partials = get_partial_rule_answers(sts, inp)

        if partials:
            # Use the first divergent partial-rule answer
            partial_name, partial_answer = next(iter(partials.items()))
            items.append(TestItem(
                input_seq=inp,
                correct_output=correct,
                item_type=ItemType.L,
                partial_rule_answer=partial_answer,
                sts_id=sts.id,
                metadata={"partial_rule": partial_name},
            ))
            training_inputs.add(inp)

    return items[:n_items]


def generate_test_items(
    sts: STSInstance,
    training_examples: list[TrainingExample],
    n_per_type: tuple[int, int, int] = (3, 1, 1),
    seq_length: int = 5,
    seed: int | None = None,
) -> list[TestItem]:
    """Generate test items for an STS instance.

    Args:
        sts: The STS instance
        training_examples: Training examples shown to the model
        n_per_type: (n_type_r, n_type_e, n_type_l) items to generate
        seq_length: Length of input sequences
        seed: Random seed

    Returns:
        List of TestItems. Type E/L items may be replaced with Type R
        if they can't be generated for this instance.
    """
    if seed is None:
        seed = hash((sts.id, len(training_examples))) & 0xFFFFFFFF

    rng = random.Random(seed + 2000)
    n_r, n_e, n_l = n_per_type
    items = []

    # Generate Type E items first (most constrained)
    type_e = _generate_type_e_items(sts, training_examples, n_e, seq_length, rng)
    items.extend(type_e)

    # Generate Type L items
    type_l = _generate_type_l_items(sts, training_examples, n_l, seq_length, rng)
    items.extend(type_l)

    # Fill any shortfall with Type R
    shortfall = (n_e - len(type_e)) + (n_l - len(type_l))
    total_r = n_r + shortfall

    type_r = _generate_type_r_items(sts, training_examples, total_r, seq_length, rng)
    items.extend(type_r)

    return items


def generate_dataset(
    tier: int,
    seed: int,
    n_examples: int = 8,
    n_test_items: tuple[int, int, int] = (3, 1, 1),
    alphabet_size: int = 6,
    seq_length: int = 5,
) -> STSDataset:
    """Generate a complete dataset for one STS instance at a given N.

    This is the main entry point for generating evaluation data.

    Returns:
        STSDataset with the STS instance, training examples, and test items.
    """
    sts = generate_sts_instance(
        tier=tier,
        seed=seed,
        alphabet_size=alphabet_size,
        seq_length=seq_length,
    )

    full_training = generate_training_set(sts, n_max=max(n_examples, 32), seq_length=seq_length, seed=seed)
    training = subset_training_set(full_training, n_examples)

    test_items = generate_test_items(
        sts=sts,
        training_examples=training,
        n_per_type=n_test_items,
        seq_length=seq_length,
        seed=seed,
    )

    return STSDataset(
        sts=sts,
        training_examples=training,
        test_items=test_items,
        n_examples=n_examples,
    )


def format_training_examples(examples: list[TrainingExample]) -> str:
    """Format training examples for inclusion in a prompt."""
    lines = []
    for i, ex in enumerate(examples, 1):
        lines.append(f"Example {i}: {ex.input_seq} → {ex.output_seq}")
    return "\n".join(lines)
