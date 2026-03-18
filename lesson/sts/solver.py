"""Deterministic STS solver.

Given an STS instance and an input sequence, produces the unique correct output.
Also implements partial-rule simulators for Type L items and misleading feedback.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .types import Rule, RuleType, STSInstance, Exception_


def apply_rule(rule: Rule, input_seq: str, alphabet: list[str]) -> str:
    """Apply a single rule to an input sequence.

    Returns the transformed sequence.
    """
    spec = rule.spec

    if rule.rule_type == RuleType.DIRECT:
        # Substring replacement: replace all occurrences of pattern with replacement
        pattern = spec["pattern"]
        replacement = spec["replacement"]
        return input_seq.replace(pattern, replacement)

    elif rule.rule_type == RuleType.POSITIONAL:
        # Position-dependent: transform symbol at specific position
        position = spec["position"]  # Index into the sequence
        mapping = spec["mapping"]    # Dict: input_symbol -> output_symbol
        chars = list(input_seq)
        if 0 <= position < len(chars) and chars[position] in mapping:
            chars[position] = mapping[chars[position]]
        return "".join(chars)

    elif rule.rule_type == RuleType.CONDITIONAL:
        # Context-dependent: check for trigger, then apply one of two branches
        trigger = spec["trigger"]         # Symbol whose presence determines branch
        if_present = spec["if_present"]   # Rule spec to apply if trigger found
        if_absent = spec["if_absent"]     # Rule spec to apply if trigger not found
        if trigger in input_seq:
            branch_rule = Rule(rule_type=if_present["type"], spec=if_present["spec"])
        else:
            branch_rule = Rule(rule_type=if_absent["type"], spec=if_absent["spec"])
        return apply_rule(branch_rule, input_seq, alphabet)

    elif rule.rule_type == RuleType.COMPOSITIONAL:
        # Chained: apply inner rule first, then outer rule
        inner = Rule(rule_type=spec["inner"]["type"], spec=spec["inner"]["spec"])
        outer = Rule(rule_type=spec["outer"]["type"], spec=spec["outer"]["spec"])
        intermediate = apply_rule(inner, input_seq, alphabet)
        return apply_rule(outer, intermediate, alphabet)

    raise ValueError(f"Unknown rule type: {rule.rule_type}")


def check_exceptions(exceptions: list[Exception_], input_seq: str) -> str | None:
    """Check if any exception applies to the input.

    Returns the override output if an exception matches, else None.
    """
    for exc in exceptions:
        if exc.position is not None:
            # Positional trigger
            chars = list(input_seq)
            if 0 <= exc.position < len(chars) and chars[exc.position] == exc.trigger:
                return exc.output_override
        else:
            # Anywhere trigger
            if exc.trigger in input_seq:
                return exc.output_override
    return None


def solve(sts: STSInstance, input_seq: str) -> str:
    """Compute the correct output for an input sequence under the full STS rules.

    Applies rules in order. Checks exceptions first (exceptions override normal rules).
    """
    # Check exceptions first
    exception_output = check_exceptions(sts.exceptions, input_seq)
    if exception_output is not None:
        return exception_output

    # Apply rules in sequence
    result = input_seq
    for rule in sts.rules:
        result = apply_rule(rule, result, sts.alphabet)

    return result


# --- Partial-rule simulators (for Type L items and misleading feedback) ---

def solve_exception_blind(sts: STSInstance, input_seq: str) -> str:
    """Apply rules WITHOUT checking exceptions.

    Partial-rule simulator #1: ignores all exception cases.
    """
    result = input_seq
    for rule in sts.rules:
        result = apply_rule(rule, result, sts.alphabet)
    return result


def solve_condition_blind(sts: STSInstance, input_seq: str) -> str:
    """Apply rules using only the DEFAULT conditional branch.

    Partial-rule simulator #2: for CONDITIONAL rules, always takes if_absent branch
    regardless of whether the trigger is present.
    """
    exception_output = check_exceptions(sts.exceptions, input_seq)
    if exception_output is not None:
        return exception_output

    result = input_seq
    for rule in sts.rules:
        if rule.rule_type == RuleType.CONDITIONAL:
            # Always use the if_absent branch (default/dominant)
            branch_spec = rule.spec["if_absent"]
            branch_rule = Rule(rule_type=branch_spec["type"], spec=branch_spec["spec"])
            result = apply_rule(branch_rule, result, sts.alphabet)
        else:
            result = apply_rule(rule, result, sts.alphabet)

    return result


def solve_order_blind(sts: STSInstance, input_seq: str) -> str:
    """Apply COMPOSITIONAL rules in WRONG order (inner/outer swapped).

    Partial-rule simulator #3: for COMPOSITIONAL rules, applies outer first then inner
    instead of the specified inner-then-outer.
    """
    exception_output = check_exceptions(sts.exceptions, input_seq)
    if exception_output is not None:
        return exception_output

    result = input_seq
    for rule in sts.rules:
        if rule.rule_type == RuleType.COMPOSITIONAL:
            spec = rule.spec
            # Swap: apply outer first, then inner (wrong order)
            outer = Rule(rule_type=spec["outer"]["type"], spec=spec["outer"]["spec"])
            inner = Rule(rule_type=spec["inner"]["type"], spec=spec["inner"]["spec"])
            intermediate = apply_rule(outer, result, sts.alphabet)
            result = apply_rule(inner, intermediate, sts.alphabet)
        else:
            result = apply_rule(rule, result, sts.alphabet)

    return result


def get_partial_rule_answers(sts: STSInstance, input_seq: str) -> dict[str, str]:
    """Get all partial-rule answers for an input.

    Returns dict mapping simulator name -> output.
    Only includes answers that DIFFER from the full-rule answer.
    """
    correct = solve(sts, input_seq)
    partials = {}

    eb = solve_exception_blind(sts, input_seq)
    if eb != correct:
        partials["exception_blind"] = eb

    cb = solve_condition_blind(sts, input_seq)
    if cb != correct:
        partials["condition_blind"] = cb

    ob = solve_order_blind(sts, input_seq)
    if ob != correct:
        partials["order_blind"] = ob

    return partials
