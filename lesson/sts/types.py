"""Core data types for Symbolic Transformation Systems."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class RuleType(Enum):
    DIRECT = "direct"           # substring replacement: ◈⬡ → ⟐
    POSITIONAL = "positional"   # position-dependent: symbol[i] → output
    CONDITIONAL = "conditional" # context-dependent: IF ⧫ in input THEN ...
    COMPOSITIONAL = "compositional"  # chained: apply(rule_1, apply(rule_2, input))


class ItemType(Enum):
    R = "R"  # Rule-consistent (any approach gives correct answer)
    E = "E"  # Extrapolation (requires generalization beyond surface similarity)
    L = "L"  # Lure (plausible partial rule gives different answer)


@dataclass(frozen=True)
class Rule:
    """A single transformation rule within an STS."""
    rule_type: RuleType
    spec: dict[str, Any]  # Rule-type-specific parameters

    def __repr__(self) -> str:
        return f"Rule({self.rule_type.value}, {self.spec})"


@dataclass(frozen=True)
class Exception_:
    """An exception to the normal rules for specific inputs."""
    trigger: str  # Symbol or pattern that triggers the exception
    position: int | None  # Position where trigger must appear (None = anywhere)
    output_override: str  # What the output becomes instead


@dataclass
class STSInstance:
    """A complete Symbolic Transformation System instance."""
    id: str
    alphabet: list[str]        # 4-12 Unicode symbols
    rules: list[Rule]          # 2-8 transformation rules
    exceptions: list[Exception_]  # 0-20% of inputs follow alternate rules
    tier: int                  # Difficulty tier 1-5
    seed: int                  # Random seed for reproducibility
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_non_direct_rule(self) -> bool:
        """Check if instance has at least one non-DIRECT rule (required for Type E items)."""
        return any(r.rule_type != RuleType.DIRECT for r in self.rules)


@dataclass(frozen=True)
class TestItem:
    """A single test item derived from an STS instance."""
    input_seq: str             # Input symbol sequence
    correct_output: str        # Verified correct output
    item_type: ItemType        # R, E, or L
    partial_rule_answer: str | None  # For Type L: what the partial rule gives
    sts_id: str               # Which STS instance this belongs to
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingExample:
    """A single input-output training example."""
    input_seq: str
    output_seq: str


@dataclass
class STSDataset:
    """Complete dataset for one STS instance at a given N."""
    sts: STSInstance
    training_examples: list[TrainingExample]
    test_items: list[TestItem]
    n_examples: int  # How many training examples shown
