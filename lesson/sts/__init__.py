"""STS (Symbolic Transformation Systems) module.

Core infrastructure for generating contamination-proof symbolic tasks.
"""

from .types import (
    RuleType, ItemType, Rule, Exception_, STSInstance,
    TestItem, TrainingExample, STSDataset,
)
from .solver import solve, get_partial_rule_answers
from .generator import (
    generate_sts_instance,
    generate_training_set,
    subset_training_set,
    generate_test_items,
    generate_dataset,
    format_training_examples,
)
from .symbols import DEFAULT_SYMBOLS, CANDIDATE_SYMBOLS, select_symbols
