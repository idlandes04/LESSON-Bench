"""Tests for lesson/eval/extraction.py — tiered answer extraction pipeline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from lesson.eval.extraction import (
    extract_answer,
    extract_answer_json,
    extract_answer_regex,
    extract_answer_symbols,
    normalize_answer,
)


# ---------------------------------------------------------------------------
# normalize_answer
# ---------------------------------------------------------------------------

def test_normalize_strips_spaces_between_symbols():
    assert normalize_answer("◈ ⬡ ⟐") == "◈⬡⟐"


def test_normalize_strips_leading_trailing_whitespace():
    assert normalize_answer("  ABC  ") == "ABC"


def test_normalize_both_internal_and_outer_spaces():
    assert normalize_answer("  A B C  ") == "ABC"


def test_normalize_empty_string():
    assert normalize_answer("") == ""


def test_normalize_no_spaces_unchanged():
    assert normalize_answer("◈⬡⟐") == "◈⬡⟐"


# ---------------------------------------------------------------------------
# extract_answer_json
# ---------------------------------------------------------------------------

def test_json_valid_output_key():
    result = extract_answer_json('{"output": "ABC"}')
    assert result == "ABC"


def test_json_in_markdown_fences():
    response = '```json\n{"output": "◈⬡⟐"}\n```'
    assert extract_answer_json(response) == "◈⬡⟐"


def test_json_in_plain_fences():
    response = '```\n{"output": "XYZ"}\n```'
    assert extract_answer_json(response) == "XYZ"


def test_json_surrounded_by_prose():
    response = 'I think the answer is {"output": "ABC"} based on the pattern.'
    assert extract_answer_json(response) == "ABC"


def test_json_truncated_output_key():
    # Gemini sometimes truncates — recovers via regex fallback
    response = '{"output": "▲'
    result = extract_answer_json(response)
    assert result == "▲"


def test_json_truncated_output_key_multi_char():
    response = '{\n  "output": "◈⬡'
    result = extract_answer_json(response)
    assert result == "◈⬡"


def test_json_multiple_objects_picks_output_key():
    # Should pick the one that has "output"
    response = '{"other": "no"} {"output": "CORRECT"}'
    assert extract_answer_json(response) == "CORRECT"


def test_json_no_output_key_returns_empty():
    response = '{"result": "ABC"}'
    assert extract_answer_json(response) == ""


def test_json_no_json_at_all_returns_empty():
    response = "The answer is probably ABC based on the rules."
    assert extract_answer_json(response) == ""


def test_json_empty_response_returns_empty():
    assert extract_answer_json("") == ""


def test_json_output_value_with_spaces_normalized():
    response = '{"output": "A B C"}'
    assert extract_answer_json(response) == "ABC"


def test_json_empty_output_value_skips_truncation_fallback():
    # '{"output": "' — empty value should not return empty string from truncated match
    response = '{"output": "'
    # The truncation regex won't capture a non-empty group, so returns ""
    result = extract_answer_json(response)
    assert result == ""


# ---------------------------------------------------------------------------
# extract_answer_symbols
# ---------------------------------------------------------------------------

VOCAB = ["◈", "⬡", "⟐", "△"]


def test_symbols_finds_answer_at_end():
    response = "Let me reason... maybe it's ◈⬡. Actually it should be ◈⬡⟐"
    result = extract_answer_symbols(response, VOCAB)
    assert result == "◈⬡⟐"


def test_symbols_output_prefix_takes_priority():
    response = "I think △◈ is possible\nOutput: ◈⬡⟐"
    result = extract_answer_symbols(response, VOCAB)
    assert result == "◈⬡⟐"


def test_symbols_answer_prefix():
    response = "Answer: △◈\nSome extra text ◈⬡"
    result = extract_answer_symbols(response, VOCAB)
    # "Answer:" prefix should be found first
    assert result == "△◈"


def test_symbols_strips_think_blocks():
    response = "<think>Let me reason: ◈⬡⟐ might be wrong</think>\nOutput: △◈"
    result = extract_answer_symbols(response, VOCAB)
    assert result == "△◈"


def test_symbols_think_block_stripped_then_fallback_to_last():
    # After stripping think block, find last symbol sequence
    response = "<think>reasoning ◈⬡⟐ wrong</think>\nThe answer is △◈△"
    result = extract_answer_symbols(response, VOCAB)
    assert result == "△◈△"


def test_symbols_empty_vocabulary_returns_empty():
    result = extract_answer_symbols("◈⬡⟐", [])
    assert result == ""


def test_symbols_no_matching_symbols_returns_empty():
    result = extract_answer_symbols("The answer is XYZ", VOCAB)
    assert result == ""


def test_symbols_single_symbol():
    result = extract_answer_symbols("Output: ◈", VOCAB)
    assert result == "◈"


def test_symbols_with_spaces_between_normalized():
    response = "Output: ◈ ⬡ ⟐"
    result = extract_answer_symbols(response, VOCAB)
    assert result == "◈⬡⟐"


def test_symbols_empty_response_after_stripping():
    response = "<think>lots of reasoning here ◈⬡⟐</think>"
    # After stripping the think block, nothing left — should return ""
    result = extract_answer_symbols(response, VOCAB)
    assert result == ""


# ---------------------------------------------------------------------------
# extract_answer_regex
# ---------------------------------------------------------------------------

def test_regex_output_prefix_uppercase():
    result = extract_answer_regex("Output: ABC")
    assert result == "ABC"


def test_regex_output_prefix_lowercase():
    result = extract_answer_regex("output: XYZ")
    assert result == "XYZ"


def test_regex_the_output_is_pattern():
    result = extract_answer_regex("The output is ABC")
    assert result == "ABC"


def test_regex_the_output_is_with_colon():
    result = extract_answer_regex("The output is: DEF")
    assert result == "DEF"


def test_regex_answer_prefix():
    result = extract_answer_regex("Answer: GHI")
    assert result == "GHI"


def test_regex_result_prefix():
    result = extract_answer_regex("Result: JKL")
    assert result == "JKL"


def test_regex_strips_think_block_before_searching():
    response = "<think>Let me reason Output: WRONG</think>\nOutput: CORRECT"
    result = extract_answer_regex(response)
    assert result == "CORRECT"


def test_regex_last_line_fallback():
    response = "Here is my reasoning.\nI considered A and B.\nABC"
    result = extract_answer_regex(response)
    assert result == "ABC"


def test_regex_empty_response_returns_empty():
    assert extract_answer_regex("") == ""


def test_regex_only_whitespace_returns_empty():
    assert extract_answer_regex("   \n  \n  ") == ""


def test_regex_multiline_output_prefix_found():
    response = "Some preamble\nOutput: ◈⬡⟐\nIgnore this trailing line"
    result = extract_answer_regex(response)
    assert result == "◈⬡⟐"


# ---------------------------------------------------------------------------
# extract_answer — tiered fallback
# ---------------------------------------------------------------------------

def test_extract_json_mode_succeeds():
    response = '{"output": "ABC"}'
    result = extract_answer(response, mode="json", vocabulary=[])
    assert result == "ABC"


def test_extract_json_fails_symbols_succeed():
    # Not valid JSON; vocabulary provided — symbol extraction takes over
    response = "The answer is ◈⬡⟐ I think"
    result = extract_answer(response, mode="json", vocabulary=VOCAB)
    assert result == "◈⬡⟐"


def test_extract_all_fail_regex_fallback():
    # No JSON, no vocab symbols, regex last-line fallback
    response = "The final output is:\nABC"
    result = extract_answer(response, mode="json", vocabulary=[])
    assert result == "ABC"


def test_extract_regex_mode_skips_json():
    # mode="regex" should skip JSON tier and go to symbols/regex
    response = '{"output": "SHOULD_BE_SKIPPED"}\nOutput: REGEX_RESULT'
    result = extract_answer(response, mode="regex", vocabulary=[])
    assert result == "REGEX_RESULT"


def test_extract_regex_mode_with_vocabulary():
    response = "Reasoning...\nThe output is ◈⬡"
    result = extract_answer(response, mode="regex", vocabulary=VOCAB)
    # symbol extraction should fire
    assert result == "◈⬡"


def test_extract_empty_response_returns_empty():
    result = extract_answer("", mode="json", vocabulary=VOCAB)
    assert result == ""
