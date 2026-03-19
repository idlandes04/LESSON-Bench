"""Tests for lesson/models/local.py — pure string helpers (zero API calls)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from unittest.mock import MagicMock, PropertyMock
from lesson.models.local import extract_thinking, _strip_thinking, _extract_response


# ---------------------------------------------------------------------------
# extract_thinking
# ---------------------------------------------------------------------------

def test_extract_thinking_returns_content_inside_tags():
    response = "<think>This is my reasoning step.</think>"
    result = extract_thinking(response)
    assert result == "This is my reasoning step."


def test_extract_thinking_strips_surrounding_whitespace():
    response = "<think>  reasoning  </think>"
    result = extract_thinking(response)
    assert result == "reasoning"


def test_extract_thinking_multiline_block():
    response = "<think>\nline one\nline two\n</think>"
    result = extract_thinking(response)
    assert result == "line one\nline two"


def test_extract_thinking_no_tags_returns_none():
    response = "The output is ABC."
    result = extract_thinking(response)
    assert result is None


def test_extract_thinking_empty_response_returns_none():
    result = extract_thinking("")
    assert result is None


def test_extract_thinking_empty_think_block():
    response = "<think></think>"
    result = extract_thinking(response)
    assert result == ""


def test_extract_thinking_ignores_text_after_closing_tag():
    response = "<think>reasoning</think>\nOutput: ABC"
    result = extract_thinking(response)
    assert result == "reasoning"


# ---------------------------------------------------------------------------
# _strip_thinking
# ---------------------------------------------------------------------------

def test_strip_thinking_removes_think_block():
    response = "<think>Some reasoning</think>\nThe answer is ABC."
    result = _strip_thinking(response)
    assert "<think>" not in result
    assert "Some reasoning" not in result
    assert "The answer is ABC." in result


def test_strip_thinking_leaves_rest_intact():
    response = "<think>reasoning</think>Output: XYZ"
    result = _strip_thinking(response)
    assert result == "Output: XYZ"


def test_strip_thinking_no_block_returns_unchanged():
    response = "Output: ABC"
    result = _strip_thinking(response)
    assert result == "Output: ABC"


def test_strip_thinking_multiline_block():
    response = "<think>\nline 1\nline 2\n</think>\nFinal answer."
    result = _strip_thinking(response)
    assert "line 1" not in result
    assert "line 2" not in result
    assert "Final answer." in result


def test_strip_thinking_empty_string():
    assert _strip_thinking("") == ""


def test_strip_thinking_only_think_block_returns_empty():
    response = "<think>all reasoning</think>"
    result = _strip_thinking(response)
    assert result == ""


# ---------------------------------------------------------------------------
# _extract_response
# ---------------------------------------------------------------------------

def _make_completion(content: str, reasoning_content=None, finish_reason: str = "stop",
                     use_model_extra: bool = False, extra_key: str = "reasoning_content"):
    """Build a minimal mock completion object."""
    msg = MagicMock()
    msg.content = content
    msg.finish_reason = finish_reason  # won't be used — it's on choices[0]

    if use_model_extra:
        # Simulate model_extra dict path
        msg.model_extra = {extra_key: reasoning_content}
        # getattr(msg, "reasoning_content", None) should return None
        del msg.reasoning_content  # remove auto-spec attribute
        type(msg).reasoning_content = PropertyMock(return_value=None)
    else:
        # Direct attribute path
        type(msg).reasoning_content = PropertyMock(return_value=reasoning_content)
        msg.model_extra = {}

    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = finish_reason

    completion = MagicMock()
    completion.choices = [choice]
    return completion


def test_extract_response_basic_content():
    completion = _make_completion("Output: ABC", reasoning_content=None)
    content, reasoning, finish_reason = _extract_response(completion)
    assert content == "Output: ABC"
    assert reasoning is None
    assert finish_reason == "stop"


def test_extract_response_with_reasoning_content_attribute():
    completion = _make_completion("Output: XYZ", reasoning_content="My deep reasoning")
    content, reasoning, finish_reason = _extract_response(completion)
    assert content == "Output: XYZ"
    assert reasoning == "My deep reasoning"


def test_extract_response_empty_content_returns_empty_string():
    completion = _make_completion("", reasoning_content=None)
    content, reasoning, finish_reason = _extract_response(completion)
    assert content == ""


def test_extract_response_none_content_coerced_to_empty():
    """If msg.content is None (API returned nothing), should yield empty string."""
    msg = MagicMock()
    msg.content = None
    type(msg).reasoning_content = PropertyMock(return_value=None)
    msg.model_extra = {}

    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = "length"

    completion = MagicMock()
    completion.choices = [choice]

    content, reasoning, finish_reason = _extract_response(completion)
    assert content == ""
    assert finish_reason == "length"


def test_extract_response_finish_reason_propagated():
    completion = _make_completion("Answer", finish_reason="length")
    _, _, finish_reason = _extract_response(completion)
    assert finish_reason == "length"
