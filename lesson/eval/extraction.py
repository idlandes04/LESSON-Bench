from __future__ import annotations

"""Answer extraction utilities for LESSON evaluation.

Provides tiered fallback extraction from raw LLM responses:
- regex mode: strip <think> blocks, look for "Output:" prefix, fall back to last line
- json mode: parse {"output": "..."} from response text
"""

import json
import re
from typing import Optional


def extract_answer_regex(response: str) -> str:
    """Extract the predicted answer from a response using regex heuristics.

    Extraction order:
    1. Strip any <think>...</think> blocks from the response.
    2. Look for a line starting with "Output:" (case-insensitive) and take the
       text following it.
    3. Fall back to the last non-empty line of the stripped response.

    Args:
        response: Raw text returned by the model.

    Returns:
        The extracted answer string, stripped of surrounding whitespace.
        Returns an empty string if the response is empty after cleaning.
    """
    # Step 1: strip <think>...</think> blocks (handles Qwen-style thinking)
    cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

    # Step 2: look for common answer patterns (case-insensitive)
    # Matches: "Output: X", "The output is X", "Answer: X", "Result: X"
    patterns = [
        r"(?i)^output\s*:\s*(.+)$",
        r"(?i)(?:the\s+)?output\s+is\s*:?\s*(.+)$",
        r"(?i)^answer\s*:\s*(.+)$",
        r"(?i)^result\s*:\s*(.+)$",
    ]
    for pat in patterns:
        m = re.search(pat, cleaned, re.MULTILINE)
        if m:
            return m.group(1).strip()

    # Step 3: fall back to the last non-empty line
    lines = [line.strip() for line in cleaned.splitlines()]
    non_empty = [line for line in lines if line]
    if non_empty:
        return non_empty[-1]

    return ""


def extract_answer_json(response: str) -> str:
    """Extract the predicted answer by parsing JSON from the response.

    Looks for a JSON object with an "output" key anywhere in the response text.
    Handles cases where the model wraps JSON in markdown code fences.

    Args:
        response: Raw text returned by the model (may contain surrounding prose).

    Returns:
        The value of the "output" key as a string, stripped of whitespace.
        Returns an empty string if no valid JSON with an "output" key is found.
    """
    # Strip markdown code fences if present (```json ... ``` or ``` ... ```)
    cleaned = re.sub(r"```(?:json)?\s*", "", response)
    cleaned = re.sub(r"```", "", cleaned).strip()

    # Try to find a JSON object in the text
    # First attempt: the whole cleaned string
    candidates = [cleaned]

    # Also search for {...} substrings in case there is surrounding prose
    brace_matches = re.findall(r"\{[^{}]*\}", cleaned, re.DOTALL)
    candidates.extend(brace_matches)

    for candidate in candidates:
        try:
            data = json.loads(candidate)
            if isinstance(data, dict) and "output" in data:
                return str(data["output"]).strip()
        except (json.JSONDecodeError, ValueError):
            continue

    return ""


def extract_answer(response: str, mode: str = "regex") -> str:
    """Dispatcher for answer extraction.

    Args:
        response: Raw text returned by the model.
        mode: Extraction mode — "regex" (default) or "json".

    Returns:
        The extracted answer string.

    Raises:
        ValueError: If an unknown mode is specified.
    """
    if mode == "regex":
        return extract_answer_regex(response)
    elif mode == "json":
        return extract_answer_json(response)
    else:
        raise ValueError(f"Unknown extraction mode: {mode!r}. Expected 'regex' or 'json'.")
