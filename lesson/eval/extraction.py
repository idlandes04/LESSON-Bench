from __future__ import annotations

"""Answer extraction utilities for LESSON evaluation.

Tiered extraction strategy (v9.0+):
1. JSON mode (primary): parse {"output": "..."} from structured API responses
2. Symbol-aware extraction: find substrings composed only of known vocabulary symbols
3. Regex fallback: strip reasoning, look for "Output:" prefix, last-line heuristic

All extractors apply normalize_answer() to strip whitespace and spaces between symbols.
"""

import json
import re
from typing import Optional, Sequence


def normalize_answer(answer: str) -> str:
    """Normalize an extracted answer for comparison.

    Strips leading/trailing whitespace and removes spaces between symbols.
    This handles models that output "◈ ⬡ ⟐" vs expected "◈⬡⟐".
    """
    return answer.strip().replace(" ", "")


def extract_answer_json(response: str) -> str:
    """Extract the predicted answer by parsing JSON from the response.

    Looks for a JSON object with an "output" key anywhere in the response text.
    Handles cases where the model wraps JSON in markdown code fences.
    Also handles TRUNCATED JSON like '{"output": "▲' where the model's
    thinking tokens consumed the output budget (common with Gemini LOW thinking).

    Args:
        response: Raw text returned by the model (may contain surrounding prose).

    Returns:
        The normalized value of the "output" key, or empty string if not found.
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
                return normalize_answer(str(data["output"]))
        except (json.JSONDecodeError, ValueError):
            continue

    # Handle TRUNCATED JSON: '{\n  "output": "▲' or '{"output": "⬡⟐'
    # Gemini often truncates JSON when thinking tokens eat the output budget.
    # Try to recover the value after "output": "
    truncated_match = re.search(
        r'"output"\s*:\s*"([^"]*?)(?:"|\Z)',
        cleaned,
    )
    if truncated_match:
        value = truncated_match.group(1)
        if value:  # Don't return empty string from truncated '{"output": "'
            return normalize_answer(value)

    return ""


def extract_answer_symbols(response: str, vocabulary: Sequence[str]) -> str:
    """Extract answer by finding substrings composed only of known vocabulary symbols.

    This is the key extraction mode for fighting reasoning leakage. When a model
    outputs "Wait, ◈◆ becomes ⬡? No, maybe ◈◆◈", this function finds "◈◆◈" as the
    longest symbol-only substring on the last relevant line.

    Strategy:
    1. Strip <think> blocks and markdown fences.
    2. Look for lines starting with "Output:" / "Answer:" — extract symbol content.
    3. Find all maximal substrings composed only of vocabulary symbols + spaces.
    4. Return the last such substring (models tend to put the answer at the end).

    Args:
        response: Raw text from the model.
        vocabulary: The STS alphabet symbols used in this problem.

    Returns:
        Normalized symbol-only answer, or empty string if none found.
    """
    if not vocabulary:
        return ""

    # Step 1: strip thinking blocks
    cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    cleaned = re.sub(r"```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"```", "", cleaned).strip()

    if not cleaned:
        return ""

    # Build a regex character class matching any vocabulary symbol
    # Escape each symbol for regex safety
    escaped = [re.escape(s) for s in vocabulary]
    sym_pattern = "|".join(escaped)
    # Match sequences of 1+ vocabulary symbols (with optional spaces between)
    seq_pattern = rf"(?:{sym_pattern})(?:\s*(?:{sym_pattern}))*"

    # Step 2: look for "Output: <symbols>" or "Answer: <symbols>" patterns
    for prefix in [r"[Oo]utput\s*:\s*", r"[Aa]nswer\s*:\s*", r"[Rr]esult\s*:\s*"]:
        m = re.search(prefix + rf"({seq_pattern})", cleaned)
        if m:
            return normalize_answer(m.group(1))

    # Step 3: find all maximal symbol-only sequences in the text
    matches = re.findall(seq_pattern, cleaned)
    if matches:
        # Return the LAST match — models put reasoning first, answer last
        return normalize_answer(matches[-1])

    return ""


def extract_answer_regex(response: str) -> str:
    """Extract the predicted answer from a response using regex heuristics.

    Last-resort fallback when structured output and symbol-aware extraction fail.

    Extraction order:
    1. Strip any <think>...</think> blocks from the response.
    2. Look for a line starting with "Output:" (case-insensitive).
    3. Fall back to the last non-empty line of the stripped response.

    Args:
        response: Raw text returned by the model.

    Returns:
        The normalized extracted answer string.
    """
    # Step 1: strip <think>...</think> blocks (handles Qwen-style thinking)
    cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

    # Step 2: look for common answer patterns (case-insensitive)
    patterns = [
        r"(?i)^output\s*:\s*(.+)$",
        r"(?i)(?:the\s+)?output\s+is\s*:?\s*(.+)$",
        r"(?i)^answer\s*:\s*(.+)$",
        r"(?i)^result\s*:\s*(.+)$",
    ]
    for pat in patterns:
        m = re.search(pat, cleaned, re.MULTILINE)
        if m:
            return normalize_answer(m.group(1))

    # Step 3: fall back to the last non-empty line
    lines = [line.strip() for line in cleaned.splitlines()]
    non_empty = [line for line in lines if line]
    if non_empty:
        return normalize_answer(non_empty[-1])

    return ""


def extract_answer(
    response: str,
    mode: str = "json",
    vocabulary: Sequence[str] = (),
) -> str:
    """Tiered answer extraction with automatic fallback.

    Extraction order:
    1. If mode="json": try JSON parsing first
    2. If vocabulary provided: try symbol-aware extraction
    3. Fall back to regex heuristics

    Args:
        response: Raw text returned by the model.
        mode: Primary extraction mode — "json" (default) or "regex".
        vocabulary: STS alphabet symbols for symbol-aware extraction.

    Returns:
        The normalized extracted answer string.
    """
    # Tier 1: JSON mode
    if mode == "json":
        result = extract_answer_json(response)
        if result:
            return result

    # Tier 2: Symbol-aware extraction (if vocabulary provided)
    if vocabulary:
        result = extract_answer_symbols(response, vocabulary)
        if result:
            return result

    # Tier 3: Regex fallback
    return extract_answer_regex(response)
