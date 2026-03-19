from __future__ import annotations

"""Gemini client using the REST API directly.

Supports thinking level control (LOW/MEDIUM/HIGH) via thinkingConfig.
The Python SDK (google-genai 1.x) doesn't expose thinkingLevel yet,
so we call the REST endpoint directly.

Gemini 3.1 Pro cannot fully disable thinking — LOW is the minimum.
"""

import json
import os
import time
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional

from lesson.models.base import LLMClient, MultiTurnSession

_MAX_RETRIES = 5
_RETRY_DELAY = 2.0
_API_BASE = "https://generativelanguage.googleapis.com/v1beta"

# Rate limiting: minimum seconds between API calls (shared across all instances)
_MIN_CALL_INTERVAL = 2.0  # seconds between calls
_last_call_time = 0.0


def _rate_limit() -> None:
    """Enforce minimum interval between API calls to avoid 429s."""
    global _last_call_time
    now = time.time()
    elapsed = now - _last_call_time
    if elapsed < _MIN_CALL_INTERVAL:
        time.sleep(_MIN_CALL_INTERVAL - elapsed)
    _last_call_time = time.time()


def _api_call(
    api_key: str,
    model_id: str,
    contents: List[Dict],
    temperature: float = 0.0,
    max_tokens: int = 512,
    thinking_level: str = "MEDIUM",
    response_mime_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Make a Gemini REST API call with retry logic and rate limiting."""
    url = f"{_API_BASE}/models/{model_id}:generateContent?key={api_key}"

    gen_config: Dict[str, Any] = {
        "thinkingConfig": {"thinkingLevel": thinking_level},
        "temperature": temperature,
        "maxOutputTokens": max_tokens,
    }
    if response_mime_type:
        gen_config["responseMimeType"] = response_mime_type

    payload = json.dumps({
        "contents": contents,
        "generationConfig": gen_config,
    }).encode("utf-8")

    last_exc = None
    for attempt in range(_MAX_RETRIES):
        _rate_limit()
        try:
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            resp = urllib.request.urlopen(req, timeout=120)
            return json.loads(resp.read())
        except (urllib.error.HTTPError, urllib.error.URLError, OSError) as exc:
            last_exc = exc
            err_str = str(exc)
            # Retry on 500, 503, 429, timeouts with exponential backoff
            if any(s in err_str for s in ["500", "503", "429", "timed out"]):
                if attempt < _MAX_RETRIES - 1:
                    backoff = _RETRY_DELAY * (2 ** attempt)  # 2, 4, 8, 16s
                    if "429" in err_str:
                        backoff = max(backoff, 10.0)  # longer wait for rate limits
                    print(f"    Retry {attempt+1}/{_MAX_RETRIES-1} after {backoff:.0f}s ({err_str})")
                    time.sleep(backoff)
                continue
            raise
    raise last_exc  # type: ignore


def _extract_answer(result: Dict[str, Any]) -> str:
    """Extract answer text from API response, skipping thought parts."""
    parts = result["candidates"][0]["content"]["parts"]
    answer_parts = []
    for part in parts:
        if part.get("thought"):
            continue
        if "text" in part:
            answer_parts.append(part["text"])
    return "".join(answer_parts).strip()


def _contents_from_text(text: str) -> List[Dict]:
    """Build single-turn contents from text."""
    return [{"role": "user", "parts": [{"text": text}]}]


class GeminiMultiTurnSession(MultiTurnSession):
    """Multi-turn session backed by the Gemini REST API."""

    def __init__(
        self,
        api_key: str,
        model_id: str,
        max_tokens: int,
        thinking_level: str,
    ) -> None:
        self._api_key = api_key
        self._model_id = model_id
        self._max_tokens = max_tokens
        self._thinking_level = thinking_level
        self._history: List[Dict] = []

    def send(self, text: str, role: str = "user") -> str:
        """Append a message and generate a response from Gemini."""
        self._history.append({
            "role": "user",
            "parts": [{"text": text}],
        })

        result = _api_call(
            api_key=self._api_key,
            model_id=self._model_id,
            contents=self._history,
            temperature=0.0,
            max_tokens=self._max_tokens,
            thinking_level=self._thinking_level,
        )

        response_text = _extract_answer(result)
        self._history.append({
            "role": "model",
            "parts": [{"text": response_text}],
        })
        return response_text

    def send_json(self, text: str, role: str = "user") -> str:
        """Append a message and generate a JSON response from Gemini."""
        self._history.append({
            "role": "user",
            "parts": [{"text": text}],
        })

        result = _api_call(
            api_key=self._api_key,
            model_id=self._model_id,
            contents=self._history,
            temperature=0.0,
            max_tokens=self._max_tokens,
            thinking_level=self._thinking_level,
            response_mime_type="application/json",
        )

        response_text = _extract_answer(result)
        self._history.append({
            "role": "model",
            "parts": [{"text": response_text}],
        })
        return response_text

    def inject(self, text: str, role: str = "assistant") -> None:
        """Inject a message into history without calling the API."""
        gemini_role = "model" if role == "assistant" else "user"
        self._history.append({
            "role": gemini_role,
            "parts": [{"text": text}],
        })

    def reset(self) -> None:
        """Clear conversation history."""
        self._history = []


class GeminiClient(LLMClient):
    """Client for Gemini models via the REST API.

    Args:
        name: Human-readable model name (e.g. "gemini-pro").
        model_id: Gemini model identifier (e.g. "gemini-3.1-pro-preview").
        api_key: Gemini API key. Reads from GEMINI_API_KEY env var if not provided.
        max_tokens: Maximum tokens to generate per response. Defaults to 512.
        thinking_level: "LOW", "MEDIUM", or "HIGH". Default "MEDIUM".
    """

    def __init__(
        self,
        name: str,
        model_id: str,
        api_key: Optional[str] = None,
        max_tokens: int = 512,
        thinking_level: str = "MEDIUM",
    ) -> None:
        self.name = name
        self._model_id = model_id
        self._max_tokens = max_tokens
        self._thinking_level = thinking_level

        self._api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "Gemini API key not provided and GEMINI_API_KEY env var is not set."
            )

    def prompt(self, text: str) -> str:
        """Single-turn completion with retry on transient errors."""
        result = _api_call(
            api_key=self._api_key,
            model_id=self._model_id,
            contents=_contents_from_text(text),
            temperature=0.0,
            max_tokens=self._max_tokens,
            thinking_level=self._thinking_level,
        )
        return _extract_answer(result)

    def prompt_json(self, text: str) -> str:
        """Single-turn completion requesting JSON output."""
        result = _api_call(
            api_key=self._api_key,
            model_id=self._model_id,
            contents=_contents_from_text(text),
            temperature=0.0,
            max_tokens=self._max_tokens,
            thinking_level=self._thinking_level,
            response_mime_type="application/json",
        )
        return _extract_answer(result)

    def multi_turn(self) -> GeminiMultiTurnSession:
        """Create a new stateful multi-turn session."""
        return GeminiMultiTurnSession(
            api_key=self._api_key,
            model_id=self._model_id,
            max_tokens=self._max_tokens,
            thinking_level=self._thinking_level,
        )
