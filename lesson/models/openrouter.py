from __future__ import annotations

"""OpenRouter client using the OpenAI-compatible API.

Supports all models available on OpenRouter (OpenAI, Anthropic, Google, DeepSeek,
Meta, Zhipu, xAI, MiniMax, Qwen, Moonshot, etc.) via a unified OpenAI SDK interface.

Key features:
- JSON mode with graceful fallback: json_object → plain text (cached per model)
- Automatic retries for transient errors (429, 500, 502, 503, 504)
- Thread-safe: multiple sessions can run in parallel across different models
- max_tokens defaults to 20,000 for thinking models that need headroom

Based on OpenRouter API docs: https://openrouter.ai/docs
"""

import os
import re
import threading
from typing import Any, Dict, List, Optional, Set

import openai

from lesson.models.base import LLMClient, MultiTurnSession


# Thread-safe set of model IDs that don't support json_object response_format.
# Populated at runtime when a model returns a 400 error on json_object mode.
# Once a model is in this set, all subsequent calls skip json mode and use
# plain text with a JSON instruction in the prompt.
_json_unsupported: Set[str] = set()
_json_unsupported_lock = threading.Lock()

_MAX_RETRIES = 5


def _safe_print(*args: Any) -> None:
    """Print with encoding error handling for terminals that can't render STS symbols."""
    try:
        print(*args)
    except UnicodeEncodeError:
        text = " ".join(str(a) for a in args)
        print(text.encode("ascii", errors="replace").decode("ascii"))


class OpenRouterMultiTurnSession(MultiTurnSession):
    """Multi-turn session backed by the OpenRouter API."""

    def __init__(
        self,
        client: openai.OpenAI,
        model_id: str,
        max_tokens: int,
    ) -> None:
        self._client = client
        self._model_id = model_id
        self._max_tokens = max_tokens
        self._messages: List[Dict[str, str]] = []

    def send(self, text: str, role: str = "user") -> str:
        """Append a message and generate a response."""
        self._messages.append({"role": role, "content": text})
        completion = self._client.chat.completions.create(
            model=self._model_id,
            messages=self._messages,  # type: ignore[arg-type]
            temperature=0.0,
            max_tokens=self._max_tokens,
        )
        content = completion.choices[0].message.content or ""
        self._messages.append({"role": "assistant", "content": content})
        return content

    def send_json(self, text: str, role: str = "user") -> str:
        """Append a message and generate a JSON response.

        Uses response_format: json_object if supported. Falls back to plain
        text with JSON instruction if the model returns a 400 error (cached
        so the fallback fires only once per model per process).
        """
        # Check if this model is known to not support json mode
        with _json_unsupported_lock:
            skip_json = self._model_id in _json_unsupported

        if skip_json:
            # Fall back: add JSON instruction to the prompt
            json_text = text + '\nRespond with ONLY valid JSON: {"output": "YOUR_ANSWER"}'
            return self.send(json_text, role=role)

        # Try with json_object mode
        self._messages.append({"role": role, "content": text})
        try:
            completion = self._client.chat.completions.create(
                model=self._model_id,
                messages=self._messages,  # type: ignore[arg-type]
                temperature=0.0,
                max_tokens=self._max_tokens,
                response_format={"type": "json_object"},
            )
            content = completion.choices[0].message.content or ""
            self._messages.append({"role": "assistant", "content": content})
            return content
        except openai.BadRequestError:
            # Model doesn't support json_object — cache and retry without
            with _json_unsupported_lock:
                _json_unsupported.add(self._model_id)
            # Remove the user message we already appended
            if self._messages and self._messages[-1].get("role") == role:
                self._messages.pop()
            # Retry with plain text + JSON instruction
            json_text = text + '\nRespond with ONLY valid JSON: {"output": "YOUR_ANSWER"}'
            return self.send(json_text, role=role)

    def inject(self, text: str, role: str = "assistant") -> None:
        """Inject a message into history without calling the API."""
        self._messages.append({"role": role, "content": text})

    def reset(self) -> None:
        """Clear conversation history."""
        self._messages = []


class OpenRouterClient(LLMClient):
    """Client for models on OpenRouter via the OpenAI-compatible API.

    Args:
        name: Human-readable model name (e.g. "glm-5", "gpt-5.3-codex").
        model_id: OpenRouter model identifier (e.g. "zhipu/glm-5").
        api_key: OpenRouter API key. Reads from OPENROUTER_API_KEY env var if not provided.
        max_tokens: Maximum tokens to generate per response. Defaults to 20000.
        max_retries: Number of automatic retries for transient errors. Defaults to 5.
    """

    def __init__(
        self,
        name: str,
        model_id: str,
        api_key: Optional[str] = None,
        max_tokens: int = 20_000,
        max_retries: int = _MAX_RETRIES,
    ) -> None:
        self.name = name
        self._model_id = model_id
        self._max_tokens = max_tokens

        api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            raise ValueError(
                "OpenRouter API key not provided and OPENROUTER_API_KEY env var is not set."
            )

        self._client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            max_retries=max_retries,
            default_headers={
                "HTTP-Referer": "https://github.com/lesson-bench",
                "X-Title": "LESSON-Bench",
            },
        )

    def prompt(self, text: str) -> str:
        """Single-turn completion with temperature=0."""
        completion = self._client.chat.completions.create(
            model=self._model_id,
            messages=[{"role": "user", "content": text}],
            temperature=0.0,
            max_tokens=self._max_tokens,
        )
        return completion.choices[0].message.content or ""

    def prompt_json(self, text: str) -> str:
        """Single-turn completion requesting JSON output.

        Tries json_object mode first, falls back to plain text with JSON
        instruction if the model doesn't support it (cached per model).
        """
        with _json_unsupported_lock:
            skip_json = self._model_id in _json_unsupported

        if skip_json:
            json_text = text + '\nRespond with ONLY valid JSON: {"output": "YOUR_ANSWER"}'
            return self.prompt(json_text)

        try:
            completion = self._client.chat.completions.create(
                model=self._model_id,
                messages=[{"role": "user", "content": text}],
                temperature=0.0,
                max_tokens=self._max_tokens,
                response_format={"type": "json_object"},
            )
            return completion.choices[0].message.content or ""
        except openai.BadRequestError:
            with _json_unsupported_lock:
                _json_unsupported.add(self._model_id)
            json_text = text + '\nRespond with ONLY valid JSON: {"output": "YOUR_ANSWER"}'
            return self.prompt(json_text)

    def multi_turn(self) -> OpenRouterMultiTurnSession:
        """Create a new stateful multi-turn session."""
        return OpenRouterMultiTurnSession(
            client=self._client,
            model_id=self._model_id,
            max_tokens=self._max_tokens,
        )


# ---------------------------------------------------------------------------
# Model configurations for SB1/SB2 evaluation
# ---------------------------------------------------------------------------
# Model IDs follow OpenRouter's "provider/model" convention.
# max_tokens is set high (20000) to accommodate thinking models.
# Verify slugs against https://openrouter.ai/models if adding new models.
# ---------------------------------------------------------------------------

OPENROUTER_MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    # --- SB2 Pilot models (8 selected for hypothesis testing) ---
    "glm-5": {
        "model_id": "z-ai/glm-5",
        "max_tokens": 20_000,
    },
    "gpt-5.3-codex": {
        "model_id": "openai/gpt-5.3-codex",
        "max_tokens": 20_000,
    },
    "gpt-5.3-chat": {
        "model_id": "openai/gpt-5.3-chat",
        "max_tokens": 20_000,
    },
    "claude-sonnet-4.6": {
        "model_id": "anthropic/claude-sonnet-4.6",
        "max_tokens": 20_000,
    },
    "deepseek-r1": {
        "model_id": "deepseek/deepseek-r1",
        "max_tokens": 20_000,
    },
    "deepseek-v3.2": {
        "model_id": "deepseek/deepseek-v3.2",
        "max_tokens": 20_000,
    },
    "claude-haiku-4.5": {
        "model_id": "anthropic/claude-haiku-4.5",
        "max_tokens": 20_000,
    },
    # --- Additional SB1 models (passed filter, deferred from SB2 pilot) ---
    "claude-opus-4.6": {
        "model_id": "anthropic/claude-opus-4.6",
        "max_tokens": 20_000,
    },
    "gpt-5.4-mini": {
        "model_id": "openai/gpt-5.4-mini",
        "max_tokens": 20_000,
    },
    "minimax-m2.7": {
        "model_id": "minimax/minimax-m2.7",
        "max_tokens": 20_000,
    },
    "qwen-3.5-397b": {
        "model_id": "qwen/qwen3.5-397b-a17b",
        "max_tokens": 20_000,
    },
    "gemini-3.1-pro": {
        "model_id": "google/gemini-3.1-pro-preview",
        "max_tokens": 20_000,
    },
    "gemini-3.1-flash-lite": {
        "model_id": "google/gemini-3.1-flash-lite-preview",
        "max_tokens": 20_000,
    },
    "kimi-k2.5": {
        "model_id": "moonshotai/kimi-k2.5",
        "max_tokens": 20_000,
    },
    "grok-4.20": {
        "model_id": "x-ai/grok-4.20-beta",
        "max_tokens": 20_000,
    },
    # --- Models that failed SB1 filter (kept for reference) ---
    "llama-3.3-70b": {
        "model_id": "meta-llama/llama-3.3-70b-instruct",
        "max_tokens": 20_000,
    },
    "llama-4-maverick": {
        "model_id": "meta-llama/llama-4-maverick",
        "max_tokens": 20_000,
    },
}


def get_openrouter_client(
    name: str, api_key: Optional[str] = None, **overrides: Any
) -> OpenRouterClient:
    """Instantiate an OpenRouterClient by registry name."""
    if name not in OPENROUTER_MODEL_CONFIGS:
        raise KeyError(
            f"Unknown OpenRouter model {name!r}. "
            f"Available: {sorted(OPENROUTER_MODEL_CONFIGS)}"
        )
    config = {**OPENROUTER_MODEL_CONFIGS[name], **overrides}
    return OpenRouterClient(name=name, api_key=api_key, **config)


def check_openrouter_key() -> bool:
    """Check if an OpenRouter API key is available."""
    return bool(os.environ.get("OPENROUTER_API_KEY"))
