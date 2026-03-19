from __future__ import annotations

"""LM Studio client using the OpenAI-compatible REST API.

LM Studio runs a local server at http://localhost:1234/v1 (default port)
with full OpenAI API compatibility. Supports:
- Chat completions with structured output (json_schema)
- Concurrent generation via continuous batching (default: 4 slots)
- Model hot-loading and discovery

Docs: https://lmstudio.ai/docs/api/openai-api
"""

import os
import re
import urllib.error
import urllib.request
import json
from typing import Any, Dict, List, Optional

import openai

from lesson.models.base import LLMClient, MultiTurnSession


_DEFAULT_PORT = 1234
_DEFAULT_BASE_URL = "http://localhost:1234/v1"


def check_lmstudio_server(base_url: str = _DEFAULT_BASE_URL) -> bool:
    """Check if LM Studio server is running and responding.

    Args:
        base_url: LM Studio API base URL (default: http://localhost:1234/v1).

    Returns:
        True if the server is reachable and returns a model list.
    """
    try:
        url = f"{base_url}/models"
        req = urllib.request.Request(url)
        resp = urllib.request.urlopen(req, timeout=5)
        data = json.loads(resp.read())
        return isinstance(data, dict) and "data" in data
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        return False


def list_lmstudio_models(base_url: str = _DEFAULT_BASE_URL) -> List[str]:
    """List models currently available in LM Studio.

    Returns model IDs. With JIT loading enabled, returns all downloaded models;
    with JIT disabled, returns only loaded models.
    """
    try:
        url = f"{base_url}/models"
        req = urllib.request.Request(url)
        resp = urllib.request.urlopen(req, timeout=5)
        data = json.loads(resp.read())
        return [m["id"] for m in data.get("data", [])]
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        return []


class LMStudioMultiTurnSession(MultiTurnSession):
    """Multi-turn session backed by a LM Studio local server."""

    def __init__(
        self,
        client: openai.OpenAI,
        model: str,
        max_tokens: int,
    ) -> None:
        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        self._messages: List[Dict[str, str]] = []

    def send(self, text: str, role: str = "user") -> str:
        """Append a message and generate a response."""
        self._messages.append({"role": role, "content": text})
        completion = self._client.chat.completions.create(
            model=self._model,
            messages=self._messages,  # type: ignore[arg-type]
            temperature=0.0,
            max_tokens=self._max_tokens,
        )
        content = completion.choices[0].message.content or ""
        self._messages.append({"role": "assistant", "content": content})
        return content

    def send_json(self, text: str, role: str = "user") -> str:
        """Append a message and generate a JSON response."""
        self._messages.append({"role": role, "content": text})
        try:
            completion = self._client.chat.completions.create(
                model=self._model,
                messages=self._messages,  # type: ignore[arg-type]
                temperature=0.0,
                max_tokens=self._max_tokens,
                response_format={"type": "json_object"},
            )
            content = completion.choices[0].message.content or ""
            self._messages.append({"role": "assistant", "content": content})
            return content
        except openai.BadRequestError:
            # Model doesn't support json_object — retry without
            if self._messages and self._messages[-1].get("role") == role:
                self._messages.pop()
            json_text = text + '\nRespond with ONLY valid JSON: {"output": "YOUR_ANSWER"}'
            return self.send(json_text, role=role)

    def inject(self, text: str, role: str = "assistant") -> None:
        """Inject a message into history without calling the API."""
        self._messages.append({"role": role, "content": text})

    def reset(self) -> None:
        """Clear conversation history."""
        self._messages = []


class LMStudioClient(LLMClient):
    """Client for models running on LM Studio's local server.

    LM Studio serves models via an OpenAI-compatible API. Supports concurrent
    generation with continuous batching (default 4 slots).

    Args:
        name: Human-readable model name (e.g. "qwen3-coder-30b").
        model_id: Model identifier as shown in LM Studio (e.g. the GGUF filename).
            If None, uses the first available model.
        port: Port the LM Studio server is listening on. Defaults to 1234.
        max_tokens: Maximum tokens to generate per response. Defaults to 2048.
        base_url: Override the full base URL (ignores port if set).
    """

    def __init__(
        self,
        name: str,
        model_id: Optional[str] = None,
        port: int = _DEFAULT_PORT,
        max_tokens: int = 2048,
        base_url: Optional[str] = None,
    ) -> None:
        self.name = name
        self._max_tokens = max_tokens

        url = base_url or f"http://localhost:{port}/v1"
        self._base_url = url

        self._client = openai.OpenAI(
            base_url=url,
            api_key="lm-studio",  # LM Studio doesn't require auth
        )

        # If no model_id specified, discover from server
        if model_id:
            self._model = model_id
        else:
            models = list_lmstudio_models(url)
            if models:
                self._model = models[0]
            else:
                self._model = "default"

    def prompt(self, text: str) -> str:
        """Single-turn completion with temperature=0."""
        completion = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": text}],
            temperature=0.0,
            max_tokens=self._max_tokens,
        )
        return completion.choices[0].message.content or ""

    def prompt_json(self, text: str) -> str:
        """Single-turn completion requesting JSON output."""
        try:
            completion = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": text}],
                temperature=0.0,
                max_tokens=self._max_tokens,
                response_format={"type": "json_object"},
            )
            return completion.choices[0].message.content or ""
        except openai.BadRequestError:
            # Model doesn't support json_object — retry with prompt instruction
            json_text = text + '\nRespond with ONLY valid JSON: {"output": "YOUR_ANSWER"}'
            return self.prompt(json_text)

    def multi_turn(self) -> LMStudioMultiTurnSession:
        """Create a new stateful multi-turn session."""
        return LMStudioMultiTurnSession(
            client=self._client,
            model=self._model,
            max_tokens=self._max_tokens,
        )


# ---------------------------------------------------------------------------
# Model configurations for LM Studio evaluation
# ---------------------------------------------------------------------------
# These configs map friendly names to LM Studio model identifiers.
# model_id should match what LM Studio shows in its model list.
# max_tokens is constrained by local context window.
# ---------------------------------------------------------------------------

LMSTUDIO_MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "qwen3-coder-30b": {
        "model_id": None,  # auto-detect from loaded model
        "max_tokens": 2048,
        "port": 1234,
    },
    "qwen3-1.7b": {
        "model_id": None,
        "max_tokens": 2048,
        "port": 1234,
    },
}


def get_lmstudio_client(
    name: str, **overrides: Any
) -> LMStudioClient:
    """Instantiate a LMStudioClient by registry name."""
    if name not in LMSTUDIO_MODEL_CONFIGS:
        raise KeyError(
            f"Unknown LM Studio model {name!r}. "
            f"Available: {sorted(LMSTUDIO_MODEL_CONFIGS)}"
        )
    config = {**LMSTUDIO_MODEL_CONFIGS[name], **overrides}
    return LMStudioClient(name=name, **config)
