from __future__ import annotations

"""Local llama-server client using OpenAI-compatible SDK.

Supports Qwen3.5 thinking mode toggle via per-request extra_body:
  extra_body={"chat_template_kwargs": {"enable_thinking": True/False}}

Requires llama-server launched with --jinja --reasoning-format deepseek.
With --reasoning-format deepseek, reasoning goes to message.reasoning_content
and the answer stays in message.content. If content is empty (reasoning exhausted
the token budget), we return "" and let the eval pipeline handle it.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

import openai

from lesson.models.base import LLMClient, MultiTurnSession


def extract_thinking(response: str) -> Optional[str]:
    """Pull <think>...</think> block from Qwen model responses.

    Returns the content inside the tags, or None if no thinking block is found.
    """
    match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def _strip_thinking(response: str) -> str:
    """Remove <think>...</think> block from response, returning clean text."""
    return re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()


def _extract_response(completion) -> Tuple[str, Optional[str], str]:
    """Extract (content, reasoning, finish_reason) from a completion.

    Handles multiple reasoning field names:
    - reasoning_content: llama.cpp with --reasoning-format deepseek (Qwen)
    - reasoning: mlx_lm.server (Nemotron)
    """
    msg = completion.choices[0].message
    content = msg.content or ""
    # Try multiple field names for reasoning content
    reasoning = getattr(msg, "reasoning_content", None)
    if reasoning is None:
        extras = msg.model_extra if hasattr(msg, "model_extra") and msg.model_extra else {}
        reasoning = extras.get("reasoning_content") or extras.get("reasoning")
    finish_reason = completion.choices[0].finish_reason or ""
    return content, reasoning, finish_reason


class LocalMultiTurnSession(MultiTurnSession):
    """Multi-turn session backed by a local llama-server instance."""

    def __init__(
        self,
        client: openai.OpenAI,
        model: str,
        max_tokens: int,
        strip_thinking: bool,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        self._strip_thinking = strip_thinking
        self._extra_body = extra_body or {}
        self._messages: List[Dict[str, str]] = []

    def send(self, text: str, role: str = "user") -> str:
        """Append a user (or custom-role) message and generate a response."""
        self._messages.append({"role": role, "content": text})
        completion = self._client.chat.completions.create(
            model=self._model,
            messages=self._messages,  # type: ignore[arg-type]
            temperature=0.0,
            max_tokens=self._max_tokens,
            **({"extra_body": self._extra_body} if self._extra_body else {}),
        )
        content, reasoning, finish_reason = _extract_response(completion)
        if self._strip_thinking:
            content = _strip_thinking(content)
        self._messages.append({"role": "assistant", "content": content})
        return content

    def send_json(self, text: str, role: str = "user") -> str:
        """Append a message and generate a JSON response."""
        self._messages.append({"role": role, "content": text})
        completion = self._client.chat.completions.create(
            model=self._model,
            messages=self._messages,  # type: ignore[arg-type]
            temperature=0.0,
            max_tokens=self._max_tokens,
            response_format={"type": "json_object"},
            **({"extra_body": self._extra_body} if self._extra_body else {}),
        )
        content, reasoning, finish_reason = _extract_response(completion)
        if self._strip_thinking:
            content = _strip_thinking(content)
        self._messages.append({"role": "assistant", "content": content})
        return content

    def inject(self, text: str, role: str = "assistant") -> None:
        """Inject a message into history without calling the API."""
        self._messages.append({"role": role, "content": text})

    def reset(self) -> None:
        """Clear conversation history."""
        self._messages = []


class LocalClient(LLMClient):
    """Client for a locally running llama-server instance.

    Args:
        name: Human-readable model name (e.g. "qwen3.5-27b-think").
        port: Port the llama-server is listening on. Defaults to 8080.
        max_tokens: Maximum tokens to generate per response. Defaults to 512.
        strip_thinking: If True, strip <think>...</think> blocks from responses.
        enable_thinking: Controls Qwen3.5 thinking mode via extra_body.
            True = reasoning tokens visible, False = no reasoning.
            None = don't send the parameter (server default).
    """

    def __init__(
        self,
        name: str,
        port: int = 8080,
        max_tokens: int = 512,
        strip_thinking: bool = False,
        enable_thinking: Optional[bool] = None,
        model_id: Optional[str] = None,
    ) -> None:
        self.name = name
        self._port = port
        self._max_tokens = max_tokens
        self._strip_thinking = strip_thinking
        self._client = openai.OpenAI(
            base_url=f"http://localhost:{port}/v1",
            api_key="not-needed",
        )
        self._model = model_id or "local-model"

        # Build extra_body for thinking mode control
        self._extra_body: Dict[str, Any] = {}
        if enable_thinking is not None:
            self._extra_body = {
                "chat_template_kwargs": {"enable_thinking": enable_thinking}
            }

    def prompt(self, text: str) -> str:
        """Single-turn completion with temperature=0."""
        completion = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": text}],
            temperature=0.0,
            max_tokens=self._max_tokens,
            **({"extra_body": self._extra_body} if self._extra_body else {}),
        )
        content, reasoning, finish_reason = _extract_response(completion)
        if self._strip_thinking:
            content = _strip_thinking(content)
        return content

    def prompt_json(self, text: str) -> str:
        """Single-turn completion requesting JSON output. Returns raw JSON string."""
        completion = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": text}],
            temperature=0.0,
            max_tokens=self._max_tokens,
            response_format={"type": "json_object"},
            **({"extra_body": self._extra_body} if self._extra_body else {}),
        )
        content, reasoning, finish_reason = _extract_response(completion)
        if self._strip_thinking:
            content = _strip_thinking(content)
        return content

    def multi_turn(self) -> LocalMultiTurnSession:
        """Create a new stateful multi-turn session."""
        return LocalMultiTurnSession(
            client=self._client,
            model=self._model,
            max_tokens=self._max_tokens,
            strip_thinking=self._strip_thinking,
            extra_body=self._extra_body,
        )
