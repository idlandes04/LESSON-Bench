from __future__ import annotations

"""Model abstraction layer for LESSON benchmarks.

Provides a unified interface (LLMClient / MultiTurnSession) for calling
local, Gemini, OpenRouter, and LM Studio models.

Quick-start
-----------
Any model by name (auto-detects provider)::

    from lesson.models import get_client, get_provider_for
    provider = get_provider_for("gpt-5.3-codex")  # -> "openrouter"
    client = get_client(provider, "gpt-5.3-codex")

Provider-specific::

    from lesson.models import get_openrouter_client
    client = get_openrouter_client("gpt-5.3-codex")

Multi-turn session::

    session = client.multi_turn()
    session.send("Start a count from 1.")
    session.inject("1, 2, 3")   # SB2 feedback injection
    reply = session.send("Continue.")
"""

from lesson.models.base import LLMClient, MultiTurnSession
from lesson.models.gemini import GeminiClient, GeminiMultiTurnSession
from lesson.models.local import LocalClient, LocalMultiTurnSession, extract_thinking
from lesson.models.openrouter import OpenRouterClient, OpenRouterMultiTurnSession
from lesson.models.lmstudio import LMStudioClient, LMStudioMultiTurnSession
from lesson.models.registry import (
    # Config dicts
    LOCAL_MODELS,
    GEMINI_MODELS,
    OPENROUTER_MODEL_CONFIGS,
    LMSTUDIO_MODEL_CONFIGS,
    # Per-provider factories
    get_local_client,
    get_gemini_client,
    get_openrouter_client,
    get_lmstudio_client,
    # Unified factory
    get_client,
    get_provider_for,
)

__all__ = [
    # Abstract bases
    "LLMClient",
    "MultiTurnSession",
    # Concrete clients
    "LocalClient",
    "LocalMultiTurnSession",
    "GeminiClient",
    "GeminiMultiTurnSession",
    "OpenRouterClient",
    "OpenRouterMultiTurnSession",
    "LMStudioClient",
    "LMStudioMultiTurnSession",
    # Config registries
    "LOCAL_MODELS",
    "GEMINI_MODELS",
    "OPENROUTER_MODEL_CONFIGS",
    "LMSTUDIO_MODEL_CONFIGS",
    # Factory functions
    "get_local_client",
    "get_gemini_client",
    "get_openrouter_client",
    "get_lmstudio_client",
    "get_client",
    "get_provider_for",
    # Utilities
    "extract_thinking",
]
