from __future__ import annotations

"""Model abstraction layer for LESSON benchmarks.

Provides a unified interface (LLMClient / MultiTurnSession) for calling both
local llama-server models and the Gemini API.

Quick-start
-----------
Local model::

    from lesson.models import get_local_client
    client = get_local_client("qwen3.5-27b-think")
    print(client.prompt("What is 2+2?"))

Gemini model::

    from lesson.models import get_gemini_client
    client = get_gemini_client("gemini-flash")
    print(client.prompt("What is 2+2?"))

Multi-turn session::

    session = client.multi_turn()
    session.send("Start a count from 1.")
    session.inject("1, 2, 3")   # SB2 feedback injection
    reply = session.send("Continue.")
"""

from lesson.models.base import LLMClient, MultiTurnSession
from lesson.models.gemini import GeminiClient, GeminiMultiTurnSession
from lesson.models.local import LocalClient, LocalMultiTurnSession, extract_thinking
from lesson.models.registry import (
    GEMINI_MODELS,
    LOCAL_MODELS,
    get_gemini_client,
    get_local_client,
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
    # Registry
    "LOCAL_MODELS",
    "GEMINI_MODELS",
    "get_local_client",
    "get_gemini_client",
    # Utilities
    "extract_thinking",
]
