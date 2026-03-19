from __future__ import annotations

"""Central model registry for all LESSON benchmark configurations.

This is THE single source of truth for model configs across all providers.
All model configuration dicts and factory functions live here.

Providers:
  - openrouter: OpenRouter API (GPT, Claude, GLM, DeepSeek, etc.)
  - gemini:     Google Gemini API (Flash, Pro)
  - lmstudio:   LM Studio local server (GGUF models)
  - local:      llama-server instances (Qwen, Nemotron, etc.)
"""

import os
from typing import Any, Dict, Optional

from lesson.models.base import LLMClient
from lesson.models.gemini import GeminiClient
from lesson.models.local import LocalClient
from lesson.models.openrouter import OpenRouterClient
from lesson.models.lmstudio import LMStudioClient


# ---------------------------------------------------------------------------
# OpenRouter model configurations
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


# ---------------------------------------------------------------------------
# LM Studio model configurations
# ---------------------------------------------------------------------------
# Model IDs must match what LM Studio reports via GET /v1/models.
# ---------------------------------------------------------------------------

LMSTUDIO_MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "lm-nemotron-nano": {
        "model_id": "nvidia/nemotron-3-nano",
        "max_tokens": 2048,
        "port": 1234,
    },
    "lm-qwen3.5-35b-a3b": {
        "model_id": "qwen/qwen3.5-35b-a3b",
        "max_tokens": 2048,
        "port": 1234,
    },
    "lm-qwen3.5-27b": {
        "model_id": "qwen3.5-27b",
        "max_tokens": 2048,
        "port": 1234,
    },
    "lm-glm-4.7-flash": {
        "model_id": "zai-org/glm-4.7-flash",
        "max_tokens": 2048,
        "port": 1234,
    },
}


# ---------------------------------------------------------------------------
# Local llama-server model configurations
# ---------------------------------------------------------------------------
# "enable_thinking" controls Qwen3.5 reasoning mode per-request.
# Server must be launched with: --jinja --reasoning-format deepseek
# Both think/nothink variants share the same server (same port).
# ---------------------------------------------------------------------------

LOCAL_MODELS: Dict[str, Dict[str, Any]] = {
    # --- Development models (share ONE server on port 8080) ---
    "qwen3.5-27b-think": {
        "port": 8080,
        "max_tokens": 8192,  # thinking uses thousands of tokens on STS tasks
        "strip_thinking": False,
        "enable_thinking": True,
    },
    "qwen3.5-27b-nothink": {
        "port": 8080,
        "max_tokens": 512,
        "strip_thinking": True,  # safety net for any residual <think> blocks
        "enable_thinking": False,
    },
    # --- Additional models for final eval (require separate server instances) ---
    "qwen3.5-35b-a3b": {
        "port": 8082,
        "max_tokens": 8192,  # STS tasks trigger 1-4K+ reasoning tokens
        "strip_thinking": False,
        "enable_thinking": True,
    },
    "qwen3.5-35b-a3b-nothink": {
        "port": 8082,
        "max_tokens": 512,
        "strip_thinking": True,  # safety net
        "enable_thinking": False,
    },
    "gemma3-27b": {
        "port": 8083,
        "max_tokens": 512,
        "strip_thinking": False,
    },
    "phi4-14b": {
        "port": 8084,
        "max_tokens": 512,
        "strip_thinking": False,
    },
    # --- Nemotron Nano (MLX model via mlx_lm.server) ---
    # Nemotron uses Qwen-style enable_thinking in its chat template
    "nemotron-nano": {
        "port": 8085,
        "max_tokens": 8192,  # Nemotron reasoning uses many tokens on STS
        "strip_thinking": False,
        "enable_thinking": True,
        "model_id": "default_model",  # mlx_lm.server uses "default_model"
    },
    "nemotron-nano-nothink": {
        "port": 8085,
        "max_tokens": 512,
        "strip_thinking": True,  # safety net
        "enable_thinking": False,
        "model_id": "default_model",
    },
}


# ---------------------------------------------------------------------------
# Gemini API model configurations
# ---------------------------------------------------------------------------
# thinking_level: "LOW" (cheapest), "MEDIUM" (daily driver), "HIGH" (deep)
# Gemini 3.1 Pro cannot fully disable thinking — LOW is the minimum.
# Default is HIGH if not set, so always specify explicitly.
# ---------------------------------------------------------------------------

GEMINI_MODELS: Dict[str, Dict[str, Any]] = {
    "gemini-flash": {
        "model_id": "gemini-3-flash-preview",
        "max_tokens": 2048,
        "thinking_level": "MEDIUM",
    },
    # --- Development models ---
    "gemini-pro": {
        "model_id": "gemini-3.1-pro-preview",
        "max_tokens": 2048,
        "thinking_level": "HIGH",
    },
    "gemini-pro-nothink": {
        "model_id": "gemini-3.1-pro-preview",
        "max_tokens": 2048,  # LOW thinking still uses ~1k tokens; need headroom for JSON output
        "thinking_level": "LOW",
    },
}

# ---------------------------------------------------------------------------
# Note: two additional models are run on Kaggle infrastructure only:
#   - claude-sonnet-4   (Anthropic Claude Sonnet 4, via Kaggle SDK)
#   - llama3.1-70b      (Meta Llama 3.1 70B Instruct, via Kaggle SDK)
# These are not listed here because they require Kaggle-specific auth and
# are instantiated directly inside the Kaggle notebook environment.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Factory functions (per-provider)
# ---------------------------------------------------------------------------

def get_local_client(name: str, **overrides: Any) -> LocalClient:
    """Instantiate a LocalClient by registry name."""
    if name not in LOCAL_MODELS:
        raise KeyError(
            f"Unknown local model {name!r}. Available: {sorted(LOCAL_MODELS)}"
        )
    config = {**LOCAL_MODELS[name], **overrides}
    return LocalClient(name=name, **config)


def get_gemini_client(
    name: str, api_key: Optional[str] = None, **overrides: Any
) -> GeminiClient:
    """Instantiate a GeminiClient by registry name."""
    if name not in GEMINI_MODELS:
        raise KeyError(
            f"Unknown Gemini model {name!r}. Available: {sorted(GEMINI_MODELS)}"
        )
    config = {**GEMINI_MODELS[name], **overrides}
    return GeminiClient(name=name, api_key=api_key, **config)


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


def get_lmstudio_client(name: str, **overrides: Any) -> LMStudioClient:
    """Instantiate a LMStudioClient by registry name."""
    if name not in LMSTUDIO_MODEL_CONFIGS:
        raise KeyError(
            f"Unknown LM Studio model {name!r}. "
            f"Available: {sorted(LMSTUDIO_MODEL_CONFIGS)}"
        )
    config = {**LMSTUDIO_MODEL_CONFIGS[name], **overrides}
    return LMStudioClient(name=name, **config)


# ---------------------------------------------------------------------------
# Unified factory
# ---------------------------------------------------------------------------

def get_provider_for(name: str) -> str:
    """Auto-detect provider from model name by scanning all registries."""
    if name in OPENROUTER_MODEL_CONFIGS:
        return "openrouter"
    if name in GEMINI_MODELS:
        return "gemini"
    if name in LMSTUDIO_MODEL_CONFIGS:
        return "lmstudio"
    if name in LOCAL_MODELS:
        return "local"
    raise KeyError(
        f"Model {name!r} not found in any registry. "
        f"Available: openrouter={sorted(OPENROUTER_MODEL_CONFIGS)}, "
        f"gemini={sorted(GEMINI_MODELS)}, "
        f"lmstudio={sorted(LMSTUDIO_MODEL_CONFIGS)}, "
        f"local={sorted(LOCAL_MODELS)}"
    )


def get_client(provider: str, name: str, **kwargs: Any) -> LLMClient:
    """Unified factory: dispatches to the right provider client.

    Args:
        provider: One of "openrouter", "gemini", "lmstudio", "local".
        name: Model registry name (e.g. "gpt-5.3-codex", "gemini-flash").
        **kwargs: Passed to the provider-specific factory (e.g. api_key, timeout).
    """
    dispatch = {
        "openrouter": get_openrouter_client,
        "gemini": get_gemini_client,
        "lmstudio": get_lmstudio_client,
        "local": get_local_client,
    }
    if provider not in dispatch:
        raise ValueError(
            f"Unknown provider {provider!r}. Valid: {sorted(dispatch)}"
        )
    return dispatch[provider](name, **kwargs)
