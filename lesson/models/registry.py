from __future__ import annotations

"""Model registry for all LESSON benchmark configurations.

Providers:
  - local:      llama-server instances (Qwen, Nemotron, etc.)
  - gemini:     Google Gemini API (Flash, Pro)
  - openrouter: OpenRouter API (GPT, Claude, GLM, DeepSeek, etc.)
  - lmstudio:   LM Studio local server (GGUF models)

Development models (4 configs):
  - qwen3.5-27b-think    — local, thinking ON  (enable_thinking=True)
  - qwen3.5-27b-nothink  — local, thinking OFF (enable_thinking=False)
  - gemini-pro            — API, thinking HIGH
  - gemini-pro-nothink    — API, thinking LOW

Both Qwen variants share ONE llama-server instance on the same port.
Think/nothink is toggled per-request via extra_body.
"""

from typing import Any, Dict, Optional

from lesson.models.gemini import GeminiClient
from lesson.models.local import LocalClient
from lesson.models.openrouter import OpenRouterClient, OPENROUTER_MODEL_CONFIGS
from lesson.models.lmstudio import LMStudioClient, LMSTUDIO_MODEL_CONFIGS

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
