"""Quick smoke test: send one prompt to each model and show what comes back.

Usage:
    python scripts/smoke_test.py [model_key ...]

If no models specified, tests all broken models from the scan.
"""
from __future__ import annotations

import json
import os
import sys
import traceback

# Ensure .env is loaded
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lesson.models import get_openrouter_client, get_lmstudio_client, OPENROUTER_MODELS, LMSTUDIO_MODELS

# A minimal STS-style prompt
TEST_PROMPT = """You are solving a symbol transformation task.

Vocabulary: ◆ ▲ ⟐ ◈ ⧫ ⬡

Training examples (input → output):
  ◆▲⟐◈⧫ → ⧫◈⟐▲◆
  ⬡◈◆▲⟐ → ⟐▲◆◈⬡
  ▲⧫⬡◆◈ → ◈◆⬡⧫▲

Now predict the output for this input:
  ◆⬡◈⧫▲ → ?

Respond with ONLY a JSON object: {"output": "YOUR_ANSWER"}
"""

# Models that had issues in the scan
DEFAULT_MODELS = [
    "or-gpt5.3-chat",
    "or-gpt5.3-codex",
    "or-glm5",
    "or-minimax-m2.7",
    "or-gemini-3.1-pro",
    "or-kimi-k2.5",
    "or-qwen3.5-397b",
    "or-deepseek-r1",
]


def test_model(model_key: str) -> None:
    print(f"\n{'='*60}")
    print(f"Testing: {model_key}")
    print(f"{'='*60}")

    try:
        if model_key in OPENROUTER_MODELS:
            client = get_openrouter_client(model_key)
        elif model_key in LMSTUDIO_MODELS:
            client = get_lmstudio_client(model_key)
        else:
            raise ValueError(f"Unknown model: {model_key}")
        print(f"  Client created: {client.name}")
    except Exception as e:
        print(f"  FAILED to create client: {e}")
        return

    # Test prompt_json
    print(f"\n  --- prompt_json ---")
    try:
        resp = client.prompt_json(TEST_PROMPT)
        print(f"  Response ({len(resp)} chars): {resp[:500]}")
        if not resp:
            print(f"  WARNING: Empty response!")
    except Exception as e:
        print(f"  EXCEPTION: {type(e).__name__}: {e}")
        traceback.print_exc()

    # Test plain prompt
    print(f"\n  --- prompt (plain) ---")
    try:
        resp = client.prompt(TEST_PROMPT)
        print(f"  Response ({len(resp)} chars): {resp[:500]}")
        if not resp:
            print(f"  WARNING: Empty response!")
    except Exception as e:
        print(f"  EXCEPTION: {type(e).__name__}: {e}")
        traceback.print_exc()


def main():
    models = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_MODELS
    print(f"Smoke testing {len(models)} models...")
    print(f"OPENROUTER_API_KEY set: {bool(os.environ.get('OPENROUTER_API_KEY'))}")

    for model_key in models:
        test_model(model_key)

    print(f"\n{'='*60}")
    print("Done.")


if __name__ == "__main__":
    main()
