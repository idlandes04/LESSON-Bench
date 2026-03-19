from __future__ import annotations

"""Abstract base classes for LLM clients used in LESSON benchmarks."""

from abc import ABC, abstractmethod


class MultiTurnSession(ABC):
    """A stateful multi-turn conversation with an LLM."""

    @abstractmethod
    def send(self, text: str, role: str = "user") -> str:
        """Send a message, get response. Maintains conversation history."""

    @abstractmethod
    def send_json(self, text: str, role: str = "user") -> str:
        """Send a message requesting JSON output. Maintains conversation history.

        The prompt should instruct the model to respond with {"output": "..."}.
        Returns the raw response text (caller parses JSON).
        """

    @abstractmethod
    def inject(self, text: str, role: str = "assistant") -> None:
        """Inject a message into history without generating a response.

        Used for feedback injection in SB2 (Symbolic Benchmark 2) so the model
        sees corrective feedback as a prior assistant turn.
        """

    @abstractmethod
    def reset(self) -> None:
        """Clear conversation history."""


class LLMClient(ABC):
    """Unified interface for calling LLMs (local llama-server or Gemini API)."""

    name: str  # Human-readable model name

    @abstractmethod
    def prompt(self, text: str) -> str:
        """Single-turn: send text, get response."""

    @abstractmethod
    def prompt_json(self, text: str) -> str:
        """Single-turn with JSON mode. Returns raw JSON string."""

    @abstractmethod
    def multi_turn(self) -> MultiTurnSession:
        """Create a new multi-turn conversation session."""
