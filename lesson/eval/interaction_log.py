from __future__ import annotations

"""Interaction logger for LESSON evaluation runs.

Records every prompt sent to an LLM and every response received, stored as
JSONL files in logs/. Automatically prunes to keep only the most recent
MAX_LOGS files.

Usage:
    log = InteractionLog("gemini-pro_sb1")
    log.record(prompt="...", raw_response="...", extracted="...", expected="...",
               correct=True, metadata={"tier": 2, "n": 4})
    log.close()  # flushes and writes summary

Each log file is a .jsonl with one JSON object per interaction, plus a
_summary.json with aggregate stats. File naming: YYYYMMDD_HHMMSS_{tag}.jsonl
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


MAX_LOGS = 15
LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"


def _prune_logs(log_dir: Path, max_keep: int = MAX_LOGS) -> None:
    """Remove oldest log files + their summaries, keeping only max_keep most recent."""
    if not log_dir.exists():
        return

    # Find all .jsonl log files (handle race condition where files are
    # deleted between glob() and stat() by concurrent threads)
    def _safe_mtime(p: Path) -> float:
        try:
            return p.stat().st_mtime
        except FileNotFoundError:
            return 0.0

    log_files = sorted(log_dir.glob("*.jsonl"), key=_safe_mtime)
    log_files = [p for p in log_files if p.exists()]

    while len(log_files) > max_keep:
        oldest = log_files.pop(0)
        oldest.unlink(missing_ok=True)
        # Also remove the companion summary
        summary = oldest.with_name(oldest.stem + "_summary.json")
        summary.unlink(missing_ok=True)


class InteractionLog:
    """Records LLM interactions to a JSONL file for debugging.

    Each entry captures the full prompt, raw response, extracted answer,
    expected answer, and whether it was correct — everything needed to
    retroactively debug extraction or model behavior.

    Args:
        tag: Short identifier for the run (e.g., "gemini-pro_sb1").
        log_dir: Directory for log files. Defaults to project_root/logs/.
        max_logs: Maximum number of log files to keep. Oldest are pruned.
    """

    def __init__(
        self,
        tag: str,
        log_dir: Path | str | None = None,
        max_logs: int = MAX_LOGS,
    ) -> None:
        self._log_dir = Path(log_dir) if log_dir else LOG_DIR
        self._log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_tag = tag.replace("/", "_").replace(" ", "_")
        self._filename = f"{timestamp}_{safe_tag}.jsonl"
        self._filepath = self._log_dir / self._filename

        self._entries: List[Dict[str, Any]] = []
        self._file = open(self._filepath, "w", encoding="utf-8")
        self._n_correct = 0
        self._n_total = 0
        self._start_time = time.time()

        # Prune old logs
        _prune_logs(self._log_dir, max_logs)

    @property
    def filepath(self) -> Path:
        return self._filepath

    def record(
        self,
        prompt: str,
        raw_response: str,
        extracted: str,
        expected: str,
        correct: bool,
        metadata: Dict[str, Any] | None = None,
    ) -> None:
        """Record a single LLM interaction.

        Args:
            prompt: The exact text sent to the LLM.
            raw_response: The exact text received from the LLM.
            extracted: The answer extracted from the response.
            expected: The correct/expected answer.
            correct: Whether extracted == expected.
            metadata: Any additional context (tier, n, condition, turn, etc.).
        """
        entry = {
            "idx": self._n_total,
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "raw_response": raw_response,
            "extracted_answer": extracted,
            "expected_answer": expected,
            "correct": correct,
        }
        if metadata:
            entry["metadata"] = metadata

        self._file.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._file.flush()

        self._n_total += 1
        if correct:
            self._n_correct += 1

    def close(self) -> None:
        """Flush the log file and write a companion summary."""
        self._file.close()

        elapsed = time.time() - self._start_time
        accuracy = self._n_correct / self._n_total if self._n_total > 0 else 0.0

        summary = {
            "log_file": self._filename,
            "total_interactions": self._n_total,
            "correct": self._n_correct,
            "accuracy": round(accuracy, 4),
            "elapsed_seconds": round(elapsed, 1),
            "timestamp": datetime.now().isoformat(),
        }

        summary_path = self._filepath.with_name(self._filepath.stem + "_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"  Log saved: {self._filepath} ({self._n_total} interactions, {accuracy:.0%} accuracy)")

    def __enter__(self) -> "InteractionLog":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
