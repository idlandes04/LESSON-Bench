"""Tests for lesson/eval/runner.py — parse_model_list and save_incremental."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import threading
import pytest
from lesson.eval.runner import parse_model_list, save_incremental


# ---------------------------------------------------------------------------
# parse_model_list
# ---------------------------------------------------------------------------

def test_parse_model_list_none_returns_defaults():
    defaults = [("openrouter", "gpt-5.3-codex"), ("gemini", "gemini-flash")]
    result = parse_model_list(None, default_models=defaults)
    assert result == defaults


def test_parse_model_list_none_no_defaults_returns_empty():
    result = parse_model_list(None, default_models=None)
    assert result == []


def test_parse_model_list_known_openrouter_model():
    result = parse_model_list("gpt-5.3-codex")
    assert len(result) == 1
    provider, name = result[0]
    assert provider == "openrouter"
    assert name == "gpt-5.3-codex"


def test_parse_model_list_known_gemini_model():
    result = parse_model_list("gemini-flash")
    assert len(result) == 1
    provider, name = result[0]
    assert provider == "gemini"
    assert name == "gemini-flash"


def test_parse_model_list_known_lmstudio_model():
    result = parse_model_list("lm-nemotron-nano")
    assert len(result) == 1
    provider, name = result[0]
    assert provider == "lmstudio"
    assert name == "lm-nemotron-nano"


def test_parse_model_list_csv_multiple_models():
    result = parse_model_list("gpt-5.3-codex,claude-sonnet-4.6")
    assert len(result) == 2
    names = [n for _, n in result]
    assert "gpt-5.3-codex" in names
    assert "claude-sonnet-4.6" in names


def test_parse_model_list_csv_with_spaces_stripped():
    result = parse_model_list(" gpt-5.3-codex , claude-sonnet-4.6 ")
    assert len(result) == 2
    names = [n for _, n in result]
    assert "gpt-5.3-codex" in names
    assert "claude-sonnet-4.6" in names


def test_parse_model_list_unknown_gemini_prefix_fallback():
    result = parse_model_list("gemini-experimental-xyz")
    assert len(result) == 1
    provider, name = result[0]
    assert provider == "gemini"
    assert name == "gemini-experimental-xyz"


def test_parse_model_list_unknown_lm_prefix_fallback():
    result = parse_model_list("lm-my-custom-model")
    assert len(result) == 1
    provider, name = result[0]
    assert provider == "lmstudio"
    assert name == "lm-my-custom-model"


def test_parse_model_list_unknown_model_falls_back_to_openrouter():
    # Completely unknown name → openrouter fallback
    result = parse_model_list("totally-unknown-model-xyz")
    assert len(result) == 1
    provider, name = result[0]
    assert provider == "openrouter"
    assert name == "totally-unknown-model-xyz"


def test_parse_model_list_empty_csv_returns_empty():
    result = parse_model_list("")
    assert result == []


def test_parse_model_list_csv_filters_empty_segments():
    # Trailing comma should not produce an empty entry
    result = parse_model_list("gpt-5.3-codex,")
    assert len(result) == 1
    assert result[0][1] == "gpt-5.3-codex"


# ---------------------------------------------------------------------------
# save_incremental
# ---------------------------------------------------------------------------

def test_save_incremental_creates_file(tmp_path):
    data = {"model": "test-model", "accuracy": 0.75}
    path = save_incremental(tmp_path, "test-model", "results", data)
    assert path.exists()


def test_save_incremental_correct_filename(tmp_path):
    data = {"x": 1}
    path = save_incremental(tmp_path, "my-model", "sb2", data)
    assert path.name == "my-model_sb2.json"


def test_save_incremental_valid_json_content(tmp_path):
    data = {"accuracy": 0.85, "turns": [0.5, 0.75, 1.0]}
    path = save_incremental(tmp_path, "model-a", "run01", data)
    loaded = json.loads(path.read_text())
    assert loaded["accuracy"] == pytest.approx(0.85)
    assert loaded["turns"] == [0.5, 0.75, 1.0]


def test_save_incremental_slashes_in_name_replaced(tmp_path):
    data = {"val": 42}
    path = save_incremental(tmp_path, "provider/model-name", "label", data)
    assert "/" not in path.name
    assert path.name == "provider_model-name_label.json"


def test_save_incremental_colons_in_name_replaced(tmp_path):
    data = {}
    path = save_incremental(tmp_path, "anthropic:claude", "test", data)
    assert ":" not in path.name
    assert path.name == "anthropic_claude_test.json"


def test_save_incremental_returns_path_object(tmp_path):
    path = save_incremental(tmp_path, "model", "label", {"k": "v"})
    assert isinstance(path, Path)


def test_save_incremental_with_lock_is_thread_safe(tmp_path):
    lock = threading.Lock()
    errors = []

    def _write(i):
        try:
            save_incremental(tmp_path, f"model_{i}", "run", {"i": i}, lock=lock)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=_write, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"Thread-safety errors: {errors}"
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 10


def test_save_incremental_overwrites_existing_file(tmp_path):
    path = save_incremental(tmp_path, "model", "label", {"v": 1})
    path = save_incremental(tmp_path, "model", "label", {"v": 2})
    loaded = json.loads(path.read_text())
    assert loaded["v"] == 2
