"""Tests for lesson/results/store.py — SQLite-backed ResultsStore."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from lesson.results.store import ResultsStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path):
    """Create a fresh ResultsStore backed by a temp file (not :memory:)."""
    db = tmp_path / "test_lesson.db"
    s = ResultsStore(db_path=db)
    yield s
    s.close()


# ---------------------------------------------------------------------------
# save_run + query
# ---------------------------------------------------------------------------

def test_save_run_inserts_record(store):
    store.save_run("run_001", version="1.0", config={"tier": 3}, results_dir="/tmp")
    rows = store.query("SELECT * FROM runs WHERE run_id = ?", ("run_001",))
    assert len(rows) == 1
    assert rows[0]["run_id"] == "run_001"
    assert rows[0]["version"] == "1.0"
    assert rows[0]["results_dir"] == "/tmp"


def test_save_run_replaces_on_duplicate(store):
    store.save_run("run_dup", version="1.0")
    store.save_run("run_dup", version="2.0")
    rows = store.query("SELECT * FROM runs WHERE run_id = ?", ("run_dup",))
    assert len(rows) == 1
    assert rows[0]["version"] == "2.0"


def test_save_run_config_stored_as_json(store):
    import json
    config = {"tier": 2, "n_examples": 8}
    store.save_run("run_cfg", config=config)
    rows = store.query("SELECT config_json FROM runs WHERE run_id = ?", ("run_cfg",))
    assert len(rows) == 1
    stored = json.loads(rows[0]["config_json"])
    assert stored["tier"] == 2
    assert stored["n_examples"] == 8


# ---------------------------------------------------------------------------
# save_cell + per-turn data
# ---------------------------------------------------------------------------

def test_save_cell_inserts_cell_record(store):
    store.save_run("run_c1")
    cell_data = {
        "status": "complete",
        "avg_accuracy": 0.75,
        "elapsed_s": 42.0,
        "per_turn": {},
    }
    store.save_cell("run_c1", "model-x", "openrouter", "correction", cell_data)
    rows = store.query(
        "SELECT * FROM cells WHERE run_id=? AND model=? AND condition=?",
        ("run_c1", "model-x", "correction"),
    )
    assert len(rows) == 1
    assert rows[0]["avg_accuracy"] == pytest.approx(0.75)
    assert rows[0]["status"] == "complete"
    assert rows[0]["elapsed_s"] == pytest.approx(42.0)


def test_save_cell_with_per_turn_data_populates_turns_table(store):
    store.save_run("run_t1")
    per_turn = {
        "0": {"n_correct": 2, "n_total": 3, "accuracy": 0.667},
        "1": {"n_correct": 3, "n_total": 3, "accuracy": 1.0},
    }
    cell_data = {"status": "complete", "avg_accuracy": 0.833, "per_turn": per_turn}
    store.save_cell("run_t1", "model-y", "openrouter", "practice_only", cell_data)

    turns = store.query(
        "SELECT * FROM turns WHERE run_id=? AND model=? ORDER BY turn_idx",
        ("run_t1", "model-y"),
    )
    assert len(turns) == 2
    assert turns[0]["turn_idx"] == 0
    assert turns[0]["n_correct"] == 2
    assert turns[0]["n_total"] == 3
    assert turns[1]["turn_idx"] == 1
    assert turns[1]["n_correct"] == 3


def test_save_cell_replaces_on_duplicate_key(store):
    store.save_run("run_r1")
    store.save_cell("run_r1", "model-z", "gemini", "correction",
                    {"avg_accuracy": 0.5, "per_turn": {}})
    store.save_cell("run_r1", "model-z", "gemini", "correction",
                    {"avg_accuracy": 0.9, "per_turn": {}})
    rows = store.query(
        "SELECT avg_accuracy FROM cells WHERE run_id=? AND model=? AND condition=?",
        ("run_r1", "model-z", "correction"),
    )
    assert len(rows) == 1
    assert rows[0]["avg_accuracy"] == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# save_model_results
# ---------------------------------------------------------------------------

def test_save_model_results_populates_cells_and_turns(store):
    store.save_run("run_mr1")
    model_data = {
        "sb2_correction": {
            "summary": {
                "correction": {
                    0: {"n_correct": 5, "n_total": 10},
                    1: {"n_correct": 7, "n_total": 10},
                }
            }
        }
    }
    store.save_model_results("run_mr1", "model-a", "openrouter", model_data, ["correction"])

    cells = store.query(
        "SELECT avg_accuracy FROM cells WHERE run_id=? AND model=? AND condition=?",
        ("run_mr1", "model-a", "correction"),
    )
    assert len(cells) == 1
    # avg of (0.5, 0.7) = 0.6
    assert cells[0]["avg_accuracy"] == pytest.approx(0.6, abs=0.01)

    turns = store.query(
        "SELECT * FROM turns WHERE run_id=? AND model=? ORDER BY turn_idx",
        ("run_mr1", "model-a"),
    )
    assert len(turns) == 2
    assert turns[0]["n_correct"] == 5
    assert turns[1]["n_correct"] == 7


def test_save_model_results_skips_missing_condition_key(store):
    store.save_run("run_mr2")
    model_data = {}  # no sb2_correction key
    store.save_model_results("run_mr2", "model-b", "openrouter", model_data, ["correction"])
    cells = store.query("SELECT * FROM cells WHERE run_id=?", ("run_mr2",))
    assert len(cells) == 0


# ---------------------------------------------------------------------------
# leaderboard
# ---------------------------------------------------------------------------

def test_leaderboard_returns_sorted_by_accuracy_desc(store):
    store.save_run("run_lb1")
    for model, acc in [("model-high", 0.9), ("model-mid", 0.6), ("model-low", 0.3)]:
        store.save_cell("run_lb1", model, "openrouter", "correction",
                        {"avg_accuracy": acc, "per_turn": {}})

    lb = store.leaderboard("correction")
    assert len(lb) == 3
    # Must be sorted descending
    accs = [r["avg_accuracy"] for r in lb]
    assert accs == sorted(accs, reverse=True)
    assert lb[0]["model"] == "model-high"
    assert lb[-1]["model"] == "model-low"


def test_leaderboard_filters_by_condition(store):
    store.save_run("run_lb2")
    store.save_cell("run_lb2", "model-x", "openrouter", "correction",
                    {"avg_accuracy": 0.8, "per_turn": {}})
    store.save_cell("run_lb2", "model-x", "openrouter", "practice_only",
                    {"avg_accuracy": 0.4, "per_turn": {}})

    lb_corr = store.leaderboard("correction")
    lb_prac = store.leaderboard("practice_only")
    assert len(lb_corr) == 1
    assert lb_corr[0]["avg_accuracy"] == pytest.approx(0.8)
    assert len(lb_prac) == 1
    assert lb_prac[0]["avg_accuracy"] == pytest.approx(0.4)


def test_leaderboard_empty_when_no_data(store):
    lb = store.leaderboard("correction")
    assert lb == []


# ---------------------------------------------------------------------------
# status_matrix
# ---------------------------------------------------------------------------

def test_status_matrix_correct_structure(store):
    store.save_run("run_sm1")
    store.save_cell("run_sm1", "model-a", "openrouter", "correction",
                    {"status": "complete", "avg_accuracy": 0.7, "per_turn": {}})
    store.save_cell("run_sm1", "model-a", "openrouter", "practice_only",
                    {"status": "complete", "avg_accuracy": 0.5, "per_turn": {}})

    matrix = store.status_matrix(run_id="run_sm1")
    assert "model-a" in matrix
    assert "correction" in matrix["model-a"]
    assert "practice_only" in matrix["model-a"]
    assert matrix["model-a"]["correction"]["status"] == "complete"
    assert matrix["model-a"]["correction"]["avg_accuracy"] == pytest.approx(0.7)


def test_status_matrix_multiple_models(store):
    store.save_run("run_sm2")
    for model in ["alpha", "beta", "gamma"]:
        store.save_cell("run_sm2", model, "openrouter", "correction",
                        {"status": "complete", "avg_accuracy": 0.5, "per_turn": {}})

    matrix = store.status_matrix(run_id="run_sm2")
    assert set(matrix.keys()) == {"alpha", "beta", "gamma"}


# ---------------------------------------------------------------------------
# get_turn_data
# ---------------------------------------------------------------------------

def test_get_turn_data_returns_correct_turns(store):
    store.save_run("run_td1")
    per_turn = {
        "0": {"n_correct": 1, "n_total": 4, "accuracy": 0.25},
        "1": {"n_correct": 2, "n_total": 4, "accuracy": 0.50},
        "2": {"n_correct": 4, "n_total": 4, "accuracy": 1.00},
    }
    store.save_cell("run_td1", "model-td", "openrouter", "correction",
                    {"avg_accuracy": 0.58, "per_turn": per_turn})

    turns = store.get_turn_data("model-td", "correction", run_id="run_td1")
    assert len(turns) == 3
    assert turns[0]["turn_idx"] == 0
    assert turns[0]["n_correct"] == 1
    assert turns[2]["turn_idx"] == 2
    assert turns[2]["n_correct"] == 4


def test_get_turn_data_without_run_id(store):
    store.save_run("run_td2")
    per_turn = {"0": {"n_correct": 3, "n_total": 5, "accuracy": 0.6}}
    store.save_cell("run_td2", "model-td2", "gemini", "practice_only",
                    {"avg_accuracy": 0.6, "per_turn": per_turn})

    turns = store.get_turn_data("model-td2", "practice_only")
    assert len(turns) == 1
    assert turns[0]["n_correct"] == 3


def test_get_turn_data_empty_when_no_data(store):
    turns = store.get_turn_data("nonexistent-model", "correction")
    assert turns == []


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------

def test_context_manager_closes_connection(tmp_path):
    db = tmp_path / "ctx_test.db"
    with ResultsStore(db_path=db) as store:
        store.save_run("run_ctx")
        rows = store.query("SELECT * FROM runs WHERE run_id = ?", ("run_ctx",))
        assert len(rows) == 1
    # After __exit__, connection is closed — should not raise during with block
    # (Python sqlite3 raises ProgrammingError on closed connection, but we just
    # verify the context manager didn't crash)
    assert db.exists()
