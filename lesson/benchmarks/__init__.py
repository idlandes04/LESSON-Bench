"""Benchmark runner entry points.

Usage::

    from lesson.benchmarks import run_sb1_pilot, run_sb2_pilot

    # Kaggle Benchmarks SDK (requires kaggle-benchmarks package):
    from lesson.benchmarks.kaggle_task import lesson_bench_sb2
"""

from lesson.eval.pilot import run_sb1_pilot
from lesson.eval.sb2_pilot import run_sb2_pilot

__all__ = ["run_sb1_pilot", "run_sb2_pilot"]

# Lazy import for Kaggle task (kaggle-benchmarks may not be installed)
def __getattr__(name):
    if name in ("lesson_bench_sb2", "lesson_sb2_cell", "lesson_bench_sb2_quick"):
        from lesson.benchmarks.kaggle_task import lesson_bench_sb2, lesson_sb2_cell, lesson_bench_sb2_quick
        return {"lesson_bench_sb2": lesson_bench_sb2, "lesson_sb2_cell": lesson_sb2_cell, "lesson_bench_sb2_quick": lesson_bench_sb2_quick}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
