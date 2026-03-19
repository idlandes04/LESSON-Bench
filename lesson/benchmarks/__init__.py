"""Benchmark runner entry points.

Usage::

    from lesson.benchmarks import run_sb1_pilot, run_sb2_pilot
"""

from lesson.eval.pilot import run_sb1_pilot
from lesson.eval.sb2_pilot import run_sb2_pilot

__all__ = ["run_sb1_pilot", "run_sb2_pilot"]
