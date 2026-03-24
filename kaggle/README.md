# Kaggle Benchmarks Files

Files for the Kaggle Benchmarks platform submission.

## Files

- `task_sb2.py` - Full benchmark task. Paste this into the Kaggle Benchmarks
  task editor ("+ New Task"). Requires the `isaaclandes/lesson-bench` dataset
  to be attached.

- `task_sb2_quick.py` - Smoke test (3 instances, 6 turns, 2 conditions). Use
  this first to verify connectivity before running the full benchmark.

- `benchmark_description.md` - Description text for the benchmark page on Kaggle.

## Dataset

The `lesson` package is uploaded as a Kaggle dataset at
`isaaclandes/lesson-bench`. Attach it to the task notebook via the "Add data"
sidebar. The task code creates a symlink so the package is importable as
`import lesson`.

## Workflow

1. Create a benchmark at https://www.kaggle.com/benchmarks
2. Click "+ New Task", paste `task_sb2_quick.py`, attach the dataset, run it
3. If the smoke test passes, create another task with `task_sb2.py`
4. Add the full task to the benchmark
5. Save version to generate leaderboard results
