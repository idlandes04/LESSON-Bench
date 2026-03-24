# Kaggle Benchmarks Files

Files for the Kaggle Benchmarks platform submission.

## Files

### SB1 (Learning Curves)
- `task_sb1.py` - Full SB1 task. 4 N-values (4, 8, 16, 32) x 25 instances x 5 test items = 500 prompts per model. Single-turn, no feedback.
- `task_sb1_quick.py` - SB1 smoke test (3 instances, 2 N-values). Run this first.

### SB2 (Feedback Learning)
- `task_sb2.py` - Full SB2 task. 4 conditions x 25 instances x 12 turns = 1,200 prompts per model. Multi-turn with feedback.
- `task_sb2_quick.py` - SB2 smoke test (3 instances, 6 turns, 2 conditions). Run this first.

### Other
- `benchmark_description.md` - Description text for the benchmark page on Kaggle.

## Dataset

The `lesson` package is uploaded as a Kaggle dataset at
`isaaclandes/lesson-bench`. Attach it to the task notebook via the "Add data"
sidebar. The task code creates a symlink so the package is importable as
`import lesson`.

## Available Models (27, as of 2026-03-23)

```
anthropic/claude-haiku-4-5@20251001      google/gemini-2.0-flash
anthropic/claude-opus-4-1@20250805       google/gemini-2.0-flash-lite
anthropic/claude-opus-4-5@20251101       google/gemini-2.5-flash
anthropic/claude-opus-4-6@default        google/gemini-2.5-pro
anthropic/claude-sonnet-4-5@20250929     google/gemini-3-flash-preview
anthropic/claude-sonnet-4-6@default      google/gemini-3-pro-preview
anthropic/claude-sonnet-4@20250514       google/gemini-3.1-flash-lite-preview
deepseek-ai/deepseek-r1-0528            google/gemini-3.1-pro-preview
deepseek-ai/deepseek-v3.1               google/gemma-3-1b
deepseek-ai/deepseek-v3.2               google/gemma-3-4b
qwen/qwen3-235b-a22b-instruct-2507      google/gemma-3-12b
qwen/qwen3-coder-480b-a35b-instruct     google/gemma-3-27b
qwen/qwen3-next-80b-a3b-instruct        zai/glm-5
qwen/qwen3-next-80b-a3b-thinking
```

To use a specific model: `kbench.llms["google/gemini-3-flash-preview"]`
Default model (`kbench.llm`) is whichever model the benchmark evaluates.

## Budget ($450 allocation, $50/day, $500/month quota)

### Per-model cost estimates (SB1 + SB2 combined = 1,700 prompts)

| Cost Tier | Models | Est. per model |
|-----------|--------|---------------|
| Low | Gemini Flash, Haiku, DeepSeek V3.x, Gemini 2.5 Flash | $3-8 |
| Medium | Sonnet, Gemini Pro, Qwen 235B, GLM-5, DeepSeek R1 | $15-30 |
| High | Opus models | $60-120 |

### Run plan

| Priority | Models | Count | Est. Total |
|----------|--------|-------|-----------|
| **Tier 1 (must run)** | Gemini 3 Flash/Pro, 3.1 Pro, 2.5 Flash, Claude Haiku/Sonnet 4.5/Opus 4.5, DeepSeek R1/V3.2, GLM-5, Qwen 235B/Coder 480B | 12 | ~$250 |
| **Tier 2 (if budget allows)** | Claude Sonnet 4.6/Opus 4.6, DeepSeek V3.1, Qwen Next Thinking | 4 | ~$120 |
| **Buffer** | Retries, overruns | — | $80 |
| **Total** | | 16 | $450 |

### Skip list (low value for cost)
- Older models (Claude Opus 4.1, Sonnet 4, Gemini 2.0 Flash/Lite) — redundant with newer versions
- Gemma 3 (1B, 4B, 12B, 27B) — too small for symbolic reasoning
- Gemini 2.5 Pro — expensive thinking model, redundant with 3.x Pro
- Gemini 3.1 Flash Lite — likely too weak for SB2

### Execution order
Run cheapest first (Gemini Flash, Haiku, DeepSeek) to validate pipeline on Kaggle, then medium (Sonnet, Pro, Qwen, GLM), then expensive (Opus). This catches bugs before spending on expensive models.

### Combined with OpenRouter (already complete)
Tier 1 + OpenRouter = **15 models across 7 labs** (Google, Anthropic, DeepSeek, ZhipuAI, Alibaba, OpenAI). Tier 2 extends to 19 models.

## Workflow

1. Create a benchmark at https://www.kaggle.com/benchmarks
2. Click "+ New Task", paste `task_sb1_quick.py` into a single cell, attach the dataset, run it
3. If it passes, create another task with `task_sb1.py` (full SB1)
4. Repeat: smoke test `task_sb2_quick.py`, then full `task_sb2.py`
5. Add both full tasks to the benchmark
6. Use "Evaluate More Models" on the Task Detail page to add models beyond the default
7. Save version to generate leaderboard results

## Daily Pacing ($50/day limit)

| Day | Models | Est. Cost |
|-----|--------|-----------|
| 1 | Gemini 3 Flash, 2.5 Flash, Haiku 4.5, DeepSeek V3.2 | ~$20 |
| 2 | DeepSeek R1, GLM-5, Qwen 235B | ~$45 |
| 3 | Gemini 3 Pro, Sonnet 4.5 | ~$40 |
| 4 | Gemini 3.1 Pro, Qwen Coder 480B | ~$45 |
| 5 | Opus 4.5 (most expensive single model) | ~$80* |
| 6-7 | Tier 2 models if budget remains | ~$120 |

*Opus may exceed $50/day — email kaggle-benchmarks-agi-hackathon@google.com for quota increase if needed.
