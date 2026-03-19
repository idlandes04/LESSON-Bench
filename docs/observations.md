# LESSON Observations & Lessons Learned
## Living document — updated as we test and iterate

---

## STS Generator Observations

### 2026-03-18: Training Set Quality Fix
- **Problem**: Random training examples often showed identity transformations (input = output), especially at Tier 1 where DIRECT rules have specific trigger patterns. 3/4 training examples could be uninformative.
- **Fix**: Added "informative example" generation — the first examples in every training set are constructed to exercise each rule, ensuring the model sees actual transformations.
- **Result**: 0% identity examples at all tiers. Models now see meaningful patterns from the first examples.

### 2026-03-18: Type E Feasibility Rates
- After tuning alphabet sizes and guaranteeing POSITIONAL rules at Tier 3+:
  - Tier 2: 82% (target >90% — slightly below, monitor)
  - Tier 3: 90% (on target)
  - Tier 4: 100%
- Key insight: Larger alphabets (8+ symbols) improve Type E feasibility because it's harder for all mapped symbols to appear in N=8 training examples.

---

## Cross-Model Results (2026-03-18 Evening Run)

### Models Tested
| Model | Type | Thinking | Active Params | Notes |
|-------|------|----------|--------------|-------|
| Gemini 3 Flash | API | MEDIUM | ~unknown | Best performer |
| Qwen3.5-35B-A3B | Local (llama.cpp) | OFF | 3B (MoE) | Thinking mode unusable (infinite loop) |
| Nemotron Nano 30B-A3B | Local (mlx_lm) | OFF | 3B (MoE) | Thinking mode unusable (infinite loop) |

### SB1: Learning Curves (Type R Accuracy)

| Model | T1 N=4 | T1 N=8 | T2 N=4 | T2 N=8 | T3 N=4 | T3 N=8 |
|-------|--------|--------|--------|--------|--------|--------|
| **Gemini Flash** | 87% (13/15) | 87% (13/15) | 25% (3/12) | 50% (6/12) | 11% (1/9) | 11% (1/9) |
| **Qwen3.5-35B-A3B** | 40% (6/15) | 47% (7/15) | 8% (1/12) | 8% (1/12) | 11% (1/9) | 0% (0/9) |
| **Nemotron Nano** | 0% (0/15) | 0% (0/15) | 0% (0/12) | 0% (0/12) | 0% (0/9) | 0% (0/9) |

### SB1: Type E/L Accuracy

| Model | T2 N=4 E | T2 N=8 E | T3 N=4 E | T3 N=8 E | T3 N=4 L | T3 N=8 L |
|-------|----------|----------|----------|----------|----------|----------|
| **Gemini Flash** | 0% (0/3) | 33% (1/3) | 33% (1/3) | 33% (1/3) | 33% (1/3) | 0% (0/3) |
| **Qwen3.5-35B-A3B** | 0% (0/3) | 0% (0/3) | 33% (1/3) | 0% (0/3) | 33% (1/3) | 0% (0/3) |
| **Nemotron Nano** | 0% (0/3) | 0% (0/3) | 0% (0/3) | 0% (0/3) | 33% (1/3) | 0% (0/3) |

### SB2: Feedback Conditions (T2, N=8, 8 turns, 3 instances)

#### Gemini 3 Flash (MEDIUM thinking)

| Turn | correction | practice_only | error_only |
|------|-----------|--------------|------------|
| 0 | 33% | 33% | 33% |
| 1 | 0% | 33% | 67% |
| 2 | 33% | 33% | 67% |
| 3 | 33% | 33% | 33% |
| 4 | 67% | 67% | 33% |
| 5 | 33% | 100% | 67% |
| 6 | 33% | 0% | 0% |
| 7 | 67% | 33% | 33% |
| **Avg** | **38%** | **42%** | **42%** |

#### Qwen3.5-35B-A3B (nothink)

| Turn | correction | practice_only | error_only |
|------|-----------|--------------|------------|
| 0 | 0% | 0% | 0% |
| 1 | 67% | 33% | 33% |
| 2 | 0% | 0% | 0% |
| 3 | 0% | 0% | 0% |
| 4 | 33% | 0% | 0% |
| 5 | 0% | 0% | 0% |
| 6 | 0% | 0% | 0% |
| 7 | 0% | 33% | 0% |
| **Avg** | **12%** | **8%** | **4%** |

#### Nemotron Nano (nothink)
- **0% across all conditions, all turns.** Complete floor effect.

---

## Key Findings

### 1. Difficulty Gradient Works
The benchmark successfully discriminates across three levels:
- **Gemini Flash** (API, thinking): T1=87%, T2=38%, T3=11%
- **Qwen 35B-A3B** (local, nothink): T1=44%, T2=8%, T3=5%
- **Nemotron Nano** (local, nothink): 0% everywhere

### 2. Thinking Mode Creates Infinite Reasoning Loops on STS
Both local MoE models (3B active params) get stuck in unbounded reasoning when thinking is enabled:
- Qwen3.5-35B-A3B: 4096+ reasoning tokens without concluding, content always empty
- Nemotron Nano: Same behavior — 13K+ chars of reasoning, never produces answer
- These models try to inductively figure out the STS rules from examples, get stuck analyzing edge cases, and exhaust the token budget
- **Implication**: MoE models with 3B active params lack the reasoning capacity for in-context rule induction. They need to be tested in nothink mode.
- **Interesting contrast**: Gemini Flash (with MEDIUM thinking) succeeds because it has much more compute per token and its thinking is bounded

### 3. FLR ≈ 0 Confirmed Across Models
For Gemini Flash (the only model with enough accuracy to measure):
- correction: 38%, practice_only: 42%, error_only: 42%
- All three conditions converge to ~40% accuracy
- **The model doesn't benefit from any specific feedback content** — it improves equally whether told the correct answer, just shown examples, or given bare correct/incorrect signals
- The learning comes from accumulated context (more examples in conversation), not from feedback

For Qwen (lower accuracy):
- correction: 12%, practice_only: 8%, error_only: 4%
- Too close to floor for meaningful FLR calculation, but directionally consistent

### 4. N=8 Helps at T1-T2 for Flash but Not for Weaker Models
- Gemini Flash: T2 N=4→N=8 doubles accuracy (25%→50%)
- Qwen3.5-35B-A3B: T2 stays at 8% regardless of N — more examples don't help
- **Interpretation**: Weaker models can't leverage additional examples because they lack the capacity to induce rules from any number of examples. The N effect is a measure of learning capacity itself.

### 5. 25-Instance Dataset Ready for Production
- 25 STS instances at Tier 2 generated and validated
- 100% Type E feasibility (all instances have extrapolation items)
- 0% Type L items (expected at Tier 2 — no exceptions, simple rules)
- 12 SB2 turns per instance (all achieved target)
- 125 total SB1 test items (100R + 25E)

---

## Thinking Mode Observations

### MoE Models with 3B Active Params (Qwen3.5-35B-A3B, Nemotron Nano)
- **With thinking ON**: Model enters analysis paralysis on STS tasks
  - Burns 4096+ tokens on reasoning
  - Tries to inductively figure out transformation rules from examples
  - Gets stuck in hypothesis-revision loops ("Wait, maybe the rule is...", "Actually, let me reconsider...")
  - Never concludes and produces an answer
  - finish_reason always "length" (hit token limit)
- **With thinking OFF**: Model produces answers quickly (~16 tokens)
  - Answers are usually wrong (0-47% depending on tier)
  - But extraction pipeline works cleanly
  - Practical for benchmarking

### Gemini Flash (MEDIUM thinking, API)
- Thinking works correctly — model reasons briefly and produces answers
- 87% accuracy at T1 shows genuine rule induction capability
- Thinking tokens are bounded (doesn't exhaust budget)

---

## Model Behavior Observations

### Qwen3.5-35B-A3B (nothink)
- Produces valid JSON responses: `{"output": "⧫⟐⧫⟐"}`
- Fast inference (~0.7s per item without thinking)
- Answers are plausible (valid symbols, correct length) but wrong
- T1 accuracy (40-47%) suggests partial pattern matching — it picks up simple direct replacements sometimes but misses multi-rule interactions
- T2+ accuracy near floor (8%) — can't handle positional rules

### Nemotron Nano (nothink, MLX)
- Hybrid Mamba-Transformer architecture (NemotronH)
- Served via mlx_lm.server on MLX (Apple Silicon native)
- Produces valid JSON but answers are nearly random
- Complete failure on STS suggests architecture is not suited for in-context rule induction
- The `reasoning` field (not `reasoning_content`) requires special handling in extraction code

---

## Answer Extraction Observations

### 2026-03-18: Initial testing
- Regex extraction handles: "Output: X", "<think>...</think>Output: X", bare symbol sequences
- Added patterns for: "The output is X", "Answer: X", "Result: X"
- JSON extraction works for `{"output": "..."}` format

### 2026-03-18: Extraction pipeline overhaul
- Structured JSON output as primary extraction
- Truncated JSON recovery for thinking-budget-exhausted responses
- Symbol-aware extraction as fallback
- Answer normalization
- Vocabulary listing in prompts

### 2026-03-18: Cross-model extraction
- **Gemini Flash**: Clean JSON extraction, 100% success rate
- **Qwen3.5-35B-A3B nothink**: Clean JSON, ~100% success
- **Qwen3.5-35B-A3B think**: Content always empty (reasoning exhausts budget). UNUSABLE.
- **Nemotron nothink**: Clean JSON with occasional `</think>` artifacts, extraction handles it
- **Nemotron think**: Reasoning field only, content empty. UNUSABLE.

---

## Difficulty Calibration

### 2026-03-18: Multi-model calibration
- **T2 N=8 confirmed as SB2 sweet spot** for Gemini Flash (~40% accuracy)
- T2 N=8 is too hard for Qwen3.5-35B-A3B (~8%) — creates floor effect
- The ideal tier for cross-model SB2 depends on the weakest model you want to include
- For the Kaggle submission: T2 N=8 is correct for Gemini-class models. Report the floor effects for weaker models as additional findings.

---

## Infrastructure Notes

### llama.cpp setup
- Binary at `llama.cpp/build/bin/llama-server`
- Rpath needed fixing: `install_name_tool -rpath` to update library search path
- Launch flags: `--jinja --reasoning-format deepseek --port 8082 --ctx-size 8192 --n-gpu-layers 99 --flash-attn on`
- `--flash-attn` requires explicit `on` value (not just the flag)

### mlx_lm setup (Nemotron)
- Installed via Python 3.13 venv at `/tmp/mlx_venv/`
- Uses `mlx_lm server` (not deprecated `mlx_lm.server`)
- Model: `~/.lmstudio/models/lmstudio-community/NVIDIA-Nemotron-3-Nano-30B-A3B-MLX-4bit`
- Model name in API: `default_model` (omit model field or use this)
- Supports `extra_body.chat_template_kwargs.enable_thinking` for thinking control
- Reasoning field is `reasoning` (not `reasoning_content`)

### Memory usage (M4 Pro 48GB)
- Qwen3.5-27B (Q4_K_M): ~16GB
- Qwen3.5-35B-A3B (Q4_K_M): ~20GB
- Nemotron Nano (MLX 4-bit): ~12.4GB loaded
- Can run 27B + 35B-A3B simultaneously (~36GB)
- Cannot add Nemotron without shutting one down

---

## Key Decisions & Pivots

### 2026-03-18
1. **Gemini model versions**: Must use gemini-3-flash-preview and gemini-3.1-pro-preview (2.0 deprecated)
2. **Training set overhaul**: Added informative example generation
3. **Retry logic**: Gemini client (5 retries, exponential backoff)

### 2026-03-18 Evening
1. **MoE thinking mode disabled**: Both Qwen3.5-35B-A3B and Nemotron Nano get stuck in infinite reasoning loops on STS. Must use nothink variants.
2. **max_tokens tuning**: Default 512-1024 is insufficient for thinking models. Set to 8192 for thinking, 512 for nothink.
3. **Nemotron served via MLX**: No GGUF available for this hybrid Mamba-Transformer architecture. mlx_lm.server works but requires Python 3.13 venv.
4. **FLR ≈ 0 confirmed early**: All three feedback conditions converge to ~40% for Gemini Flash. The error signal adds nothing beyond accumulated context.

---

## Cost Tracking

### 2026-03-18
- Gemini API calls: ~15 initial + ~200 calibration
- Post-fix eval: ~200 calls

### 2026-03-18 Evening
- Gemini Flash: ~160 API calls (90 SB1 + 48 SB2 main + 24 SB2 error_only)
- Qwen3.5-35B-A3B: ~160 local calls (similar breakdown)
- Nemotron Nano: ~160 local calls (similar breakdown)
- Total API cost: <$0.05 (Gemini Flash is very cheap)
- Local compute: ~20 min for nothink models, unusable for thinking mode

---

## Priority 3: 25-Instance Dataset Generated

- Tier: 2
- Instances: 25
- Avg rules/instance: 3.0
- SB1 items: 100R + 25E + 0L = 125
- SB2 turns per instance: 12 (all achieved)
- Type E feasibility: 25/25 (100%)
- Saved to: `results/priorities_20260318_194841/dataset_t2_25inst/dataset.json`
