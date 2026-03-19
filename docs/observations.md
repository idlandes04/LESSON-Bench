# LESSON Observations & Lessons Learned
## Living document — updated as we test and iterate
### Status: Pilot results only (3 STS instances). v11.0 spec finalized — adds no-feedback baseline, mechanistic probes, 15+ models via OpenRouter. Production runs pending.

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

**Important: N=3 per cell — these numbers are noise, not signal. Do not compute RII or HTR from these. The 25-instance dataset (25 Type E items per cell) is required for meaningful strategy decomposition.**

| Model | T2 N=4 E | T2 N=8 E | T3 N=4 E | T3 N=8 E | T3 N=4 L | T3 N=8 L |
|-------|----------|----------|----------|----------|----------|----------|
| **Gemini Flash** | 0% (0/3) | 33% (1/3) | 33% (1/3) | 33% (1/3) | 33% (1/3) | 0% (0/3) |
| **Qwen3.5-35B-A3B** | 0% (0/3) | 0% (0/3) | 33% (1/3) | 0% (0/3) | 33% (1/3) | 0% (0/3) |
| **Nemotron Nano** | 0% (0/3) | 0% (0/3) | 0% (0/3) | 0% (0/3) | 33% (1/3) | 0% (0/3) |

*95% CI for 1/3 ≈ [0.8%, 91%]. These results confirm the extraction pipeline works and items are answerable, but carry no diagnostic power for strategy classification.*

### SB2: Feedback Conditions (T2, N=8, 8 turns, 3 instances — PILOT)

**This is a 3-instance pilot (24 observations per condition). Treat all results as directional observations, not findings. Production run requires the 25-instance dataset (300 observations per condition).**

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
The benchmark discriminates across model capability levels:
- **Gemini Flash** (API, thinking): T1=87%, T2=38%, T3=11%
- **Qwen 35B-A3B** (local, nothink): T1=44%, T2=8%, T3=5%
- **Nemotron Nano** (local, nothink): 0% everywhere (complete floor effect — cannot engage with STS at all)
- **Caveat:** Discrimination here is primarily between model scale tiers (frontier API vs. 3B-active local vs. hybrid architecture), not between cognitive profiles. The more interesting discrimination (feedback type effects, strategy differences) requires the full production dataset.

### 2. Thinking Mode Creates Infinite Reasoning Loops on STS
Both local MoE models (3B active params) get stuck in unbounded reasoning when thinking is enabled:
- Qwen3.5-35B-A3B: 4096+ reasoning tokens without concluding, content always empty
- Nemotron Nano: Same behavior — 13K+ chars of reasoning, never produces answer
- These models try to inductively figure out the STS rules from examples, get stuck analyzing edge cases, and exhaust the token budget
- **Implication**: MoE models with 3B active params lack the reasoning capacity for in-context rule induction at the STS difficulty level. They need to be tested in nothink mode.
- **Interesting contrast**: Gemini Flash (with MEDIUM thinking) succeeds because it has much more compute per token and its thinking is bounded
- **Note**: This is itself a finding worth reporting — STS tasks expose a thinking-mode failure mode in small-active-param MoE models that wouldn't surface on typical benchmarks

### 3. FLR ≈ 0 Directionally Observed (Pilot — Not Yet Confirmed)
For Gemini Flash (the only model with enough accuracy to measure):
- correction: 38% (~9/24), practice_only: 42% (~10/24), error_only: 42% (~10/24)
- All three conditions converge to ~40% accuracy
- The difference between 38% and 42% is ~1 item out of 24 — well within noise (95% CI for 9/24 ≈ [19%, 59%])
- **Directional observation:** feedback type appears not to matter, but this pilot cannot confirm or reject FLR ≈ 0. The 25-instance production run (300 observations per condition) is needed for statistical power.
- **Open question — SB2 baseline gap:** SB1 T2 N=8 accuracy is 50%, but SB2 correction averages only 38% at the same tier/N. The multi-turn format may itself degrade performance (context pollution from the model's own wrong answers, format overhead, or harder STS instances in the pilot set). This gap needs investigation before interpreting feedback effects.

For Qwen (lower accuracy):
- correction: 12%, practice_only: 8%, error_only: 4%
- Too close to floor for meaningful FLR calculation, but directionally consistent
- With ~3/24, ~2/24, ~1/24 correct, these are effectively indistinguishable from noise

### 4. N=8 Helps at T1-T2 for Flash but Not for Weaker Models
- Gemini Flash: T2 N=4→N=8 doubles accuracy (25%→50%)
- Qwen3.5-35B-A3B: T2 stays at 8% regardless of N — more examples don't help
- **Note on information vs. capacity:** The "informative example" design guarantees all rules are exercised even at N=4, so the N=4→N=8 improvement reflects the model's ability to use *redundant* evidence (more diverse inputs per rule), not just seeing more rules. This is a genuine measure of learning capacity — a model that can already induce the rule from N=4 doesn't need the redundancy.
- **Interpretation**: Qwen at 3B active params likely lacks the capacity to induce positional rules from any number of in-context examples. The N effect is a measure of learning capacity itself — but only above the minimum capability threshold. Models at floor (Nemotron, Qwen at T2+) don't benefit from more examples because they can't engage with the task, not necessarily because they "can't learn."

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
- **T2 N=8 selected as SB2 operating point** for Gemini Flash (~50% SB1 accuracy, ~38% in multi-turn SB2 pilot)
- T2 N=8 is too hard for Qwen3.5-35B-A3B (~8%) — creates floor effect
- The ideal tier for cross-model SB2 depends on the weakest model you want to include
- For the Kaggle submission: T2 N=8 is appropriate for Gemini-class models. Report the floor effects for weaker models as additional findings.
- **Open question**: The ~12-point drop from SB1 (50%) to SB2 (38%) at the same tier/N needs investigation. If multi-turn format inherently degrades accuracy, the operating point may need recalibration.

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
4. **FLR ≈ 0 directionally observed**: All three feedback conditions converge to ~40% for Gemini Flash in pilot (3 instances, 24 obs/condition). Suggestive but far too few observations to confirm. Needs 25-instance production run.

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

---

## Known Gaps & Open Questions (Pre-Production)

### Statistical
1. **All SB2 results are from a 3-instance pilot.** Claims about FLR, condition effects, or trajectory patterns require the 25-instance production run (300 obs/condition) before they carry weight.
2. **Type E/L cell sizes (N=3) make RII/HTR meaningless.** Strategy decomposition analysis must wait for production data.
3. **Confidence intervals should accompany all reported percentages** in the production writeup. Point estimates from small samples are misleading.

### Design
4. **SB2 < SB1 baseline gap is unexplained.** Gemini Flash scores 50% at T2 N=8 in single-prompt SB1 but only 38% in multi-turn SB2 correction (same tier, same starting N). Possible causes: context pollution from wrong answers, multi-turn format overhead, or harder STS instances in the pilot. Must be investigated — if the format itself degrades accuracy, feedback effects are confounded with format effects.
   - **v11.0 STATUS: ADDRESSED.** Three new conditions target this directly: **no-feedback** (multi-turn, "Next question." only — isolates format effect), **clean-context** (correct pairs as clean examples, wrong answers stripped — tests context pollution), and **SB1 at N=16/N=32** (provides learning curve comparison). Run on existing 3 pilot instances first (24 API calls, trivial cost).
5. **Missing multi-turn no-feedback baseline.** The 2×2 factorial's "not evaluated, no answer" cell uses SB1 (single-prompt), which is a different format.
   - **v11.0 STATUS: ADDRESSED.** No-feedback condition added as 4th core condition. True 2×2 with all cells in same multi-turn format.
6. **Only one model above floor for SB2.** Gemini Flash is the only model with enough T2 accuracy to measure feedback effects. The cognitive profiling story (2D map, radar chart) needs multiple models above floor. The production run on Kaggle SDK models (Gemini Pro, larger Qwen, etc.) should address this.
   - **v11.0 STATUS: ADDRESSED.** Expanded to 15-20 models via OpenRouter. Two-phase selection: broad SB1 scan filters for models in 15-70% accuracy range.

### Interpretive
7. **In correction and practice_only conditions, each turn adds a correct input→output pair to context.** After 8 turns, the model effectively has N≈16. The SB2 accuracy should be compared to SB1 N=16 (currently not tested) to determine whether SB2 is showing anything beyond the SB1 learning curve extended by more examples.
   - **v11.0 STATUS: ADDRESSED.** SB1 at N=16 and N=32 will be run on existing T2 instances. Clean-context at Turn 8 is functionally equivalent to SB1 N=16 — validates the comparison.
8. **Nemotron at 0% tells us about minimum capability thresholds, not about learning.** Worth reporting as a finding about task engagement floors, but should not be framed as evidence about cognitive profiles or feedback responsiveness.

### v11.0 New Questions
9. **Code-training hypothesis untested.** Do code-tuned models (DeepSeek-Coder-V2, Codestral) show higher FLR than chat-tuned counterparts? This is the most exciting hypothesis — if confirmed, feedback blindness is a training data gap, not an architectural limitation.
10. **Mechanistic probes are pilot-only.** Clean-context, prompted-correction, structured-correction, and reformatted-correction run on 3-5 instances on 2-3 models. If they show clear signal, consider expanding. If not, they still provide evidence about WHY FLR ≈ 0.
11. **OpenRouter rate limiting and availability.** Need rate-limited client with exponential backoff. Some models may be unavailable or have long queue times.
