# LESSON Observations & Lessons Learned
## Living document — updated as we test and iterate
### Status: v1.5.0 SB1 scan complete (19 models, 16 pass). SB2 pilot next.

---

## v1.5.0 SB1 Scan Results (2026-03-19) — COMPLETE

### Protocol
- **Tiers**: T1 (easy), T2 (medium) — T2 N=8 is the SB2 operating point
- **N-values**: 4 and 8 training examples
- **Instances**: 3 STS instances per cell (different random rule sets)
- **Items**: 5 test items per instance (3R + 1E + 1L) = 60 items per model
- **Format**: Single-prompt, JSON output with extraction fallback
- **Filter**: T2 N=8 Type R accuracy between 15-70% → passes to SB2

### Full Results (sorted by T2N8 accuracy)

| Model | Provider | T1N4 | T1N8 | T2N4 | T2N8 | Filter |
|-------|----------|------|------|------|------|--------|
| **GLM-5** | OpenRouter | 67% | 93% | 17% | **67%** | PASS |
| **GPT-5.3-Codex** | OpenRouter | 73% | 100% | 33% | **58%** | PASS |
| **GPT-5.3-Chat** | OpenRouter | 67% | 100% | 25% | **50%** | PASS |
| **MiniMax-M2.7** | OpenRouter | 60% | 87% | 17% | **50%** | PASS |
| **Qwen-3.5-397B** | OpenRouter | 60% | 87% | 8% | **50%** | PASS |
| **Gemini-3.1-Pro** | OpenRouter | 73% | 93% | 8% | **42%** | PASS |
| **Claude-Opus-4.6** | OpenRouter | 60% | 73% | 42% | **42%** | PASS |
| **Claude-Sonnet-4.6** | OpenRouter | 73% | 80% | 25% | **42%** | PASS |
| **DeepSeek-V3.2** | OpenRouter | 47% | 80% | 17% | **33%** | PASS |
| **Claude-Haiku-4.5** | OpenRouter | 53% | 80% | 25% | **33%** | PASS |
| **GPT-5.4-Mini** | OpenRouter | 73% | 73% | 25% | **33%** | PASS |
| **DeepSeek-R1** | OpenRouter | 53% | 20% | 17% | **33%** | PASS |
| **Kimi-K2.5** | OpenRouter | 67% | 80% | 17% | **33%** | PASS |
| **Gemini-3.1-Flash-Lite** | OpenRouter | 47% | 80% | 8% | **17%** | PASS |
| **Grok-4.20** | OpenRouter | — | — | — | **17%** | PASS |
| **Qwen-3-Coder-30B** | LM Studio | 60% | 67% | 42% | **17%** | PASS |
| Llama-3.3-70B | OpenRouter | 40% | 60% | 8% | 8% | FAIL |
| Llama-4-Maverick | OpenRouter | — | — | — | 0% | FAIL |
| Qwen-3-1.7B | LM Studio | 27% | 27% | 8% | 8% | FAIL |

**16 of 19 models pass the SB2 filter.**

### Local Models with Empty Responses (excluded from table above)
- **Qwen-3.5-27B** (LM Studio): 0% — thinking mode exhausts token budget, returns empty content. Context window too small for max_tokens needed.
- **Qwen-3.5-27B-NoThink** (LM Studio): 0% — still hits context size limits on N=8 prompts. Needs larger context window in LM Studio.
- **GLM-4.7-Flash** (LM Studio): 0% — all empty responses, same context size issue.

---

## Interpretation & Analysis

### 1. Discriminatory Power — Outstanding

T2N8 ranges from 67% (GLM-5) to 0% (Llama-4-Maverick). This is not a benchmark where everything clusters — it reveals dramatic differences across models on a task none of them have seen before. The judges' rubric explicitly asks for "a gradient of performance" and "can the benchmark significantly distinguish model performance?" — 16 models spread across a 50-point range satisfies this decisively.

### 2. Learning Gradient (N=4 -> N=8) — Nearly Universal

Almost every model improves at T1 when given more examples, confirming they use additional training examples for in-context learning. But the SIZE of the improvement differs — some models jump 30 points, others barely move. This is the sample efficiency axis producing real signal. At T2, the N effect is weaker — positional rules are harder to induce regardless of example count. This is itself a finding about the nature of in-context learning.

### 3. T1 Ceiling as Raw Capacity Indicator

GLM-5, GPT-5.3 variants, Gemini Pro, Claude Sonnet all hit 80-100% at T1N8. These models can reliably induce simple DIRECT rules from 8 examples. That's the prerequisite for SB2 — they have the capacity, the question is whether they can USE feedback.

### 4. GPT-5.3 Codex vs Chat — First Code Hypothesis Test

Codex (58%) outperforms Chat (50%) at T2N8. Same base model, different fine-tuning. Both hit 100% at T1N8 — the difference only appears on harder positional rules. The code-tuned variant is better at in-context rule induction. At N=15 per cell this isn't statistically significant, but it's directionally consistent with Hypothesis 2 (code training improves pattern induction). If Codex also shows higher FLR in SB2, that's a headline finding.

### 5. DeepSeek-R1 Anomaly — Overthinking Effect?

R1 scores T1N4=53% -> T1N8=20%. It gets WORSE with more examples on the EASIEST tier. V3.2 goes 47% -> 80% on the same items. This could be noise (only 15 items), but it could also be a real overthinking effect — reasoning models spending their thinking budget on hypothesis-revision loops instead of just applying the obvious pattern. Echoes the MoE thinking-loop finding with local models. Flagged for targeted replication; do not build narrative around this until it replicates at scale.

### 6. Claude Family — Scale Gradient Flattens at T2

Opus and Sonnet both score 42% at T2N8, Haiku at 33%. Opus is supposed to be the more capable model, but on in-context rule induction it's tied with Sonnet. This suggests that whatever makes Opus "better" generally doesn't translate to better pattern learning from examples. A genuine insight about what model scaling buys you — and what it doesn't.

### 7. Llama Models — Strikingly Weak

Llama 3.3 70B at 8%, Maverick at 0%. These are large models that perform well on standard benchmarks. Their failure here suggests Meta's training pipeline doesn't emphasize the kind of abstract pattern induction STS requires. Compare to GLM-5 at 67% — a less-discussed model that crushes this task. The benchmark is revealing something about training methodology that standard evals miss.

### 8. GLM-5 as the Leader — A Surprise

If this holds at scale, "GLM-5 outperforms GPT-5.3 and Claude Opus on novel in-context rule induction" is an attention-grabbing finding. It suggests the Zhipu/GLM training approach develops stronger in-context learning abilities, at least on this kind of task.

### 9. Statistical Caveats

- With 15 items per cell (3 instances x 5 items), CIs are roughly +/-20%. The ranking is directional, not definitive.
- The DeepSeek-R1 and Qwen-3-Coder anomalies (getting WORSE with more examples) are probably noise at this sample size.
- GLM-5 at 67% vs Claude Opus at 42% — the true values might be closer. SB2 production runs with 25 instances will provide definitive rankings.

---

## SB2 Pilot Model Selection (8 models)

Each model earns its slot by testing a specific hypothesis:

| Model | T2N8 | Why Include |
|-------|------|-------------|
| GLM-5 | 67% | Highest performer — ceiling reference |
| GPT-5.3-Codex | 58% | Code hypothesis H2 (compare to Chat) |
| GPT-5.3-Chat | 50% | Code hypothesis H2 (compare to Codex) |
| Gemini-3.1-Flash* | — | Kaggle SDK model, Google judge appeal. Use Flash instead of Pro to avoid reasoning token cost blowup |
| Claude-Sonnet-4.6 | 42% | Different architecture family |
| DeepSeek-R1 | 33% | Reasoning-trained model hypothesis H3 |
| DeepSeek-V3.2 | 33% | Same family, NOT reasoning-trained (control for R1) |
| Claude-Haiku-4.5 | 33% | Scale comparison within Claude family |

**Deferred:** Grok (incomplete data), Flash-Lite (too low for meaningful feedback signal), Kimi and MiniMax (interesting but don't test unique hypotheses), Qwen-3-Coder local (anomalous). Can add back for production if pilot reveals something worth chasing.

*Note: Gemini-3.1-Pro used >50% of total run cost in SB1 due to long reasoning traces. Flash provides a Gemini architecture test at manageable cost.

---

## Early Pilot Results (2026-03-18) — Preserved for Reference

### SB2 Feedback Pilot (Gemini 3 Flash, 3 instances, T2 N=8)

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

**Directional observation:** All three feedback conditions converge to ~40%. FLR appears near zero, but N=3 instances (24 obs/condition) is far too small to confirm. The 8-model SB2 pilot will test this with more statistical power.

### Thinking Mode Failure in Small MoE Models
- Qwen3.5-35B-A3B and Nemotron Nano (3B active params) enter infinite reasoning loops on STS tasks
- Burns 4096+ tokens on hypothesis-revision without concluding
- Works fine in nothink mode (fast, wrong answers, but extractable)
- STS tasks expose a thinking-mode failure mode in small-active-param MoE models that wouldn't surface on typical benchmarks

---

## Infrastructure Fixes Applied (v1.5.0)

- **max_tokens**: 20,000 for OpenRouter (thinking models need headroom), 2,048 for LM Studio (constrained by local context window).
- **Empty-response fallback**: `prompt_json()` can return empty string when JSON schema mode fails silently. Added `if not raw_response.strip()` check -> falls back to plain `prompt()` in both `pilot.py` and `sb2_pilot.py`.
- **JSON schema compatibility caching**: OpenRouter models that don't support `json_schema` response format (GPT-5.3, MiniMax) are remembered in `_json_schema_unsupported` set — fallback fires once per model per process, then skips silently.
- **Log pruning race condition**: Fixed `_prune_logs()` in `interaction_log.py` — concurrent threads could crash when a file was deleted between `glob()` and `stat()`.
- **Unicode console fix**: Added `_safe_print()` in `pilot.py` for Windows cp1252 terminals that can't encode STS symbols.

---

## STS Generator Notes

### Training Set Quality
- Informative example generation ensures all rules are exercised even at N=4
- 0% identity examples at all tiers
- N=4->N=8 improvement reflects ability to use redundant evidence, not just seeing more rules

### Type E Feasibility
- Tier 2: 82% (slightly below 90% target)
- Tier 3: 90%
- Tier 4: 100%
- 25-instance T2 dataset: 100% feasibility (all instances have extrapolation items)

---

## Open Questions for SB2

1. **Does FLR ~ 0 hold across 8 models?** The Gemini Flash pilot (1 model, 3 instances) is suggestive but not confirmatory. The 8-model SB2 pilot will test with statistical power across model families.
2. **Which component of feedback matters?** The 2x2 factorial (correction vs practice_only vs error_only vs no_feedback) decomposes whether models benefit from the error signal, the correct answer, or neither.
3. **Does code training predict feedback responsiveness?** GPT-5.3-Codex vs Chat is a clean within-family test of Hypothesis 2.
4. **Does reasoning training help or hurt?** DeepSeek-R1 vs V3.2 at the same T2N8 accuracy (33%) provides a natural pair for Hypothesis 3.
5. **SB2 baseline gap**: Gemini Flash pilot showed SB1 T2N8=50% but SB2 correction avg=38%. The no-feedback condition will isolate whether multi-turn format itself degrades performance.
6. **Does the R1 overthinking anomaly replicate?** T1N8 regression needs more instances to confirm or dismiss.

---

## Cost & Infrastructure

### API Costs (approximate)
- 2026-03-18: ~600 Gemini Flash calls (<$0.05)
- 2026-03-19: ~19 models x 60 items x 2 calls (prompt_json + fallback) ~ 2,280 OpenRouter calls, estimated <$10 total
- SB2 pilot (8 models, 3 instances, 4 conditions, 8 turns): estimated $15-20 on OpenRouter
- SB2 production (5-6 models, 25 instances): estimated $50-100 on OpenRouter

### Local Setup (RTX 5090 32GB via LM Studio)
- Qwen-3-Coder-30B-A3B: Works, 17% T2N8, 7-12s per model scan
- Qwen-3-1.7B: Works, 8% T2N8 (too weak for SB2)
- Qwen-3.5-27B: Context too small for thinking, needs 8K+ context in LM Studio
- GLM-4.7-Flash: Same context issue, all empty responses

### Results Saved
- `results/production_20260319_004154/` — first OpenRouter run (Claude, GPT-5.4-Mini, DeepSeek-V3.2, Gemini-Flash-Lite, Grok, Llama models)
- `results/rescan_or_v2/` — OpenRouter re-scan (GPT-5.3-Chat/Codex, GLM-5, MiniMax, Gemini-Pro, Kimi, Qwen-397B, DeepSeek-R1)
- `results/rescan_local_v3/` — Local LM Studio re-scan (5 models)
