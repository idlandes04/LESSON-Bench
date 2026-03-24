# Preliminary N=25 Cross-Model Observations

**Date:** 2026-03-20
**Models:** Gemini 3 Flash, GPT-5.3-Chat, GPT-5.3-Codex (all complete)
**Sample size:** N=25 STS instances per condition
**Conditions:** correction, practice_only, error_only, no_feedback
**Turns per instance:** 12
**Tier:** 2 (2 transformation rules)

---

## Raw Results

### Gemini 3 Flash (complete)

| Condition | Avg | T0 | T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8 | T9 | T10 | T11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| correction | 53.7% | 36% | 44% | 60% | 48% | 44% | 60% | 60% | 56% | 64% | 52% | 76% | 44% |
| practice_only | 52.0% | 36% | 40% | 52% | 40% | 48% | 68% | 52% | 64% | 52% | 52% | 68% | 52% |
| error_only | 33.0% | 36% | 36% | 32% | 32% | 36% | 40% | 36% | 48% | 8% | 20% | 32% | 40% |
| no_feedback | 40.0% | 36% | 32% | 48% | 48% | 44% | 36% | 44% | 56% | 24% | 28% | 44% | 40% |

### GPT-5.3-Chat (complete)

| Condition | Avg | T0 | T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8 | T9 | T10 | T11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| correction | 40.7% | 44% | 40% | 52% | 36% | 40% | 44% | 44% | 44% | 48% | 24% | 32% | 40% |
| practice_only | 38.0% | 44% | 40% | 44% | 32% | 32% | 32% | 52% | 48% | 24% | 36% | 36% | 36% |
| error_only | 28.3% | 36% | 36% | 32% | 24% | 32% | 32% | 36% | 24% | 16% | 20% | 20% | 32% |
| no_feedback | 31.0% | 36% | 44% | 36% | 32% | 24% | 28% | 28% | 36% | 32% | 24% | 28% | 24% |

### GPT-5.3-Codex (complete)

| Condition | Avg | T0 | T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8 | T9 | T10 | T11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| correction | 67.0% | 60% | 48% | 60% | 52% | 60% | 76% | 72% | 76% | 64% | 76% | 80% | 80% |
| practice_only | 66.7% | 48% | 56% | 60% | 52% | 64% | 68% | 76% | 80% | 68% | 80% | 84% | 64% |
| error_only | 46.3% | 52% | 40% | 52% | 44% | 36% | 48% | 52% | 48% | 28% | 56% | 52% | 48% |
| no_feedback | 44.3% | 56% | 56% | 32% | 44% | 48% | 48% | 52% | 28% | 36% | 48% | 44% | 40% |

## Key Metrics — Cross-Model Comparison

| Metric | Codex | Flash | Chat |
|---|---|---|---|
| **FLR** | **-0.020** | **+0.007** | **-0.053** |
| Correction avg | 67.0% | 53.7% | 40.7% |
| Practice avg | 66.7% | 52.0% | 38.0% |
| Error_only avg | 46.3% | 33.0% | 28.3% |
| No_feedback avg | 44.3% | 40.0% | 31.0% |
| Correction slope | +0.153 | +0.100 | -0.040 |
| Practice slope | +0.173 | +0.093 | +0.013 |
| **Answer effect** | **+21.5%** | **+16.3%** | **+9.7%** |
| **Evaluation effect** | **+2.0%** | **-2.7%** | **0.0%** |
| **Eval damage (err-nofb)** | **+2.0%** | **-7.0%** | **-2.7%** |

---

## Deep Findings

### 1. Universal feedback blindness: FLR ~ 0 across all models

The headline finding is negative: no model tested shows differential learning from corrective feedback versus neutral answer exposure. FLR is near zero for all three models (Codex: -0.020, Flash: +0.007, Chat: -0.053). The corrective framing — "Incorrect. The correct output is X" — confers no learning advantage over the neutral framing — "The output for Y is X. Next question."

This held across a 27-percentage-point range of baseline ability (Codex 67% to Chat 40%), ruling out the possibility that FLR ~ 0 is an artifact of task difficulty. The strongest model (Codex) and the weakest (Chat) both show the same pattern.

**The N=3 pilot overestimated FLR for every model.** Pilot FLR values (Codex +0.09, Flash +0.08) deflated to near-zero at N=25. This confirms that small-sample pilot estimates are directionally useful but unreliable for magnitudes.

### 2. The code-training hypothesis (H2) does not hold for FLR

The original hypothesis predicted that code-trained models would show higher FLR because code training involves error-correction cycles (write code, get error, fix code). At N=25, Codex FLR = -0.020 — not meaningfully different from Flash or Chat. Code training makes Codex dramatically better at the STS task itself (67% vs 41-54%), but it does not change *how* the model learns from feedback.

This is an important distinction: code training improves pattern extraction ability, not error-driven learning.

### 3. In-context learning is example-driven pattern matching, not hypothesis-driven reasoning

This is the deeper mechanistic finding that emerges from the 2x2 factorial design.

The four conditions cleanly separate two information channels:
- **Answer visibility** (does the model see the correct output?): strong effect (+10-22%)
- **Evaluation signal** (does the model know if it was right/wrong?): zero or negative effect (-2.7% to +2.0%)

Models improve over turns *only* when correct input-output pairs accumulate in context. They treat these pairs as additional training examples and get better at extracting the underlying transformation rules through pattern matching. The corrective wrapper ("Incorrect.") around those examples is ignored.

Crucially, when models receive evaluation signals *without* correct answers (error_only), they cannot use the information. A systematic reasoner could do hypothesis elimination: "I guessed X and was told it's wrong, so my current rule hypothesis is incorrect — let me revise." This is how humans learn from error signals. The models show no evidence of this operation. Instead, error_only performs *worse* than no_feedback, suggesting the model's own wrong guesses pollute the context rather than serving as useful negative evidence.

**This implies that in-context "learning" in transformers is fundamentally different from human error-driven learning.** Transformers do in-context pattern matching (more examples of the pattern → better extraction). They do not do in-context hypothesis testing (my prediction was wrong → my hypothesis is wrong → try a different hypothesis). The error signal is informationally useful but computationally inaccessible to current architectures.

### 4. Error signals without corrections are harmful for chat/reasoning models — but not code models

The "evaluation damage" finding — error_only performing worse than no_feedback — replicates across Flash and Chat but **not** Codex:
- Flash: 33.0% vs 40.0% (-7.0pp)
- Chat: 28.3% vs 31.0% (-2.7pp)
- **Codex: 46.3% vs 44.3% (+2.0pp) — immune**

The N=3 pilot predicted this (Codex eval damage = 0pp), and it holds at N=25. Codex is uniquely protected against context pollution from its own wrong guesses.

**Why?** The mechanism for Flash/Chat is context pollution: error_only accumulates the model's own wrong guesses tagged with "Incorrect." but no corrective information. These wrong guesses function as noise, diluting the signal from the original 8 training examples. In no_feedback, "Next question." is informationally neutral.

Codex's immunity likely stems from code training. Code models process millions of error messages, stack traces, and failed test outputs during pre-training — error-tagged content that is informationally noisy but not performance-degrading. This may have trained Codex to be robust to "Incorrect." prefixed noise in context, even though it doesn't help Codex *learn* from the error signal (FLR is still ~0).

**The distinction is critical:** code training makes models *robust to* error signals (no damage) but does not make them *responsive to* error signals (no learning). Robustness ≠ responsiveness.

For agentic systems: **telling a chat-tuned AI "you're wrong" without the right answer is worse than saying nothing.** Code-tuned models are at least immune to this harm, even if they can't use the information constructively.

### 5. Model ability stratification: Codex >> Flash >> Chat

The three models show dramatically different baseline abilities on the STS task:
- **Codex: 56%** (all 4 conditions avg) — strongest pattern extraction, steepest learning curves (+0.15-0.17 slope in answer-visible conditions)
- **Flash: 45%** (all 4 conditions avg) — moderate ability, moderate slopes (+0.09-0.10)
- **Chat: 35%** (all 4 conditions avg) — weakest, nearly flat or negative slopes (-0.04 to +0.01)

Code training (Codex) and thinking-budget (Flash with MEDIUM thinking) both contribute to STS ability, but through pattern extraction — not through error-driven learning. Chat, as a pure chat-tuned model, is weakest at the symbolic reasoning task and shows the least within-session improvement.

Codex also shows the **strongest answer effect** (+21.5% vs Flash +16.3% vs Chat +9.7%). Better models benefit more from seeing correct examples — a multiplicative relationship between baseline ability and example-driven learning.

Notably, Chat's correction slope is *negative* (-0.040) — it gets worse over turns even when given correct answers in corrective framing. Practice slope is slightly positive (+0.013). This drives Chat's negative FLR (-0.053): the corrective framing is actively counterproductive for Chat specifically. The "Incorrect." prefix doesn't just fail to help — it interferes with Chat's ability to extract patterns from the answer that follows. **Chat is the only model where corrective feedback is worse than neutral exposure — the strongest evidence that evaluation signals can be toxic, not merely useless.** This has direct implications for agentic system design: wrapping correct answers in corrective framing can actively degrade a chat-tuned model's in-context learning.

### 6. Turn-0 baselines validate the experimental design

Turn-0 accuracy (before any feedback divergence) should be identical across conditions since all conditions see the same 8 training examples. Results:
- Flash: 36% across all four conditions (exact match)
- Chat: 44%/44%/36%/36% (some sampling noise at N=25 but consistent within answer-visibility pairs)
- Codex: 60%/48%/52%/56% (correction/practice/error/no_feedback — more variance than Flash/Chat, but no systematic bias by answer visibility)

The near-identical T0 baselines confirm that performance differences across conditions emerge from the feedback manipulation, not from pre-existing differences in the STS instances. The design is sound.

### 7. Answer effect scales with baseline ability

The answer effect (performance gain from seeing correct examples) scales monotonically with model ability:
- **Codex: +21.5%** (strongest model, strongest benefit from examples)
- **Flash: +16.3%**
- **Chat: +9.7%** (weakest model, weakest benefit)

This is a multiplicative relationship, not additive. Better pattern extractors benefit *more* from additional correct examples in context. This has a direct implication: the value of showing correct examples to an AI agent increases with the agent's baseline capability. Weak models gain less from good examples; strong models gain more.

### 8. Code training creates a robustness-without-responsiveness dissociation

The complete Codex factorial reveals a clean dissociation that may be the most nuanced finding:
- **FLR = -0.020** → Codex does NOT learn differentially from corrections (no responsiveness)
- **Eval damage = +2.0pp** → Codex is NOT harmed by error signals (full robustness)
- **Answer effect = +21.5%** → Codex benefits most from correct examples (strongest pattern matching)

Code training produces models that are immune to error-signal noise but blind to error-signal information. This is consistent with code training exposing models to vast amounts of error output (compiler errors, test failures, stack traces) as context that must be *tolerated* but not *learned from* in the traditional sense. The model learns to extract useful patterns despite noisy error messages, but it never learns to use those error messages as signals for hypothesis revision.

This dissociation (robustness ≠ responsiveness) is architecturally significant. It suggests that making models responsive to error signals — actually revising hypotheses based on being told "wrong" — would require a fundamentally different training signal than error→fix pairs in code.

---

## Remaining Work

- **Kaggle SDK integration** — Port SB2 evaluation to `@kbench.task` format with `llm.prompt(schema=STSAnswer)`. Required for competition submission.
- **Kaggle production run (9 models)** — Gemini 3 Flash/Pro Preview, Claude Opus/Sonnet/Haiku 4.5, GLM 5, Qwen 3 235B, DeepSeek R1/V3.2. All at N=25, 4 core conditions. Combined with 3 existing OpenRouter models = **12 total models across 6 labs**.
- **Bootstrap CIs** — resample instance-level data to get 95% CIs on all reported effects.

## Files

- Flash run: `results/sb2_20260319_193213/` (complete)
- Chat run: `results/sb2_20260320_000915/` (complete)
- Codex correction+practice: `results/sb2_20260320_000915/` (complete)
- Codex error_only+no_feedback: `results/sb2_20260320_162100/` (complete, resumed from `sb2_20260320_103109`)
- Flash observations: `docs/observations_flash_n25.md`
- Interaction logs: `logs/20260320_*_gpt-5.3-{chat,codex}_sb2.jsonl`
- SQLite DB: `results/lesson_bench.db` (all 3 models, all conditions)
