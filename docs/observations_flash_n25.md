# Gemini 3 Flash N=25 Run — Observations

**Date:** 2026-03-19
**Model:** gemini-3-flash-preview (thinking_level=MEDIUM)
**Sample size:** N=25 STS instances per condition
**Conditions:** correction, practice_only, error_only, no_feedback
**Turns per instance:** 12
**Tier:** 2 (2 transformation rules)
**Total API calls:** 1,200
**Runtime:** 180.5 minutes (~3 hours)
**Errors/retries:** 0

---

## Raw Results

| Condition | Avg Accuracy | T0 | T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8 | T9 | T10 | T11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| correction | 53.7% | 36% | 44% | 60% | 48% | 44% | 60% | 60% | 56% | 64% | 52% | 76% | 44% |
| practice_only | 52.0% | 36% | 40% | 52% | 40% | 48% | 68% | 52% | 64% | 52% | 52% | 68% | 52% |
| error_only | 33.0% | 36% | 36% | 32% | 32% | 36% | 40% | 36% | 48% | 8% | 20% | 32% | 40% |
| no_feedback | 40.0% | 36% | 32% | 48% | 48% | 44% | 36% | 44% | 56% | 24% | 28% | 44% | 40% |

## Key Metrics

| Metric | Value |
|---|---|
| **FLR (Feedback Learning Rate)** | +0.007 |
| Correction slope (2nd half - 1st half) | +0.100 |
| Practice slope | +0.093 |
| **Answer effect (2x2 factorial)** | +0.163 (+16.3%) |
| **Evaluation effect (2x2 factorial)** | -0.027 (-2.7%) |
| Interaction effect | +0.087 |

## Observations

### 1. Answer visibility is the dominant factor, not error signals

The **answer effect (+16.3%)** is the largest factorial effect by far. Conditions that reveal the correct answer (correction, practice_only) average ~53% accuracy, while conditions that withhold answers (error_only, no_feedback) average ~36%. This means Gemini Flash benefits primarily from *seeing the right answer*, not from knowing whether it was right or wrong.

The **evaluation effect is near zero (-2.7%)** — knowing "correct/incorrect" without the answer provides essentially no learning benefit. This is a strong result against the hypothesis that error signals alone drive in-context learning.

### 2. FLR is near zero — Flash does not differentially learn from correction vs practice

FLR = +0.007 is negligible. Both correction and practice_only show nearly identical learning slopes (+0.100 vs +0.093). This means Gemini Flash improves over turns when given correct answers, but it doesn't matter whether those answers come as corrections ("Incorrect. The correct output is X") or as practice examples ("The output for Y is X. Next question.").

**Implication:** For Gemini Flash, the corrective framing adds nothing beyond the answer content itself. The model does not preferentially encode information from error-corrective feedback compared to neutral answer exposure.

### 3. Conditions rank: correction > practice > no_feedback > error_only

The condition ranking is informative:
- **correction (53.7%)** and **practice_only (52.0%)** are nearly tied — both provide answer visibility
- **no_feedback (40.0%)** outperforms error_only — surprising, since no_feedback gets zero information while error_only gets correctness signals
- **error_only (33.0%)** is the worst condition — knowing "incorrect" without the answer may actually be confusing the model

The error_only result is notable: it performs *worse* than no_feedback, which gets no information at all. This suggests that error signals without corrective information may introduce noise or negative interference in Flash's in-context processing.

### 4. Learning trajectories differ by answer availability

**Improving conditions** (positive slope per turn):
- correction: +0.015/turn — steady improvement
- practice_only: +0.018/turn — slightly steeper than correction

**Flat/declining conditions** (zero or negative slope):
- error_only: -0.006/turn — slight decline, no learning
- no_feedback: -0.002/turn — flat, no learning

The learning trajectories confirm that answer visibility enables within-session learning while withholding answers prevents it entirely. Flash can extract patterns from correct examples but cannot figure out the rules from correctness signals alone.

### 5. Turn 0 baseline is identical across all conditions (36%)

All four conditions start at exactly 36% accuracy on turn 0. This is expected and validates the experimental design — the model sees the same 8 training examples before any feedback divergence. The 36% baseline (above the ~12.5% random chance for 8-symbol sequences) shows Flash can partially learn STS rules from examples alone.

### 6. Anomalous turns 8-9 in error_only and no_feedback

Both error_only and no_feedback show a sharp accuracy dip at turns 8-9 (8%/20% and 24%/28% respectively). This may indicate context degradation — as the conversation grows longer, the model's ability to apply partial rules deteriorates without reinforcement. This pattern is absent in correction and practice_only, where answer visibility counteracts context degradation.

## Implications for the Full Study

1. **FLR may not be the right headline metric for Flash-class models.** The answer effect is the real signal. The 5-model production run should focus on whether code-trained or reasoning-tuned models show a *different* pattern — specifically whether they show FLR > 0 (correction > practice for the same answer content).

2. **The benchmark is working.** The 2x2 factorial cleanly separates answer effect from evaluation effect. The identical turn-0 baselines validate the design. N=25 produces interpretable, differentiable results.

3. **error_only < no_feedback is a publishable finding** if it replicates across models. It suggests that uninformative error signals (without corrective information) can be actively harmful to in-context processing.

4. **Tier 2 difficulty is appropriate.** 36% turn-0 accuracy means the task is not trivially easy (would be ~80%+) nor impossibly hard (would be ~12.5%), leaving room for learning effects to manifest.

## Files

- Run config: `results/sb2_20260319_193213/run_config.json`
- Per-condition JSON: `results/sb2_20260319_193213/gemini-flash_sb2_*.json`
- Combined results: `results/sb2_20260319_193213/combined_results.json`
- Interaction logs: `logs/20260319_19321*_gemini-flash_sb2.jsonl` (4 files, 1 per condition)
- Analysis PDF: `results/flash_n25_report.pdf`
- SQLite DB run_id: `20260319_193213`
