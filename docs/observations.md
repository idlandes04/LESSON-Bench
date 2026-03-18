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

## Model Behavior Observations
(What we see when running models on STS tasks)

### Local Models
#### Qwen3.5-27B (thinking mode)
(awaiting local model download)

#### Qwen3.5-27B (non-thinking mode)
(awaiting local model download)

#### Qwen3.5-35B-A3B
(awaiting local model download)

#### Gemma 3 27B
(awaiting local model download)

#### Phi-4 14B
(awaiting local model download)

### API Models
#### Gemini 3.0 Flash (gemini-3-flash-preview)
- **2026-03-18 first test**: Tier 2, N=4 — got first STS item wrong but close (one symbol off: `⟐▲◈◈◈` vs expected `⟐◈◈◈◈`). Shows the model is *trying* to apply rules but making errors.
- API connection works via OpenAI-compatible endpoint at generativelanguage.googleapis.com
- Note: gemini-2.0-flash is deprecated for new users (404 error). Must use 3.x models.

#### Gemini 3.1 Pro (gemini-3.1-pro-preview)
- **2026-03-18 first test**: 0/5 at Tier 3 N=8, 0/5 at Tier 2 N=4
- **Critical observation**: Pro produces verbose reasoning even with "respond with output only" instruction. Responses include: "Wait, Example 4 input: ▲", "Wait, ◈ is", "(5) →"
- Some responses echo the input verbatim (identity), suggesting the model doesn't understand the transformation
- The extraction pipeline catches fragments of reasoning as answers — need stricter prompt formatting
- **Intermittent 500 errors** from the API — added retry logic (3 retries with exponential backoff)

### Early Signal (pre-informative-examples fix)
- Both Gemini models scored 0% on initial testing, but this was BEFORE the training set quality fix. The old training examples were mostly identity transformations — models had nothing to learn from. **Must re-test with improved generator.**

## Answer Extraction Observations

### 2026-03-18: Initial testing
- Regex extraction handles: "Output: X", "<think>...</think>Output: X", bare symbol sequences
- Added patterns for: "The output is X", "Answer: X", "Result: X"
- **Problem with verbose models**: Gemini 3.1 Pro includes reasoning fragments ("Wait, ...") that get caught by last-line fallback. May need to strip common reasoning prefixes.
- JSON extraction works for `{"output": "..."}` format — untested on live models yet.

## Difficulty Calibration
(Which tiers/N-values give ~35-45% accuracy for SB2 targeting)

### 2026-03-18: Pre-calibration notes
- Initial tests suggest Tier 2+ may be harder than expected — 0% on early Gemini tests (but with uninformative training examples)
- **ACTION**: Re-run calibration after generator fix with informative training examples
- May need to reconsider: Tier 2 at N=8 could be the sweet spot instead of Tier 3 at N=4

## Feedback Condition Observations (SB2)
(Early signals from correction vs practice-only comparisons)
- No data yet — SB2 pilot was stopped to fix generator and model issues first

## Type E/L Item Quality
(Do diagnostic items actually discriminate?)
- Type E items generated successfully at all tiers 2+
- Type L items with divergent partial-rule answers confirmed working (e.g., correct `◆◆⧫⟐◈` vs partial `◈▲◈⧫⟐`)
- Full discrimination testing awaits model runs

## Tokenization Notes
(Symbol rendering/tokenization across models)
- Using symbols from Miscellaneous Technical and Geometric Shapes Unicode blocks
- tiktoken screening not yet run (deferred — will test when local models are downloaded)
- All models use same symbols — within-model contrasts unaffected regardless

## Key Decisions & Pivots

### 2026-03-18
1. **Gemini model versions**: Must use gemini-3-flash-preview and gemini-3.1-pro-preview (2.0 deprecated)
2. **Training set overhaul**: Added informative example generation to ensure rules are exercised. This is a fundamental quality improvement — without it, models have nothing to learn from.
3. **Retry logic**: Added to Gemini client (3 retries, exponential backoff) for 500 errors
4. **Prompt refinement needed**: "Respond with output only" isn't sufficient for Gemini 3.1 Pro — produces reasoning. Need to test stricter formats like ending prompt with "Output:" to force completion.

## Cost Tracking
(API costs and quota usage)

### 2026-03-18
- Gemini API calls: ~15 total (testing connectivity + 5 Pro items + 5 Flash items)
- Estimated cost: <$0.01 (Gemini Flash/Pro are very cheap per call)
- Local model downloads: in progress (~63GB total)
