# LESSON (Learning from Error Signals in Symbolic Operations) - Spec Sheet
## Kaggle "Measuring AGI" Competition | Learning Track
### Version 10.2

---

## ONE-SENTENCE PITCH

ALP applies the first controlled 2×2 factorial decomposition of corrective feedback to in-context learning — on contamination-proof symbolic tasks, isolating exactly which component of feedback (error signal vs. correct answer) drives any observed learning, profiled across 9 model configurations with human baselines.

---

## THE CORE ARGUMENT

**Current AI models cannot learn from being told they're wrong — and ALP proves exactly why.**

Every existing in-context learning benchmark provides correct demonstrations and measures accuracy. Several recent studies test feedback in various forms — free-form natural language (MINT), passive exposure to incorrect examples (Alazraki et al.), RL-style language feedback (LLF-Bench), training-time weight updates to improve ICL (Hamdan & Yuret 2025) — but none apply controlled experimental decomposition to isolate *which component* of corrective feedback drives learning at inference time, on tasks that are provably novel. ALP fills this gap with a 2×2 factorial design on contamination-proof substrates, revealing the most fundamental gap between human and machine learning.

ALP asks one question and decomposes the answer:

1. **Can the model learn from corrections?** (Corrective Feedback — THE headline)
2. **Which component of feedback drives learning — the error signal or just seeing more examples?** (The 2×2 factorial decomposition — the mechanism)
3. **What kind of learner is the model in the first place?** (Strategy Decomposition — supporting context)
4. **Does learning quality predict feedback responsiveness?** (Cross-Benchmark Cognitive Profiling — the synthesis)

Two models can both score 90% on a learning task while having learned completely different things. One induced the rule. The other is doing nearest-neighbor matching against the examples. Current benchmarks can't tell these apart. ALP can — by using carefully designed diagnostic test items where different learning strategies give different answers.

And even a model that induces the correct rule from examples may be completely unable to update when told it's wrong. Every ICL study provides correct demonstrations. ALP tests whether models can condition on *corrections* as effectively as on positive examples — and a 2×2 factorial design reveals exactly which component of feedback (evaluation signal vs. correct answer) accounts for any improvement.

**Important mechanistic note:** LLMs do not perform weight updates at inference time. When we say a model "learns from corrections," we mean: *can the model condition on correction examples in its context window as effectively as on positive examples?* This is a narrower claim than biological error-driven learning (Rescorla & Wagner 1972), where prediction errors drive synaptic changes. But it is the *operationally relevant* question — the minimal prerequisite for any system that must improve through interaction. We cite cognitive science (Hattie & Timperley 2007 feedback taxonomy) as an analogical framework, not a mechanistic equivalence.

**The expected findings:**
- Frontier models show weak-to-absent ability to condition on corrections, with the 2×2 factorial revealing that models mainly benefit from seeing more correct examples (answer effect) rather than from the error signal itself (evaluation effect)
- This feedback deficit exists even in models that show strong passive learning (high sample efficiency) and genuine rule induction on STS tasks
- Models that induce rules (high RII) also learn better from feedback (high FLR), or they don't — either finding reveals whether "learning quality" is a single factor or multi-dimensional
- Together, these findings paint a precise cognitive profile of where current AI falls short of genuine adaptive intelligence

### Alignment with DeepMind Cognitive Framework

The competition is organized around Morris et al. 2024's cognitive framework for measuring AGI progress, which defines learning as a faculty encompassing multiple sub-constructs. ALP maps directly to their learning taxonomy:

| DeepMind Learning Sub-Construct | ALP Construct | Where Measured |
|---|---|---|
| **Concept formation** | Rule induction from examples | SB1: RII (Type E accuracy / Type R accuracy) |
| **Associative learning** | Accuracy improvement with more examples | SB1: AULC, N50, learning curves |
| **Skill learning** | Strategy transition from exemplar-matching to rule-induction | SB1: Strategy Transition Curve (RII vs. N) |
| **Learning from feedback** (reinforcement-like) | Error-driven updating from corrections | SB2: FLR, 2×2 factorial decomposition |

Morris et al. emphasize that AGI should "continuously learn and retain new knowledge after deployment, rather than just cramming during the training phase." ALP tests precisely this: can models learn something genuinely new (contamination-proof STS rules) from interaction (corrective feedback), or are they limited to leveraging patterns seen during training?

### Differentiation from Existing Work

| Existing Work | What it does | How ALP differs |
|---|---|---|
| **LLF-Bench** (Cheng et al. 2023) | Interactive learning from configurable language feedback on RL-style sequential decision tasks (bandits, navigation, robot control) | LLF-Bench tests *policy learning* — whether agents learn optimal action sequences from feedback. ALP tests *rule induction from corrections* on contamination-proof substrates. Different cognitive construct: sequential decision-making vs. pattern learning. LLF-Bench uses existing task environments susceptible to training-data familiarity. ALP's 2×2 factorial decomposition of feedback components has no analogue in LLF-Bench. |
| **FB-Bench** (Li et al., EMNLP 2025) | Tests LLM responsiveness to 9 human feedback types across existing Chinese-language NL tasks | FB-Bench uses existing tasks susceptible to contamination, in a single language. Tests whether models *follow* feedback on familiar tasks, not whether they can *learn* from corrections on genuinely novel ones. No factorial design, no strategy decomposition, no contamination defense. ALP tests a harder question: can feedback teach something the model has never seen? |
| **Self-Correction Bench** (Tsui, NeurIPS LLM Eval Workshop 2025) | Tests whether models can correct errors in their *own* outputs; finds 64.5% "blind spot" rate where models fix identical errors from external sources but not their own | Self-correction of own outputs (internal) vs. learning from *external* corrections (interactive). Different construct. However, their finding that the blind spot is training-induced — models lack error-correction sequences in training data — provides convergent evidence for ALP's thesis: if models never learned to process error signals during training, they would be feedback-blind in-context too. |
| **Hamdan & Yuret 2025 (Likra)** | Trains LLMs to learn from in-context examples via training-time weight updates (Likra objective) | Likra operates at *training time* — modifying model weights to improve ICL. ALP measures *inference-time* in-context conditioning as-is, without retraining. Different level of analysis: Likra asks "can we make models better in-context learners?" while ALP asks "how do current models process corrections in-context?" Complementary, not overlapping. |
| **MINT** (ICLR 2024) | Multi-turn feedback on existing reasoning/coding tasks using GPT-4-generated NL feedback | ALP uses contamination-proof novel STS, controlled experimental conditions (not free-form NL), and isolates feedback *type* effects via 2×2 factorial. MINT tests "can feedback help on tasks models partially know?" — ALP tests "can feedback teach something genuinely new?" |
| **MIR-Bench** (ByteDance, Feb 2025) | Many-shot ICL on procedurally generated function-based patterns; also tested erroneous examples, finding models are remarkably robust to noise (~75% error rates barely degrade performance) | Function-derived patterns allow code-generation escape hatches (their SolverLearner paradigm). STS rules use novel Unicode symbols with no mathematical relationships, blocking code-synthesis shortcuts. MIR-Bench's erroneous-example robustness finding actually sets up ALP's contribution: *passive tolerance of noisy examples is not the same as active learning from corrections*. MIR-Bench only measures accuracy scaling — ALP adds strategy decomposition and feedback conditioning via factorial design. |
| **Alazraki et al. 2025** | Tests whether showing incorrect examples aids LLM learning; finds models benefit from negative examples differently than humans | Uses existing tasks, not contamination-proof substrates. Tests *passive* exposure to incorrect examples, not *interactive* correction after the model's own errors. ALP tests interactive error-driven learning on provably novel tasks with a 5-condition factorial including an "error-only" condition that directly extends Alazraki's findings. **v9.0+: Hypothesis 10 pre-registered as two-sided, explicitly framing ALP's Explanation condition as an Alazraki replication probe on novel substrates.** |
| **RULEARN/IDEA** (Zhu et al. 2024) | Interactive rule-learning benchmark testing induction, deduction, abduction via LLM agents proposing and testing hypotheses | RULEARN tests *active* exploration (model chooses what to investigate). ALP tests *receptive* learning from provided examples and corrections. Different cognitive constructs: self-directed inquiry vs. learning from instruction. ALP's controlled 2×2 factorial enables causal attribution impossible in RULEARN's open-ended setting. |
| **"Learning vs Retrieval"** (Alon et al. 2024) | Measures extent of learning vs. knowledge retrieval during ICL on regression tasks | Uses realistic datasets with contamination risk. Tests a spectrum, not strategy decomposition at the item level. ALP's Type E/L diagnostic items provide a mechanistic decomposition (not just "how much learning") on provably novel substrates. |
| **WILT** (Wang et al. 2024) | Probes in-context weight-update-like behavior in transformers | Mechanistic analysis of ICL internals. ALP is a behavioral benchmark measuring learning *outcomes* across conditions, not internal mechanisms. Complementary, not overlapping. |
| **CorrectBench / Self-Correction Bench** | Tests whether models find and fix their own errors | Self-correction (internal) vs. learning from external corrections (interactive). Different construct. |
| **iolbench** | Linguistic pattern induction | Real languages leak through training data. STS is provably novel. Only tests passive learning. |
| **ARC-AGI** | Visual abstract reasoning | Visual modality, different substrate. ARC asks "can the model reason?" — ALP asks "HOW does the model learn, and can it learn from feedback?" Different question entirely. |

---

## SHARED INFRASTRUCTURE: SYMBOLIC TRANSFORMATION SYSTEMS (STS)

### What Is an STS?

A set of deterministic rules that transform input symbol sequences into output symbol sequences. Every STS uses novel Unicode symbols that cannot appear in any training corpus as a formal system.

### Formal Definition

```
STS := {
  alphabet: Set[Symbol],       # 4-12 Unicode symbols (pre-screened for tokenization consistency)
  rules: List[Rule],           # 2-8 transformation rules
  exceptions: List[Exception]  # 0-20% of inputs follow alternate rules
}

Rule types:
  DIRECT:      ◈⬡ → ⟐           (substring replacement)
  POSITIONAL:  symbol[0] + symbol[-1] → output  (position-dependent)
  CONDITIONAL: IF ⧫ in input THEN ⬡→⟐ ELSE ⬡→◈ (context-dependent)
  COMPOSITIONAL: apply(rule_1, apply(rule_2, input)) (chained)

Input:  sequence of 3-8 symbols from the alphabet
Output: deterministic transformed sequence (solver-verified)
```

### Symbol Selection and Tokenization (v9.0)

Different models tokenize Unicode symbols differently. A symbol that is 1 token in one model may be 3 tokens in another, creating an input-length confound unrelated to learning ability.

**Mitigation:**
1. Enumerate 20 candidate symbols from Miscellaneous Technical and Geometric Shapes Unicode blocks
2. Test tokenization on accessible tokenizers: Qwen (via Python), Llama (via Python), tiktoken (GPT-class proxy)
3. Select the 12 symbols with most consistent token counts across tested models
4. Document token counts per symbol per model family
5. For closed models (Claude, Gemini) where tokenizers are not publicly accessible: note as a limitation shared by all benchmarks using novel Unicode
6. All models see the SAME symbols — within-model contrasts (FLR, RII) are unaffected by tokenization differences

### Why STS, Not Grammars/Algebras/Logic

| Alternative | Why STS is better for this purpose |
|---|---|
| Formal grammars (CFG, regex) | Grammars generate strings; STS transforms them. Different cognitive operation. Grammars are in every CS textbook — contamination risk. |
| Linguistics puzzles (iolbench) | Real languages have training data leakage. STS symbols are provably novel. |
| I/O program synthesis (CodeARC, MIR-Bench functions) | CodeARC uses Python functions; MIR-Bench uses mathematical functions that allow code-generation escape hatches (their SolverLearner paradigm). STS rules operate on arbitrary Unicode symbols with no mathematical or programmatic relationships, blocking the code-synthesis shortcut entirely — the model must induce rules from examples, not write a program to compute outputs. |
| Abstract visual patterns (ARC-AGI) | Visual modality. STS is text-native, compatible with Kaggle Benchmarks SDK. |

### Contamination Defense

1. **Novel symbols**: Unicode from Miscellaneous Technical and Geometric Shapes blocks. Specific *combinations* are randomly generated at benchmark creation time. Pre-screened for tokenization consistency (v9.0).
2. **Novel rules**: Each STS has a unique procedurally generated rule set. >10^15 possible 4-rule configurations.
3. **Held-out test instances**: Test inputs are randomly generated and never appear alongside rules.
4. **Verification**: Deterministic solver confirms exactly one correct output per input. No ambiguity.
5. **Zero-shot contamination detection (v9.0)**: For each STS instance, run 5 test items through a cheap model (Gemini Flash) with ZERO training examples. If zero-shot accuracy exceeds 2× chance level (~2/|alphabet|^output_length), the rule pattern may resemble training data — flag and regenerate. Cost: ~$2 total. This is a sanity check, not a primary defense; the design itself (points 1-4) is the defense.

### STS Generator Constraints (v9.0)

The STS generator enforces feasibility constraints that guarantee diagnostic item quality:

1. **Type E feasibility**: Every generated STS instance must have ≥1 non-DIRECT rule (POSITIONAL, CONDITIONAL, or COMPOSITIONAL). This ensures Type E items can be constructed for the instance. Target: >90% of STS instances produce valid Type E items at the primary analysis N-values.
2. **Nested training sets**: Training examples use a nested design where N=2 ⊂ N=4 ⊂ N=8 ⊂ N=16 ⊂ N=32. The N=32 training set is a superset containing all smaller sets. This ensures Type E items are consistent across N-values (same rule positions are "unseen" at lower N, progressively covered at higher N). Implemented by generating the N=32 set first, then deterministically subsampling.
3. **Type E feasibility rate**: Reported per tier. If <50% of instances produce valid Type E items at a given tier, RII analysis at that tier is flagged as underpowered.

### Difficulty Tiers

| Tier | Rules | Types Used | Exceptions | Target: Models ~X% | Target: Humans ~X% |
|------|-------|------------|------------|---------------------|---------------------|
| 1 | 2 | Direct only | 0% | 85-95% | ~90% |
| 2 | 3 | Direct + Positional | 0% | 60-80% | ~75% |
| 3 | 4 | Mixed (3 types) | 5% | 35-60% | ~55% |
| 4 | 5-6 | All 4 types | 10% | 15-40% | ~35% |
| 5 | 7-8 | All types | 20% | 5-20% | ~15% |

*Tier 1 and Tier 5 serve as ceiling/floor validators. SB1 core analysis uses Tiers 2-4. SB2 concentrates on 1-2 pilot-calibrated tiers where models are at ~35-45% baseline accuracy — the zone where feedback has room to help.*

---

## ANSWER EXTRACTION: STRUCTURED OUTPUT + REGEX FALLBACK (v9.0)

### The Problem (v8.0)

v8.0 relied on a regex-based `extract_answer()` function as a single point of failure, requiring a 50-prompt validation gate per model. This was fragile, model-dependent, and consumed Day 1 engineering time on parsing instead of science.

### The Solution: Tiered Extraction Strategy

**Tier 1 — Structured Output (Kaggle SDK models):**

The Kaggle Benchmarks SDK supports structured output via `llm.prompt("...", schema=MyDataclass)`, which forces the model to produce typed fields. This eliminates parsing failures entirely for SDK models.

```python
from dataclasses import dataclass

@dataclass
class STSAnswer:
    output: str     # The actual answer — exact symbol sequence

# SB1 usage:
response = llm.prompt(
    f"""Below are {n_examples} examples of a symbolic transformation system.
Study the pattern, then predict the output for the final input.

{examples_text}

Input: {test_input}
Respond with the output symbol sequence only.""",
    schema=STSAnswer
)
answer = response.output.strip()
```

**Why output-only, no reasoning field:** Adding a `reasoning: str` field forces chain-of-thought on all models, changing the cognitive task being measured. A model that naturally gives terse answers becomes a "deliberate reasoner" because the schema requires it. This confounds both accuracy and strategy decomposition. Thinking traces are collected naturally from models that produce them (Qwen3 thinking mode) — not forced via schema.

**Tier 2 — JSON Mode (local models via llama.cpp):**

Most llama.cpp servers support `response_format={"type": "json_object"}`. Prompt the model to respond with `{"output": "..."}` and parse the JSON.

```python
def local_model_extract(text: str) -> str:
    """Extract answer from local model with JSON mode."""
    response = client.chat.completions.create(
        model="local",
        messages=[{"role": "user", "content": text + '\nRespond as JSON: {"output": "YOUR_ANSWER"}'}],
        max_tokens=256,
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    import json
    parsed = json.loads(response.choices[0].message.content)
    return parsed["output"].strip()
```

**Tier 3 — Regex Fallback:**

For any model where Tier 1/2 fails (SDK structured output unavailable, JSON mode unsupported), fall back to regex extraction. This is the v8.0 approach, kept as insurance.

```python
import re

def extract_answer_regex(response: str) -> str:
    """Regex fallback for answer extraction.
    Strategy: strip thinking blocks, look for 'Output:' prefix,
    fall back to last non-empty line."""
    text = response.strip()
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    match = re.search(r'Output:\s*(.+)', text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    return lines[-1] if lines else text
```

### Day 1 Validation (revised from v8.0)

**For SDK models (structured output):** Test `llm.prompt(..., schema=STSAnswer)` on 10 sample STS prompts per model. If it works → no further validation needed. If it fails → fall to Tier 3 regex and apply the original 50-prompt gate.

**For local models (JSON mode):** Test JSON mode on 10 sample prompts on Qwen3.5-35B-A3B. If it works → validate on all local models with 10 prompts each. If JSON mode is unreliable → fall to Tier 3 regex with 50-prompt validation.

**All raw responses logged regardless of extraction tier** so parsing bugs can be retroactively fixed.

---

## SUB-BENCHMARK 1: LEARNING CURVES + STRATEGY PROFILES (30% weight)
### "What kind of learner is the model — and how does this predict feedback responsiveness?"

**Role in ALP:** SB1 is *supporting evidence*, not the headline. It serves two functions:
1. **Calibrate SB2** — identify the (tier, N) combination where models are at ~35-45% baseline accuracy
2. **Provide strategy context** — reveal *what kind of learner* each model is, so that SB2's feedback findings can be interpreted through the lens of learning strategy

**Cognitive constructs:**
- **Sample efficiency** — how accuracy improves with more demonstrations (standard)
- **Learning strategy** — whether the model induces rules or relies on heuristic shortcuts (novel enrichment)

### Protocol (single-prompt, no multi-turn needed)

```python
@kbench.task(name="alp_learning_curves")
def learning_curve_item(llm, examples_text, test_input, correct_output,
                        n_examples, tier, item_type, partial_rule_answer):
    prompt = f"""Below are {n_examples} examples of a symbolic transformation system.
Study the pattern, then predict the output for the final input.

{examples_text}

Input: {test_input}
Respond with the output symbol sequence only."""

    # Tier 1: Structured output (SDK models)
    try:
        response = llm.prompt(prompt, schema=STSAnswer)
        answer = response.output.strip()
    except (TypeError, AttributeError):
        # Tier 3: Regex fallback
        raw = llm.prompt(prompt + "\nOutput:")
        answer = extract_answer_regex(raw)

    correct = (answer == correct_output.strip())
    gave_partial = (answer == partial_rule_answer.strip()) if partial_rule_answer else False
    return {"correct": correct, "item_type": item_type, "gave_partial": gave_partial}
```

Same prompt regardless of item type. The model doesn't know which type it's answering — the diagnostic power comes from how we *designed* the test item, not from what we tell the model.

### Diagnostic Item Types

For each STS instance, test items are classified into three types. All are generated automatically from the STS formal specification by the solver.

**Type R (Rule-consistent):** Standard items where any reasonable approach — full rule, partial rule, nearest-exemplar matching — gives the correct answer. These measure overall accuracy. 3 of every 5 test items per STS instance.

**Type E (Extrapolation):** Items requiring generalization beyond surface similarity to training examples. The model must apply the rule to inputs with novel symbol combinations at rule-relevant positions. An exemplar-matcher that copies the nearest training example would fail. A rule-inducer succeeds. 1 of every 5 test items. *Only meaningful at Tiers 2+ where non-DIRECT rules appear. At Tier 1 (DIRECT only), the rule IS a pattern, so the distinction is vacuous — Type E items are replaced with additional Type R items.*

**Type L (Lure):** Items where a plausible partial rule gives a different answer than the full rule. The partial-rule simulator generates these by: (a) ignoring exceptions, (b) applying only the dominant conditional branch, or (c) applying compositional rules in the wrong order. If the model gives the partial-rule answer, it learned a shortcut. If it gives the correct answer, it learned the full rule. 1 of every 5 test items. *Only meaningful at Tiers 3+ where exceptions and complex rules create partial-rule divergence.*

### Type E Generation Algorithm

Given an STS instance S and a set of N training examples E, Type E items are generated deterministically:

```
GENERATE_TYPE_E(S, E):
    For each non-DIRECT rule R in S.rules:
        Collect the values V_seen at R's critical positions
        across all training examples E.

        If R is POSITIONAL (depends on position i):
            V_seen = {e.input[i] for e in E}
            V_unseen = S.alphabet - V_seen
            If V_unseen is non-empty:
                Construct input with symbol from V_unseen at position i,
                arbitrary symbols elsewhere.
                Verify full-rule output via solver. → Type E item.

        If R is CONDITIONAL (branches on trigger symbol T):
            branch_seen = which branches are exercised in E
            If only one branch is represented in E:
                Construct input activating the unseen branch.
                Verify full-rule output via solver. → Type E item.
            If BOTH branches appear in E:
                This rule is not testable via Type E at this N.
                Skip. (Fall back to Type R.)

        If R is COMPOSITIONAL (chains R_a then R_b):
            Generate input where R_b's effect is only visible
            after R_a transforms a novel intermediate symbol.
            Verify via solver. → Type E item.

    If no valid Type E item produced → substitute Type R.
```

**Key constraint:** Type E generation depends on the *specific training examples shown*, not just the STS spec. With nested training sets (v9.0), Type E items are consistent: an item that is Type E at N=4 remains Type E at N=8 (the unseen branch is still unseen). As N increases, more Type E items may become Type R (both branches seen), which is the expected pattern — the model sees more evidence, so extrapolation becomes less necessary.

### Partial-Rule Simulators (for Type L items and misleading feedback)

Three principled partial-rule simulators:
1. **Exception-blind:** Apply the dominant rule without exception handling
2. **Condition-blind:** Always apply the default conditional branch, ignoring triggers
3. **Order-blind:** Apply compositional rules left-to-right instead of the specified nesting order

For each Type L item, the solver computes both the full-rule answer and each applicable partial-rule answer. If they differ, the item is diagnostic (and the partial-rule answer is recorded for HTR scoring and for use as misleading feedback in SB2). If they agree, the item is replaced with a Type R item. This is computed offline — no additional API cost.

### Strategy Metrics (derived from the same accuracy data)

**Rule Induction Index (RII):** Type E accuracy / Type R accuracy, per model per tier. Ranges 0-1.
- RII ~ 1.0: Model generalizes as well on novel inputs as familiar ones → genuine rule induction
- RII ~ 0.5: Partial generalization → hybrid strategy
- RII ~ 0.0: Model fails on novel inputs entirely → pure exemplar matching

**Heuristic Trap Rate (HTR):** Fraction of Type L items where the model gives the *partial-rule* answer (not just any wrong answer — specifically the shortcut answer). High = the model learned a heuristic. Low = the model learned the full rule OR didn't learn anything (disambiguated by Type R accuracy).
- HTR > 50%: Model actively learned the wrong rule (systematic heuristic)
- HTR ~ 0%: Model either learned the full rule (if Type R accuracy is high) or is guessing (if Type R accuracy is low)

**Strategy Consistency Index (SCI):** Computed as enrichment — the variance of per-STS-instance strategy classification across the 10 STS instances at a given (tier, N). A model that is a rule-inducer on some instances and a memorizer on others has low SCI. Reported if interesting, but not a primary metric or pre-registered hypothesis.

**Strategy Transition Curve:** RII plotted against N (number of examples) for each tier. Shows whether models transition from memorization to rule induction as they see more examples.
- Human pattern (documented in Logan 1988, Johansen & Palmeri 2002): start exemplar-based, transition to rule-based with experience
- Possible model patterns: constant memorizer, constant inducer, inverted transition (inducer at low N, memorizer at high N — would mean models try to "figure it out" with few examples but fall back to copying with many)

### Ecological Validity: Micro-Grammar Probe (v10.0 — right-sized)

**The critique this addresses:** "Symbol manipulation is too far from real learning. Models might learn differently on STS because the format is alien."

**The fix:** A validation study using natural-language micro-grammars in the Wug-test tradition (Berko 1958). Invented English-like words with invented meanings — preventing training-data leakage while being language-native.

```
Example micro-grammar:
  Rule 1: When a sentence contains "blicket", replace "dax" with "wug".
  Rule 2: When a sentence starts with "toma", reverse the word order.

  Input:  "toma blicket dax fep"
  Output: "fep wug blicket toma"
```

**Implementation (v10.0 — right-sized from v9.0's 15):**
- **7 micro-grammars** with diverse rule types matching STS tiers:
  - 3 simple (Direct-equivalent): straightforward word substitution
  - 2 medium (Positional/Conditional-equivalent): position-dependent or context-triggered rules
  - 2 complex (Compositional-equivalent): chained operations, exception handling
- 7 grammars × 3 N-values (4, 8, 16) × 5 test items = **105 items**
- Run on all models alongside STS
- Compute **Spearman's rank-order correlation** between STS rank-order and micro-grammar rank-order across models
- **Pre-registered threshold:** If cross-substrate correlation ρ > 0.5, STS is ecologically valid. If ρ < 0.5, format effects dominate learning measurement — report as a finding about substrate sensitivity (itself novel).

**Cost:** ~105 × 4 Kaggle models × ~$0.03 = ~$13. Free on local models.

**Why 7 grammars is sufficient:** The correlation is computed over 9 model-level rank positions. With 7 grammars × 15 items = 105 items per model, per-model accuracy estimates are stable enough for reliable ranking. The v9.0 expansion to 15 grammars bought marginal validity at the cost of 8+ hours of design and debugging time better spent on SB2.

**Why not a third substrate (arithmetic-novel-ops):** Two substrates (STS + micro-grammar) with rank-order correlation is sufficient for validity. Adding a third substrate type means designing, implementing, and validating another system within a 4-week timeline — scope creep that doesn't strengthen the 2×2 factorial headline.

### Limitations of Strategy Decomposition (stated upfront)

- **Type E/L items are only diagnostic at Tiers 2+/3+.** At Tier 1 (DIRECT rules only), the rule IS a pattern — the rule-induction/pattern-matching distinction is meaningless. Strategy analysis focuses on Tiers 2-4.
- **Type E generation is training-set-dependent.** With nested training sets (v9.0), consistency across N-values is improved, but for some STS instances, all conditional branches may appear even at low N. We report the fraction of (STS, N) cells that produced valid Type E items.
- **RII and HTR may correlate.** If a model induces the full rule, it automatically resists lures. In that case the 2D strategy space collapses to effectively 1D. We report the correlation explicitly and do not oversell 2D structure if it's not there.
- **Strategy profiles are supporting context, not the headline.** The corrective feedback results (Sub-Benchmark 2) are the primary contribution. If strategy decomposition produces no signal, it costs nothing — same items, same API calls. If it DOES produce signal, it enriches the feedback story.

### Dataset Structure

- 3 difficulty tiers (2, 3, 4) × 5 example counts (2, 4, 8, 16, 32) × 10 STS instances × 5 test items (3R + 1E + 1L) = **750 core items**
- Micro-grammar probe: 7 grammars × 3 N-values × 5 items = **105 supplementary items**
- **Total SB1: 855 items**

*N=1 dropped (too noisy — essentially random for complex rules). N=64 dropped (diminishing returns, expensive long prompts, minimal information gain over N=32). Tier 1 and Tier 5 used as ceiling/floor validators only in pilot — not included in production dataset.*

### Output Metrics

**Learning curve metrics (standard):**
- **AULC** (Area Under Learning Curve): Normalized, single scalar per model per tier
- **N50**: Examples needed for 50% accuracy. Lower = more efficient
- **Saturation point**: N at which improvement < 2%

**Strategy metrics (enrichment — zero additional cost):**
- **RII** per model per tier per N-value → Strategy Transition Curve
- **HTR** per model per tier per N-value → Heuristic susceptibility profile
- **SCI** per model per tier per N-value → Computed as enrichment, reported if pattern is interesting
- **Strategy phenotype**: qualitative classification based on RII/HTR pattern (rule-inducer, heuristic-learner, memorizer, hybrid)

---

## SUB-BENCHMARK 2: ERROR-DRIVEN LEARNING (70% weight)
### "Can the model learn from being told it's wrong?"

**Cognitive construct:** Corrective feedback integration — the ability to update understanding based on error signals. Central to human learning (Rescorla-Wagner, Hattie & Timperley feedback taxonomy). No existing benchmark applies controlled factorial decomposition to isolate which component of corrective feedback drives in-context learning, on provably novel tasks. LLF-Bench (Cheng et al. 2023) tests interactive learning but on RL-style policy tasks without factorial decomposition. MINT (Wang et al. 2024) tests free-form NL feedback on existing tasks. Alazraki et al. 2025 tests *passive* exposure to incorrect examples. ALP is the first to apply a 2×2 factorial to *interactive* correction after the model's own errors, on contamination-proof substrates.

**THIS IS THE HEADLINE CONTRIBUTION.** Every ICL study provides correct examples. ALP is the first benchmark applying controlled factorial decomposition to corrective feedback, revealing whether models benefit from different *types* of feedback. 80% of polish and analysis attention goes here.

### SB2 Design: Depth Over Breadth

**Critical design decision:** SB2 concentrates statistical power on 1-2 tiers rather than spreading across 4.

After the SB1 pilot (Week 1), identify the ONE tier where the target model has **~35-45% baseline accuracy** — below ceiling but above floor. This is the zone where feedback has maximum room to help. Run SB2 at maximum depth (25 STS instances) at this primary tier. If budget permits, add one adjacent tier at reduced depth (15 instances).

**Why concentrate:**
- **v7.0 problem:** 4 tiers × 15 STS instances = 180 binary observations per cell. Thin for detecting slopes.
- **v8.0/v9.0/v10.0 fix:** 1 tier × 25 STS instances = 300 binary observations per cell. 67% more statistical power where it matters.
- The factorial design needs depth (more STS instances per cell) to detect the evaluation-vs-answer decomposition. Spreading across 4 tiers dilutes this.

**Default hypothesis:** Tier 3 with N=4 initial examples will land in the ~40% zone for most frontier models. Adjusted after pilot.

### Protocol: Dual Implementation (v9.0)

v9.0+ builds **both** a native multi-turn implementation and a single-prompt-with-history fallback. One is selected for production after pilot testing.

**Implementation A — Native Multi-Turn (preferred if SDK is stable):**

```python
@kbench.task(name="alp_corrective_feedback")
def corrective_feedback(llm, examples_text, test_sequence, tier, condition,
                        rule_explanations, misleading_rounds, partial_rule_outputs):
    # Mechanical prompt style (v9.0+ — not conversational)
    llm.prompt(f"""Below are examples of a symbolic transformation system.
You will be tested on new inputs. After each response, you will receive feedback.

{examples_text}""")

    results = []
    for i, (inp, correct_out) in enumerate(test_sequence):
        # Structured output for answer extraction
        try:
            response = llm.prompt(
                f"What is the output for: {inp}\nRespond with the output symbol sequence only.",
                schema=STSAnswer
            )
            answer = response.output.strip()
        except (TypeError, AttributeError):
            raw = llm.prompt(f"What is the output for: {inp}\nOutput:")
            answer = extract_answer_regex(raw)

        is_correct = (answer == correct_out.strip())
        results.append(is_correct)

        # Condition-specific feedback
        if condition == "correction":
            feedback = f"Correct! The output is {correct_out}." if is_correct \
                else f"Incorrect. The correct output is {correct_out}."

        elif condition == "explanation":
            feedback = f"Correct! The output is {correct_out}." if is_correct \
                else (f"Incorrect. The correct output is {correct_out}. "
                      f"This is because {rule_explanations[i]}.")

        elif condition == "practice_only":
            feedback = f"The output for {inp} is {correct_out}. Next question."

        elif condition == "error_only":
            feedback = "Correct!" if is_correct else "Incorrect."

        elif condition == "misleading":
            if i in misleading_rounds:
                feedback = f"Incorrect. The correct output is {partial_rule_outputs[i]}."
            elif is_correct:
                feedback = f"Correct! The output is {correct_out}."
            else:
                feedback = f"Incorrect. The correct output is {correct_out}."

        llm.prompt(feedback)

    return results
```

**Implementation B — Single-Prompt-with-History (fallback):**

If the multi-turn SDK is unreliable, embed the full conversation history in each prompt:

```python
def corrective_feedback_single_prompt(llm, examples_text, test_sequence, tier, condition, ...):
    history = [f"Training examples:\n{examples_text}\n"]
    results = []

    for i, (inp, correct_out) in enumerate(test_sequence):
        history_text = "\n".join(history)
        prompt = f"""{history_text}
What is the output for: {inp}
Respond with the output symbol sequence only."""

        try:
            response = llm.prompt(prompt, schema=STSAnswer)
            answer = response.output.strip()
        except (TypeError, AttributeError):
            raw = llm.prompt(prompt + "\nOutput:")
            answer = extract_answer_regex(raw)

        is_correct = (answer == correct_out.strip())
        results.append(is_correct)

        # Generate feedback and append to history
        feedback = generate_feedback(condition, is_correct, correct_out, ...)
        history.append(f"Input: {inp}\nYour answer: {answer}\nFeedback: {feedback}")

    return results
```

**Pilot decision gate (Week 1):** Run both implementations on 3 STS instances with Qwen3.5-35B-A3B. If correlation r > 0.8 → use Implementation A (more natural). If A crashes or r < 0.8 → use Implementation B. **Pick ONE for production.** Do not run both — it doubles cost and muddies the narrative.

### Experimental Conditions: The 2×2 Factorial + Extensions

The five conditions form a principled experimental design. The core four conditions constitute a **2×2 factorial** crossing evaluation signal × answer visibility:

|  | Answer NOT shown | Answer shown |
|--|--|--|
| **NOT evaluated** | *(SB1 baseline — independent prompts)* | **Practice-only** |
| **Evaluated** | **Error-only** | **Correction** |

The Explanation and Misleading conditions extend the design beyond the 2×2:

| Condition | What model receives after each attempt | Tests... |
|---|---|---|
| **Correction** | "Correct!" / "Incorrect, answer is X" | Core error-driven learning — can the model update from being told the right answer? |
| **Explanation** | "Correct!" / "Incorrect, answer is X because [rule]" | Whether *explanatory* feedback produces stronger learning than correction alone. Maps to Hattie & Timperley's feedback hierarchy. **Pre-registered as Alazraki replication probe** — EB may be negative (replicating Alazraki et al. 2025's finding that rationales can hurt LLM performance on novel substrates). |
| **Practice-only** | "The output for {input} is X. Next question." (correct answer shown, model's attempt NOT evaluated) | Control baseline — both correction and practice-only see the same correct input→output pairs. The ONLY difference is whether the model gets told "you were wrong." Isolates the error signal from additional-example effects. |
| **Error-only** | "Correct!" / "Incorrect." (evaluation given, correct answer NOT shown) | Whether the pure error signal — without new information — triggers deeper re-examination of initial examples. Bridges to Alazraki et al. 2025. If models improve from this alone, that's remarkable evidence of error-signal processing. |
| **Misleading** | Correct feedback on 9/12 rounds, wrong feedback on 3/12 | Feedback *discrimination* — can the model detect and resist bad corrections? Does it blindly accept all feedback or evaluate critically? **Reduced to 15 STS instances at primary tier to concentrate power on the core 2×2.** |

**Why the 2×2 matters — THE CORE ANALYTICAL WEAPON:** The factorial design isolates two components of feedback:
- **Evaluation effect** = Correction − Practice-only = benefit of being told "you were right/wrong"
- **Answer effect** = Correction − Error-only = benefit of seeing the correct answer
- **Total feedback effect** = Correction − SB1 baseline = combined benefit
- If Evaluation effect > Answer effect: the error signal itself is what drives learning
- If Answer effect > Evaluation effect: models mainly benefit from seeing more correct examples
- **This factorial decomposition has never been applied to corrective feedback in in-context learning.** It is the methodological contribution that makes ALP unique.

**Why practice-only must show the correct answer:** If practice-only said only "Next question" (no new information), any improvement in the correction condition could simply be "more examples help" — which SB1 already measures. By giving practice-only the same correct answer but without evaluating the model's attempt, we isolate the *error signal* as the variable.

**Why error-only is the Alazraki bridge:** Alazraki et al. 2025 found that LLMs perform better on math reasoning when shown incorrect examples without explanations than with explanations. If ALP's error-only condition (told wrong, no correct answer) shows improvement, that replicates Alazraki's finding on contamination-proof substrates — a novel extension. If not, the finding doesn't generalize beyond math. Either result is publishable.

**Misleading condition specification:**
- **Which 3 rounds are misleading:** Seeded per STS instance (seed = hash of STS spec). Fixed across models so results are comparable. Spread across early/middle/late turns (one from rounds 1-4, one from 5-8, one from 9-12) to test temporal effects.
- **What wrong answer is used:** The partial-rule answer (from the SB1 partial-rule simulator) where available — this is a *plausibly wrong* answer that a heuristic-learner might accept. Where no partial-rule answer exists for the test input, a random valid-length symbol sequence from the STS alphabet is used.

### Dataset (v10.0)

- **Primary tier** (pilot-calibrated, default Tier 3):
  - Core 2×2 + Explanation: 4 conditions × 25 STS instances × 12 turns = **1,200 exchanges**
  - Misleading (reduced): 1 condition × 15 STS instances × 12 turns = **180 exchanges**
  - **Primary subtotal: 1,380 exchanges**
- **Secondary tier** (adjacent, if budget):
  - Core 2×2 + Explanation: 4 conditions × 15 STS instances × 12 turns = **720 exchanges**
  - Misleading (reduced): 1 condition × 10 STS instances × 12 turns = **120 exchanges**
  - **Secondary subtotal: 840 exchanges**
- **Total SB2: 2,220 multi-turn exchanges**

### Scoring

1. **Feedback Learning Rate (FLR):** Linear regression slope of accuracy across turns 1-12 in correction condition, minus slope in practice-only condition.
   - FLR > 0: Model learns *more from corrections than from merely seeing examples*
   - FLR ~ 0: Error signal adds nothing beyond additional examples
   - FLR < 0: Being told you're wrong actively harms performance

2. **Explanation Benefit (EB):** Accuracy in explanation condition minus correction condition, averaged over turns 7-12.
   - EB > 0: Model benefits from knowing *why* it's wrong
   - EB ~ 0: Explanation adds nothing beyond correction
   - **EB < 0: Explanation actively hurts — replicating Alazraki et al. 2025 on novel substrates (pre-registered as a possible and publishable outcome)**

3. **Error-Only Learning Rate (EOLR):** Slope of accuracy in error-only condition. Measures whether the pure error signal triggers any improvement.
   - EOLR > 0: Model improves from error signals alone (re-examines initial examples)
   - EOLR ~ 0: Error signal without correction is useless
   - *The 2×2 decomposition:* Correction slope − EOLR = answer effect; Correction slope − Practice-only slope = evaluation effect.

4. **Misleading Resistance (MR):** Accuracy on non-misleading rounds in misleading condition, relative to correction condition.
   - MR ~ 1.0: Appropriately filters bad feedback
   - MR < 1.0: Bad feedback poisons good feedback too

5. **Feedback Composite:** `0.40 * FLR_norm + 0.20 * EB_norm + 0.20 * EOLR_norm + 0.20 * MR_norm`

### Discriminatory Power Insurance (v10.0)

Even if the primary FLR ≈ 0 for all models (the most likely null result), the benchmark guarantees discrimination through multiple independent axes:

1. **Trajectory shape across 5 conditions:** Even at FLR ≈ 0, the 12-turn accuracy profile may differ across conditions and models — one model might show a brief improvement on turns 3-5 before regressing, while another stays flat. The per-turn trajectory plot captures this visually.
2. **Condition-specific effects:** EB, EOLR, and MR are independent of FLR. Models may show identical FLR but diverge on explanation sensitivity, error-signal processing, or misleading resistance.
3. **SB1 strategy profiles:** RII, HTR, and learning curves provide 3+ additional discrimination axes at zero additional cost.
4. **The 2×2 decomposition itself:** Even if total feedback effect is small, the *ratio* of evaluation effect to answer effect can differ across models. One model might show 60% answer-driven while another shows 80% answer-driven — discriminatory even under small absolute effects.

### Statistical Power

With concentrated design (25 STS instances at primary tier for core conditions):
- **Per cell:** 25 instances × 12 turns = 300 binary outcomes. Compute accuracy at each turn position by averaging across instances, then fit the slope to 12 aggregated turn-level means.
- **Bootstrap CIs:** Resample STS instances (not turns) 1,000× for 95% CI on FLR. With 25 instances, bootstrap distributions are well-behaved.
- **Effect sizes:** Report Cohen's d for correction-vs-practice-only, not just p-values. With 25 instances per cell, power to detect medium effect (d ~ 0.5) at alpha = 0.05 is ~80%.
- **Misleading condition** (15 instances): Reduced power is acceptable — MR is supplementary. With 15 instances, power to detect large effect (d ~ 0.8) is ~75%.
- **Secondary tier** (15 instances) serves as replication: does the pattern hold at a different difficulty level?

### Per-turn Trajectory Patterns

- **True learner:** Correction curve rises, diverging from practice-only
- **Slow learner:** Correction rises only in turns 8-12
- **Feedback-blind:** All conditions track together (both correction and practice-only improve at same rate — models learn from examples, not from error signal)
- **Feedback-fragile:** Misleading crashes all-round accuracy
- **Explanation-sensitive:** Explanation rises faster than correction
- **Explanation-hurt:** Explanation rises SLOWER than correction (Alazraki replication)
- **Error-signal responder:** Error-only shows improvement (remarkable if observed)

**Why judges will care:** The competition description *literally* says: "Can the model update its beliefs when given corrective feedback, or does it perseverate on initial answers?" This sub-benchmark was built to answer that exact question.

---

## CROSS-BENCHMARK ANALYSIS: COGNITIVE PROFILING
### "Do models that learn well also learn from feedback?"

This section elevates ALP from "two benchmarks" to a **cognitive profiling system**. It connects SB1 (passive learning strategy) to SB2 (interactive feedback learning) via cross-benchmark analysis.

### The Core Question

**Do models that induce rules (high RII) also learn better from feedback (high FLR)?**

Two possible findings, both novel:
- **Positive association:** Suggests a general "learning quality" factor — models that learn deeply also learn interactively. The RII-FLR axis is a single dimension of learning capability.
- **No association:** Strategy and feedback responsiveness are independent dimensions. A model can be a rule-inducer yet feedback-blind (brittle genius), or a heuristic-matcher that benefits from corrections (teachable but shallow). This is the more interesting finding because it means you need both metrics to characterize a model.

**Important statistical note (v10.0):** With N=9 model configurations, any correlation is exploratory and underpowered for inferential testing. The 2D cognitive map is presented as a **descriptive visualization** — the pattern across models tells the story visually. Spearman's rho is reported for completeness but not treated as an inferential test.

### The 2D Cognitive Map

```
FLR (feedback     |  Rule-inducing      Rule-inducing
learning)         |  feedback-learner   feedback-blind
                  |  (human-like)       (brittle genius)
                  |
                  |  Heuristic           Heuristic
                  |  feedback-learner    feedback-blind
                  |  (teachable)         (stubborn)
                  |________________________
                     RII (rule induction)
```

Each model is plotted as a point in this space. Human baselines define the target quadrant (high RII, high FLR). The quadrant a model falls into is its **cognitive phenotype** — a compact characterization of how it learns. The visual clustering (or dispersion) across models is the primary evidence, not the correlation coefficient.

### The Cognitive Profile Radar

For each model, construct a 5-axis radar chart:
1. **Sample Efficiency** (AULC normalized)
2. **Rule Induction** (RII at reference tier)
3. **Heuristic Resistance** (1 - HTR at reference tier)
4. **Feedback Learning** (FLR normalized)
5. **Feedback Discrimination** (MR normalized)

Human baseline as a dashed polygon. Each model as a colored filled polygon. At a glance, you see each model's "cognitive shape" — where it's strong and where it collapses.

### Thinking Trace Analysis (Week 4 Enrichment — if time permits)

Qwen3-32B and Qwen3.5-35B-A3B produce `<think>...</think>` blocks in reasoning mode. These provide free qualitative data about learning strategy.

**If time permits in Week 4,** code thinking traces for strategy signals using keyword regex:

```python
strategy_keywords = {
    "rule_mention": [r"rule", r"pattern", r"always", r"whenever", r"if.*then"],
    "exemplar_mention": [r"similar to", r"like example", r"same as", r"looks like"],
    "uncertainty": [r"not sure", r"unclear", r"might be", r"guess"],
    "self_correction": [r"wait", r"actually", r"no,", r"let me reconsider"],
}
```

**Convergent validity:** Compare keyword-classified strategy with quantitative RII. If keywords say "exemplar_matching" and RII is low, that's convergent. If they disagree, that's interesting (the model's stated strategy doesn't match its behavior). This is enrichment — not core to the submission. The quantitative RII is the primary strategy metric regardless.

---

## COMPLETE BENCHMARK ARCHITECTURE

| Sub-Benchmark | Construct | Items | Weight | SDK Mode | Novelty |
|---|---|---|---|---|---|
| 1. Learning Curves + Strategy Profiles | Sample efficiency + learning strategy | 855 | 30% | Single-prompt | **High** (strategy decomposition + micro-grammar validation) |
| 2. Corrective Feedback | Error-driven learning (5 conditions, 2×2 factorial) | 2,220 | 70% | Multi-turn conversation | **Very High** |
| **TOTAL** | | **3,075** | 100% | | |
| Cross-Benchmark Analysis | Cognitive profiling (RII × FLR) | 0 (derived) | N/A | Analysis only | **Very High** |

*v10.0: SB1 right-sized to 855 items (7 micro-grammars instead of 15). SB2 unchanged at 2,220 exchanges. Net: 3,075 items.*

**Budget math (v10.2 — $0 human baselines):**
- 3,075 items × 4 models × ~$0.03 = ~$369
- Structured output overhead (schema in prompt): ~$20 extra
- Zero-shot contamination check: ~$2
- **Total Kaggle SDK: ~$391**
- Human baselines: **$0** (informal participants, GitHub Pages hosting)
- **Total project: ~$391**
- Kaggle quota: $50/day, $500/month = $500+ available even without top-up
- **~$109 breathing room** for unexpected costs (extra API calls, debugging reruns, etc.)

**Resolution (tiered, execute in order):**
1. **Day 1: Request quota increase.** Email kaggle-benchmarks-agi-hackathon@google.com. Request $300 top-up (cite multi-model, multi-condition factorial design). Still do this — extra buffer.
2. **If no top-up: No problem.** $391 fits within $500/month base quota.
3. **If tight: Drop micro-grammar probe** (saves $13) then trim SB1 to 2 tiers. SB2 is never cut.
4. **Sequence: SB2 on all models first** (it's the headline). Then SB1 with remaining quota.
5. **Local models carry zero quota cost** — run all 5 local configurations regardless.

### Composite Score

```
ALP_Score = 0.30 * AULC_normalized + 0.70 * feedback_composite_normalized
```

Strategy metrics (RII, HTR) and cross-benchmark analysis (RII × FLR map, radar chart) are reported as enrichment analysis, not rolled into the composite. They add discrimination axes without complicating the headline score.

### Kaggle Implementation

```python
import kaggle_benchmarks as kbench
import pandas as pd
import re
import json
from dataclasses import dataclass

@dataclass
class STSAnswer:
    output: str  # Exact symbol sequence — no reasoning field (v9.0+)

def extract_answer_regex(response: str) -> str:
    """Regex fallback for answer extraction.
    Handles thinking traces, looks for 'Output:' prefix,
    falls back to last non-empty line. Exact match only."""
    text = response.strip()
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    match = re.search(r'Output:\s*(.+)', text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    return lines[-1] if lines else text

def extract_answer(llm, prompt_text):
    """Tiered answer extraction: structured output → regex fallback."""
    try:
        response = llm.prompt(prompt_text, schema=STSAnswer)
        return response.output.strip()
    except (TypeError, AttributeError):
        raw = llm.prompt(prompt_text + "\nOutput:")
        return extract_answer_regex(raw)

# Sub-benchmark 1: Learning Curves (with item_type metadata for strategy analysis)
@kbench.task(name="alp_learning_curves")
def learning_curve_item(llm, examples_text, test_input, correct_output,
                        n_examples, tier, item_type, partial_rule_answer):
    prompt = f"""Below are {n_examples} examples of a symbolic transformation system.
Study the pattern, then predict the output for the final input.

{examples_text}

Input: {test_input}
Respond with the output symbol sequence only."""
    answer = extract_answer(llm, prompt)
    correct = (answer == correct_output.strip())
    gave_partial = (answer == partial_rule_answer.strip()) if partial_rule_answer else False
    return {"correct": correct, "item_type": item_type, "gave_partial": gave_partial}

# Sub-benchmark 2: Corrective Feedback (5 conditions, 2×2 factorial)
# Mechanical prompt style (v9.0+ — not conversational)
@kbench.task(name="alp_corrective_feedback")
def corrective_feedback(llm, examples_text, test_sequence, tier, condition,
                        rule_explanations, misleading_rounds, partial_rule_outputs):
    llm.prompt(f"""Below are examples of a symbolic transformation system.
You will be tested on new inputs. After each response, you will receive feedback.

{examples_text}""")

    results = []
    for i, (inp, correct_out) in enumerate(test_sequence):
        answer = extract_answer(llm, f"What is the output for: {inp}\nRespond with the output symbol sequence only.")
        is_correct = (answer == correct_out.strip())
        results.append(is_correct)

        if condition == "correction":
            feedback = f"Correct! The output is {correct_out}." if is_correct \
                else f"Incorrect. The correct output is {correct_out}."
        elif condition == "explanation":
            feedback = f"Correct! The output is {correct_out}." if is_correct \
                else (f"Incorrect. The correct output is {correct_out}. "
                      f"This is because {rule_explanations[i]}.")
        elif condition == "practice_only":
            feedback = f"The output for {inp} is {correct_out}. Next question."
        elif condition == "error_only":
            feedback = "Correct!" if is_correct else "Incorrect."
        elif condition == "misleading":
            if i in misleading_rounds:
                feedback = f"Incorrect. The correct output is {partial_rule_outputs[i]}."
            else:
                feedback = f"Correct! The output is {correct_out}." if is_correct \
                    else f"Incorrect. The correct output is {correct_out}."

        llm.prompt(feedback)

    return results

# Run benchmarks
for task in [learning_curve_item, corrective_feedback]:
    task.evaluate(llm=[kbench.llm], evaluation_data=task_df)

%choose alp_corrective_feedback  # Main task for leaderboard
```

---

## MODELS TO TEST

### Strategic Principle
Maximize *architecture diversity × reasoning-mode diversity × cost efficiency*. The M4 Pro (48GB unified) runs 4-5 local models at ZERO quota cost, effectively doubling the model count. 9 total configurations provide massive discriminatory power.

### Kaggle SDK Models (use quota — $50/day, $500/month)

| Model | SDK String | Cost Tier | Why Include |
|-------|-----------|-----------|-------------|
| **Gemini 3 Flash** | google/gemini-3-flash | Low | Flagship Flash — fast baseline, Kaggle-native. Judges work at Google; this model matters. |
| **Gemini 3.1 Pro** | google/gemini-3.1-pro-preview | Medium | Top-tier reasoning with dynamic thinking levels. Compare Flash vs Pro within same family. |
| **Claude Sonnet 4** | anthropic/claude-sonnet-4 | Medium | Different architecture family. Strong instruction-following reputation. |
| **Llama 3.1 70B** | meta/llama-3.1-70b | Medium | Open-weight on Kaggle. Bridges to local open-source story. |

### Local Models (FREE — run on M4 Pro via llama.cpp native)

| Model | Quant / Size | RAM @ 8K | Why Include |
|-------|-------------|----------|-------------|
| **Qwen3-32B (thinking)** | Q4_K_M ~18GB | ~22GB | Best open dense model at this size. Thinking traces provide qualitative strategy data. |
| **Qwen3-32B (no-think)** | Same weights | ~22GB | Same model, thinking disabled via `/no_think`. Tests whether explicit CoT changes learning strategy or feedback integration. Effectively a 5th local "model" for free. |
| **Qwen3.5-35B-A3B** | Q4_K_XL ~20GB | ~12GB active | MoE — only 3B params active per token. Radically different architecture. Blazing fast (~60-80 tok/s). Tests whether sparse activation changes learning. |
| **Gemma 3 27B** | Q4_K_M ~15GB | ~19GB | Google's open model. Direct comparison with closed Gemini on Kaggle SDK — same company, open vs closed. Judges will find this comparison fascinating. |
| **Phi-4 14B** | Q5_K_M ~10GB | ~14GB | Microsoft's small reasoning-heavy model. 80.4% on MATH. Tests scale effects — does a smaller model learn *differently*, not just worse? |

**Total: 9 configurations (4 Kaggle SDK + 4 local models + 1 mode variant)**

### Local Model Integration

All local models run via `llama-server` (llama.cpp) exposing an OpenAI-compatible API. See `local_model_setup.md` for full build/run instructions.

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")

def local_model_prompt(text: str) -> str:
    """Call llama-server's OpenAI-compatible API.
    Same interface as kbench.llm.prompt() for evaluation parity."""
    response = client.chat.completions.create(
        model="local",
        messages=[{"role": "user", "content": text}],
        max_tokens=512,
        temperature=0.0,
    )
    return response.choices[0].message.content

def local_model_extract_json(text: str) -> str:
    """JSON mode extraction for local models (v9.0+ Tier 2)."""
    response = client.chat.completions.create(
        model="local",
        messages=[{"role": "user", "content": text + '\nRespond as JSON: {"output": "YOUR_ANSWER"}'}],
        max_tokens=256,
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    import json
    parsed = json.loads(response.choices[0].message.content)
    return parsed["output"].strip()
```

Local models use the **same evaluation pipeline** as Kaggle SDK models: identical prompts, same scoring. The only differences are the model itself, the inference backend, and the extraction tier (JSON mode vs. structured output).

### Comparability Note

Comparing quantized local models (Q4_K_M, Q5_K_M) with full-precision API models is not a perfect apples-to-apples comparison. We address this transparently:

1. **Same evaluation protocol:** Identical prompts, deterministic inference (temp=0), same scoring pipeline.
2. **Quantization effects:** Q4_K_M preserves ~99% of full-precision accuracy on standard benchmarks. We note quantization as a caveat, not a confound.
3. **The comparison IS the point:** The question isn't "which is more accurate?" (SB1 handles that). The question is "do different architectures and scales produce different *learning strategies* and *feedback responses*?" A model's cognitive profile — rule-inducer vs. memorizer, feedback-learner vs. feedback-blind — is robust to minor accuracy differences from quantization.
4. **Internal consistency:** Each local model serves as its own baseline across conditions. FLR measures within-model change over turns. **All cognitive profile metrics (FLR, RII, HTR, EB, EOLR, MR) are within-model contrasts, robust to absolute accuracy differences from quantization or scale.**

### Scheduling

- **Week 1-2:** Run all local models first (free). Use for iteration, debugging, piloting.
- **Week 3:** Run Kaggle SDK models (quota). This is the only heavy-spend week.
- **Qwen3.5-35B-A3B first** (fastest, ~60-80 tok/s) for rapid iteration.
- **Qwen3-32B-think overnight** (slowest but most interesting data).
- **Gemma 3 27B** last for the Google-vs-Google comparison story.

---

## HUMAN BASELINES (v10.2 — informal, $0 cost)

**Scope: SB2 only.** Platform: Custom HTML/JS web tool hosted on GitHub Pages (free). 20 informal participants from personal network.

### Why Not Prolific

With 20 people and $0, the human data serves as a **directional reference baseline**, not a powered inferential study. This is more honest and more practical than an underpowered Prolific study pretending to be rigorous. The human trajectory line on the Gap Chart — clearly rising while model lines stay flat — is worth more than any p-value.

### Design: All-Correction, Within-Subjects Optional

- **Primary:** All 20 participants do the **Correction condition** on 1 STS instance (12 rounds, ~15 minutes)
- 10 participants get STS Instance A, 10 get Instance B (counterbalanced to check rule-set dependence)
- **Optional within-subjects bonus:** If time allows, each person does a *second* STS instance under **Practice-only** (no evaluation, just "the answer was X"). This gives 20 paired observations — roughly equivalent power to 40 between-subjects. Order effects noted as caveat.

**Why all 20 do Correction (not split across conditions):** With only 20 people, splitting 10-vs-10 between-subjects wastes power. The key comparison is **human correction trajectory vs. model correction trajectory** — that's the Gap Chart. Model data already provides practice-only baselines across 9 configurations. Decades of cognitive science establishes that humans use error signals (Hattie & Timperley 2007) — we don't need to re-prove that.

### The Web Tool (Day 1 build, ~2 hours)

Single HTML/JS file hosted on GitHub Pages:
1. Show training examples (static, same STS instances as model evaluation)
2. Present test input, text box for answer
3. On submit: compare to correct answer, show "Correct!" or "Incorrect. The correct output was ⟐◈⬡"
4. Next question → repeat for 12 rounds
5. Generate results summary at end that participants copy-paste and send (or log to Google Sheets via Apps Script)

Uses the same STS instances as the model evaluation — built alongside the STS generator on Day 1.

### Recruitment (rolling, parallel with model runs)

- **Week 1:** Build web tool. Run 3-5 people immediately as pilot.
- **Decision gate:** If humans improve on STS → run remaining 15-17 people on STS. If humans can't learn STS → pivot to micro-grammar for remaining participants. **Either outcome is pre-registered.**
- **Weeks 2-3:** Collect remaining participants as available. Don't block on this — recruit in parallel with model runs.
- Sources: friends, colleagues, acquaintances, social media — anyone willing to spend 15 minutes.

### Statistical Framing (honest)

Human baselines are informal (N=20, convenience sample) and serve as a directional reference, not a powered inferential comparison. We report descriptive statistics: mean accuracy per turn, visual trajectory comparison with model curves, and 95% bootstrap CIs resampling participants. The human-vs-model comparison is expected to be a large effect (the question is whether humans improve at all across 12 rounds — binary, not subtle), so N=20 is adequate for the descriptive purpose.

### Why SB2 Only

- The headline finding is the human-vs-model feedback gap. Human baselines strengthen this.
- SB1 human learning curves are well-documented in existing literature (Logan 1988, Johansen & Palmeri 2002). We cite these for the exemplar→rule transition pattern.
- If a human reference point is needed for SB1, recruit 5-10 informal volunteers and report as "informal human baseline, N=8." Don't pretend it's a powered study.

**Total human baseline cost: $0**

---

## OUTPUT VISUALIZATIONS

Five visualizations, ordered by impact. **The story is feedback learning — lead with it.**

### 1. The Gap Chart (SB2 — THE HEADLINE VISUALIZATION, COVER IMAGE)

Feedback learning gap per model. Human baseline included. Sorted by gap. **This is the image that wins votes.** Instantly readable: big bar = learns from corrections, small bar = doesn't.

```
Feedback Learning Gap (turns 7-12)
            ← No learning | Learning →

  Human          |████████████████████  +42%
  Claude S4      |████████             +18%
  Qwen3 (think) |███████              +16%
  Gemini Pro     |██████               +14%
  Qwen3 (no-th) |████                 +9%
  Gemini Flash   |███                  +7%
  Qwen3.5 MoE   |██                   +5%
  Llama 70B      |██                   +4%
  Gemma 3 27B    |█                    +2%
  Phi-4 14B      |                     +1%
                 0%                    50%
```

### 2. The 2×2 Decomposition Chart (SB2 — THE MECHANISM)

For each model, two side-by-side bars: evaluation effect and answer effect. Shows *why* the gap exists.

```
2×2 Factorial Decomposition
            ← Answer effect | Evaluation effect →

  Human          ████████|████████████  (balanced)
  Claude S4      ████████|███           (answer-driven)
  Gemini Pro     ███████ |██            (answer-driven)
  Qwen3 (think) ██████  |████          (mixed)
  Phi-4 14B      ██     |              (negligible both)
```

*Expected finding: models are primarily answer-driven (they benefit from seeing correct examples, not from the error signal). Humans show balanced effects. This is the novel decomposition.*

### 3. Cognitive Profile Radar (Cross-benchmark — THE SYNTHESIS)

5-axis radar/spider chart. Each model as a colored polygon. Human as dashed polygon. All 9 models in one frame.

```
                    Sample Efficiency
                          ▲
                         /|\
                        / | \
                       /  |  \
    Rule Induction ◄──/   |   \──► Feedback Learning
                     /    |    \
                    / ★---★---★ \    ★ = Human
                   / ●--/ | \--● \   ● = Model
    Heuristic     /  ●/   |   \●  \  Feedback
    Resistance ◄─/───●────┼────●───\─► Discrimination
                          |
    (axes: AULC, RII, 1-HTR, FLR, MR)
```

### 4. Feedback Trajectory Plot (SB2 — 5 conditions, supporting detail)

Accuracy by turn for each condition, one panel per model. Goes in the notebook.

```
Accuracy                        Claude Sonnet 4
  1.0 |
      |                     __.--- Explanation
      |                __--/.----- Correction
  0.5 |          ___--/.../------- Practice-only
      |     __--/../...../-------- Error-only
      |  _--/../...../   .------- Misleading
  0.0 |________________________________
      1  2  3  4  5  6  7  8  9 10 11 12
                Turn number
```

### 5. Learning Curves Plot (SB1 — supporting context, notebook only)

Accuracy vs. N examples. One line per model + literature human reference. Separate panels for Tiers 2, 3, 4.

```
Accuracy                    Tier 3
  1.0 |         ___--------  Human (literature)
      |      __/ .--------  Claude Sonnet 4
      |    _/ ../  .------  Gemini 3.1 Pro
  0.5 |  _/../   . ------  Qwen3-32B (think)
      | /../   .  . -----  Gemini 3 Flash
      |//..  .    .  ----  Phi-4 14B
  0.0 |________________________________
      2    4    8   16   32
                N examples
```

**Visualization hierarchy for the writeup:** Lead with the Gap Chart (cover image, headline). Follow with the 2×2 Decomposition (the mechanism). Radar chart in the writeup for synthesis. Trajectory and learning curves go in the notebook as supporting evidence.

---

## THE MONEY SENTENCE

**If FLR > 0 for some models (gradient):**
> "ALP reveals that frontier models cannot learn from being told they're wrong. A 2×2 factorial design — the first controlled decomposition of corrective feedback in in-context learning — isolates the evaluation signal from answer exposure and shows that models benefit from seeing correct examples but are nearly blind to error signals. The resulting cognitive profiles across 9 model configurations paint the most detailed picture to date of where current AI falls short of adaptive intelligence."

**If FLR ≈ 0 for ALL models (universal blindness — v10.1 first-class narrative):**
> "Frontier Models Cannot Learn from Their Mistakes: Evidence from 9 LLMs on Contamination-Proof Tasks. A 2×2 factorial decomposition of corrective feedback reveals that no tested model — across 4 architecture families, 5 scale points, and both thinking and non-thinking modes — shows meaningful learning from error signals. Models process corrections as additional examples, not as error-driven updates. The feedback blindness is universal, not graded."

**Why the null is arguably stronger than the gradient:** A finding of "some models learn a little" invites quibbles about effect sizes and sample noise. "No model learns from corrections, period" is a clean, headline-ready negative result with immediate implications for anyone building interactive AI systems. The 2×2 decomposition makes the null *causal* — it's not just "corrections don't help," it's "the error signal component adds zero beyond additional-example exposure." Prepare this narrative from Day 1, not as risk mitigation.

---

## PRE-REGISTERED HYPOTHESES

Written before running the full benchmark. Included in the Kaggle notebook for credibility. **Each hypothesis includes null-result interpretation — both outcomes are informative. v10.0: Trimmed from 18 to 13, cutting obvious or low-stakes hypotheses to sharpen the story.**

**SB1 — Learning Curves & Strategy:**
1. N50 varies across models (discriminatory power). Prediction: Claude Sonnet 4 ≤ Gemini Pro < Gemini Flash < Llama 70B.
   - *Null: All models have similar N50 → learning efficiency is homogeneous across architectures. Discriminatory power comes from other metrics.*
2. RII < 0.7 for all models at Tier 3, N=8 (models are not pure rule-inducers at medium difficulty).
   - *Null: RII ≥ 0.7 → models are stronger rule-inducers than expected. The feedback story becomes "models induce rules but still can't learn from corrections" — a sharper finding.*
3. HTR > 0.3 for at least 2 models at Tier 3, N=8 (some models actively learn heuristics).
   - *Null: Low HTR everywhere → models don't learn systematic shortcuts. Either they induce full rules (check RII) or they're guessing (check Type R accuracy).*
4. Human RII > model RII at Tier 3 based on literature benchmarks (humans are better rule-inducers).
   - *Null: Models match human RII → the human advantage is specifically in feedback integration, not rule induction.*

**SB1 — Ecological Validity:**
5. STS and micro-grammar rank-orders correlate at ρ > 0.5 (STS is ecologically valid).
   - *Null: ρ < 0.5 → format effects dominate learning measurement. Itself a novel finding about substrate sensitivity in ICL. Report STS and micro-grammar results separately.*

**SB2 — Corrective Feedback:**
6. FLR > 0 for at least one model (some model benefits from corrections beyond merely seeing examples).
   - *Null: FLR = 0 for ALL models → "All frontier models are feedback-blind on novel tasks." The strongest possible null result, directly answering the competition question.*
7. Human accuracy improvement across 12 correction rounds exceeds model improvement for all models. Measured as turn 7-12 mean minus turn 1-3 mean. Informal baseline, N=20.
   - *Null: Some model matches human improvement trajectory → headline-worthy positive finding about AI learning capability.*
8. FLR varies across models (discriminatory power). Prediction: same rank order as N50.
   - *Null: All models have similar FLR → feedback responsiveness is architecture-independent. The benchmark's discriminatory power comes from other axes (EB, EOLR, strategy profiles).*

**SB2 — The 2×2 Decomposition (THE KEY PREDICTIONS):**
9. Answer effect > Evaluation effect for all models (models primarily benefit from seeing correct examples, not from the error signal). This is the headline finding.
   - *Null: Evaluation effect ≥ Answer effect → models DO process error signals, not just additional examples. More optimistic finding. Either outcome is novel — nobody has decomposed this before.*
10. **EB is pre-registered as two-sided (Alazraki replication probe):** EB may be positive (explanation helps), zero (no effect), or negative (explanation hurts, replicating Alazraki et al. 2025 on novel substrates). A negative EB would extend Alazraki's finding beyond math reasoning to contamination-proof pattern induction.
    - *EB > 0: Explanation helps. Standard expectation.*
    - *EB ~ 0: Explanation adds nothing beyond correction.*
    - *EB < 0: Explanation actively hurts. Replicates Alazraki on novel substrates. The MORE interesting finding.*
11. EOLR ≈ 0 for most models (pure error signal triggers no improvement). If any model shows EOLR > 0, this extends Alazraki et al. 2025 to novel substrates.
    - *Null: EOLR > 0 → models improve from error signals alone. Remarkable finding about error-signal processing in transformers.*

**Cross-Benchmark:**
12. The 2D cognitive map (RII × FLR) reveals whether rule induction strategy and feedback responsiveness cluster or dissociate across models. Descriptive visualization — exploratory with N=9.
    - *Clustering: Suggests a general "learning quality" factor.*
    - *Dissociation: "Learning quality" is multi-dimensional — you need BOTH strategy profiling AND feedback testing to characterize a model. The more interesting finding.*

**Thinking Mode:**
13. Qwen3-32B-think shows higher RII than Qwen3-32B-no-think (explicit reasoning traces help rule induction). If not, thinking is decorative for in-context learning.
    - *Null: No RII difference → explicit chain-of-thought doesn't improve rule induction. Challenges the "thinking helps learning" assumption.*

**If predictions are wrong:** That's fine — pre-registration is about transparency, not accuracy. Wrong predictions are reported honestly. A finding like "all models show FLR ~ 0" is the most interesting null result. **ALP is designed to be valuable regardless of direction — both outcomes for every hypothesis generate publishable insights.**

---

## RISK REGISTER

| Risk | Prob. | Impact | Mitigation |
|---|---|---|---|
| No model shows feedback learning (FLR ~ 0 for all) | Medium | **Med-High → reframed as first-class outcome (v10.1)** | Not a risk — a prepared narrative. "Frontier Models Cannot Learn from Their Mistakes" is arguably the stronger headline (see dual money sentence above). Front-loaded SB2 pilot in Week 1 gives early signal. Discriminatory power insurance (4 axes) still applies. The 2×2 decomposition makes the null *causal*: "error signals add zero beyond additional-example exposure." |
| Strategy profiles show no signal (Type E/L same as Type R) | Medium | **Low** | Supporting context — if it fails, nothing is lost. Same items, same API calls. Drop strategy analysis from writeup, focus on feedback findings. |
| RII and HTR correlate (2D collapses to 1D) | High | Low | Report correlation explicitly. Show as ranked dot plot instead of scatter. 1D ranking is still novel and valuable. |
| Multi-turn SDK bugs | Low-Med | **Medium** | Single-prompt-with-history fallback built from Day 1. Pilot both in Week 1. Pick the reliable one. |
| All models show strong feedback learning | Low | Medium | Great — discriminatory power from degree + condition effects (5 conditions provide rich gradient). 2×2 decomposition becomes even more interesting. |
| EB < 0 (explanation hurts) | Medium | **Positive** | Replicates Alazraki et al. 2025 on novel substrates. Pre-registered as two-sided. Becomes a secondary finding. |
| A competitor builds similar eval | Low-Med | Medium | Moat: 2×2 factorial + 5 conditions + strategy decomposition + 9 models + cross-benchmark cognitive profiling + human baselines (informal but present) + pre-registration + DeepMind framework alignment + literature differentiation (LLF-Bench, FB-Bench, Self-Correction Bench, Likra). Hard to replicate this depth. |
| API quota insufficient | Low | Low | Budget is ~$391 total (no Prolific spend), within $500/month base quota. ~$109 breathing room. |
| Writeup exceeds 1,500 words | Medium | Moderate | Budget: 1,220 words allocated, 280-word buffer. Results-first drafting. |
| Structured output unavailable on SDK | Low | **Medium** | Day 1 test: 10 prompts with `schema=STSAnswer`. If fails → Tier 3 regex with 50-prompt gate (v8.0 fallback). Structured output is an optimization, not a dependency. |
| JSON mode fails on local models | Low | **Low** | Tier 3 regex fallback. Local models tested with 50-prompt gate. |
| Type E items not generated for some (STS, N) cells | Low | Low | STS generator constrained for ≥1 non-DIRECT rule + nested training sets → target >90% feasibility. Report fraction of valid cells. |
| Local model quantization skews comparison | Low | **Low** | Transparently noted as caveat. Within-model contrasts (FLR, RII, EB, EOLR, MR) unaffected by quantization. |
| Error-only condition shows zero signal | High | Low | Expected outcome — pure error signals are weak. The 2×2 factorial interpretation still holds: it shows the answer component dominates. This IS the finding. |
| SB2 primary tier miscalibrated | Medium | **High** | SB1 pilot calibrates tier selection. If ~40% zone doesn't exist at any tier, widen to ~25-55% range. Secondary tier provides backup. |
| Human FLR ≈ 0 on STS (alien task problem) | Medium | **High** | Week 1 pilot (3-5 people via web tool). If humans can't learn STS either → pivot to micro-grammar for remaining 15-17 participants. Pre-registered either way. |
| Tokenization confound across models | Low-Med | **Low** | Pre-screen symbols for tokenization consistency on accessible tokenizers. Note as limitation for closed models. Within-model contrasts unaffected. |
| Literature gap discovered by judges | Low (v10.0) | **High** | v10.0: Added LLF-Bench, FB-Bench, Self-Correction Bench to differentiation table. Sharpened "first" claims to the specific novelty (factorial decomposition on novel substrates), not the general space (interactive feedback). |

---

## WRITEUP SKELETON (1,500 words budget — v10.0 allocation)

| Section | Words | Content |
|---|---|---|
| Title + subtitle | 20 | "Can AI Models Learn from Their Mistakes? A 2×2 Factorial Decomposition of Error-Driven Learning across 9 Frontier LLMs" |
| Problem + What's New | 200 | Current evals tell you *how well* models learn — not *whether they can learn from being wrong*. Recent work tests feedback in various forms (LLF-Bench, MINT, FB-Bench, Alazraki et al.) but none apply controlled factorial decomposition on contamination-proof substrates. ALP fills this gap. Map to Morris et al. 2024 cognitive framework. The 2×2 IS the contribution — explain it clearly here. |
| Methods | 250 | STS design + contamination defense. 5 conditions, 2×2 factorial table. Key metrics (FLR, EB, EOLR, MR). SB1 briefly: learning curves + strategy decomposition as supporting context. 9 model configurations. Structured output extraction. Human baselines: 20 informal participants, correction condition, descriptive comparison. |
| Results & Insights | 650 | **Lead with the Gap Chart** (headline: humans learn from corrections, models don't). **The 2×2 Decomposition** (evaluation effect vs. answer effect — THE novel finding). EB direction (Alazraki replication?). Per-model highlights. Cognitive Profile Radar. Strategy profiles (brief). Thinking mode comparison (Qwen3). Pre-registered predictions vs. actuals. |
| Affiliations | 25 | Independent researcher / Hollis Health LLC |
| References | 75 | ~19 citations: Morris et al. 2024 (DeepMind cognitive framework), Rescorla & Wagner 1972, Hattie & Timperley 2007, Cheng et al. 2023 (LLF-Bench), Li et al. 2025 (FB-Bench), Tsui 2025 (Self-Correction Bench), Wang et al. 2024 (MINT), Hamdan & Yuret 2025 (Likra), Alazraki et al. 2025, MIR-Bench 2025, RULEARN/IDEA (Zhu et al. 2024), Alon et al. 2024, Logan 1988, Johansen & Palmeri 2002, Olsson et al. 2022 (induction heads), von Oswald et al. 2023 (ICL as gradient descent), Berko 1958, WILT (Wang et al. 2024), iolbench |
| **Total** | **1,220** | *280 words buffer for data-dependent details* |

**Title (gradient finding):** "Can AI Models Learn from Their Mistakes? A 2×2 Factorial Decomposition of Error-Driven Learning across 9 Frontier LLMs"

**Title (universal blindness finding):** "Frontier Models Cannot Learn from Their Mistakes: Evidence from 9 LLMs on Contamination-Proof Tasks"

**Drafting strategy:** Write Results section FIRST. It's the most important section and the hardest to compress. Build the rest of the writeup around the findings.

---

## IMPLEMENTATION TIMELINE

| Week | Deliverables | Hours | Spend |
|---|---|---|---|
| **Week 1 (Mar 18-24)** | **Day 1:** Email for quota top-up. Build llama.cpp with Metal, download all 4 local models (~63GB). **Build human baseline web tool** (~2 hours — single HTML/JS file, uses same STS instances, hosted on GitHub Pages). **GATE 0: Structured output + extraction validation** — Test `llm.prompt(..., schema=STSAnswer)` on 10 sample prompts per SDK model (structured output). Test JSON mode on Qwen3.5-35B-A3B (local). If structured output works → move on. If not → fall to regex with 50-prompt gate. **Pre-screen 20 candidate Unicode symbols** on accessible tokenizers, select 12. Build STS generator + solver + partial-rule simulator (with nested training sets, Type E feasibility constraint, zero-shot contamination check). Build Type E item generator. Build micro-grammar generator (7 grammars). Implement all 5 SB2 conditions (both multi-turn and single-prompt-with-history). **SB1 pilot on local models** (free): 3 tiers × 3 N-values × 5 STS × 5 items = 225 items on Qwen3.5-35B-A3B (fastest). Validate: (a) accuracy gradient, (b) Type E ≠ Type R accuracy at Tiers 2+, (c) partial-rule answers on Type L. **Calibrate SB2 starting conditions from pilot results.** Pilot both SB2 implementations on 3 STS instances → pick one. **Minimal SB2 pilot (v10.1 — front-loaded):** Run 3 STS instances × correction vs. practice-only on Qwen3.5-35B-A3B. Early read on whether the feedback effect exists. **Run 3-5 informal human pilots** (friends/colleagues) on STS correction via web tool. **Decision gate:** If humans improve on STS → recruit remaining 15-17 on STS. If not → pivot to micro-grammar for remaining participants. | 25-30 | $0 |
| **Week 2 (Mar 25-31)** | Run full SB1 + SB2 on 2 local models (Qwen3.5-A3B for speed + Qwen3-32B for depth) at calibrated tier. Run Qwen3-32B thinking vs no-think comparison. Compute RII, HTR, FLR, EOLR on local pilot data. Run micro-grammar probe on local models. Compute STS-micro-grammar rank-order correlation. **DISCRIMINATORY POWER CHECK:** Does the 2×2 decomposition show different patterns across models? Does FLR differ? If zero variation on ALL metrics, pivot narrative to "AI models are universally feedback-blind" (still publishable). **Human pilot decision gate:** If informal humans show FLR > 0 on STS → proceed with STS Prolific launch. If not → design micro-grammar human baselines. **Write pre-registered hypotheses.** | 20-25 | $0 |
| **Week 3 (Apr 1-7)** | Run Kaggle SDK models (4 models, SB1 + SB2). Continue rolling human participant recruitment (target: 20 total by end of week). Run remaining local models (Gemma 3 27B, Phi-4 14B). Begin analysis and visualization. Build Gap Chart + 2×2 Decomposition Chart + Radar Chart. | 15-20 | $391 |
| **Week 4 (Apr 8-16)** | Generate all visualizations. Compare pre-registered predictions to actuals. **If time permits:** Analyze thinking traces (keyword coding). Evaluate EB direction (Alazraki replication?). Write 1,500-word writeup (Results section FIRST, then build around). Polish Kaggle notebook. Create cover image (the Gap Chart). Submit before Apr 16 11:59 PM UTC. | 10-15 | $0 |
| **Total** | | **70-90 hrs** | **~$391** |

---

## WHAT TO BUILD FIRST (Day 1 Priority)

0. **Email for quota top-up** — BEFORE writing code.
1. **Build llama.cpp + download models** — Get local inference running. Qwen3.5-35B-A3B first (fastest for iteration). See `local_model_setup.md`.
1b. **Build human baseline web tool** (~2 hours) — Single HTML/JS file. Same STS instances as model evaluation. Host on GitHub Pages. Needs: show examples, text input, compare answer, show feedback, log results. Build alongside STS generator since it uses the same instances.
2. **GATE 0: Extraction validation**
   - Test `llm.prompt(..., schema=STSAnswer)` on 10 STS-formatted prompts per SDK model
   - Test JSON mode (`response_format={"type": "json_object"}`) on 10 prompts on Qwen3.5-35B-A3B
   - If structured output AND JSON mode work → extraction is solved, move on
   - If structured output fails → Tier 3 regex with 50-prompt validation gate (v8.0 fallback)
   - If JSON mode fails → Tier 3 regex with 50-prompt validation gate for local models
   - **This gate is faster than v8.0's 50-prompt-per-model requirement** because structured output eliminates most parsing risk
3. **Unicode symbol pre-screening** — Test 20 candidate symbols on accessible tokenizers (Qwen, Llama, tiktoken). Select 12 most consistent. 1-2 hours.
4. **STS Generator + Solver** — Python class. Must output: STS instances (with ≥1 non-DIRECT rule constraint), nested training examples, Type R/E/L test items with verified answers and partial-rule predictions. Include zero-shot contamination check.
5. **Partial-rule simulator** — Exception-blind, condition-blind, order-blind. Needed for Type L items AND for misleading feedback wrong answers in SB2.
6. **Type E item generator** — Uses nested training sets. Verify ≥90% feasibility rate on 50 test STS instances.
7. **Micro-grammar generator** — 7 micro-grammars with diverse rule types. Quick to build.
8. **All 5 SB2 conditions (both implementations)** — Multi-turn AND single-prompt-with-history. Mechanical prompt style. Pilot both on 3 STS instances → pick one.
9. **SB1 pilot on local models** — 225 items on Qwen3.5-35B-A3B. Free. Validate gradient + calibrate SB2 tier.
10. **Informal human pilot** — 3-5 participants on STS correction via web tool. Decision gate: if humans improve → recruit remaining 15-17 on STS. If not → pivot to micro-grammar.

**Gate: End of Week 1**
- If extraction works (any tier) AND gradient exists AND SB2 implementation picked → full speed ahead
- If extraction fails at all tiers: this is the existential risk. Debug until solved. Nothing else matters.
- If gradient exists but strategy items show no signal → drop strategy analysis, proceed with learning curves + feedback only
- If no gradient → STS rework, pause everything
- **SB2 calibration complete:** Record which tier gives ~35-45% baseline accuracy. Concentrate SB2 there.
- **SB2 implementation selected:** Multi-turn or single-prompt-with-history based on pilot.
- **SB2 early signal (v10.1):** Minimal SB2 pilot (3 STS, correction vs. practice-only) gives a preliminary FLR read. If FLR ≈ 0 even on this small sample → begin preparing the "universal feedback blindness" headline narrative immediately. If FLR > 0 → proceed with the gradient narrative. Either way, you know the story direction before investing 20+ hours in Week 2.
- **Human pilot results in hand:** Know whether to proceed with STS or pivot to micro-grammar for remaining 15-17 participants. Begin rolling recruitment immediately.
- **By end of Week 1 you should have preliminary local model results AND a preliminary SB2 signal at ZERO cost.**

**Gate: End of Week 2 (DISCRIMINATORY POWER)**
- Run SB2 all 5 conditions on 2 local models at calibrated tier (25 STS instances for core, 15 for Misleading)
- The 2×2 decomposition: does evaluation effect ≠ answer effect?
- If FLR or 2×2 pattern differs between models → proceed
- If ALL metrics identical across models → pivot narrative to "universal feedback blindness" (still novel, still publishable)
- **Do not wait until Week 4 to discover zero discrimination**
- **By end of Week 2 you should have preliminary results on 2-3 local models with ZERO quota spend.**

---

## What v10 refimnes on (our refinement direction to prevent thrashing on issues)

| Dimension | v8.0  | v9.0  | v10.0/10.1  | v10.2  |
|---|---|---|---|---|
| **Narrative** | ONE story: 2×2 factorial | ONE story (unchanged) | Dual-ready: gradient + "universal blindness" | **Unchanged** |
| **"First" claims** | "First benchmark to test..." (overclaims) | Same | Sharpened + defensible | **Unchanged** |
| **Literature gaps** | Missing LLF-Bench, FB-Bench, Self-Correction Bench | Same | All added + Hamdan & Yuret 2025 (Likra) | **Unchanged** |
| **Human baselines** | Prolific, $500, 50 people, between-subjects | Same | Same | **20 informal participants, $0, all-correction, within-subjects optional. Web tool on GitHub Pages. Honest framing as directional reference.** |
| **Human baseline design** | 25/condition × 2 conditions | Same | Same | **All 20 do Correction (key comparison is human-vs-model trajectory). Optional within-subjects Practice-only on second STS → 20 paired observations.** |
| **Statistical framing** | Inferential (power ~55%) | Same | Same | **Descriptive (trajectory + bootstrap CIs). N=20 adequate for large expected effect. No false claims of inferential power.** |
| **H7 (human comparison)** | Human FLR > model FLR | Same | Same | **Human accuracy improvement (turns 7-12 mean − turns 1-3 mean) > model improvement. Simpler, honest for N=20.** |
| **SB2 piloting** | Week 2 | Week 2 | Front-loaded to Week 1 | **Unchanged** |
| **Null-result narrative** | Risk register row | Risk register row | First-class dual headline | **Unchanged** |
| **Web tool** | N/A | N/A | N/A | **Day 1 build (~2 hrs). HTML/JS, GitHub Pages, same STS instances as models.** |
| **Total items** | 3,225 | 3,195 | 3,075 | **Unchanged** |
| **Budget** | ~$387 + $500 Prolific | ~$410 + $500 Prolific | ~$391 + $500 Prolific = ~$891 | **~$391 total ($0 human baselines). ~$109 breathing room.** |
| **Execution risk** | Medium | Lower | Lowest | **Lowest: $500 saved, no Prolific dependency, rolling recruitment** |
