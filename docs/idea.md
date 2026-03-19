# LESSON (Learning from Error Signals in Symbolic Operations) - Spec Sheet
## Kaggle "Measuring AGI" Competition | Learning Track
### Version 12.0

---

## ONE-SENTENCE PITCH

ALP applies the first controlled 2×2 factorial decomposition of corrective feedback to in-context learning — on contamination-proof symbolic tasks, isolating exactly which component of feedback (error signal vs. correct answer) drives any observed learning, profiled across 15+ model configurations (hypothesis-driven selection spanning scale, code-tuning, and reasoning-training) with human baselines and mechanistic probes into *why* models fail to use error signals.

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
| **CL-Bench** (Tencent/Fudan, Feb 2026) | Tests "context learning" on novel knowledge/rules beyond pre-training scope, including rule system application. Four categories: knowledge recall, analogy, rule application, creative writing. | CL-Bench is single-turn (no feedback loop), no factorial design, no strategy decomposition. Tests whether models can learn novel rules from context, not whether they can learn from *corrections*. ALP fills the specific interactive feedback gap CL-Bench doesn't touch. Complementary: CL-Bench strengthens the case that context learning is underexplored; ALP adds the feedback dimension. |
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

**Tier 2 — JSON Mode (local models via LM Studio / llama.cpp):**

LM Studio and llama.cpp servers support `response_format={"type": "json_schema"}`. Prompt the model to respond with `{"output": "..."}` and parse the JSON.

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

**Why 7 grammars is sufficient:** The correlation is computed over 15+ model-level rank positions. With 7 grammars × 15 items = 105 items per model, per-model accuracy estimates are stable enough for reliable ranking. The v9.0 expansion to 15 grammars bought marginal validity at the cost of 8+ hours of design and debugging time better spent on SB2.

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

**v11.0 addition:** Beyond measuring WHETHER models learn from feedback, SB2 now probes WHY they don't. Four mechanistic probe conditions (clean-context, prompted-correction, structured-correction, reformatted-correction) isolate the mechanism behind feedback blindness — is it context pollution from wrong answers, lack of metacognitive triggers, or format sensitivity to conversational vs. code-like error presentation?

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

**Clean-context implementation note (v11.0):** The clean-context condition is always implemented as single-prompt-with-history by design — it strips the model's wrong guesses and presents accumulated correct pairs as additional training examples:

```python
def clean_context_prompt(examples_text, accumulated_correct_pairs, test_input, turn_num):
    """Clean-context: correct pairs formatted as additional training examples."""
    additional = "\n".join([f"{inp} → {out}" for inp, out in accumulated_correct_pairs])
    return f"""Below are {8 + turn_num} examples of a symbolic transformation system.
Study the pattern, then predict the output for the final input.

{examples_text}
{additional}

Input: {test_input}
Respond with the output symbol sequence only."""
```

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

The conditions form a principled experimental design. The core four conditions constitute a **true 2×2 factorial** crossing evaluation signal × answer visibility — all in the same multi-turn conversational format:

|  | Answer NOT shown | Answer shown |
|--|--|--|
| **NOT evaluated** | **No-feedback** | **Practice-only** |
| **Evaluated** | **Error-only** | **Correction** |

**v11.0 change: No-feedback replaces the SB1 single-prompt baseline in the 2×2.** Previously, the "not evaluated / answer not shown" cell was the SB1 baseline, which used a different format (single-prompt, no multi-turn). This confounded format with condition. The new **no-feedback** condition uses the same multi-turn conversation but gives zero information after each guess — just "Next question." The conversation history accumulates (the model sees its own guesses) but receives no new information. This creates a clean 2×2 where format is held constant across all four cells.

The clean 2×2 decomposition becomes:
- **Answer effect** = (Practice-only + Correction)/2 − (No-feedback + Error-only)/2
- **Evaluation effect** = (Error-only + Correction)/2 − (No-feedback + Practice-only)/2
- **Interaction** = does evaluation help MORE when the answer is also shown?

The Explanation, Misleading, and Mechanistic Probe conditions extend the design beyond the 2×2:

**Core 2×2 conditions (run at full 25 instances on all production models):**

| Condition | What model receives after each attempt | Tests... |
|---|---|---|
| **Correction** | "Correct!" / "Incorrect, answer is X" | Core error-driven learning — can the model update from being told the right answer? |
| **Practice-only** | "The output for {input} is X. Next question." (correct answer shown, model's attempt NOT evaluated) | Control baseline — both correction and practice-only see the same correct input→output pairs. The ONLY difference is whether the model gets told "you were wrong." Isolates the error signal from additional-example effects. |
| **Error-only** | "Correct!" / "Incorrect." (evaluation given, correct answer NOT shown) | Whether the pure error signal — without new information — triggers deeper re-examination of initial examples. Bridges to Alazraki et al. 2025. If models improve from this alone, that's remarkable evidence of error-signal processing. |
| **No-feedback** | "Next question." (nothing revealed — no evaluation, no answer) | Pure multi-turn format baseline. Isolates whether the multi-turn conversation format itself (accumulating the model's own guesses in context) helps or hurts relative to SB1 single-prompt. **Critical for explaining the SB2 < SB1 gap.** |

**Extended conditions (run at full 25 instances on top 3-4 models):**

| Condition | What model receives after each attempt | Tests... |
|---|---|---|
| **Explanation** | "Correct!" / "Incorrect, answer is X because [rule]" | Whether *explanatory* feedback produces stronger learning than correction alone. Maps to Hattie & Timperley's feedback hierarchy. **Pre-registered as Alazraki replication probe** — EB may be negative (replicating Alazraki et al. 2025's finding that rationales can hurt LLM performance on novel substrates). **v12.0: Run on top 3 models only (not 3-4).** |
| **Misleading** | Correct feedback on 9/12 rounds, wrong feedback on 3/12 | Feedback *discrimination* — can the model detect and resist bad corrections? **v12.0: Reduced to 3-5 instances on 2 models only.** Interesting but not the story — the 2×2 is the headline. |

**Mechanistic probe conditions (run at 3-5 instances on 2-3 models, pilot only):**

| Condition | What model receives | Tests... |
|---|---|---|
| **Clean-context** | Same information as correction, but implemented as single-prompt-with-history where only correct pairs are included (formatted as additional training examples). The model's wrong guesses are stripped entirely. | **Context pollution hypothesis:** If clean-context >> correction, wrong answers in conversation history are poisoning performance. The model CAN use feedback information, but only when presented as clean examples. Also serves as a functional SB1-at-N=16 data point. |
| **Prompted-correction** | Same as correction, but after feedback: "Before answering the next question, briefly state what pattern you think explains why your previous answer was wrong." | **Metacognitive trigger hypothesis:** If prompted-correction > correction, models CAN process error signals but don't do so spontaneously — they need explicit prompting to reflect. |
| **Structured-correction** | Same information as correction, formatted as code test output: `TEST FAILED\nInput: ◈⬡⧫\nExpected: ⟐◈⬡\nYour output: ⟐⟐⟐\nDiff: position 1 (expected ◈, got ⟐)...` | **Training data format hypothesis:** If structured-correction > correction, models have learned to process error signals in code-like formats but not conversational ones. Direct implications for agentic feedback system design. |
| **Reformatted-correction** | Same feedback information, but correct answer formatted as an additional example: "Example: ◈⬡⧫ → ⟐◈⬡" (no "Incorrect" framing) | **Format sensitivity:** Isolates whether the model can parse conversational corrections vs. clean example format. |

**Why the 2×2 matters — THE CORE ANALYTICAL WEAPON:** The factorial design isolates two components of feedback with proper marginal averaging:
- **Evaluation effect** = (Error-only + Correction)/2 − (No-feedback + Practice-only)/2 = benefit of being told "you were right/wrong"
- **Answer effect** = (Practice-only + Correction)/2 − (No-feedback + Error-only)/2 = benefit of seeing the correct answer
- **Interaction** = Correction − Practice-only − Error-only + No-feedback = does evaluation help MORE when the answer is also shown?
- If Evaluation effect > Answer effect: the error signal itself is what drives learning
- If Answer effect > Evaluation effect: models mainly benefit from seeing more correct examples
- **This factorial decomposition has never been applied to corrective feedback in in-context learning.** It is the methodological contribution that makes ALP unique.

**Why no-feedback is critical (v11.0):** It directly explains the SB2 < SB1 gap observed in pilot data (correction at 38% vs. SB1 single-prompt at 50%). If no-feedback scores lower than SB1, the multi-turn format itself hurts (context pollution from the model's own wrong answers). If no-feedback matches SB1, the format is neutral and something else explains the gap. Without this cell, the 2×2 decomposition conflates format effects with feedback effects.

**Why clean-context is the key mechanistic probe:** In the correction condition, by Turn 7 the model sees 7 wrong answers interspersed with 7 correct answers. The wrong answers may interfere with pattern induction — the model tries to induce rules from a context containing both correct and incorrect mappings. Clean-context strips this pollution:
- If clean-context >> correction: Wrong answers poison performance. The model CAN use feedback information when presented cleanly. Finding: "Models can learn from additional examples but cannot distinguish correct from incorrect information in their own conversation history."
- If clean-context ≈ correction: Context pollution isn't the issue. The model genuinely hits a ceiling.
- If clean-context matches SB1 at N=16: The model follows its learning curve, and SB2 adds nothing beyond "more examples."

**Why practice-only must show the correct answer:** If practice-only said only "Next question" (no new information), any improvement in the correction condition could simply be "more examples help" — which SB1 already measures. By giving practice-only the same correct answer but without evaluating the model's attempt, we isolate the *error signal* as the variable.

**Why error-only is the Alazraki bridge:** Alazraki et al. 2025 found that LLMs perform better on math reasoning when shown incorrect examples without explanations than with explanations. If ALP's error-only condition (told wrong, no correct answer) shows improvement, that replicates Alazraki's finding on contamination-proof substrates — a novel extension. If not, the finding doesn't generalize beyond math. Either result is publishable.

**Misleading condition specification:**
- **Which 3 rounds are misleading:** Seeded per STS instance (seed = hash of STS spec). Fixed across models so results are comparable. Spread across early/middle/late turns (one from rounds 1-4, one from 5-8, one from 9-12) to test temporal effects.
- **What wrong answer is used:** The partial-rule answer (from the SB1 partial-rule simulator) where available — this is a *plausibly wrong* answer that a heuristic-learner might accept. Where no partial-rule answer exists for the test input, a random valid-length symbol sequence from the STS alphabet is used.

### Dataset (v11.0)

- **Primary tier** (pilot-calibrated, default Tier 3):
  - Core 2×2: 4 conditions × 25 STS instances × 12 turns = **1,200 exchanges**
  - Extended — Explanation: 1 condition × 25 STS instances × 12 turns = **300 exchanges** (top 3 models only)
  - Extended — Misleading: 1 condition × 3-5 STS instances × 12 turns = **36-60 exchanges** (2 models only, v12.0 reduction)
  - **Primary subtotal: 1,200-1,560 exchanges per model**
- **Mechanistic probes** (pilot only, 2-3 models):
  - 4 probe conditions × 3-5 STS instances × 12 turns = **144-240 exchanges**
- **Secondary tier** (adjacent, if budget):
  - Core 2×2: 4 conditions × 15 STS instances × 12 turns = **720 exchanges**
  - **Secondary subtotal: 720 exchanges**
- **Total SB2 per model: ~1,200-2,400 multi-turn exchanges** (depending on model tier)
- **Total SB2 across 15+ models: ~25,000-35,000 exchanges**

**v11.0 change:** Extended from 8 to 12 turns per sequence (spec always said 12, pilot used 8). More turns = more data per instance = more power per model. Added no-feedback as fourth core condition. Mechanistic probes (conditions 7-10) do not go on the leaderboard — they appear in "Technical Details" and "Results & Insights" as evidence about WHY FLR ≈ 0.

**v12.0 change:** Misleading condition reduced from 15 instances to 3-5 instances on 2 models only. Explanation condition limited to top 3 models. Thinking trace analysis cut entirely. These reductions concentrate effort on the core 2×2 factorial, which is the headline contribution.

**Critical:** Use the SAME 3 STS instances for ALL pilot runs across ALL models and conditions. Per-instance difficulty is controlled when comparing Gemini vs Llama vs DeepSeek on identical instances. Different instances for different models would confound model effects with instance effects.

### Scoring

1. **Feedback Learning Rate (FLR):** Linear regression slope of accuracy across turns 1-12 in correction condition, minus slope in practice-only condition.
   - FLR > 0: Model learns *more from corrections than from merely seeing examples*
   - FLR ~ 0: Error signal adds nothing beyond additional examples
   - FLR < 0: Being told you're wrong actively harms performance

2. **2×2 Factorial Decomposition (v11.0 — proper marginal averaging):**
   - **Answer effect** = (Practice-only + Correction)/2 − (No-feedback + Error-only)/2
   - **Evaluation effect** = (Error-only + Correction)/2 − (No-feedback + Practice-only)/2
   - **Interaction** = Correction − Practice-only − Error-only + No-feedback
   - All computed with bootstrap 95% CIs (resample STS instances)

3. **Explanation Benefit (EB):** Accuracy in explanation condition minus correction condition, averaged over turns 7-12.
   - EB > 0: Model benefits from knowing *why* it's wrong
   - EB ~ 0: Explanation adds nothing beyond correction
   - **EB < 0: Explanation actively hurts — replicating Alazraki et al. 2025 on novel substrates (pre-registered as a possible and publishable outcome)**

4. **Error-Only Learning Rate (EOLR):** Slope of accuracy in error-only condition. Measures whether the pure error signal triggers any improvement.
   - EOLR > 0: Model improves from error signals alone (re-examines initial examples)
   - EOLR ~ 0: Error signal without correction is useless

5. **Misleading Resistance (MR):** Accuracy on non-misleading rounds in misleading condition, relative to correction condition.
   - MR ~ 1.0: Appropriately filters bad feedback
   - MR < 1.0: Bad feedback poisons good feedback too

6. **Feedback Composite:** `0.40 * FLR_norm + 0.20 * EB_norm + 0.20 * EOLR_norm + 0.20 * MR_norm`

### Discriminatory Power Insurance (v11.0)

Even if the primary FLR ≈ 0 for all models (the most likely null result), the benchmark guarantees discrimination through multiple independent axes:

1. **Trajectory shape across conditions:** Even at FLR ≈ 0, the 12-turn accuracy profile may differ across conditions and models — one model might show a brief improvement on turns 3-5 before regressing, while another stays flat. The per-turn trajectory plot captures this visually.
2. **Model grouping effects (v11.0):** Even if individual FLR ≈ 0, the *distribution* of FLR across model groups (code-tuned vs chat, reasoning vs base) may differ. Permutation tests detect group-level effects that per-model tests miss.
3. **Condition-specific effects:** EB, EOLR, and MR are independent of FLR. Models may show identical FLR but diverge on explanation sensitivity, error-signal processing, or misleading resistance.
4. **Mechanistic probe results (v11.0):** Clean-context vs correction, structured vs conversational feedback — these reveal *why* FLR ≈ 0, even if they don't change the headline.
5. **SB1 strategy profiles:** RII, HTR, and learning curves provide 3+ additional discrimination axes at zero additional cost.
6. **The 2×2 decomposition itself:** Even if total feedback effect is small, the *ratio* of evaluation effect to answer effect can differ across models. One model might show 60% answer-driven while another shows 80% answer-driven — discriminatory even under small absolute effects.
7. **No-feedback baseline (v11.0):** The no-feedback condition reveals whether multi-turn format itself helps or hurts, independent of feedback content. This is a new discrimination axis.

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
- **Format-degraded:** No-feedback < SB1 single-prompt baseline — multi-turn format itself hurts (context pollution from own wrong answers)
- **Feedback-fragile:** Misleading crashes all-round accuracy
- **Explanation-sensitive:** Explanation rises faster than correction
- **Explanation-hurt:** Explanation rises SLOWER than correction (Alazraki replication)
- **Error-signal responder:** Error-only shows improvement (remarkable if observed)
- **Context-pollution-sensitive:** Clean-context >> correction — wrong answers in history poison performance (mechanistic probe)

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

**Important statistical note (v11.0):** With N=15+ model configurations, cross-model correlations become more meaningful than in v10.0 (N=9). The 2D cognitive map is still primarily a **descriptive visualization**, but Spearman's rho is now adequately powered for moderate effects. Additionally, the hypothesis-driven model grouping (code vs chat, reasoning vs base) enables permutation tests for group-level differences.

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

Each model is plotted as a point in this space, **color-coded by training type** (code-tuned, chat-tuned, reasoning-RL, base). Human baselines define the target quadrant (high RII, high FLR). The quadrant a model falls into is its **cognitive phenotype** — a compact characterization of how it learns. With 15+ models (v11.0), visual clustering by training type becomes the primary evidence: do code-tuned models cluster in a different quadrant than chat models?

### The Cognitive Profile Radar

For each model, construct a 5-axis radar chart:
1. **Sample Efficiency** (AULC normalized)
2. **Rule Induction** (RII at reference tier)
3. **Heuristic Resistance** (1 - HTR at reference tier)
4. **Feedback Learning** (FLR normalized)
5. **Feedback Discrimination** (MR normalized)

Human baseline as a dashed polygon. Each model as a colored filled polygon. At a glance, you see each model's "cognitive shape" — where it's strong and where it collapses.

### Thinking Trace Analysis — CUT (v12.0)

~~Qualitative thinking trace coding was planned as Week 4 enrichment.~~ **Cut in v12.0.** The quantitative RII is the primary strategy metric. Thinking trace coding is qualitative fluff that won't change the headline and consumes polish time better spent on the 2×2 writeup. Thinking traces are still *logged* (free data) and can be revisited post-submission if interesting.

---

## THE SB2 < SB1 GAP: INVESTIGATION AND MECHANISTIC PROBES (v11.0)

### The Finding

Pilot data shows SB2 correction accuracy at 38% is BELOW SB1 single-prompt accuracy at 50%, despite the model having MORE information (original 8 examples plus accumulated correct pairs from feedback). Something is actively hurting performance.

### Candidate Explanations

1. **Context pollution from wrong answers:** By Turn 7, the model sees 7 wrong answers interspersed with 7 correct answers. The wrong answers interfere with pattern induction — the model can't fully distinguish correct from incorrect mappings despite "Incorrect" labels.

2. **Multi-turn format degradation:** The conversational format itself (system prompt → user → assistant → user → ...) may be less effective for pattern induction than a single prompt with all examples presented together.

3. **Lost-in-the-middle effect:** As conversation grows, attention over original training examples decreases. The model loses track of early context.

4. **Novel substrate interaction:** The format effect may be specific to novel symbolic tasks (STS) and not present on familiar tasks.

### How the New Conditions Test Each Explanation

| Explanation | Test | Condition | Expected result if explanation is correct |
|---|---|---|---|
| Context pollution | Strip wrong answers from history | **Clean-context** | Clean-context >> correction |
| Format degradation | Compare no-feedback to SB1 single-prompt | **No-feedback** | No-feedback < SB1 baseline |
| Lost-in-the-middle | Re-present training examples at end of prompt before question | **(Manual test on 1 instance)** | Re-presentation helps |
| Substrate-specific | Run same protocol on familiar arithmetic task | **(Optional probe, if time)** | FLR > 0 on arithmetic, FLR ≈ 0 on STS |

### Format Sensitivity Sub-Experiment (v11.0)

Run 1 STS instance where feedback is formatted in three ways:
1. **Conversational:** "Incorrect. The correct output is ⟐◈⬡."
2. **Tabular:** "Input: ◈⬡⧫ | Correct: ⟐◈⬡ | Your answer: ⟐⟐⟐ | Result: FAIL"
3. **Example-formatted:** "Example: ◈⬡⧫ → ⟐◈⬡"

Same information content, different surface format. If (3) >> (1), the model can't parse conversational corrections into usable evidence.

### Framing for Writeup

This finding is NOT a flaw — it's an additional finding about how multi-turn conversation interacts with in-context learning. Frame as: "Models not only fail to learn from corrections, but the multi-turn correction process itself degrades performance relative to single-prompt learning. Clean-context analysis reveals that [context pollution / format degradation / attention decay] is the mechanism."

---

## COMPLETE BENCHMARK ARCHITECTURE

| Sub-Benchmark | Construct | Items per model | Weight | SDK Mode | Novelty |
|---|---|---|---|---|---|
| 1. Learning Curves + Strategy Profiles | Sample efficiency + learning strategy | 855 (+ SB1 N=16, N=32 for key models) | 30% | Single-prompt | **High** (strategy decomposition + micro-grammar validation) |
| 2. Corrective Feedback — Core 2×2 | Error-driven learning (4 core conditions) | 1,200 | 70% | Multi-turn conversation | **Very High** |
| 2b. Corrective Feedback — Extended | Explanation + Misleading | 480 (top 3-4 models) | (included in 70%) | Multi-turn conversation | High |
| 2c. Mechanistic Probes | Clean-context, prompted, structured, reformatted | 144-240 (2-3 models, pilot) | (not scored) | Mixed | **Very High** |
| **TOTAL per model** | | **~1,200-2,535** | 100% | | |
| Cross-Benchmark Analysis | Cognitive profiling (RII × FLR) + model grouping tests | 0 (derived) | N/A | Analysis only | **Very High** |

*v11.0: Added no-feedback as 4th core condition. Extended to 12 turns. Added 4 mechanistic probe conditions. SB1 extended with N=16/N=32 for key models. Total items vary by model tier (core models get full battery, broad-scan models get SB1 only).*

**Budget math (updated 2026-03-19 — post-SB2 pilot):**
- **Spent so far:**
  - Gemini Flash SB2 pilot (Day 1): ~$0.05
  - OpenRouter SB1 broad scan (19 models, 60 items each): <$10
  - OpenRouter SB2 pilot (8 models, N=3, 4 core conditions): ~$15-20
  - **Total spent: ~$25-30**
- **Remaining estimates:**
  - Gemini Flash N=25 production run (via Gemini API, 4 conditions): ~$0.11
  - Kaggle SDK production (5 models, N=25, 4 core conditions + extended): ~$200-400 (reserved from $500 Kaggle allocation)
  - **Total remaining: ~$200-400**
- Human baselines: **$0** (informal participants, GitHub Pages hosting)
- Local models: **$0** (RTX 5090 via LM Studio)
- **Total project estimate: ~$225-430**
- **Strategy:** Run Gemini Flash at full statistical power (N=25) first via cheap Gemini API (~$0.11). Use those results to validate the eval protocol and refine hypotheses before spending any Kaggle SDK budget on the 5-model production run. Kaggle $500 allocation is reserved exclusively for the final production run — no exploratory spending.

**Note:** Gemini-3.1-Pro consumed >50% of SB1 scan cost due to long reasoning traces (20K max_tokens). For SB2, use Gemini Flash where possible to manage cost. Kaggle SDK models run last to maximize remaining quota.

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

# Sub-benchmark 2: Corrective Feedback (10 conditions: 4 core + 2 extended + 4 mechanistic)
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
        elif condition == "no_feedback":
            feedback = "Next question."
        elif condition == "explanation":
            feedback = f"Correct! The output is {correct_out}." if is_correct \
                else (f"Incorrect. The correct output is {correct_out}. "
                      f"This is because {rule_explanations[i]}.")
        elif condition == "practice_only":
            feedback = f"The output for {inp} is {correct_out}. Next question."
        elif condition == "error_only":
            feedback = "Correct!" if is_correct else "Incorrect."
        elif condition == "prompted_correction":
            if is_correct:
                feedback = f"Correct! The output is {correct_out}."
            else:
                feedback = (f"Incorrect. The correct output is {correct_out}. "
                           f"Before answering the next question, briefly state what "
                           f"pattern you think explains why your previous answer was wrong.")
        elif condition == "structured_correction":
            if is_correct:
                feedback = f"TEST PASSED\nInput: {inp}\nExpected: {correct_out}\nYour output: {answer}\nResult: PASS"
            else:
                feedback = f"TEST FAILED\nInput: {inp}\nExpected: {correct_out}\nYour output: {answer}\nResult: FAIL"
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

### Strategic Principle (v11.0 — hypothesis-driven selection)

Each model tests a **specific hypothesis** about what drives in-context learning and feedback sensitivity. Two-phase approach: broad scan → filtered deep dive.

### Hypothesis-Driven Selection Framework

**Hypothesis 1: Scale matters (within-family comparisons)**
- Claude Haiku 4.5 vs Sonnet 4.6 vs Opus 4.6 (same family, scaling)
- GPT-5.3-Chat vs GPT-5.4-Mini (different scale points)

**Hypothesis 2: Code training enables error signal processing**
- GPT-5.3-Codex vs GPT-5.3-Chat (code-tuned vs general, same base model)
- This is the most exciting hypothesis. Code models see millions of error→fix sequences in training. If they show FLR > 0 while chat models show FLR ≈ 0: **"Error-signal sensitivity is not an architectural limitation — it's a training data gap."**

**Hypothesis 3: Reasoning training (RL) changes feedback processing**
- DeepSeek-R1 vs DeepSeek-V3.2 (reasoning RL vs base, same family)
- RL-trained reasoning models are optimized to evaluate their own outputs. If they show better feedback sensitivity, RLHF/RLVR partially teaches error processing.

**Hypothesis 4: Architecture differences**
- GLM-5, MiniMax-M2.7, Kimi-K2.5, Grok-4.20 — diverse architectures from different labs
- Qwen-3.5-397B MoE vs dense models at similar capability

### Phase 1 — Broad Scan: COMPLETE (v1.5.0, 2026-03-19)

**19 models scanned** via OpenRouter (16) + LM Studio (5, 2 excluded for context issues). 60 STS items per model (T1/T2 x N=4/N=8 x 3 instances x 5 items). Cost: <$10 total.

**Filter:** T2 N=8 accuracy between 15% and 70%. **16 of 19 models pass.**

See `docs/observations.md` for full results table.

**SB2 pilot (N=3) also complete.** 8 models × 4 core conditions × 3 instances × 12 turns = 1,152 exchanges. All results in SQLite DB (`results/lesson_bench.db`). Pilot confirmed pipeline works end-to-end but N=3 is statistically insufficient — 95% CIs on FLR are ±0.066, wider than the entire observed range of effects. Directional patterns only: Claude Sonnet 4.6 and DeepSeek-R1 show positive FLR, GPT-5.3 Chat shows negative FLR, but none are statistically significant at N=3.

### Phase 2 — SB2 Pilot (8 models, COMPLETE at N=3)

| Model | T2N8 | Hypothesis | Why |
|-------|------|-----------|-----|
| **GLM-5** | 67% | Ceiling reference | Highest SB1 performer |
| **GPT-5.3-Codex** | 58% | Code training (H2) | Compare to Chat |
| **GPT-5.3-Chat** | 50% | Code training (H2) | Compare to Codex |
| **Gemini-3.1-Flash*** | — | Architecture diversity | Kaggle SDK model, Google judge appeal |
| **Claude-Sonnet-4.6** | 42% | Architecture diversity | Different model family |
| **DeepSeek-R1** | 33% | Reasoning RL (H3) | Compare to V3.2 |
| **DeepSeek-V3.2** | 33% | Reasoning baseline (H3) | Same family, NOT reasoning-trained |
| **Claude-Haiku-4.5** | 33% | Scale (H1) | Scale comparison within Claude family |

*Gemini-3.1-Pro used >50% of total SB1 run cost due to long reasoning traces. Flash provides a Gemini architecture test at manageable cost.

**Deferred:** Grok (incomplete data), Flash-Lite (too low), Kimi/MiniMax (don't test unique hypotheses), Qwen-3-Coder local (anomalous). Can add back for production if pilot reveals something worth chasing.

### Phase 2.5 — Gemini Flash Production (N=25, NEXT)

Run Gemini 3 Flash at full statistical power (N=25, 4 core conditions) via Gemini API before touching the Kaggle budget. Flash is cheap (~$0.11) and serves as the baseline model (Google judge appeal). With N=25 we get 4% accuracy resolution per turn (vs 33% at N=3), smooth trajectory curves, and 95% CIs tight enough to detect effects as small as ±2.3 percentage points on FLR. This run informs:
- Whether the eval protocol produces clean, interpretable results at scale
- Whether Flash shows any feedback learning signal (baseline expectation: FLR ≈ 0)
- Whether any modifications are needed before the expensive 5-model Kaggle run
- Refinement of hypotheses H1-H18 based on the first real data

### Phase 3 — Production (5 models via Kaggle SDK, $500 allocation)

Run full 25-instance SB2 (4 core conditions) on 5 selected models via Kaggle Benchmarks SDK. Model selection informed by Flash N=25 results + pilot directional patterns. Kaggle $500 is reserved exclusively for this phase — no exploratory spending.

### Kaggle SDK Models (use quota — $50/day, $500/month)

| Model | SDK String | Cost Tier | Hypothesis |
|-------|-----------|-----------|------------|
| **Gemini 3 Flash** | google/gemini-3-flash | Low | Baseline. Judges work at Google. |
| **Gemini 3.1 Pro** | google/gemini-3.1-pro-preview | Medium | Scale (Flash vs Pro). Expensive due to long reasoning traces. |

### OpenRouter Models (SB1 complete, SB2 pilot ~$15-20)

| Model | T2N8 | SB2 Slot | Hypothesis |
|-------|------|----------|-----------|
| **GLM-5** | 67% | Yes | Ceiling reference |
| **GPT-5.3-Codex** | 58% | Yes | Code training (H2) |
| **GPT-5.3-Chat** | 50% | Yes | Code training baseline (H2) |
| **MiniMax-M2.7** | 50% | Deferred | Architecture diversity |
| **Qwen-3.5-397B** | 50% | Deferred | Scale (large MoE) |
| **Gemini-3.1-Pro** | 42% | Deferred (cost) | Scale (Google) |
| **Claude-Opus-4.6** | 42% | Deferred | Scale (Claude) |
| **Claude-Sonnet-4.6** | 42% | Yes | Architecture diversity |
| **DeepSeek-V3.2** | 33% | Yes | Reasoning baseline (H3) |
| **Claude-Haiku-4.5** | 33% | Yes | Scale (H1, Claude) |
| **GPT-5.4-Mini** | 33% | Deferred | Scale (OpenAI) |
| **DeepSeek-R1** | 33% | Yes | Reasoning RL (H3) |
| **Kimi-K2.5** | 33% | Deferred | Architecture diversity |
| **Gemini-3.1-Flash-Lite** | 17% | Deferred | Floor reference |
| **Grok-4.20** | 17% | Deferred | Incomplete data |

### Local Models (FREE — RTX 5090 32GB via LM Studio)

| Model | T2N8 | Status | Notes |
|-------|------|--------|-------|
| **Qwen-3-Coder-30B-A3B** | 17% | Working | 7-12s per call, passes SB2 filter |
| **Qwen-3-1.7B** | 8% | Working | Too weak for SB2 (FAIL) |
| **Qwen-3.5-27B** | 0% | Broken | Context window too small for thinking mode |
| **Qwen-3.5-27B-NoThink** | 0% | Broken | Still hits context limits on N=8 |
| **GLM-4.7-Flash** | 0% | Broken | All empty responses, context issue |

**Total scanned: 22 configurations (16 OpenRouter + 5 LM Studio + 1 excluded). 16 pass SB2 filter. 8 selected for SB2 pilot.**

### Per-Model Documentation

For each model, record in a table: Name, parameter count, active params (if MoE), architecture type, training method (base/SFT/RLHF/RLVR/code), context length, and which hypothesis it tests. This table goes in the writeup to show judges the selection was scientifically motivated.

### Local Model Integration

Local models run via LM Studio (v0.4.0+) exposing an OpenAI-compatible API on port 1234. See `lesson/models/lmstudio.py` for the client implementation.

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# Same evaluation pipeline as OpenRouter models: identical prompts, same scoring.
# max_tokens=2048 for LM Studio (constrained by local context window).
# JSON schema mode (response_format=json_schema) supported.
```

Local models use the **same evaluation pipeline** as API models: identical prompts, same scoring, same extraction tiers (JSON schema → plain prompt fallback). The only differences are the model itself and max_tokens (2,048 vs 20,000 for OpenRouter).

### Comparability Note

Comparing quantized local models (Q4_K_M, Q5_K_M) with full-precision API models is not a perfect apples-to-apples comparison. We address this transparently:

1. **Same evaluation protocol:** Identical prompts, deterministic inference (temp=0), same scoring pipeline.
2. **Quantization effects:** Q4_K_M preserves ~99% of full-precision accuracy on standard benchmarks. We note quantization as a caveat, not a confound.
3. **The comparison IS the point:** The question isn't "which is more accurate?" (SB1 handles that). The question is "do different architectures and scales produce different *learning strategies* and *feedback responses*?" A model's cognitive profile — rule-inducer vs. memorizer, feedback-learner vs. feedback-blind — is robust to minor accuracy differences from quantization.
4. **Internal consistency:** Each local model serves as its own baseline across conditions. FLR measures within-model change over turns. **All cognitive profile metrics (FLR, RII, HTR, EB, EOLR, MR) are within-model contrasts, robust to absolute accuracy differences from quantization or scale.**
5. **Within-family comparisons are cleanest:** The hypothesis-driven selection (v11.0) prioritizes within-family comparisons (DeepSeek-V3 vs DeepSeek-Coder-V2, GPT-4o-mini vs o1-mini) where the only variable is training method, not architecture or scale.

### Scheduling (updated 2026-03-19)

- **Day 1 (Mar 18): DONE.** STS generator, evaluation pipeline, SB2 pilot framework, LM Studio + OpenRouter clients built and validated. Gemini Flash SB2 pilot (3 instances, 3 conditions) completed.
- **Day 2 (Mar 19): DONE.** Full SB1 broad scan across 19 models (16 OpenRouter + 5 LM Studio). 16 pass filter. Infrastructure bugs fixed (log pruning race, Unicode, context size, JSON schema caching). 8-model SB2 pilot selection complete. SB2 pilot (8 models, N=3, 4 core conditions) completed via OpenRouter. Full codebase refactoring: consolidated model configs into registry, extracted shared runner infrastructure (CircuitBreaker, retry_with_backoff, resume), built SQLite results store, unified CLI (`python -m lesson`), filled package stubs, 160 tests. Built analysis pipeline with publication-quality PDF report generation (gap chart, 2×2 factorial, trajectory plots, Codex-vs-Chat comparison, model grouping). All pilot results imported to SQLite DB.
- **Day 3 (Mar 20):** Code inspection and cleanup. Gemini 3 Flash production run (N=25, 4 core conditions) via Gemini API. This is the first statistically powered run (~3h, ~$0.11). Analyze Flash results for real signal before committing Kaggle budget.
- **Days 4-5 (Mar 21-22):** Analyze Flash N=25 results. Evaluate whether eval protocol needs modification. Finalize 5-model selection for production. Adjust hypotheses based on Flash findings.
- **Days 6-10 (Mar 23-27):** Production SB2 runs (N=25, 4 core conditions) on 5 selected models via Kaggle Benchmarks SDK ($500 allocation). Extended conditions on top 3.
- **Days 11-14 (Mar 28-31):** Full statistical analysis. Cross-benchmark cognitive profiling. Visualizations.
- **Days 15-28 (Apr 1-16):** Polish, writeup, submission.

---

## HUMAN BASELINES (v10.2 — informal, $0 cost)

**Scope: SB2 only.** Platform: Custom HTML/JS web tool hosted on GitHub Pages (free). 20 informal participants from personal network.

### Why Not Prolific

With 20 people and $0, the human data serves as a **directional reference baseline**, not a powered inferential study. This is more honest and more practical than an underpowered Prolific study pretending to be rigorous. The human trajectory line on the Gap Chart — clearly rising while model lines stay flat — is worth more than any p-value.

### Design: Correction + Clean-Context Within-Subjects (v11.0)

- **Primary:** All 20 participants do the **Correction condition** on 1 STS instance (12 rounds, ~15 minutes)
- 10 participants get STS Instance A, 10 get Instance B (counterbalanced to check rule-set dependence)
- **Within-subjects clean-context condition (v11.0):** Each person does a *second* STS instance under **Clean-context** (no evaluation shown — just accumulating examples without showing the human's wrong answer back to them). This gives 20 paired observations for the context pollution question.

**Why correction + clean-context (not correction + practice-only):** The v11.0 plan adds the context pollution hypothesis as a major mechanistic probe. By running the SAME conditions on humans:
- If humans are unaffected by seeing their own wrong answers but models are degraded: human-vs-AI difference in filtering ability (novel finding).
- If humans are ALSO degraded: context pollution is a universal cognitive effect, not model-specific.
- This strengthens the Gap Chart narrative with a mechanistic dimension.

**Why all 20 do Correction (not split across conditions):** With only 20 people, splitting 10-vs-10 between-subjects wastes power. The key comparison is **human correction trajectory vs. model correction trajectory** — that's the Gap Chart. Model data provides practice-only baselines across 15+ configurations. Decades of cognitive science establishes that humans use error signals (Hattie & Timperley 2007) — we don't need to re-prove that.

### The Web Tool (Day 1 build, ~2 hours)

Single HTML/JS file hosted on GitHub Pages. Accepts any STS instance definition as a **JSON config** so it can be reused across different instances and conditions:
1. Load STS config (training examples, test sequence, condition type) from JSON
2. Show training examples (static, same STS instances as model evaluation)
3. Present test input, text box for answer
4. On submit: compare to correct answer, show condition-appropriate feedback:
   - Correction: "Correct!" or "Incorrect. The correct output was ⟐◈⬡"
   - Clean-context: Just shows the correct pair as a new example, no evaluation
5. Next question → repeat for 12 rounds
6. Log per-turn responses and timestamps
7. Generate results summary at end that participants copy-paste and send (or log to Google Sheets via Apps Script)

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

5-axis radar/spider chart. Each model as a colored polygon. Human as dashed polygon. Top 8-10 models in one frame (full set in notebook appendix).

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

### 4. Feedback Trajectory Plot (SB2 — all conditions, supporting detail)

Accuracy by turn for each condition, one panel per model. Goes in the notebook.

```
Accuracy                        Claude Sonnet 4
  1.0 |
      |                     __.--- Explanation
      |                __--/.----- Correction
  0.5 |          ___--/.../------- Practice-only
      |     __--/../...../-------- Error-only
      |  _--/../...../   .------- No-feedback
  0.0 |________________________________
      1  2  3  4  5  6  7  8  9 10 11 12
                Turn number
         ---- SB1 single-prompt baseline (dashed reference line)
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

### 6. Model Grouping Comparison (v11.0 — THE TRAINING DATA STORY)

Box plots of FLR grouped by training type: code-tuned vs. chat-tuned, reasoning-RL vs. base. Permutation test p-values annotated. If code-tuned models show systematically higher FLR, this is visually immediate.

```
FLR by Training Type
                Code-tuned    Chat-tuned    Reasoning-RL    Base
  +0.10  |         ┬              ┬               ┬            ┬
         |         │              │               │            │
  +0.05  |      ┌──┤           ┌──┤            ┌──┤         ┌──┤
         |      │  │           │  │            │  │         │  │
   0.00  |    ──┤  │         ──┤  │          ──┤  │       ──┤  │
         |      │  │           │  │            │  │         │  │
  -0.05  |      └──┤           └──┤            └──┤         └──┤
         |         ┴              ┴               ┴            ┴
```

**Visualization hierarchy for the writeup:** Lead with the Gap Chart (cover image, headline). Follow with the 2×2 Decomposition (the mechanism). Model Grouping for the training data story. Radar chart in the writeup for synthesis. Trajectory and learning curves go in the notebook as supporting evidence.

---

## THE MONEY SENTENCE

**If FLR > 0 for some models (gradient):**
> "ALP reveals that frontier models cannot learn from being told they're wrong. A 2×2 factorial design — the first controlled decomposition of corrective feedback in in-context learning — isolates the evaluation signal from answer exposure and shows that models benefit from seeing correct examples but are nearly blind to error signals. The resulting cognitive profiles across 15+ model configurations paint the most detailed picture to date of where current AI falls short of adaptive intelligence."

**If FLR ≈ 0 for ALL models (universal blindness — v10.1 first-class narrative):**
> "Frontier Models Cannot Learn from Their Mistakes: Evidence from 15+ LLMs on Contamination-Proof Tasks. A 2×2 factorial decomposition of corrective feedback reveals that no tested model — across 6 architecture families, multiple scale points, code-tuned and reasoning-tuned variants — shows meaningful learning from error signals. Models process corrections as additional examples, not as error-driven updates. The feedback blindness is universal, not graded."

**If code-tuned models show FLR > 0 but chat models don't (v11.0 — the training data hypothesis):**
> "Error-signal sensitivity is not an architectural limitation — it's a training data gap. Code-trained models, exposed to millions of error→fix sequences during training, show measurable feedback learning on contamination-proof symbolic tasks where chat-trained models show none. This has direct implications for agentic AI: feedback-sensitive systems may need code-style error formatting, not conversational corrections."

**Why the null is arguably stronger than the gradient:** A finding of "some models learn a little" invites quibbles about effect sizes and sample noise. "No model learns from corrections, period" is a clean, headline-ready negative result with immediate implications for anyone building interactive AI systems. The 2×2 decomposition makes the null *causal* — it's not just "corrections don't help," it's "the error signal component adds zero beyond additional-example exposure." Prepare this narrative from Day 1, not as risk mitigation.

---

## PRE-REGISTERED HYPOTHESES

Written before running the full benchmark. Included in the Kaggle notebook for credibility. **Each hypothesis includes null-result interpretation — both outcomes are informative. v11.0: Expanded from 13 to 18, adding mechanistic probes (H14-H18) motivated by pilot data showing SB2 < SB1 gap.**

**SB1 — Learning Curves & Strategy:**
1. N50 varies across models (discriminatory power). **SB1 scan confirms:** T2N8 accuracy ranges from 67% (GLM-5) to 0% (Llama-4-Maverick) — strong discriminatory power.
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

**v11.0 — New Hypotheses (Mechanistic Probes + Model Selection):**
14. Clean-context > correction for all models (context pollution hypothesis). The model's own wrong answers in conversation history poison pattern induction.
    - *Null: Clean-context ≈ correction → context pollution isn't the issue. The model genuinely hits a ceiling regardless of how information is presented.*
15. Code-tuned models (GPT-5.3-Codex) show higher FLR than their chat-tuned counterparts (GPT-5.3-Chat). Code training teaches error-signal processing through millions of error→fix sequences. **SB1 directionally supports:** Codex (58%) > Chat (50%) at T2N8.
    - *Null: Code-tuned ≈ chat-tuned on FLR → feedback blindness is architecture/scale-independent, not a training data gap. This would strengthen the "universal blindness" narrative.*
16. Reasoning-tuned models (DeepSeek-R1) show higher FLR than base models (DeepSeek-V3.2). RL training partially teaches error processing. **SB1 note:** R1 shows anomalous *negative* learning at T1 (53% → 20% with more examples) — overthinking effect?
    - *Null: Reasoning-tuned ≈ base on FLR → RL training doesn't transfer to in-context error-signal processing.*
17. Structured-correction ≈ correction (format is not the issue) OR structured-correction > correction (format sensitivity). Tests whether models have learned to process error signals in code-like formats but not conversational ones.
    - *Either direction is informative. If format matters → direct implications for agentic feedback system design.*
18. No-feedback < all other conditions (information from feedback IS used, just not differentially by type). The multi-turn format itself may hurt via context pollution from wrong answers.
    - *Null: No-feedback ≈ correction → feedback adds nothing beyond the multi-turn format. The strongest possible null.*

**Thinking Mode:**
13. Qwen3-32B-think shows higher RII than Qwen3-32B-no-think (explicit reasoning traces help rule induction). If not, thinking is decorative for in-context learning.
    - *Null: No RII difference → explicit chain-of-thought doesn't improve rule induction. Challenges the "thinking helps learning" assumption.*

**If predictions are wrong:** That's fine — pre-registration is about transparency, not accuracy. Wrong predictions are reported honestly. A finding like "all models show FLR ~ 0" is the most interesting null result. **ALP is designed to be valuable regardless of direction — both outcomes for every hypothesis generate publishable insights.**

---

## STATISTICAL ANALYSIS INFRASTRUCTURE (v11.0 — Build Once, Use Throughout)

Build a single Python analysis module that all experiments pipe results through. Takes a standardized results JSONL and produces all outputs. Every time a new model is run, feed its results in and get updated figures.

### Logging Standard

For every turn of every condition of every model, save to JSONL: complete prompt sent, complete raw response, extracted answer, correct answer, match boolean, timestamp, model name, condition name, instance ID, turn number. One JSONL file per model. This is the audit trail and analysis input.

### Per-Condition Metrics
- Accuracy per turn with 95% bootstrap CIs (resample instances, not turns — instances are the independent unit)
- Slope of accuracy across turns (linear regression) with CI on the slope
- Cohen's d for each pairwise condition comparison (correction vs practice-only, etc.)

### The 2×2 Decomposition (v11.0 — proper marginal averaging)
- Answer effect = mean(practice-only, correction) − mean(no-feedback, error-only), with bootstrap CI
- Evaluation effect = mean(error-only, correction) − mean(no-feedback, practice-only), with bootstrap CI
- Interaction term = Correction − Practice-only − Error-only + No-feedback

### Per-Model Profile
- AULC from SB1 (area under learning curve, normalized)
- RII (Type E accuracy / Type R accuracy) with CI
- FLR (correction slope minus practice-only slope) with CI
- All metrics on same scale for radar chart

### Cross-Model Analysis
- Rank-order correlations (Spearman) between SB1 metrics and SB2 metrics across models
- Permutation test for whether model groupings (code vs chat, reasoning vs standard) differ on FLR
- Mixed-effects model: accuracy ~ turn × condition × model_group + (1|instance) for the full dataset

### Visualization Pipeline
1. **Gap chart** (the headline viz — FLR per model, sorted, with CIs)
2. **2×2 decomposition chart** (answer effect vs evaluation effect per model)
3. **Per-turn trajectory plot** (all conditions, one panel per model)
4. **Learning curve plot** (SB1, accuracy vs N, per model)
5. **Radar chart** (5 axes per model)
6. **Model grouping comparison** (code-tuned vs chat, reasoning vs base — box plots of FLR by group)

### SB1 Extended Data Points (v11.0)

Run SB1 at **N=16 and N=32** on at least Gemini Flash (using the 5 existing STS instances from the original SB1 pilot). This provides the comparison point needed to interpret the SB2 < SB1 gap:
- If SB1 T2 N=16 ≈ SB2 correction at Turn 8 (~38%): the model follows its learning curve and feedback adds nothing beyond "more examples."
- If SB1 T2 N=16 >> SB2 correction at Turn 8: multi-turn format actively hurts beyond just the learning curve.
- The clean-context condition at Turn 8 is functionally equivalent to SB1 at N=16 — this validates the comparison.

---

## RISK REGISTER

| Risk | Prob. | Impact | Mitigation |
|---|---|---|---|
| No model shows feedback learning (FLR ~ 0 for all) | Medium | **Med-High → reframed as first-class outcome (v10.1)** | Not a risk — a prepared narrative. "Frontier Models Cannot Learn from Their Mistakes" is arguably the stronger headline (see dual money sentence above). Front-loaded SB2 pilot in Week 1 gives early signal. Discriminatory power insurance (4 axes) still applies. The 2×2 decomposition makes the null *causal*: "error signals add zero beyond additional-example exposure." |
| Strategy profiles show no signal (Type E/L same as Type R) | Medium | **Low** | Supporting context — if it fails, nothing is lost. Same items, same API calls. Drop strategy analysis from writeup, focus on feedback findings. |
| RII and HTR correlate (2D collapses to 1D) | High | Low | Report correlation explicitly. Show as ranked dot plot instead of scatter. 1D ranking is still novel and valuable. |
| Multi-turn SDK bugs | Low-Med | **Medium** | Single-prompt-with-history fallback built from Day 1. Pilot both in Week 1. Pick the reliable one. |
| All models show strong feedback learning | Low | Medium | Great — discriminatory power from degree + condition effects (10 conditions provide rich gradient). 2×2 decomposition + mechanistic probes become even more interesting. |
| EB < 0 (explanation hurts) | Medium | **Positive** | Replicates Alazraki et al. 2025 on novel substrates. Pre-registered as two-sided. Becomes a secondary finding. |
| A competitor builds similar eval | Low-Med | Medium | Moat: 2×2 factorial + 10 conditions + strategy decomposition + 15+ models (hypothesis-driven) + mechanistic probes + cross-benchmark cognitive profiling + human baselines (informal but present) + pre-registration + DeepMind framework alignment + literature differentiation (LLF-Bench, FB-Bench, Self-Correction Bench, Likra). Hard to replicate this depth. |
| API quota insufficient | Low | Low | Budget is ~$391 total (no Prolific spend), within $500/month base quota. ~$109 breathing room. |
| Writeup exceeds 1,500 words | Medium | Moderate | Budget: 1,220 words allocated, 280-word buffer. Results-first drafting. |
| Structured output unavailable on SDK | Low | **Medium** | Day 1 test: 10 prompts with `schema=STSAnswer`. If fails → Tier 3 regex with 50-prompt gate (v8.0 fallback). Structured output is an optimization, not a dependency. |
| JSON mode fails on local models | Low | **Low** | Tier 3 regex fallback. Local models tested with 50-prompt gate. |
| Type E items not generated for some (STS, N) cells | Low | Low | STS generator constrained for ≥1 non-DIRECT rule + nested training sets → target >90% feasibility. Report fraction of valid cells. |
| Local model quantization skews comparison | Low | **Low** | Transparently noted as caveat. Within-model contrasts (FLR, RII, EB, EOLR, MR) unaffected by quantization. |
| Error-only condition shows zero signal | High | Low | Expected outcome — pure error signals are weak. The 2×2 factorial interpretation still holds: it shows the answer component dominates. This IS the finding. |
| Context pollution confirmed (clean-context >> correction) | Medium | **Positive** | Explains the SB2 < SB1 gap. Finding: "Models cannot distinguish correct from incorrect information in their own conversation history." Direct implications for agentic system design. |
| Code-tuned models show no FLR advantage | Medium | Low | Strengthens "universal blindness" narrative. Training data doesn't help. |
| OpenRouter rate limits or outages | Medium | Medium | Build rate-limiting client with exponential backoff. Spread calls across multiple hours. Have fallback models if primary choices are unavailable. |
| SB2 primary tier miscalibrated | Medium | **High** | SB1 pilot calibrates tier selection. If ~40% zone doesn't exist at any tier, widen to ~25-55% range. Secondary tier provides backup. |
| Human FLR ≈ 0 on STS (alien task problem) | Medium | **High** | Week 1 pilot (3-5 people via web tool). If humans can't learn STS either → pivot to micro-grammar for remaining 15-17 participants. Pre-registered either way. |
| Tokenization confound across models | Low-Med | **Low** | Pre-screen symbols for tokenization consistency on accessible tokenizers. Note as limitation for closed models. Within-model contrasts unaffected. |
| Literature gap discovered by judges | Low (v12.0) | **High** | v10.0: Added LLF-Bench, FB-Bench, Self-Correction Bench. v12.0: Added CL-Bench (Tencent/Fudan, Feb 2026) — most relevant recent competitor, strengthens our argument. Sharpened "first" claims to the specific novelty (factorial decomposition on novel substrates), not the general space (interactive feedback). |

---

## WRITEUP SKELETON (1,500 words budget — v10.0 allocation)

| Section | Words | Content |
|---|---|---|
| Title + subtitle | 20 | "Can AI Models Learn from Their Mistakes? A 2×2 Factorial Decomposition of Error-Driven Learning across 15+ LLMs" |
| Problem + What's New | 200 | Current evals tell you *how well* models learn — not *whether they can learn from being wrong*. Recent work tests feedback in various forms (LLF-Bench, MINT, FB-Bench, Alazraki et al.) but none apply controlled factorial decomposition on contamination-proof substrates. ALP fills this gap. Map to Morris et al. 2024 cognitive framework. The 2×2 IS the contribution — explain it clearly here. |
| Methods | 250 | STS design + contamination defense. 10 conditions (4 core + 2 extended + 4 mechanistic probes), 2×2 factorial table. Key metrics (FLR, 2×2 decomposition, EB, EOLR, MR). SB1 briefly: learning curves + strategy decomposition as supporting context. 15+ model configurations (hypothesis-driven: scale, code-tuning, reasoning-RL). Structured output extraction. Human baselines: 20 informal participants, correction + clean-context, descriptive comparison. |
| Results & Insights | 650 | **Lead with the Gap Chart** (headline: humans learn from corrections, models don't). **The 2×2 Decomposition** (evaluation effect vs. answer effect — proper marginal averaging with no-feedback baseline). **The SB2 < SB1 finding** (context pollution from wrong answers — clean-context probe). **Code-training hypothesis** (do code-tuned models show higher FLR?). EB direction (Alazraki replication?). Per-model highlights by hypothesis group. Cognitive Profile Radar. Strategy profiles (brief). Pre-registered predictions vs. actuals (18 hypotheses). |
| Affiliations | 25 | Independent researcher / Hollis Health LLC |
| References | 75 | ~20 citations: Morris et al. 2024 (DeepMind cognitive framework), Rescorla & Wagner 1972, Hattie & Timperley 2007, Cheng et al. 2023 (LLF-Bench), Li et al. 2025 (FB-Bench), Tsui 2025 (Self-Correction Bench), Wang et al. 2024 (MINT), Hamdan & Yuret 2025 (Likra), Alazraki et al. 2025, MIR-Bench 2025, **CL-Bench (Tencent/Fudan, Feb 2026)**, RULEARN/IDEA (Zhu et al. 2024), Alon et al. 2024, Logan 1988, Johansen & Palmeri 2002, Olsson et al. 2022 (induction heads), von Oswald et al. 2023 (ICL as gradient descent), Berko 1958, WILT (Wang et al. 2024), iolbench, Liu et al. 2024 ("lost in the middle" attention) |
| **Total** | **1,220** | *280 words buffer for data-dependent details* |

**Title (gradient finding):** "Can AI Models Learn from Their Mistakes? A 2×2 Factorial Decomposition of Error-Driven Learning across 15+ LLMs"

**Title (universal blindness finding):** "Frontier Models Cannot Learn from Their Mistakes: Evidence from 15+ LLMs on Contamination-Proof Tasks"

**Title (code-training finding):** "Code-Trained Models Process Error Signals; Chat Models Don't: Evidence from Controlled Feedback Experiments on 15+ LLMs"

**Drafting strategy:** Write Results section FIRST. It's the most important section and the hardest to compress. Build the rest of the writeup around the findings.

---

## IMPLEMENTATION TIMELINE

| Phase | Dates | Deliverables | Status | Spend |
|---|---|---|---|---|
| **Day 1 (Mar 18)** | Foundation | STS generator, eval pipeline, SB2 pilot framework, LM Studio + OpenRouter clients, Gemini Flash SB2 pilot (3 instances, 3 conditions). | **DONE** | ~$0.05 |
| **Day 2 (Mar 19)** | Broad scan | Full SB1 scan across 19 models (16 OpenRouter + 5 LM Studio). 16 pass filter. Infrastructure bugs fixed. 8-model SB2 pilot selection. Spec v12.0 finalized. | **DONE** | ~$10 |
| **Day 3 (Mar 20)** | SB2 pilot (EXECUTE) | Run SB2 pilot (8 models, 3 instances, 4 core conditions). **In parallel:** no-feedback + clean-context probes on 1-2 models. Build analysis module so results flow immediately into visualizations. | **ACTIVE** | ~$15-20 |
| **Day 4 (Mar 21)** | Analyze + decide | Analyze SB2 pilot. Story direction clear: universal blindness? code-tuning advantage? context pollution? Select 5-6 models for production. | PENDING | ~$5 |
| **Days 5-7 (Mar 22-24)** | Production ramp | Full 25-instance SB2 (4 core conditions) on selected models. Mechanistic probes on 2-3 models. Build human baseline web tool. Start recruiting. | PENDING | ~$30-50 |
| **Days 8-12 (Mar 25-29)** | Production complete | Remaining production SB2 runs. Explanation on top 3 models. Misleading on 2 models (3-5 instances). Human baseline collection. **DISCRIMINATORY POWER CHECK.** | PENDING | ~$30-60 |
| **Days 13-16 (Mar 30-Apr 2)** | Kaggle SDK + analysis | Run Kaggle SDK models (2 models, full SB1 + SB2). Full statistical analysis. All visualizations. Pre-registered hypotheses vs actuals. | PENDING | ~$200-300 |
| **Days 17-28 (Apr 3-16)** | Polish + submit | Writeup (Results section FIRST). Kaggle benchmark + notebook. Cover image (Gap Chart). Submit before Apr 16 11:59 PM UTC. | PENDING | $0 |
| **Total** | | | **2 of 7 phases done** | **~$290-445** |

---

## PROGRESS & NEXT STEPS (updated 2026-03-19, v12.0)

### COMPLETED (Days 1-2)

1. ~~**STS Generator + Solver**~~ — DONE. Generates STS instances with tier-appropriate rules, informative training examples, Type R/E/L test items. 0% identity examples. Type E feasibility: T2=82%, T3=90%, T4=100%.
2. ~~**Extraction pipeline**~~ — DONE. Tiered: JSON schema → plain prompt fallback. JSON schema compatibility caching for models that don't support it (GPT-5.3, MiniMax). Symbol-aware extraction with vocabulary matching.
3. ~~**OpenRouter client**~~ — DONE. Rate-limited with exponential backoff, logging, JSON schema fallback caching. See `lesson/models/openrouter.py`.
4. ~~**LM Studio client**~~ — DONE. OpenAI-compatible API on port 1234, max_tokens=2048. See `lesson/models/lmstudio.py`.
5. ~~**SB1 evaluation runner**~~ — DONE. Sequential and parallel modes. Thread-safe with print/logging locks. See `lesson/eval/pilot.py`.
6. ~~**SB2 pilot framework**~~ — DONE. Multi-turn sessions with condition-specific feedback. See `lesson/eval/sb2_pilot.py`.
7. ~~**OpenRouter broad scan**~~ — DONE. 19 models scanned, 16 pass filter. See `docs/observations.md`.
8. ~~**LM Studio scan**~~ — DONE. 5 models tested, 1 passes filter (Qwen-3-Coder-30B), 3 broken (context issues), 1 too weak.
9. ~~**Gemini Flash SB2 pilot**~~ — DONE. 3 instances, 3 conditions. FLR ~ 0 directionally. Far too small to confirm.
10. ~~**Infrastructure hardening**~~ — DONE. Log pruning race condition, Unicode console fix, empty-response fallback, JSON schema caching.
11. ~~**Spec v12.0**~~ — DONE. Execution-focused scope tightening. CL-Bench added. Thinking traces cut. Misleading/explanation reduced.

### NEXT 48 HOURS — EXECUTE (Days 3-4, Mar 20-21)

**Priority 1 (Day 3):** Run the 8-model SB2 pilot (3 instances, 4 core conditions). ~$15-20. This gives you data to make every subsequent decision.

**Priority 2 (Day 3, in parallel):** Run no-feedback + clean-context probes on 1-2 models to understand the SB2 < SB1 gap.

**Priority 3 (Day 3-4, in parallel):** Build the analysis module so results flow immediately into visualizations. Don't wait for all data before building the pipeline.

**Priority 4 (Day 4):** SB1 extended (N=16, N=32) on existing T2 instances for comparison with SB2's "accumulated examples" effect.

### WHAT TO DEFER OR CUT

- ~~Thinking trace analysis~~ — CUT (v12.0). Qualitative, won't change headline.
- **Misleading at full scale** — reduced to 3-5 instances on 2 models. Not the story.
- **Extended conditions (explanation) on more than 3 models** — the 2×2 is the headline, not extensions.
- **Micro-grammar ecological validity probe** — run late, only if SB1 strategy data is interesting enough.
- **Human baselines** — don't block SB2 model runs on human data collection. Run in parallel.

### Gates

**Gate: End of Day 4 (Mar 21)**
- SB2 pilot data on 8 models. Early FLR read across model groups.
- Story direction clear: Universal blindness? Code-tuning advantage? Context pollution?
- Models selected for production.

**Gate: End of Day 12 (Mar 29) — DISCRIMINATORY POWER**
- Full 25-instance SB2 on 5-6 models complete.
- The 2×2 decomposition: does evaluation effect ≠ answer effect?
- Permutation test: do model groupings differ on FLR?
- If ALL metrics identical → pivot narrative to "universal feedback blindness"
- **Do not wait until Week 4 to discover zero discrimination**

---

## VERSION HISTORY (refinement direction to prevent thrashing)

| Dimension | v8.0  | v9.0  | v10.0/10.1  | v10.2  | v11.0 | v12.0 |
|---|---|---|---|---|---|---|
| **Narrative** | ONE story: 2×2 factorial | ONE story (unchanged) | Dual-ready: gradient + "universal blindness" | Unchanged | Triple-ready: gradient + universal blindness + code-training hypothesis | **Unchanged. Execution-focused revision.** |
| **2×2 Design** | SB1 as baseline cell | Same | Same | Same | No-feedback condition fills baseline cell. True 2×2, all same format. Proper marginal averaging. | **Unchanged** |
| **Conditions** | 5 (correction, explanation, practice, error, misleading) | Same | Same | Same | 10 total: 4 core + 2 extended + 4 mechanistic probes | **10 total. Misleading reduced to 3-5 instances on 2 models. Explanation limited to top 3. Thinking trace analysis cut.** |
| **Models** | 9 (4 SDK + 5 local) | Same | Same | Same | 15-20. Two-phase hypothesis-driven selection. | **Unchanged** |
| **Model selection** | Architecture diversity | Same | Same | Same | Hypothesis-driven: scale (H1), code-training (H2), reasoning-RL (H3), architecture (H4). | **Unchanged** |
| **SB2 < SB1 gap** | Not observed | Not observed | Not observed | Not observed | Dedicated section + mechanistic probes. | **Unchanged** |
| **SB1 extended** | N=2,4,8,16,32 | Same | Same | Same | Added N=16, N=32 on existing instances. | **Unchanged** |
| **Statistical infrastructure** | Ad hoc | Same | Same | Same | Dedicated analysis module. | **Unchanged** |
| **Human baselines** | Prolific, $500, 50 people | Same | Same | 20 informal, $0 | 20 informal, $0, correction + clean-context within-subjects. | **Unchanged. Don't block SB2 runs on human data.** |
| **Literature gaps** | Missing LLF-Bench, FB-Bench, Self-Correction Bench | Same | All added + Hamdan & Yuret 2025 (Likra) | Unchanged | Unchanged | **Added CL-Bench (Tencent/Fudan, Feb 2026)** |
| **SB2 piloting** | Week 2 | Week 2 | Front-loaded to Week 1 | Unchanged | Days 2-3. | **Day 3 (TODAY). Stop planning, execute.** |
| **Hypotheses** | 13 | Same | Same | Same | 18 (added H14-H18). | **Unchanged** |
| **Budget** | ~$387 + $500 Prolific | ~$410 + $500 Prolific | ~$391 + $500 Prolific | ~$391 total | ~$450-520 total | **~$290-445 total (reduced misleading/extended saves ~$30-50)** |
| **Execution risk** | Medium | Lower | Lowest | Lowest | Lowest | **Lowest. v12.0 is a scope-tightening release. No new complexity.** |

### v1.5.0 Status (2026-03-19)

**SB1 broad scan complete.** 19 models tested across OpenRouter + LM Studio. 16 pass the SB2 filter (T2N8 15-70%). Key early findings:
- **Discriminatory power confirmed**: 50-point spread from GLM-5 (67%) to Llama-4-Maverick (0%)
- **Code hypothesis directionally supported**: GPT-5.3-Codex (58%) > GPT-5.3-Chat (50%)
- **Reasoning anomaly flagged**: DeepSeek-R1 gets *worse* with more examples at T1 (53% → 20%)
- **Claude scale gradient flat**: Opus = Sonnet = 42% at T2N8
- **GLM-5 surprise leader**: Outperforms GPT-5.3 and Claude Opus on novel rule induction
- **Llama models strikingly weak**: 70B at 8%, Maverick at 0% — training methodology effect
- **Infrastructure hardened**: JSON schema caching, empty-response fallback, log pruning race fix, Unicode console fix

### v12.0 Changes (2026-03-19)

**Execution-focused revision. No new complexity — only scope tightening and prioritization.**

1. **Added CL-Bench** (Tencent/Fudan, Feb 2026) to differentiation table. Single-turn context learning benchmark — complementary, not overlapping. Strengthens the case that context learning is underexplored.
2. **Cut thinking trace analysis** entirely. Qualitative fluff that won't change the headline.
3. **Reduced misleading condition** to 3-5 instances on 2 models (was 15 instances on top 3-4). Interesting but not the story.
4. **Limited explanation condition** to top 3 models (was 3-4).
5. **Accelerated timeline**: SB2 pilot begins Day 3 (Mar 20). Analysis module built in parallel so results flow directly into visualizations.
6. **Micro-grammar probe**: run late, only if SB1 strategy data warrants validation. Don't prioritize over SB2.

**The plan is good enough. Execute.**
