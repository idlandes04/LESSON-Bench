## LESSON: Learning from Error Signals in Symbolic Operations

**Track: Learning**

This benchmark measures whether large language models can learn from corrective feedback within a single session, or whether they only benefit from seeing additional correct examples. It isolates the learning process itself — separating the contribution of error signals from the contribution of additional example data — using a factorial design that has not previously been applied to in-context learning.

### Method

The evaluation uses procedurally generated symbolic transformation systems (STS): sets of deterministic rewriting rules over abstract symbol alphabets. Each STS instance defines a mapping from input sequences to output sequences. Because each instance is generated from a random seed at evaluation time, the specific test material cannot appear in any model's training data. A deterministic solver verifies exactly one correct output per input, eliminating answer ambiguity.

The benchmark uses a 2x2 factorial design that separates two information channels present in corrective feedback:

- **Answer visibility**: whether the model sees the correct output after each turn
- **Evaluation signal**: whether the model is told if its answer was right or wrong

This produces four feedback conditions:

| Condition | Answer visible | Evaluation signal |
|---|---|---|
| Correction | Yes | Yes |
| Practice only | Yes | No |
| Error only | No | Yes |
| No feedback | No | No |

The benchmark has two sub-benchmarks. **SB1 (learning curves)** tests each model with an escalating number of training examples (e.g., 4, 8, 16, 32) to measure how sample-efficiently the model induces the transformation rules from a single prompt. **SB2 (feedback learning)** tests each model across 25 STS instances per condition, with 12 multi-turn interactions per instance. The model first sees 8 training examples of the transformation, then answers 12 novel test inputs with condition-specific feedback between turns.

The choice of 8 initial examples for SB2 is calibrated from SB1 learning curve data: 8 examples places most frontier models in the 30–50% accuracy range — high enough to demonstrate partial rule induction, but low enough to leave room for feedback-driven improvement. Fewer examples (2–4) would put many models near floor, making feedback effects undetectable; more examples (16–32) would risk ceiling effects where models saturate from examples alone, masking any feedback signal.

### Statistical design

Each SB2 condition produces 25 instances × 12 turns = 300 binary observations per model per condition, yielding bootstrap confidence intervals narrow enough to detect effect sizes of ~5 percentage points. SB1 provides 3–5 test items per instance across multiple N values for learning curve estimation.

### Metrics

- **Feedback Learning Rate (FLR)**: Late-turn accuracy under correction minus late-turn accuracy under practice only. Measures whether corrective framing provides learning benefit beyond the answer itself.
- **Answer effect**: Average accuracy with answer visible minus average accuracy with answer hidden. Measures the contribution of seeing correct examples.
- **Evaluation effect**: Average accuracy with evaluation present minus average accuracy with evaluation absent. Measures the contribution of right/wrong signals.
- **Evaluation damage**: Error-only accuracy minus no-feedback accuracy. Measures whether evaluation signals without answers cause harm.

### Discriminatory power

SB1 testing across 16+ frontier models shows accuracy ranging from 67% to under 10% on the same task instances — a 50+ point spread that clearly distinguishes model capabilities. Models also differ in sample efficiency (the slope from N=4 to N=8), and within-family comparisons (e.g., code-tuned vs. chat-tuned variants of the same base model) reveal training methodology effects not visible on standard benchmarks.

### What this benchmark reveals

Standard evaluations conflate a model's knowledge with its capacity to learn from new information during a session. This benchmark isolates the learning process itself and further decomposes it into distinct information channels. The factorial design makes it possible to determine whether a model processes corrective feedback as a learning signal or simply as additional example data — a distinction with direct implications for agentic systems that rely on iterative error correction.
