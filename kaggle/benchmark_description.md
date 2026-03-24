## LESSON: Learning from Error Signals in Symbolic Operations

This benchmark measures whether large language models can learn from corrective feedback within a single session, or whether they only benefit from seeing additional correct examples.

### Method

The evaluation uses procedurally generated symbolic transformation systems (STS): sets of deterministic rewriting rules over abstract symbol alphabets. Each STS instance defines a mapping from input sequences to output sequences. Because the rules are generated at evaluation time from a random seed, the test material cannot appear in any model's training data.

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

Each model is tested across 25 STS instances per condition, with 12 multi-turn interactions per instance. The model first sees 8 training examples of the transformation, then answers 12 novel test inputs with condition-specific feedback between turns.

### Metrics

- **Feedback Learning Rate (FLR)**: Late-turn accuracy under correction minus late-turn accuracy under practice only. Measures whether corrective framing provides learning benefit beyond the answer itself.
- **Answer effect**: Average accuracy with answer visible minus average accuracy with answer hidden. Measures the contribution of seeing correct examples.
- **Evaluation effect**: Average accuracy with evaluation present minus average accuracy with evaluation absent. Measures the contribution of right/wrong signals.
- **Evaluation damage**: Error-only accuracy minus no-feedback accuracy. Measures whether evaluation signals without answers cause harm.

### What this benchmark reveals

Standard evaluations conflate a model's knowledge with its capacity to learn from new information during a session. This benchmark isolates the learning process itself and further decomposes it into distinct information channels. The factorial design makes it possible to determine whether a model processes corrective feedback as a learning signal or simply as additional example data.
