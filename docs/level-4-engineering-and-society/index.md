# Level 4: Engineering and society

Running, measuring, understanding and governing AI systems: MLOps, efficient training and inference, evaluation and benchmarks, interpretability, safety and governance, and research skills and careers.

| ID | Module | Prerequisites | Time | Scripts |
|---|---|---|---|---|
| L4-01 | MLOps and Deployment | L1-01, L1-05, L2-02 | 6 to 8 hours | 4 |
| L4-02 | Efficient Training and Inference | L0-01, L0-05, L2-02, L2-05, L3-01 | 7 to 9 hours | 5 |
| L4-03 | Evaluation and Benchmarks | L0-04, L1-01, L3-02, L3-06, L3-07 | 7 to 9 hours | 5 |
| L4-04 | Interpretability | L1-02, L2-01, L2-02, L2-05, L4-03 | 7 to 9 hours | 4 |
| L4-05 | Safety, Alignment and Governance | L0-04, L1-01, L1-05, L3-02, L3-07, L4-03 (L3-08 and L3-10 help for sections 7 and 10) | 10 to 12 hours | 5 |
| L4-06 | Research Skills and Careers | L0-04, L1-01, L2-02, L4-03 (L4-01 helps for section 4) | 6 to 8 hours, plus the capstone | 4 |

## What each module teaches

### L4-01: MLOps and Deployment

- Turn a notebook model into a pipeline whose data, code, environment and model are versioned, so any prediction can be traced to what produced it.
- Write data-validation checks and tests that run offline in continuous integration, and say what each kind of test can and cannot catch.
- Choose between batch, online and streaming serving for a problem, and estimate what each costs.
- Release a model safely with an offline gate, a canary compared against a concurrent control, and a rollback that is a one-line change.

### L4-02: Efficient Training and Inference

- Account for the memory of a training run (weights, gradients, optimizer state, activations) and predict what mixed precision, activation checkpointing and sharding each save.
- Derive the $6ND$ training-compute rule, state its assumptions, and turn FLOPs into a run time through model FLOPs utilization.
- Explain fp32, fp16, bf16 and fp8, and why mixed precision keeps fp32 master weights and (for fp16) scales the loss.
- Compare data, tensor, pipeline and expert parallelism by the memory each removes and the communication each adds.

### L4-03: Evaluation and Benchmarks

- Design an evaluation that starts from a decision and a target population, and choose a metric and a grader that measure what that decision needs.
- Report an eval score with an honest interval: clustered standard errors, resampled answers, paired comparisons, and a power calculation done before the eval is built.
- Explain how benchmarks saturate and become contaminated, detect simple contamination, and say what a "decontaminated" score does and does not guarantee.
- Measure agreement between human annotators with Cohen's kappa and Krippendorff's alpha, and read those numbers correctly when classes are imbalanced.

### L4-04: Interpretability

- Explain what different interpretability methods claim to show, and separate *information is present* (correlation) from *the model uses it* (causation).
- Train and evaluate probes with proper baselines and control tasks, and test whether a probed direction matters by removing it.
- Compute gradient, gradient-times-input, integrated-gradients and guided-backprop attributions, and run the sanity checks that some of them fail.
- Train a two-layer attention-only transformer on CPU, watch induction heads form in a sudden phase change, and confirm their role with ablations.

### L4-05: Safety, Alignment and Governance

- Classify an AI failure (hallucination, misuse, adversarial attack, privacy leak, unfair outcome, misaligned behavior) and name the evidence and defenses that fit it.
- Explain why adversarial examples, jailbreaks and prompt injection are hard to defend against, and measure robustness with an adaptive attack rather than a fixed one.
- Run a membership-inference test against a model, explain what differential privacy guarantees, and compute the privacy cost of noisy training.
- State the main group-fairness criteria, prove that they conflict when base rates differ, and show the conflict on data.

### L4-06: Research Skills and Careers

- Read a machine learning paper in passes, extract its central claim and the evidence for it, and spot the most common weaknesses in that evidence.
- Plan and carry out a reproduction of a published result, and write down where and why your numbers differ.
- Design an ablation study with several seeds, paired comparisons and a held-out test set, and report it with honest uncertainty.
- Track experiments so that every number in a report can be traced to a configuration, a code version and an environment.
