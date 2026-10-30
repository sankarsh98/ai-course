---
title: "Level 3: Modern AI across domains"
---

Language-model pretraining and post-training, generative models and diffusion, vision and multimodal models, reinforcement learning, retrieval, agents, recommenders, time series and graphs, and AI in science and healthcare. Time-sensitive claims are dated and verified as of October 2026.

| ID | Module | Prerequisites | Time | Scripts |
|---|---|---|---|---|
| L3-01 | Language Model Pretraining | L0-04, L0-05, L2-02, L2-04, L2-05 | 10 to 12 hours | 4 |
| L3-02 | Post-Training and Alignment Methods | L0-04, L0-05, L2-05, L3-01 (L3-05 helps for section 4) | 10 to 12 hours | 3 |
| L3-03 | Generative Models and Diffusion | L0-04, L0-05, L2-02, L2-05, L2-06 | 8 to 10 hours | 6 |
| L3-04 | Vision and Multimodal Models | L2-03, L2-05, L2-06 | 7 to 9 hours | 4 |
| L3-05 | Reinforcement Learning Foundations | L0-04, L0-05, L2-01, L2-02 | 8 to 10 hours | 6 |
| L3-06 | Retrieval and RAG | L0-02, L0-04, L1-01, L2-06 | 7 to 9 hours | 5 |
| L3-07 | Agents and Tool Use | L0-04, L1-01, L3-02, L3-06, and Track A of the "AI Systems" course | 7 to 9 hours | 5 |
| L3-08 | Recommenders and Ranking | L0-02, L1-01, L1-02, L2-02, L2-06 | 7 to 9 hours | 5 |
| L3-09 | Time Series and Graphs | L1-01, L1-03, L1-04, L2-04, L2-05 | 8 to 10 hours | 5 |
| L3-10 | AI for Science and Healthcare | L1-01, L1-05, L2-03, L3-09 | 6 to 8 hours | 4 |

## What each module teaches

### L3-01: Language Model Pretraining

- Implement byte-level BPE from scratch, and explain how vocabulary size, pre-tokenization and language change sequence length and cost.
- State the next-token objective, and compare models with different tokenizers in bits per byte.
- Describe a modern pretraining data pipeline and why each step exists.
- Compute training FLOPs with $C \approx 6ND$, derive a compute-optimal allocation from a fitted scaling law, and explain why practice deviates from it.

### L3-02: Post-Training and Alignment Methods

- Explain what supervised fine-tuning, reward modeling and reinforcement learning each add to a base model, and what each cannot do.
- Fit a Bradley-Terry reward model and show, with numbers, how optimizing against it can lower the true reward.
- Derive the optimum of the KL-regularized RLHF objective, and from it the DPO loss; verify both numerically.
- Implement a GRPO-style policy-gradient update with a verifiable reward, and compare it with offline DPO on pass@1, pass@k and KL.

### L3-03: Generative Models and Diffusion

- State what a generative model is for, and compare VAEs, GANs, diffusion and flow models by what they optimize and how they fail.
- Derive the DDPM training loss from the forward noising process and connect noise prediction to the score $\nabla_x \log q_t(x)$.
- Implement and train a diffusion model and a flow-matching model on CPU, and sample from them with DDPM, DDIM and Euler steps.
- Explain classifier-free guidance as a reweighting of scores, and measure the trade-off it makes between fidelity and diversity.

### L3-04: Vision and Multimodal Models

- Turn an image into a token sequence, explain what a vision transformer gives up relative to a CNN, and measure it on small data.
- Explain the bookkeeping of detection and segmentation (IoU, non-maximum suppression, average precision) and how set-prediction and promptable models changed it.
- Train a CLIP-style contrastive image-text model and test the claims usually made for it, including zero-shot transfer and compositional understanding.
- Describe the main ways vision-language models connect an image encoder to a language model, and their documented weaknesses.

### L3-05: Reinforcement Learning Foundations

- Formulate a decision problem as a Markov decision process and write its Bellman equations.
- Solve small MDPs with policy evaluation, value iteration and policy iteration, and explain why they converge.
- Implement Monte Carlo, TD(0), SARSA and Q-learning, and predict how on-policy and off-policy learning behave.
- Derive the policy gradient, show why baselines cut variance without adding bias, and compute GAE advantages.

### L3-06: Retrieval and RAG

- Implement BM25 and a dense retriever, and explain when each fails.
- Combine retrievers with reciprocal rank fusion, and explain how HNSW finds neighbors without scanning everything.
- Choose a chunking scheme by measuring retrieval at a fixed context budget.
- Build a retrieve, rerank, answer and cite pipeline that abstains when evidence is weak.

### L3-07: Agents and Tool Use

- Decide between a single model call, a fixed workflow and an agent, and back the decision with reliability and cost arithmetic.
- Explain the main reasoning, planning, memory and multi-agent designs, and how strong the evidence for each is.
- Evaluate an agent properly: grade outcomes, run repeated trials, choose between pass@k and pass^k, report intervals and cost.
- Read an agent benchmark claim critically, checking for contamination, weak graders, leaked test signals and scaffold differences.

### L3-08: Recommenders and Ranking

- Frame recommendation as rating prediction or top-N ranking from explicit or implicit feedback, and set up the baselines that a new method has to beat.
- Derive and implement biased matrix factorization and implicit ALS, and explain why a plain SVD is the wrong tool for a matrix with missing entries.
- Build a two-tower retrieval model with in-batch negatives, explain the sampling-bias correction, and use side features for cold start.
- Lay out a multi-stage system (candidate generation, ranking, re-ranking) and say what each stage optimizes.

### L3-09: Time Series and Graphs

- Evaluate a forecaster with rolling-origin backtests and MASE, and show why a shuffled split misleads.
- Implement seasonal naive, Holt-Winters and a seasonal autoregressive model, and check whether their prediction intervals are honest.
- Judge claims about deep and foundation-model forecasters against the evidence, as of October 2026.
- Score time-series anomalies from forecast residuals, and evaluate detectors without the point-adjustment trap.

### L3-10: AI for Science and Healthcare

- Place a published claim on an evidence ladder, from retrospective benchmark to randomised trial, and say who validated it and how.
- Summarise what has been demonstrated, and what is still claimed or contested, for protein structure prediction, machine-learning weather forecasting, materials and drug discovery, and medical imaging and clinical prediction (as of October 2026).
- Show with code how shortcut learning and site differences inflate internal validation, and why external validation is necessary but not sufficient.
- Audit a model by subgroup with confidence intervals, and trace a disparity to the choice of label or feature.
