# Syllabus

Five levels, 32 modules. Each level builds on the one before. Module IDs are `L<level>-<NN>-<slug>`. A module's page goes in the level's directory as `<NN>-<slug>.md`, and its code goes in `code/<module-id>/`.

| Level | Directory | Theme |
|---|---|---|
| 0 | `docs/level-0-foundations/` | The math and tools everything else uses |
| 1 | `docs/level-1-classical-ml/` | Learning from data without deep networks |
| 2 | `docs/level-2-deep-learning/` | Neural networks, from backprop to transformers |
| 3 | `docs/level-3-modern-ai/` | Modern AI across domains |
| 4 | `docs/level-4-engineering-and-society/` | Running, measuring, understanding and governing AI |

Scope notes tell you what a module **must** cover and what belongs to **another** module (so we don't repeat each other). Stay inside your scope. Where a topic belongs elsewhere, give one sentence and a pointer such as "see L2-05".

---

## Level 0: Foundations

**L0-01-python-numpy-for-ml.** Arrays, dtypes, broadcasting, vectorization vs loops (with measured timings), indexing, reproducibility (seeds), plotting basics, profiling, floating-point pitfalls (overflow, underflow, log-sum-exp). *Not here:* linear algebra theory (L0-02).

**L0-02-linear-algebra.** Vectors, dot products, norms, matrices as linear maps, matrix multiplication cost, rank, eigendecomposition, SVD and low-rank approximation, projections, the geometry of embeddings and cosine similarity. Always tie to what ML uses it for. *Not here:* PCA as an algorithm (L1-04).

**L0-03-calculus-and-autodiff.** Derivatives, partial derivatives, gradients, the chain rule, Jacobians, computational graphs, forward vs reverse mode, and building a small scalar autograd engine from scratch with tests against numerical gradients. *Not here:* optimizers (L0-05), backprop through full networks (L2-01).

**L0-04-probability-and-statistics.** Random variables, common distributions, expectation and variance, Bayes' rule, maximum likelihood and MAP, entropy, cross-entropy, KL divergence, sampling and Monte Carlo, confidence intervals, and the basics of hypothesis testing and what they mean for comparing models. *Not here:* evaluation methodology for ML (L1-01, L4-03).

**L0-05-optimization.** Gradient descent, learning rate behavior, conditioning, SGD and minibatches, momentum, Adam and its relatives, convex vs non-convex, saddle points, line search basics, and why optimization in deep learning works as well as it does (with honest uncertainty). *Not here:* normalization, schedules in practice (L2-02).

## Level 1: Classical machine learning

**L1-01-ml-workflow-and-evaluation.** Problem framing, baselines, train/validation/test splits, data leakage, metrics for classification and regression (accuracy, precision/recall, ROC/PR, calibration), cross-validation, the bias-variance trade-off, over- and underfitting, reporting uncertainty. *Not here:* LLM evaluation (L4-03).

**L1-02-linear-and-logistic-models.** Linear regression (closed form and gradient descent), regularization (ridge, lasso), logistic regression, softmax, the probabilistic view, from-scratch implementations compared with scikit-learn, interpretation of coefficients and its limits.

**L1-03-trees-and-ensembles.** Decision trees, splitting criteria, random forests, bagging, boosting and gradient boosting, practical tabular modeling, why tree ensembles still often beat deep nets on tabular data (with sources and honest caveats), feature importance and its pitfalls.

**L1-04-unsupervised-learning.** k-means, Gaussian mixtures and EM, hierarchical clustering, PCA (derivation via SVD from L0-02), t-SNE and UMAP and how they mislead, anomaly detection, evaluating unsupervised results.

**L1-05-data-and-features.** Data quality and collection, labeling, feature engineering, encoding categorical variables, scaling, missing data, class imbalance, dataset shift, train/serve skew, and data-centric AI. *Not here:* privacy and fairness (L4-05), MLOps (L4-01).

## Level 2: Deep learning

**L2-01-neural-networks-and-backprop.** Perceptrons, MLPs, activation functions, the universal approximation idea (with its real limits), loss functions, deriving backpropagation, implementing an MLP from scratch in NumPy, gradient checking. *Not here:* the autograd engine itself (L0-03), which this module reuses by reference.

**L2-02-training-deep-networks.** PyTorch fundamentals (tensors, autograd, `nn.Module`, data loading, training loops), initialization, normalization (batch, layer), regularization (dropout, weight decay, early stopping), learning-rate schedules, debugging a network that won't train, overfitting a single batch, reproducibility.

**L2-03-convolutional-networks.** Convolution, pooling, receptive fields, classic architectures (LeNet to ResNet), residual connections, data augmentation, transfer learning and fine-tuning, a small image classifier you can train on CPU. *Not here:* vision transformers and detection (L3-04).

**L2-04-sequence-models.** Recurrent networks, vanishing and exploding gradients, LSTM and GRU, sequence-to-sequence, the attention mechanism's origin in translation, teacher forcing, decoding (greedy, beam, sampling), a small character-level language model.

**L2-05-transformers.** Self-attention from first principles, multi-head attention, positional encodings (sinusoidal, learned, rotary), residuals and layer norm placement, the decoder-only architecture, masking, a complete small GPT in PyTorch trained on a tiny corpus on CPU, attention cost and why it matters. *Not here:* scaling laws and pretraining at scale (L3-01).

**L2-06-representation-learning-and-embeddings.** What an embedding is, word2vec and the distributional hypothesis, contrastive learning, self-supervised learning, autoencoders, similarity search basics (exact vs approximate), the geometry and the failure modes of embeddings. *Not here:* RAG systems (L3-06).

## Level 3: Modern AI across domains

**L3-01-language-model-pretraining.** Tokenization (BPE, unigram, byte-level) with a from-scratch BPE, the next-token objective, data sources and curation, scaling laws and the compute-optimal debate, architecture choices (decoder-only, mixture of experts at concept level), long-context techniques, emergent abilities and the debate about them, what pretraining at scale involves. *Not here:* instruction tuning and RLHF (L3-02), serving (L4-02).

**L3-02-post-training-and-alignment-methods.** Supervised fine-tuning, reward modeling, RLHF with PPO, direct preference optimization and relatives, reinforcement learning from verifiable rewards and GRPO-style methods, reward hacking, constitutional approaches, and how post-training shapes behavior. Include a small runnable preference-optimization or policy-gradient toy. *Not here:* RL basics (L3-05), safety and governance (L4-05).

**L3-03-generative-models-and-diffusion.** Generative modeling goals, VAEs, GANs (including why they were hard to train), diffusion models (the forward and reverse process, score matching view, DDPM, samplers), classifier-free guidance, latent diffusion, flow matching, evaluation (FID and its limits). Include a small 2D diffusion toy that trains on CPU.

**L3-04-vision-and-multimodal-models.** Vision transformers, object detection and segmentation concepts, CLIP-style contrastive image-text training, vision-language models, speech recognition and audio models at a conceptual level, how modalities are tokenized and combined. *Not here:* CNN basics (L2-03).

**L3-05-reinforcement-learning-foundations.** Markov decision processes, returns and value functions, Bellman equations, dynamic programming, Monte Carlo and TD learning, Q-learning, DQN, policy gradients, actor-critic, PPO, exploration, and what makes RL hard. Include tabular and small deep RL examples that run on CPU.

**L3-06-retrieval-and-rag.** Retrieval basics (BM25, dense retrieval, hybrid), vector indexes (HNSW, IVF) conceptually, chunking, reranking, RAG architectures and failure modes, RAG evaluation, long context versus retrieval trade-offs, grounding and citations. Include a small working RAG over a bundled corpus without external APIs.

**L3-07-agents-and-tool-use.** The agent loop, tool design, planning and reasoning patterns, memory, multi-agent setups, agent benchmarks and their weaknesses, failure modes, security (prompt injection), when not to use an agent. This builds on, and should link to, the existing "AI Systems" course (https://sankarsh98.github.io/ai-systems/) rather than repeat it: do not re-teach the basic loop or MCP in depth, go broader and more critical.

**L3-08-recommenders-and-ranking.** Collaborative filtering, matrix factorization, implicit feedback, two-tower models, candidate generation vs ranking, evaluation (offline metrics vs online experiments), feedback loops, cold start, bias and the societal effects of recommendation. Include a runnable matrix factorization on a small bundled or synthetic dataset.

**L3-09-time-series-and-graphs.** Forecasting basics (baselines, ARIMA/ETS ideas, validation for time data), deep and foundation-model approaches to forecasting with honest evidence about when they help, anomaly detection in time series, graph data, message passing and GNNs, applications. Include runnable examples.

**L3-10-ai-for-science-and-healthcare.** Case studies with verified detail: protein structure prediction, weather and climate modeling, materials and drug discovery, medical imaging and clinical prediction, and the specific risks (distribution shift, evaluation, regulation, equity). Emphasize what has been demonstrated versus claimed. Heavy verification of all factual claims is required.

## Level 4: Engineering and society

**L4-01-mlops-and-deployment.** From notebook to production: experiment tracking, data and model versioning, reproducible pipelines, packaging, serving patterns (batch, online, streaming), monitoring, drift detection, rollback, CI for ML, cost. Include a small runnable pipeline with tests and a drift check.

**L4-02-efficient-training-and-inference.** Mixed precision, memory and compute accounting for training (parameters, gradients, optimizer state, activations), data, tensor and pipeline parallelism, communication, quantization, distillation, pruning, serving techniques at a survey level. This builds on, and should link to, the "AI Systems" course (KV cache, quantization, speculative decoding) rather than repeat it. Include a runnable memory-and-FLOPs calculator.

**L4-03-evaluation-and-benchmarks.** Designing evals, metrics vs real goals, benchmark saturation and contamination, human evaluation, LLM-as-judge and its biases, statistical rigor (confidence intervals, significance with small eval sets), building a small eval harness, red flags in published claims.

**L4-04-interpretability.** Why interpret, probing, attribution methods and their limits, mechanistic interpretability (circuits, induction heads, superposition, sparse autoencoders), what has and hasn't been shown, a hands-on experiment you can run on a small model on CPU.

**L4-05-safety-alignment-and-governance.** Failure modes (hallucination, misuse, deceptive or misaligned behavior as a research concern), adversarial robustness and jailbreaks, prompt injection, privacy (memorization, differential privacy basics), fairness definitions and their conflicts, safety evaluations and red teaming, the current governance landscape (verify everything as of October 2026), and honest discussion of open disagreements.

**L4-06-research-skills-and-careers.** Reading a paper efficiently, reproducing results, ablations and experimental hygiene, writing and tracking experiments, finding problems, open-source contribution, the shape of careers in AI, how to keep learning, and a curated map of the field's main venues, textbooks and communities (all verified).

---

## Capstones (written by the coordinator, not by module writers)

Five projects that combine modules. Listed so module writers can mention them where relevant.

1. **Tabular predictor with honest evaluation** (L0 to L1).
2. **A GPT from scratch on your own text** (L2).
3. **RAG over your own notes with an evaluation set** (L3-06, L4-03).
4. **Reproduce a small paper result** (L4-06).
5. **An agent with guardrails and an eval harness** (L3-07, L4-03, L4-05).
