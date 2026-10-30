---
title: "Level 2: Deep learning"
---

From backpropagation by hand to a working GPT: training practice, convolutional networks, sequence models, transformers, and embeddings. Everything trains on a CPU in under a minute.

| ID | Module | Prerequisites | Time | Scripts |
|---|---|---|---|---|
| L2-01 | Neural Networks and Backpropagation | L0-02, L0-03, L0-05, L1-02 | 6 to 8 hours | 6 |
| L2-02 | Training Deep Networks | L0-05, L1-01, L2-01 | 6 to 8 hours | 9 |
| L2-03 | Convolutional Networks | L0-02, L1-01, L2-01, L2-02 | 6 to 8 hours | 5 |
| L2-04 | Sequence Models | L0-03, L0-04, L2-01, L2-02, L2-03 | 6 to 8 hours | 4 |
| L2-05 | Transformers | L0-02, L0-04, L2-01, L2-02, L2-04 | 8 to 10 hours | 7 |
| L2-06 | Representation Learning and Embeddings | L0-02, L1-04, L2-02, L2-05 | 7 to 9 hours | 5 |

## What each module teaches

### L2-01: Neural Networks and Backpropagation

- Explain why a single perceptron cannot learn XOR and why one hidden layer with a nonlinearity can.
- State what the universal approximation theorems promise, and name three things they do not promise.
- Choose an output layer and loss for regression and classification, and justify computing the loss from logits.
- Derive backpropagation for a multilayer perceptron in matrix form and implement it in NumPy.

### L2-02: Training Deep Networks

- Write a complete PyTorch training loop with validation, checkpointing and early stopping.
- Derive why the initial weight scale must depend on fan-in and activation, and detect a network whose signal dies at initialization.
- Use batch norm and layer norm correctly, including train and eval modes, and summarize the debate about why batch norm helps.
- Apply dropout, weight decay and early stopping, and predict when each matters.

### L2-03: Convolutional Networks

- Compute a convolution by hand and in code, predict output shapes, parameter counts and cost, and derive its backward pass.
- Explain translation equivariance, why pooling gives only approximate invariance, and test both experimentally.
- Calculate receptive fields and explain why the effective receptive field is smaller.
- Trace the main ideas from LeNet to ResNet, and show why residual connections make deep networks trainable.

### L2-04: Sequence Models

- Write down a recurrent network, unroll it, and explain backpropagation through time.
- Derive why RNN gradients vanish or explode, measure it, and use gradient clipping.
- Explain how LSTM and GRU gates carry information and gradients, and why the forget-gate bias matters.
- Build an encoder-decoder with teacher forcing and additive attention, and explain why attention was invented.

### L2-05: Transformers

- Derive scaled dot-product self-attention as a soft lookup, compute it by hand, and explain the $1/\sqrt{d_k}$ factor with numbers.
- Explain why attention needs positional information, and compare sinusoidal, learned and rotary encodings.
- Assemble a decoder-only transformer block and explain why pre-LN trains more easily than post-LN.
- Train a small GPT on CPU, judge its loss against baselines, and sample from it.

### L2-06: Representation Learning and Embeddings

- Explain what an embedding is, and why "good embedding" only means something relative to a use.
- Train word2vec (skip-gram with negative sampling) from scratch and connect it to factorizing a PMI matrix with the SVD.
- Train autoencoders and a contrastive encoder without labels, and evaluate representations with probes rather than by their training loss.
- Build exact and approximate (IVF) nearest-neighbor search and measure the recall you give up for speed.
