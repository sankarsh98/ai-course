---
title: "Transformers"
---

> **Level 2 · Module 5** · Prerequisites: L0-02, L0-04, L2-01, L2-02, L2-04 · Time: 8 to 10 hours · Code: `code/L2-05-transformers/`

## What you will be able to do

- Derive scaled dot-product self-attention as a soft lookup, compute it by hand, and explain the $1/\sqrt{d_k}$ factor with numbers.
- Explain why attention needs positional information, and compare sinusoidal, learned and rotary encodings.
- Assemble a decoder-only transformer block and explain why pre-LN trains more easily than post-LN.
- Train a small GPT on CPU, judge its loss against baselines, and sample from it.
- Count the FLOPs and memory of attention, and judge claims about long context and fast kernels.

## 1. Why replace recurrence

A recurrent network (L2-04) reads tokens one at a time, so training cannot run in parallel across positions, and information from token 1 must survive every state update to reach token $t$. L2-04 showed attention added to a recurrent model so the decoder could look back directly. The transformer keeps the attention and drops the recurrence [Vaswani et al., 2017]. Their Table 1 states the trade: a self-attention layer costs $O(n^2 d)$ for length $n$ and width $d$, with $O(1)$ sequential steps and an $O(1)$ path between any two positions; a recurrent layer costs $O(nd^2)$, with $O(n)$ sequential steps and an $O(n)$ path. The $O(1)$ steps make training fast on parallel hardware. The $n^2$ makes long inputs expensive (section 9).

## 2. Self-attention as a soft lookup

A Python `dict` returns the value whose key matches the query exactly. Attention makes the lookup soft: compare the query with every key, turn the similarities into weights summing to 1, and return the weighted average of the values. In self-attention every token supplies all three. With token vectors as the rows of $X \in \mathbb{R}^{T \times d}$ and learned $W_Q, W_K \in \mathbb{R}^{d \times d_k}$, $W_V \in \mathbb{R}^{d \times d_v}$:

$$Q = XW_Q,\quad K = XW_K,\quad V = XW_V,\qquad \operatorname{Attention}(Q,K,V) = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.$$

Entry $(i, j)$ of $QK^\top$ is a dot-product similarity (L0-02) between token $i$'s query and token $j$'s key. The softmax runs along each row, so row $i$ of the weights is a distribution over the tokens that token $i$ reads, and output $i$ is the matching average of values. The separate maps let a token be found for one reason (its key) and pass along something else (its value).

**Worked example** (script 01). Tokens $x_1 = (1,0)$, $x_2 = (0,1)$, $x_3 = (1,1)$; $W_Q = W_K = I$; $W_V$ swaps the coordinates. Then $QK^\top = \begin{pmatrix} 1&0&1\\0&1&1\\1&1&2 \end{pmatrix}$. Row 1 divided by $\sqrt 2$ is $(0.707, 0, 0.707)$; exponentiating gives $(2.03, 1, 2.03)$, so the weights are $(0.401, 0.198, 0.401)$. With values $(0,1), (1,0), (1,1)$, output 1 is $0.401(0,1) + 0.198(1,0) + 0.401(1,1) = (0.599, 0.802)$.

**No notion of order.** The formula never mentions positions. Shuffle the rows of $X$ and the outputs shuffle the same way (script 01: largest difference $5 \times 10^{-7}$). To self-attention alone, "dog bites man" and "man bites dog" are the same bag of vectors. Section 6 fixes this.

## 3. Why divide by $\sqrt{d_k}$

If $q$ and $k$ have independent entries of mean 0 and variance 1, then $q \cdot k$ is a sum of $d_k$ unit-variance terms, with standard deviation $\sqrt{d_k}$. Large scores push the softmax toward one-hot, where its Jacobian $\operatorname{diag}(p) - pp^\top$ is nearly zero and learning stalls; Vaswani et al. give this reason for the scale [Vaswani et al., 2017]. Script 01, with 64 keys:

| $d_k$ | std of $q\cdot k$ | max weight, unscaled | gradient, unscaled | max weight, scaled | gradient, scaled |
|---|---|---|---|---|---|
| 16 | 4.02 | 0.568 | 0.268 | 0.104 | 0.176 |
| 256 | 15.95 | 0.896 | 0.098 | 0.107 | 0.177 |
| 1024 | 31.85 | 0.948 | 0.055 | 0.107 | 0.177 |

Unscaled, wide heads attend almost one-hot and the gradient (with respect to the scores) shrinks fivefold; scaled, both stay constant. This is about the starting point: a trained model can still grow its query and key norms when it wants sharp attention.

## 4. Multi-head attention

One softmax gives one distribution per query, but a token may need several things at once: the previous character, the start of the word, the speaker named three lines up. **Multi-head attention** runs $h$ attentions in parallel on $d/h$-dimensional slices and mixes the results with an output matrix $W_O$:

$$\operatorname{MultiHead}(X) = [\operatorname{head}_1; \dots; \operatorname{head}_h]\,W_O,\qquad \operatorname{head}_i = \operatorname{Attention}(XW_Q^{(i)}, XW_K^{(i)}, XW_V^{(i)}).$$

The base model of Vaswani et al. had $d = 512$ and 8 heads of width 64. With head width $d/h$, the parameter count equals one full-width head ($4d^2$), and in code it is a reshape: in `gpt.py`, `nn.Linear(C, 3C)` makes all queries, keys and values, `.view(B, T, H, D).transpose(1, 2)` splits heads, and `proj` is $W_O$.

## 5. Masking and the decoder-only model

A language model predicts token $t+1$ from tokens $1..t$. If position $t$ could attend to $t+1$ it would learn to copy. The **causal mask** sets scores above the diagonal to $-\infty$, so their weights are exactly zero. Script 01 changes the last of five tokens: without the mask all five outputs change, with it only the last.

The mask makes training efficient: one pass over a 128-token window gives 128 predictions, each with exactly the right context, computed in parallel. Inputs are `data[i : i+T]`, targets the same window shifted by one, and the loss is the mean cross-entropy (L0-04) over positions. This is teacher forcing (L2-04) with no loop.

A **decoder-only** model is just a stack of masked self-attention and MLP blocks, with no encoder. The original transformer was an encoder-decoder for translation; GPT-style models keep only the masked stack, and the GPT-2 paper describes its model as a Transformer that largely follows the earlier OpenAI GPT with a few modifications [Radford et al., 2019]. Generation is a loop: run the model, sample from the distribution at the last position, append, repeat (`GPT.generate`). Recomputing the whole prefix each step is wasteful; real systems cache keys and values. The KV cache, quantization and speculative decoding are covered in the companion "AI Systems" course (https://sankarsh98.github.io/ai-systems/) and are not repeated here.

## 6. Positional information

**Sinusoidal encodings** [Vaswani et al., 2017] add a fixed vector to each token embedding:

$$PE(t, 2i) = \sin(t\,\theta_i),\qquad PE(t, 2i+1) = \cos(t\,\theta_i),\qquad \theta_i = 10000^{-2i/d}.$$

Each pair of dimensions is a clock hand turning at its own speed. The authors chose this because $PE(t+k)$ is a linear function of $PE(t)$ for fixed $k$, which might let the model attend by relative position; in fact it is a rotation of each pair by $k\theta_i$, the same for every $t$. Script 02 confirms this to $2 \times 10^{-6}$ and shows $PE(t) \cdot PE(t+5) = 6.137$ for all $t$. They also reported that learned position embeddings gave nearly identical results.

**Learned embeddings** are a trainable table, one vector per position, as in GPT-2 and `gpt.py`. Simple, but there is no row for position `block_size`, so the model cannot run on longer inputs at all.

**Rotary embeddings (RoPE)** [Su et al., 2021] add nothing to the token. They rotate each coordinate pair of every query and key by an angle proportional to its position. In two dimensions, with $R(\alpha)$ a rotation,

$$\big(R(m\theta)q\big)^\top \big(R(n\theta)k\big) = q^\top R(m\theta)^\top R(n\theta)\,k = q^\top R\big((n-m)\theta\big)\,k,$$

since $R(\alpha)^\top = R(-\alpha)$ and rotations add angles. Position enters only through the offset. Script 02: the same token pair at offset 2 scores 5.2376 at positions (3, 1), (103, 101) and (1003, 1001) (5.2375 for the last, float32 rounding), while with added sinusoids it scores 2.55, 10.04 and $-0.94$. Rotations preserve length, so $\lVert q \rVert$ is unchanged.

Even with no positional encoding at all, causal language models were found to be competitive, apparently because the mask lets each token infer how many predecessors it has [Haviv et al., 2022]. Extending context length in practice is part of L3-01.

## 7. The block: residual connections, MLP and layer norm

A pre-LN block computes

$$x \leftarrow x + \operatorname{Attn}(\operatorname{LN}(x)),\qquad x \leftarrow x + \operatorname{MLP}(\operatorname{LN}(x)),$$

where the MLP has hidden width $4d$ and a GELU, applied to each position separately. Attention moves information between positions; the MLP transforms it within one. The MLP holds two-thirds of a block's weights ($8d^2$ of $12d^2$). Each block reads the running vector $x$ and adds a correction, as in the residual networks of L2-03, so gradients flow straight down the sums and deep stacks train. **Layer norm** rescales each token's vector to zero mean and unit variance over its features, then applies a learned scale and shift; unlike batch norm (L2-02) it ignores the rest of the batch.

**Where the norm goes matters.** The original transformer used **post-LN**, $x \leftarrow \operatorname{LN}(x + \operatorname{Sublayer}(x))$, with learning-rate warmup over the first 4,000 steps [Vaswani et al., 2017]. GPT-2 moved layer norm to the input of each sub-block (**pre-LN**), added a final layer norm, and scaled residual-layer weights at initialization by $1/\sqrt{N}$ for $N$ residual layers [Radford et al., 2019]; `gpt.py` copies all three. Xiong et al. argue that at initialization post-LN has large gradients near the output, which is why it needs warmup, while pre-LN is well behaved and can drop warmup; their evidence includes the last layer's FFN gradient staying constant with depth for post-LN and shrinking for pre-LN [Xiong et al., 2020]. Script 03 checks both claims:

- **Gradient scale.** With Xavier initialization, as in their analysis, the last-layer gradient stays at 0.17 to 0.25 for post-LN from 3 to 24 layers and falls from 0.215 to 0.071 for pre-LN, matching their result. With our GPT-2-style initialization the pattern differs (post-LN about 0.04 at every depth, pre-LN about 1.0), so the gradient story depends on initialization.
- **Training.** Six layers, Adam at $10^{-3}$, 200 steps, batch 16, three seeds. Post-LN without warmup ended at 3.318, 3.309 and 3.305: exactly the unigram entropy of the corpus (3.31). It learned the character frequencies and ignored context. With 100 warmup steps, post-LN reached 2.52; pre-LN without warmup reached 2.46 to 2.52.

Treat this as an illustration, not a general law: in exploratory runs not shown, a width-64 model never got stuck, and at batch 32 one of five post-LN seeds escaped without warmup. Post-LN is fragile, not hopeless.

**Weight tying.** The output layer has the same shape as the embedding table, and Vaswani et al. shared one matrix between them; `gpt.py` does too.

## 8. A small GPT on CPU

`gpt.py` is the whole model in about 110 lines: token and learned position embeddings, pre-LN blocks, a final layer norm, a tied output layer, and sampling. Each block has $12C^2 + 13C$ parameters (attention $4C^2 + 4C$, MLP $8C^2 + 5C$, layer norms $4C$), so four blocks of width $C = 96$ with 66 characters and 128 positions give $4 \times 111{,}840 + 6{,}336 + 12{,}288 + 192 = 466{,}176$, as the script prints.

**Data.** `data/romeo_and_juliet.txt` is Shakespeare's *Romeo and Juliet*, Project Gutenberg eBook #1513: 142,281 characters, 66 distinct. The play is in the public domain. Project Gutenberg's license page says that once its license and all references to Project Gutenberg are removed, the public-domain text can be used freely in the United States (users elsewhere should check local law). `prepare_corpus.py` does that stripping and maps curly quotes to ASCII. Tokens are characters; real models use subword tokenizers (L3-01).

**Baselines.** Knowing nothing scores $\ln 66 = 4.19$ nats per character; knowing only character frequencies scores 3.27 on validation. Below 3.27, the model is using context.

**Training.** AdamW (L0-05), peak learning rate $4 \times 10^{-3}$ with 50 warmup steps and cosine decay, 800 steps of 32 windows of 128 characters, gradient clipping at 1.0, the last 10% of the play held out. It took 52 s on an Intel i7-13700HX that other jobs kept busy (load average 2 to 8 during the run), so an idle machine should be faster.

| step | 0 | 200 | 400 | 600 | 800 |
|---|---|---|---|---|---|
| train loss | 4.203 | 2.283 | 1.863 | 1.633 | 1.510 |
| validation loss | 4.209 | 2.325 | 1.926 | 1.770 | 1.704 |

A validation loss of 1.704 nats is 2.46 bits per character, a perplexity of $e^{1.704} \approx 5.5$: as uncertain as a fair choice among 5.5 characters, against 66 at the start. The 0.19 gap to training loss is partly overfitting (no dropout, about 26 passes over the text) and partly that Act V differs from the rest.

**What it learned.** Before training the sample is noise. After it, the sample has the play's layout: speaker tags in capitals on their own line ("NURSE.", "ROMEO."), blank lines between speeches, capitalized verse lines, bracketed stage directions, and English-like fragments ("What the so lovines, poisc to mother."). Of five tags (one is the prompt), three are real speakers; "FIRSTAR STAMPSON." blends "FIRST" and "SAMPSON". Of the lowercase word-like strings, 75% occur in the play, though short words make that easy. It has learned spelling and format, not meaning, which is about what half a million parameters and a minute of CPU can buy.

## 9. What attention costs

Forward FLOPs for one layer on $T$ tokens of width $d$ (2 FLOPs per multiply-add, L0-02): projections $8Td^2$, MLP $16Td^2$, and the scores plus weighted sum $4T^2d$. The quadratic part dominates when $T > 6d$. For $d = 768$, the width of the smallest GPT-2 [Radford et al., 2019], that is 4,608 tokens; script 04 shows attention at 18% of the layer's FLOPs for $T = 1024$ and 88% at 32,768. Memory bites first: a naive implementation stores a $T \times T$ matrix per head, 3.2 GB for 12 heads at $T = 8192$ in float32, for one layer of one sequence.

On CPU, naive causal attention went from 0.98 ms at $T = 256$ to 174 ms at $T = 4096$, multiplying by 2.9 to 4.4 per doubling (ideal: 4). PyTorch's fused `scaled_dot_product_attention` (SDPA) was 6 to 9 times faster at the longest lengths, with outputs matching to within $6 \times 10^{-7}$. The machine was loaded, so treat these as rough.

FlashAttention computes exact attention in tiles that fit in fast on-chip GPU memory and never writes the $T \times T$ matrix to slow memory [Dao et al., 2022]; SDPA can dispatch to FlashAttention-2, a memory-efficient kernel, or a plain C++ implementation depending on the inputs [PyTorch, SDPA docs]. Such kernels remove the quadratic memory and much of the time, but the FLOPs stay $4T^2d$. When you hear "linear attention", ask whether it is exact (then the claim is about memory) or approximate (then ask what quality it loses on long-range tasks).

## Hands-on

Run each script from the repository root, for example `.venv/bin/python code/L2-05-transformers/train_gpt.py`. Outputs come from Python 3.12.3, PyTorch 2.15.0 (CPU) and NumPy 2.5.4 on Linux (WSL2) with 12 PyTorch threads, on a machine shared with other heavy jobs. Seeded results reproduced exactly across reruns here; other versions or CPUs may differ slightly. Timings depend on load.

### Corpus preparation (optional; the cleaned file ships with the code)

```python
"""Download Romeo and Juliet from Project Gutenberg and save a clean copy.

The play is in the public domain. Project Gutenberg's license page says that
once the Project Gutenberg license and all references to Project Gutenberg
are stripped, the remaining public-domain text can be used freely. This
script keeps only the play itself and normalizes typographic quotes and
line endings so the character vocabulary stays small.

The cleaned file is already in the repository (data/romeo_and_juliet.txt);
run this only to reproduce it.
"""
import pathlib
import urllib.request

URL = "https://www.gutenberg.org/cache/epub/1513/pg1513.txt"
OUT = pathlib.Path(__file__).parent / "data" / "romeo_and_juliet.txt"

raw = urllib.request.urlopen(URL, timeout=60).read().decode("utf-8")
raw = raw.replace("\r\n", "\n")
start = raw.index("THE TRAGEDY OF ROMEO AND JULIET")
end = raw.index("*** END OF THE PROJECT GUTENBERG EBOOK")
text = raw[start:end].strip() + "\n"

# Map curly quotes and dashes to ASCII; drop the stage-direction underscores.
for a, b in {"\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"',
             "\u2014": "--", "\u2013": "-", "_": ""}.items():
    text = text.replace(a, b)
assert "Gutenberg" not in text, "a Project Gutenberg reference survived"

OUT.parent.mkdir(exist_ok=True)
OUT.write_text(text, encoding="utf-8")
print(f"saved {OUT.name}: {len(text):,} characters, "
      f"{len(set(text))} distinct, {text.count(chr(10)):,} lines")
print("non-ASCII characters left:", sorted({c for c in text if ord(c) > 127}))
```

```text
saved romeo_and_juliet.txt: 142,281 characters, 66 distinct, 5,265 lines
non-ASCII characters left: ['æ']
```

The one non-ASCII character left is the "æ" of "Dramatis Personæ".

### Script 01: attention from first principles

```python
"""Self-attention from first principles: a hand-sized example, the sqrt(d_k)
scale, a check against PyTorch, permutation equivariance, the causal mask."""
import math

import torch
import torch.nn.functional as F

torch.manual_seed(0)
torch.set_printoptions(precision=3, sci_mode=False)


def attention(q, k, v, causal=False):
    """softmax(q k^T / sqrt(d_k)) v for tensors shaped (..., T, d)."""
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
    if causal:
        T = q.size(-2)
        mask = torch.ones(T, T, dtype=torch.bool).tril()
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = scores.softmax(dim=-1)
    return weights @ v, weights


# ---- 1. Worked example: W_Q = W_K = identity, W_V swaps the coordinates. ----
X = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
W_v = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
Q, K, Vv = X, X, X @ W_v
out, w = attention(Q, K, Vv)
print("1. worked example")
print("raw scores Q K^T:\n", Q @ K.T)
print("weights (rows sum to 1):\n", w)
print("outputs:\n", out)

# ---- 2. Why divide by sqrt(d_k)? q.k has variance d_k for unit-variance entries.
print("\n2. score scale vs head width d_k (64 keys, 2000 random queries)")
print(f"{'d_k':>5} {'std(q.k)':>9} | {'max weight':>10} {'grad norm':>10}"
      f" | {'max w, scaled':>13} {'grad, scaled':>12}")
n_keys = 64
for d in (4, 16, 64, 256, 1024):
    q = torch.randn(2000, d)
    k = torch.randn(n_keys, d)
    raw = q @ k.T
    row = []
    for s in (raw, raw / math.sqrt(d)):
        s = s.clone().requires_grad_(True)
        p = s.softmax(dim=-1)
        # Stand-in for the loss gradient: a random linear function of p.
        (p * torch.randn_like(p)).sum().backward()
        row += [p.max(dim=-1).values.mean().item(), s.grad.norm(dim=-1).mean().item()]
    print(f"{d:>5} {raw.std().item():>9.2f} | {row[0]:>10.3f} {row[1]:>10.4f}"
          f" | {row[2]:>13.3f} {row[3]:>12.4f}")

# ---- 3. Check against PyTorch ------------------------------------------------
B, H, T, D = 2, 4, 10, 16
q, k, v = (torch.randn(B, H, T, D) for _ in range(3))
mine, _ = attention(q, k, v, causal=True)
ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
print(f"\n3. max |ours - F.scaled_dot_product_attention| = "
      f"{(mine - ref).abs().max().item():.2e}")

# ---- 4. Permutation equivariance ----------------------------------------------
d_model = 8
Wq, Wk, Wv = (torch.randn(d_model, d_model) for _ in range(3))
x = torch.randn(5, d_model)
perm = torch.tensor([3, 0, 4, 1, 2])


def self_attn(x):
    return attention(x @ Wq, x @ Wk, x @ Wv)[0]


diff = (self_attn(x[perm]) - self_attn(x)[perm]).abs().max().item()
print(f"\n4. shuffle inputs, outputs shuffle the same way: max diff {diff:.2e}")
print("   so 'dog bites man' and 'man bites dog' get the same set of vectors")

# ---- 5. Causal mask -------------------------------------------------------------
x2 = x.clone()
x2[-1] = torch.randn(d_model)          # change only the last token
for causal in (False, True):
    a = attention(x @ Wq, x @ Wk, x @ Wv, causal)[0]
    b = attention(x2 @ Wq, x2 @ Wk, x2 @ Wv, causal)[0]
    changed = (a - b).abs().amax(dim=-1) > 1e-6
    print(f"5. causal={causal!s:5}: outputs changed at positions "
          f"{changed.nonzero().flatten().tolist()}")
```

```text
1. worked example
raw scores Q K^T:
 tensor([[1., 0., 1.],
        [0., 1., 1.],
        [1., 1., 2.]])
weights (rows sum to 1):
 tensor([[0.401, 0.198, 0.401],
        [0.198, 0.401, 0.401],
        [0.248, 0.248, 0.503]])
outputs:
 tensor([[0.599, 0.802],
        [0.802, 0.599],
        [0.752, 0.752]])

2. score scale vs head width d_k (64 keys, 2000 random queries)
  d_k  std(q.k) | max weight  grad norm | max w, scaled grad, scaled
    4      1.82 |      0.221     0.2294 |         0.085       0.1652
   16      4.02 |      0.568     0.2676 |         0.104       0.1760
   64      7.89 |      0.785     0.1849 |         0.106       0.1767
  256     15.95 |      0.896     0.0976 |         0.107       0.1766
 1024     31.85 |      0.948     0.0554 |         0.107       0.1769

3. max |ours - F.scaled_dot_product_attention| = 3.58e-07

4. shuffle inputs, outputs shuffle the same way: max diff 4.77e-07
   so 'dog bites man' and 'man bites dog' get the same set of vectors
5. causal=False: outputs changed at positions [0, 1, 2, 3, 4]
5. causal=True : outputs changed at positions [4]
```

### Script 02: positional encodings

```python
"""Sinusoidal (added) and rotary (RoPE) positional encodings, checked numerically:
1. PE(t + k) is a rotation of PE(t) that does not depend on t.
2. With added encodings, a q.k score depends on absolute positions.
3. With RoPE, it depends only on the offset m - n."""
import torch

torch.manual_seed(0)
d = 16                                    # must be even: dimensions come in pairs
freqs = 10000.0 ** (-torch.arange(0, d, 2) / d)   # theta_i, one per pair


def sinusoidal(positions):
    """Rows are PE(t): sin in even dims, cos in odd dims (Vaswani et al., 2017)."""
    ang = positions[:, None].float() * freqs[None, :]       # (T, d/2)
    pe = torch.zeros(len(positions), d)
    pe[:, 0::2], pe[:, 1::2] = ang.sin(), ang.cos()
    return pe


def rotate_pairs(x, angles):
    """Rotate each (even, odd) coordinate pair of x by the given angles."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c, s = angles.cos(), angles.sin()
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * c - x2 * s
    out[..., 1::2] = x1 * s + x2 * c
    return out


def rope(x, pos):
    """RoPE (Su et al., 2021): rotate pair i of a vector at position pos by pos * theta_i."""
    return rotate_pairs(x, pos * freqs)


# ---- 1. Sinusoidal: shifting by k is a rotation that does not depend on t.
pos = torch.arange(0, 200)
pe = sinusoidal(pos)
k = 7
# Pairs are stored as (sin, cos), so in rotate_pairs' convention the shift by
# k is a rotation by -k * theta_i, the same for every t.
shifted = torch.stack([rotate_pairs(pe[t], -k * freqs) for t in range(len(pos) - k)])
err = (shifted - pe[k:]).abs().max().item()
print(f"1. max |R_k PE(t) - PE(t+k)| over t < {len(pos) - k}: {err:.1e}")
for t in (0, 50, 120):
    print(f"   PE({t:3d}) . PE({t + 5:3d}) = {(pe[t] @ pe[t + 5]).item():.4f}")

# ---- 2. Added encodings: the score depends on absolute position. -----------
Wq, Wk = torch.randn(d, d) / d ** 0.5, torch.randn(d, d) / d ** 0.5
a, b = torch.randn(d), torch.randn(d)      # two token embeddings ("cat", "sat")
print("\n2. added PE: score of the same token pair at offset 2, different places")
for m in (3, 103, 1003):
    xm = a + sinusoidal(torch.tensor([m]))[0]
    xn = b + sinusoidal(torch.tensor([m - 2]))[0]
    print(f"   positions ({m:4d}, {m - 2:4d}): {((xm @ Wq) @ (xn @ Wk)).item():.4f}")

# ---- 3. RoPE: the score depends only on m - n. -------------------------------
q, kk = a @ Wq, b @ Wk
print("\n3. RoPE: same pair, offset 2, different places")
for m in (3, 103, 1003):
    s = rope(q, m) @ rope(kk, m - 2)
    print(f"   positions ({m:4d}, {m - 2:4d}): {s.item():.4f}")
print("   offsets 0..5 at position 1000:",
      [round((rope(q, 1000) @ rope(kk, 1000 - j)).item(), 3) for j in range(6)])
# Rotations preserve length, so RoPE never changes |q| or |k|.
print(f"   |q| before {q.norm():.4f}, after rotation {rope(q, 1003).norm():.4f}")
```

```text
1. max |R_k PE(t) - PE(t+k)| over t < 193: 1.9e-06
   PE(  0) . PE(  5) = 6.1370
   PE( 50) . PE( 55) = 6.1370
   PE(120) . PE(125) = 6.1370

2. added PE: score of the same token pair at offset 2, different places
   positions (   3,    1): 2.5453
   positions ( 103,  101): 10.0442
   positions (1003, 1001): -0.9354

3. RoPE: same pair, offset 2, different places
   positions (   3,    1): 5.2376
   positions ( 103,  101): 5.2376
   positions (1003, 1001): 5.2375
   offsets 0..5 at position 1000: [3.471, 4.405, 5.238, 5.193, 4.282, 3.285]
   |q| before 3.1014, after rotation 3.1014
```

### The model, `gpt.py` (imported by the other scripts; prints nothing)

```python
"""A small decoder-only transformer (GPT) in plain PyTorch.

Shapes: B = batch, T = sequence length, C = model width,
H = number of heads, D = C // H = head width.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 128      # maximum context length
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128          # C
    pre_ln: bool = True        # False gives the original post-LN layout


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)  # all heads at once
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)     # W_O, mixes heads

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, H, T, D): each head attends independently.
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
                   for t in (q, k, v))
        # softmax(q k^T / sqrt(D)) v with the causal mask, as a fused kernel.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # concatenate heads
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.out = nn.Linear(4 * cfg.n_embd, cfg.n_embd)

    def forward(self, x):
        return self.out(F.gelu(self.fc(x)))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.pre_ln = cfg.pre_ln
        self.ln1, self.ln2 = nn.LayerNorm(cfg.n_embd), nn.LayerNorm(cfg.n_embd)
        self.attn, self.mlp = CausalSelfAttention(cfg), MLP(cfg)

    def forward(self, x):
        if self.pre_ln:   # normalize sublayer inputs; residual path untouched
            x = x + self.attn(self.ln1(x))
            return x + self.mlp(self.ln2(x))
        x = self.ln1(x + self.attn(x))   # post-LN: normalize after the sum
        return self.ln2(x + self.mlp(x))


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)  # learned
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd) if cfg.pre_ln else nn.Identity()
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight   # weight tying
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)
        # GPT-2 style: shrink layers that write into the residual stream by
        # 1/sqrt(number of residual layers) so its variance does not grow.
        for name, p in self.named_parameters():
            if name.endswith(("proj.weight", "out.weight")):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def forward(self, idx, targets=None):
        T = idx.shape[1]
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T))
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))          # (B, T, vocab)
        loss = None
        if targets is not None:   # every position predicts the next token
            loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, n_new, temperature=1.0, top_k=None):
        for _ in range(n_new):
            logits, _ = self(idx[:, -self.cfg.block_size:])  # crop context
            logits = logits[:, -1] / temperature              # last position
            if top_k is not None:
                kth = torch.topk(logits, top_k).values[:, [-1]]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            nxt = torch.multinomial(logits.softmax(-1), num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx


def count_params(model):
    return sum(p.numel() for p in model.parameters())  # tied weights once
```

### Training, `train_gpt.py`

```python
"""Train a small character-level GPT on Romeo and Juliet, on CPU.

Prints baselines, losses during training, samples before and after, and a
check of whether the samples follow the play's format.
"""
import math
import pathlib
import re
import time

import torch

from gpt import GPT, GPTConfig, count_params

torch.manual_seed(0)
text = (pathlib.Path(__file__).parent / "data" / "romeo_and_juliet.txt").read_text(
    encoding="utf-8")

# Tokenize: one token per character. The last 10% of the play is validation.
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
encode = lambda s: torch.tensor([stoi[c] for c in s])
decode = lambda t: "".join(chars[i] for i in t.tolist())
data = encode(text)
n_train = int(0.9 * len(data))
train_data, val_data = data[:n_train], data[n_train:]
V = len(chars)
print(f"corpus: {len(text):,} chars, vocab {V}, "
      f"train {len(train_data):,} / val {len(val_data):,}")

# Baselines a model must beat (nats per character).
unigram = torch.bincount(train_data, minlength=V).float() + 1
unigram /= unigram.sum()
print(f"uniform baseline {math.log(V):.3f}, "
      f"unigram baseline {-unigram.log()[val_data].mean():.3f}")

cfg = GPTConfig(vocab_size=V, block_size=128, n_layer=4, n_head=4, n_embd=96)
model = GPT(cfg)
print(f"parameters: {count_params(model):,}")
BATCH, STEPS, LR, WARMUP = 32, 800, 4e-3, 50


def get_batch(src):
    ix = torch.randint(len(src) - cfg.block_size - 1, (BATCH,))
    x = torch.stack([src[i:i + cfg.block_size] for i in ix])
    y = torch.stack([src[i + 1:i + cfg.block_size + 1] for i in ix])  # shifted
    return x, y


@torch.no_grad()
def estimate_loss(src, n_batches=10):
    return sum(model(*get_batch(src))[1].item() for _ in range(n_batches)) / n_batches


def lr_at(step):   # linear warmup, then cosine decay to 10% of the peak
    if step < WARMUP:
        return LR * (step + 1) / WARMUP
    progress = (step - WARMUP) / (STEPS - WARMUP)
    return LR * (0.1 + 0.45 * (1 + math.cos(math.pi * progress)))


def sample(n, seed=1):
    torch.manual_seed(seed)
    start = encode("ROMEO.\n").unsqueeze(0)
    return decode(model.generate(start, n, temperature=0.8, top_k=20)[0])


print("\n--- sample before training ---\n" + sample(120))

opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.99), weight_decay=0.1)
t0 = time.time()
for step in range(STEPS + 1):
    if step % 200 == 0:
        print(f"step {step:4d}  train {estimate_loss(train_data):.3f}  "
              f"val {estimate_loss(val_data):.3f}  ({time.time() - t0:5.1f} s)")
    if step == STEPS:
        break
    for g in opt.param_groups:
        g["lr"] = lr_at(step)
    _, loss = model(*get_batch(train_data))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
print(f"training time {time.time() - t0:.1f} s, {torch.get_num_threads()} threads")

generated = sample(600)
print("\n--- sample after training ---\n" + generated)

# Did it learn the format? A speaker tag is a line like "JULIET." (the
# prompt "ROMEO." counts as one).
tag = re.compile(r"^([A-Z][A-Z ]+)\.$", re.M)
real_speakers = set(tag.findall(text))
gen_tags = tag.findall(generated)
words = re.findall(r"[a-z]+", generated)
real_words = set(re.findall(r"[a-z]+", text))
print(f"\nspeaker tags: {gen_tags}, real: {sum(t in real_speakers for t in gen_tags)}")
print(f"lowercase words: {len(words)}, "
      f"found in the play: {sum(w in real_words for w in words) / len(words):.0%}")
```

```text
corpus: 142,281 chars, vocab 66, train 128,052 / val 14,229
uniform baseline 4.190, unigram baseline 3.272
parameters: 466,176

--- sample before training ---
ROMEO.

Sstznon]zhhIac
:dr
vkkpz
psLCrr,Vld"&UEE;y;;e,:r;g]-BWWaaYFh[[A&YnVFnG'in:nnL--Tæ  VNbq!!jæJ,,]Ja''ZIa&pO""EEE-,,mQGP!!
step    0  train 4.203  val 4.209  (  0.4 s)
step  200  train 2.283  val 2.325  ( 12.1 s)
step  400  train 1.863  val 1.926  ( 24.2 s)
step  600  train 1.633  val 1.770  ( 37.7 s)
step  800  train 1.510  val 1.704  ( 52.2 s)
training time 52.2 s, 12 threads

--- sample after training ---
ROMEO.
Come, and hence edgeries with her lays;
Signees him so that heaven that in not dow him
wist to suden to lady of the love,
Thou hath fall the-geicant she cancee and.

 [Kitt he leving in the sperss.]

SECONE tpers, I cheek the be is thee, unly be my his.

FIRSTAR STAMPSON.
Then be I canouse, we she thy heaven me,
We here for foll that a e'er ancels aggains
That thing and long werch we pers.
Colls oume, and lady the show, that lad, men yet
Then the heads driesss neved the well.

ROMEO.
What the so lovines, poisc to mother.

PARION.
Whost hame, and thou should you best.

NURSE.
Say, we I fall I s

speaker tags: ['ROMEO', 'FIRSTAR STAMPSON', 'ROMEO', 'PARION', 'NURSE'], real: 3
lowercase words: 109, found in the play: 75%
```

### Script 03: pre-LN versus post-LN

```python
"""Pre-LN versus post-LN: gradient scale at initialization, and a training race.

Xiong et al. (2020) show that at initialization, the gradient of the last
layer's FFN weights stays roughly constant with depth in post-LN
transformers but shrinks with depth in pre-LN ones, and argue that the
large post-LN gradients near the output are why post-LN needs learning-rate
warmup. Part 1 repeats that measurement on our model. Part 2 trains both
layouts with and without warmup.
"""
import math
import pathlib
import time

import torch
import torch.nn as nn

from gpt import GPT, GPTConfig

text = (pathlib.Path(__file__).parent / "data" / "romeo_and_juliet.txt").read_text(encoding="utf-8")
chars = sorted(set(text))
data = torch.tensor([chars.index(c) for c in text])
V, T = len(chars), 64


def batch(gen, size=32):
    ix = torch.randint(len(data) - T - 1, (size,), generator=gen)
    return (torch.stack([data[i:i + T] for i in ix]),
            torch.stack([data[i + 1:i + T + 1] for i in ix]))


def make(pre_ln, n_layer, seed=0):
    torch.manual_seed(seed)
    return GPT(GPTConfig(vocab_size=V, block_size=T, n_layer=n_layer,
                         n_head=4, n_embd=128, pre_ln=pre_ln))


# ---- 1. Gradient norm of the last FFN output matrix at init, vs depth. -----
def xavier(model):
    # Xavier-uniform matrices, the setting Xiong et al. analyze.
    for name, p in model.named_parameters():
        if p.dim() == 2 and "emb" not in name:
            nn.init.xavier_uniform_(p)
    return model


print("1. |grad| of the last block's MLP output weights at init (mean of 3 batches)")
print(f"   {'init':<12} {'layout':<8} " + " ".join(f"{f'L={L}':>7}" for L in (3, 6, 12, 24)))
for init_name, init in (("Xavier", xavier), ("GPT-2 style", lambda m: m)):
    for pre_ln in (False, True):
        row = []
        for L in (3, 6, 12, 24):
            model = init(make(pre_ln, L))
            gen = torch.Generator().manual_seed(1)
            g = 0.0
            for _ in range(3):
                model.zero_grad()
                model(*batch(gen))[1].backward()
                g += model.blocks[-1].mlp.out.weight.grad.norm().item() / 3
            row.append(g)
        name = "pre-LN" if pre_ln else "post-LN"
        print(f"   {init_name:<12} {name:<8} " + " ".join(f"{g:>7.3f}" for g in row))

# ---- 2. Training race: 6 layers, Adam at peak lr 1e-3, 200 steps, 3 seeds. ---
STEPS, PEAK = 200, 1e-3


def train(pre_ln, warmup, seed):
    model = make(pre_ln, 6, seed)
    opt = torch.optim.Adam(model.parameters(), lr=PEAK)
    gen = torch.Generator().manual_seed(seed + 10)
    losses = []
    for step in range(STEPS):
        lr = PEAK * min(1.0, (step + 1) / warmup) if warmup else PEAK
        for g in opt.param_groups:
            g["lr"] = lr
        loss = model(*batch(gen, size=16))[1]
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return sum(losses[-25:]) / 25          # mean of the last 25 steps


unigram = torch.bincount(data, minlength=V).float() / len(data)
print(f"\n2. training loss after {STEPS} steps (mean of last 25), per seed")
print(f"   uniform baseline {math.log(V):.2f}, unigram baseline "
      f"{-(unigram * unigram.log()).sum():.2f}")
t0 = time.time()
for pre_ln, warmup in ((False, 0), (False, 100), (True, 0)):
    r = [train(pre_ln, warmup, seed) for seed in range(3)]
    name = "pre-LN" if pre_ln else "post-LN"
    print(f"   {name:<8} warmup {warmup:>3}: " + "  ".join(f"{x:.3f}" for x in r))
print(f"   ({time.time() - t0:.0f} s)")
```

```text
1. |grad| of the last block's MLP output weights at init (mean of 3 batches)
   init         layout       L=3     L=6    L=12    L=24
   Xavier       post-LN    0.173   0.222   0.251   0.246
   Xavier       pre-LN     0.215   0.165   0.126   0.071
   GPT-2 style  post-LN    0.038   0.040   0.041   0.038
   GPT-2 style  pre-LN     0.852   1.034   1.025   1.022

2. training loss after 200 steps (mean of last 25), per seed
   uniform baseline 4.19, unigram baseline 3.31
   post-LN  warmup   0: 3.318  3.309  3.305
   post-LN  warmup 100: 2.524  2.518  2.517
   pre-LN   warmup   0: 2.489  2.518  2.457
   (58 s)
```

The race took 58 s (load average rising from 1 to 8 during the run), so the whole script runs slightly over the course's 60-second guideline.

### Script 04: the cost of attention

```python
"""What attention costs: FLOP accounting and measured CPU time versus length.

Per layer and per sequence of T tokens with model width d (forward pass,
counting a multiply-add as 2 FLOPs):
  Q, K, V and output projections: 4 matrices of d x d -> 8 T d^2
  MLP with hidden width 4d:       2 matrices of d x 4d -> 16 T d^2
  scores Q K^T and weights @ V:   2 products of T x T x d -> 4 T^2 d
So the T^2 part overtakes the rest when 4 T^2 d > 24 T d^2, that is T > 6d.
"""
import math
import time

import torch
import torch.nn.functional as F

torch.manual_seed(0)


def layer_flops(T, d):
    linear = 24 * T * d * d
    attn = 4 * T * T * d
    return linear, attn


print("1. forward FLOPs per layer, width d = 768 (crossover at T = 6d = 4608)")
print(f"{'T':>7} {'linear GFLOP':>13} {'T^2 GFLOP':>10} {'T^2 share':>10}"
      f" {'scores MB (12 heads, fp32)':>27}")
for T in (512, 1024, 2048, 4096, 8192, 32768):
    lin, att = layer_flops(T, 768)
    score_mb = 12 * T * T * 4 / 1e6           # one T x T matrix per head
    print(f"{T:>7} {lin / 1e9:>13.1f} {att / 1e9:>10.1f} {att / (lin + att):>10.0%}"
          f" {score_mb:>27,.0f}")


def naive(q, k, v):
    s = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    mask = torch.ones(s.shape[-2:], dtype=torch.bool).tril()
    return s.masked_fill(~mask, float("-inf")).softmax(-1) @ v


def best_time(fn, *args, reps=5):
    fn(*args)                                   # warm up
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t0)
    return min(times)                           # min is least affected by load


print(f"\n2. measured causal attention, 4 heads x 64 dims, batch 1, "
      f"{torch.get_num_threads()} threads")
print(f"{'T':>6} {'naive ms':>9} {'x prev':>7} {'SDPA ms':>8} {'x prev':>7}")
prev = None
with torch.no_grad():
    for T in (256, 512, 1024, 2048, 4096):
        q, k, v = (torch.randn(1, 4, T, 64) for _ in range(3))
        tn = best_time(naive, q, k, v)
        ts = best_time(lambda a, b, c: F.scaled_dot_product_attention(
            a, b, c, is_causal=True), q, k, v)
        ratio = "" if prev is None else f"{tn / prev[0]:.1f}"
        ratio_s = "" if prev is None else f"{ts / prev[1]:.1f}"
        print(f"{T:>6} {tn * 1e3:>9.2f} {ratio:>7} {ts * 1e3:>8.2f} {ratio_s:>7}")
        prev = (tn, ts)

# Exactness: the fused kernel computes the same function, not an approximation.
q, k, v = (torch.randn(1, 4, 1024, 64) for _ in range(3))
diff = (naive(q, k, v) - F.scaled_dot_product_attention(q, k, v, is_causal=True)).abs().max()
print(f"\nmax difference naive vs SDPA at T=1024: {diff:.1e}")
```

```text
1. forward FLOPs per layer, width d = 768 (crossover at T = 6d = 4608)
      T  linear GFLOP  T^2 GFLOP  T^2 share  scores MB (12 heads, fp32)
    512           7.2        0.8        10%                          13
   1024          14.5        3.2        18%                          50
   2048          29.0       12.9        31%                         201
   4096          58.0       51.5        47%                         805
   8192         116.0      206.2        64%                       3,221
  32768         463.9     3298.5        88%                      51,540

2. measured causal attention, 4 heads x 64 dims, batch 1, 12 threads
     T  naive ms  x prev  SDPA ms  x prev
   256      0.98             0.25        
   512      2.86     2.9     0.47     1.9
  1024     10.32     3.6     1.88     4.0
  2048     39.83     3.9     7.02     3.7
  4096    173.63     4.4    19.89     2.8

max difference naive vs SDPA at T=1024: 6.3e-07
```

## Common mistakes

- **Unshifted targets.** If `y == x`, the model learns to copy the current token: training loss plunges and generation is garbage.
- **A missing mask.** Without it, loss falls far below what generation can achieve, because each position reads its own answer. A suspiciously low loss is a leak until proven otherwise.
- **Splitting heads with `view` alone.** `view(B, H, T, D)` on a `(B, T, C)` tensor scrambles tokens and heads. Use `view(B, T, H, D).transpose(1, 2)`.
- **Softmax over the wrong axis.** Weights must sum to 1 over keys; check `weights.sum(-1)`.
- **Comparing losses across tokenizers.** Nats per character and per subword token are different units; convert to bits per character or byte.
- **Running past `block_size`.** A learned position table has no entry for longer inputs; crop the context, as `generate` does.
- **Mistaking a plateau for convergence.** A loss stuck at the unigram entropy (3.31 here) means the model ignores context. Check learning rate, warmup and norm placement.

## Exercises

1. [warm-up] Compute rows 2 and 3 of the worked example by hand.
2. [warm-up] Derive $12C^2 + 13C$ per block from `gpt.py` and predict the parameter count for $C = 128$, 6 layers.
3. [warm-up] Why does a stuck post-LN model end at almost exactly 3.31?
4. [core] Train with `is_causal=False`. Report train and validation loss and a sample, and explain them.
5. [core] Delete the position embedding and retrain. How much worse is validation loss? Relate this to section 6.
6. [core] Replace learned positions with RoPE (apply `rope` from script 02 to each head's `q` and `k`). Compare validation loss over two or more seeds.
7. [core] Sample at temperatures 0.5, 0.8, 1.0 and 1.5; measure the fraction of real words and the number of distinct words.
8. [core] At width 96, train with 1, 4 and 8 heads. How would you tell a real difference from seed noise?
9. [stretch] Add a key-value cache to `generate`, verify identical output with `top_k=1`, and measure the speed-up.
10. [stretch] Map where post-LN without warmup gets stuck: learning rates from $3 \times 10^{-4}$ to $3 \times 10^{-3}$, warmup 0 to 100, three seeds each.
11. [stretch] Add `nn.Dropout(0.1)` on each residual branch, train 3,000 steps with and without it, and find where validation loss bottoms out.

### Hints and answers

1. Row 2 mirrors row 1: weights $(0.198, 0.401, 0.401)$, output $(0.802, 0.599)$. Row 3: scores $(1,1,2)/\sqrt2$, weights $(0.248, 0.248, 0.503)$, output $(0.752, 0.752)$.
2. QKV $3C^2 + 3C$, projection $C^2 + C$, MLP $4C^2 + 4C$ and $4C^2 + C$, layer norms $4C$. For $C = 128$: $6 \times 198{,}272 + 8{,}448 + 16{,}384 + 256 = 1{,}214{,}720$.
3. Ignoring context, the best prediction is the character frequencies, whose expected loss is the unigram entropy (L0-04).
4. Expect both losses (validation is also unmasked) to collapse far below 1.5 because each position sees the next character, and generation to fail because at sampling time the future does not exist.
5. Expect worse but not catastrophic: the causal mask still leaks position [Haviv et al., 2022].
6. Use the same pairing convention for `q` and `k`, positions `0..T-1`. Differences at this scale may be within seed noise.
7. Low temperature: repetitive text, many real words. High: more variety, more invented words.
8. Run at least three seeds per setting and compare the spread between settings with the spread across seeds (L1-01).
9. Keep per-layer `k` and `v` of shape `(B, H, T_so_far, D)`, append each new token's, and attend from one query; pass the right position index. The speed-up grows with length. Compare with the AI Systems KV-cache page.
10. At $10^{-3}$ with no warmup, all three seeds here got stuck. Expect a threshold below which no warmup is needed, rising with warmup length.
11. Without dropout, validation loss should flatten then rise while training loss keeps falling; dropout should delay that.

## Check your understanding

1. **Why does self-attention need positional information when a recurrent network does not?** A recurrent network processes tokens in order, so order is built in. Self-attention computes the same weighted averages whatever the row order: permuting inputs permutes outputs. Without encodings a non-causal transformer sees a bag of tokens (a causal one gets weak position information from the mask).

2. **What goes wrong without $1/\sqrt{d_k}$, and why is it worse for wider heads?** Random dot products have standard deviation $\sqrt{d_k}$, so wider heads give larger scores, a saturated softmax, near one-hot attention and vanishing gradients. The scale keeps score variance near 1 at any width.

3. **What does RoPE give that added sinusoids do not?** With RoPE, rotating query and key and taking their dot product leaves a single rotation by the offset, so the score depends on position only through $m - n$. With added encodings, content and absolute position mix, and the same pair at the same offset scores differently in different places.

4. **Why is pre-LN easier to train?** Its residual path is a plain sum from input to output, so gradients pass through unchanged and each sublayer only adds a correction. Xiong et al. argue post-LN has large gradients near the output at initialization, so it needs warmup. In script 03 post-LN without warmup stuck at the unigram loss for three of three seeds; pre-LN did not.

5. **"Our kernel makes attention linear." What do you ask?** Is it exact? If so, the claim is about memory: exact attention still costs $4T^2d$ FLOPs per layer, though it can be much faster in wall-clock time because memory traffic dominates. If it approximates, ask what it loses on long-range tasks. And ask at what lengths: below about $6d$ tokens the linear layers dominate anyway.

## Further reading

- **Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin, "Attention Is All You Need"** (2017, NeurIPS 30; arXiv:1706.03762). https://arxiv.org/abs/1706.03762
  *Read for:* the original architecture, the scaling argument, sinusoidal encodings and the complexity table. *Level:* intermediate.
- **Alexander (Sasha) Rush, "The Annotated Transformer"** (2018, Workshop for NLP Open Source Software; 2022 update by Austin Huang, Suraj Subramanian, Jonathan Sum, Khalid Almubarak, Stella Biderman). https://nlp.seas.harvard.edu/annotated-transformer/
  *Read for:* the paper rebuilt line by line in PyTorch. *Level:* intro to intermediate.
- **Simon J. D. Prince, "Understanding Deep Learning"** (2023, MIT Press). https://udlbook.github.io/udlbook/
  *Read for:* the transformers chapter's clear figures of attention and heads. Free PDF on the author's site. *Level:* intro.
- **Andrej Karpathy, "nanoGPT"** (GitHub repository, MIT license, 2022). https://github.com/karpathy/nanoGPT
  *Read for:* a compact GPT training codebase with a character-level Shakespeare example close to this module's. *Level:* intro to intermediate.
- **Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, "Language Models are Unsupervised Multitask Learners"** (2019, OpenAI report). https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
  *Read for:* GPT-2's pre-LN layout, scaled residual initialization and model sizes (section 2.3). *Level:* intermediate.
- **Ruibin Xiong, Yunchang Yang, Di He, Kai Zheng, Shuxin Zheng, Chen Xing, Huishuai Zhang, Yanyan Lan, Liwei Wang, Tie-Yan Liu, "On Layer Normalization in the Transformer Architecture"** (2020, ICML; arXiv:2002.04745). https://arxiv.org/abs/2002.04745
  *Read for:* why post-LN needs warmup and pre-LN does not. *Level:* advanced.
- **Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, Yunfeng Liu, "RoFormer: Enhanced Transformer with Rotary Position Embedding"** (2021, arXiv:2104.09864). https://arxiv.org/abs/2104.09864
  *Read for:* the derivation of rotary embeddings. *Level:* intermediate.
- **Adi Haviv, Ori Ram, Ofir Press, Peter Izsak, Omer Levy, "Transformer Language Models without Positional Encodings Still Learn Positional Information"** (2022, Findings of EMNLP; arXiv:2203.16634). https://arxiv.org/abs/2203.16634
  *Read for:* evidence that the causal mask alone carries position information. *Level:* intermediate.
- **Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré, "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"** (2022, arXiv:2205.14135). https://arxiv.org/abs/2205.14135
  *Read for:* why attention speed is about memory traffic, and exact attention without the $T \times T$ matrix. *Level:* advanced.
- **PyTorch Contributors, "torch.nn.functional.scaled_dot_product_attention"** (PyTorch 2.15 documentation, accessed October 2026). https://docs.pytorch.org/docs/2.15/generated/torch.nn.functional.scaled_dot_product_attention.html
  *Read for:* the `is_causal` and `scale` arguments and the available backends. *Level:* intro.
