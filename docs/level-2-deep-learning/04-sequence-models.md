---
title: "Sequence Models"
---

> **Level 2 · Module 4** · Prerequisites: L0-03, L0-04, L2-01, L2-02, L2-03 · Time: 6 to 8 hours · Code: `code/L2-04-sequence-models/`

## What you will be able to do

- Write down a recurrent network, unroll it, and explain backpropagation through time.
- Derive why RNN gradients vanish or explode, measure it, and use gradient clipping.
- Explain how LSTM and GRU gates carry information and gradients, and why the forget-gate bias matters.
- Build an encoder-decoder with teacher forcing and additive attention, and explain why attention was invented.
- Train a character-level language model, report bits per character against baselines, and compare greedy, beam, temperature and nucleus decoding.

## 1. Recurrence

Text, audio and sensor logs are sequences: order matters and lengths vary. A recurrent neural network (**RNN**) reads one element at a time and keeps a **hidden state** $h_t \in \mathbb{R}^H$ summarising everything so far:

$$h_t = \tanh(W h_{t-1} + U x_t + b), \qquad \hat{y}_t = V h_t + c.$$

The same $W$, $U$, $V$ are used at every step: weight sharing across time, as a convolution shares weights across space (L2-03). The parameter count does not depend on sequence length.

Unroll the loop for $T$ steps and you get a deep feed-forward graph with tied weights. Backpropagating through it (L2-01) is **backpropagation through time (BPTT)**; the gradient for $W$ sums its gradients at every step. For long streams such as a book, you cut the sequence into windows (say 64 steps) and backpropagate only within each: **truncated BPTT**. Dependencies longer than the window are then hard to learn.

## 2. Vanishing and exploding gradients

How does the state at step $T$ depend on the state at step 0? By the chain rule (L0-03),

$$\frac{\partial h_T}{\partial h_0} = \prod_{t=1}^{T} J_t, \qquad J_t = \frac{\partial h_t}{\partial h_{t-1}} = \operatorname{diag}\big(1 - h_t^2\big)\, W,$$

where $1 - h_t^2 \le 1$ is the derivative of $\tanh$. A product of $T$ matrices behaves like a power. In the scalar case $h_t = w h_{t-1}$ the gradient is $w^T$: $0.9^{50} = 0.005$, $1.1^{50} = 117$. In general $\|\partial h_T/\partial h_0\| \le \prod_t \|W\|$, so if the largest singular value of $W$ is below 1 the gradient vanishes exponentially; if it is well above 1 it can explode [Bengio et al., 1994; Pascanu et al., 2013].

Script 1 measures this. With $W$ a scaled orthogonal matrix and no nonlinearity, the norm is exactly $\rho^T$. With $\tanh$ and a Gaussian $W$ of scale $g$, the gradient after 100 steps is about $10^{-13}$ at $g = 1$ and $10^{12}$ at $g = 3$, even with half the units saturated. Note that the one-step norm at $g = 0.5$ is 0.87, not 0.5: for a large random Gaussian matrix the largest singular value is about twice the spectral radius (largest eigenvalue magnitude), and singular values set the norm. The well-behaved range is narrow.

**Exploding gradients have a cheap fix.** **Gradient clipping** rescales the whole gradient when its norm exceeds a threshold $c$: $g \leftarrow g \cdot \min(1, c/\|g\|)$, keeping its direction [Pascanu et al., 2013]. Every training script here clips at 1. **Vanishing gradients have no cheap fix**: the signal is absent, not too big. The solution was architectural.

## 3. LSTM and GRU

The **long short-term memory** network [Hochreiter & Schmidhuber, 1997] adds a **cell state** $c_t$ updated by addition rather than by repeated matrix multiplication. With the forget gate added by Gers et al. (2000), and in the form PyTorch's `nn.LSTM` documentation uses:

$$
\begin{aligned}
i_t &= \sigma(W_i x_t + R_i h_{t-1} + b_i), & f_t &= \sigma(W_f x_t + R_f h_{t-1} + b_f), \\
o_t &= \sigma(W_o x_t + R_o h_{t-1} + b_o), & g_t &= \tanh(W_g x_t + R_g h_{t-1} + b_g), \\
c_t &= f_t \odot c_{t-1} + i_t \odot g_t, & h_t &= o_t \odot \tanh(c_t).
\end{aligned}
$$

$\sigma$ is the logistic sigmoid, so the **gates** (input $i_t$, forget $f_t$, output $o_t$) lie between 0 and 1 per unit, and $\odot$ is element-wise multiplication. The forget gate decides how much memory to keep, the input gate how much new content $g_t$ to write, the output gate how much to expose [Gers et al., 2000].

**Why it helps.** Along the cell path, $\partial c_t/\partial c_{t-1} = \operatorname{diag}(f_t)$ (plus indirect terms through the gates): no weight matrix, no squashing. If $f_t \approx 1$, the stored value and its gradient survive many steps. It is the same idea as a residual connection (L2-03).

**The forget-gate bias.** Over $T$ steps the cell path scales by $\prod_t f_t$. PyTorch initializes LSTM biases uniformly in $\pm 1/\sqrt{H}$ (per its documentation), so initially $f_t \approx \sigma(0) = 0.5$ and memory decays as $0.5^{20} \approx 10^{-6}$: the LSTM starts as forgetful as a plain RNN. Jozefowicz et al. (2015) found that a forget bias of 1 closed the gap between the LSTM and the GRU [Jozefowicz et al., 2015]. With bias $b$, $f \approx \sigma(b)$: $\sigma(1)^{50} \approx 1.6 \times 10^{-7}$, $\sigma(3)^{50} \approx 0.09$. Script 2 needs the model to remember the first of $T$ random tokens. At $T = 20$ and $50$, the RNN, GRU and default LSTM stay at chance after 800 steps; an LSTM with forget bias 3 solves both. In extra runs not shown, bias 1 solved $T = 20$ but not 50, and the GRU needed about 3,000 steps for $T = 20$. Initialization, not just architecture, decides what is learnable.

**The GRU** [Cho et al., 2014] uses two gates and no separate cell. In PyTorch's notation:

$$
\begin{aligned}
r_t &= \sigma(W_r x_t + R_r h_{t-1} + b_r), \qquad z_t = \sigma(W_z x_t + R_z h_{t-1} + b_z), \\
n_t &= \tanh\big(W_n x_t + b_n + r_t \odot (R_n h_{t-1} + b'_n)\big), \qquad h_t = (1 - z_t) \odot n_t + z_t \odot h_{t-1}.
\end{aligned}
$$

The **update gate** $z_t$ interpolates between the old state and the candidate $n_t$ (doing the job of forget and input gates together); the **reset gate** $r_t$ lets the candidate ignore the past. A GRU has three weight blocks to the LSTM's four. Neither wins everywhere; try both if it matters.

## 4. Language modelling and teacher forcing

A **language model** factorises $p(x_1, \dots, x_T) = \prod_t p(x_t \mid x_{<t})$, and an RNN outputs each factor as a softmax over the vocabulary. The loss is the average cross-entropy $-\frac{1}{T}\sum_t \log p(x_t \mid x_{<t})$ (L0-04). For characters it is reported in **bits per character (bpc)**, the loss in nats divided by $\ln 2$. **Perplexity** $2^{\text{bpc}}$ is the effective number of equally likely choices per step; a uniform guess over 70 characters has 6.13 bpc and perplexity 70.

**Baselines first** (L1-01). In Script 4 a count-based 3-gram model reaches 2.74 bpc on held-out text, and a 5-gram does *worse* (2.82): most 4-character contexts are rare in 130,000 training characters. A neural model shares strength across similar contexts; the small LSTM reaches 2.17 bpc (perplexity 4.5).

**Teacher forcing** means feeding the *true* previous token as input during training. It is parallel and stable. But at test time the model consumes its own outputs, so one mistake leads to states never seen in training, and errors compound. Bengio et al. (2015) described this mismatch and proposed **scheduled sampling**, gradually replacing true inputs with the model's own predictions during training [Bengio et al., 2015]. In Script 3 the no-attention model's token accuracy is 0.805 teacher-forced but 0.612 free-running.

## 5. Sequence to sequence

Translation maps a sequence to one of different length. The **encoder-decoder** (**seq2seq**) design [Cho et al., 2014; Sutskever et al., 2014] reads the source into a fixed-length vector with one RNN, then generates the target with a second RNN started from that vector, until it emits an end token. Sutskever et al. used two 4-layer LSTMs with 1,000 cells per layer (384 million parameters) and reported 34.8 BLEU on WMT'14 English to French, against 33.3 for a phrase-based system. Reversing the source sentence helped markedly, because it put the first source words near the first target words [Sutskever et al., 2014].

That trick hints at the problem: the whole source must squeeze through one vector. In Script 3 a GRU encoder-decoder reverses digit strings almost perfectly up to length 7, gets 42% exactly right at lengths 8 to 12, and almost none beyond.

## 6. Attention: where it came from

Bahdanau, Cho and Bengio (2015) called the fixed-length vector a bottleneck and let the decoder look back at every encoder state $h_1, \dots, h_S$ (the **annotations**) [Bahdanau et al., 2015]. At decoder step $t$, with previous decoder state $s_{t-1}$:

$$e_{t,j} = v^\top \tanh(W s_{t-1} + U h_j), \qquad \alpha_{t,j} = \frac{\exp(e_{t,j})}{\sum_k \exp(e_{t,k})}, \qquad c_t = \sum_j \alpha_{t,j} h_j.$$

The **attention weights** $\alpha_{t,\cdot}$ form a probability distribution over source positions: a soft, differentiable "where to look", learned without alignment labels. The **context vector** $c_t$ feeds the decoder step and its output layer. The paper's encoder was bidirectional, and its attention model held up on sentences of 50 words or more, while the plain encoder-decoder degraded sharply with length.

Script 3 reproduces this: with attention, exact-match stays at 91% or more up to length 20. For a length-10 input the most-attended position walks backwards, as reversal requires. Two caveats. The alignment is off by one at the start (the second step still looks at position 9) yet the output is right: a one-directional encoder's $h_9$ also carries earlier digits, so annotations are not tokens and attention maps are not exact explanations. And the example has a real error: in the run of three zeros the model emits two, at the step with the lowest peak weight (0.46). Content-based attention cannot tell identical tokens apart.

Drop recurrence and let a sequence attend to itself, and you have the transformer (L2-05).

## 7. Decoding

Turning $p(x_t \mid x_{<t})$ into text is a separate choice, and it matters as much as the model.

- **Greedy**: take the most likely token. Deterministic, often repetitive.
- **Beam search**: keep the $B$ partial sequences with the highest total log-probability, extend each by every token, keep the best $B$. When candidates end at different lengths, scores are usually length-normalized, since each token lowers the total. Sutskever et al. found a beam of 2 gave most of the benefit for translation.
- **Temperature** $\tau$: sample from $\operatorname{softmax}(z/\tau)$ for logits $z$; $\tau < 1$ sharpens toward greedy, $\tau > 1$ flattens.
- **Top-k**: sample among the $k$ most likely tokens.
- **Nucleus (top-p)**: sample from the smallest set of tokens with total probability at least $p$, renormalized, so the set adapts to the model's confidence [Holtzman et al., 2020].

Holtzman et al. found that likelihood-maximizing decoding such as beam search produces bland, repetitive "degenerate" text, and that truncating the unreliable tail while still sampling works better for open-ended generation [Holtzman et al., 2020]. Script 4 shows it in miniature: beam search finds the highest log-probability ($-0.48$ nats per character) and the worst text, "and she went on" forever; greedy loops with a longer period; plain sampling at $\tau = 1$ invents words; nucleus sampling is varied and much more probable. Beam search suits tasks with one right answer, such as translation.

## Hands-on

Run from the repository root, for example `.venv/bin/python code/L2-04-sequence-models/rnn_gradients.py`. Outputs are from PyTorch 2.15.0 (CPU) on Linux, on a machine busy with other jobs, so times are rough; each script ran in under 40 seconds. Last digits may vary by platform and thread count.

**The corpus.** `data/alice.txt` is Lewis Carroll's *Alice's Adventures in Wonderland* (1865), downloaded from Project Gutenberg (eBook #11, `https://www.gutenberg.org/cache/epub/11/pg11.txt`), with the Project Gutenberg header, footer, table of contents and illustration markers removed and curly quotes converted to ASCII (144,215 characters). It is public domain in the United States; Project Gutenberg's license page says that once its license and all references to it are removed, US law places no restriction on the text. Elsewhere, check local copyright rules.

### Script 1: measuring vanishing and exploding gradients

```python
"""Vanishing and exploding gradients in a vanilla RNN, measured directly.
dh_T/dh_0 is a product of T Jacobians J_t = diag(phi'(a_t)) W. We compute its norm."""
import numpy as np

rng = np.random.default_rng(0)
H = 64                                        # hidden size


def random_orthogonal(n):
    q, r = np.linalg.qr(rng.normal(size=(n, n)))
    return q * np.sign(np.diag(r))            # all singular values exactly 1


def jacobian_norms(W, U, T_list, tanh=True, x_scale=1.0, seed=1):
    """Run h_t = phi(W h_{t-1} + U x_t) and return the spectral norm of dh_T/dh_0 for each T."""
    r = np.random.default_rng(seed)
    h = np.zeros(H)
    J = np.eye(H)
    out = {}
    for t in range(1, max(T_list) + 1):
        a = W @ h + U @ (x_scale * r.normal(size=H))
        h = np.tanh(a) if tanh else a
        d = 1 - h ** 2 if tanh else np.ones(H)    # phi'(a_t); for tanh it is 1 - tanh(a)^2
        J = (d[:, None] * W) @ J                  # chain rule: J_t times everything before
        if t in T_list:
            out[t] = np.linalg.norm(J, 2)
    return out, h


# 1. The scalar picture: h_t = w h_{t-1}, so dh_T/dh_0 = w^T.
for w in [0.9, 1.0, 1.1]:
    print(f"scalar w={w}: w^10={w ** 10:.3g}, w^50={w ** 50:.3g}, w^100={w ** 100:.3g}")

# 2. Linear RNN, W = rho * orthogonal: every direction scales by exactly rho per step.
T_list = [1, 10, 25, 50, 100]
Q = random_orthogonal(H)
U = rng.normal(size=(H, H)) / np.sqrt(H)
print(f"\n||dh_T/dh_0|| (spectral norm), linear RNN")
print("rho   " + "".join(f"{'T=' + str(T):>11s}" for T in T_list))
for rho in [0.9, 1.0, 1.1]:
    norms, _ = jacobian_norms(rho * Q, U, T_list, tanh=False)
    print(f"{rho:<5} " + "".join(f"{norms[T]:11.3g}" for T in T_list))

# 3. tanh RNN with a Gaussian W of scale g/sqrt(H) (spectral radius about g).
print(f"\n||dh_T/dh_0||, tanh RNN, W ~ N(0, g^2/H)")
print("g     " + "".join(f"{'T=' + str(T):>11s}" for T in T_list) + "   mean |h_T|  saturated")
for g in [0.5, 1.0, 1.5, 3.0]:
    W = g * rng.normal(size=(H, H)) / np.sqrt(H)
    norms, h = jacobian_norms(W, U, T_list, tanh=True, x_scale=0.5)
    print(f"{g:<5} " + "".join(f"{norms[T]:11.3g}" for T in T_list)
          + f"   {np.abs(h).mean():10.2f}  {np.mean(np.abs(h) > 0.95):9.0%}")

# 4. Why clipping helps: rescale the gradient if its norm exceeds a threshold.
grad = rng.normal(size=1000) * 50
c = 1.0
clipped = grad * min(1.0, c / np.linalg.norm(grad))
cos = grad @ clipped / (np.linalg.norm(grad) * np.linalg.norm(clipped))
print(f"\nclip by norm: ||g|| {np.linalg.norm(grad):.1f} -> {np.linalg.norm(clipped):.1f}, "
      f"direction unchanged (cosine {cos:.3f})")
```

```
scalar w=0.9: w^10=0.349, w^50=0.00515, w^100=2.66e-05
scalar w=1.0: w^10=1, w^50=1, w^100=1
scalar w=1.1: w^10=2.59, w^50=117, w^100=1.38e+04

||dh_T/dh_0|| (spectral norm), linear RNN
rho           T=1       T=10       T=25       T=50      T=100
0.9           0.9      0.349     0.0718    0.00515   2.66e-05
1.0             1          1          1          1          1
1.1           1.1       2.59       10.8        117   1.38e+04

||dh_T/dh_0||, tanh RNN, W ~ N(0, g^2/H)
g             T=1       T=10       T=25       T=50      T=100   mean |h_T|  saturated
0.5         0.874     0.0012   1.37e-08    4.3e-17   1.49e-33         0.36         0%
1.0          1.66      0.296     0.0038   1.47e-06   6.24e-13         0.45         0%
1.5          2.57       6.41       5.33       4.62      0.165         0.54         3%
3.0          5.34        241   4.79e+04   3.81e+07   3.75e+12         0.81        55%

clip by norm: ||g|| 1545.0 -> 1.0, direction unchanged (cosine 1.000)
```

The linear rows are exactly $\rho^T$, as they must be for an orthogonal $W$.

### Script 2: a long-range memory task

```python
"""A long-range memory task: read a sequence, then output its FIRST token.
Everything after the first token is a distraction. Chance accuracy is 1/8."""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(4)  # fixed thread count, so runs on a shared machine are comparable
V, HID = 8, 64            # vocabulary size, hidden size


def batch(n, T, gen):
    x = torch.randint(0, V, (n, T), generator=gen)
    return x, x[:, 0].clone()            # target = the first token


class Reader(nn.Module):
    def __init__(self, cell, forget_bias=None):
        super().__init__()
        self.emb = nn.Embedding(V, 16)
        self.rnn = {"RNN": nn.RNN, "GRU": nn.GRU, "LSTM": nn.LSTM}[cell](16, HID, batch_first=True)
        self.out = nn.Linear(HID, V)
        if forget_bias is not None:   # PyTorch orders LSTM gate biases (input, forget, cell, output)
            with torch.no_grad():
                self.rnn.bias_ih_l0[HID:2 * HID] = forget_bias
                self.rnn.bias_hh_l0[HID:2 * HID] = 0.0

    def forward(self, x):
        h_all, _ = self.rnn(self.emb(x))
        return self.out(h_all[:, -1])    # classify from the last hidden state only


def run(cell, T, forget_bias=None, steps=800, seed=0):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = Reader(cell, forget_bias)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    max_norm = 0.0
    for _ in range(steps):
        x, y = batch(128, T, gen)
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        # Clip the global gradient norm at 1 (Pascanu et al., 2013); record the largest raw norm.
        max_norm = max(max_norm, nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())
        opt.step()
    x, y = batch(2000, T, torch.Generator().manual_seed(999))
    with torch.no_grad():
        acc = (model(x).argmax(1) == y).float().mean().item()
    return acc, max_norm


configs = [("RNN", "RNN", None), ("GRU", "GRU", None), ("LSTM", "LSTM", None),
           ("LSTM, forget bias 3", "LSTM", 3.0)]
print("test accuracy (largest raw gradient norm seen in training); chance = 0.125")
print(f"{'length T':>8s} " + " ".join(f"{name:>19s}" for name, _, _ in configs))
for T in [5, 20, 50]:
    t0 = time.time()
    cells = []
    for _, cell, fb in configs:
        acc, gmax = run(cell, T, fb)
        cells.append(f"{acc:8.3f} ({gmax:6.2f})")
    print(f"{T:8d} " + " ".join(f"{c:>19s}" for c in cells) + f"   {time.time() - t0:.0f}s")
```

```
test accuracy (largest raw gradient norm seen in training); chance = 0.125
length T                 RNN                 GRU                LSTM LSTM, forget bias 3
       5      1.000 (  0.88)      1.000 (  0.64)      1.000 (  1.01)      1.000 (  0.82)   4s
      20      0.135 (  0.41)      0.119 (  0.26)      0.125 (  0.17)      1.000 (  7.60)   9s
      50      0.115 (  0.42)      0.116 (  0.32)      0.123 (  0.20)      1.000 ( 30.88)   20s
```

Failing runs never see a gradient norm above about 0.4, because no useful gradient reaches the first token; the long-range successes see 7.6 and 30.9 once learning starts, and clipping keeps those steps safe. One seed per cell, so rerun with others before trusting any single entry.

### Script 3: encoder-decoder with and without attention

```python
"""Encoder-decoder GRUs on a toy translation task: output the input digits reversed.
One decoder sees only the encoder's final state; the other uses additive (Bahdanau) attention.
Both are trained with teacher forcing and evaluated with greedy decoding."""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(4)  # fixed thread count, so runs on a shared machine are comparable
PAD, BOS, EOS = 10, 11, 12  # tokens 0-9 are digits
V, E, H = 13, 32, 128
MIN_LEN, MAX_LEN = 3, 20


def make_batch(n, gen, lo=MIN_LEN, hi=MAX_LEN):
    lens = torch.randint(lo, hi + 1, (n,), generator=gen)
    src = torch.full((n, hi), PAD)
    tgt = torch.full((n, hi + 1), PAD)                 # reversed digits followed by EOS
    for i, L in enumerate(lens.tolist()):
        s = torch.randint(0, 10, (L,), generator=gen)
        src[i, :L] = s
        tgt[i, :L] = s.flip(0)
        tgt[i, L] = EOS
    return src, tgt, lens


class Seq2Seq(nn.Module):
    def __init__(self, attention):
        super().__init__()
        self.attention = attention
        self.emb = nn.Embedding(V, E)
        self.encoder = nn.GRU(E, H, batch_first=True)
        self.cell = nn.GRUCell(E + (H if attention else 0), H)
        self.W, self.U, self.v = nn.Linear(H, H, bias=False), nn.Linear(H, H), nn.Linear(H, 1, bias=False)
        self.out = nn.Linear(H + (H if attention else 0), V)

    def encode(self, src, lens):
        packed = nn.utils.rnn.pack_padded_sequence(self.emb(src), lens, batch_first=True,
                                                   enforce_sorted=False)
        hs, h_last = self.encoder(packed)
        hs, _ = nn.utils.rnn.pad_packed_sequence(hs, batch_first=True, total_length=src.shape[1])
        return hs, h_last[0], src != PAD          # annotations, final state, mask

    def step(self, y_prev, s, hs, keys, mask):
        """One decoder step. Additive attention: e_j = v^T tanh(W s + U h_j)."""
        x = self.emb(y_prev)
        alpha = None
        if self.attention:
            e = self.v(torch.tanh(self.W(s)[:, None] + keys)).squeeze(-1)
            alpha = torch.softmax(e.masked_fill(~mask, -1e9), dim=1)   # weights over source
            ctx = (alpha[:, :, None] * hs).sum(1)
            s = self.cell(torch.cat([x, ctx], 1), s)
            return self.out(torch.cat([s, ctx], 1)), s, alpha
        s = self.cell(x, s)
        return self.out(s), s, alpha

    def forward(self, src, lens, tgt):
        """Teacher forcing: the decoder input at step t is the TRUE token t-1."""
        hs, s, mask = self.encode(src, lens)
        keys = self.U(hs)
        y_prev = torch.full((src.shape[0],), BOS)
        logits = []
        for t in range(tgt.shape[1]):
            lo, s, _ = self.step(y_prev, s, hs, keys, mask)
            logits.append(lo)
            y_prev = tgt[:, t]
        return torch.stack(logits, 1)

    @torch.no_grad()
    def greedy(self, src, lens, max_steps):
        """Free running: feed back the model's own previous prediction."""
        hs, s, mask = self.encode(src, lens)
        keys = self.U(hs)
        y_prev = torch.full((src.shape[0],), BOS)
        preds, alphas = [], []
        for _ in range(max_steps):
            lo, s, a = self.step(y_prev, s, hs, keys, mask)
            y_prev = lo.argmax(1)
            preds.append(y_prev)
            alphas.append(a)
        return torch.stack(preds, 1), alphas


def train(attention, steps=600, seed=0):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = Seq2Seq(attention)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    for _ in range(steps):
        src, tgt, lens = make_batch(128, gen)
        logits = model(src, lens, tgt)
        loss = F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1), ignore_index=PAD)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model


def evaluate(model, gen):
    src, tgt, lens = make_batch(3000, gen)
    with torch.no_grad():
        tf_pred = model(src, lens, tgt).argmax(-1)          # teacher-forced predictions
    pred, _ = model.greedy(src, lens, MAX_LEN + 1)
    valid = tgt != PAD
    tf_tok = ((tf_pred == tgt) & valid).sum().item() / valid.sum().item()
    fr_tok = ((pred == tgt) & valid).sum().item() / valid.sum().item()
    exact = ((pred == tgt) | ~valid).all(1)                 # every target token right, incl. EOS
    buckets = {f"{a}-{b}": exact[(lens >= a) & (lens <= b)].float().mean().item()
               for a, b in [(3, 7), (8, 12), (13, 16), (17, 20)]}
    return tf_tok, fr_tok, buckets


models = {}
for attention in [False, True]:
    t0 = time.time()
    name = "with attention" if attention else "no attention"
    models[name] = train(attention)
    tf_tok, fr_tok, buckets = evaluate(models[name], torch.Generator().manual_seed(7))
    print(f"{name}: trained in {time.time() - t0:.0f}s")
    print(f"  token accuracy: teacher-forced {tf_tok:.3f}, free-running {fr_tok:.3f}")
    print("  exact-match by source length: " + ", ".join(f"{k}: {v:.2f}" for k, v in buckets.items()))

# Where does the attention model look? For a length-10 input, print the argmax source position
# and its weight at each output step. For reversal it should walk backwards: 9, 8, ..., 0.
src, tgt, lens = make_batch(1, torch.Generator().manual_seed(3), lo=10, hi=10)
pred, alphas = models["with attention"].greedy(src, lens, 11)
print("\nsource :", src[0, :10].tolist())
print("output :", pred[0, :10].tolist())
print("argmax :", [a[0].argmax().item() for a in alphas[:10]])
print("weight :", [round(a[0].max().item(), 2) for a in alphas[:10]])
```

```
no attention: trained in 12s
  token accuracy: teacher-forced 0.805, free-running 0.612
  exact-match by source length: 3-7: 0.97, 8-12: 0.42, 13-16: 0.02, 17-20: 0.00
with attention: trained in 24s
  token accuracy: teacher-forced 0.996, free-running 0.979
  exact-match by source length: 3-7: 0.99, 8-12: 0.98, 13-16: 0.95, 17-20: 0.91

source : [8, 7, 7, 0, 0, 0, 5, 1, 9, 4]
output : [4, 9, 1, 5, 0, 0, 7, 7, 8, 12]
argmax : [9, 9, 8, 7, 6, 5, 4, 2, 1, 0]
weight : [0.99, 0.99, 0.98, 0.97, 0.97, 0.79, 0.46, 0.66, 0.65, 0.96]
```

The attention model is about twice as slow per step: it scores every source position at every output step.

### Script 4: a character-level language model and decoding

```python
"""A character-level LSTM language model on "Alice's Adventures in Wonderland" (Lewis Carroll,
1865; public domain text, see data/alice.txt), compared with count baselines, then decoded
with greedy search, beam search, temperature sampling and nucleus (top-p) sampling."""
import math
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(4)  # fixed thread count, so runs on a shared machine are comparable
torch.manual_seed(0)
text = (Path(__file__).parent / "data" / "alice.txt").read_text()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text])
V = len(chars)
split = int(0.9 * len(data))                  # first 90% train, last 10% validation
train_data, val_data = data[:split], data[split:]
print(f"{len(text):,} characters, vocabulary {V}, train {split:,}, validation {len(val_data):,}")


# 1. Count baselines, in bits per character (bpc). Perplexity = 2^bpc.
def ngram_bpc(order, alpha=0.1):
    """Add-alpha smoothed n-gram model: P(c | previous order-1 chars)."""
    counts, ctx_counts = {}, {}
    tr = train_data.tolist()
    for i in range(order - 1, len(tr)):
        ctx, c = tuple(tr[i - order + 1:i]), tr[i]
        counts[ctx, c] = counts.get((ctx, c), 0) + 1
        ctx_counts[ctx] = ctx_counts.get(ctx, 0) + 1
    va, total = val_data.tolist(), 0.0
    for i in range(order - 1, len(va)):
        ctx, c = tuple(va[i - order + 1:i]), va[i]
        p = (counts.get((ctx, c), 0) + alpha) / (ctx_counts.get(ctx, 0) + alpha * V)
        total -= math.log2(p)
    return total / (len(va) - order + 1)


print(f"uniform over {V} chars: {math.log2(V):.3f} bpc")
for order in [1, 3, 5]:
    bpc = ngram_bpc(order)
    print(f"{order}-gram (add-0.1): {bpc:.3f} bpc, perplexity {2 ** bpc:.2f}")


# 2. The LSTM language model.
class CharLSTM(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.emb = nn.Embedding(V, 64)
        self.lstm = nn.LSTM(64, hidden, batch_first=True)
        self.head = nn.Linear(hidden, V)

    def forward(self, x, state=None):
        h, state = self.lstm(self.emb(x), state)
        return self.head(h), state


def windows(src, n, L, gen):   # n random windows; target = input shifted by one character
    i = torch.randint(0, len(src) - L - 1, (n,), generator=gen)
    x = torch.stack([src[j:j + L] for j in i.tolist()])
    y = torch.stack([src[j + 1:j + L + 1] for j in i.tolist()])
    return x, y


@torch.no_grad()
def val_bpc(model, L=200):     # validation text in chunks of 200 (state reset per chunk)
    n = (len(val_data) - 1) // L
    x = val_data[:n * L].view(n, L)
    y = val_data[1:n * L + 1].view(n, L)
    logits, _ = model(x)
    return F.cross_entropy(logits.reshape(-1, V), y.reshape(-1)).item() / math.log(2)


model = CharLSTM()
opt = torch.optim.Adam(model.parameters(), lr=3e-3)
gen = torch.Generator().manual_seed(0)
t0 = time.time()
for step in range(1, 2001):
    x, y = windows(train_data, 32, 64, gen)        # truncated BPTT over 64 characters
    logits, _ = model(x)                           # teacher forcing: inputs are the true text
    loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if step % 400 == 0:
        print(f"step {step:4d}: train {loss.item() / math.log(2):.3f} bpc, "
              f"validation {val_bpc(model):.3f} bpc  ({time.time() - t0:.0f}s)")
model.eval()


# 3. Decoding. Every method continues the same prompt for N characters.
def encode(s):
    return torch.tensor([[stoi[c] for c in s]])


@torch.no_grad()
def sample(prompt, n, temperature=1.0, top_p=None, greedy=False, seed=0):
    g = torch.Generator().manual_seed(seed)
    logits, state = model(encode(prompt))
    out, logp = [], 0.0
    for _ in range(n):
        lp = F.log_softmax(logits[0, -1], -1)
        if greedy:
            c = lp.argmax().item()
        else:
            probs = F.softmax(logits[0, -1] / temperature, -1)
            if top_p is not None:                 # keep the smallest set with mass >= top_p
                p_sorted, idx = probs.sort(descending=True)
                keep = p_sorted.cumsum(0) - p_sorted < top_p
                probs = torch.zeros_like(probs).scatter(0, idx[keep], p_sorted[keep])
                probs /= probs.sum()
            c = torch.multinomial(probs, 1, generator=g).item()
        logp += lp[c].item()                       # score under the UNMODIFIED model
        out.append(c)
        logits, state = model(torch.tensor([[c]]), state)
    return "".join(chars[c] for c in out), logp / n


@torch.no_grad()
def beam_search(prompt, n, width=5):
    logits, (h, c) = model(encode(prompt))
    beams = [([], 0.0)]                           # (tokens, total log-prob)
    lp = F.log_softmax(logits[:, -1], -1)         # one row per beam
    for _ in range(n):
        cand = [(beams[b][1] + lp[b, t].item(), b, t)
                for b in range(len(beams)) for t in lp[b].topk(width).indices.tolist()]
        cand.sort(reverse=True)
        cand = cand[:width]
        idx = torch.tensor([b for _, b, _ in cand])
        beams = [(beams[b][0] + [t], s) for s, b, t in cand]
        logits, (h, c) = model(torch.tensor([[t] for _, _, t in cand]), (h[:, idx], c[:, idx]))
        lp = F.log_softmax(logits[:, -1], -1)
    tokens, score = beams[0]
    return "".join(chars[t] for t in tokens), score / n


def distinct4(s):
    grams = [s[i:i + 4] for i in range(len(s) - 3)]
    return len(set(grams)) / len(grams)


prompt, N = "Alice was beginning to ", 120
print(f'\nprompt: "{prompt}"  (avg log-prob per char under the model; distinct 4-grams)')
results = {
    "greedy": sample(prompt, N, greedy=True),
    "beam, width 5": beam_search(prompt, N),
    "sample, T=0.5": sample(prompt, N, temperature=0.5),
    "sample, T=1.0": sample(prompt, N, temperature=1.0),
    "top-p 0.9, T=1.0": sample(prompt, N, top_p=0.9),
}
for name, (s, lp) in results.items():
    print(f"\n[{name}] logp/char {lp:.2f}, distinct 4-grams {distinct4(s):.2f}")
    print(s.replace("\n", " "))
```

```
144,215 characters, vocabulary 70, train 129,793, validation 14,422
uniform over 70 chars: 6.129 bpc
1-gram (add-0.1): 4.560 bpc, perplexity 23.58
3-gram (add-0.1): 2.737 bpc, perplexity 6.67
5-gram (add-0.1): 2.820 bpc, perplexity 7.06
step  400: train 2.330 bpc, validation 2.466 bpc  (3s)
step  800: train 2.062 bpc, validation 2.271 bpc  (5s)
step 1200: train 1.880 bpc, validation 2.193 bpc  (8s)
step 1600: train 1.899 bpc, validation 2.174 bpc  (10s)
step 2000: train 1.825 bpc, validation 2.166 bpc  (13s)

prompt: "Alice was beginning to "  (avg log-prob per char under the model; distinct 4-grams)

[greedy] logp/char -0.64, distinct 4-grams 0.47
herself to her side of the thing with a little finished to her side of the thing with a little finished to her side of t

[beam, width 5] logp/char -0.48, distinct 4-grams 0.21
herself, and she went on, and she went on, and she went on, and she went on, and she went on, and she went on, and she w

[sample, T=0.5] logp/char -0.94, distinct 4-grams 0.94
see it was not itself, "I am lablay then she see, and the dise. She had been into a little hattered the ground at the tr

[sample, T=1.0] logp/char -1.22, distinct 4-grams 0.99
speen to herself, and, looking buny you jave, for stupiculdial. She!" thought Alice, I don't talking, of bright):  "Well

[top-p 0.9, T=1.0] logp/char -0.88, distinct 4-grams 0.93
speen to herself, and she looked at the jury, for stupiculdial. She had the Queen, and the Mock Turtle, this far my," sa
```

Validation bpc is still falling slowly at 2,000 steps. Log-probabilities are nats per character under the unmodified model, so they compare across methods. The $\tau = 1$ and top-p samples share their first words because they use the same seed and those choices fell inside the nucleus.

## Common mistakes

- **No gradient clipping.** One exploding batch turns the loss into `nan`. Clip, and log the pre-clip norm.
- **Comparing perplexities across vocabularies or tokenizers.** Per-character and per-word numbers are not comparable. Use bpc on the same text.
- **No baseline.** Report an n-gram number next to your model.
- **Leaking the future.** Targets are inputs shifted by one; shift the wrong way and the loss drops near zero.
- **Padding in the loss or attention.** Use `ignore_index` and masks (Script 3 does both).
- **Trusting teacher-forced metrics or likelihood alone.** Evaluate free-running output and read samples.
- **Random splits of one text.** Overlapping windows leak; split by position, as Script 4 does.

## Exercises

1. [warm-up] With input size 16 and hidden size 64, how many parameters do `nn.RNN`, `nn.GRU` and `nn.LSTM` have? Check in code.
2. [warm-up] A model scores 2.17 bpc. What is its perplexity, and its geometric-mean probability on the correct character?
3. [warm-up] Do two steps of beam search with $B = 2$ by hand on a made-up 3-token distribution.
4. [core] In Script 1, find the scale $g$ at which $\|\partial h_{50}/\partial h_0\|$ is closest to 1. How does `x_scale` change it?
5. [core] In Script 2, try forget biases 0, 1, 2, 3 and 5 at $T = 50$ and record when training accuracy first exceeds 90%.
6. [core] Make Script 3's encoder bidirectional, as in the paper. Does the off-by-one alignment go away?
7. [core] Add top-k sampling to Script 4 and compare $k = 5$ and $k = 20$ with top-p 0.9 over 5 seeds.
8. [core] In Script 4, use hidden size 256, windows of 100, batch 64, 1,200 steps, printing validation bpc every 300. Does it overfit?
9. [stretch] Implement scheduled sampling for Script 3's no-attention model and measure the teacher-forced versus free-running gap.

### Hints and answers

1. Per layer, $G(HI + H^2 + 2H)$ with $G = 1, 3, 4$ (PyTorch keeps two bias vectors): 5,248, 15,744 and 20,992.
2. $2^{2.17} \approx 4.5$; $2^{-2.17} \approx 0.22$.
3. Expand each beam by all 3 tokens, score the 6 candidates by total log-probability, keep the top 2.
4. Between 1 and 1.5 with the default inputs; stronger inputs saturate more units and push it up.
5. Expect bias 0 to fail and larger biases to learn sooner with diminishing returns; very large biases make the cell slow to overwrite.
6. Use `bidirectional=True`, annotations of size $2H$, and project the final states for the decoder. Alignments often become cleaner.
7. Small $k$ acts like low temperature; large $k$ admits the tail on confident steps, where top-p would not.
8. In a run during writing, validation bpc reached 2.13 at step 600 and rose to 2.25 at 1,200 while training bpc fell. Use dropout, early stopping or more data.
9. Feed the model's own argmax instead of the true token with probability $\epsilon$, raising $\epsilon$ during training. The gap should shrink.

## Check your understanding

1. **Why do vanilla RNN gradients vanish or explode?** The gradient across $T$ steps is a product of $T$ Jacobians $\operatorname{diag}(\tanh')\,W$, which shrinks or grows exponentially depending on the singular values of $W$, with $\tanh' \le 1$ adding shrinkage. Clipping handles explosion; vanishing needs a different architecture.

2. **What lets an LSTM remember, and what can still stop it?** The additive cell update $c_t = f_t \odot c_{t-1} + i_t \odot g_t$, whose path through time has Jacobian $\operatorname{diag}(f_t)$ and no weight matrix. A forget gate near 0.5 at initialization still erases memory exponentially, hence the forget bias (Script 2).

3. **What is teacher forcing and what problem does it create?** Training on the true previous token. At test time the model consumes its own predictions, can drift into unseen states, and errors compound; Script 3's teacher-forced versus free-running gap measures this.

4. **Why was attention introduced?** A plain encoder-decoder squeezes the source into one vector, and quality falls with length. Attention gives each decoder step a learned weighted average of all encoder states. In Script 3, exact-match at lengths 17 to 20 goes from 0% to 91%.

5. **Beam search gives the most likely text, but it reads badly. Why, and what instead?** Language models give high probability to repetitive, generic continuations, so maximizing likelihood finds loops. For open-ended text, use truncated sampling such as nucleus sampling; for one-right-answer tasks, beam search with length normalization.

## Further reading

- **Andrej Karpathy, "The Unreasonable Effectiveness of Recurrent Neural Networks"** (2015, blog post). https://karpathy.github.io/2015/05/21/rnn-effectiveness/
  *Read for:* character-level RNN language models, samples, a minimal NumPy implementation, and temperature. *Level:* intro.
- **Christopher Olah, "Understanding LSTM Networks"** (2015, blog post). https://colah.github.io/posts/2015-08-Understanding-LSTMs/
  *Read for:* clear step-by-step diagrams of LSTM gates and variants. *Level:* intro.
- **Yoshua Bengio, Patrice Simard, Paolo Frasconi, "Learning Long-Term Dependencies with Gradient Descent is Difficult"** (1994, IEEE Transactions on Neural Networks 5:157ff). https://doi.org/10.1109/72.279181
  *Read for:* the original analysis of why long-range credit assignment is hard. May need a subscription. *Level:* advanced.
- **Sepp Hochreiter, Jürgen Schmidhuber, "Long Short-Term Memory"** (1997, Neural Computation 9(8):1735-1780). https://doi.org/10.1162/neco.1997.9.8.1735
  *Read for:* the original LSTM and its long-time-lag experiments. May need a subscription. *Level:* advanced.
- **Felix A. Gers, Jürgen Schmidhuber, Fred Cummins, "Learning to Forget: Continual Prediction with LSTM"** (2000, Neural Computation 12:2451-2471). https://doi.org/10.1162/089976600300015015
  *Read for:* why the forget gate was added. May need a subscription. *Level:* advanced.
- **Razvan Pascanu, Tomas Mikolov, Yoshua Bengio, "On the difficulty of training Recurrent Neural Networks"** (2013, ICML, PMLR 28(3):1310-1318; arXiv:1211.5063). https://arxiv.org/abs/1211.5063
  *Read for:* the geometry of exploding and vanishing gradients, and norm clipping. *Level:* intermediate.
- **Kyunghyun Cho et al., "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation"** (2014, EMNLP; arXiv:1406.1078). https://arxiv.org/abs/1406.1078
  *Read for:* the encoder-decoder and the gated unit now called the GRU. *Level:* intermediate.
- **Ilya Sutskever, Oriol Vinyals, Quoc V. Le, "Sequence to Sequence Learning with Neural Networks"** (2014, NeurIPS 27; arXiv:1409.3215). https://arxiv.org/abs/1409.3215
  *Read for:* deep LSTM seq2seq, source reversal, and beam search in practice. *Level:* intermediate.
- **Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio, "Neural Machine Translation by Jointly Learning to Align and Translate"** (2015, ICLR; arXiv:1409.0473). https://arxiv.org/abs/1409.0473
  *Read for:* the origin of attention, with alignment plots. Cited as [Bahdanau et al., 2015]. *Level:* intermediate.
- **Rafal Jozefowicz, Wojciech Zaremba, Ilya Sutskever, "An Empirical Exploration of Recurrent Network Architectures"** (2015, ICML, PMLR 37:2342-2350). https://proceedings.mlr.press/v37/jozefowicz15.html
  *Read for:* a large LSTM versus GRU comparison and the forget-bias finding. *Level:* intermediate.
- **Samy Bengio, Oriol Vinyals, Navdeep Jaitly, Noam Shazeer, "Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks"** (2015, NeurIPS 28; arXiv:1506.03099). https://arxiv.org/abs/1506.03099
  *Read for:* the train/test mismatch of teacher forcing and a curriculum fix. *Level:* intermediate.
- **Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, Yejin Choi, "The Curious Case of Neural Text Degeneration"** (2020, ICLR; arXiv:1904.09751). https://arxiv.org/abs/1904.09751
  *Read for:* why likelihood-maximizing decoding degenerates, and nucleus sampling. *Level:* intermediate.
