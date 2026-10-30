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
