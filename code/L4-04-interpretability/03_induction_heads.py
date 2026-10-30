"""Watch induction heads form in a tiny attention-only transformer, then test them.

Data: a random segment of random length, repeated. The first copy cannot be
predicted; later copies can, by "find where the current token appeared before
and copy the token that followed it", which is what an induction head does.
"""
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

V, T = 64, 40            # vocabulary size, sequence length
LMIN, LMAX = 8, 20       # segment length varies, so position alone cannot solve the task


def make_batch(n, gen):
    """A random segment of length L, tiled to fill T positions (2 to 5 copies).
    The first copy is unpredictable; after it, every next token is predictable
    by "find the earlier occurrence of the current token, copy its successor"."""
    seg = torch.randint(V, (n, T), generator=gen)
    L = torch.randint(LMIN, LMAX + 1, (n,), generator=gen)
    ar = torch.arange(T)
    x = torch.gather(seg, 1, ar[None, :] % L[:, None])
    pred = ar[None, :] >= (L[:, None] - 1)   # positions whose next token repeats
    return x, L, pred


class Attention(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h, self.dh = h, d // h
        self.qkv, self.out = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.head_mask = None    # set to a (h,) tensor of 0/1 to ablate heads
        self.pattern = None      # last attention pattern, (B, h, T, T)

    def forward(self, x):
        B, Tn, d = x.shape
        q, k, v = self.qkv(x).view(B, Tn, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(self.dh)
        causal = torch.ones(Tn, Tn, dtype=torch.bool).tril()
        att = att.masked_fill(~causal, float("-inf")).softmax(-1)
        self.pattern = att.detach()
        y = att @ v                                    # (B, h, T, dh)
        if self.head_mask is not None:
            y = y * self.head_mask.view(1, -1, 1, 1)   # zero-ablate chosen heads
        return self.out(y.transpose(1, 2).reshape(B, Tn, d))


class AttnOnly(nn.Module):
    """Embedding -> n_layer attention layers on the residual stream -> unembed. No MLPs."""

    def __init__(self, n_layer, d=64, h=4):
        super().__init__()
        self.emb, self.pos = nn.Embedding(V, d), nn.Embedding(T, d)
        self.layers = nn.ModuleList(Attention(d, h) for _ in range(n_layer))
        self.lns = nn.ModuleList(nn.LayerNorm(d) for _ in range(n_layer))
        self.ln_f, self.unemb = nn.LayerNorm(d), nn.Linear(d, V)

    def forward(self, x):
        r = self.emb(x) + self.pos(torch.arange(x.shape[1]))
        for ln, att in zip(self.lns, self.layers):
            r = r + att(ln(r))                 # each layer adds to the residual stream
        return self.unemb(self.ln_f(r))


def losses(model, x, pred):
    """Mean next-token loss on induction-predictable positions and on all others."""
    logits = model(x)[:, :-1]
    nll = F.cross_entropy(logits.reshape(-1, V), x[:, 1:].reshape(-1), reduction="none")
    nll = nll.view(x.shape[0], -1)
    m = pred[:, :-1]
    return nll[m].mean().item(), nll[~m].mean().item()


def head_scores(model, x, pred):
    """Per head, two attention scores averaged over query positions.
    induction: attention mass on keys j whose PREVIOUS token equals the query's
               current token (the successors of earlier occurrences), measured
               at positions where the next token is a repeat.
    prev-token: attention mass on the immediately preceding position."""
    model(x)
    Tn = x.shape[1]
    match = torch.zeros(x.shape[0], Tn, Tn, dtype=torch.bool)
    match[:, :, 1:] = x[:, :, None] == x[:, None, :-1]   # x[j-1] == x[i]
    match &= torch.ones(Tn, Tn, dtype=torch.bool).tril()  # keys j <= i only
    out = []
    for att in model.layers:
        P = att.pattern                                   # (B, h, T, T)
        ind = (P * match[:, None]).sum(-1)                # (B, h, T)
        ind = ind.permute(0, 2, 1)[pred].mean(0)          # (h,)
        q = torch.arange(1, Tn)
        prev = P[:, :, q, q - 1].mean((0, 2))             # (h,)
        out.append((ind, prev))
    return out


def train(n_layer, steps, seed, log_every=None):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = AttnOnly(n_layer)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    xe, Le, pe = make_batch(256, torch.Generator().manual_seed(999))  # fixed eval batch
    history = []
    for step in range(steps + 1):
        if log_every and step % log_every == 0:
            with torch.no_grad():
                rep, other = losses(model, xe, pe)
                ind = max(s[0].max().item() for s in head_scores(model, xe, pe))
            history.append((step, rep, other, ind))
        x, _, _ = make_batch(64, gen)
        logits = model(x)[:, :-1]
        loss = F.cross_entropy(logits.reshape(-1, V), x[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model, history, (xe, Le, pe)


if __name__ == "__main__":
    torch.set_num_threads(4)
    t0 = time.time()
    print(f"chance loss = ln({V}) = {math.log(V):.3f} nats\n")

    model, hist, (xe, Le, pe) = train(n_layer=2, steps=1000, seed=0, log_every=50)
    print("2-layer attention-only model during training")
    print(" step  loss(repeat)  loss(other)  best induction score")
    for step, rep, other, ind in hist:
        print(f"{step:5d}  {rep:12.3f}  {other:11.3f}  {ind:20.3f}")

    with torch.no_grad():
        scores = head_scores(model, xe, pe)
        print("\nPer-head scores after training (4 heads per layer)")
        for li, (ind, prev) in enumerate(scores):
            print(f"layer {li}: induction  " + " ".join(f"{v:.2f}" for v in ind))
            print(f"layer {li}: prev-token " + " ".join(f"{v:.2f}" for v in prev))
        prev_heads = scores[0][1].argsort(descending=True).tolist()

        def ablate(layer, heads):
            mask = torch.ones(4)
            mask[heads] = 0
            model.layers[layer].head_mask = mask
            r = losses(model, xe, pe)
            model.layers[layer].head_mask = None
            return r

        # Causal test: zero some heads' outputs and see what breaks.
        print("\nAblations: loss on repeated part / elsewhere")
        tests = [("nothing", 0, []),
                 ("layer 1, best induction head", 1, [int(scores[1][0].argmax())]),
                 ("layer 1, all four heads", 1, [0, 1, 2, 3]),
                 (f"layer 0, best prev-token head ({prev_heads[0]})", 0, prev_heads[:1]),
                 (f"layer 0, second prev-token head ({prev_heads[1]})", 0, prev_heads[1:2]),
                 ("layer 0, the other two heads", 0, prev_heads[2:])]
        for name, layer, heads in tests:
            r = ablate(layer, heads)
            print(f"  {name:38s} {r[0]:.3f} / {r[1]:.3f}")
        # A model that only knew "the next token is one of the segment's L
        # tokens" and spread its bet evenly would score mean(ln L).
        Lq = Le[:, None].expand_as(pe)[pe].float()
        print(f"  reference: uniform over the segment    {Lq.log().mean():.3f}")

    # Control: one attention layer cannot compose a previous-token head with a
    # matching head, so it should not learn induction (Elhage et al., 2021).
    m1, h1, _ = train(n_layer=1, steps=1000, seed=0, log_every=1000)
    print(f"\n1-layer control after 1000 steps: loss(repeat) {h1[-1][1]:.3f}, "
          f"loss(other) {h1[-1][2]:.3f}, best induction score {h1[-1][3]:.3f}")
    print(f"\ntotal time {time.time() - t0:.1f} s")
