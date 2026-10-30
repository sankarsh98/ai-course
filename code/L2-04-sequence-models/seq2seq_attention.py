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
