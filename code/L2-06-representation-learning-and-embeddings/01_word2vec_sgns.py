"""Skip-gram with negative sampling (word2vec) from scratch, on a corpus
whose structure we know, compared with SVD of the PPMI matrix.

The corpus is generated from templates, so words in the same category
(animals, foods, ...) occur in the same contexts and nowhere else. The
distributional hypothesis predicts that their vectors should cluster.
Word frequencies within each category follow a Zipf-like law, so we can
also see what happens to rare words.
"""
import numpy as np
import torch
import torch.nn.functional as F

rng = np.random.default_rng(0)
torch.manual_seed(0)

CATS = {
    "animal": "cat dog horse cow sheep goat duck fish",
    "food": "bread apple cheese rice soup cake egg fish",   # "fish" is in both
    "vehicle": "car bus train bike boat truck tram van",
    "place": "city park school river market farm town beach",
    "person": "man woman child doctor teacher farmer nurse pilot",
}
CATS = {c: w.split() for c, w in CATS.items()}
TEMPLATES = [
    "the animal eats the food", "the person eats the food",   # shared contexts
    "the person rides the vehicle to the place", "the person drives the vehicle",
    "the animal sleeps in the place", "the person buys food at the place",
    "the person sees the animal", "a vehicle stops near the place",
    "the person cooks food", "the animal runs to the place",
]
zipf = 1.0 / np.arange(1, 9) ** 1.5           # rank 1 is ~23x more common than rank 8
zipf /= zipf.sum()


def sentence():
    words = []
    for slot in rng.choice(TEMPLATES).split():
        words.append(str(rng.choice(CATS[slot], p=zipf)) if slot in CATS else slot)
    return words


corpus = [w for _ in range(3000) for w in sentence() + ["."]]
vocab = sorted(set(corpus))
idx = {w: i for i, w in enumerate(vocab)}
ids = np.array([idx[w] for w in corpus])
V = len(vocab)
counts = np.bincount(ids, minlength=V)
print(f"corpus {len(ids):,} tokens, vocabulary {V}")

# ---- (center, context) pairs within a window of 2. --------------------------
W = 2
pairs = np.concatenate([np.stack([ids[W:-W], ids[W + o:len(ids) - W + o]], 1)
                        for o in (-2, -1, 1, 2)])
print(f"training pairs {len(pairs):,}")

# ---- SGNS: maximize log sig(w.c) + sum_k log sig(-w.c_k), c_k ~ unigram^0.75 --
DIM, K, EPOCHS, BS = 16, 5, 3, 1024
w_emb = torch.nn.Embedding(V, DIM)
c_emb = torch.nn.Embedding(V, DIM)
torch.nn.init.normal_(w_emb.weight, std=0.1)
torch.nn.init.zeros_(c_emb.weight)
noise = torch.tensor(counts ** 0.75, dtype=torch.float)
opt = torch.optim.Adam(list(w_emb.parameters()) + list(c_emb.parameters()), lr=0.01)
P = torch.from_numpy(pairs)
for epoch in range(EPOCHS):
    perm = torch.randperm(len(P))
    total = 0.0
    for s in range(0, len(P), BS):
        b = P[perm[s:s + BS]]
        w, c = w_emb(b[:, 0]), c_emb(b[:, 1])
        neg = c_emb(torch.multinomial(noise, len(b) * K, replacement=True)).view(len(b), K, DIM)
        pos_score = (w * c).sum(-1)
        neg_score = torch.einsum("bd,bkd->bk", w, neg)
        loss = -(F.logsigmoid(pos_score) + F.logsigmoid(-neg_score).sum(-1)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item() * len(b)
    print(f"epoch {epoch}: loss {total / len(P):.3f}")

E_sgns = w_emb.weight.detach().numpy()

# ---- Count-based alternative: PPMI matrix + truncated SVD. -------------------
C = np.zeros((V, V))
np.add.at(C, (pairs[:, 0], pairs[:, 1]), 1)
pmi = np.log(C * C.sum() / np.outer(C.sum(1), C.sum(0)) + 1e-12)
ppmi = np.maximum(pmi, 0)
U, S, _ = np.linalg.svd(ppmi)
E_svd = U[:, :DIM] * np.sqrt(S[:DIM])


def neighbors(E, word, n=4):
    En = E / np.linalg.norm(E, axis=1, keepdims=True)
    sims = En @ En[idx[word]]
    order = [j for j in np.argsort(-sims) if j != idx[word]][:n]
    return " ".join(f"{vocab[j]}:{sims[j]:.2f}" for j in order)


def centroid_report(E, name):
    """Cosine of words to their own category centroid vs the best other one.
    Centroids use the four most common words (ranks 1-4) of each category."""
    En = E / np.linalg.norm(E, axis=1, keepdims=True)
    cents = {c: En[[idx[w] for w in ws[:4]]].mean(0) for c, ws in CATS.items()}
    cents = {c: v / np.linalg.norm(v) for c, v in cents.items()}

    def margin(word, cat):
        own = En[idx[word]] @ cents[cat]
        other = max(En[idx[word]] @ v for c, v in cents.items() if c != cat)
        return own, other

    for label, ranks in (("ranks 1-2", (0, 1)), ("ranks 7-8", (6, 7))):
        words = [(ws[r], c) for c, ws in CATS.items() for r in ranks if ws[r] != "fish"]
        m = np.array([margin(w, c) for w, c in words])
        print(f"  {name:<9} {label}: own centroid {m[:, 0].mean():.2f}, "
              f"best other {m[:, 1].mean():.2f}")
    fa, ff = En[idx["fish"]] @ cents["animal"], En[idx["fish"]] @ cents["food"]
    print(f"  {name:<9} 'fish': animal centroid {fa:.2f}, food centroid {ff:.2f}")


print("\nfrequency and polysemy (cosine to category centroids)")
centroid_report(E_sgns, "SGNS")
centroid_report(E_svd, "PPMI+SVD")
print("  counts:", {w: int(counts[idx[w]]) for w in ("cat", "dog", "duck", "fish", "egg")})
fish_ctx = [vocab[j] for j in np.argsort(-C[idx["fish"]])[:6]]
print("  most common contexts of 'fish':", fish_ctx)

# ---- Levy and Goldberg (2014): SGNS implicitly fits w.c = PMI(w, c) - log K. --
obs = C > 20                                   # pairs seen often enough to estimate
wc = (E_sgns @ c_emb.weight.detach().numpy().T)[obs]
target = pmi[obs] - np.log(K)
print(f"\nobserved pairs: corr(w.c, PMI - log k) = {np.corrcoef(wc, target)[0, 1]:.3f}, "
      f"mean w.c {wc.mean():.2f} vs mean PMI - log k {target.mean():.2f}")
