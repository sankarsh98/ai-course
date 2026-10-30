# Representation Learning and Embeddings

> **Level 2 · Module 6** · Prerequisites: L0-02, L1-04, L2-02, L2-05 · Time: 7 to 9 hours · Code: `code/L2-06-representation-learning-and-embeddings/`

## What you will be able to do

- Explain what an embedding is, and why "good embedding" only means something relative to a use.
- Train word2vec (skip-gram with negative sampling) from scratch and connect it to factorizing a PMI matrix with the SVD.
- Train autoencoders and a contrastive encoder without labels, and evaluate representations with probes rather than by their training loss.
- Build exact and approximate (IVF) nearest-neighbor search and measure the recall you give up for speed.
- Recognize and measure embedding failure modes: polysemy, rare items, collapse, hubness, out-of-distribution queries and social bias.

## 1. What an embedding is

An **embedding** maps items (words, images, users, products) to vectors so that geometry carries useful information: similar items end up close, and a simple model can use the vector. L0-02 covered the geometry (dot product versus cosine, anisotropy, when cosine misleads); this module is about where the vectors come from. For discrete items an embedding is a lookup table $E \in \mathbb{R}^{V \times d}$, row $i$ for item $i$, equal to a one-hot vector times $E$; `nn.Embedding` in L2-05 is exactly that. More generally any hidden layer is a **representation**, and **representation learning** means training so that some layer is useful beyond the task it was trained on.

Labels are expensive, so most representation learning is **self-supervised**: invent a task whose answer is in the data. Predict a word from its neighbors (section 2), reconstruct the input through a bottleneck (section 3), recognize two views of the same image (section 4), or predict the next token, as the GPT of L2-05 does. The pretext task is a means; the representation is judged afterwards with a **probe**: freeze it, train a simple model (linear classifier or $k$-nearest neighbors) on a labeled task, and measure accuracy. Labels are used only for this evaluation.

## 2. The distributional hypothesis and word2vec

Harris argued that differences in meaning correlate with differences in distribution, the contexts in which words occur [Harris, 1954]. If "dog" and "cat" appear in the same contexts ("the ___ sleeps", "feed the ___"), a method that only sees text can still learn that they are similar.

**Skip-gram with negative sampling (SGNS)** turns this into a classification problem [Mikolov et al., 2013]. Each word $w$ has a vector $u_w$ and, separately, a context vector $c_w$. For each observed (word, context) pair within a small window, and $k$ "negative" contexts drawn from a noise distribution $P_n$, maximize

$$\log \sigma(u_w \cdot c_{\text{pos}}) + \sum_{j=1}^{k} \log \sigma(-u_w \cdot c_{\text{neg}_j}), \qquad \sigma(z) = \frac{1}{1 + e^{-z}}.$$

Real pairs should score high, random pairs low: $k + 1$ binary decisions instead of a softmax over the vocabulary. Mikolov et al. used the unigram distribution raised to the 3/4 power as $P_n$ (boosting rare words), found $k$ of 5 to 20 useful for small datasets and 2 to 5 for large ones, and randomly discarded frequent words during training.

**What SGNS computes.** Levy and Goldberg showed that, under assumptions, SGNS implicitly factorizes a word-context matrix whose entries are the pointwise mutual information (PMI) shifted by $\log k$:

$$u_w \cdot c_{w'} \approx \operatorname{PMI}(w, w') - \log k, \qquad \operatorname{PMI}(w, w') = \log \frac{P(w, w')}{P(w)\,P(w')}$$

[Levy and Goldberg, 2014]. So a count-based alternative is natural: build the co-occurrence counts, compute positive PMI (negative values set to 0), and take a truncated SVD (L0-02). They found that SVD can match SGNS on word-similarity tasks, while SGNS did better on analogies.

**The experiment** (script 01). Corpora big enough for good word vectors are large, so the script generates 3,000 sentences (20,049 tokens) from templates like "the *animal* eats the *food*". Within each of five categories, frequencies follow a Zipf-like law (the most common word is about 23 times more frequent than the eighth). Two traps are planted: persons and animals share the context "eats the *food*", and "fish" is both an animal and a food. SGNS with 16 dimensions, window 2 and $k = 5$ trains in seconds. Each word is compared with the normalized mean vector (centroid) of its category's four most common words:

| method | common words: own centroid | best other centroid | rare words: own centroid | "fish": animal / food |
|---|---|---|---|---|
| SGNS | 0.99 | 0.78 | 0.98 | 0.91 / 0.82 |
| PPMI + SVD | 0.99 | 0.26 | 0.85 | 0.74 / 0.50 |

Four lessons. **The distributional hypothesis works**: the neighbors of "dog" are "horse, cow, cat, sheep" and of "train" "bike, bus, boat, car", with no category labels. **SGNS vectors share a common direction**: every word has cosine at least 0.78 with some other category's centroid, because frequent contexts like "the" push everything the same way (the anisotropy of L0-02); rankings are fine, raw cosines are not comparable across methods. **Rare words are noisier, especially for PPMI**, whose estimates from few counts are unreliable (0.85 against 0.99). **One vector per word type blends senses**: "fish" lands between animals and foods. Transformers (L2-05) give each occurrence its own contextual vector, which avoids this (L3-01).

Finally, on frequently observed pairs the learned $u_w \cdot c_{w'}$ correlates with $\operatorname{PMI} - \log k$ at 0.69 (means $-1.05$ and $-0.78$): the theory holds roughly after three epochs in 16 dimensions. For more depth, see [Jurafsky and Martin, 2026] and [Alammar, 2019].

## 3. Autoencoders

An **autoencoder** is an encoder $f$ that maps $x$ to a short code $z = f(x)$ and a decoder $g$ that maps it back, trained to make $g(f(x)) \approx x$. The bottleneck forces the code to keep what matters most for reconstruction.

**Linear autoencoders are PCA.** With linear $f$ and $g$ and squared error, the optimum projects onto the subspace of the leading principal components, and Baldi and Hornik proved that this error surface has no other local minima [Baldi and Hornik, 1989]. Script 02 trains one on the 8×8 digits (bundled with scikit-learn, L1-04) with an 8-number code. Its test error equals PCA's (1.568). The cosines of the principal angles between the decoder's subspace and PCA's are all 1.000, yet the individual decoder directions match PCA axes only with cosines of 0.59 to 0.87. Any invertible mix of the code gives the same reconstruction, so the subspace is determined and the axes are not. Do not interpret individual autoencoder units as if they were principal components.

**Nonlinear and denoising.** Results on the 540 test digits:

| code (8 numbers unless noted) | reconstruction error | linear probe accuracy |
|---|---|---|
| raw pixels (64 numbers) | 0 | 0.970 |
| PCA | 1.568 | 0.904 |
| linear autoencoder | 1.568 | 0.898 |
| nonlinear autoencoder (64-128-8-128-64) | 0.821 | 0.950 |
| denoising autoencoder (30% of pixels dropped) | 1.062 | 0.959 |
| random untrained encoder | (not trained) | 0.583 |

The nonlinear autoencoder halves the error and gives a better code. The denoising one, which must reconstruct clean images from inputs with 30% of pixels zeroed, reconstructs worse yet gives the best probe. **Reconstruction error does not measure representation quality**: reconstructing every pixel rewards keeping details a classifier does not need, while denoising forces the code to keep structure that survives corruption. And no 8-number code beats the raw 64 pixels here; compression has a price. Masking part of the input and predicting it is the same idea at scale, used by text and vision models (L3-01, L3-04). Variational autoencoders, which make the code a probability distribution you can sample from, belong to L3-03.

## 4. Contrastive learning

Instead of reconstructing the input, **contrastive learning** asks the encoder to recognize it. Make two random **augmentations** of each image (here: shift by up to one pixel, drop 20% of pixels, change brightness, add noise). The two views of the same image are a **positive pair**; views of other images in the batch are **negatives**. With a batch of $N$ images there are $2N$ normalized embeddings $z$, and for each view $i$ with partner $j$ the loss is

$$\ell_i = -\log \frac{\exp(z_i \cdot z_j / \tau)}{\sum_{m \ne i} \exp(z_i \cdot z_m / \tau)},$$

a softmax cross-entropy whose correct class is the partner. The temperature $\tau$ sets how sharply the loss focuses on the hardest negatives. SimCLR studied this recipe and reported three findings: the composition of augmentations matters a lot, a small nonlinear projection head between the representation and the loss substantially improves the representation, and contrastive learning benefits from larger batches and longer training than supervised learning [Chen et al., 2020]. As in SimCLR, script 03 computes the loss on the head's output but evaluates the representation before the head.

Wang and Isola describe what the loss optimizes as two properties of the normalized features: **alignment**, positives close together, measured as the mean squared distance between views; and **uniformity**, features spread over the sphere, measured as $\log \mathbb{E}\, e^{-2\lVert z - z' \rVert^2}$ over random pairs (lower is more uniform, 0 means everything is in one spot) [Wang and Isola, 2020].

Script 03 trains a 64-256-64 MLP encoder on the 1,257 training digits for 60 epochs, without labels:

| representation | kNN | kNN, test shifted 1 px | linear probe | alignment | uniformity | mean cosine | effective rank |
|---|---|---|---|---|---|---|---|
| raw pixels | 0.978 | 0.570 | 0.970 | 1.066 | $-2.196$ | 0.685 | 13.1 |
| random encoder | 0.961 | 0.517 | 0.909 | 0.619 | $-1.250$ | 0.803 | 11.7 |
| contrastive, $\tau = 0.05$ | 0.963 | 0.681 | 0.943 | 0.473 | $-1.148$ | 0.820 | 7.0 |
| contrastive, $\tau = 0.2$ | 0.961 | 0.756 | 0.943 | 0.470 | $-1.299$ | 0.746 | 6.4 |
| contrastive, $\tau = 1.0$ | 0.946 | 0.726 | 0.928 | 0.344 | $-1.213$ | 0.706 | 4.2 |
| positives only (no negatives) | 0.957 | 0.541 | 0.904 | 0.023 | $-0.049$ | 0.994 | 2.5 |

How to read it honestly:

- **On clean digits nothing beats raw pixels** (kNN 0.978), and a random encoder is close (0.961) because random projections roughly preserve distances. Without these baselines, any representation looks good.
- **The gain is the invariance you asked for.** Shift every test image one pixel and raw-pixel kNN drops to 0.570, while the $\tau = 0.2$ encoder keeps 0.756. The augmentations define what "same" means; choosing them is choosing which differences to ignore. Temperature matters too (0.681, 0.756, 0.726).
- **Without negatives the representation collapses.** Pulling views together is solved perfectly by mapping everything to one point: alignment 0.023, uniformity near 0, mean cosine between different digits 0.994. kNN still scores 0.957, because after 60 epochs the collapse is incomplete and rankings survive in tiny residual differences, but shift robustness is gone and any similarity threshold would be meaningless. Methods without negatives need another mechanism against this.
- **Few directions are used.** The effective rank, $(\sum_i \lambda_i)^2 / \sum_i \lambda_i^2$ over covariance eigenvalues, is 6.4 for the contrastive encoder against 13.1 for raw pixels, out of 64 dimensions.

Image-text contrastive models such as CLIP apply the same loss across modalities (L3-04).

## 5. Similarity search: exact and approximate

Once items are vectors, "find similar items" is nearest-neighbor search. **Exact search** on normalized vectors is one matrix product and a top-$k$ selection: $O(Nd)$ per query. Script 04 does it for $N = 100{,}000$ vectors of dimension 64 in about 1 ms per query (batched, on a loaded machine). For many applications that is enough, and you should measure before reaching for an index.

At millions or billions of vectors, exact search is too slow, and **approximate nearest-neighbor (ANN)** indexes trade a little recall for a lot of speed. The **inverted file (IVF)** index clusters the database with $k$-means (L1-04) into `nlist` cells; a query is compared with the cell centroids and scans only the `nprobe` closest cells. Script 04 builds one with 256 cells and measures **recall@10**, the fraction of the true 10 nearest neighbors found:

| `nprobe` | fraction scanned | recall@10, clustered data | recall@10, structureless data |
|---|---|---|---|
| 1 | 0.4% | 0.770 | 0.086 |
| 4 | 1.3% to 1.6% | 0.915 | 0.206 |
| 16 | 5.5% to 6.3% | 0.975 | 0.436 |
| 64 | 24% to 25% | 0.994 | 0.787 |

On clustered data (500 topics plus noise), scanning 5.5% of the database finds 97.5% of the true neighbors. On i.i.d. Gaussian data the same index is nearly useless: with no cluster structure, a point's neighbors are spread across many cells, and distances concentrate (L0-02). ANN indexes exploit structure in real embeddings; they do not beat the curse of dimensionality in general. Always measure recall on your own data. Our Python loop is slow per query, so its timings understate what a compiled library achieves; the recall numbers are the point.

Production libraries offer many index types and vector compression; the Faiss paper lays out the trade-offs of vector search from the authors of one such library [Douze et al., 2024]. Graph indexes like HNSW and their use in retrieval systems are covered in L3-06.

## 6. Failure modes

Several have already appeared: blended senses for polysemous words, noisy vectors for rare items, and collapse without negatives. L0-02 covered anisotropy and why cosine values can be arbitrary. Four more:

**Hubness.** In high dimensions, some points appear in the $k$-nearest-neighbor lists of very many other points, and many appear in none [Radovanović et al., 2010]. Script 05 counts, for 2,000 Gaussian points, how often each is among another's 10 nearest neighbors (the average is always 10):

| $d$ | skewness | max count | never a neighbor |
|---|---|---|---|
| 3 | $-0.30$ | 19 | 0.2% |
| 30 | 3.59 | 161 | 10.8% |
| 300 | 5.33 | 298 | 24.2% |

At $d = 300$ one point is a neighbor of 298 others, and a quarter of the points are never returned. Hubs tend to be the points closest to the data mean (rank correlation 0.93), as Radovanović et al. also report. In retrieval or recommendation, hubs are shown to everyone and anti-hubs to no one, for geometric rather than relevance reasons.

**There is always a nearest neighbor.** Search returns the closest item whether or not anything relevant exists. Script 05 embeds digits with 16-dimensional PCA and sets a threshold at the 5th percentile of in-distribution top-1 cosine scores (0.767). Uniform noise and inverted digits score lower (medians 0.60 and 0.61) and are flagged 98% to 99% of the time, but digits with a bright bar painted over them score a median 0.84 and 70% pass the threshold. A cosine of 0.84 looks confident. If downstream code trusts it (say, to choose a document for a language model to cite, L3-06), corrupted or out-of-scope queries get confident answers.

**Embeddings inherit the biases of their data.** Bolukbasi et al. found that word embeddings trained on Google News articles encode gender stereotypes, that the bias is captured by a direction in the embedding, and proposed removing that direction from gender-neutral words [Bolukbasi et al., 2016]. Their analogy examples, such as the one in their title, were widely cited. Nissim, van Noord and van der Goot argued that analogy tests are a poor tool for measuring bias: the standard analogy code excludes the input words from the answers, so "man is to doctor as woman is to X" cannot return "doctor"; allowing input words changes several widely cited results, often to the input word itself [Nissim et al., 2020]. Both sides agree that the embeddings reflect biases in the text. They disagree about how to measure them, and simple measurements can exaggerate or hide the effect. L4-05 takes this further.

**A good loss does not guarantee a useful representation.** Sections 3 and 4 showed it twice: lower reconstruction error with a worse probe, and near-zero alignment loss from a collapsed encoder. Evaluate representations on the use you care about, against raw-input and random-encoder baselines.

## Hands-on

Run each script from the repository root, for example `.venv/bin/python code/L2-06-representation-learning-and-embeddings/01_word2vec_sgns.py`. Outputs come from Python 3.12.3, PyTorch 2.15.0 (CPU), NumPy 2.5.4 and scikit-learn 1.9.1 on Linux (WSL2), on a machine shared with other heavy jobs. Each script ran in under 10 s; seeded results reproduced exactly across reruns, but other library versions or CPUs may differ slightly. Timings depend on load.

### Script 01: word2vec (SGNS) versus PPMI + SVD

```python
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
```

```text
corpus 20,049 tokens, vocabulary 55
training pairs 80,180
epoch 0: loss 2.956
epoch 1: loss 2.415
epoch 2: loss 2.208

frequency and polysemy (cosine to category centroids)
  SGNS      ranks 1-2: own centroid 0.99, best other 0.78
  SGNS      ranks 7-8: own centroid 0.98, best other 0.80
  SGNS      'fish': animal centroid 0.91, food centroid 0.82
  PPMI+SVD  ranks 1-2: own centroid 0.99, best other 0.26
  PPMI+SVD  ranks 7-8: own centroid 0.85, best other 0.24
  PPMI+SVD  'fish': animal centroid 0.74, food centroid 0.50
  counts: {'cat': 661, 'dog': 219, 'duck': 37, 'fish': 56, 'egg': 34}
  most common contexts of 'fish': ['the', '.', 'eats', 'sees', 'in', 'sleeps']

observed pairs: corr(w.c, PMI - log k) = 0.689, mean w.c -1.05 vs mean PMI - log k -0.78
```

### Script 02: autoencoders and PCA

```python
"""Autoencoders on the 8x8 digits: linear AE vs PCA, nonlinear AE, denoising AE.

1. A linear autoencoder trained with squared error learns the PCA subspace
   (Baldi and Hornik, 1989), though not the PCA axes themselves.
2. A nonlinear autoencoder reconstructs better at the same code size.
3. Lower reconstruction error does not guarantee a more useful code:
   we check each code with a linear probe (labels used only for evaluation).
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

torch.manual_seed(0)
X, y = load_digits(return_X_y=True)
X = (X / 16.0).astype(np.float32)               # pixels in [0, 1]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
mean = Xtr.mean(0)
K = 8                                           # code size: 64 pixels -> 8 numbers


def train_ae(encoder, decoder, noise=0.0, epochs=300):
    params = list(encoder.parameters()) + list(decoder.parameters())
    opt = torch.optim.Adam(params, lr=3e-3)
    xt = torch.from_numpy(Xtr - mean)
    for epoch in range(epochs):
        for b in torch.randperm(len(xt)).split(128):
            x = xt[b]
            # Denoising: corrupt the input, reconstruct the clean target.
            x_in = x * (torch.rand_like(x) > noise) if noise else x
            loss = ((decoder(encoder(x_in)) - x) ** 2).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return encoder, decoder


def recon_error(encoder, decoder, X_):
    with torch.no_grad():
        x = torch.from_numpy(X_ - mean)
        return ((decoder(encoder(x)) - x) ** 2).sum(1).mean().item()


def probe(Z_tr, Z_te):
    """Linear probe: logistic regression on the frozen code."""
    clf = LogisticRegression(max_iter=3000).fit(Z_tr, ytr)
    return clf.score(Z_te, yte)


def codes(encoder):
    with torch.no_grad():
        return [encoder(torch.from_numpy(A - mean)).numpy() for A in (Xtr, Xte)]


# ---- PCA baseline ---------------------------------------------------------------
pca = PCA(n_components=K).fit(Xtr)
pca_err = (((pca.inverse_transform(pca.transform(Xte)) - Xte) ** 2).sum(1)).mean()

# ---- 1. Linear autoencoder ------------------------------------------------------
lin_enc, lin_dec = train_ae(nn.Linear(64, K, bias=False), nn.Linear(K, 64, bias=False))
# Compare the subspace spanned by the decoder's columns with PCA's subspace:
# cosines of the principal angles between them are the singular values of Q1^T Q2.
Q1, _ = np.linalg.qr(lin_dec.weight.detach().numpy())
Q2, _ = np.linalg.qr(pca.components_.T)
cosines = np.linalg.svd(Q1.T @ Q2, compute_uv=False)
axis_cos = np.abs(Q1.T @ pca.components_.T).max(0)    # best match per PCA axis

# ---- 2. Nonlinear and denoising autoencoders ------------------------------------
def mlp_ae():
    enc = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, K))
    dec = nn.Sequential(nn.Linear(K, 128), nn.ReLU(), nn.Linear(128, 64))
    return enc, dec


torch.manual_seed(1)
mlp_enc, mlp_dec = train_ae(*mlp_ae())
torch.manual_seed(1)
den_enc, den_dec = train_ae(*mlp_ae(), noise=0.3)

print(f"test reconstruction error (sum of squared pixel errors), code size {K}")
print(f"  PCA                    {pca_err:.3f}")
print(f"  linear AE              {recon_error(lin_enc, lin_dec, Xte):.3f}")
print(f"  nonlinear AE           {recon_error(mlp_enc, mlp_dec, Xte):.3f}")
print(f"  denoising AE, 30% drop {recon_error(den_enc, den_dec, Xte):.3f}")
print(f"  predict the mean image {((Xte - mean) ** 2).sum(1).mean():.3f}")
print(f"\nlinear AE vs PCA: cosines of principal angles between subspaces:\n  "
      f"{np.round(cosines, 3)}")
print(f"  but individual axes do not match PCA's: best |cos| per PCA axis\n  "
      f"{np.round(axis_cos, 2)}")

print("\nlinear probe accuracy on the 8-number code (labels only for evaluation)")
print(f"  raw pixels (64 numbers) {probe(Xtr, Xte):.3f}")
print(f"  PCA                     {probe(pca.transform(Xtr), pca.transform(Xte)):.3f}")
print(f"  linear AE               {probe(*codes(lin_enc)):.3f}")
print(f"  nonlinear AE            {probe(*codes(mlp_enc)):.3f}")
print(f"  denoising AE            {probe(*codes(den_enc)):.3f}")
torch.manual_seed(2)
print(f"  random untrained MLP    {probe(*codes(mlp_ae()[0])):.3f}")
```

```text
test reconstruction error (sum of squared pixel errors), code size 8
  PCA                    1.568
  linear AE              1.568
  nonlinear AE           0.821
  denoising AE, 30% drop 1.062
  predict the mean image 4.728

linear AE vs PCA: cosines of principal angles between subspaces:
  [1. 1. 1. 1. 1. 1. 1. 1.]
  but individual axes do not match PCA's: best |cos| per PCA axis
  [0.66 0.87 0.59 0.83 0.86 0.6  0.62 0.7 ]

linear probe accuracy on the 8-number code (labels only for evaluation)
  raw pixels (64 numbers) 0.970
  PCA                     0.904
  linear AE               0.898
  nonlinear AE            0.950
  denoising AE            0.959
  random untrained MLP    0.583
```

### Script 03: contrastive learning

```python
"""SimCLR-style contrastive learning on the 8x8 digits, with no labels.

Two random augmentations of each image form a positive pair; the other
images in the batch are negatives. The InfoNCE / NT-Xent loss pulls
positives together and pushes everything else apart. We then measure:
  - kNN and linear-probe accuracy (labels used only for evaluation),
  - accuracy on test images shifted by one pixel (the invariance we asked for),
  - alignment and uniformity (Wang and Isola, 2020),
  - the mean cosine between different images and the effective rank,
and compare temperatures and a "positives only" loss, which collapses.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

torch.manual_seed(0)
X, y = load_digits(return_X_y=True)
X = (X / 16.0).astype(np.float32)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
Xtr_t, Xte_t = torch.from_numpy(Xtr), torch.from_numpy(Xte)


def shift(imgs, dx, dy):
    """Translate 8x8 images by (dx, dy) pixels, filling with zeros."""
    im = F.pad(imgs.view(-1, 1, 8, 8), (1, 1, 1, 1))
    return im[:, :, 1 - dy:9 - dy, 1 - dx:9 - dx].reshape(-1, 64)


def augment(x):
    """Random shift of up to 1 pixel, random pixel dropout, brightness, noise."""
    out = torch.empty_like(x)
    d = torch.randint(-1, 2, (len(x), 2))
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            m = (d[:, 0] == dx) & (d[:, 1] == dy)
            out[m] = shift(x[m], dx, dy)
    out = out * (torch.rand_like(out) > 0.2)
    out = out * torch.empty(len(x), 1).uniform_(0.7, 1.3)
    return out + 0.05 * torch.randn_like(out)


def make_encoder():
    return nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 64))


def nt_xent(z1, z2, tau):
    """InfoNCE over 2N views: for each view, its partner is the positive."""
    z = F.normalize(torch.cat([z1, z2]), dim=1)
    sim = z @ z.T / tau
    sim.fill_diagonal_(float("-inf"))            # a view is not its own negative
    n = len(z1)
    target = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)])
    return F.cross_entropy(sim, target)


def train(tau=0.2, positives_only=False, epochs=60, seed=0):
    torch.manual_seed(seed)
    enc = make_encoder()
    head = nn.Sequential(nn.ReLU(), nn.Linear(64, 32))   # projection head
    opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()), lr=1e-3)
    for _ in range(epochs):
        for b in torch.randperm(len(Xtr_t)).split(256):
            x = Xtr_t[b]
            z1, z2 = head(enc(augment(x))), head(enc(augment(x)))
            if positives_only:   # no negatives: only pull pairs together
                loss = (F.normalize(z1, dim=1) - F.normalize(z2, dim=1)).pow(2).sum(1).mean()
            else:
                loss = nt_xent(z1, z2, tau)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return enc


def evaluate(name, embed):
    with torch.no_grad():
        h_tr, h_te = embed(Xtr_t).numpy(), embed(Xte_t).numpy()
        h_sh = embed(shift(Xte_t, 1, 0)).numpy()          # every test image moved right
        # Alignment and uniformity on L2-normalized vectors.
        a = F.normalize(embed(augment(Xte_t)), dim=1)
        b = F.normalize(embed(augment(Xte_t)), dim=1)
        align = (a - b).pow(2).sum(1).mean().item()
        unif = torch.pdist(a).pow(2).mul(-2).exp().mean().log().item()
    norm = lambda M: M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    knn = KNeighborsClassifier(5).fit(norm(h_tr), ytr)
    lin = LogisticRegression(max_iter=3000).fit(h_tr, ytr)
    hn = norm(h_te)
    mean_cos = (hn @ hn.T)[~np.eye(len(hn), dtype=bool)].mean()
    # Effective rank (participation ratio of the covariance eigenvalues):
    # 64 if variance is spread evenly over 64 directions, 1 if all in one.
    ev = np.linalg.svd(h_tr - h_tr.mean(0), compute_uv=False) ** 2
    eff_rank = ev.sum() ** 2 / (ev ** 2).sum()
    print(f"{name:<24} {knn.score(hn, yte):>5.3f} {knn.score(norm(h_sh), yte):>8.3f}"
          f" {lin.score(h_te, yte):>7.3f} {align:>7.3f} {unif:>7.3f}"
          f" {mean_cos:>8.3f} {eff_rank:>6.1f}")


print(f"{'representation':<24} {'kNN':>5} {'kNN,shift':>8} {'linear':>7}"
      f" {'align':>7} {'unif':>7} {'mean cos':>8} {'eff.rk':>6}")
evaluate("raw pixels", lambda x: x)
torch.manual_seed(0)
evaluate("random encoder", make_encoder())
for tau in (0.05, 0.2, 1.0):
    evaluate(f"contrastive, tau={tau}", train(tau))
evaluate("positives only", train(positives_only=True))
```

```text
representation             kNN kNN,shift  linear   align    unif mean cos eff.rk
raw pixels               0.978    0.570   0.970   1.066  -2.196    0.685   13.1
random encoder           0.961    0.517   0.909   0.619  -1.250    0.803   11.7
contrastive, tau=0.05    0.963    0.681   0.943   0.473  -1.148    0.820    7.0
contrastive, tau=0.2     0.961    0.756   0.943   0.470  -1.299    0.746    6.4
contrastive, tau=1.0     0.946    0.726   0.928   0.344  -1.213    0.706    4.2
positives only           0.957    0.541   0.904   0.023  -0.049    0.994    2.5
```

### Script 04: exact and IVF search

```python
"""Exact versus approximate nearest-neighbor search, with an IVF index from scratch.

Exact search scores the query against every vector: one matrix product.
An inverted-file (IVF) index clusters the database with k-means and, at
query time, scans only the vectors in the `nprobe` clusters whose centroids
are closest to the query. Fewer vectors scanned means faster search, at the
price of sometimes missing a true neighbor. We measure that trade-off as
recall@10 and show that it depends on the data having cluster structure.
"""
import time

import numpy as np
from sklearn.cluster import KMeans

rng = np.random.default_rng(0)
N, D, NQ, K = 100_000, 64, 200, 10


def normalize(M):
    return (M / np.linalg.norm(M, axis=1, keepdims=True)).astype(np.float32)


def clustered_data(n):
    # 500 "topics"; each vector is a topic center plus noise.
    centers = rng.normal(size=(500, D))
    return normalize(centers[rng.integers(0, 500, n)] + 0.6 * rng.normal(size=(n, D)))


def exact_topk(db, q, k=K):
    scores = q @ db.T                                    # (NQ, N) cosine scores
    top = np.argpartition(-scores, k, axis=1)[:, :k]     # unordered top k, O(N)
    return top


class IVF:
    def __init__(self, db, nlist):
        km = KMeans(nlist, n_init=1, max_iter=20, random_state=0)
        km.fit(db[rng.choice(len(db), 20_000, replace=False)])  # train on a sample
        self.centroids = km.cluster_centers_.astype(np.float32)
        assign = np.argmax(db @ self.centroids.T, axis=1)
        self.lists = [np.flatnonzero(assign == c) for c in range(nlist)]
        # Store each list's vectors contiguously, as real IVF indexes do.
        self.vecs = [db[ids] for ids in self.lists]

    def search(self, q, nprobe, k=K):
        out, scanned = [], 0
        near = np.argsort(-(q @ self.centroids.T), axis=1)[:, :nprobe]
        for qi, cl in zip(q, near):
            cand = np.concatenate([self.lists[c] for c in cl])
            s = np.concatenate([self.vecs[c] @ qi for c in cl])
            scanned += len(cand)
            out.append(cand[np.argpartition(-s, k)[:k]])
        return out, scanned / len(q)


def recall(found, truth):
    return np.mean([len(set(f) & set(t)) / K for f, t in zip(found, truth)])


for name, db in (("clustered data", clustered_data(N)),
                 ("structureless Gaussian data", normalize(rng.normal(size=(N, D))))):
    q = normalize(db[rng.choice(N, NQ)] + 0.1 * rng.normal(size=(NQ, D)))
    t0 = time.perf_counter()
    truth = exact_topk(db, q)
    t_exact = (time.perf_counter() - t0) / NQ * 1e3
    t0 = time.perf_counter()
    index = IVF(db, nlist=256)
    t_build = time.perf_counter() - t0
    print(f"\n{name}: N={N:,}, d={D}; exact search {t_exact:.3f} ms/query; "
          f"IVF build {t_build:.1f} s")
    print(f"  {'nprobe':>6} {'scanned':>9} {'recall@10':>10} {'ms/query':>9}")
    for nprobe in (1, 4, 16, 64):
        t0 = time.perf_counter()
        found, scanned = index.search(q, nprobe)
        ms = (time.perf_counter() - t0) / NQ * 1e3
        print(f"  {nprobe:>6} {scanned / N:>9.1%} {recall(found, truth):>10.3f} {ms:>9.3f}")
```

```text
clustered data: N=100,000, d=64; exact search 0.932 ms/query; IVF build 0.7 s
  nprobe   scanned  recall@10  ms/query
       1      0.4%      0.770     0.029
       4      1.3%      0.915     0.060
      16      5.5%      0.975     0.119
      64     24.2%      0.994     0.466

structureless Gaussian data: N=100,000, d=64; exact search 1.248 ms/query; IVF build 0.7 s
  nprobe   scanned  recall@10  ms/query
       1      0.4%      0.086     0.027
       4      1.6%      0.206     0.048
      16      6.3%      0.436     0.135
      64     25.0%      0.787     0.562
```

### Script 05: hubness and out-of-distribution queries

```python
"""Two failure modes of embedding search you can measure.

1. Hubness (Radovanovic et al., 2010): in high dimensions a few points
   appear in many other points' k-nearest-neighbor lists, and many points
   appear in none. Hubs tend to be the points closest to the data mean.
2. A nearest neighbor always exists: an out-of-distribution query still
   gets a top result with a respectable-looking cosine score.
"""
import numpy as np
from scipy.stats import skew, spearmanr
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

rng = np.random.default_rng(0)


def k_occurrence(X, k=10):
    """N_k(x): how many points have x among their k nearest neighbors (Euclidean)."""
    sq = (X ** 2).sum(1)
    D2 = sq[:, None] + sq[None, :] - 2 * X @ X.T
    np.fill_diagonal(D2, np.inf)
    nn = np.argpartition(D2, k, axis=1)[:, :k]
    return np.bincount(nn.ravel(), minlength=len(X))


print("1. hubness: N_k for k = 10, 2000 i.i.d. Gaussian points (mean of N_k is always 10)")
print(f"{'d':>5} {'skewness':>9} {'max N_k':>8} {'N_k = 0':>8} {'corr(N_k, -dist to mean)':>25}")
for d in (3, 30, 300):
    X = rng.normal(size=(2000, d))
    Nk = k_occurrence(X)
    rho = spearmanr(Nk, -np.linalg.norm(X - X.mean(0), axis=1))[0]
    print(f"{d:>5} {skew(Nk):>9.2f} {Nk.max():>8d} {np.mean(Nk == 0):>8.1%} {rho:>25.2f}")

# ---- 2. Out-of-distribution queries ----------------------------------------------
X, y = load_digits(return_X_y=True)
X = X / 16.0
db, queries = X[:1500], X[1500:]
pca = PCA(16, random_state=0).fit(db)


def embed(A):
    Z = pca.transform(A)
    return Z / np.linalg.norm(Z, axis=1, keepdims=True)


E = embed(db)


def top1(A):
    return (embed(A) @ E.T).max(1)


ood = {
    "held-out digits (in distribution)": queries,
    "uniform noise images": rng.uniform(0, 1, size=(200, 64)),
    "digits plus a bright 3-column bar": np.clip(
        queries[:200] + np.repeat(rng.uniform(0, 1, (200, 1)), 64, 1)
        * (np.arange(64) % 8 < 3), 0, 1),
    "inverted digits (1 - x)": 1 - queries[:200],
}
print("\n2. top-1 cosine score in a 16-d PCA embedding of digits")
thr = np.quantile(top1(queries), 0.05)   # flags 5% of in-distribution queries
print(f"   threshold = 5th percentile of in-distribution scores = {thr:.3f}")
print(f"   {'query set':<36} {'median':>7} {'min':>6} {'flagged':>8}")
for name, A in ood.items():
    s = top1(A)
    print(f"   {name:<36} {np.median(s):>7.3f} {s.min():>6.3f} {np.mean(s < thr):>8.0%}")
```

```text
1. hubness: N_k for k = 10, 2000 i.i.d. Gaussian points (mean of N_k is always 10)
    d  skewness  max N_k  N_k = 0  corr(N_k, -dist to mean)
    3     -0.30       19     0.2%                      0.24
   30      3.59      161    10.8%                      0.94
  300      5.33      298    24.2%                      0.93

2. top-1 cosine score in a 16-d PCA embedding of digits
   threshold = 5th percentile of in-distribution scores = 0.767
   query set                             median    min  flagged
   held-out digits (in distribution)      0.926  0.671       5%
   uniform noise images                   0.596  0.423      99%
   digits plus a bright 3-column bar      0.838  0.544      30%
   inverted digits (1 - x)                0.611  0.466      98%
```

## Common mistakes

- **Judging an embedding by its training loss.** Reconstruction or contrastive loss says little about downstream use. Probe it, and compare with raw inputs and a random encoder.
- **No baselines.** On easy data, raw features or random projections may already be as good (script 03).
- **Reading individual dimensions.** Autoencoder and word2vec axes are not identifiable; any rotation of the code works equally well.
- **Mixing embeddings from different models or versions.** Vectors from two training runs live in unrelated coordinate systems. Re-embed the whole index after retraining.
- **Trusting top-1 scores.** Search always returns something. Calibrate thresholds on held-out in-distribution and out-of-distribution queries.
- **Assuming ANN recall.** Measure recall@k against exact search on your own data; structureless or shifted data can wreck an index tuned elsewhere.
- **Treating analogy tests as definitive.** Excluding input words and choosing which analogies to report can change conclusions.

## Exercises

1. [warm-up] For the word "dog" in script 01, list three contexts it shares with "cat" and one it shares with "man".
2. [warm-up] Why do the linear autoencoder and PCA have identical reconstruction error but different axes?
3. [warm-up] In the IVF table, why does recall rise with `nprobe`, and what is recall when `nprobe` equals `nlist`?
4. [core] In script 01, remove the shared "the person eats the food" template and rerun. How do the "best other centroid" numbers change, and why?
5. [core] Change the SGNS noise distribution from $U^{3/4}$ to $U^{1}$ and to uniform. Compare neighbors of rare words.
6. [core] In script 02, sweep the code size (2, 4, 8, 16, 32) for PCA and the nonlinear autoencoder. Plot reconstruction error and probe accuracy against code size.
7. [core] In script 03, remove the shift augmentation and retrain. What happens to "kNN, test shifted"? Explain in terms of what the augmentations define.
8. [stretch] Implement random-hyperplane hashing (sign of projections on 16 random vectors) as another ANN index and compare its recall/scan curve with IVF.
9. [stretch] Reduce hubness in script 05 by centering and then normalizing, or by replacing raw similarity with a rank-based score. Measure the skewness of $N_k$ before and after.
10. [stretch] Plant a "gender" direction in a synthetic embedding (as in L0-02 script 06), then test analogies with and without excluding input words.

### Hints and answers

1. "the ___ eats", "sees the ___", "the ___ sleeps in"; shared with "man": "the ___ eats the".
2. Reconstruction depends only on the subspace the code spans; any invertible linear map applied to the code and undone by the decoder gives the same output.
3. Each extra cell scanned can only add true neighbors. With `nprobe = nlist` the search is exhaustive and recall is 1.
4. Persons and animals share fewer contexts, so their vectors should move apart; expect lower "best other" cosines for those two categories.
5. With $U^1$, negatives are mostly frequent words and rare words get few gradient updates as negatives; uniform oversamples rare words. Look at the rare-word centroid cosines.
6. Error falls with code size for both; the probe usually saturates sooner than reconstruction, another sign the two measure different things.
7. Shifted-test kNN should fall toward the raw-pixel or random-encoder level, because nothing told the encoder that shifted images are the same.
8. Bucket by the 16-bit code; probe buckets in order of Hamming distance. Expect lower recall than IVF on clustered data at the same fraction scanned.
9. Radovanović et al. link hubs to proximity to the data mean, so centering is a natural first try; measure rather than assume.
10. With exclusion, the planted direction can make "man : doctor :: woman : ?" return a different word even when "doctor" is the true nearest point.

## Check your understanding

1. **What makes an embedding good?** Only its usefulness for a stated purpose: classes separable by a simple probe, nearest neighbors that a user would call similar, invariance to the nuisances you care about. Training loss, reconstruction error or pretty plots do not settle it. Always compare with raw inputs and a random encoder.

2. **Why does word2vec learn that "dog" and "cat" are similar without being told?** They occur in the same contexts, and SGNS pushes a word's vector toward the context vectors it co-occurs with. Words with similar context distributions get similar vectors. Levy and Goldberg showed this amounts to factorizing a shifted PMI matrix, which is why an SVD of PPMI gives similar neighbors.

3. **Why does contrastive learning need negatives (or a substitute)?** Pulling positive pairs together is minimized perfectly by mapping every input to the same vector. Negatives push different inputs apart, which keeps the representation spread out (uniformity). Script 03 showed the collapse: mean cosine 0.994 without negatives.

4. **When is approximate search a bad idea?** When exact search is fast enough (100,000 vectors took about 1 ms per query here), when the data lacks the cluster structure the index relies on, or when missing a true neighbor is costly. In all cases, measure recall against exact search.

5. **A similarity search returns a top result with cosine 0.84. Is it relevant?** Not necessarily. Scores depend on the model, dimension and anisotropy, and the nearest neighbor always exists. In script 05, 70% of corrupted digits passed a threshold set on clean data. Calibrate thresholds on held-out queries, including out-of-scope ones, and check the result.

## Further reading

- **Daniel Jurafsky and James H. Martin, "Speech and Language Processing", 3rd edition draft, chapter 5 "Embeddings"** (draft of August 19, 2026). https://web.stanford.edu/~jurafsky/slp3/
  *Read for:* the distributional hypothesis, PMI and skip-gram with negative sampling, explained for beginners. Free PDF. *Level:* intro.
- **Jay Alammar, "The Illustrated Word2vec"** (2019, blog). https://jalammar.github.io/illustrated-word2vec/
  *Read for:* a picture-first walk through skip-gram and negative sampling. *Level:* intro.
- **Zellig S. Harris, "Distributional Structure"** (1954, *Word* 10(2-3), 146 to 162). https://doi.org/10.1080/00437956.1954.11659520
  *Read for:* the original statement of the distributional hypothesis. Publisher version may be paywalled. *Level:* intermediate.
- **Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, Jeffrey Dean, "Distributed Representations of Words and Phrases and their Compositionality"** (2013, NeurIPS 26; arXiv:1310.4546). https://arxiv.org/abs/1310.4546
  *Read for:* negative sampling, the $U^{3/4}$ noise distribution and subsampling of frequent words. *Level:* intermediate.
- **Omer Levy and Yoav Goldberg, "Neural Word Embedding as Implicit Matrix Factorization"** (2014, NeurIPS 27). https://proceedings.neurips.cc/paper/2014/hash/b78666971ceae55a8e87efb7cbfd9ad4-Abstract.html
  *Read for:* why SGNS is close to factorizing a shifted PMI matrix, and how SVD compares. *Level:* advanced.
- **Pierre Baldi and Kurt Hornik, "Neural Networks and Principal Component Analysis: Learning from Examples Without Local Minima"** (1989, *Neural Networks* 2(1), 53 to 58). https://doi.org/10.1016/0893-6080(89)90014-2
  *Read for:* the proof that linear autoencoders learn the PCA subspace. Publisher version may be paywalled. *Level:* advanced.
- **Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton, "A Simple Framework for Contrastive Learning of Visual Representations"** (2020, ICML; arXiv:2002.05709). https://arxiv.org/abs/2002.05709
  *Read for:* SimCLR's ablations of augmentations, projection head, temperature and batch size. *Level:* intermediate.
- **Tongzhou Wang and Phillip Isola, "Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere"** (2020, ICML; arXiv:2005.10242). https://arxiv.org/abs/2005.10242
  *Read for:* the alignment and uniformity metrics used in script 03. *Level:* intermediate.
- **Miloš Radovanović, Alexandros Nanopoulos, Mirjana Ivanović, "Hubs in Space: Popular Nearest Neighbors in High-Dimensional Data"** (2010, *JMLR* 11, 2487 to 2531). https://jmlr.org/papers/v11/radovanovic10a.html
  *Read for:* hubness as a property of high-dimensional distributions. Open access. *Level:* advanced.
- **Matthijs Douze, Alexandr Guzhva, Chengqi Deng, Jeff Johnson, Gergely Szilvasy, Pierre-Emmanuel Mazaré, Maria Lomeli, Lucas Hosseini, Hervé Jégou, "The Faiss library"** (2024, arXiv:2401.08281). https://arxiv.org/abs/2401.08281
  *Read for:* the trade-offs of vector search (speed, memory, accuracy) and the indexing and compression methods in a widely used library. *Level:* intermediate.
- **Tolga Bolukbasi, Kai-Wei Chang, James Zou, Venkatesh Saligrama, Adam Kalai, "Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings"** (2016, NeurIPS 29; arXiv:1607.06520). https://arxiv.org/abs/1607.06520
  *Read for:* gender bias as a direction in word2vec space, and a debiasing method. *Level:* intermediate.
- **Malvina Nissim, Rik van Noord, Rob van der Goot, "Fair Is Better than Sensational: Man Is to Doctor as Woman Is to Doctor"** (2020, *Computational Linguistics* 46(2), 487 to 497). https://aclanthology.org/2020.cl-2.7
  *Read for:* why analogy tests mislead as bias measurements. Open access. *Level:* intermediate.
