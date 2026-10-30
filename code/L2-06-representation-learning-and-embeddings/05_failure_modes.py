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
