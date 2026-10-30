"""k-means (Lloyd's algorithm) from scratch, with random and k-means++ seeding.

Shows: the objective never increases, different seeds land in different local
minima, k-means++ seeding helps, and our result matches scikit-learn.
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score


def sq_dists(X, C):
    # ||x - c||^2 = ||x||^2 - 2 x.c + ||c||^2, computed for all pairs at once.
    # Rounding can make this slightly negative when x == c, so clip at 0.
    d2 = (X**2).sum(1)[:, None] - 2 * X @ C.T + (C**2).sum(1)[None, :]
    return np.maximum(d2, 0.0)


def init_random(X, k, rng):
    return X[rng.choice(len(X), size=k, replace=False)]


def init_kmeanspp(X, k, rng):
    # First center uniform; each next center sampled with prob. proportional
    # to squared distance from the nearest center chosen so far.
    centers = [X[rng.integers(len(X))]]
    for _ in range(k - 1):
        d2 = sq_dists(X, np.array(centers)).min(1)
        centers.append(X[rng.choice(len(X), p=d2 / d2.sum())])
    return np.array(centers)


def lloyd(X, C, max_iter=100):
    history = []
    for _ in range(max_iter):
        d2 = sq_dists(X, C)
        labels = d2.argmin(1)                     # assignment step
        history.append(d2[np.arange(len(X)), labels].sum())
        newC = np.array([X[labels == j].mean(0) if np.any(labels == j) else C[j]
                         for j in range(len(C))])  # update step
        if np.allclose(newC, C):
            break
        C = newC
    return C, labels, history


X, y = make_blobs(n_samples=1500, centers=10, cluster_std=0.6,
                  center_box=(-10, 10), random_state=3)
k = 10

_, _, hist = lloyd(X, init_random(X, k, np.random.default_rng(0)))
print("Objective per iteration (seed 0, random init):")
print("  " + ", ".join(f"{h:.0f}" for h in hist))
print("  never increases:", all(b <= a + 1e-9 for a, b in zip(hist, hist[1:])))

for name, init in [("random", init_random), ("k-means++", init_kmeanspp)]:
    finals, aris = [], []
    for seed in range(50):
        C, labels, hist = lloyd(X, init(X, k, np.random.default_rng(seed)))
        finals.append(hist[-1])
        aris.append(adjusted_rand_score(y, labels))
    finals = np.array(finals)
    best = finals.min()
    print(f"\n{name} init, 50 seeds:")
    print(f"  final objective  min {best:.0f}  median {np.median(finals):.0f}"
          f"  max {finals.max():.0f}")
    print(f"  runs within 1% of best: {(finals <= best * 1.01).sum()}/50")
    print(f"  ARI vs true labels: median {np.median(aris):.3f}  min {min(aris):.3f}")

sk = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
print(f"\nscikit-learn KMeans (k-means++, n_init=10): inertia {sk.inertia_:.0f}"
      f"  ARI {adjusted_rand_score(y, sk.labels_):.3f}")
