"""Measure how t-SNE distorts cluster sizes, between-cluster distances and noise.

Every effect is printed as a number, and a 2x2 figure is saved for the lesson.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

rng = np.random.default_rng(0)
D = 10                                    # dimension of the synthetic data
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
OUT = Path(__file__).resolve().parents[2] / "docs/assets/L1-04-unsupervised-learning"


def tsne(X, perplexity, seed=0):
    return TSNE(perplexity=perplexity, init="pca", random_state=seed).fit_transform(X)


def spread(P):
    # Root-mean-square distance of points from their cluster centroid.
    return np.sqrt(((P - P.mean(0)) ** 2).sum(1).mean())


def knn_kept(X, E, k=10):
    # Fraction of each point's k nearest neighbors that stay neighbors in E.
    a = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)[1][:, 1:]
    b = NearestNeighbors(n_neighbors=k + 1).fit(E).kneighbors(E)[1][:, 1:]
    return np.mean([len(set(r) & set(s)) / k for r, s in zip(a, b)])


# 1. Cluster sizes: one tight cluster (std 1) and one diffuse cluster (std 10).
tight = rng.normal(0, 1, (200, D))
wide = rng.normal(0, 10, (200, D)) + 100
E1 = tsne(np.vstack([tight, wide]), 30)
print("1. cluster size")
print(f"   spread ratio wide/tight, original 10-D: {spread(wide) / spread(tight):.1f}")
print(f"   spread ratio wide/tight, t-SNE 2-D    : {spread(E1[200:]) / spread(E1[:200]):.1f}")

# 2. Between-cluster distances: A and B are close, C is 10x farther from A.
centers = np.zeros((3, D))
centers[1, 0], centers[2, 0] = 10, 100
X2 = np.vstack([rng.normal(c, 1, (100, D)) for c in centers])
print("\n2. between-cluster distance, ratio dist(A,C)/dist(A,B); original = 10.0")
E2 = {}
for p in (5, 30, 100):
    E2[p] = tsne(X2, p)
    m = [E2[p][i * 100:(i + 1) * 100].mean(0) for i in range(3)]
    print(f"   perplexity {p:>3}: {np.linalg.norm(m[0] - m[2]) / np.linalg.norm(m[0] - m[1]):.1f}")

# 3. Pure noise: one Gaussian blob, no clusters at all.
noise = rng.normal(size=(300, D))
E3 = {p: tsne(noise, p) for p in (2, 30)}
print("\n3. pure Gaussian noise: plots saved (see figure, panels c and d)")

# 4. Local vs global structure on real data (digits, 64-D).
X4 = load_digits().data
dist_hi = pdist(X4)
print("\n4. digits: neighbors kept and global distance correlation")
for name, E in [("PCA 2-D", PCA(2).fit_transform(X4)), ("t-SNE 2-D", tsne(X4, 30))]:
    rho = spearmanr(dist_hi, pdist(E))[0]
    print(f"   {name:9s}: 10-NN kept {knn_kept(X4, E):.2f}, "
          f"Spearman corr. of all pairwise distances {rho:.2f}")

# Figure: what a reader would see, side by side.
fig, ax = plt.subplots(2, 2, figsize=(9, 8), facecolor="#fcfcfb")
panels = [
    (E1, [0, 200, 400], "(a) std 1 vs std 10: look the same size"),
    (E2[30], [0, 100, 200, 300], "(b) A to C is 10x A to B in 10-D; perplexity 30"),
    (E3[2], [0, 300], "(c) pure noise, perplexity 2: fake clumps"),
    (E3[30], [0, 300], "(d) pure noise, perplexity 30"),
]
for a, (E, cuts, title) in zip(ax.flat, panels):
    for i in range(len(cuts) - 1):
        color = "#8a8985" if len(cuts) == 2 else COLORS[i]
        a.scatter(*E[cuts[i]:cuts[i + 1]].T, s=8, color=color, linewidths=0)
    a.set_title(title, fontsize=10, color="#0b0b0b", loc="left")
    a.set_xticks([]), a.set_yticks([])
    a.set_facecolor("#fcfcfb")
    for s in a.spines.values():
        s.set_color("#d0cfca")
# Direct labels so identity never depends on color alone.
for E, n, labels, a in [(E1, 200, ["std 1", "std 10"], ax[0, 0]),
                        (E2[30], 100, ["A", "B", "C"], ax[0, 1])]:
    for i, label in enumerate(labels):
        a.annotate(label, E[i * n:(i + 1) * n].mean(0), fontsize=11,
                   color="#0b0b0b", weight="bold", ha="center")
fig.tight_layout()
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "tsne_pitfalls.png", dpi=110)
print(f"\nsaved {OUT.name}/tsne_pitfalls.png")
