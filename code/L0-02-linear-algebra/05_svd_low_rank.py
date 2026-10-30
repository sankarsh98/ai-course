"""SVD and low-rank approximation, checked against the Eckart-Young theorem.

Run from the repository root so the plot lands in docs/assets/.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits

rng = np.random.default_rng(0)
OUT = Path("docs/assets/L0-02-linear-algebra")

# 1. SVD of a small matrix: A = U diag(s) V^T.
A = np.array([[3.0, 1.0, 1.0],
              [-1.0, 3.0, 1.0]])
U, s, Vt = np.linalg.svd(A, full_matrices=False)
print("singular values:", np.round(s, 4))
print("reconstruct A:", np.allclose(A, U @ np.diag(s) @ Vt))
print("s^2 == eigenvalues of A A^T:", np.round(s**2, 4), np.round(np.linalg.eigvalsh(A @ A.T)[::-1], 4))
print("||A||_2 (largest stretch) =", round(np.linalg.norm(A, 2), 4),
      "| ||A||_F =", round(np.linalg.norm(A, "fro"), 4),
      "= sqrt(sum s^2) =", round(float(np.sqrt((s**2).sum())), 4))


def rank_k(U, s, Vt, k):
    return (U[:, :k] * s[:k]) @ Vt[:k]       # sum of the top-k rank-1 pieces


# 2. Eckart-Young: truncated SVD is the best rank-k approximation, with
#    Frobenius error sqrt(s_{k+1}^2 + ...). Check against random rank-k guesses.
M = rng.normal(size=(60, 40)) @ np.diag(np.linspace(3, 0.1, 40)) @ rng.normal(size=(40, 40))
Um, sm, Vtm = np.linalg.svd(M, full_matrices=False)
k = 5
err_svd = np.linalg.norm(M - rank_k(Um, sm, Vtm, k), "fro")
print(f"\nrank-{k} SVD error = {err_svd:.4f}; theory sqrt(sum s[k:]^2) = "
      f"{np.sqrt((sm[k:]**2).sum()):.4f}")
best_random = np.inf
for _ in range(200):
    # best possible fit of M using a random k-dim column space (projection)
    Qr, _ = np.linalg.qr(rng.normal(size=(60, k)))
    best_random = min(best_random, np.linalg.norm(M - Qr @ (Qr.T @ M), "fro"))
print(f"best of 200 random rank-{k} subspaces: error = {best_random:.4f}")

# 3. Real data: 8x8 handwritten digits (bundled with scikit-learn), 1797 x 64.
X = load_digits().data                       # pixel values 0..16
Ux, sx, Vtx = np.linalg.svd(X, full_matrices=False)
energy = np.cumsum(sx**2) / np.sum(sx**2)
print(f"\ndigits matrix {X.shape}, numerical rank {np.linalg.matrix_rank(X)}")
print("first 3 singular values:", np.round(sx[:3], 1), "| last 4:", sx[-4:])
print(f"{'k':>4}{'rel. Frobenius error':>22}{'energy kept':>13}{'numbers stored':>16}")
for k in (1, 2, 5, 10, 20, 40):
    rel = np.linalg.norm(X - rank_k(Ux, sx, Vtx, k)) / np.linalg.norm(X)
    stored = k * (X.shape[0] + X.shape[1] + 1)
    print(f"{k:>4}{rel:>22.4f}{energy[k-1]:>13.4f}{stored:>16,}")
print(f"full matrix stores {X.size:,} numbers")

# 4. Plot: original digits and their rank-k reconstructions.
OUT.mkdir(parents=True, exist_ok=True)
ks = [1, 3, 5, 10, 20]
idx = [0, 1, 2, 3]
fig, axes = plt.subplots(len(idx), len(ks) + 1, figsize=(7, 5), layout="constrained")
for row, i in enumerate(idx):
    axes[row, 0].imshow(X[i].reshape(8, 8), cmap="gray_r")
    for col, k in enumerate(ks, start=1):
        rec = rank_k(Ux, sx, Vtx, k)[i]
        axes[row, col].imshow(rec.reshape(8, 8), cmap="gray_r")
for col, title in enumerate(["original"] + [f"rank {k}" for k in ks]):
    axes[0, col].set_title(title, fontsize=9)
for ax in axes.ravel():
    ax.set_xticks([])
    ax.set_yticks([])
fig.savefig(OUT / "digits_low_rank.png", dpi=110)
print(f"saved {OUT / 'digits_low_rank.png'}")
