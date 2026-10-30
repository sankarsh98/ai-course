"""Minibatch SGD on least-squares regression: gradient noise and the noise floor."""
import numpy as np

rng = np.random.default_rng(0)
n, d = 20_000, 20
X = rng.normal(size=(n, d))
w_true = rng.normal(size=d)
y = X @ w_true + 0.5 * rng.normal(size=n)          # noise std 0.5

def loss(w):
    r = X @ w - y
    return 0.5 * np.mean(r ** 2)

def grad(w, idx):
    r = X[idx] @ w - y[idx]
    return X[idx].T @ r / len(idx)

w_star = np.linalg.lstsq(X, y, rcond=None)[0]
f_star = loss(w_star)
print(f"optimal loss f* = {f_star:.5f}  (noise variance/2 = {0.25 / 2:.5f})")

print("\nPart 1: minibatch gradient error at w = 0 (200 draws per batch size)")
w = np.zeros(d)
g_full = grad(w, np.arange(n))
print(f"{'B':>6} {'E|g_B - g|^2':>14} {'x B':>8}")
for B in [1, 10, 100, 1000]:
    errs = [np.sum((grad(w, rng.choice(n, B, replace=False)) - g_full) ** 2)
            for _ in range(200)]
    print(f"{B:6d} {np.mean(errs):14.4f} {np.mean(errs) * B:8.2f}")

def sgd(B, lr, epochs, decay=False):
    """Run minibatch SGD for a fixed number of passes over the data."""
    w = np.zeros(d)
    t = 0
    for _ in range(epochs):
        perm = rng.permutation(n)
        for start in range(0, n, B):
            idx = perm[start:start + B]
            step = lr / (1 + t / 2000) if decay else lr   # 1/t-style decay
            w -= step * grad(w, idx)
            t += 1
    return loss(w) - f_star, t

print("\nPart 2: 2 epochs (40,000 examples seen), constant lr = 0.05")
print(f"{'B':>6} {'updates':>8} {'f - f*':>12}")
for B in [1, 10, 100, 1000, n]:
    gap, t = sgd(B, 0.05, 2)
    print(f"{B:6d} {t:8d} {gap:12.3e}")

print("\nPart 3: B = 1, 2 epochs, constant vs decaying lr")
for lr in [0.05, 0.01, 0.002]:
    gap_c, _ = sgd(1, lr, 2)
    gap_d, _ = sgd(1, lr, 2, decay=True)
    print(f"lr0 = {lr:5.3f}: constant f - f* = {gap_c:.2e}, decaying f - f* = {gap_d:.2e}")
