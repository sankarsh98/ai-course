"""Differential privacy: noise versus accuracy (educational demo, synthetic data).

Part 1: the Laplace mechanism on a count and a mean, and what composition costs.
Part 2: differentially private full-batch gradient descent for logistic regression
        (per-example clipping + Gaussian noise), with zCDP accounting, at several
        privacy levels and two dataset sizes.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")  # small matrices: threads only add overhead
import numpy as np

rng = np.random.default_rng(0)

# ---------------------------------------------------------------- Part 1
# 10,000 synthetic incomes. To bound the sensitivity of a mean we must clip
# every value to a public range [0, U]; the clipping itself is part of the
# mechanism (and introduces bias if U is too small).
n = 10_000
U = 200_000.0
incomes = np.clip(rng.lognormal(mean=10.8, sigma=0.6, size=n), 0, U)
over_50k = (incomes > 50_000).sum()
true_mean = incomes.mean()

print("Part 1: Laplace mechanism (replace-one neighbors, n public)")
print(f"  true count(income > 50k) = {over_50k}, true mean = {true_mean:,.0f}")
print(f"  {'eps':>5} {'count: mean |err|':>18} {'mean: mean |err|':>17} {'mean: rel err':>14}")
for eps in [0.01, 0.1, 0.5, 1.0, 5.0]:
    # Count: changing one person's record changes the count by at most 1.
    count_noise = rng.laplace(scale=1.0 / eps, size=2000)
    # Mean of values in [0, U] with n fixed: replacing one record moves it by <= U/n.
    mean_noise = rng.laplace(scale=(U / n) / eps, size=2000)
    print(f"  {eps:5.2f} {np.abs(count_noise).mean():18.1f} "
          f"{np.abs(mean_noise).mean():17.1f} {np.abs(mean_noise).mean() / true_mean:14.2%}")

# Basic composition: k queries each at eps_i cost sum(eps_i). Splitting a fixed
# budget over more queries makes each one noisier.
print("\n  fixed total budget eps=1 split across k count queries:")
for k in [1, 10, 100]:
    print(f"    k={k:3d}: each query eps={1 / k:.3f}, expected |error| per query = {k:.0f}")

# ---------------------------------------------------------------- Part 2
D = 20
W_TRUE = np.random.default_rng(123).normal(size=D) / np.sqrt(D) * 3  # fixed "world"


def make_data(n, seed=0):
    r = np.random.default_rng(seed)
    X = r.normal(size=(n, D))
    p = 1 / (1 + np.exp(-X @ W_TRUE))
    y = (r.random(n) < p).astype(float)
    return X, y


def dp_gd(X, y, z, steps=100, lr=0.5, C=1.0, seed=0):
    """Full-batch GD on logistic loss. Each step: clip every per-example gradient
    to L2 norm <= C, sum, add N(0, (z * 2C)^2 I), divide by n.
    Replace-one sensitivity of the clipped sum is 2C, so the noise std is z * sensitivity.
    z = 0 means no noise (but still clipped, so the comparison isolates the noise)."""
    r = np.random.default_rng(seed)
    n, d = X.shape
    w = np.zeros(d)
    for _ in range(steps):
        p = 1 / (1 + np.exp(-X @ w))
        g = (p - y)[:, None] * X                       # per-example gradients, (n, d)
        norms = np.linalg.norm(g, axis=1, keepdims=True)
        g = g / np.maximum(1.0, norms / C)             # clip each row to norm <= C
        noisy_sum = g.sum(0) + r.normal(scale=z * 2 * C, size=d)
        w -= lr * noisy_sum / n
    return w


def zcdp_epsilon(z, steps, delta):
    """Gaussian mechanism with noise std = z * sensitivity is rho = 1/(2 z^2) zCDP;
    zCDP composes by adding rho; rho-zCDP implies (rho + 2 sqrt(rho ln(1/delta)), delta)-DP
    [Bun and Steinke, 2016]. This bound is valid but not the tightest accountant."""
    rho = steps / (2 * z ** 2)
    return rho + 2 * np.sqrt(rho * np.log(1 / delta))


def accuracy(w, X, y):
    return ((X @ w > 0) == (y == 1)).mean()


X_test, y_test = make_data(20_000, seed=99)
delta = 1e-5
print("\nPart 2: DP gradient descent, logistic regression, d=20, 100 steps, delta=1e-5")
print(f"  {'z':>6} {'epsilon':>9} | {'acc n=500 (mean +- sd, 5 seeds)':>32} | {'acc n=20,000':>15}")
for z in [0.0, 200.0, 50.0, 20.0, 8.0, 3.0]:
    eps = np.inf if z == 0 else zcdp_epsilon(z, 100, delta)
    row = []
    for n_train in [500, 20_000]:
        X, y = make_data(n_train, seed=1)
        accs = [accuracy(dp_gd(X, y, z, seed=s), X_test, y_test) for s in range(5)]
        row.append((np.mean(accs), np.std(accs)))
    eps_s = "inf" if z == 0 else f"{eps:.2f}"
    print(f"  {z:6.1f} {eps_s:>9} | {row[0][0]:22.3f} +- {row[0][1]:.3f}  | "
          f"{row[1][0]:7.3f} +- {row[1][1]:.3f}")

# The Bayes-optimal classifier for this data uses the true weights.
print(f"  accuracy of the true weights on the test set: {accuracy(W_TRUE, X_test, y_test):.3f}")
