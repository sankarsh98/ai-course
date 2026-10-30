"""Gradient descent and momentum on a quadratic bowl.

f(w) = 0.5 * w^T A w, with A diagonal so each coordinate is one eigen-direction.
Part 1 shows the learning-rate regimes; part 2 shows how conditioning sets speed.
"""
import numpy as np


def run(A, w0, lr, steps, beta=0.0):
    """Heavy-ball momentum (beta=0 gives plain gradient descent). Returns losses."""
    w, v = w0.copy(), np.zeros_like(w0)
    losses = []
    for _ in range(steps):
        v = beta * v - lr * (A @ w)   # velocity accumulates past gradients
        w = w + v
        losses.append(0.5 * w @ A @ w)
    return np.array(losses)


def steps_to_tol(A, w0, lr, beta=0.0, tol=1e-10, max_steps=200_000):
    """Number of steps until f(w) < tol * f(w0)."""
    w, v = w0.copy(), np.zeros_like(w0)
    f0 = 0.5 * w0 @ A @ w0
    for t in range(1, max_steps + 1):
        v = beta * v - lr * (A @ w)
        w = w + v
        if 0.5 * w @ A @ w < tol * f0:
            return t
    return None


w0 = np.array([1.0, 1.0])

print("Part 1: learning-rate regimes, eigenvalues (1, 10), so L = 10, 2/L = 0.2")
A = np.diag([1.0, 10.0])
print(f"{'lr':>6} {'|1-lr*1|':>9} {'|1-lr*10|':>10} {'loss@5':>10} {'loss@50':>10}")
for lr in [0.01, 0.1, 0.18, 0.2, 0.21]:
    losses = run(A, w0, lr, 50)
    print(f"{lr:6.2f} {abs(1 - lr):9.2f} {abs(1 - 10 * lr):10.2f} "
          f"{losses[4]:10.3e} {losses[49]:10.3e}")

print("\nPart 2: steps to reduce loss by 1e10, eigenvalues (1, kappa)")
print(f"{'kappa':>6} {'GD steps':>9} {'GD theory':>10} {'momentum':>9} {'mom theory':>11}")
for kappa in [1, 10, 100, 1000, 10000]:
    A = np.diag([1.0, float(kappa)])
    mu, L = 1.0, float(kappa)
    # Best fixed step for GD, and the optimal heavy-ball parameters.
    lr_gd = 2.0 / (L + mu)
    rate_gd = (kappa - 1) / (kappa + 1)
    lr_hb = 4.0 / (np.sqrt(L) + np.sqrt(mu)) ** 2
    rate_hb = (np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)
    beta = rate_hb ** 2
    n_gd = steps_to_tol(A, w0, lr_gd)
    n_hb = steps_to_tol(A, w0, lr_hb, beta)
    # Loss is quadratic in the error, so it shrinks like rate^(2t).
    th_gd = np.log(1e-10) / (2 * np.log(rate_gd)) if kappa > 1 else 1
    th_hb = np.log(1e-10) / (2 * np.log(rate_hb)) if kappa > 1 else 1
    print(f"{kappa:6d} {n_gd:9d} {th_gd:10.0f} {n_hb:9d} {th_hb:11.0f}")
