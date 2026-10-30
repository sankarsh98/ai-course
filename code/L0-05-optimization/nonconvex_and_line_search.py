"""Saddle points, multiple minima, and backtracking line search."""
import numpy as np

rng = np.random.default_rng(0)

# Part 1: a saddle. f(x, y) = x^2/2 + y^4/4 - y^2/2.
# Origin: gradient zero, Hessian eigenvalues +1 (x) and -1 (y), so it is a saddle.
# Minima at (0, +1) and (0, -1).
def saddle_grad(p):
    x, y = p
    return np.array([x, y ** 3 - y])

def escape_steps(y0, lr=0.1, noise=0.0, max_steps=10_000):
    p = np.array([1.0, y0])
    for t in range(1, max_steps + 1):
        p = p - lr * (saddle_grad(p) + noise * rng.normal(size=2))
        if abs(p[1]) > 0.5:          # left the saddle region along y
            return t
    return None

print("Part 1: GD near a saddle, start (1, y0), lr 0.1")
for y0 in [1e-1, 1e-3, 1e-6, 1e-9, 0.0]:
    n = escape_steps(y0)
    print(f"  y0 = {y0:7.0e}: steps to |y| > 0.5 = {n if n else 'never (10,000 steps)'}")
n = escape_steps(0.0, noise=0.01)
print(f"  y0 = 0 with gradient noise std 0.01: steps = {n}")

# Part 2: a 1-D non-convex function with several local minima.
f1 = lambda w: 0.1 * w ** 2 + np.sin(3 * w)
g1 = lambda w: 0.2 * w + 3 * np.cos(3 * w)
grid = np.linspace(-10, 10, 200_001)
print(f"\nPart 2: f(w) = 0.1 w^2 + sin(3w); global min f = {f1(grid).min():.4f} "
      f"at w = {grid[f1(grid).argmin()]:.3f}")
ends = []
for w in rng.uniform(-8, 8, size=20):
    for _ in range(2000):
        w -= 0.05 * g1(w)
    ends.append(round(float(w), 2))
values, counts = np.unique(ends, return_counts=True)
for v, c in zip(values, counts):
    print(f"  ended at w = {v:6.2f} (f = {f1(v):7.4f}): {c} of 20 starts")

# Part 3: backtracking (Armijo) line search vs a fixed step on Rosenbrock.
f = lambda w: (1 - w[0]) ** 2 + 100 * (w[1] - w[0] ** 2) ** 2
g = lambda w: np.array([-2 * (1 - w[0]) - 400 * w[0] * (w[1] - w[0] ** 2),
                        200 * (w[1] - w[0] ** 2)])

def gd_fixed(lr, iters):
    w = np.array([-1.5, 2.0])
    for i in range(iters):
        w = w - lr * g(w)
        if np.abs(w).max() > 1e3:          # blew up: stop and report it
            return np.inf, i + 1
    return f(w), iters                     # one gradient evaluation per step

def gd_backtracking(iters, t0=1.0, c=1e-4, shrink=0.5):
    w, evals = np.array([-1.5, 2.0]), 0
    for _ in range(iters):
        grad, fw, t = g(w), f(w), t0
        # Shrink t until the step gives "sufficient decrease" (Armijo condition).
        while f(w - t * grad) > fw - c * t * grad @ grad:
            t *= shrink
            evals += 1
        w = w - t * grad
        evals += 1
    return f(w), evals

print("\nPart 3: Rosenbrock from (-1.5, 2), 5000 iterations of steepest descent")
for lr in [1e-4, 1e-3, 2e-3, 5e-3]:
    val, ev = gd_fixed(lr, 5000)
    if np.isinf(val):
        print(f"  fixed lr {lr:.0e}: diverged after {ev} steps")
    else:
        print(f"  fixed lr {lr:.0e}: f = {val:.3e}, function evals = {ev}")
val, ev = gd_backtracking(5000)
print(f"  backtracking:   f = {val:.3e}, function evals = {ev}")
