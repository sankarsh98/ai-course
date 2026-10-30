"""Measure loops vs vectorized NumPy on three typical ML kernels.

Timings depend on your CPU, BLAS library, and machine load. Ratios are more
portable than absolute numbers, and even ratios vary.
"""
import timeit

import numpy as np

rng = np.random.default_rng(0)


def best_time(fn, repeat=5, number=1):
    """Best of `repeat` runs, in seconds per call (timeit docs recommend the min)."""
    return min(timeit.repeat(fn, repeat=repeat, number=number)) / number


# Kernel 1: sum of squares (a reduction).
x = rng.normal(size=1_000_000)
x_list = x.tolist()


def sumsq_loop():
    s = 0.0
    for v in x_list:
        s += v * v
    return s


def sumsq_numpy():
    return np.dot(x, x)


# Kernel 2: ReLU applied elementwise.
def relu_loop():
    return [v if v > 0 else 0.0 for v in x_list]


def relu_numpy():
    return np.maximum(x, 0.0)


# Kernel 3: pairwise squared distances, 300 x 300 points in 64 dimensions.
P = rng.normal(size=(300, 64))
Q = rng.normal(size=(300, 64))


def dist_loop():
    n, m = len(P), len(Q)
    out = np.empty((n, m))
    for i in range(n):
        for j in range(m):
            diff = P[i] - Q[j]           # still uses NumPy for the inner vector
            out[i, j] = diff @ diff
    return out


def dist_numpy():
    return (P**2).sum(1)[:, None] - 2 * P @ Q.T + (Q**2).sum(1)[None, :]


# Check the fast versions give the same answers before timing anything.
assert np.isclose(sumsq_loop(), sumsq_numpy())
assert np.allclose(relu_loop(), relu_numpy())
assert np.allclose(dist_loop(), dist_numpy())

print(f"{'kernel':<28}{'loop (ms)':>12}{'numpy (ms)':>12}{'speedup':>10}")
for name, slow, fast, rep in [
    ("sum of squares, n=1e6", sumsq_loop, sumsq_numpy, 3),
    ("ReLU, n=1e6", relu_loop, relu_numpy, 3),
    ("pairwise dist 300x300x64", dist_loop, dist_numpy, 3),
]:
    t_slow = best_time(slow, repeat=rep)
    t_fast = best_time(fast, repeat=20)
    print(f"{name:<28}{t_slow*1e3:>12.2f}{t_fast*1e3:>12.3f}{t_slow/t_fast:>10.0f}x")

# Vectorization is not free: a temporary array costs memory. The broadcast
# version of pairwise distance materializes an (n, m, d) array.
n, m, d = 300, 300, 64
print(f"\nbroadcast temp (n,m,d) float64 = {n*m*d*8/2**20:.0f} MiB; "
      f"expanded form temp (n,m) = {n*m*8/2**20:.1f} MiB")
