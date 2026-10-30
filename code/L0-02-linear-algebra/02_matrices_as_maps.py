"""Matrices as linear maps, composition, and the cost of matrix multiplication."""
import time

import numpy as np

rng = np.random.default_rng(0)

# 1. A matrix is determined by where it sends the basis vectors: its columns.
theta = np.pi / 2
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])    # rotate 90 degrees
S = np.array([[3.0, 0.0],
              [0.0, 0.5]])                          # stretch x, squash y
H = np.array([[1.0, 1.0],
              [0.0, 1.0]])                          # shear
F = np.array([[1.0, 0.0],
              [0.0, -1.0]])                         # reflect across the x-axis
Pj = np.array([[1.0, 0.0],
               [0.0, 0.0]])                         # project onto the x-axis
e1, e2 = np.eye(2)
print("R e1 =", np.round(R @ e1, 3), "= first column of R;",
      "R e2 =", np.round(R @ e2, 3))

# 2. Composition: (R S) x means "apply S first, then R". Order matters.
x = np.array([1.0, 1.0])
print("\nR @ (S @ x) =", np.round(R @ (S @ x), 3), "| (R @ S) @ x =", np.round((R @ S) @ x, 3))
print("S @ R @ x   =", np.round(S @ R @ x, 3), "-> RS != SR:", not np.allclose(R @ S, S @ R))

# 3. Determinant = signed area scaling of the unit square.
#    Negative flips orientation; zero squashes the plane onto a line.
for name, M in [("R", R), ("S", S), ("H", H), ("F", F), ("Pj", Pj), ("S@F", S @ F)]:
    print(f"det({name:<3}) = {np.linalg.det(M):+.3f}")

# 4. Cost: an (m x k) @ (k x n) product needs m*n*k multiply-adds = 2mnk flops.
def gflops(n, dtype=np.float64, reps=10):
    A = rng.normal(size=(n, n)).astype(dtype)
    B = rng.normal(size=(n, n)).astype(dtype)
    for _ in range(3):
        A @ B                                # warm-up (thread pool start, caches)
    best = min(_time(lambda: A @ B) for _ in range(reps))   # best of reps
    return 2 * n**3 / best / 1e9, best


def _time(fn):
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


print(f"\n{'n':>6}{'time (ms)':>12}{'GFLOP/s f64':>14}{'GFLOP/s f32':>14}")
for n in (256, 1024, 2048):
    g64, t64 = gflops(n)
    g32, _ = gflops(n, np.float32)
    print(f"{n:>6}{t64*1e3:>12.2f}{g64:>14.1f}{g32:>14.1f}")

# BLAS uses many threads by default. Pin to one to see the per-core rate.
try:
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1, user_api="blas"):
        g1, t1 = gflops(1024)
    gall, tall = gflops(1024)
    print(f"n=1024 f64: 1 thread {g1:.1f} GFLOP/s ({t1*1e3:.0f} ms), "
          f"default threads {gall:.1f} GFLOP/s ({tall*1e3:.0f} ms)")
except ImportError:
    print("threadpoolctl not installed; skipping thread comparison")

# 5. Same math, different cost: associativity lets you choose the order.
n = 2000
A = rng.normal(size=(n, n))
B = rng.normal(size=(n, n))
v = rng.normal(size=n)
t_mm = min(_time(lambda: (A @ B) @ v) for _ in range(3))   # ~2n^3 flops
t_mv = min(_time(lambda: A @ (B @ v)) for _ in range(3))   # ~4n^2 flops
print(f"\n(A@B)@v: {t_mm*1e3:.1f} ms   A@(B@v): {t_mv*1e3:.2f} ms   "
      f"same result: {np.allclose((A @ B) @ v, A @ (B @ v))}")
