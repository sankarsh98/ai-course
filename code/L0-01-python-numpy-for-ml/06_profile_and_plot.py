"""Profile a small pipeline with cProfile, then plot loop vs NumPy scaling.

Run from the repository root so the plot lands in docs/assets/.
"""
import cProfile
import io
import pstats
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                   # no display needed; write a file
import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(0)
OUT = Path("docs/assets/L0-01-python-numpy-for-ml")


# A toy "feature pipeline" with one deliberately slow step.
def load(n):
    return rng.normal(size=(n, 20))


def standardize(X):
    return (X - X.mean(0)) / X.std(0)


def clip_outliers_slow(X, k=3.0):
    out = X.copy()
    for i in range(out.shape[0]):       # Python loop over rows: the bottleneck
        for j in range(out.shape[1]):
            if out[i, j] > k:
                out[i, j] = k
            elif out[i, j] < -k:
                out[i, j] = -k
    return out


def clip_outliers_fast(X, k=3.0):
    return np.clip(X, -k, k)


def pipeline(clip):
    X = load(50_000)
    Z = standardize(X)
    return clip(Z).sum()


def profile(fn, *args):
    pr = cProfile.Profile()
    pr.enable()
    fn(*args)
    pr.disable()
    s = io.StringIO()
    # Keep only our own functions in the report so it stays readable.
    st = pstats.Stats(pr, stream=s).strip_dirs().sort_stats("cumulative")
    st.print_stats(r"06_profile_and_plot")
    lines = [ln for ln in s.getvalue().splitlines() if ln.strip()]
    print("\n".join(ln for ln in lines if "List reduced" not in ln))


print("=== profile: slow clip ===")
profile(pipeline, clip_outliers_slow)
print("\n=== profile: fast clip ===")
profile(pipeline, clip_outliers_fast)
Xc = rng.normal(size=(1000, 20)) * 2
assert np.array_equal(clip_outliers_slow(Xc), clip_outliers_fast(Xc))  # same result


# Scaling plot: time per call vs n for loop and NumPy ReLU.
def timed(fn, *args, repeat=3):
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - t0)
    return best


sizes = np.logspace(1, 6, 6).astype(int)      # 10, 100, ..., 1,000,000
t_loop, t_np = [], []
for n in sizes:
    v = rng.normal(size=n)
    lst = v.tolist()
    t_loop.append(timed(lambda: [a if a > 0 else 0.0 for a in lst]))
    t_np.append(timed(lambda: np.maximum(v, 0.0)))

for n, a, b in zip(sizes, t_loop, t_np):
    print(f"n={n:>8}  loop={a*1e6:>10.1f} us  numpy={b*1e6:>8.1f} us  ratio={a/b:>6.1f}")

OUT.mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
ax.loglog(sizes, t_loop, "o-", label="Python list comprehension")
ax.loglog(sizes, t_np, "s-", label="np.maximum (vectorized)")
ax.set_xlabel("number of elements n")
ax.set_ylabel("best time per call (s)")
ax.set_title("ReLU: loop vs vectorized")
ax.grid(True, which="major", alpha=0.3)
ax.legend()
fig.savefig(OUT / "relu_scaling.png", dpi=120)
print(f"saved {OUT / 'relu_scaling.png'}")
