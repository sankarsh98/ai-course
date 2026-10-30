"""Monte Carlo estimation: error shrinks like 1/sqrt(N), and rare events need importance sampling."""
import numpy as np
from scipy import stats

rng = np.random.default_rng(0)

# ---- 1. Estimate E[g(X)] for X ~ Uniform(0,1), g(x) = 4 / (1 + x^2). The exact answer is pi. ----
print("Estimating pi = E[4/(1+X^2)], X ~ Uniform(0,1); 200 independent runs per N")
print(f"{'N':>9} {'RMS error':>16} {'error * sqrt(N)':>16} {'reported std err':>17}")
for N in [100, 1_000, 10_000, 100_000]:
    errors, ses = [], []
    for _ in range(200):
        g = 4 / (1 + rng.uniform(size=N) ** 2)
        errors.append(g.mean() - np.pi)
        ses.append(g.std(ddof=1) / np.sqrt(N))  # what one run can report about itself
    rmse = np.sqrt(np.mean(np.square(errors)))
    print(f"{N:>9,} {rmse:16.6f} {rmse * np.sqrt(N):16.4f} {np.mean(ses):17.6f}")

# ---- 2. A rare event: P(Z > 4) for Z ~ N(0,1). ----
exact = stats.norm.sf(4)
N = 100_000
z = rng.normal(size=N)
naive = np.mean(z > 4)
print(f"\nP(Z > 4): exact {exact:.3e}")
print(f"  naive MC, N={N:,}: {naive:.3e} ({int(np.sum(z > 4))} of {N:,} samples landed in the region)")

# Importance sampling: draw from N(4, 1), which puts half its mass in the region,
# and reweight each sample by p(y)/q(y) so the estimate stays unbiased.
y = rng.normal(loc=4, size=N)
w = stats.norm.pdf(y) / stats.norm.pdf(y, loc=4)
vals = (y > 4) * w
est, se = vals.mean(), vals.std(ddof=1) / np.sqrt(N)
print(f"  importance sampling, N={N:,}: {est:.3e} +/- {se:.1e} (relative error {abs(est - exact) / exact:.2%})")

# ---- 3. A bad proposal: importance weights with huge variance. ----
# Target p = N(0, 1), proposal q = N(0, 0.5^2), which is too narrow and misses the tails.
y = rng.normal(scale=0.5, size=N)
w = stats.norm.pdf(y) / stats.norm.pdf(y, scale=0.5)
vals = y ** 2 * w  # estimates E_p[X^2] = 1
ess = w.sum() ** 2 / np.sum(w ** 2)  # effective sample size
print(f"\nE[X^2] under N(0,1) with too-narrow proposal N(0, 0.5^2), N={N:,}:")
print(f"  estimate {vals.mean():.3f} +/- {vals.std(ddof=1) / np.sqrt(N):.3f} (truth 1.000), "
      f"effective sample size {ess:,.0f}")
print(f"  largest single weight {w.max():.1f}; top 10 samples carry {np.sort(vals)[-10:].sum() / vals.sum():.1%} of the sum")
