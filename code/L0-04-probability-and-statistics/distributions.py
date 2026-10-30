"""Common distributions: check sample moments against theory, then watch the CLT work (and fail)."""
import numpy as np
from scipy import stats

rng = np.random.default_rng(0)
N = 200_000

# (name, sampler, theoretical mean, theoretical variance)
cases = [
    ("Bernoulli(p=0.3)", lambda: rng.binomial(1, 0.3, N), 0.3, 0.3 * 0.7),
    ("Binomial(n=20, p=0.3)", lambda: rng.binomial(20, 0.3, N), 20 * 0.3, 20 * 0.3 * 0.7),
    ("Poisson(lam=4)", lambda: rng.poisson(4, N), 4.0, 4.0),
    ("Uniform(0, 1)", lambda: rng.uniform(0, 1, N), 0.5, 1 / 12),
    ("Normal(mu=2, sigma=3)", lambda: rng.normal(2, 3, N), 2.0, 9.0),
    ("Exponential(rate=0.5)", lambda: rng.exponential(1 / 0.5, N), 2.0, 4.0),
]
print(f"{'distribution':<24} {'mean':>8} {'theory':>8} {'var':>8} {'theory':>8}")
for name, sample, mu, var in cases:
    x = sample()
    print(f"{name:<24} {x.mean():8.4f} {mu:8.4f} {x.var():8.4f} {var:8.4f}")

# Central limit theorem: the mean of n Exponential(1) draws.
# Exponential(1) has skewness 2, so the mean of n draws has skewness 2/sqrt(n).
print("\nMean of n Exponential(1) draws (50,000 repetitions each)")
print(f"{'n':>5} {'mean':>8} {'std':>8} {'1/sqrt(n)':>9} {'skew':>7} {'2/sqrt(n)':>9} {'P(z<-1.96)':>11} {'P(z>1.96)':>10}")
for n in [1, 2, 5, 30, 200]:
    means = rng.exponential(1.0, size=(50_000, n)).mean(axis=1)
    z = (means - 1.0) / (1.0 / np.sqrt(n))  # standardize with the true mean and std
    lo, hi = np.mean(z < -1.96), np.mean(z > 1.96)  # each 0.025 if exactly normal
    print(f"{n:>5} {means.mean():8.4f} {means.std():8.4f} {1 / np.sqrt(n):9.4f} "
          f"{stats.skew(means):7.3f} {2 / np.sqrt(n):9.3f} {lo:11.4f} {hi:10.4f}")

# A distribution with no mean: the running average of Cauchy draws never settles.
print("\nRunning mean of standard Cauchy draws (no finite mean, so no LLN or CLT)")
c = rng.standard_cauchy(1_000_000)
g = rng.normal(size=1_000_000)
for k in [1_000, 10_000, 100_000, 1_000_000]:
    print(f"  first {k:>9,} draws: Cauchy mean {c[:k].mean():9.3f}   Normal mean {g[:k].mean():8.4f}")
