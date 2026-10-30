"""Bayes' rule, maximum likelihood, and MAP on small examples you can check by hand."""
import numpy as np
from scipy import stats

rng = np.random.default_rng(0)

# ---- 1. Bayes' rule: a rare condition and a good test ----
prior = 0.01          # P(condition)
sensitivity = 0.99    # P(positive | condition)
false_pos = 0.05      # P(positive | no condition)
evidence = sensitivity * prior + false_pos * (1 - prior)  # P(positive), law of total probability
posterior = sensitivity * prior / evidence
print(f"P(condition | positive) by Bayes' rule: {posterior:.4f}")

# Same answer by brute-force simulation of one million people.
n = 1_000_000
has = rng.random(n) < prior
positive = np.where(has, rng.random(n) < sensitivity, rng.random(n) < false_pos)
print(f"P(condition | positive) by simulation:  {has[positive].mean():.4f} "
      f"({positive.sum():,} positives, {has[positive].sum():,} true)")

# ---- 2. Coin bias: MLE vs MAP vs full posterior with a Beta(a, b) prior ----
a, b = 2.0, 2.0  # prior pseudo-counts: a gentle pull toward 0.5
print(f"\nCoin bias with Beta({a:.0f},{b:.0f}) prior")
print(f"{'heads/flips':>12} {'MLE':>7} {'MAP':>7} {'post. mean':>10} {'95% credible interval':>24}")
for heads, flips in [(3, 3), (7, 10), (70, 100), (700, 1000)]:
    mle = heads / flips
    map_ = (heads + a - 1) / (flips + a + b - 2)  # mode of Beta(heads+a, tails+b)
    post = stats.beta(heads + a, flips - heads + b)
    lo, hi = post.ppf([0.025, 0.975])
    print(f"{heads:>5}/{flips:<6} {mle:7.3f} {map_:7.3f} {post.mean():10.3f}      [{lo:.3f}, {hi:.3f}]")

# ---- 3. The Gaussian MLE of variance is biased low for small samples ----
sigma2, n_small = 4.0, 5
samples = rng.normal(0, np.sqrt(sigma2), size=(200_000, n_small))
mle_var = samples.var(axis=1, ddof=0)      # divide by n: the MLE
unbiased = samples.var(axis=1, ddof=1)     # divide by n-1
print(f"\nTrue variance {sigma2}, n={n_small}: average MLE = {mle_var.mean():.3f} "
      f"(theory {sigma2 * (n_small - 1) / n_small:.3f}), average with n-1 = {unbiased.mean():.3f}")
