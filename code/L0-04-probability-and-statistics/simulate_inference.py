"""Simulations of what confidence intervals and p-values do and do not promise."""
import numpy as np

from intervals import two_proportion_p, wald, wilson

rng = np.random.default_rng(0)
TRIALS = 20_000

# 1. Coverage: how often does a "95%" interval contain the true accuracy?
print(f"Coverage of nominal 95% intervals ({TRIALS:,} simulated test sets per row)")
print(f"{'true acc':>8} {'n':>6} {'Wald':>7} {'Wilson':>7}")
for p_true in [0.80, 0.95, 0.99]:
    for n in [50, 200, 1000]:
        k = rng.binomial(n, p_true, size=TRIALS)
        cover = [np.mean((lo <= p_true) & (p_true <= hi)) for lo, hi in (wald(k, n), wilson(k, n))]
        print(f"{p_true:8.2f} {n:6d} {cover[0]:7.3f} {cover[1]:7.3f}")

# 2. When the null is true (both models 0.85 accurate), p-values are uniform.
p = two_proportion_p(rng.binomial(500, 0.85, TRIALS), rng.binomial(500, 0.85, TRIALS), 500)
print(f"\nNull true: fraction p < 0.05 = {np.mean(p < 0.05):.3f}, p < 0.01 = {np.mean(p < 0.01):.3f}")
print("  p-value deciles:", np.round(np.quantile(p, np.linspace(0.1, 0.9, 9)), 2))

# 3. Twenty variants, none truly better than the baseline: report the best one.
hits = [np.any(two_proportion_p(rng.binomial(500, 0.85), rng.binomial(500, 0.85, 20), 500) < 0.05)
        for _ in range(2_000)]
print(f"\n20 equally good variants: P(at least one p < 0.05) = {np.mean(hits):.3f}")

# 4. Power: how large a test set is needed to detect 0.85 -> 0.87?
print("\nPower to detect 0.85 -> 0.87 (independent test sets, alpha = 0.05)")
for n in [500, 2000, 5000]:
    p = two_proportion_p(rng.binomial(n, 0.85, 5000), rng.binomial(n, 0.87, 5000), n)
    print(f"  n = {n:>5} per model: power = {np.mean(p < 0.05):.3f}")
