"""Two ways that looking at the test set many times manufactures progress.

1. Twenty prompt variants with exactly the same true accuracy: pick the best
   on the test set, then see what corrections and a fresh test set say.
2. A leaderboard attack in the style of Blum and Hardt (2015): random guesses,
   kept or dropped according to their public score, then combined.
"""
import numpy as np
from scipy.stats import norm

from evalkit import benjamini_hochberg, holm, paired_permutation_p

rng = np.random.default_rng(1)

# ---- 1. selecting the best of 20 equal variants ------------------------------
N_ITEMS, N_VARIANTS = 300, 20
p_item = 1 / (1 + np.exp(-rng.normal(1.0, 2.0, size=N_ITEMS)))   # true per-item accuracy


def run_variants(p, rng):
    """Variant 0 is the baseline prompt. Every variant agrees with the baseline
    on a random 70% of items and is redrawn on the rest: correlated results,
    identical expected accuracy."""
    base = rng.random(len(p)) < p
    out = [base]
    for _ in range(N_VARIANTS):
        redraw = rng.random(len(p)) < 0.3
        out.append(np.where(redraw, rng.random(len(p)) < p, base))
    return np.array(out, float)


res = run_variants(p_item, rng)
acc = res.mean(1)
pvals = np.array([paired_permutation_p(res[v], res[0], n_perm=4000, seed=v)
                  for v in range(1, N_VARIANTS + 1)])
best = 1 + int(np.argmax(acc[1:]))
print(f"1. true accuracy of every variant: {p_item.mean():.3f}")
print(f"   baseline on test set {acc[0]:.3f}; best of {N_VARIANTS} variants (#{best}) {acc[best]:.3f}")
print(f"   best variant p-value {pvals[best - 1]:.4f}; smallest p {pvals.min():.4f}; "
      f"variants with p < 0.05: {(pvals < 0.05).sum()}")
print(f"   after Holm: {(holm(pvals) < 0.05).sum()} significant; "
      f"after Benjamini-Hochberg: {(benjamini_hochberg(pvals) < 0.05).sum()}")

# Fresh items from the same distribution, and fresh runs of every variant.
fresh = run_variants(1 / (1 + np.exp(-rng.normal(1.0, 2.0, size=N_ITEMS))), rng)
print(f"   on a fresh test set: baseline {fresh[0].mean():.3f}, chosen variant {fresh[best].mean():.3f}")

# How big is the winner's curse on average? Repeat the whole selection 500 times.
# (A paired z-test, sum(d) / sqrt(sum(d^2)), stands in for the slower permutation test.)
gains, any_raw, any_holm = [], 0, 0
for _ in range(500):
    r = run_variants(p_item, rng)
    a = r.mean(1)
    gains.append(a[1:].max() - a[0])
    d = r[1:] - r[0]
    z = d.sum(1) / np.sqrt((d ** 2).sum(1)).clip(min=1)
    p = 2 * norm.sf(np.abs(z))
    any_raw += (p < 0.05).any()
    any_holm += (holm(p) < 0.05).any()
print(f"   over 500 repeats, best-minus-baseline averaged {np.mean(gains):+.3f} "
      f"(true gain 0); 90th percentile {np.quantile(gains, 0.9):+.3f}")
print(f"   repeats with at least one 'significant' variant: uncorrected {any_raw / 500:.3f}, "
      f"Holm {any_holm / 500:.3f}")

# ---- 2. climbing a public leaderboard with no skill ---------------------------
N_TEST, N_SUB = 4000, 400
labels = rng.integers(0, 2, N_TEST)
public = np.arange(N_TEST) < N_TEST // 2            # half the test set is scored publicly
subs = rng.integers(0, 2, (N_SUB, N_TEST))          # random guesses: true accuracy 0.5
pub_score = (subs[:, public] == labels[public]).mean(1)
keep = subs[pub_score > 0.5]                        # keep guesses that "did well"
combined = (keep.mean(0) > 0.5).astype(int)         # majority vote of the keepers
print(f"\n2. {N_SUB} random submissions, kept {len(keep)} that scored above 0.5 publicly")
print(f"   best single submission: public {pub_score.max():.3f}")
print(f"   majority vote of kept:  public {(combined[public] == labels[public]).mean():.3f}, "
      f"private {(combined[~public] == labels[~public]).mean():.3f}")
