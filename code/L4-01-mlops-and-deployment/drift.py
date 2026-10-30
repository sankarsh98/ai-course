"""How PSI and the two-sample KS test behave, on synthetic data.

1. With no drift at all: how often does each rule raise an alarm, by sample size?
2. With a tiny shift (0.05 standard deviations) and a large one (0.5):
   which rule notices, and does noticing mean it matters?
3. Many features: the chance of at least one false alarm per check.
Each cell is the fraction of 200 repetitions that alarm.
"""
import numpy as np
from scipy import stats

from pipeline import psi

rng = np.random.default_rng(0)
REPS, BINS = 200, 10


def alarm_rates(n, shift):
    psi_vals, ks_hits = [], 0
    for _ in range(REPS):
        ref = rng.normal(0, 1, n)
        cur = rng.normal(shift, 1, n)
        psi_vals.append(psi(ref, cur, bins=BINS))
        ks_hits += stats.ks_2samp(ref, cur).pvalue < 0.05
    v = np.array(psi_vals)
    return v.mean(), (v > 0.1).mean(), (v > 0.25).mean(), ks_hits / REPS


print("reference and current windows both of size n; alarm rate over 200 repetitions")
for shift in (0.0, 0.05, 0.5):
    print(f"\nshift = {shift} standard deviations")
    print(f"  {'n':>7} {'mean PSI':>9} {'PSI>0.10':>9} {'PSI>0.25':>9} {'KS p<0.05':>10}"
          f" {'(B-1)(2/n)':>11}")
    for n in (100, 1_000, 10_000, 100_000):
        m, a10, a25, ks = alarm_rates(n, shift)
        print(f"  {n:>7} {m:>9.4f} {a10:>9.0%} {a25:>9.0%} {ks:>10.0%} {(BINS - 1) * 2 / n:>11.4f}")

print("\n20 independent features, no drift, n=5000, one check")
hits = 0
for _ in range(REPS):
    p = [stats.ks_2samp(rng.normal(size=5000), rng.normal(size=5000)).pvalue for _ in range(20)]
    hits += min(p) < 0.05
print(f"  at least one KS p<0.05: {hits / REPS:.0%}   (theory 1 - 0.95^20 = {1 - 0.95 ** 20:.0%});"
      f" Bonferroni threshold 0.05/20 would give about 5%")
