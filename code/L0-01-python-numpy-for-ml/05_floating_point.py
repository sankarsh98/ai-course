"""Floating-point pitfalls: rounding, overflow, underflow, cancellation, log-sum-exp."""
import math

import numpy as np
from scipy.special import logsumexp as scipy_lse

np.seterr(all="ignore")   # we want to SEE inf/nan here, not warnings

# 1. Decimal fractions are not exact in binary.
print("0.1 + 0.2 == 0.3 ?", 0.1 + 0.2 == 0.3, "| value:", repr(0.1 + 0.2))
print("np.isclose(0.1 + 0.2, 0.3):", np.isclose(0.1 + 0.2, 0.3))

# 2. Overflow and underflow of exp in float64 and float32.
for dt in (np.float64, np.float32):
    print(f"{dt.__name__}: exp(710)={np.exp(dt(710))}, exp(89)={np.exp(dt(89))}, "
          f"exp(-746)={np.exp(dt(-746))}, exp(-104)={np.exp(dt(-104))}")


# 3. Naive vs shifted log-sum-exp and softmax.
def lse_naive(z):
    return np.log(np.sum(np.exp(z)))


def lse_stable(z):
    m = np.max(z)
    return m + np.log(np.sum(np.exp(z - m)))   # every exponent is <= 0


def softmax_naive(z):
    e = np.exp(z)
    return e / e.sum()


def softmax_stable(z):
    e = np.exp(z - np.max(z))
    return e / e.sum()


for z in (np.array([1.0, 2.0, 3.0]),
          np.array([1000.0, 1001.0, 1002.0]),
          np.array([-1000.0, -1001.0, -1002.0])):
    print(f"\nz={z}")
    print(f"  naive lse={lse_naive(z)!s:<20} stable lse={lse_stable(z):.6f}  "
          f"scipy={scipy_lse(z):.6f}")
    print(f"  naive softmax={softmax_naive(z)}  stable={np.round(softmax_stable(z), 6)}")

# 4. Cross-entropy from logits: never take log of a softmax that rounded to 0.
logits = np.array([50.0, 0.0, -50.0])
true_class = 2
p = softmax_stable(logits)
print(f"\nlog(softmax)[2] = {np.log(p[true_class])}  (p rounds to {p[true_class]:.3e})")
print(f"log-softmax via lse = {logits[true_class] - lse_stable(logits):.4f}")
p32 = softmax_stable(logits.astype(np.float32))
print(f"same in float32: log(p) = {np.log(p32[true_class])}")

# 5. Catastrophic cancellation: the one-pass variance formula E[x^2] - E[x]^2.
rng = np.random.default_rng(0)
x = 1e8 + rng.normal(size=10_000)      # true variance ~1, huge mean
one_pass = np.mean(x**2) - np.mean(x) ** 2
two_pass = np.mean((x - x.mean()) ** 2)
print(f"\nvariance one-pass={one_pass:.4f}  two-pass={two_pass:.4f}  np.var={np.var(x):.4f}")

# 6. Accumulating many small numbers in float32.
#    np.sum uses pairwise summation, so its error grows slowly; a running
#    total in float32 stalls once it is so large that adding 0.1 rounds away.
vals = np.full(10_000_000, 0.1, dtype=np.float32)
loop32 = np.float32(0.0)
for v in vals[:2_000_000]:             # strictly one-at-a-time, 2e6 terms
    loop32 += v
exact = 10_000_000 * float(np.float32(0.1))   # float32(0.1) is 0.10000000149...
print(f"\nsum of 1e7 copies of float32(0.1), exact {exact:.3f}:")
print(f"  float32 np.sum={vals.sum():.3f}  float64 accumulate="
      f"{vals.sum(dtype=np.float64):.3f}  math.fsum={math.fsum(vals):.3f}")
print(f"one-at-a-time float32 running total over 2e6 terms "
      f"(exact {exact / 5:.3f}): {loop32:.3f}")

# 7. NaN propagates, and NaN != NaN.
y = np.array([1.0, np.nan, 3.0])
print(f"\nmean with nan={y.mean()}, nanmean={np.nanmean(y)}, nan==nan: {np.nan == np.nan}")
