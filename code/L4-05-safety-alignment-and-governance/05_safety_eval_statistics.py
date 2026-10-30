"""The statistics of safety evaluations (educational demo, simulated numbers only).

1. Zero failures in n red-team attempts: what upper bound does that support?
2. Repeated attempts: per-attempt success rates that vary across prompts give
   attack success that keeps rising with the number of attempts N.
3. Is model B safer than model A? Confidence intervals on small eval sets.
4. Hallucination and grading: binary grading rewards guessing; a penalty for
   wrong answers makes "I don't know" the better choice when unsure.
"""
import numpy as np
from scipy import stats

rng = np.random.default_rng(0)

# ---------------------------------------------------------------- 1
print("1. Observed 0 harmful outputs in n attempts: 95% upper bound on the true rate")
print(f"  {'n':>6} {'rule of three 3/n':>18} {'exact one-sided bound':>24}")
for n in [30, 100, 1000, 10000]:
    exact = 1 - 0.05 ** (1 / n)  # solves (1 - p)^n = 0.05 (one-sided Clopper-Pearson)
    print(f"  {n:6d} {3 / n:18.4f} {exact:24.4f}")

# ---------------------------------------------------------------- 2
# 1,000 harmful-request prompts. For each, the chance that one randomly
# perturbed attempt gets a harmful answer is p_i ~ Beta(0.3, 30): mean 1%,
# but most prompts are much harder than that and a few are easy.
p = rng.beta(0.3, 30, size=1000)
print("\n2. Attack success rate (ASR) when the attacker may try N times per prompt")
print(f"  mean per-attempt success = {p.mean():.4f}")
print(f"  {'N':>6} {'ASR (exact)':>12} {'ASR (simulated)':>16} {'never broken':>13}")
for N in [1, 10, 100, 1000, 10000]:
    exact = 1 - np.mean((1 - p) ** N)
    sim = np.mean(rng.binomial(N, p) > 0)
    print(f"  {N:6d} {exact:12.3f} {sim:16.3f} {1 - exact:13.3f}")
Ns = np.array([100, 1000, 10000])
never = np.array([np.mean((1 - p) ** n) for n in Ns])
slope = np.polyfit(np.log(Ns), np.log(never), 1)[0]
print(f"  log-log slope of 'never broken' vs N (100..10000): {slope:.2f} "
      f"(a power law with exponent near the Beta's first parameter, 0.3)")

# ---------------------------------------------------------------- 3
def wilson(k, n, z=1.96):
    ph = k / n
    c = (ph + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


print("\n3. Model A fails on 12 of 200 red-team prompts, model B on 6 of 200")
for name, k in [("A", 12), ("B", 6)]:
    lo, hi = wilson(k, 200)
    print(f"  {name}: rate {k / 200:.3f}, 95% Wilson CI [{lo:.3f}, {hi:.3f}]")
# Same 200 prompts for both models -> paired test on discordant prompts (McNemar).
# Suppose B fails on a subset of A's failures: 6 prompts only A fails, 0 only B fails.
for only_a, only_b in [(6, 0), (9, 3)]:
    pval = stats.binomtest(only_a, only_a + only_b, 0.5).pvalue
    print(f"  paired: {only_a} prompts fail only on A, {only_b} only on B -> "
          f"exact McNemar p = {pval:.3f}")
# How many prompts to detect 4% vs 2% (two-sided alpha 0.05, power 0.8, unpaired)?
z_a, z_b = stats.norm.ppf(0.975), stats.norm.ppf(0.8)
p1, p2 = 0.04, 0.02
pbar = (p1 + p2) / 2
n_req = ((z_a * np.sqrt(2 * pbar * (1 - pbar)) + z_b * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2)))
         / (p1 - p2)) ** 2
print(f"  prompts per model to detect 4% vs 2% (unpaired, power 0.8): {int(np.ceil(n_req))}")

# ---------------------------------------------------------------- 4
# 10,000 questions. The model's confidence c that its best answer is right is
# calibrated (it is right with probability c). It can answer or abstain.
c = rng.beta(2, 1.2, size=10_000)
correct = rng.random(10_000) < c
print("\n4. Grading and hallucination (calibrated model, 10,000 questions)")
print(f"  {'policy':28} {'answered':>8} {'wrong':>6} {'binary score':>13} "
      f"{'score, wrong=-1':>16} {'score, wrong=-3':>16}")
for t in [0.0, 0.5, 0.75, 0.9]:
    ans = c >= t
    wrong = ans & ~correct
    right = ans & correct
    binary = right.mean()                    # right = 1, wrong = 0, abstain = 0
    pen1 = right.mean() - 1 * wrong.mean()   # right = 1, wrong = -1, abstain = 0
    pen3 = right.mean() - 3 * wrong.mean()   # right = 1, wrong = -3
    label = "always answer" if t == 0 else f"answer if confidence >= {t}"
    print(f"  {label:28} {ans.mean():8.3f} {wrong.mean():6.3f} {binary:13.3f} "
          f"{pen1:16.3f} {pen3:16.3f}")
print("  With penalty k for a wrong answer, answering pays iff c > k / (1 + k): "
      "0.5 for k=1, 0.75 for k=3.")
