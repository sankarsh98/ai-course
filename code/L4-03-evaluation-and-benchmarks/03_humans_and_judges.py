"""Annotator agreement, and the biases of a (simulated) LLM judge.

Nothing here calls a real model. The judge is a formula with known position
and length biases, so we can check whether standard diagnostics find them.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression

from evalkit import bootstrap_ci, cohen_kappa, krippendorff_alpha_nominal

rng = np.random.default_rng(2)
sigmoid = lambda z: 1 / (1 + np.exp(-z))

# ---- 1. Agreement between human annotators -----------------------------------
# 400 responses; 90% are truly acceptable. Each annotator labels correctly with
# probability 0.9, except annotator 3, who barely reads and says "ok" 97% of the time.
N = 400
truth = (rng.random(N) < 0.9).astype(int)
noisy = lambda acc: np.where(rng.random(N) < acc, truth, 1 - truth)
ann = np.array([noisy(0.9), noisy(0.9), (rng.random(N) < 0.97).astype(int)])  # 3 ignores the item
print("1. three annotators, binary acceptable / not, 90% truly acceptable")
for i, j in [(0, 1), (0, 2), (1, 2)]:
    print(f"   annotators {i + 1} & {j + 1}: raw agreement {(ann[i] == ann[j]).mean():.3f}, "
          f"Cohen's kappa {cohen_kappa(ann[i], ann[j]):.3f}")
print(f"   Krippendorff's alpha, all three: {krippendorff_alpha_nominal(ann):.3f}")
# Missing data is normal: each item labeled by only two of the three.
sparse = ann.astype(float)
sparse[rng.integers(0, 3, N), np.arange(N)] = np.nan
print(f"   alpha with one random label missing per item: {krippendorff_alpha_nominal(sparse):.3f}")
# The same raters on a balanced set: same accuracy, very different kappa.
truth_b = (rng.random(N) < 0.5).astype(int)
b1, b2 = [np.where(rng.random(N) < 0.9, truth_b, 1 - truth_b) for _ in range(2)]
print(f"   balanced set, two 90%-accurate raters: raw {(b1 == b2).mean():.3f}, "
      f"kappa {cohen_kappa(b1, b2):.3f}")

# ---- 2. A pairwise judge with position and length bias -------------------------
# 600 prompts. Model X writes answers 40% longer than the reference model R and
# is slightly worse on true quality. Humans judge quality only.
P = 600
q_x, q_r = rng.normal(-0.15, 1, P), rng.normal(0, 1, P)
len_x, len_r = rng.normal(1.4, 0.3, P), rng.normal(1.0, 0.3, P)   # in units of R's mean length


def human_prefers_first(qa, qb):
    return rng.random(len(qa)) < sigmoid(2.0 * (qa - qb))


def judge_prefers_first(qa, qb, la, lb, pos_bias=0.6, len_bias=1.5):
    """Sees quality (a bit noisily), likes longer answers, and likes whatever is shown first."""
    z = 2.0 * (qa - qb) + len_bias * (la - lb) + pos_bias
    return rng.random(len(qa)) < sigmoid(z)


human = human_prefers_first(q_x, q_r).astype(float)
j_xfirst = judge_prefers_first(q_x, q_r, len_x, len_r).astype(float)        # X shown first
j_rfirst = 1 - judge_prefers_first(q_r, q_x, len_r, len_x).astype(float)    # R shown first
print("\n2. win rate of model X against reference R over 600 prompts")
print(f"   humans:                                   {human.mean():.3f}")
print(f"   judge, X always shown first:              {j_xfirst.mean():.3f}")
print(f"   judge, X always shown second:             {j_rfirst.mean():.3f}")
both = (j_xfirst + j_rfirst) / 2
print(f"   judge, both orders averaged:              {both.mean():.3f}  "
      "95% CI [{:.3f}, {:.3f}]".format(*bootstrap_ci(both)))
print(f"   verdict flips when order is swapped:      {(j_xfirst != j_rfirst).mean():.3f} of prompts")

# Length control in the spirit of length-controlled AlpacaEval: model the
# judge's verdict as logistic in the length difference, then predict the win
# rate the judge would give if the lengths were equal.
dlen = np.concatenate([len_x - len_r, len_x - len_r])
y = np.concatenate([j_xfirst, j_rfirst])
pos = np.concatenate([np.ones(P), -np.ones(P)])                # +1: X shown first
lr = LogisticRegression(C=1e6).fit(np.column_stack([dlen, pos]), y)
lc = sigmoid(lr.intercept_[0])                                 # dlen = 0, positions balanced
print(f"   length-controlled judge win rate:         {lc:.3f}  "
      f"(fitted length coefficient {lr.coef_[0][0]:.2f})")

# ---- 3. Agreement of the judge with humans --------------------------------------
print("\n3. judge vs human labels (per prompt, both-order majority; ties broken by X-first)")
j_major = np.where(both == 0.5, j_xfirst, both > 0.5).astype(int)
h2 = human_prefers_first(q_x, q_r).astype(int)                 # a second human
print(f"   judge-human agreement {(j_major == human).mean():.3f}, kappa {cohen_kappa(j_major, human):.3f}")
print(f"   human-human agreement {(h2 == human).mean():.3f}, kappa {cohen_kappa(h2, human):.3f}")

# ---- 4. When longer really is better, length control removes real signal --------
q_x2 = q_r + 0.5 * (len_x - len_r) + rng.normal(0, 0.5, P)    # quality now grows with length
human2 = human_prefers_first(q_x2, q_r).mean()
jx = judge_prefers_first(q_x2, q_r, len_x, len_r).astype(float)
jr = 1 - judge_prefers_first(q_r, q_x2, len_r, len_x).astype(float)
lr2 = LogisticRegression(C=1e6).fit(np.column_stack([dlen, pos]), np.concatenate([jx, jr]))
print("\n4. same judge, but now longer answers are genuinely better")
print(f"   humans {human2:.3f}   judge both orders {(jx + jr).mean() / 2:.3f}   "
      f"length-controlled {sigmoid(lr2.intercept_[0]):.3f}")
