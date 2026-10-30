"""A minimal offline eval harness: items, models, a grader, and honest statistics.

The "models" are simulated so that we know the truth: each has a skill, each
item a difficulty, and each passage (a cluster of 6 questions written about
the same scenario) a shared difficulty. Model B is slightly more accurate
than A but phrases its answers more verbosely.
"""
import re

import numpy as np

from evalkit import (bootstrap_ci, clustered_se, mean_se, n_for_power,
                     paired_permutation_p)

rng = np.random.default_rng(0)
N_PASSAGES, Q_PER = 40, 6
PASSAGE_SD, ITEM_SD = 1.0, 2.0          # spread of difficulty, in logits


def make_items(n_passages, rng):
    """Arithmetic word problems; questions about one passage share its difficulty."""
    items = []
    for p in range(n_passages):
        p_diff = rng.normal(0, PASSAGE_SD)
        for q in range(Q_PER):
            a, b = rng.integers(2, 30, size=2)
            c = rng.integers(1, a * b)
            items.append(dict(
                passage=p,
                question=f"A crate holds {a} boxes of {b} pens; {c} pens are lost. How many remain?",
                answer=int(a * b - c),
                difficulty=p_diff + rng.normal(0, ITEM_SD)))
    return items


class SimulatedModel:
    def __init__(self, name, skill, formats):
        self.name, self.skill, self.formats = name, skill, formats

    def p_correct(self, item):
        return 1 / (1 + np.exp(-(self.skill - item["difficulty"])))

    def answer(self, item, rng):
        """One sampled response: right with probability p_correct, else off by a bit."""
        value = item["answer"]
        if rng.random() > self.p_correct(item):
            value += int(rng.choice([-10, -2, -1, 1, 2, 10]))
        return rng.choice(self.formats).format(value)


def grade_strict(response, item):
    return float(response.strip() == str(item["answer"]))


def grade_normalized(response, item):
    """Take the last number in the response; compare numerically."""
    nums = re.findall(r"-?\d+(?:\.\d+)?", response.replace(",", ""))
    return float(bool(nums) and float(nums[-1]) == item["answer"])


A = SimulatedModel("A", skill=1.0, formats=["{}", "{}", "{}", "{}."])
B = SimulatedModel("B", skill=1.4, formats=["{}", "The answer is {}.", "{} pens"])

items = make_items(N_PASSAGES, rng)
clusters = np.array([it["passage"] for it in items])
print(f"{len(items)} items in {N_PASSAGES} passages; example: {items[0]['question']} -> {items[0]['answer']}")

# ---- 1. The grader is part of the metric ------------------------------------
resp = {m.name: [m.answer(it, rng) for it in items] for m in (A, B)}
print("\n1. accuracy, one sample per item")
for m in (A, B):
    strict = np.mean([grade_strict(r, it) for r, it in zip(resp[m.name], items)])
    norm_ = np.mean([grade_normalized(r, it) for r, it in zip(resp[m.name], items)])
    print(f"   model {m.name}: strict match {strict:.3f}   normalized {norm_:.3f}   "
          f"responses look like {sorted(set(resp[m.name][:8]), key=len)[-1]!r}")

score = {m.name: np.array([grade_normalized(r, it) for r, it in zip(resp[m.name], items)])
         for m in (A, B)}

# ---- 2. Standard errors: items are not independent ---------------------------
s = score["A"]
print("\n2. model A, normalized accuracy {:.3f}".format(s.mean()))
print(f"   naive SE {mean_se(s):.4f}   clustered SE {clustered_se(s, clusters):.4f}")
print("   95% bootstrap CI, resampling items:    [{:.3f}, {:.3f}]".format(*bootstrap_ci(s)))
print("   95% bootstrap CI, resampling passages: [{:.3f}, {:.3f}]".format(*bootstrap_ci(s, clusters)))

# ---- 3. Several samples per item shrink only the sampling part ----------------
K = 10
multi = np.array([[grade_normalized(A.answer(it, rng), it) for _ in range(K)] for it in items])
print(f"\n3. model A, mean of {K} samples per item: accuracy {multi.mean():.3f}")
print(f"   clustered SE with 1 sample {clustered_se(multi[:, 0], clusters):.4f}, "
      f"with {K} samples {clustered_se(multi.mean(1), clusters):.4f}")

# ---- 4. Comparing A and B on the same items ------------------------------------
d = score["B"] - score["A"]
unpaired = np.sqrt(mean_se(score["A"]) ** 2 + mean_se(score["B"]) ** 2)
print(f"\n4. B - A = {d.mean():+.3f}")
print(f"   SE if treated as independent samples {unpaired:.4f}; paired SE {mean_se(d):.4f}; "
      f"paired clustered SE {clustered_se(d, clusters):.4f}")
print(f"   paired permutation p: items {paired_permutation_p(score['B'], score['A']):.3f}, "
      f"passages {paired_permutation_p(score['B'], score['A'], clusters):.3f}")
true_gap = np.mean([B.p_correct(it) - A.p_correct(it) for it in make_items(5000, rng)])
print(f"   true expected gap on this item distribution (from 30,000 fresh items): {true_gap:+.3f}")
sd_d = d.std(ddof=1) * clustered_se(d, clusters) / mean_se(d)   # inflate for clustering
print(f"   items needed for 80% power at that gap: about {n_for_power(sd_d, true_gap):,}")

# ---- 5. Do the intervals cover the truth? -----------------------------------------
# Repeat the whole experiment on 1,000 fresh eval sets (model A, one sample per
# item). The truth is A's expected accuracy over the item distribution.
truth = np.mean([A.p_correct(it) for it in make_items(5000, rng)])
cover_naive = cover_clust = 0
for rep in range(1000):
    its = make_items(N_PASSAGES, rng)
    cl = np.array([it["passage"] for it in its])
    sc = np.array([float(rng.random() < A.p_correct(it)) for it in its])
    cover_naive += abs(sc.mean() - truth) < 1.96 * mean_se(sc)
    cover_clust += abs(sc.mean() - truth) < 1.96 * clustered_se(sc, cl)
print(f"\n5. coverage of nominal 95% intervals over 1,000 fresh eval sets "
      f"(true accuracy {truth:.3f}):")
print(f"   naive {cover_naive / 1000:.3f}   clustered {cover_clust / 1000:.3f}")
