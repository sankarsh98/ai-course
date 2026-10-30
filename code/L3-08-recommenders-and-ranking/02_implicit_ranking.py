"""Top-N recommendation from implicit clicks: popularity, item-kNN and implicit ALS,
scored three ways: full-catalogue metrics on held-out clicks, the sampled-negatives
shortcut, and (only possible because the data are synthetic) true user preference.
"""
import itertools

import numpy as np

from recdata import (ials, item_knn_scores, make_world, organic_clicks, sampled_hit_rate,
                     split_holdout, topk_metrics)

world = make_world()
X = organic_clicks(world)                 # exposure was driven by popularity
train, test = split_holdout(X)
cp = world["click_prob"]
top10_share = np.sort(X.sum(0))[-10:].sum() / X.sum()
print(f"{X.shape[0]} users, {X.shape[1]} items, {int(X.sum())} clicks "
      f"({X.sum(1).mean():.1f} per user); the top 10 items get {top10_share:.0%} of clicks")


def true_precision(scores, k=10):
    """Mean TRUE click probability of the k items recommended to each user."""
    s = np.where(train > 0, -np.inf, scores)
    top = np.argsort(-s, axis=1)[:, :k]
    return float(np.take_along_axis(cp, top, axis=1).mean())


rng = np.random.default_rng(0)
U, V = ials(train, k=8, alpha=10, reg=1)
U16, V16 = ials(train, k=16, alpha=10, reg=1)
models = {
    "random": rng.random(X.shape),
    "popularity": np.tile(train.sum(0), (X.shape[0], 1)),
    "item-kNN (cosine)": item_knn_scores(train),
    "iALS k=8": U @ V.T,
    "iALS k=16": U16 @ V16.T,
    "oracle: true prefs": cp,
}
print("\n1. Scores                Recall@10  NDCG@10  sampled HR@10  true prec@10")
for name, s in models.items():
    r, n = topk_metrics(s, train, test)
    print(f"   {name:20s}  {r:8.3f} {n:8.3f}  {sampled_hit_rate(s, train, test):12.3f}"
          f"  {true_precision(s):11.3f}")

# ---- 2. Tune iALS by the offline metric, or by what users actually want --------
print("\n2. iALS grid: choose by held-out Recall@10, or by true precision@10")
rows = []
for k, alpha, reg in itertools.product((8, 32), (2, 10, 40), (0.1, 10, 50)):
    U, V = ials(train, k=k, alpha=alpha, reg=reg, iters=8)
    s = U @ V.T
    rows.append((k, alpha, reg, topk_metrics(s, train, test)[0], true_precision(s)))
rows = np.array(rows)
by_offline, by_truth = rows[rows[:, 3].argmax()], rows[rows[:, 4].argmax()]
for label, r in [("best offline Recall", by_offline), ("best true precision", by_truth)]:
    print(f"   {label:20s} k={r[0]:<3.0f} alpha={r[1]:<4.0f} reg={r[2]:<5g} "
          f"Recall@10 {r[3]:.3f}  true prec@10 {r[4]:.3f}")
corr = np.corrcoef(rows[:, 3], rows[:, 4])[0, 1]
print(f"   correlation between the two criteria over {len(rows)} configs: {corr:+.2f}")
pop_share = []
for k, alpha, reg in (by_offline[:3], by_truth[:3]):
    U, V = ials(train, k=int(k), alpha=alpha, reg=reg, iters=8)
    top = np.argsort(-np.where(train > 0, -np.inf, U @ V.T), axis=1)[:, :10]
    pop_share.append(np.isin(top, np.argsort(-world["pop"])[:20]).mean())
print(f"   share of recommendations from the 20 most-exposed items: "
      f"{pop_share[0]:.0%} (offline pick) vs {pop_share[1]:.0%} (truth pick)")
