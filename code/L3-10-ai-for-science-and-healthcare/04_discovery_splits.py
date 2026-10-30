"""SYNTHETIC DATA ONLY. 'Compounds' are random vectors grouped into families
(think scaffolds or chemical systems); the property is an invented function.

Two questions a discovery paper must answer:
1. Does the model work on new families, or only on new members of known ones?
2. Can it rank candidates better than anything it was trained on?"""
import os

# Many small fits: extra threads mostly add overhead, and on a shared machine
# they can slow things down a lot. Must be set before numpy is imported.
for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(var, "4")

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, KFold

rng = np.random.default_rng(0)
F, PER, D = 80, 30, 12                         # families, members per family, descriptors
centres = rng.normal(0, 1, (F, D))
fam = np.repeat(np.arange(F), PER)
X = centres[fam] + rng.normal(0, 0.25, (F * PER, D))
w = rng.normal(0, 1, D)
fam_offset = rng.normal(0, 0.8, F)             # family effect not explained by descriptors
y = X @ w * 0.6 + np.sin(2 * X[:, 0]) + fam_offset[fam] + rng.normal(0, 0.3, len(fam))
print(f"{len(y)} compounds in {F} families, {D} descriptors")


def cv_r2(model, splitter):
    scores = []
    groups = fam if isinstance(splitter, GroupKFold) else None
    for tr, te in splitter.split(X, y, groups):
        scores.append(r2_score(y[te], model.fit(X[tr], y[tr]).predict(X[te])))
    return np.mean(scores), np.std(scores)


models = {"ridge": Ridge(alpha=1.0),
          "random forest": RandomForestRegressor(300, min_samples_leaf=2, n_jobs=4, random_state=0)}
print(f"\n{'model':<14} {'random 5-fold R2':>17} {'family-grouped 5-fold R2':>25}")
for name, m in models.items():
    r, rs = cv_r2(m, KFold(5, shuffle=True, random_state=0))
    g, gs = cv_r2(m, GroupKFold(5))
    print(f"{name:<14} {r:10.2f} ({rs:.2f}) {g:17.2f} ({gs:.2f})")

# ---- Prospective-style test: hide the best 5% and see if they can be found.
top = y >= np.quantile(y, 0.95)
train = ~top & (rng.random(len(y)) < 0.7)              # known, ordinary compounds
pool = ~train                                          # unseen: all top + 30% of others
print(f"\ntraining set: {train.sum()} ordinary compounds, best property {y[train].max():.2f}")
print(f"candidate pool: {pool.sum()} unseen compounds, {top.sum()} of them better than every training compound")
base_rate = top[pool].mean()
print(f"\n{'model':<14} {'max predicted':>14} {'hits in top 50':>15} {'enrichment vs random':>21}")
for name, m in models.items():
    pred = m.fit(X[train], y[train]).predict(X[pool])
    chosen = np.argsort(-pred)[:50]
    hits = top[pool][chosen].sum()
    print(f"{name:<14} {pred.max():14.2f} {hits:15d} {hits / 50 / base_rate:21.1f}x")
print(f"{'random pick':<14} {'':>14} {50 * base_rate:15.1f} {1.0:21.1f}x")
