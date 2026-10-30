"""Gradient boosting from scratch.
Part 1: squared loss (fit trees to residuals) on California housing, with and
without shrinkage, checked against scikit-learn.
Part 2: log-loss with Newton leaf values (the XGBoost-style second-order step)
on the breast cancer data.
"""
import time

import numpy as np
from sklearn.datasets import fetch_california_housing, load_breast_cancer
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                              HistGradientBoostingRegressor)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

X, y = fetch_california_housing(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)
CHECK = [1, 10, 50, 100, 200, 500]


def boost_l2(lr, n_rounds=500, depth=3):
    f_tr = np.full(len(y_tr), y_tr.mean())      # F_0: the best constant
    f_te = np.full(len(y_te), y_tr.mean())
    out = {}
    for m in range(1, n_rounds + 1):
        resid = y_tr - f_tr                     # negative gradient of (1/2)(y - F)^2
        t = DecisionTreeRegressor(max_depth=depth, random_state=0).fit(X_tr, resid)
        f_tr += lr * t.predict(X_tr)
        f_te += lr * t.predict(X_te)
        if m in CHECK:
            out[m] = (np.mean((y_tr - f_tr) ** 2), np.mean((y_te - f_te) ** 2))
    return out


print("rounds   lr=1.0 train/test    lr=0.1 train/test")
r1, r01 = boost_l2(1.0), boost_l2(0.1)
for m in CHECK:
    print(f"{m:6d}   {r1[m][0]:.3f} / {r1[m][1]:.3f}       {r01[m][0]:.3f} / {r01[m][1]:.3f}")

t0 = time.perf_counter()
gb = GradientBoostingRegressor(n_estimators=500, learning_rate=0.1, max_depth=3,
                               random_state=0).fit(X_tr, y_tr)
t1 = time.perf_counter()
hgb = HistGradientBoostingRegressor(random_state=0).fit(X_tr, y_tr)
t2 = time.perf_counter()
print(f"\nsklearn GradientBoostingRegressor (500 x depth 3, lr 0.1): test MSE "
      f"{np.mean((gb.predict(X_te) - y_te) ** 2):.3f}  [{t1 - t0:.1f} s]")
print(f"sklearn HistGradientBoostingRegressor (defaults): test MSE "
      f"{np.mean((hgb.predict(X_te) - y_te) ** 2):.3f}, {hgb.n_iter_} rounds  [{t2 - t1:.1f} s]")

# ---- Part 2: log-loss with Newton leaves ----
Xb, yb = load_breast_cancer(return_X_y=True)
Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(Xb, yb, test_size=0.3, random_state=0, stratify=yb)


def boost_logloss(n_rounds=200, lr=0.1, depth=3, lam=1.0):
    p0 = yb_tr.mean()
    f_tr = np.full(len(yb_tr), np.log(p0 / (1 - p0)))   # F_0 = log-odds of the base rate
    f_te = np.full(len(yb_te), f_tr[0])
    for _ in range(n_rounds):
        p = 1 / (1 + np.exp(-f_tr))
        g, h = p - yb_tr, p * (1 - p)           # first and second derivative wrt F
        # The tree picks the partition using the gradient ...
        t = DecisionTreeRegressor(max_depth=depth, random_state=0).fit(Xb_tr, -g)
        # ... then each leaf gets the Newton step -sum(g) / (sum(h) + lambda).
        leaf_tr, leaf_te = t.apply(Xb_tr), t.apply(Xb_te)
        value = {L: -g[leaf_tr == L].sum() / (h[leaf_tr == L].sum() + lam)
                 for L in np.unique(leaf_tr)}
        f_tr += lr * np.array([value[L] for L in leaf_tr])
        f_te += lr * np.array([value[L] for L in leaf_te])
    return f_te


gbc = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3,
                                 random_state=0).fit(Xb_tr, yb_tr)
pg = gbc.predict_proba(Xb_te)[:, 1]


def report(name, p):
    ll = -np.mean(yb_te * np.log(p) + (1 - yb_te) * np.log(1 - p))
    print(f"{name:38s} test accuracy {np.mean((p > 0.5) == yb_te):.3f}, test log-loss {ll:.4f}")


print()
for lam in [0.0, 1.0]:
    report(f"log-loss boosting, ours, lambda={lam}:", 1 / (1 + np.exp(-boost_logloss(lam=lam))))
report("sklearn GradientBoostingClassifier:", pg)
