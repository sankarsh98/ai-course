"""Impurity-based (MDI) importance vs permutation importance, with planted
traps: a pure-noise continuous column, a pure-noise binary column, and a
near-copy of the strongest feature.
"""
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

data = fetch_california_housing()
X, y, names = data.data, data.target, list(data.feature_names)
rng = np.random.default_rng(0)
n = len(y)
X = np.column_stack([
    X,
    rng.normal(size=n),                          # noise, ~20k distinct values
    rng.integers(0, 2, n),                       # noise, 2 distinct values
    X[:, 0] + rng.normal(0, 0.1 * X[:, 0].std(), n),   # near-copy of MedInc
])
names += ["noise_cont", "noise_bin", "MedInc_copy"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)


def importances(X_tr, X_te, cols):
    rf = RandomForestRegressor(n_estimators=100, max_features=0.5, n_jobs=-1,
                               random_state=0).fit(X_tr, y_tr)
    pi_tr = permutation_importance(rf, X_tr, y_tr, n_repeats=5, random_state=0, n_jobs=-1)
    pi_te = permutation_importance(rf, X_te, y_te, n_repeats=5, random_state=0, n_jobs=-1)
    print(f"  train R^2 {rf.score(X_tr, y_tr):.3f}, test R^2 {rf.score(X_te, y_te):.3f}")
    print(f"  {'feature':12s} {'MDI':>7s} {'perm(train)':>12s} {'perm(test)':>11s}")
    for i in np.argsort(-rf.feature_importances_):
        print(f"  {cols[i]:12s} {rf.feature_importances_[i]:7.3f} "
              f"{pi_tr.importances_mean[i]:12.3f} {pi_te.importances_mean[i]:11.3f}")
    return rf


print("All features (permutation importance = drop in R^2 when a column is shuffled):")
importances(X_tr, X_te, names)

# Remove the copy: what MedInc "really" carries becomes visible again.
keep = [i for i, c in enumerate(names) if c != "MedInc_copy"]
print("\nWithout MedInc_copy:")
importances(X_tr[:, keep], X_te[:, keep], [names[i] for i in keep])
