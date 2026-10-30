"""Why averaging trees works: bagging by hand, then random forests with
different max_features, measuring single-tree error, tree-to-tree error
correlation, and ensemble error on the California housing data.
"""
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

X, y = fetch_california_housing(return_X_y=True)  # downloads ~400 KB once, then cached
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)
rng = np.random.default_rng(0)


def mse(pred):
    return np.mean((pred - y_te) ** 2)


# 1. Bagging by hand: fit full-depth trees on bootstrap samples, average them.
preds = []
for b in range(100):
    idx = rng.integers(0, len(y_tr), len(y_tr))     # sample n rows with replacement
    tree = DecisionTreeRegressor(random_state=b).fit(X_tr[idx], y_tr[idx])
    preds.append(tree.predict(X_te))
preds = np.array(preds)
print(f"baseline (predict training mean): test MSE {mse(np.full_like(y_te, y_tr.mean())):.3f}")
print("bagged trees:  B    test MSE")
for B in [1, 2, 5, 10, 25, 50, 100]:
    print(f"            {B:4d}   {mse(preds[:B].mean(0)):.3f}")


# 2. Random forests: decorrelate trees by restricting features at each split.
def forest_stats(max_features):
    rf = RandomForestRegressor(n_estimators=100, max_features=max_features,
                               oob_score=True, n_jobs=-1, random_state=0).fit(X_tr, y_tr)
    P = np.array([t.predict(X_te) for t in rf.estimators_])
    E = P - y_te                                    # each tree's errors on test rows
    C = np.corrcoef(E)
    rho = C[np.triu_indices_from(C, k=1)].mean()   # mean pairwise error correlation
    return np.mean(E ** 2), rho, mse(rf.predict(X_te)), 1 - rf.oob_score_


print("\nmax_features  single-tree MSE  error corr rho  formula  forest MSE  OOB MSE")
var_y = np.var(y_tr)  # OOB R^2 is computed against training targets, so this converts it to MSE
for mf in [1, 2, 3, 5, 8]:
    single, rho, forest, oob_unexpl = forest_stats(mf)
    formula = single * (rho + (1 - rho) / 100)   # rho*s + (1 - rho)*s/B with B = 100
    print(f"{mf:12d}  {single:15.3f}  {rho:14.3f}  {formula:7.3f}  {forest:10.3f}"
          f"  {oob_unexpl * var_y:7.3f}")
