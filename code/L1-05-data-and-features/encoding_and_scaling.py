"""Scaling for distance-based models, and target encoding done wrong and right.

Part 1: k-NN on the wine data with and without standardization.
Part 2: two high-cardinality columns, a useful "city" and a pure-noise
        "user_id", encoded three ways.
"""
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.datasets import load_wine
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

# Part 1 -------------------------------------------------------------------
Xw, yw = load_wine(return_X_y=True)
for name, model in [("k-NN, raw features   ", KNeighborsClassifier()),
                    ("k-NN, standardized   ", make_pipeline(StandardScaler(), KNeighborsClassifier()))]:
    acc = cross_val_score(model, Xw, yw, cv=KFold(5, shuffle=True, random_state=0))
    print(f"{name} 5-fold accuracy {acc.mean():.3f}")

# Part 2 -------------------------------------------------------------------
rng = np.random.default_rng(0)
n = 20000
city = rng.integers(0, 500, n)            # 500 cities, about 40 rows each
city_effect = rng.normal(0, 1.0, 500)     # each city has a real effect
user_id = rng.integers(0, 10000, n)       # about 2 rows per id, no effect at all
x_num = rng.normal(size=n)
logit = 0.8 * x_num + city_effect[city]
y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
X = np.column_stack([x_num, city, user_id])
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)


def naive_target_encode(col_tr, col_te):
    # Mean target per category computed on ALL training rows, including the
    # row being encoded. That row's own label leaks into its feature.
    prior = ytr.mean()
    means = {c: ytr[col_tr == c].mean() for c in np.unique(col_tr)}
    return (np.array([means[c] for c in col_tr]),
            np.array([means.get(c, prior) for c in col_te]))


def report(name, Ftr, Fte, model):
    model.fit(Ftr, ytr)
    tr = roc_auc_score(ytr, model.predict_proba(Ftr)[:, 1])
    te = roc_auc_score(yte, model.predict_proba(Fte)[:, 1])
    print(f"  {name:34s} train AUC {tr:.3f}   test AUC {te:.3f}")


def gbm():
    return HistGradientBoostingClassifier(random_state=0)


# Best possible test AUC: score test rows with the true logit we simulated.
oracle = 0.8 * Xte[:, 0] + city_effect[Xte[:, 1].astype(int)]
print(f"\nHigh-cardinality categoricals (oracle test AUC {roc_auc_score(yte, oracle):.3f})")

enc = [naive_target_encode(Xtr[:, j], Xte[:, j]) for j in (1, 2)]
report("naive target encoding + GBM",
       np.column_stack([Xtr[:, 0], enc[0][0], enc[1][0]]),
       np.column_stack([Xte[:, 0], enc[0][1], enc[1][1]]), gbm())

te_enc = TargetEncoder(cv=KFold(5, shuffle=True, random_state=0))
cats_tr = te_enc.fit_transform(Xtr[:, 1:].astype(int), ytr)   # cross-fitted
cats_te = te_enc.transform(Xte[:, 1:].astype(int))
report("cross-fitted TargetEncoder + GBM",
       np.column_stack([Xtr[:, 0], cats_tr]), np.column_stack([Xte[:, 0], cats_te]), gbm())

# One-hot the two categorical columns (sparse), keep the numeric column as is.
oh = OneHotEncoder(handle_unknown="ignore").fit(Xtr[:, 1:].astype(int))
Ftr = hstack([csr_matrix(Xtr[:, :1]), oh.transform(Xtr[:, 1:].astype(int))]).tocsr()
Fte = hstack([csr_matrix(Xte[:, :1]), oh.transform(Xte[:, 1:].astype(int))]).tocsr()
report("one-hot + L2 logistic regression", Ftr, Fte, LogisticRegression(C=0.5, max_iter=2000))
