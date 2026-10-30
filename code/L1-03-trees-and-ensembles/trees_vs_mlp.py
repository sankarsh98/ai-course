"""A small, single-dataset echo of two experiments in Grinsztajn et al. (2022):
how do a tree ensemble and an MLP react to (a) extra uninformative features and
(b) a random rotation of the feature space? This is a demonstration of a
mechanism, not a benchmark: one dataset, one split, no tuning.
"""
import time

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = fetch_california_housing(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)
rng = np.random.default_rng(0)

# Standardize once so that the rotation below mixes features on equal footing.
sc = StandardScaler().fit(X_tr)
Z_tr, Z_te = sc.transform(X_tr), sc.transform(X_te)


def models():
    return {
        "ridge": RidgeCV(alphas=np.logspace(-3, 3, 13)),
        "random forest": RandomForestRegressor(n_estimators=100, min_samples_leaf=2,
                                               n_jobs=-1, random_state=0),
        "hist GBDT": HistGradientBoostingRegressor(random_state=0),
        "MLP": make_pipeline(StandardScaler(), MLPRegressor(
            hidden_layer_sizes=(128, 64), early_stopping=True, max_iter=300,
            random_state=0)),
    }


def run(A_tr, A_te):
    row = {}
    for name, m in models().items():
        m.fit(A_tr, y_tr)
        row[name] = np.mean((m.predict(A_te) - y_te) ** 2)
    return row


# (a) 24 extra columns of pure noise. (b) a random orthogonal rotation Q.
noise_tr, noise_te = rng.normal(size=(len(Z_tr), 24)), rng.normal(size=(len(Z_te), 24))
Q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
settings = {
    "original 8 features": (Z_tr, Z_te),
    "+24 noise features": (np.hstack([Z_tr, noise_tr]), np.hstack([Z_te, noise_te])),
    "randomly rotated": (Z_tr @ Q, Z_te @ Q),
}

t0 = time.perf_counter()
print(f"{'test MSE':22s}" + "".join(f"{k:>15s}" for k in models()))
for label, (A_tr, A_te) in settings.items():
    row = run(A_tr, A_te)
    print(f"{label:22s}" + "".join(f"{v:15.3f}" for v in row.values()))
print(f"(total time {time.perf_counter() - t0:.0f} s)")
