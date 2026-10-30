"""Baselines first, then a train/validation/test workflow, and how much one split can lie."""
import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
print(f"breast cancer: {X.shape[0]} samples, {X.shape[1]} features, "
      f"{y.mean():.1%} labelled benign (class 1)")

# 60/20/20 split: test is locked away; validation is for choosing C.
X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
X_tr, X_val, y_tr, y_val = train_test_split(X_tmp, y_tmp, test_size=0.25, stratify=y_tmp, random_state=0)

base = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
print(f"\nmajority-class baseline, validation accuracy = {base.score(X_val, y_val):.3f}")

print("choosing C for logistic regression on the validation set:")
best = None
for C in [0.001, 0.01, 0.1, 1, 10, 100]:
    model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))
    acc = model.fit(X_tr, y_tr).score(X_val, y_val)
    print(f"  C = {C:<6} validation accuracy = {acc:.3f}")
    if best is None or acc > best[0]:
        best = (acc, C)
C = best[1]
# Refit on train+validation with the chosen C, then touch the test set exactly once.
final = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000)).fit(X_tmp, y_tmp)
print(f"chosen C = {C}; test accuracy (reported once) = {final.score(X_test, y_test):.3f}")

print("\nsame model, 20 different random 80/20 splits:")
accs = []
for seed in range(20):
    a, b, c, d = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000)).fit(a, c)
    accs.append(m.score(b, d))
print(f"  test accuracy min {min(accs):.3f}, max {max(accs):.3f}, "
      f"mean {np.mean(accs):.3f}, std {np.std(accs, ddof=1):.3f}")

Xr, yr = load_diabetes(return_X_y=True)
a, b, c, d = train_test_split(Xr, yr, test_size=0.25, random_state=0)
print(f"\ndiabetes regression: {Xr.shape[0]} samples, target range {yr.min():.0f} to {yr.max():.0f}")
for name, reg in [("predict-the-mean", DummyRegressor()), ("linear regression", LinearRegression())]:
    p = reg.fit(a, c).predict(b)
    print(f"  {name:>17}: MAE {mean_absolute_error(d, p):6.1f}  "
          f"RMSE {root_mean_squared_error(d, p):6.1f}  R^2 {r2_score(d, p):6.3f}")
