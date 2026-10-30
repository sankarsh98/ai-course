"""Train/serve skew and label errors, two data problems a model can't see.

Part 1: serving computes features differently; a statistics check catches it.
Part 2: 10% of training labels are wrong; out-of-fold predictions flag suspects.
"""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(0)

# Part 1 -------------------------------------------------------------------
n = 20000
tenure_days = rng.exponential(400, n)
sessions = rng.poisson(5, n).astype(float)
sessions[rng.random(n) < 0.2] = np.nan     # some users have no session log
logit = -1 + 1.5 * (tenure_days < 90) - 0.3 * np.nan_to_num(sessions, nan=2.0)
churn = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
X = np.column_stack([tenure_days, sessions])
Xtr, Xsv, ytr, ysv = train_test_split(X, churn, test_size=0.3, random_state=0)
model = HistGradientBoostingClassifier(random_state=0).fit(Xtr, ytr)

# The serving team re-implemented the features: tenure in weeks, and missing
# session counts filled with 0 instead of being left missing.
Xbug = Xsv.copy()
Xbug[:, 0] = Xbug[:, 0] / 7
Xbug[:, 1] = np.nan_to_num(Xbug[:, 1], nan=0.0)
print("Part 1: train/serve skew")
print(f"  AUC, serving features computed like training: {roc_auc_score(ysv, model.predict_proba(Xsv)[:, 1]):.3f}")
print(f"  AUC, serving features from buggy pipeline   : {roc_auc_score(ysv, model.predict_proba(Xbug)[:, 1]):.3f}")


def skew_report(train, serve, names):
    # Standardized mean difference and missing-rate difference, per feature.
    for j, name in enumerate(names):
        a, b = train[:, j], serve[:, j]
        smd = (np.nanmean(b) - np.nanmean(a)) / np.nanstd(a)
        dmiss = np.isnan(b).mean() - np.isnan(a).mean()
        flag = "  <-- check" if abs(smd) > 0.25 or abs(dmiss) > 0.05 else ""
        print(f"    {name:12s} std. mean diff {smd:+.2f}  missing-rate diff {dmiss:+.2f}{flag}")


print("  skew check, buggy serving data vs training data:")
skew_report(Xtr, Xbug, ["tenure_days", "sessions"])

# Part 2 -------------------------------------------------------------------
Xd, yd = load_digits(return_X_y=True)
Xa, Xb, ya, yb = train_test_split(Xd, yd, test_size=0.5, stratify=yd, random_state=0)
flip = rng.random(len(ya)) < 0.10
ya_noisy = ya.copy()
ya_noisy[flip] = (ya[flip] + rng.integers(1, 10, flip.sum())) % 10   # wrong label


def clf():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))


print(f"\nPart 2: label errors ({flip.sum()} of {len(ya)} training labels flipped)")
acc_noisy = (clf().fit(Xa, ya_noisy).predict(Xb) == yb).mean()
print(f"  test accuracy, trained on noisy labels: {acc_noisy:.3f}")

# Out-of-fold probability the model assigns to each row's *given* label.
# Low values mark suspects (a simple form of the idea behind confident learning).
P = cross_val_predict(clf(), Xa, ya_noisy, cv=5, method="predict_proba")
self_conf = P[np.arange(len(ya)), ya_noisy]
suspects = np.argsort(self_conf)[:flip.sum()]
print(f"  of the {len(suspects)} lowest-confidence rows, {flip[suspects].mean():.0%} were truly flipped")
keep = np.setdiff1d(np.arange(len(ya)), suspects)
acc_clean = (clf().fit(Xa[keep], ya_noisy[keep]).predict(Xb) == yb).mean()
print(f"  test accuracy after dropping suspects : {acc_clean:.3f}")
print(f"  test accuracy, trained on true labels : {(clf().fit(Xa, ya).predict(Xb) == yb).mean():.3f}")

# Now corrupt 10% of the *test* labels and compare two models.
yb_noisy = yb.copy()
tf = rng.random(len(yb)) < 0.10
yb_noisy[tf] = (yb[tf] + rng.integers(1, 10, tf.sum())) % 10
print("  measured on clean vs noisy test labels:")
for name, m in [("trained on noisy labels ", clf().fit(Xa, ya_noisy)),
                ("after dropping suspects ", clf().fit(Xa[keep], ya_noisy[keep]))]:
    p = m.predict(Xb)
    print(f"    {name} clean {np.mean(p == yb):.3f}   noisy {np.mean(p == yb_noisy):.3f}")
