"""Accuracy vs precision/recall, ROC-AUC vs PR-AUC under imbalance, and calibration."""
import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             log_loss, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


def data(pos_rate, n=40_000, seed=0):
    X, y = make_classification(n_samples=n, n_features=20, n_informative=5,
                               weights=[1 - pos_rate], flip_y=0.01, random_state=seed)
    return train_test_split(X, y, test_size=0.5, stratify=y, random_state=seed)

# --- 1. Thresholds on an imbalanced problem (about 2% positives) ---
X_tr, X_te, y_tr, y_te = data(0.02)
p = LogisticRegression(max_iter=2000).fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
print(f"1. test set: {len(y_te)} examples, {y_te.sum()} positives ({y_te.mean():.1%})")
print(f"   always-negative accuracy: {1 - y_te.mean():.3f}")
print(f"   {'threshold':>9} {'accuracy':>9} {'precision':>10} {'recall':>7}  TN FP FN TP")
for t in [0.5, 0.2, 0.1, 0.05]:
    pred = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
    print(f"   {t:9.2f} {(tn + tp) / len(y_te):9.3f} "
          f"{precision_score(y_te, pred, zero_division=0):10.3f} "
          f"{recall_score(y_te, pred):7.3f}  {tn} {fp} {fn} {tp}")

# --- 2. Same model, same score distributions, different prevalence ---
# Train once, then build test sets with fewer positives by dropping some of them.
# ROC-AUC depends only on how positive and negative scores compare, so it stays put.
a, b, c, d = data(0.5)
s = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=0).fit(a, c).predict_proba(b)[:, 1]
rng = np.random.default_rng(0)
pos, neg = np.flatnonzero(d == 1), np.flatnonzero(d == 0)
print("\n2. one random forest, test sets with fewer and fewer positives")
print(f"   {'pos rate':>8} {'ROC-AUC':>8} {'avg precision':>14}")
for rate in [0.5, 0.1, 0.02, 0.005]:
    k = int(rate / (1 - rate) * len(neg))          # positives to keep for this rate
    idx = np.concatenate([neg, rng.choice(pos, size=min(k, len(pos)), replace=False)])
    # A random scorer has ROC-AUC 0.5 and average precision equal to the positive rate.
    print(f"   {d[idx].mean():8.3f} {roc_auc_score(d[idx], s[idx]):8.3f} "
          f"{average_precision_score(d[idx], s[idx]):14.3f}")

# --- 3. Calibration: do predicted probabilities match observed frequencies? ---
X_tr, X_te, y_tr, y_te = data(0.3)
models = {
    "logistic": LogisticRegression(max_iter=2000),
    "naive Bayes": GaussianNB(),
    "NB + isotonic": CalibratedClassifierCV(GaussianNB(), method="isotonic", cv=5),
    "random forest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=0),
}
print("\n3. calibration on a 30%-positive problem")
print(f"   {'model':>14} {'ROC-AUC':>8} {'Brier':>7} {'log loss':>9} {'ECE':>6}")
reliability = {}
for name, m in models.items():
    q = m.fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
    # Expected calibration error: bin-weighted gap between confidence and frequency.
    bins = np.minimum((q * 10).astype(int), 9)
    ece = sum(abs(q[bins == k].mean() - y_te[bins == k].mean()) * (bins == k).mean()
              for k in range(10) if (bins == k).any())
    print(f"   {name:>14} {roc_auc_score(y_te, q):8.3f} {brier_score_loss(y_te, q):7.4f} "
          f"{log_loss(y_te, q):9.4f} {ece:6.3f}")
    reliability[name] = calibration_curve(y_te, q, n_bins=5)
print("\n   reliability table (mean predicted -> observed fraction positive, 5 bins)")
for name, (frac, mean_pred) in reliability.items():
    pairs = "  ".join(f"{mp:.2f}->{fr:.2f}" for fr, mp in zip(frac, mean_pred))
    print(f"   {name:>14}: {pairs}")
