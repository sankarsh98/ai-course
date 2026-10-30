"""Class imbalance: resampling and reweighting vs simply moving the threshold.

Logistic regression on data with about 3% positives, trained four ways, scored
on ranking (ROC AUC, average precision), calibration (Brier score, mean
predicted probability) and precision/recall at a threshold.
"""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

rng = np.random.default_rng(0)
X, y = make_classification(n_samples=30000, n_features=10, n_informative=5,
                           weights=[0.97], flip_y=0.01, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, stratify=y, random_state=0)


def oversample(X, y):
    # Duplicate random minority rows until the classes are equal in size.
    pos = np.flatnonzero(y == 1)
    extra = rng.choice(pos, (y == 0).sum() - len(pos))
    return np.vstack([X, X[extra]]), np.r_[y, y[extra]]


def smote(X, y, k=5):
    # SMOTE [Chawla et al., 2002]: new minority points on line segments
    # between a minority point and one of its k nearest minority neighbors.
    P = X[y == 1]
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(P).kneighbors(P)[1][:, 1:]
    m = (y == 0).sum() - len(P)
    i = rng.integers(len(P), size=m)
    j = nbrs[i, rng.integers(k, size=m)]
    new = P[i] + rng.random((m, 1)) * (P[j] - P[i])
    return np.vstack([X, new]), np.r_[y, np.ones(m, int)]


variants = {
    "no correction": (Xtr, ytr, None),
    "class_weight='balanced'": (Xtr, ytr, "balanced"),
    "random oversampling": (*oversample(Xtr, ytr), None),
    "SMOTE": (*smote(Xtr, ytr), None),
}
print(f"positive rate: train {ytr.mean():.3f}, test {yte.mean():.3f}\n")
print("variant                   ROC AUC  avg prec  Brier   mean p   prec@0.5 rec@0.5")
probs = {}
for name, (Xf, yf, cw) in variants.items():
    p = LogisticRegression(class_weight=cw, max_iter=1000).fit(Xf, yf).predict_proba(Xte)[:, 1]
    probs[name] = p
    pred = p >= 0.5
    print(f"{name:25s} {roc_auc_score(yte, p):7.3f} {average_precision_score(yte, p):8.3f} "
          f"{brier_score_loss(yte, p):7.4f} {p.mean():7.3f} "
          f"{precision_score(yte, pred, zero_division=0):8.3f} {recall_score(yte, pred):7.3f}")

# Threshold moving: keep the uncorrected model, lower its threshold until it
# flags as many test cases as the balanced model did at 0.5. (In practice,
# choose the threshold on a validation set from the costs of each error.)
n_flag = (probs["class_weight='balanced'"] >= 0.5).sum()
t = np.sort(probs["no correction"])[::-1][n_flag - 1]
pred = probs["no correction"] >= t
print(f"\nno correction, threshold moved to {t:.3f} (flags {n_flag} cases): "
      f"precision {precision_score(yte, pred):.3f}, recall {recall_score(yte, pred):.3f}")

# Reweighting to 50/50 multiplies the model's odds by about (1 - pi) / pi,
# where pi is the real positive rate. Undo that to get usable probabilities.
pi = ytr.mean()
pw = probs["class_weight='balanced'"]
odds = pw / (1 - pw) * pi / (1 - pi)
p_fix = odds / (1 + odds)
print(f"balanced model after odds correction: Brier {brier_score_loss(yte, p_fix):.4f}, "
      f"mean p {p_fix.mean():.3f}, ROC AUC {roc_auc_score(yte, p_fix):.3f} (ranking unchanged)")
