"""SYNTHETIC DATA ONLY. NOT MEDICAL ADVICE. All rates are invented.

What a fixed model means at the bedside: (1) the same alert threshold at
different prevalences, (2) net benefit (decision curves) for a calibrated
and a miscalibrated version, (3) calibration drift over time and the effect
of recalibrating on recent data."""
import os

# Many small fits: extra threads mostly add overhead, and on a shared machine
# they can slow things down a lot. Must be set before numpy is imported.
for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(var, "4")

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(0)
COEF = np.array([1.0, 0.7, 0.5])


def cohort(n, intercept, coef=COEF):
    X = rng.normal(0, 1, (n, 3))
    p = 1 / (1 + np.exp(-(intercept + X @ coef)))
    return X, (rng.random(n) < p).astype(int)


# ---- 1. one model, one threshold, three prevalences -------------------------
X_dev, y_dev = cohort(20000, -2.6)
model = LogisticRegression().fit(X_dev, y_dev)
thr = np.quantile(model.predict_proba(X_dev)[:, 1], 0.90)   # alert on top 10% at development
print(f"development prevalence {y_dev.mean():.1%}; alert threshold {thr:.3f}\n")
print(f"{'prevalence':>10} {'AUC':>6} {'alerts/100':>11} {'PPV':>6} {'sensitivity':>12} {'alerts per true case':>21}")
for intercept in [-4.6, -2.6, -1.2]:
    X, y = cohort(20000, intercept)
    p = model.predict_proba(X)[:, 1]
    alert = p >= thr
    tp = np.sum(alert & (y == 1))
    print(f"{y.mean():10.1%} {roc_auc_score(y, p):6.3f} {100 * alert.mean():11.1f} "
          f"{tp / alert.sum():6.2f} {tp / y.sum():12.2f} {alert.sum() / tp:21.1f}")

# ---- 2. decision curves -------------------------------------------------------
# Net benefit at threshold probability pt (Vickers and Elkin, 2006):
#   NB = TP/n - FP/n * pt / (1 - pt)
# pt encodes how many false alarms one true case is worth: at pt = 0.2, 4.
X, y = cohort(20000, -2.6)
p_cal = model.predict_proba(X)[:, 1]
logit = np.log(p_cal / (1 - p_cal))
p_over = 1 / (1 + np.exp(-(logit + 1.5)))             # same ranking, risks inflated
print(f"\nnet benefit per 100 patients (prevalence {y.mean():.1%}; AUC identical for both models)")
print(f"{'pt':>5} {'treat none':>11} {'treat all':>10} {'calibrated':>11} {'overestimating':>15}")
for pt in [0.05, 0.10, 0.20, 0.30, 0.50]:
    w = pt / (1 - pt)
    nb = lambda q: (np.sum((q >= pt) & (y == 1)) - w * np.sum((q >= pt) & (y == 0))) / len(y)
    nb_all = y.mean() - w * (1 - y.mean())
    print(f"{pt:5.2f} {0:11.2f} {100 * nb_all:10.2f} {100 * nb(p_cal):11.2f} {100 * nb(p_over):15.2f}")

# ---- 3. temporal drift --------------------------------------------------------
# Each quarter the baseline risk creeps up and one predictor matters less
# (think of a change in practice or in how a test is ordered).
print("\nquarterly monitoring of the frozen model, and of a copy recalibrated each quarter")
print("on the previous quarter's data (intercept and slope of the logit only):")
print(f"{'quarter':>7} {'rate':>6} {'AUC':>6} {'frozen: obs-pred':>17} {'recalibrated: obs-pred':>23}")
prev = None
for q in range(8):
    coef_q = COEF * np.array([1, 1 - 0.08 * q, 1])
    Xq, yq = cohort(5000, -2.6 + 0.15 * q, coef_q)
    pq = model.predict_proba(Xq)[:, 1]
    lq = np.log(pq / (1 - pq))[:, None]
    line = f"{q + 1:7d} {yq.mean():6.1%} {roc_auc_score(yq, pq):6.3f} {yq.mean() - pq.mean():+17.3f}"
    if prev is not None:
        recal = LogisticRegression(C=1e6).fit(*prev)       # fit on last quarter only
        line += f" {yq.mean() - recal.predict_proba(lq)[:, 1].mean():+23.3f}"
    print(line)
    prev = (lq, yq)
