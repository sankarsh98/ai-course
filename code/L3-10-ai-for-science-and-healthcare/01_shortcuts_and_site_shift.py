"""SYNTHETIC DATA ONLY. NOT MEDICAL ADVICE. No real patients, features or
effect sizes are represented; the numbers are invented to show a method.

A clinical-prediction toy with four hospitals. At the two development sites,
sicker patients are more often imaged on a portable device (a shortcut), and
the sites differ in prevalence and in a scanner signature. Compare internal,
leave-one-site-out and external validation, with and without the shortcut."""
import os

# Many small fits: extra threads mostly add overhead, and on a shared machine
# they can slow things down a lot. Must be set before numpy is imported.
for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(var, "4")

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(0)
PHYS = ["age", "lab_a", "lab_b", "lab_c"]
ALL = PHYS + ["portable", "scanner"]


def make_site(n, base_logit, portable_if_ill, portable_if_well, scanner_mu, age_shift=0.0):
    """Outcome depends only on physiology (plus a site-level baseline risk).
    'portable' depends on the outcome; 'scanner' only identifies the site."""
    age = rng.normal(age_shift, 1, n)
    lab_a, lab_b, lab_c = rng.normal(0, 1, (3, n))
    logit = base_logit + 0.8 * age + 1.0 * lab_a + 0.6 * lab_b + 0.5 * lab_a * lab_b
    y = rng.random(n) < 1 / (1 + np.exp(-logit))
    portable = np.where(y, rng.random(n) < portable_if_ill, rng.random(n) < portable_if_well)
    scanner = rng.normal(scanner_mu, 0.3, n)
    return np.column_stack([age, lab_a, lab_b, lab_c, portable, scanner]), y.astype(int)


sites = {  # development sites A, B; external C (shortcut absent), D (shortcut present)
    "A": make_site(5000, -2.5, 0.80, 0.10, 0.0),
    "B": make_site(5000, -0.5, 0.80, 0.10, 2.0),
    "C": make_site(5000, -1.5, 0.30, 0.30, 1.0, age_shift=0.5),
    "D": make_site(5000, -1.5, 0.80, 0.10, 1.0),
}
for s, (X, y) in sites.items():
    print(f"site {s}: n={len(y)}, outcome rate {y.mean():.1%}, "
          f"portable among ill {X[y == 1, 4].mean():.0%}, among well {X[y == 0, 4].mean():.0%}")


def metrics(model, X, y):
    """AUC (discrimination), calibration-in-the-large (observed minus mean
    predicted risk), calibration slope (1 is ideal) and Brier score."""
    p = np.clip(model.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p))[:, None]
    slope = LogisticRegression(C=1e6).fit(logit, y).coef_[0, 0]
    return roc_auc_score(y, p), y.mean() - p.mean(), slope, brier_score_loss(y, p)


def make_model():
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=0)


XA, yA = sites["A"]; XB, yB = sites["B"]
X_dev, y_dev = np.vstack([XA, XB]), np.r_[yA, yB]
site_dev = np.r_[np.zeros(len(yA)), np.ones(len(yB))]

# Can a model tell the development sites apart from the inputs alone?
Xs_tr, Xs_te, ss_tr, ss_te = train_test_split(X_dev, site_dev, test_size=0.3, random_state=0)
site_clf = make_model().fit(Xs_tr, ss_tr)
print(f"\nsite A vs B identifiable from the inputs: AUC {roc_auc_score(ss_te, site_clf.predict_proba(Xs_te)[:, 1]):.3f}")

print(f"\n{'features':<11} {'evaluation':<26} {'AUC':>5} {'obs-pred':>9} {'slope':>6} {'Brier':>6}")
for label, cols in [("all", list(range(6))), ("physiology", list(range(4)))]:
    Xtr, Xte, ytr, yte = train_test_split(X_dev[:, cols], y_dev, test_size=0.25,
                                          stratify=y_dev, random_state=0)
    model = make_model().fit(Xtr, ytr)
    rows = [("internal (random split)", metrics(model, Xte, yte))]
    # Leave one development site out: train on A, test on B, and vice versa.
    for tr, te in [("A", "B"), ("B", "A")]:
        m = make_model().fit(sites[tr][0][:, cols], sites[tr][1])
        rows.append((f"train {tr} -> test {te}", metrics(m, sites[te][0][:, cols], sites[te][1])))
    for ext in ["C", "D"]:
        rows.append((f"external site {ext}", metrics(model, sites[ext][0][:, cols], sites[ext][1])))
    for name, (auc, citl, slope, brier) in rows:
        print(f"{label:<11} {name:<26} {auc:5.3f} {citl:+9.3f} {slope:6.2f} {brier:6.3f}")
    if label == "all":
        imp = permutation_importance(model, Xte, yte, scoring="roc_auc", n_repeats=5, random_state=0)
        order = np.argsort(-imp.importances_mean)
        print("  permutation importance (drop in internal AUC): "
              + ", ".join(f"{ALL[i]} {imp.importances_mean[i]:.3f}" for i in order[:4]))
