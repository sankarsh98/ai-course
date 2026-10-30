"""SYNTHETIC DATA ONLY. NOT MEDICAL ADVICE. Groups, features and rates are
invented to show a method; they do not describe any real population.

Two groups with the same distribution of true illness. G2 has less access to
care, so it has fewer prior visits at the same severity, and its illness is
recorded (diagnosed) less often. One measurement is also noisier for G2.
Audit the model per group, against the recorded label and against the true
condition, with bootstrap intervals. Then change the training label."""
import os

# Many small fits: extra threads mostly add overhead, and on a shared machine
# they can slow things down a lot. Must be set before numpy is imported.
for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(var, "4")

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(0)


def make_population(n, frac_g2=0.15):
    g2 = rng.random(n) < frac_g2
    severity = rng.normal(0, 1, n)                    # same distribution in both groups
    ill = severity > 0.85                             # about 20% truly ill
    x1 = severity + rng.normal(0, 0.6, n)
    x2 = severity + rng.normal(0, np.where(g2, 1.2, 0.6), n)   # noisier for G2
    access = np.where(g2, 0.5, 1.0)
    visits = rng.poisson(np.exp(0.5 + 0.7 * severity) * access)   # utilization
    recorded = ill & (rng.random(n) < np.where(g2, 0.55, 0.9))   # under-recorded in G2
    X = np.column_stack([x1, x2, np.log1p(visits)])
    return X, ill.astype(int), recorded.astype(int), g2


def boot_ci(fn, *arrays, reps=1000):
    """Percentile bootstrap 95% interval of fn over resampled rows."""
    n = len(arrays[0])
    vals = []
    for _ in range(reps):
        i = rng.integers(0, n, n)
        vals.append(fn(*(a[i] for a in arrays)))
    return np.percentile(vals, [2.5, 97.5])


def sens(y, flag):
    return flag[y == 1].mean()


X_tr, ill_tr, rec_tr, g_tr = make_population(20000)
X_te, ill_te, rec_te, g_te = make_population(4000)
print(f"test set: {np.sum(~g_te)} in G1, {np.sum(g_te)} in G2")
print(f"true illness rate   G1 {ill_te[~g_te].mean():.1%}   G2 {ill_te[g_te].mean():.1%}")
print(f"recorded label rate G1 {rec_te[~g_te].mean():.1%}   G2 {rec_te[g_te].mean():.1%}")

configs = [("recorded label", rec_tr, [0, 1, 2]),
           ("true condition", ill_tr, [0, 1, 2]),
           ("true condition, no visits", ill_tr, [0, 1])]
print(f"\n{'trained on':<26} {'group':<5} {'AUC':>5} {'95% CI':>14} {'sens.':>6} "
      f"{'95% CI':>14} {'flagged':>8} {'mean risk':>9}")
for train_label, y_train, cols in configs:
    model = LogisticRegression().fit(X_tr[:, cols], y_train)   # group is NOT an input
    p = model.predict_proba(X_te[:, cols])[:, 1]
    flag = p >= np.quantile(p, 0.80)                  # flag the top 20% for follow-up
    for gname, mask in [("G1", ~g_te), ("G2", g_te)]:
        y_g, p_g, f_g = ill_te[mask], p[mask], flag[mask]   # judged against the truth
        a_lo, a_hi = boot_ci(roc_auc_score, y_g, p_g)
        s_lo, s_hi = boot_ci(sens, y_g, f_g)
        print(f"{train_label:<26} {gname:<5} {roc_auc_score(y_g, p_g):5.3f} [{a_lo:.3f}, {a_hi:.3f}] "
              f"{sens(y_g, f_g):6.2f} [{s_lo:.2f}, {s_hi:.2f}] {f_g.mean():8.1%} {p_g.mean():9.3f}")
    if train_label == "recorded label":
        coef_note = ", ".join(f"{c:.2f}" for c in model.coef_[0])
        audit = [roc_auc_score(rec_te[m], p[m]) for m in (~g_te, g_te)]
        print(f"{'':<26} (coefficients x1, x2, log visits: {coef_note}; audited against the"
              f" recorded label instead, AUC G1 {audit[0]:.3f}, G2 {audit[1]:.3f})")
print(f"\ntrue illness rate: G1 {ill_te[~g_te].mean():.3f}, G2 {ill_te[g_te].mean():.3f}")
