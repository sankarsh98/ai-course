"""Putting error bars on a test score, and comparing two models on the same test set."""
import numpy as np
from scipy.stats import chi2 as chi2_dist
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(0)

def bootstrap_ci(values, n_boot=10_000, level=0.95):
    """Percentile bootstrap CI for the mean of per-example values (e.g. 0/1 correct)."""
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - level) / 2, (1 + level) / 2])
    return values.mean(), lo, hi

# --- 1. How wide is the interval for a test-set accuracy? ---
print("1. 95% intervals for an accuracy of about 0.85")
print(f"   {'n test':>7} {'observed':>9} {'normal approx':>16} {'bootstrap':>16}")
for n in [100, 1000, 10_000]:
    correct = (rng.random(n) < 0.85).astype(float)
    acc = correct.mean()
    half = 1.96 * np.sqrt(acc * (1 - acc) / n)          # binomial standard error
    _, lo, hi = bootstrap_ci(correct)
    print(f"   {n:7d} {acc:9.3f}   [{acc - half:.3f}, {acc + half:.3f}]   [{lo:.3f}, {hi:.3f}]")

# --- 2. Two models, one test set: use paired differences ---
X, y = make_classification(n_samples=3000, n_features=20, n_informative=6,
                           flip_y=0.05, random_state=1)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=500, random_state=1)
a = LogisticRegression(max_iter=1000).fit(X_tr, y_tr).predict(X_te) == y_te
b = RandomForestClassifier(n_estimators=300, random_state=0).fit(X_tr, y_tr).predict(X_te) == y_te
print(f"\n2. test n = {len(y_te)}: logistic acc {a.mean():.3f}, random forest acc {b.mean():.3f}")
for name, v in [("logistic", a), ("forest", b)]:
    m, lo, hi = bootstrap_ci(v.astype(float))
    print(f"   {name:>8} alone:      {m:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
m, lo, hi = bootstrap_ci(b.astype(float) - a.astype(float))
print(f"   paired difference: {m:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")
n01, n10 = int(np.sum(a & ~b)), int(np.sum(~a & b))
# McNemar's test uses only the examples where the two models disagree.
chi2 = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
print(f"   disagreements: logistic-only right {n01}, forest-only right {n10}, "
      f"McNemar chi2 = {chi2:.2f}, p = {chi2_dist.sf(chi2, df=1):.3f}")

# --- 3. Is the gap stable across splits? (Splits overlap, so these are not
#        independent replications; treat the spread as a rough guide only.) ---
gaps = []
for seed in range(10):
    Xa, Xb, ya, yb = train_test_split(X, y, test_size=500, random_state=seed)
    la = LogisticRegression(max_iter=1000).fit(Xa, ya).score(Xb, yb)
    fa = RandomForestClassifier(n_estimators=300, random_state=seed).fit(Xa, ya).score(Xb, yb)
    gaps.append(fa - la)
print(f"\n3. forest minus logistic over 10 random splits: "
      f"mean {np.mean(gaps):+.3f}, min {min(gaps):+.3f}, max {max(gaps):+.3f}")
