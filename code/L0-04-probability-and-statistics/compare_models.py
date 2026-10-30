"""Two classifiers, one test set: confidence intervals and a paired test for the difference."""
import numpy as np
from scipy import stats
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from intervals import wald, wilson

X, y = load_breast_cancer(return_X_y=True)  # 569 examples, bundled with scikit-learn
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
model_a = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X_tr, y_tr)
model_b = DecisionTreeClassifier(random_state=0).fit(X_tr, y_tr)
correct_a = model_a.predict(X_te) == y_te
correct_b = model_b.predict(X_te) == y_te
n = len(y_te)


def bootstrap_ci(values, reps=10_000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(reps, len(values)))  # resample test items
    return np.percentile(values[idx].mean(axis=1), [2.5, 97.5])


print(f"test set size n = {n}")
for name, c in [("A: logistic regression", correct_a), ("B: decision tree", correct_b)]:
    k = c.sum()
    (w1, w2), (s1, s2), (b1, b2) = wald(k, n), wilson(k, n), bootstrap_ci(c.astype(float))
    print(f"{name:<23} acc={k / n:.4f}  Wald [{w1:.3f}, {w2:.3f}]  Wilson [{s1:.3f}, {s2:.3f}]  "
          f"bootstrap [{b1:.3f}, {b2:.3f}]")

# Paired view: both models saw the same items, so look at where they disagree.
only_a = int(np.sum(correct_a & ~correct_b))
only_b = int(np.sum(~correct_a & correct_b))
print(f"\nboth right {np.sum(correct_a & correct_b)}, both wrong {np.sum(~correct_a & ~correct_b)}, "
      f"only A right {only_a}, only B right {only_b}")
# Exact McNemar test: under 'equal accuracy', each disagreement is a fair coin flip.
print(f"exact McNemar p-value: {stats.binomtest(only_a, only_a + only_b, 0.5).pvalue:.4f}")

diff = correct_a.astype(float) - correct_b.astype(float)
lo, hi = bootstrap_ci(diff)  # resampling items keeps each A/B pair together
print(f"accuracy difference A - B = {diff.mean():+.4f}, paired bootstrap 95% CI [{lo:+.4f}, {hi:+.4f}]")
se_unpaired = np.sqrt(correct_a.var(ddof=1) / n + correct_b.var(ddof=1) / n)
print(f"standard error of the difference: unpaired {se_unpaired:.4f}, "
      f"paired {diff.std(ddof=1) / np.sqrt(n):.4f}")
