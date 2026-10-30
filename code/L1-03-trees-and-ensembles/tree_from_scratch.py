"""A CART-style classification tree from scratch (Gini or entropy), compared
with scikit-learn's DecisionTreeClassifier at several depths.
"""
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def impurity(counts, criterion):
    """counts: (..., K) class counts. Returns Gini or entropy per row."""
    n = counts.sum(-1, keepdims=True)
    p = counts / np.maximum(n, 1)
    if criterion == "gini":
        return 1 - (p ** 2).sum(-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.nansum(np.where(p > 0, p * np.log2(p), 0.0), axis=-1)


def best_split(X, y, K, criterion):
    """Scan every feature and every midpoint between sorted distinct values.
    Returns (feature, threshold, weighted child impurity) or None."""
    n = len(y)
    best = (None, None, impurity(np.bincount(y, minlength=K), criterion))
    for j in range(X.shape[1]):
        order = np.argsort(X[:, j], kind="stable")
        xs, ys = X[order, j], y[order]
        left = np.cumsum(np.eye(K)[ys], axis=0)[:-1]   # counts left of each cut
        right = left[-1] + np.eye(K)[ys[-1]] - left     # total minus left
        n_left = np.arange(1, n)
        score = (n_left * impurity(left, criterion)
                 + (n - n_left) * impurity(right, criterion)) / n
        score[xs[1:] == xs[:-1]] = np.inf               # can't cut between equal values
        i = np.argmin(score)
        if score[i] < best[2] - 1e-12:
            best = (j, (xs[i] + xs[i + 1]) / 2, score[i])
    return best if best[0] is not None else None


def build(X, y, K, depth, max_depth, criterion, min_samples=2):
    counts = np.bincount(y, minlength=K)
    leaf = {"leaf": True, "pred": int(np.argmax(counts))}
    if depth == max_depth or len(y) < min_samples or counts.max() == len(y):
        return leaf
    split = best_split(X, y, K, criterion)
    if split is None:
        return leaf
    j, t, _ = split
    go_left = X[:, j] <= t
    return {"leaf": False, "j": j, "t": t,
            "l": build(X[go_left], y[go_left], K, depth + 1, max_depth, criterion),
            "r": build(X[~go_left], y[~go_left], K, depth + 1, max_depth, criterion)}


def predict_one(node, x):
    while not node["leaf"]:
        node = node["l"] if x[node["j"]] <= node["t"] else node["r"]
    return node["pred"]


def predict(tree, X):
    return np.array([predict_one(tree, x) for x in X])


# A worked example: 10 points, one feature.
x_toy = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]], dtype=float)
y_toy = np.array([0, 0, 0, 0, 1, 0, 1, 1, 1, 1])
j, t, s = best_split(x_toy, y_toy, 2, "gini")
print(f"toy: parent Gini = {impurity(np.bincount(y_toy), 'gini'):.3f}; "
      f"best cut x <= {t}, weighted child Gini = {s:.3f}")

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
print("\ndepth  criterion  ours train/test   sklearn train/test")
for crit in ["gini", "entropy"]:
    for d in [1, 2, 3, 5, 8, None]:
        tree = build(X_tr, y_tr, 2, 0, d if d is not None else 10**9, crit)
        sk = DecisionTreeClassifier(criterion=crit, max_depth=d, random_state=0).fit(X_tr, y_tr)
        print(f"{str(d):>5s}  {crit:9s}  {np.mean(predict(tree, X_tr) == y_tr):.3f} / "
              f"{np.mean(predict(tree, X_te) == y_te):.3f}     "
              f"{sk.score(X_tr, y_tr):.3f} / {sk.score(X_te, y_te):.3f}")

# sklearn visits features in a random order, so tied splits are broken
# differently per seed, even with all features considered.
accs = [DecisionTreeClassifier(random_state=s).fit(X_tr, y_tr).score(X_te, y_te)
        for s in range(10)]
print("\nfull-depth sklearn tree, test accuracy over 10 seeds:", np.round(accs, 3))
