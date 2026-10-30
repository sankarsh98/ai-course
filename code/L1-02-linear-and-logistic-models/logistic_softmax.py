"""Logistic regression and softmax regression from scratch with gradient
descent, compared with scikit-learn, plus what happens on separable data.
"""
import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def sigmoid(z):
    # Split by sign so exp never overflows.
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1 / (1 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1 + ez)
    return out


def fit_logistic(X, y, C=np.inf, lr=0.5, n_iter=5000):
    """Minimize mean log-loss + ||w||^2 / (2 C n), i.e. sklearn's objective / n.
    The intercept b is not penalized."""
    n, p = X.shape
    w, b = np.zeros(p), 0.0
    lam = 0.0 if np.isinf(C) else 1.0 / (C * n)
    for _ in range(n_iter):
        err = sigmoid(X @ w + b) - y          # gradient of log-loss wrt the logit
        w -= lr * (X.T @ err / n + lam * w)
        b -= lr * err.mean()
    return w, b


def log_loss(y, p, eps=1e-15):
    p = np.clip(p, eps, 1 - eps)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


# ---- 1. Binary: breast cancer data ----
X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
sc = StandardScaler().fit(X_tr)
X_tr, X_te = sc.transform(X_tr), sc.transform(X_te)

w, b = fit_logistic(X_tr, y_tr, C=1.0)
sk = LogisticRegression(C=1.0, tol=1e-10, max_iter=10_000).fit(X_tr, y_tr)
p_ours, p_sk = sigmoid(X_te @ w + b), sk.predict_proba(X_te)[:, 1]
print("binary, C=1:  max |w_ours - w_sklearn| =", f"{np.abs(w - sk.coef_[0]).max():.2e}")
print(f"  test accuracy ours {np.mean((p_ours > 0.5) == y_te):.3f}, "
      f"sklearn {sk.score(X_te, y_te):.3f}")
print(f"  test log-loss ours {log_loss(y_te, p_ours):.4f}, sklearn {log_loss(y_te, p_sk):.4f}")
# Odds ratio per standard deviation for the largest-magnitude weight.
j = np.argmax(np.abs(w))
print(f"  largest |w|: feature '{load_breast_cancer().feature_names[j]}', w = {w[j]:.2f}, "
      f"odds multiplier per +1 SD = exp(w) = {np.exp(w[j]):.2f}")

# ---- 2. Separable data: without a penalty, weights grow without bound ----
x_sep = np.array([[-2.0], [-1.0], [-0.5], [0.5], [1.0], [2.0]])
y_sep = np.array([0, 0, 0, 1, 1, 1])
print("\nseparable toy data, |w| after k gradient steps:")
for k in [100, 1_000, 10_000, 100_000]:
    w0, _ = fit_logistic(x_sep, y_sep, C=np.inf, lr=1.0, n_iter=k)
    w1, _ = fit_logistic(x_sep, y_sep, C=1.0, lr=1.0, n_iter=k)
    print(f"  k={k:6d}   no penalty: {abs(w0[0]):6.2f}    C=1: {abs(w1[0]):5.2f}")


# ---- 3. Multiclass: softmax regression on 8x8 digit images ----
def softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)  # log-sum-exp trick: same result, no overflow
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def fit_softmax(X, y, K, C=1.0, lr=0.5, n_iter=3000):
    n, p = X.shape
    W, b = np.zeros((p, K)), np.zeros(K)
    Y = np.eye(K)[y]                          # one-hot targets
    lam = 1.0 / (C * n)
    for _ in range(n_iter):
        G = softmax(X @ W + b) - Y            # gradient of cross-entropy wrt logits
        W -= lr * (X.T @ G / n + lam * W)
        b -= lr * G.mean(0)
    return W, b


Xd, yd = load_digits(return_X_y=True)
Xd_tr, Xd_te, yd_tr, yd_te = train_test_split(Xd, yd, test_size=0.3, random_state=0, stratify=yd)
scd = StandardScaler().fit(Xd_tr)
Xd_tr, Xd_te = scd.transform(Xd_tr), scd.transform(Xd_te)
W, bd = fit_softmax(Xd_tr, yd_tr, K=10, C=1.0)
P = softmax(Xd_te @ W + bd)
skd = LogisticRegression(C=1.0, max_iter=5000).fit(Xd_tr, yd_tr)
print(f"\ndigits (10 classes): test accuracy ours {np.mean(P.argmax(1) == yd_te):.3f}, "
      f"sklearn {skd.score(Xd_te, yd_te):.3f}")
print(f"  rows of P sum to 1: {np.allclose(P.sum(1), 1)}; "
      f"mean max-probability {P.max(1).mean():.3f}")
