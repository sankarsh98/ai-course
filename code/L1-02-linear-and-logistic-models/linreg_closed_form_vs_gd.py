"""Linear regression three ways: normal equations, gradient descent, scikit-learn.

Also shows why feature scaling matters for gradient descent: the same problem
needs far more iterations when features live on very different scales.
"""
import numpy as np
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng(0)
n = 200
# Two features on very different scales (think "rooms" and "square metres").
x1 = rng.normal(3, 1, n)
x2 = rng.normal(100, 30, n)
X = np.column_stack([x1, x2])
w_true, b_true = np.array([2.0, 0.05]), 1.0
y = X @ w_true + b_true + rng.normal(0, 0.5, n)


def add_bias(X):
    """Prepend a column of ones so the intercept is just another weight."""
    return np.column_stack([np.ones(len(X)), X])


def closed_form(X, y):
    A = add_bias(X)
    # Solve (A^T A) w = A^T y. lstsq is more stable than forming an inverse.
    w, *_ = np.linalg.lstsq(A, y, rcond=None)
    return w


def gradient_descent(X, y, lr, tol=1e-10, max_iter=200_000):
    A = add_bias(X)
    w = np.zeros(A.shape[1])
    for it in range(1, max_iter + 1):
        grad = (2 / len(y)) * A.T @ (A @ w - y)  # gradient of the mean squared error
        w -= lr * grad
        if np.linalg.norm(grad) < tol:
            return w, it
    return w, max_iter


def mse(X, y, w):
    return np.mean((add_bias(X) @ w - y) ** 2)


w_cf = closed_form(X, y)
print("closed form       [b, w1, w2] =", np.round(w_cf, 4), " MSE =", round(mse(X, y, w_cf), 4))

sk = LinearRegression().fit(X, y)
print("scikit-learn      [b, w1, w2] =", np.round(np.r_[sk.intercept_, sk.coef_], 4))

mu, sd = X.mean(0), X.std(0)
# Largest stable step is 2 / (largest eigenvalue of the Hessian 2/n A^T A).
# We use 1 / largest eigenvalue, a safe choice for both versions.
for name, Xs in [("raw features", X), ("standardized", (X - mu) / sd)]:
    A = add_bias(Xs)
    eig = np.linalg.eigvalsh(2 / n * A.T @ A)
    w, iters = gradient_descent(Xs, y, lr=1.0 / eig.max())
    print(f"GD on {name:13s}: condition number = {eig.max() / eig.min():10.1f}, "
          f"iterations = {iters:6d}, MSE = {mse(Xs, y, w):.4f}")

# The loop ends with w fitted on standardized features: map back to original units.
w_std = w
w_orig = w_std[1:] / sd
b_orig = w_std[0] - np.sum(w_std[1:] * mu / sd)
print("GD (standardized, mapped back) [b, w1, w2] =", np.round(np.r_[b_orig, w_orig], 4))
