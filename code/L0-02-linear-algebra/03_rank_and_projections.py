"""Rank, null space, least squares as projection, and conditioning."""
import numpy as np

rng = np.random.default_rng(0)

# 1. Rank = number of independent columns = dimension of the column space.
A = np.array([[1.0, 2.0, 3.0],
              [2.0, 4.0, 6.1],
              [1.0, 0.0, 1.0]])
B = A.copy()
B[1, 2] = 6.0                 # now row 2 = 2 * row 1 exactly
print("rank(A) =", np.linalg.matrix_rank(A), "| rank(B) =", np.linalg.matrix_rank(B))

# Null space of B: directions B sends to zero. From the SVD (see script 05),
# right singular vectors with zero singular value span it.
_, s, Vt = np.linalg.svd(B)
null = Vt[-1]
print("singular values of B:", np.round(s, 6))
print("null vector n =", np.round(null, 4), "| B @ n =", np.round(B @ null, 12))

# 2. Rank of a product never exceeds the smaller rank (a "bottleneck").
U = rng.normal(size=(100, 5))
V = rng.normal(size=(5, 80))
print("\nrank of (100x5)@(5x80) =", np.linalg.matrix_rank(U @ V), "out of max 80")

# 3. Least squares: find w minimizing ||X w - y||^2.
#    Geometrically, X w_hat is the projection of y onto the column space of X.
n = 50
x = rng.uniform(0, 10, size=n)
X = np.column_stack([np.ones(n), x])          # intercept and slope
y = 1.5 + 0.8 * x + rng.normal(scale=1.0, size=n)
w_hat, *_ = np.linalg.lstsq(X, y, rcond=None)
w_normal = np.linalg.solve(X.T @ X, X.T @ y)  # normal equations X^T X w = X^T y
print("\nlstsq w =", np.round(w_hat, 4), "| normal equations w =", np.round(w_normal, 4))

P = X @ np.linalg.inv(X.T @ X) @ X.T          # projection matrix onto col(X)
y_hat = P @ y
r = y - y_hat
print("P is symmetric:", np.allclose(P, P.T), "| P @ P == P:", np.allclose(P @ P, P))
print("residual orthogonal to every column of X: X^T r =", np.round(X.T @ r, 10))
print(f"trace(P) = {np.trace(P):.6f} (= rank of X = number of parameters)")
print(f"Pythagoras: ||y||^2={y@y:.3f} = ||y_hat||^2 + ||r||^2 = {y_hat@y_hat + r@r:.3f}")

# 4. Conditioning: nearly dependent columns make solutions unstable.
x2 = x + rng.normal(scale=1e-6, size=n)       # an almost duplicate feature
Xbad = np.column_stack([np.ones(n), x, x2])
print(f"\ncond(X) = {np.linalg.cond(X):.1f}  cond(Xbad) = {np.linalg.cond(Xbad):.2e}")
print(f"cond(Xbad^T Xbad) = {np.linalg.cond(Xbad.T @ Xbad):.2e}  (squared!)")
w1, *_ = np.linalg.lstsq(Xbad, y, rcond=None)
w2, *_ = np.linalg.lstsq(Xbad, y + rng.normal(scale=1e-3, size=n), rcond=None)
print("weights:                    ", np.round(w1, 2))
print("weights after tiny y change:", np.round(w2, 2))
print("fitted values barely move:  max |change| =",
      f"{np.abs(Xbad @ w1 - Xbad @ w2).max():.2e}")
