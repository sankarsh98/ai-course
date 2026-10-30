"""Ridge (closed form) and lasso (coordinate descent) from scratch on the
scikit-learn diabetes data, checked against scikit-learn, plus a look at how
the penalty strength trades training fit for test error.
"""
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Lasso, LassoCV, Ridge
from sklearn.model_selection import train_test_split

X, y = load_diabetes(return_X_y=True)
names = load_diabetes().feature_names
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

# Standardize with training statistics only, and center y, so the intercept
# drops out of the penalized problem (we never penalize the intercept).
mu, sd, y_mu = X_tr.mean(0), X_tr.std(0), y_tr.mean()
Z_tr, Z_te = (X_tr - mu) / sd, (X_te - mu) / sd
yc = y_tr - y_mu
n, p = Z_tr.shape


def ridge(Z, yc, alpha):
    """Minimize ||yc - Z w||^2 + alpha ||w||^2 (same objective as sklearn Ridge)."""
    return np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ yc)


def soft_threshold(rho, lam):
    return np.sign(rho) * max(abs(rho) - lam, 0.0)


def lasso_cd(Z, yc, alpha, n_sweeps=1000, tol=1e-8):
    """Minimize (1/2n)||yc - Z w||^2 + alpha ||w||_1 (sklearn Lasso objective)."""
    n, p = Z.shape
    w = np.zeros(p)
    col_sq = (Z ** 2).sum(0)
    r = yc.copy()                      # residual yc - Z w, kept up to date
    for _ in range(n_sweeps):
        max_change = 0.0
        for j in range(p):
            r += Z[:, j] * w[j]        # remove feature j's contribution
            rho = Z[:, j] @ r          # correlation of feature j with what is left
            w_new = soft_threshold(rho, n * alpha) / col_sq[j]
            r -= Z[:, j] * w_new
            max_change = max(max_change, abs(w_new - w[j]))
            w[j] = w_new
        if max_change < tol:
            break
    return w


def test_mse(w):
    return np.mean((Z_te @ w + y_mu - y_te) ** 2)


# 1. Agreement with scikit-learn.
w_r = ridge(Z_tr, yc, alpha=10.0)
w_r_sk = Ridge(alpha=10.0).fit(Z_tr, y_tr).coef_
w_l = lasso_cd(Z_tr, yc, alpha=1.0)
w_l_sk = Lasso(alpha=1.0, tol=1e-10, max_iter=100_000).fit(Z_tr, y_tr).coef_
print("max |ours - sklearn|  ridge:", f"{np.abs(w_r - w_r_sk).max():.2e}",
      " lasso:", f"{np.abs(w_l - w_l_sk).max():.2e}")

# 2. Coefficients as the penalty grows. Ridge shrinks; lasso zeroes out.
w_ols = np.linalg.lstsq(Z_tr, yc, rcond=None)[0]
print("\nfeature    OLS   ridge(a=100)  lasso(a=5)")
w_r100, w_l5 = ridge(Z_tr, yc, 100.0), lasso_cd(Z_tr, yc, 5.0)
for j, nm in enumerate(names):  # "+ 0.0" turns -0.0 into 0.0 for printing
    print(f"{nm:6s} {w_ols[j]:7.1f} {w_r100[j] + 0.0:10.1f} {w_l5[j] + 0.0:10.1f}")

# 3. Train vs test error along the lasso path.
print("\nalpha   nonzero  train MSE  test MSE")
for a in [0.01, 0.1, 0.5, 1, 2, 5, 10, 20, 40]:
    w = lasso_cd(Z_tr, yc, a)
    tr = np.mean((Z_tr @ w - yc) ** 2)
    print(f"{a:5.2f}   {np.sum(w != 0):5d}   {tr:8.0f}  {test_mse(w):8.0f}")

# 4. Let cross-validation on the training set choose alpha.
cv = LassoCV(cv=5, random_state=0).fit(Z_tr, y_tr)
print(f"\nLassoCV alpha = {cv.alpha_:.3f}, nonzero = {np.sum(cv.coef_ != 0)}, "
      f"test MSE = {test_mse(cv.coef_):.0f}")
print(f"OLS test MSE = {test_mse(w_ols):.0f}; predicting the mean: "
      f"{np.mean((y_mu - y_te) ** 2):.0f}")

# 5. A high-variance regime: only 40 training rows plus 30 pure-noise columns.
rng = np.random.default_rng(0)
idx = rng.choice(n, 40, replace=False)
noise_tr, noise_te = rng.normal(size=(40, 30)), rng.normal(size=(len(y_te), 30))
S_tr, S_te = np.hstack([Z_tr[idx], noise_tr]), np.hstack([Z_te, noise_te])
ys = y_tr[idx]
ys_mu = ys.mean()
print("\n40 rows, 10 real + 30 noise features:")
w = np.linalg.lstsq(S_tr, ys - ys_mu, rcond=None)[0]
print(f"  OLS        test MSE = {np.mean((S_te @ w + ys_mu - y_te) ** 2):8.0f}")
for a in [10.0, 100.0]:
    w = ridge(S_tr, ys - ys_mu, a)
    print(f"  ridge a={a:<4.0f} test MSE = {np.mean((S_te @ w + ys_mu - y_te) ** 2):8.0f}")
for a in [2.0, 5.0, 15.0]:
    w = lasso_cd(S_tr, ys - ys_mu, a)
    print(f"  lasso a={a:<4.0f} test MSE = {np.mean((S_te @ w + ys_mu - y_te) ** 2):8.0f}"
          f"  (nonzero real: {np.sum(w[:10] != 0)}, nonzero noise: {np.sum(w[10:] != 0)})")
