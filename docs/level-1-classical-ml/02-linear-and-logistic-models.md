# Linear and Logistic Models

> **Level 1 · Module 2** · Prerequisites: L0-02, L0-03, L0-04, L0-05, L1-01 · Time: 6 to 8 hours · Code: `code/L1-02-linear-and-logistic-models/`

## What you will be able to do

- Derive the least-squares solution, fit it in closed form and by gradient descent, and explain why feature scaling changes how fast gradient descent converges.
- Implement ridge (closed form) and lasso (coordinate descent) from scratch, match scikit-learn, and choose the penalty by cross-validation.
- Explain least squares, ridge, lasso, and logistic regression as maximum-likelihood or MAP estimates under explicit assumptions.
- Implement binary logistic and multiclass softmax regression from scratch, and explain what goes wrong on linearly separable data.
- Read a coefficient correctly, and name the situations in which it tells you less than it seems to.

## 1. The linear model and least squares

A linear model predicts a number as a weighted sum of the inputs plus an offset:

$$\hat{y} = w_1 x_1 + \dots + w_p x_p + b = \mathbf{w}^\top \mathbf{x} + b.$$

Here $\mathbf{x} \in \mathbb{R}^p$ is one example with $p$ features, $\mathbf{w}$ is the weight (coefficient) vector, and $b$ is the intercept. Stack $n$ examples as rows of $X \in \mathbb{R}^{n \times p}$ and the targets into $\mathbf{y} \in \mathbb{R}^n$. Appending a column of ones to $X$ turns the intercept into just another weight, so predictions are $X\mathbf{w}$; I do this silently when it shortens the algebra.

"Linear" means linear in the weights. You can feed in $x^2$, $\log x$, or products of features and everything below still applies (building such features is L1-05).

**Least squares** minimizes the sum of squared residuals:

$$L(\mathbf{w}) = \lVert \mathbf{y} - X\mathbf{w} \rVert^2 = \sum_{i=1}^n (y_i - \mathbf{w}^\top \mathbf{x}_i)^2.$$

The gradient is $\nabla L = -2X^\top(\mathbf{y} - X\mathbf{w})$. Setting it to zero gives the **normal equations**:

$$X^\top X\, \mathbf{w} = X^\top \mathbf{y} \quad\Longrightarrow\quad \hat{\mathbf{w}} = (X^\top X)^{-1} X^\top \mathbf{y} \text{ when } X^\top X \text{ is invertible.}$$

$L$ is a convex quadratic, so this is the global minimum. Geometrically (L0-02), $X\hat{\mathbf{w}}$ is the orthogonal projection of $\mathbf{y}$ onto the column space of $X$: the residual is perpendicular to every column, which is what $X^\top(\mathbf{y} - X\hat{\mathbf{w}}) = 0$ says.

**A worked example.** Three points: $(x, y) = (0, 1), (1, 3), (2, 4)$. With one feature, $w = \sum (x_i - \bar{x})(y_i - \bar{y}) / \sum (x_i - \bar{x})^2$ and $b = \bar{y} - w\bar{x}$. Here $\bar{x} = 1$, $\bar{y} = 8/3$, the numerator is $(-1)(-5/3) + 0 + (1)(4/3) = 3$, and the denominator is $2$. So $w = 1.5$, $b \approx 1.167$, and the residuals are $-0.167, +0.333, -0.167$: they sum to zero (orthogonal to the ones column) and are orthogonal to $x - \bar x$.

$X^\top X$ is singular when columns are linearly dependent (a feature in two units, a full one-hot encoding plus an intercept) or when $p > n$, and nearly singular when columns are strongly correlated. Never form the inverse in code: use `np.linalg.lstsq` or `np.linalg.solve`. The cost is about $O(np^2 + p^3)$, fine for thousands of features, painful for millions.

## 2. Gradient descent for least squares

For large $p$, or losses without a closed form, we iterate. On the mean squared error $\frac{1}{n}\lVert X\mathbf{w} - \mathbf{y}\rVert^2$ the step with learning rate $\eta$ is

$$\mathbf{w} \leftarrow \mathbf{w} - \eta \cdot \tfrac{2}{n} X^\top (X\mathbf{w} - \mathbf{y}).$$

The loss is a quadratic bowl with Hessian $H = \frac{2}{n}X^\top X$. Along an eigenvector of $H$ with eigenvalue $\lambda_i$, each step multiplies the error by $(1 - \eta\lambda_i)$, so we need $\eta < 2/\lambda_{\max}$. With $\eta = 1/\lambda_{\max}$ the slowest direction shrinks by $1 - 1/\kappa$ per step, where $\kappa = \lambda_{\max}/\lambda_{\min}$ is the **condition number**, and reducing the error by a factor $\varepsilon$ takes about $\kappa \ln(1/\varepsilon)$ steps (L0-05).

Hence the rule: **standardize features before gradient descent.** In script 1, one feature sits around 3 with spread 1 and the other around 100 with spread 30; $\kappa \approx 238{,}000$, and 200,000 steps do not converge, since $(1 - 1/238{,}000)^{200{,}000} \approx e^{-0.84} \approx 0.43$. After standardizing (subtract the mean, divide by the standard deviation), $\kappa \approx 1.1$ and 13 steps suffice. Both describe the same model: map back with $w_j = w^{\text{std}}_j / s_j$. Scaling changes the optimizer's job, not the least-squares answer. Penalized models are different.

## 3. Regularization: ridge and lasso

With few rows, many features, or correlated features, least-squares weights have high variance. **Regularization** adds a penalty that pulls weights toward zero, trading some bias for less variance: the bias-variance trade-off from L1-01 with a dial attached.

**Ridge regression** [Hoerl and Kennard, 1970] adds a squared L2 penalty:

$$L_{\text{ridge}}(\mathbf{w}) = \lVert \mathbf{y} - X\mathbf{w}\rVert^2 + \lambda \lVert \mathbf{w}\rVert^2, \qquad \hat{\mathbf{w}}_{\text{ridge}} = (X^\top X + \lambda I)^{-1} X^\top \mathbf{y}.$$

Hoerl and Kennard proposed exactly this: add a small positive quantity to the diagonal of $X^\top X$, which makes it invertible for any $\lambda > 0$, even when $p > n$. In the worked example (centered, $\lambda = 1$) the slope becomes $3/(2 + 1) = 1$ instead of $1.5$.

The SVD (L0-02) shows what ridge does. With $X = UDV^\top$ and singular values $d_j$, ridge's fitted values are $\sum_j \mathbf{u}_j \frac{d_j^2}{d_j^2 + \lambda} \mathbf{u}_j^\top \mathbf{y}$. Directions where the data vary a lot (large $d_j$) are barely touched; directions where the data say little, and least squares is noisiest, are shrunk hard [Hastie et al., 2009].

**The lasso** [Tibshirani, 1996] uses an L1 penalty. In scikit-learn's scaling (so the code matches; scikit-learn calls the penalty `alpha`):

$$L_{\text{lasso}}(\mathbf{w}) = \frac{1}{2n}\lVert \mathbf{y} - X\mathbf{w}\rVert^2 + \alpha \lVert \mathbf{w}\rVert_1, \qquad \lVert \mathbf{w}\rVert_1 = \sum_j |w_j|.$$

There is no closed form, but one coordinate at a time there is. Fix all weights except $w_j$, let $\mathbf{r}$ be the residual with feature $j$'s contribution removed, and $\rho_j = \mathbf{x}_j^\top \mathbf{r}$. Then

$$w_j = \frac{S(\rho_j,\; n\alpha)}{\mathbf{x}_j^\top \mathbf{x}_j}, \qquad S(\rho, t) = \operatorname{sign}(\rho)\max(|\rho| - t, 0).$$

$S$ is **soft-thresholding**: if feature $j$'s correlation with what is left to explain is below $n\alpha$, its weight is exactly zero. Cycling over coordinates until nothing changes is **coordinate descent**, which scikit-learn's `Lasso` uses [scikit-learn, Linear Models].

Tibshirani (1996) states the contrast for an orthonormal design: ridge scales each least-squares coefficient by a constant factor, while the lasso shifts each toward zero by a constant and truncates at zero. So the lasso does **feature selection** and ridge does not. Geometrically, the L1 constraint region is a diamond with corners on the axes, and the squared-error contours usually touch it first at a corner, where some coordinates are zero.

Rules that follow:

- **Standardize features** before penalizing; otherwise the penalty depends on units.
- **Do not penalize the intercept.** Center the data, fit, add the mean back (scikit-learn does this).
- **Choose the penalty by cross-validation** on training data (`RidgeCV`, `LassoCV`, `LogisticRegressionCV`).
- **The elastic net** [Zou and Hastie, 2005] mixes L1 and L2. It keeps sparsity while handling strongly correlated predictors better, with a "grouping effect" where correlated predictors tend to enter or leave together.

Regularization is not a free win. With $n$ large relative to $p$, least squares may already be near the best available. Script 2 shows both regimes.

## 4. The probabilistic view

Assume $y_i = \mathbf{w}^\top\mathbf{x}_i + \varepsilon_i$ with independent $\varepsilon_i \sim \mathcal{N}(0, \sigma^2)$. The negative log-likelihood (L0-04) is

$$-\log p(\mathbf{y} \mid X, \mathbf{w}) = \frac{1}{2\sigma^2}\sum_i (y_i - \mathbf{w}^\top\mathbf{x}_i)^2 + \frac{n}{2}\log(2\pi\sigma^2),$$

so **maximum likelihood under Gaussian noise is least squares**. Add a prior $\mathbf{w} \sim \mathcal{N}(0, \tau^2 I)$ and the negative log-posterior gains $\frac{1}{2\tau^2}\lVert\mathbf{w}\rVert^2$: the **MAP estimate is ridge** with $\lambda = \sigma^2/\tau^2$. A Laplace prior $p(w_j) \propto e^{-|w_j|/\beta}$ gives the **lasso** as a MAP estimate. Bishop (2006) and Murphy (2022) develop this view in depth.

The view makes assumptions explicit. Squared error assumes symmetric, light-tailed noise, so a few large outliers can drag the fit. The penalty encodes how large you believe weights are relative to noise. Change the noise model and the loss changes; for 0/1 labels the natural model is Bernoulli.

## 5. Logistic regression

For $y \in \{0, 1\}$ we want a probability. Logistic regression passes the linear score through the **sigmoid** $\sigma(z) = 1/(1 + e^{-z})$:

$$p(y = 1 \mid \mathbf{x}) = \sigma(\mathbf{w}^\top\mathbf{x} + b).$$

Inverting the sigmoid shows the linear part models the **log-odds**: $\log\frac{p}{1-p} = \mathbf{w}^\top\mathbf{x} + b$. A one-unit increase in $x_j$, others fixed, multiplies the odds by $e^{w_j}$ (a weight of $0.7$ roughly doubles them). The boundary $p = 0.5$ is the hyperplane $\mathbf{w}^\top\mathbf{x} + b = 0$: a linear classifier.

With $p_i = \sigma(\mathbf{w}^\top\mathbf{x}_i + b)$ and Bernoulli labels, the negative log-likelihood is the **log-loss** (binary cross-entropy):

$$L(\mathbf{w}) = -\sum_{i=1}^n \big[ y_i \log p_i + (1 - y_i)\log(1 - p_i) \big].$$

Since $\sigma'(z) = \sigma(z)(1 - \sigma(z))$, each term's derivative with respect to its score is $p_i - y_i$, so

$$\nabla_{\mathbf{w}} L = X^\top(\mathbf{p} - \mathbf{y}), \qquad \nabla^2_{\mathbf{w}} L = X^\top \operatorname{diag}\big(p_i(1 - p_i)\big) X.$$

Same "features times error" shape as least squares. The Hessian is positive semidefinite, so the loss is **convex**. There is no closed form; we use gradient descent, Newton's method (iteratively reweighted least squares), or L-BFGS, scikit-learn's default solver [scikit-learn, Linear Models].

**Separable data breaks unpenalized logistic regression.** If a hyperplane separates the classes, scaling $\mathbf{w}$ up always lowers the loss, so the minimum is at infinity. In script 3 the unpenalized weight keeps growing (about 4.5 per tenfold increase in steps), while with a penalty it stops at 1.18. With many features and few rows, separability is common, so you usually want a penalty.

> **Note:** scikit-learn's `LogisticRegression` is regularized by default: it minimizes $C\sum_i \ell_i + \frac{1}{2}\lVert\mathbf{w}\rVert^2$ with $C = 1.0$ (smaller $C$, stronger penalty). As of scikit-learn 1.9.1 (October 2026) the `penalty` argument is deprecated (since 1.8, removal planned for 1.10); the API reference says to choose the penalty type with `l1_ratio` and use `C=np.inf` for no penalty.

Because it minimizes log-loss, a proper scoring rule, logistic regression often gives reasonable probabilities when the model is well specified. That is not guaranteed: strong penalties pull probabilities toward 0.5 and misspecification can distort them. Check calibration (L1-01).

## 6. Softmax regression

For $K$ classes, give each class a score $z_k = \mathbf{w}_k^\top\mathbf{x} + b_k$ and normalize with the **softmax**:

$$p(y = k \mid \mathbf{x}) = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}}.$$

The loss is cross-entropy, $-\sum_i \log p(y_i \mid \mathbf{x}_i)$. With one-hot targets $Y$ and predicted probabilities $P$, the gradient with respect to the scores is again $P - Y$, and with respect to the weight matrix it is $X^\top(P - Y)$. Three details:

- **Stability.** Subtract $\max_k z_k$ before exponentiating (the log-sum-exp trick, L0-01). Softmax is unchanged and nothing overflows.
- **Over-parameterization.** Adding the same vector to every $\mathbf{w}_k$ leaves probabilities unchanged, so the unpenalized solution is not unique; a penalty picks one.
- **$K = 2$ is logistic regression**: $e^{z_1}/(e^{z_0} + e^{z_1}) = \sigma(z_1 - z_0)$.

Softmax plus cross-entropy is the output layer of almost every neural classifier (L2-01): the last layer is softmax regression on learned features.

## 7. Interpreting coefficients, and the limits

Linear models are readable, which makes it easy to read too much into them. The scikit-learn example on pitfalls in interpreting coefficients [scikit-learn, coefficient pitfalls] is an excellent short treatment; script 4 reproduces its main lessons on synthetic data with known truth.

1. **Units.** A distance effect of $-0.004$ per metre is $-3.96$ per kilometre. Raw coefficients compare units, not importance. Standardize, or multiply each coefficient by its feature's standard deviation.
2. **Conditional meaning.** $w_j$ is the change per unit of $x_j$ *holding the other features in the model fixed*. Add or drop a feature and other coefficients can change, even in sign.
3. **Correlated features.** If $x_2$ nearly copies $x_1$, the data pin down $w_1 + w_2$ but not the split. In script 4 the truth is $(3, 0)$; least squares gives $(4.92, -1.94)$, each weight has bootstrap standard deviation about 1.1, and the sum has 0.06. Ridge splits the weight evenly, $(1.59, 1.34)$: stable, but no better at saying which feature matters.
4. **Omitted variables.** If an unmeasured $z$ drives both $x$ and $y$, $x$'s coefficient absorbs part of $z$'s effect. In script 4 the true effect is $-1$; regressing on $x$ alone gives $+1.42$, the wrong sign. More data does not fix a missing confounder.
5. **Association is not causation.** A precise coefficient says how predictions change, not what happens if you intervene.
6. **Regularized coefficients are biased by design.** They are shrunk toward zero, naive standard errors do not apply, and with correlated features the lasso's selected set can change between resamples.
7. **A coefficient describes the model.** If the model fits poorly, its coefficients describe a poor approximation.

What you can do: describe the fitted model with standardized coefficients, check their stability by bootstrap or repeated cross-validation, report uncertainty, and treat causal claims as needing separate evidence.

## Hands-on

Run from the repository root, for example `.venv/bin/python code/L1-02-linear-and-logistic-models/linreg_closed_form_vs_gd.py`. All data are synthetic or bundled with scikit-learn. Outputs are from scikit-learn 1.9.1 and NumPy 2.5.4 on Linux; trailing digits may differ elsewhere.

### Script 1: closed form vs gradient descent

```python
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
```

Output:

```text
closed form       [b, w1, w2] = [0.8048 2.0336 0.051 ]  MSE = 0.248
scikit-learn      [b, w1, w2] = [0.8048 2.0336 0.051 ]
GD on raw features : condition number =   237806.0, iterations = 200000, MSE = 0.2498
GD on standardized : condition number =        1.1, iterations =     13, MSE = 0.2480
GD (standardized, mapped back) [b, w1, w2] = [0.8048 2.0336 0.051 ]
```

Closed form and scikit-learn agree. Gradient descent on raw features hits the 200,000-step cap with a slightly worse MSE (0.2498 vs 0.2480), as the condition number predicted; standardized, it converges in 13 steps to the same coefficients. The intercept estimate (0.80 vs a true 1.0) is off because of noise, not the method.

### Script 2: ridge and lasso from scratch

The diabetes data (bundled) has 442 patients, 10 baseline measurements (age, sex, BMI, blood pressure, six blood serum measurements `s1` to `s6`), and a measure of disease progression one year later.

```python
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
```

Output:

```text
max |ours - sklearn|  ridge: 2.35e-14  lasso: 5.48e-09

feature    OLS   ridge(a=100)  lasso(a=5)
age       -2.5       -0.9        0.0
sex       -9.2       -6.4       -0.9
bmi       28.0       22.2       26.6
bp        13.0       11.5        7.4
s1       -23.9       -1.0        0.0
s2        11.5       -4.3        0.0
s3        -3.4      -10.0       -8.0
s4         4.2        5.4        0.0
s5        35.4       19.7       24.0
s6         1.4        3.9        0.0

alpha   nonzero  train MSE  test MSE
 0.01      10       2804      3099
 0.10      10       2806      3115
 0.50       8       2811      3122
 1.00       8       2818      3118
 2.00       7       2845      3121
 5.00       5       2971      3197
10.00       4       3125      3262
20.00       2       3644      3527
40.00       2       5278      4448

LassoCV alpha = 1.300, nonzero = 7, test MSE = 3116
OLS test MSE = 3097; predicting the mean: 5101

40 rows, 10 real + 30 noise features:
  OLS        test MSE = 25997078
  ridge a=10   test MSE =     4184
  ridge a=100  test MSE =     3638
  lasso a=2    test MSE =     4518  (nonzero real: 7, nonzero noise: 20)
  lasso a=5    test MSE =     4119  (nonzero real: 5, nonzero noise: 15)
  lasso a=15   test MSE =     3895  (nonzero real: 3, nonzero noise: 1)
```

- **Correctness.** Ridge matches scikit-learn to $10^{-14}$, lasso to about $10^{-8}$ (coordinate descent stops at a tolerance).
- **Shrinkage vs selection.** At `alpha=5` the lasso zeroes five features; ridge keeps all. `s1` (total cholesterol) and `s2` (LDL) correlate at about 0.9 here, and least squares gives them large opposite-signed weights ($-23.9$, $+11.5$). Ridge collapses both and `s2` flips sign: point 3 of section 7 on real data.
- **Regularization does not always help.** With 309 training rows and 10 features, no lasso penalty beats OLS on test MSE (3099 at the smallest alpha vs 3097), and cross-validation picks a near-tie. That is a real result.
- **When it does help.** With 40 rows and 30 noise columns, $p = n = 40$ and least squares interpolates the training data: test MSE in the tens of millions. Ridge gives 3638; a strong lasso gives 3895 while keeping one noise column. Weak lasso penalties let many noise columns in.

### Script 3: logistic and softmax regression from scratch

Breast cancer data (bundled): 569 tumours, 30 features, label 1 = benign. Digits (bundled): 1,797 8-by-8 images of handwritten digits.

```python
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
```

Output:

```text
binary, C=1:  max |w_ours - w_sklearn| = 2.76e-04
  test accuracy ours 0.959, sklearn 0.959
  test log-loss ours 0.0847, sklearn 0.0847
  largest |w|: feature 'radius error', w = -1.09, odds multiplier per +1 SD = exp(w) = 0.34

separable toy data, |w| after k gradient steps:
  k=   100   no penalty:   4.95    C=1:  1.18
  k=  1000   no penalty:   8.98    C=1:  1.18
  k= 10000   no penalty:  13.47    C=1:  1.18
  k=100000   no penalty:  18.06    C=1:  1.18

digits (10 classes): test accuracy ours 0.974, sklearn 0.972
  rows of P sum to 1: True; mean max-probability 0.952
```

Gradient descent matches scikit-learn's L-BFGS to about $3 \times 10^{-4}$ in the weights, with identical accuracy and log-loss. On standardized features, a $+1$ standard deviation change in `radius error` multiplies the odds of "benign" by 0.34, holding the other 29 (heavily correlated) features fixed, so read it with section 7 in mind. On separable data the unpenalized weight grows roughly logarithmically in the number of steps. Softmax regression reaches about 97% on digits, matching scikit-learn.

### Script 4: what coefficients do and do not tell you

```python
"""What linear-model coefficients do and do not tell you: units, collinearity,
and omitted variables, on synthetic data where we know the truth.
"""
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge

rng = np.random.default_rng(1)
n = 300

# 1. Units: the same model, distance in metres vs kilometres.
dist_m = rng.uniform(100, 5000, n)
price = 50 - 0.004 * dist_m + rng.normal(0, 2, n)
c_m = LinearRegression().fit(dist_m[:, None], price).coef_[0]
c_km = LinearRegression().fit((dist_m / 1000)[:, None], price).coef_[0]
print(f"1. coefficient per metre {c_m:.5f}, per km {c_km:.3f} (same model, same fit)")

# 2. Collinearity: x2 is almost a copy of x1; y depends only on x1.
x1 = rng.normal(size=n)
x2 = x1 + rng.normal(0, 0.05, n)
y = 3 * x1 + rng.normal(0, 1, n)
X = np.column_stack([x1, x2])
print("2. OLS on the full sample: (w1, w2) =",
      np.round(LinearRegression().fit(X, y).coef_, 2))
ols, rid = [], []
for _ in range(200):                     # refit on bootstrap resamples
    i = rng.integers(0, n, n)
    ols.append(LinearRegression().fit(X[i], y[i]).coef_)
    rid.append(Ridge(alpha=10.0).fit(X[i], y[i]).coef_)
ols, rid = np.array(ols), np.array(rid)
print("   truth (w1, w2) = (3, 0); mean +/- sd over 200 bootstrap refits:")
print(f"   OLS   w1 {ols[:, 0].mean():5.2f} +/- {ols[:, 0].std():.2f}   "
      f"w2 {ols[:, 1].mean():5.2f} +/- {ols[:, 1].std():.2f}   "
      f"w1+w2 {ols.sum(1).mean():.2f} +/- {ols.sum(1).std():.2f}")
print(f"   ridge w1 {rid[:, 0].mean():5.2f} +/- {rid[:, 0].std():.2f}   "
      f"w2 {rid[:, 1].mean():5.2f} +/- {rid[:, 1].std():.2f}   "
      f"w1+w2 {rid.sum(1).mean():.2f} +/- {rid.sum(1).std():.2f}")
print(f"   correlation of OLS w1 and w2 across resamples: {np.corrcoef(ols.T)[0, 1]:.3f}")

# 3. Omitted variable: z drives both x and y. Truly, more x lowers y.
z = rng.normal(size=n)
x = z + rng.normal(0, 0.5, n)
y = -1.0 * x + 3.0 * z + rng.normal(0, 1, n)
c_alone = LinearRegression().fit(x[:, None], y).coef_[0]
c_both = LinearRegression().fit(np.column_stack([x, z]), y).coef_
print(f"3. true effect of x is -1.0; y ~ x alone gives {c_alone:+.2f}; "
      f"y ~ x + z gives {c_both[0]:+.2f} (z: {c_both[1]:+.2f})")
```

Output:

```text
1. coefficient per metre -0.00396, per km -3.960 (same model, same fit)
2. OLS on the full sample: (w1, w2) = [ 4.92 -1.94]
   truth (w1, w2) = (3, 0); mean +/- sd over 200 bootstrap refits:
   OLS   w1  4.89 +/- 1.14   w2 -1.91 +/- 1.13   w1+w2 2.98 +/- 0.06
   ridge w1  1.59 +/- 0.05   w2  1.34 +/- 0.05   w1+w2 2.93 +/- 0.06
   correlation of OLS w1 and w2 across resamples: -0.999
3. true effect of x is -1.0; y ~ x alone gives +1.42; y ~ x + z gives -1.13 (z: +3.19)
```

Section 7 discussed these numbers. Note that the bootstrap mean of OLS $w_1$ (4.89) sits near the full-sample estimate (4.92), not the truth (3): the bootstrap shows how much the estimate wobbles around *this sample's* answer, not where the truth is. The well-determined sum $w_1 + w_2$ is on target at 2.98.

## Common mistakes

- **Fitting scalers or choosing penalties on all data before splitting.** This leaks test information (L1-01). Put scaling in a pipeline fitted on training folds.
- **Forgetting scikit-learn's logistic regression is penalized.** Its coefficients will not match an unpenalized statistics package. Use `C=np.inf` for maximum likelihood, and expect trouble on separable data.
- **Penalizing unscaled features**, so the lasso's "selected" features partly reflect units.
- **Reading raw coefficients as importance**, ignoring units and correlated features.
- **Inverting $X^\top X$ explicitly** instead of `lstsq` or `solve`.
- **Reusing a learning rate.** The stable step depends on $\lambda_{\max}$ of the Hessian, which depends on feature scales. Exploding loss means the step is too big; creeping loss suggests ill-conditioning.
- **Full one-hot encoding plus an intercept, no penalty.** The columns are dependent and $X^\top X$ is singular.
- **Trusting a lasso's selected set or a good fit as causal.** Check stability by bootstrap; prediction quality says nothing about confounding.

## Exercises

1. [warm-up] Fit the three-point example with `np.linalg.lstsq` and confirm $w = 1.5$. Compute the ridge slope for $\lambda = 1$ and $10$ by hand and check in code.
2. [warm-up] In script 1, use learning rates $1.9/\lambda_{\max}$ and $2.1/\lambda_{\max}$ on standardized data. What happens, and why is the boundary at 2?
3. [warm-up] Show that softmax is unchanged when you add a constant to every score, and explain why subtracting the maximum is safe.
4. [core] Derive the lasso coordinate update (soft-thresholding) from section 3.
5. [core] Add an L1 option to `fit_logistic` by proximal gradient descent (gradient step, then soft-threshold). Compare its sparsity with `LogisticRegression(l1_ratio=1, solver="saga")` on breast cancer.
6. [core] On diabetes, fit `Lasso(alpha=5)` on 100 bootstrap samples of standardized data and count how often each feature is nonzero. Repeat with `ElasticNet(alpha=5, l1_ratio=0.5)`.
7. [core] Implement Newton's method for logistic regression using the Hessian in section 5. Count iterations on breast cancer and state each iteration's cost in $n$ and $p$.
8. [core] Prove that a Gaussian prior gives ridge with $\lambda = \sigma^2/\tau^2$. What does $\tau \to \infty$ mean?
9. [stretch] Derive the SVD form of ridge and compute the ridge path for 100 values of $\lambda$ with a single SVD. Check against `Ridge`.
10. [stretch] Explain why the unpenalized weight in script 3 grows like $\log k$ and predict the growth per factor of 10.

### Hints and answers

1. On centered data $w = 3/(2 + \lambda)$: $1.0$ and $0.25$.
2. At $1.9$ the top direction's factor is $-0.9$: it oscillates but converges. At $2.1$ it is $-1.1$ and diverges.
3. $e^{z_k + c} / \sum_j e^{z_j + c} = e^{z_k}/\sum_j e^{z_j}$. After subtracting the max, the largest term is $e^0 = 1$, so nothing overflows and the denominator is at least 1.
4. Minimize $\frac{1}{2n}\lVert\mathbf{r} - \mathbf{x}_j w_j\rVert^2 + \alpha|w_j|$; the subgradient of $|w|$ at 0 is $[-1, 1]$. Solve the cases $w_j > 0$, $< 0$, $= 0$.
5. Update $\mathbf{w} \leftarrow S(\mathbf{w} - \eta\nabla, \eta\lambda)$ elementwise. Expect similar but not identical nonzero sets.
6. When I ran this (seed 0), the lasso kept `bmi`, `bp`, `s5` in all 100 resamples and `s3` in 99, but `s2` in only 20 and `s6` in 44. The elastic net kept most features most of the time, including both `s1` (83) and `s4` (100).
7. With $C = 1$ my implementation converged in 10 iterations, versus thousands of plain gradient steps. Each costs $O(np^2 + p^3)$ versus $O(np)$. Without a penalty, Newton can diverge on near-separable data.
8. Log-posterior: $-\frac{1}{2\sigma^2}\lVert\mathbf{y} - X\mathbf{w}\rVert^2 - \frac{1}{2\tau^2}\lVert\mathbf{w}\rVert^2 + \text{const}$; multiply by $-2\sigma^2$. $\tau \to \infty$ is a flat prior: least squares.
9. $\hat{\mathbf{w}}_\lambda = V \operatorname{diag}\big(d_j/(d_j^2 + \lambda)\big) U^\top\mathbf{y}$.
10. Once all points are correct, the gradient is dominated by the smallest margin, roughly $e^{-0.5 w}$ here (smallest $|x| = 0.5$). So $dw/dk \propto e^{-0.5w}$, giving $w \approx 2\ln k$: about $2\ln 10 \approx 4.6$ per decade, close to the observed 4.0 to 4.6.

## Check your understanding

1. **Why does standardizing speed up gradient descent but not change the least-squares fit?** Rescaling a column does not change the column space, so the fit is the same and the weight rescales inversely. Gradient descent's speed depends on the condition number of $X^\top X$, which scaling changes a lot: badly scaled features make a long thin valley where a stable step is tiny along the shallow direction.

2. **Why does the lasso give exact zeros and ridge does not?** Coordinate-wise, the lasso solution is a soft-threshold: if a feature's correlation with the residual is below the threshold, the minimizer is exactly zero, because the L1 penalty's slope does not vanish at zero. Ridge's penalty has zero slope at zero, so any nonzero correlation gives a nonzero (shrunk) weight.

3. **What makes least squares the "right" loss, and what changes for binary labels?** Least squares is maximum likelihood under independent Gaussian noise with constant variance. For binary labels the natural model is Bernoulli with a sigmoid of a linear score, whose negative log-likelihood is log-loss.

4. **"Feature A has coefficient 0.02 and B has 15, so B matters 750 times more." What do you ask?** What units, and were features standardized? Are A and B correlated with each other or with other features? Was the model regularized? And "matters" for predictions, or causally? The coefficients address at most the first reading.

5. **Why does unpenalized logistic regression fail on separable data, and why is that common?** Scaling up a separating $\mathbf{w}$ pushes every probability toward its label and always lowers the loss, so the optimum is at infinity. Separation is likely when features are numerous relative to examples, or when one feature perfectly predicts the label in a small sample. A penalty or early stopping gives a finite answer.

## Further reading

- **Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani, Jonathan Taylor, "An Introduction to Statistical Learning with Applications in Python"** (2023, Springer). https://www.statlearning.com/
  *Read for:* the clearest gentle treatment of linear and logistic regression and ridge/lasso, with Python labs; free PDF on the official site. *Level:* intro.
- **scikit-learn developers, "1.1. Linear Models"** (scikit-learn 1.9.1 User Guide, accessed October 2026). https://scikit-learn.org/stable/modules/linear_model.html
  *Read for:* the exact objectives and solvers behind `Ridge`, `Lasso`, `ElasticNet`, and `LogisticRegression`. *Level:* intro.
- **scikit-learn developers, "Common pitfalls in the interpretation of coefficients of linear models"** (scikit-learn 1.9.1 examples, accessed October 2026). https://scikit-learn.org/stable/auto_examples/inspection/plot_linear_model_coefficient_interpretation.html
  *Read for:* a real-data tour of units, conditional dependence, correlated features, coefficient variability, and causality. *Level:* intro.
- **Trevor Hastie, Robert Tibshirani, Jerome Friedman, "The Elements of Statistical Learning: Data Mining, Inference, and Prediction"** (2nd ed., 2009, Springer). https://hastie.su.domains/ElemStatLearn/
  *Read for:* the SVD view of ridge, lasso geometry, and a deeper statistical treatment; free PDF on the authors' site. *Level:* intermediate.
- **Christopher M. Bishop, "Pattern Recognition and Machine Learning"** (2006, Springer). https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/
  *Read for:* the probabilistic and Bayesian view of linear and logistic regression; free PDF from Microsoft Research. *Level:* intermediate.
- **Kevin P. Murphy, "Probabilistic Machine Learning: An Introduction"** (2022, MIT Press). https://probml.github.io/pml-book/book1.html
  *Read for:* a modern probabilistic treatment of linear models, regularization as MAP estimation, and optimization; free draft PDF (CC BY-NC-ND). *Level:* intermediate.
- **Arthur E. Hoerl, Robert W. Kennard, "Ridge Regression: Biased Estimation for Nonorthogonal Problems"** (1970, Technometrics 12(1):55-67). https://www.stat.cmu.edu/technometrics/70-79/VOL-12-01/v1201055.pdf
  *Read for:* the original case for adding a constant to the diagonal of $X^\top X$, and the "ridge trace". *Level:* intermediate.
- **Robert Tibshirani, "Regression Shrinkage and Selection via the Lasso"** (1996, Journal of the Royal Statistical Society Series B 58(1):267-288). https://academic.oup.com/jrsssb/article/58/1/267/7027929
  *Read for:* the original lasso paper, including the orthonormal-design comparison with ridge. Publisher page; may require a subscription. *Level:* advanced.
- **Hui Zou, Trevor Hastie, "Regularization and variable selection via the elastic net"** (2005, Journal of the Royal Statistical Society Series B 67(2):301-320). https://ideas.repec.org/a/bla/jorssb/v67y2005i2p301-320.html
  *Read for:* why mixing L1 and L2 helps with correlated predictors and $p \gg n$. Bibliographic record; the journal version may require a subscription. *Level:* advanced.
