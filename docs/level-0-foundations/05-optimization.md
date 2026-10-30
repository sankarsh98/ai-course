# Optimization

> **Level 0 · Module 5** · Prerequisites: L0-01, L0-02, L0-03 · Time: 5 to 7 hours · Code: `code/L0-05-optimization/`

## What you will be able to do

- Predict from a loss's curvature which learning rates converge, oscillate or diverge.
- Explain how the condition number sets the speed of gradient descent, and why momentum helps.
- Implement SGD, momentum, RMSProp, Adam and AdamW from scratch and test claims about them.
- Recognise convex problems, local minima and saddle points, and use a backtracking line search.
- Separate what is known from what is still argued about in why deep networks are trainable.

## 1. Gradient descent and the learning rate

Training means choosing parameters $w \in \mathbb{R}^d$ to make a **loss** $f(w)$ small. Usually the loss is an average over $n$ examples, $f(w) = \frac{1}{n}\sum_{i=1}^{n} \ell_i(w)$. The gradient $\nabla f(w)$ (L0-03) points uphill, so **gradient descent** steps the other way:

$$w_{t+1} = w_t - \eta \, \nabla f(w_t),$$

where $\eta > 0$ is the **learning rate**. The rest of this module is about when this is slow or unstable, and what to do about it.

Start in one dimension with $f(w) = 5w^2$. The gradient is $10w$, so each step multiplies $w$ by $1 - 10\eta$:

| $\eta$ | factor $1 - 10\eta$ | from $w_0 = 1$ |
|---|---|---|
| 0.01 | 0.9 | 1, 0.9, 0.81, ... slow and steady |
| 0.1 | 0 | reaches the minimum in one step |
| 0.15 | -0.5 | 1, -0.5, 0.25, ... overshoots but shrinks |
| 0.25 | -1.5 | 1, -1.5, 2.25, ... diverges |

With second derivative $f'' = 10$, gradient descent converges precisely when $|1 - \eta f''| < 1$, that is $0 < \eta < 2/f''$. The ideal step $1/f''$ is Newton's method. **Curvature sets the scale of the learning rate.**

## 2. Many dimensions: conditioning

In $d$ dimensions curvature is the **Hessian** $H$, the matrix of second derivatives. Near a minimum $w^*$ any smooth loss looks like a quadratic bowl, $f(w) \approx f(w^*) + \tfrac12 (w - w^*)^\top H (w - w^*)$. So the quadratic $f(w) = \tfrac12 w^\top A w$, with $A$ symmetric positive definite, describes the end of training. Its gradient is $Aw$ and the update is $w \leftarrow (I - \eta A) w$.

Write $A = Q\Lambda Q^\top$ (L0-02). In the rotated coordinates $z = Q^\top w$ the update splits into independent one-dimensional problems, $z_i \leftarrow (1 - \eta\lambda_i) z_i$, each a copy of the $5w^2$ example with curvature $\lambda_i$. Let $L = \lambda_{\max}$ and $\mu = \lambda_{\min}$.

1. **The sharpest direction sets stability.** Every direction converges only if $\eta < 2/L$.
2. **The flattest direction sets speed.** It shrinks by $1 - \eta\mu$ per step, which is close to 1 when $\mu \ll L$.

The ratio $\kappa = L/\mu$ is the **condition number**. The best fixed step, $\eta = 2/(L + \mu)$, gives an error reduction of $\rho_{\text{GD}} = (\kappa - 1)/(\kappa + 1)$ per step [Goh, 2017]. For large $\kappa$, $\ln\rho \approx -2/\kappa$. Cutting the error by a factor $\epsilon$ therefore takes about $(\kappa/2)\ln(1/\epsilon)$ steps, **linear in the condition number**.

Picture a long, narrow valley. The gradient points mostly at the steep walls, so gradient descent zigzags across and creeps along the floor. Features on different scales cause exactly this in linear models, which is one reason to standardize (L1-05).

## 3. Momentum

**Momentum** (the "heavy ball" method) keeps a velocity:

$$v_{t+1} = \beta v_t - \eta \nabla f(w_t), \qquad w_{t+1} = w_t + v_{t+1}, \qquad \beta \in [0, 1).$$

Components that keep their sign (along the valley floor) accumulate, up to $1/(1-\beta) = 10$ times a single step for $\beta = 0.9$, while components that flip sign cancel. With

$$\eta = \frac{4}{(\sqrt{L} + \sqrt{\mu})^2}, \qquad \beta = \left(\frac{\sqrt\kappa - 1}{\sqrt\kappa + 1}\right)^2,$$

the rate on a quadratic becomes $\rho_{\text{HB}} = (\sqrt\kappa - 1)/(\sqrt\kappa + 1)$: $\kappa$ is replaced by $\sqrt\kappa$ [Goh, 2017]. Goh also shows that a broad class of gradient-only methods cannot beat the $\sqrt\kappa$ dependence on a badly conditioned quadratic.

One subtlety shows up in the output. At these settings, the two-by-two update for each extreme eigen-direction has a repeated eigenvalue, so the error decays like $t\rho^t$, not $\rho^t$. Step counts exceed the simple formula but still scale with $\sqrt\kappa$. **Nesterov momentum** is a close variant that evaluates the gradient at the look-ahead point $w_t + \beta v_t$ [Ruder, 2016].

## 4. Stochastic gradients and minibatches

An exact gradient costs a full pass over the data. **Stochastic gradient descent (SGD)** uses a random minibatch $\mathcal{B}$ of size $B$ instead, $g_{\mathcal{B}} = \frac1B \sum_{i\in\mathcal{B}} \nabla\ell_i(w)$. It is **unbiased** (its average is the true gradient), and its variance falls like $1/B$, as for any sample mean (L0-04). Sampling without replacement multiplies the variance by $(n-B)/(n-1)$, which only matters when $B$ is a large fraction of $n$. Three facts follow [Goodfellow et al., 2016, ch. 8]:

1. **Diminishing returns from bigger batches.** The standard error falls only like $1/\sqrt{B}$, so 100 times the computation buys 10 times less noise. Per example processed, small batches make far more updates.
2. **A noise floor.** With a constant $\eta$, SGD does not converge. Near $w^*$ the true gradient vanishes but the noise does not, so the iterates wander in a region that grows with $\eta$.
3. **Decaying learning rates.** A sufficient condition for convergence is $\sum_t \eta_t = \infty$ and $\sum_t \eta_t^2 < \infty$, met for example by $\eta_t \propto 1/t$.

Practical schedules (warm-up, cosine decay) belong to L2-02.

## 5. Adaptive methods: RMSProp, Adam, AdamW

Another remedy for poor conditioning is a separate step size per parameter: divide each coordinate's step by a running estimate of its gradient magnitude. **RMSProp**, introduced by Geoff Hinton in a lecture rather than a paper [Ruder, 2016], keeps $r_t = \rho r_{t-1} + (1-\rho)g_t^2$ (elementwise) and steps by $\eta g_t/(\sqrt{r_t} + \epsilon)$.

**Adam** [Kingma & Ba, 2015] adds momentum:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \quad v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2, \quad \hat m_t = \frac{m_t}{1-\beta_1^t}, \quad \hat v_t = \frac{v_t}{1-\beta_2^t},$$

$$w_t = w_{t-1} - \eta\, \hat m_t / (\sqrt{\hat v_t} + \epsilon),$$

with suggested defaults $\eta = 0.001$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$ [Kingma & Ba, 2015].

- **Bias correction.** Because $m_0 = v_0 = 0$, at $t = 1$ we get $m_1 = 0.1 g_1$ and $v_1 = 0.001 g_1^2$. Uncorrected, the first step would be $\eta \cdot 0.1/\sqrt{0.001} \approx 3.16\eta$. The corrected estimates are exactly $g_1$ and $g_1^2$, so the first step is $\eta$.
- **Roughly scale-free steps.** With steady gradients, $\hat m/\sqrt{\hat v} \approx \pm1$, so each coordinate moves about $\eta$ whatever its gradient scale. That makes $\eta$ easier to choose. It is **diagonal preconditioning**: each axis is rescaled.
- **Diagonal means axis-aligned.** Rescaling axes cures ill-conditioning only when the steep and flat directions line up with the axes. A diagonal valley stays diagonal. Script 3 shows this.

**AdamW.** For plain SGD, an L2 penalty $\tfrac\lambda2\|w\|^2$ in the loss is equivalent to shrinking weights by a constant factor each step ("weight decay"). Loshchilov and Hutter [2019] showed the equivalence fails for Adam. The penalty gradient $\lambda w$ is divided by $\sqrt{\hat v}$, so weights with large gradients are barely regularized. **AdamW** applies the Adam step to the loss gradient only and subtracts $\eta\lambda w$ separately. It is widely used for large networks (L2-02).

**A caution.** Wilson et al. [2017] built a problem where adaptive methods generalize far worse than SGD, and reported deep learning experiments where tuned SGD generalized better than Adam despite Adam's lower training loss. Adam-family methods remain heavily used, so read the paper as a warning about tuning and fair comparisons, not a verdict. Ruder [2016] summarises later variants such as AMSGrad, proposed to fix problems in Adam's convergence analysis.

## 6. Convex and non-convex problems

A function is **convex** if the chord between any two points on its graph lies on or above the graph. For twice-differentiable $f$, that means the Hessian is positive semidefinite everywhere. Then **every local minimum is global**, and gradient descent with a small enough step reaches it [Boyd & Vandenberghe, 2004]. Linear and logistic regression are convex (L1-02).

Neural network losses are **non-convex**, which allows two new problems.

- **Local minima.** A point can be lower than its neighbours yet not lowest overall, so the starting point decides the outcome. Script 4 runs 20 starts on $0.1w^2 + \sin 3w$ and finds six different endpoints.
- **Saddle points** have zero gradient but a Hessian with both positive and negative eigenvalues. A component $\delta$ along a negative-curvature direction $\lambda_-$ grows by $1 + \eta|\lambda_-|$ per step, so escape takes about $\ln(1/\delta)/(\eta|\lambda_-|)$ steps. Near a saddle, training can stall on a plateau for a long time.

Dauphin et al. [2014] argued that in high dimensions saddles, not bad local minima, are the main obstacle, because a random critical point is unlikely to have all $d$ eigenvalues positive. The argument rests on random-matrix theory plus experiments on networks of that era, so treat it as a strong heuristic, not a theorem about today's models.

## 7. Line search

A **line search** chooses the step size anew at each iteration. **Backtracking** [Boyd & Vandenberghe, 2004] starts from $t = t_0$ and halves $t$ until the **Armijo** (sufficient decrease) condition holds:

$$f(w - t\nabla f(w)) \le f(w) - c\,t\,\|\nabla f(w)\|^2, \qquad c \approx 10^{-4}.$$

It needs no tuning and cannot diverge, but it changes only the step size, not the direction. On a narrow valley, steepest descent zigzags even with ideal steps [Goodfellow et al., 2016, ch. 8], so line search does not cure bad conditioning. Fixing the direction is the job of Newton and quasi-Newton methods such as L-BFGS, which Goodfellow et al. also cover. L-BFGS is the default solver of scikit-learn's `LogisticRegression` (L1-02). Deep learning rarely uses line searches. Each trial costs a forward pass, and on minibatches the comparison of losses is itself noisy.

## 8. Why does deep learning optimization work at all?

Deep network losses are non-convex and badly conditioned, yet SGD and Adam reliably push training loss near zero. There is no complete explanation, only evidence and disagreement.

- **Fitting is easy, even too easy.** Zhang et al. [2017] trained standard image networks with SGD to zero training error on random labels, and even on random-noise images. For large networks, reaching low loss is not the hard part. The puzzle shifts to why the solutions found on real labels generalize (L1-01).
- **Saddles rather than bad minima.** This is the Dauphin et al. [2014] picture above: consistent with experiments, but not proven for realistic architectures.
- **Flat versus sharp minima.** Keskar et al. [2017] found that large-batch training reaches sharper minima that generalize worse, and attributed small-batch SGD's flatter minima to gradient noise. Dinh et al. [2017] showed that ReLU networks can be reparametrized so the same function sits at an arbitrarily sharp minimum, so sharpness alone cannot be the explanation. How to define flatness so that it predicts generalization is unsettled.
- **Training breaks the classical stability rule.** Section 2 says $\eta < 2/L$. Cohen et al. [2021] tracked the top Hessian eigenvalue during full-batch gradient descent on neural networks. It typically rises to about $2/\eta$ and hovers just above it, while the loss keeps falling over the long run with short-term oscillation. Networks routinely train where the quadratic theory predicts instability.

In short, the tools of this module explain local behaviour well and are what you use to debug training. Why large non-convex models end up in good places is only partly understood.

## Hands-on

The scripts need only NumPy and run in seconds from `code/L0-05-optimization/`. Outputs are from a Linux laptop. Trailing digits may differ elsewhere, but the patterns will not.

### Script 1: learning rates and conditioning

```python
"""Gradient descent and momentum on a quadratic bowl.

f(w) = 0.5 * w^T A w, with A diagonal so each coordinate is one eigen-direction.
Part 1 shows the learning-rate regimes; part 2 shows how conditioning sets speed.
"""
import numpy as np


def run(A, w0, lr, steps, beta=0.0):
    """Heavy-ball momentum (beta=0 gives plain gradient descent). Returns losses."""
    w, v = w0.copy(), np.zeros_like(w0)
    losses = []
    for _ in range(steps):
        v = beta * v - lr * (A @ w)   # velocity accumulates past gradients
        w = w + v
        losses.append(0.5 * w @ A @ w)
    return np.array(losses)


def steps_to_tol(A, w0, lr, beta=0.0, tol=1e-10, max_steps=200_000):
    """Number of steps until f(w) < tol * f(w0)."""
    w, v = w0.copy(), np.zeros_like(w0)
    f0 = 0.5 * w0 @ A @ w0
    for t in range(1, max_steps + 1):
        v = beta * v - lr * (A @ w)
        w = w + v
        if 0.5 * w @ A @ w < tol * f0:
            return t
    return None


w0 = np.array([1.0, 1.0])

print("Part 1: learning-rate regimes, eigenvalues (1, 10), so L = 10, 2/L = 0.2")
A = np.diag([1.0, 10.0])
print(f"{'lr':>6} {'|1-lr*1|':>9} {'|1-lr*10|':>10} {'loss@5':>10} {'loss@50':>10}")
for lr in [0.01, 0.1, 0.18, 0.2, 0.21]:
    losses = run(A, w0, lr, 50)
    print(f"{lr:6.2f} {abs(1 - lr):9.2f} {abs(1 - 10 * lr):10.2f} "
          f"{losses[4]:10.3e} {losses[49]:10.3e}")

print("\nPart 2: steps to reduce loss by 1e10, eigenvalues (1, kappa)")
print(f"{'kappa':>6} {'GD steps':>9} {'GD theory':>10} {'momentum':>9} {'mom theory':>11}")
for kappa in [1, 10, 100, 1000, 10000]:
    A = np.diag([1.0, float(kappa)])
    mu, L = 1.0, float(kappa)
    # Best fixed step for GD, and the optimal heavy-ball parameters.
    lr_gd = 2.0 / (L + mu)
    rate_gd = (kappa - 1) / (kappa + 1)
    lr_hb = 4.0 / (np.sqrt(L) + np.sqrt(mu)) ** 2
    rate_hb = (np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)
    beta = rate_hb ** 2
    n_gd = steps_to_tol(A, w0, lr_gd)
    n_hb = steps_to_tol(A, w0, lr_hb, beta)
    # Loss is quadratic in the error, so it shrinks like rate^(2t).
    th_gd = np.log(1e-10) / (2 * np.log(rate_gd)) if kappa > 1 else 1
    th_hb = np.log(1e-10) / (2 * np.log(rate_hb)) if kappa > 1 else 1
    print(f"{kappa:6d} {n_gd:9d} {th_gd:10.0f} {n_hb:9d} {th_hb:11.0f}")
```

```text
Part 1: learning-rate regimes, eigenvalues (1, 10), so L = 10, 2/L = 0.2
    lr  |1-lr*1|  |1-lr*10|     loss@5    loss@50
  0.01      0.99       0.90  2.196e+00  1.831e-01
  0.10      0.90       0.00  1.743e-01  1.328e-05
  0.18      0.82       0.80  6.056e-01  2.222e-09
  0.20      0.80       1.00  5.054e+00  5.000e+00
  0.21      0.79       1.10  1.302e+01  6.890e+04

Part 2: steps to reduce loss by 1e10, eigenvalues (1, kappa)
 kappa  GD steps  GD theory  momentum  mom theory
     1         1          1         1           1
    10        58         57        23          18
   100       576        576        83          57
  1000      5757       5756       282         182
 10000     57565      57565       953         576
```

Part 1 matches section 2. The larger of the two factors controls the loss. At $\eta = 0.18$ both factors are about 0.8, the best balance. At exactly $2/L = 0.2$ the steep coordinate flips sign forever, so its loss term stays at $\tfrac12 \cdot 10 \cdot 1^2 = 5$. At 0.21 it diverges. In part 2, gradient descent matches $(\kappa/2)\ln(1/\epsilon)$ almost exactly and grows tenfold per row. Momentum grows only about $\sqrt{10} \approx 3.2$ times per row. It sits above its formula for the repeated-eigenvalue reason in section 3.

### Script 2: minibatch noise and the noise floor

```python
"""Minibatch SGD on least-squares regression: gradient noise and the noise floor."""
import numpy as np

rng = np.random.default_rng(0)
n, d = 20_000, 20
X = rng.normal(size=(n, d))
w_true = rng.normal(size=d)
y = X @ w_true + 0.5 * rng.normal(size=n)          # noise std 0.5

def loss(w):
    r = X @ w - y
    return 0.5 * np.mean(r ** 2)

def grad(w, idx):
    r = X[idx] @ w - y[idx]
    return X[idx].T @ r / len(idx)

w_star = np.linalg.lstsq(X, y, rcond=None)[0]
f_star = loss(w_star)
print(f"optimal loss f* = {f_star:.5f}  (noise variance/2 = {0.25 / 2:.5f})")

print("\nPart 1: minibatch gradient error at w = 0 (200 draws per batch size)")
w = np.zeros(d)
g_full = grad(w, np.arange(n))
print(f"{'B':>6} {'E|g_B - g|^2':>14} {'x B':>8}")
for B in [1, 10, 100, 1000]:
    errs = [np.sum((grad(w, rng.choice(n, B, replace=False)) - g_full) ** 2)
            for _ in range(200)]
    print(f"{B:6d} {np.mean(errs):14.4f} {np.mean(errs) * B:8.2f}")

def sgd(B, lr, epochs, decay=False):
    """Run minibatch SGD for a fixed number of passes over the data."""
    w = np.zeros(d)
    t = 0
    for _ in range(epochs):
        perm = rng.permutation(n)
        for start in range(0, n, B):
            idx = perm[start:start + B]
            step = lr / (1 + t / 2000) if decay else lr   # 1/t-style decay
            w -= step * grad(w, idx)
            t += 1
    return loss(w) - f_star, t

print("\nPart 2: 2 epochs (40,000 examples seen), constant lr = 0.05")
print(f"{'B':>6} {'updates':>8} {'f - f*':>12}")
for B in [1, 10, 100, 1000, n]:
    gap, t = sgd(B, 0.05, 2)
    print(f"{B:6d} {t:8d} {gap:12.3e}")

print("\nPart 3: B = 1, 2 epochs, constant vs decaying lr")
for lr in [0.05, 0.01, 0.002]:
    gap_c, _ = sgd(1, lr, 2)
    gap_d, _ = sgd(1, lr, 2, decay=True)
    print(f"lr0 = {lr:5.3f}: constant f - f* = {gap_c:.2e}, decaying f - f* = {gap_d:.2e}")
```

```text
optimal loss f* = 0.12337  (noise variance/2 = 0.12500)

Part 1: minibatch gradient error at w = 0 (200 draws per batch size)
     B   E|g_B - g|^2      x B
     1       395.9374   395.94
    10        39.3804   393.80
   100         3.9847   398.47
  1000         0.3916   391.62

Part 2: 2 epochs (40,000 examples seen), constant lr = 0.05
     B  updates       f - f*
     1    40000    9.414e-02
    10     4000    4.099e-03
   100      400    3.489e-04
  1000       40    1.515e-01
 20000        2    7.851e+00

Part 3: B = 1, 2 epochs, constant vs decaying lr
lr0 = 0.050: constant f - f* = 1.02e-01, decaying f - f* = 2.24e-03
lr0 = 0.010: constant f - f* = 1.63e-02, decaying f - f* = 4.38e-04
lr0 = 0.002: constant f - f* = 2.44e-03, decaying f - f* = 1.76e-05
```

In part 1, the error times $B$ is constant (the $1/B$ law), dipping at $B = 1000$ by the without-replacement factor 0.95. In part 2, with the work fixed, $B = 100$ wins by combining low noise with 400 updates. The large batches made only 40 and 2 updates at a learning rate suited to small batches (exercise 5 makes this fair). Part 3 shows the noise floor. With constant $\eta$ the final gap is roughly proportional to $\eta$, and decay gets 40 to 140 times closer for the same work.

### Script 3: optimizers from scratch

```python
"""SGD, momentum, RMSProp, Adam and AdamW written from scratch, then compared.

Each optimizer is a function step(w, g, state, lr) -> new w that keeps its own state.
"""
import numpy as np


def sgd(w, g, s, lr):
    return w - lr * g

def momentum(w, g, s, lr, beta=0.9):
    s["v"] = beta * s.get("v", 0.0) + g          # running sum of past gradients
    return w - lr * s["v"]

def rmsprop(w, g, s, lr, rho=0.99, eps=1e-8):
    s["r"] = rho * s.get("r", 0.0) + (1 - rho) * g * g
    return w - lr * g / (np.sqrt(s["r"]) + eps)  # per-coordinate scaling

def adam(w, g, s, lr, b1=0.9, b2=0.999, eps=1e-8, wd=0.0, bias_correct=True):
    t = s["t"] = s.get("t", 0) + 1
    s["m"] = b1 * s.get("m", 0.0) + (1 - b1) * g
    s["v"] = b2 * s.get("v", 0.0) + (1 - b2) * g * g
    m_hat = s["m"] / (1 - b1 ** t) if bias_correct else s["m"]
    v_hat = s["v"] / (1 - b2 ** t) if bias_correct else s["v"]
    # AdamW: weight decay applied directly to w, outside the adaptive scaling.
    return w - lr * m_hat / (np.sqrt(v_hat) + eps) - lr * wd * w

OPTS = {"sgd": sgd, "momentum": momentum, "rmsprop": rmsprop, "adam": adam}


def steps_to_tol(opt, f, grad, w0, lr, tol=1e-8, max_steps=5000):
    """Steps until f(w) < tol * f(w0); None if it diverges or runs out of steps."""
    w, s, f0 = w0.copy(), {}, f(w0)
    for t in range(1, max_steps + 1):
        w = opt(w, grad(w), s, lr)
        if not np.all(np.isfinite(w)) or np.abs(w).max() > 1e6:
            return None
        if f(w) < tol * f0:
            return t
    return None


def best_lr(opt, f, grad, w0, lrs=np.logspace(-5, 0, 11)):
    """Tune the learning rate on a log grid so each method gets a fair chance."""
    runs = [(steps_to_tol(opt, f, grad, w0, lr), lr) for lr in lrs]
    runs = [r for r in runs if r[0] is not None]
    return min(runs) if runs else (None, None)


def quadratic(R, kappa=1e4):
    """f(w) = 0.5 w^T A w with eigenvalues 1 and kappa, eigenvectors = columns of R."""
    A = R @ np.diag([1.0, kappa]) @ R.T
    return (lambda w: 0.5 * w @ A @ w), (lambda w: A @ w)

c = np.cos(np.pi / 4)
rotated = np.array([[c, -c], [c, c]])
w_start = np.array([1.0, 0.3])
problems = {
    "quadratic, axis-aligned": (*quadratic(np.eye(2)), w_start),
    "quadratic, rotated 45deg": (*quadratic(rotated), w_start),
    "rosenbrock": (
        lambda w: (1 - w[0]) ** 2 + 100 * (w[1] - w[0] ** 2) ** 2,
        lambda w: np.array([-2 * (1 - w[0]) - 400 * w[0] * (w[1] - w[0] ** 2),
                            200 * (w[1] - w[0] ** 2)]),
        np.array([-1.5, 2.0]),
    ),
}

print("Steps to reduce f by 1e8 (max 5000), best lr from an 11-point log grid")
print(f"{'problem':>25} " + " ".join(f"{k:>17}" for k in OPTS))
for name, (f, g, w0) in problems.items():
    cells = []
    for opt in OPTS.values():
        n, lr = best_lr(opt, f, g, w0)
        cells.append(">5000 (all lrs)" if n is None else f"{n} (lr {lr:.0e})")
    print(f"{name:>25} " + " ".join(f"{cell:>17}" for cell in cells))

print("\nAdam bias correction: size of the first 3 steps, lr = 0.001, g = 0.01 each step")
for bc in [True, False]:
    w, s, sizes = np.zeros(1), {}, []
    for _ in range(3):
        w_new = adam(w, np.array([0.01]), s, 0.001, bias_correct=bc)
        sizes.append(abs(w_new - w)[0])
        w = w_new
    print(f"bias_correct={bc!s:5}: " + ", ".join(f"{x:.2e}" for x in sizes))

print("\nL2-in-the-gradient Adam vs AdamW, 3000 steps, lr 1e-3, strength 0.1")
print("Loss gradient is pure noise (no signal), so only the decay moves w on average.")
rng = np.random.default_rng(0)
noise_scale = np.repeat([10.0, 0.01], 500)      # 500 noisy and 500 quiet weights
for label, l2, wd in [("Adam + L2", 0.1, 0.0), ("AdamW", 0.0, 0.1)]:
    w, s = np.ones(1000), {}
    for _ in range(3000):
        g = noise_scale * rng.normal(size=1000) + l2 * w
        w = adam(w, g, s, 1e-3, wd=wd)
    print(f"{label:>10}: mean w, noisy weights = {w[:500].mean():.3f}, "
          f"quiet weights = {w[500:].mean():.3f}")
```

```text
Steps to reduce f by 1e8 (max 5000), best lr from an 11-point log grid
                  problem               sgd          momentum           rmsprop              adam
  quadratic, axis-aligned   >5000 (all lrs)   1790 (lr 3e-04)      5 (lr 1e+00)     96 (lr 1e-01)
 quadratic, rotated 45deg   >5000 (all lrs)   1611 (lr 3e-04)   >5000 (all lrs)   1952 (lr 1e+00)
               rosenbrock   >5000 (all lrs)   1734 (lr 1e-03)   >5000 (all lrs)    777 (lr 1e+00)

Adam bias correction: size of the first 3 steps, lr = 0.001, g = 0.01 each step
bias_correct=True : 1.00e-03, 1.00e-03, 1.00e-03
bias_correct=False: 3.16e-03, 4.25e-03, 4.95e-03

L2-in-the-gradient Adam vs AdamW, 3000 steps, lr 1e-3, strength 0.1
Loss gradient is pure noise (no signal), so only the decay moves w on average.
 Adam + L2: mean w, noisy weights = 0.969, quiet weights = 0.000
     AdamW: mean w, noisy weights = 0.743, quiet weights = 0.741
```

With $\kappa = 10^4$, plain gradient descent fails everywhere, as script 1 predicts (about 57,000 steps needed). Momentum is unaffected by rotation. RMSProp and Adam are dramatically faster on the axis-aligned bowl (5 and 96 steps). Rotate it by 45 degrees and RMSProp fails while Adam needs 20 times more steps. They are diagonal preconditioners, not magic. On Rosenbrock, Adam won, but its best learning rate sat at the grid's edge, so the ranking is tuning-sensitive. The bias-correction lines reproduce section 5's $3.16\eta$.

In the AdamW test, the loss gradient is pure noise, so only regularization pulls on $w$. With L2 inside the gradient, the noisy weights (large $\sqrt{\hat v}$) barely shrink and the quiet ones collapse to zero. Decoupled decay shrinks every weight by the same $(1 - 10^{-4})^{3000} \approx e^{-0.3} \approx 0.74$.

### Script 4: saddles, local minima, line search

```python
"""Saddle points, multiple minima, and backtracking line search."""
import numpy as np

rng = np.random.default_rng(0)

# Part 1: a saddle. f(x, y) = x^2/2 + y^4/4 - y^2/2.
# Origin: gradient zero, Hessian eigenvalues +1 (x) and -1 (y), so it is a saddle.
# Minima at (0, +1) and (0, -1).
def saddle_grad(p):
    x, y = p
    return np.array([x, y ** 3 - y])

def escape_steps(y0, lr=0.1, noise=0.0, max_steps=10_000):
    p = np.array([1.0, y0])
    for t in range(1, max_steps + 1):
        p = p - lr * (saddle_grad(p) + noise * rng.normal(size=2))
        if abs(p[1]) > 0.5:          # left the saddle region along y
            return t
    return None

print("Part 1: GD near a saddle, start (1, y0), lr 0.1")
for y0 in [1e-1, 1e-3, 1e-6, 1e-9, 0.0]:
    n = escape_steps(y0)
    print(f"  y0 = {y0:7.0e}: steps to |y| > 0.5 = {n if n else 'never (10,000 steps)'}")
n = escape_steps(0.0, noise=0.01)
print(f"  y0 = 0 with gradient noise std 0.01: steps = {n}")

# Part 2: a 1-D non-convex function with several local minima.
f1 = lambda w: 0.1 * w ** 2 + np.sin(3 * w)
g1 = lambda w: 0.2 * w + 3 * np.cos(3 * w)
grid = np.linspace(-10, 10, 200_001)
print(f"\nPart 2: f(w) = 0.1 w^2 + sin(3w); global min f = {f1(grid).min():.4f} "
      f"at w = {grid[f1(grid).argmin()]:.3f}")
ends = []
for w in rng.uniform(-8, 8, size=20):
    for _ in range(2000):
        w -= 0.05 * g1(w)
    ends.append(round(float(w), 2))
values, counts = np.unique(ends, return_counts=True)
for v, c in zip(values, counts):
    print(f"  ended at w = {v:6.2f} (f = {f1(v):7.4f}): {c} of 20 starts")

# Part 3: backtracking (Armijo) line search vs a fixed step on Rosenbrock.
f = lambda w: (1 - w[0]) ** 2 + 100 * (w[1] - w[0] ** 2) ** 2
g = lambda w: np.array([-2 * (1 - w[0]) - 400 * w[0] * (w[1] - w[0] ** 2),
                        200 * (w[1] - w[0] ** 2)])

def gd_fixed(lr, iters):
    w = np.array([-1.5, 2.0])
    for i in range(iters):
        w = w - lr * g(w)
        if np.abs(w).max() > 1e3:          # blew up: stop and report it
            return np.inf, i + 1
    return f(w), iters                     # one gradient evaluation per step

def gd_backtracking(iters, t0=1.0, c=1e-4, shrink=0.5):
    w, evals = np.array([-1.5, 2.0]), 0
    for _ in range(iters):
        grad, fw, t = g(w), f(w), t0
        # Shrink t until the step gives "sufficient decrease" (Armijo condition).
        while f(w - t * grad) > fw - c * t * grad @ grad:
            t *= shrink
            evals += 1
        w = w - t * grad
        evals += 1
    return f(w), evals

print("\nPart 3: Rosenbrock from (-1.5, 2), 5000 iterations of steepest descent")
for lr in [1e-4, 1e-3, 2e-3, 5e-3]:
    val, ev = gd_fixed(lr, 5000)
    if np.isinf(val):
        print(f"  fixed lr {lr:.0e}: diverged after {ev} steps")
    else:
        print(f"  fixed lr {lr:.0e}: f = {val:.3e}, function evals = {ev}")
val, ev = gd_backtracking(5000)
print(f"  backtracking:   f = {val:.3e}, function evals = {ev}")
```

```text
Part 1: GD near a saddle, start (1, y0), lr 0.1
  y0 =   1e-01: steps to |y| > 0.5 = 19
  y0 =   1e-03: steps to |y| > 0.5 = 67
  y0 =   1e-06: steps to |y| > 0.5 = 139
  y0 =   1e-09: steps to |y| > 0.5 = 212
  y0 =   0e+00: steps to |y| > 0.5 = never (10,000 steps)
  y0 = 0 with gradient noise std 0.01: steps = 53

Part 2: f(w) = 0.1 w^2 + sin(3w); global min f = -0.9732 at w = -0.512
  ended at w =  -6.65 (f =  3.5308): 4 of 20 starts
  ended at w =  -4.61 (f =  1.1720): 5 of 20 starts
  ended at w =  -0.51 (f = -0.9732): 3 of 20 starts
  ended at w =   1.54 (f = -0.7586): 1 of 20 starts
  ended at w =   3.58 (f =  0.3141): 3 of 20 starts
  ended at w =   5.63 (f =  2.2443): 4 of 20 starts

Part 3: Rosenbrock from (-1.5, 2), 5000 iterations of steepest descent
  fixed lr 1e-04: f = 4.485e+00, function evals = 5000
  fixed lr 1e-03: f = 6.514e-03, function evals = 5000
  fixed lr 2e-03: f = 3.707e-07, function evals = 5000
  fixed lr 5e-03: diverged after 4 steps
  backtracking:   f = 7.211e-03, function evals = 52718
```

Part 1: $|y|$ grows by 1.1 per step, so from $10^{-9}$ the prediction is $\ln(0.5/10^{-9})/\ln 1.1 \approx 210$ steps. The script took 212. Each factor of 1,000 closer adds about 70 steps. Exactly on the saddle's stable line, gradient descent never leaves, but small gradient noise frees it in 53 steps. Part 2: only 3 of 20 starts found the global minimum. In part 3, backtracking needed no tuning and never diverged. But it used ten times more evaluations and ended worse than the best fixed step, which sits close to one that diverges in 4 steps. Line search buys robustness, not a better direction.

## Common mistakes

- **Loss becomes `nan` or explodes.** The learning rate is too high for the current curvature. Cut it by 3 to 10 times before changing the model.
- **Loss falls very slowly.** Sweep the learning rate on a log grid before judging the model.
- **Unscaled features.** Very different input scales create ill-conditioning, so standardize them.
- **Comparing optimizers at one learning rate.** Without a per-method sweep you mostly measure luck with defaults.
- **Expecting constant-$\eta$ SGD to converge exactly.** It plateaus at a noise floor, and a decaying schedule is the usual fix.
- **L2 when you meant weight decay.** PyTorch's documentation for `torch.optim.Adam` (version 2.15, checked October 2026) adds `weight_decay` to the gradient as L2 unless `decoupled_weight_decay=True`. `torch.optim.AdamW` decouples it. Check your library and version.

## Exercises

1. [warm-up] For $f(w) = 3w^2$, find by hand the one-step learning rate and the divergence threshold. Confirm them in NumPy.
2. [warm-up] In `gd_quadratic.py`, use eigenvalues $(1, 50)$. Predict the stability threshold before running it.
3. [core] For $f = \tfrac12 w^\top A w$, show that $\eta = 2/(L+\mu)$ minimises $\max_i |1 - \eta\lambda_i|$ and gives $\rho = (\kappa-1)/(\kappa+1)$.
4. [core] Add Nesterov momentum to `optimizers.py`. Does it change the axis-aligned versus rotated picture?
5. [core] In `sgd_minibatch.py`, tune $\eta$ separately for each batch size. Does $B = 1000$ still lose at a fixed budget of examples?
6. [core] With eigenvalues $(1, 1000)$, run plain gradient descent at $\eta = 1/L$ from $w_0 = (0, 1)$. How many steps does it take, and why is that misleading?
7. [core] Find a problem and learning rate where Adam oscillates without converging while momentum converges. Explain it with the scale-free step.
8. [core] In the saddle experiment, start at $(1, 0)$ with noise for 20 seeds. Report which minimum each run reaches, and check the Hessian there.
9. [stretch] Add a Newton direction $-H^{-1}\nabla f$ to the backtracking search on Rosenbrock and compare with script 4.
10. [stretch] After L2-01, estimate the top Hessian eigenvalue of a small MLP during full-batch training (power iteration on finite-difference Hessian-vector products) and compare it with $2/\eta$, as in Cohen et al. [2021].

### Hints and answers

1. $f'' = 6$, so the one-step rate is $\eta = 1/6$. It diverges for $\eta > 1/3$ and oscillates at exactly $1/3$.
2. The threshold is $2/50 = 0.04$.
3. One factor falls with $\eta$ and the other rises. The maximum is smallest where $1 - \eta\mu = \eta L - 1$.
4. Nesterov is rotation-invariant like momentum, so expect similar results in both orientations.
5. Bigger batches tolerate a larger $\eta$, which narrows the gap. But at a fixed number of examples they still make few updates, so expect a sweet spot at moderate $B$.
6. One step, because $1 - \eta L = 0$ and there is no error along the flat direction. Rates describe generic starts. Any flat component pays the full $\kappa$ cost.
7. Adam's step stays near $\eta$ in size while gradients shrink, until $\hat v$ catches up. A large $\eta$ on a quadratic shows sustained oscillation of amplitude about $\eta$.
8. The Hessian at $(0, \pm1)$ is $\mathrm{diag}(1, 2)$. Expect a roughly even split between the two minima.
9. Near the solution most steps accept $t = 1$, so convergence is fast.
10. $Hv \approx (\nabla f(w+\epsilon v) - \nabla f(w-\epsilon v))/(2\epsilon)$. Expect the sharpness to rise and level off near $2/\eta$, though small models may not get there.

## Check your understanding

**1. Why does one steep direction slow learning in all the others?** There is one learning rate. Stability needs $\eta < 2/L$, and then the flattest direction moves only a fraction $\eta\mu < 2/\kappa$ of the way per step.

**2. What does momentum buy?** It accumulates consistent gradient components and cancels alternating ones. On quadratics with tuned settings, the step count scales with $\sqrt\kappa$ instead of $\kappa$, the best possible for gradient-only methods in the worst case.

**3. Why doesn't constant-step SGD converge exactly?** Minibatch gradients stay noisy at the minimum, so each step jitters by an amount proportional to $\eta$. A decaying $\eta$ (with $\sum\eta_t = \infty$ and $\sum\eta_t^2 < \infty$) or a larger batch shrinks the jitter.

**4. When does Adam's per-coordinate scaling help?** When gradient scales differ by parameter and the ill-conditioning is aligned with the axes. For a rotated valley it behaves roughly like tuned momentum.

**5. Are saddle points dangerous?** Rarely as permanent traps, since negative-curvature components grow exponentially. But escape from very close takes time proportional to $\ln(1/\delta)$, causing plateaus. Noise helps.

## Further reading

- **Gabriel Goh, "Why Momentum Really Works"** (2017, Distill). https://distill.pub/2017/momentum/
  *Read for:* an interactive explanation of gradient descent and momentum on quadratics. *Level:* intro.
- **Sebastian Ruder, "An overview of gradient descent optimization algorithms"** (2016, arXiv:1609.04747). https://arxiv.org/abs/1609.04747
  *Read for:* a compact tour of momentum, Nesterov, Adagrad, RMSProp, Adam and later variants in one notation. *Level:* intro.
- **Ian Goodfellow, Yoshua Bengio, Aaron Courville, "Deep Learning", Chapter 8** (2016, MIT Press; free online). https://www.deeplearningbook.org/
  *Read for:* the textbook view of SGD, ill-conditioning, saddles, adaptive and second-order methods. *Level:* intermediate.
- **Stephen Boyd, Lieven Vandenberghe, "Convex Optimization"** (2004, Cambridge University Press; free PDF from the authors). https://web.stanford.edu/~boyd/cvxbook/
  *Read for:* convexity, gradient descent, backtracking line search and Newton's method with proofs (Chapters 2, 3 and 9). *Level:* intermediate.
- **Diederik P. Kingma, Jimmy Ba, "Adam: A Method for Stochastic Optimization"** (2015, ICLR; arXiv:1412.6980). https://arxiv.org/abs/1412.6980
  *Read for:* the original algorithm, its defaults and the bias-correction argument. *Level:* intermediate.
- **Ilya Loshchilov, Frank Hutter, "Decoupled Weight Decay Regularization"** (2019, ICLR; arXiv:1711.05101). https://arxiv.org/abs/1711.05101
  *Read for:* why L2 and weight decay differ under Adam, and AdamW. *Level:* intermediate.
- **Ashia C. Wilson, Rebecca Roelofs, Mitchell Stern, Nathan Srebro, Benjamin Recht, "The Marginal Value of Adaptive Gradient Methods in Machine Learning"** (2017, NeurIPS; arXiv:1705.08292). https://arxiv.org/abs/1705.08292
  *Read for:* the case that adaptive methods can generalize worse than SGD. *Level:* intermediate.
- **Yann Dauphin, Razvan Pascanu, Caglar Gulcehre, Kyunghyun Cho, Surya Ganguli, Yoshua Bengio, "Identifying and attacking the saddle point problem in high-dimensional non-convex optimization"** (2014, arXiv:1406.2572). https://arxiv.org/abs/1406.2572
  *Read for:* the argument that saddles, not local minima, dominate high-dimensional losses. *Level:* advanced.
- **Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, Oriol Vinyals, "Understanding deep learning requires rethinking generalization"** (2017, ICLR; arXiv:1611.03530). https://arxiv.org/abs/1611.03530
  *Read for:* the random-label experiments that reframe what optimization success means. *Level:* intermediate.
- **Nitish Shirish Keskar, Dheevatsa Mudigere, Jorge Nocedal, Mikhail Smelyanskiy, Ping Tak Peter Tang, "On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima"** (2017, ICLR; arXiv:1609.04836). https://arxiv.org/abs/1609.04836
  *Read for:* the evidence linking large batches, sharp minima and worse generalization. *Level:* advanced.
- **Laurent Dinh, Razvan Pascanu, Samy Bengio, Yoshua Bengio, "Sharp Minima Can Generalize For Deep Nets"** (2017, ICML, PMLR 70; arXiv:1703.04933). https://arxiv.org/abs/1703.04933
  *Read for:* why sharpness changes under reparametrization and cannot explain generalization alone. *Level:* advanced.
- **Jeremy M. Cohen, Simran Kaur, Yuanzhi Li, J. Zico Kolter, Ameet Talwalkar, "Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability"** (2021, ICLR; arXiv:2103.00065). https://arxiv.org/abs/2103.00065
  *Read for:* evidence that real training runs at sharpness near $2/\eta$. *Level:* advanced.
