---
title: "Neural Networks and Backpropagation"
---

> **Level 2 · Module 1** · Prerequisites: L0-02, L0-03, L0-05, L1-02 · Time: 6 to 8 hours · Code: `code/L2-01-neural-networks-and-backprop/`

## What you will be able to do

- Explain why a single perceptron cannot learn XOR and why one hidden layer with a nonlinearity can.
- State what the universal approximation theorems promise, and name three things they do not promise.
- Choose an output layer and loss for regression and classification, and justify computing the loss from logits.
- Derive backpropagation for a multilayer perceptron in matrix form and implement it in NumPy.
- Verify a hand-written backward pass with gradient checking, and recognize a false alarm when you see one.
- Train a from-scratch MLP on real data and compare it honestly with a linear baseline.

## 1. From linear models to neurons

L1-02 ended with softmax regression: scores $\mathbf{z} = W^\top\mathbf{x} + \mathbf{b}$, a softmax, and a cross-entropy loss. Every decision boundary such a model can draw is a hyperplane. This module is about what happens when you put learned features in front of that linear layer.

The oldest version of the idea is the **perceptron** [Rosenblatt, 1958]: compute $\mathbf{w}^\top\mathbf{x} + b$ and output 1 if it is positive, 0 otherwise. Its learning rule is simple. With labels $y \in \{-1, +1\}$, visit the examples one at a time and, whenever one is on the wrong side of the boundary, move the boundary toward it: $\mathbf{w} \leftarrow \mathbf{w} + y\,\mathbf{x}$, $b \leftarrow b + y$. A classical result (the perceptron convergence theorem) says that if a hyperplane separates the classes, the rule stops making mistakes after finitely many updates.

Script 1 runs it on the four points of the unit square. AND converges after 9 passes, OR after 6. XOR (output 1 when exactly one input is 1) never converges: after 100 passes the weights are cycling and happen to be back at zero. That is not bad luck. Suppose some $w_1, w_2, b$ separated XOR. The points $(0,1)$ and $(1,0)$ need $w_2 + b > 0$ and $w_1 + b > 0$; the points $(0,0)$ and $(1,1)$ need $b \le 0$ and $w_1 + w_2 + b \le 0$. Adding the first two gives $w_1 + w_2 + 2b > 0$, adding the last two gives $w_1 + w_2 + 2b \le 0$. Contradiction: no line separates XOR.

## 2. Multilayer perceptrons

The fix is to transform the inputs before the linear classifier. A **multilayer perceptron (MLP)** stacks layers. With $\mathbf{h}_0 = \mathbf{x}$,

$$\mathbf{z}_l = W_l^\top \mathbf{h}_{l-1} + \mathbf{b}_l, \qquad \mathbf{h}_l = \phi(\mathbf{z}_l), \qquad l = 1, \dots, L-1,$$

and a final linear layer $\mathbf{z}_L = W_L^\top\mathbf{h}_{L-1} + \mathbf{b}_L$ produces the outputs. $W_l$ has shape (fan-in, fan-out), $\mathbf{z}_l$ is the **pre-activation**, $\phi$ is an elementwise **activation function**, and $\mathbf{h}_l$ is the vector of **hidden units**. The intermediate layers are "hidden" because the data never says what they should be; training decides.

The nonlinearity is essential. Without it, $W_2^\top(W_1^\top\mathbf{x}) = (W_1 W_2)^\top\mathbf{x}$ and the stack collapses into one linear layer, which script 1 confirms numerically. With it, XOR needs only two hidden ReLU units, where $\mathrm{ReLU}(z) = \max(0, z)$:

$$h_1 = \mathrm{ReLU}(x_1 + x_2), \quad h_2 = \mathrm{ReLU}(x_1 + x_2 - 1), \quad \hat y = h_1 - 2h_2.$$

The hidden layer maps the four inputs to $(0,0), (1,0), (1,0), (2,1)$. The two "exactly one" inputs land on the same point, and in the new coordinates a line separates the classes. That is the general picture: **hidden layers learn a representation in which the last, linear layer can do its job.** For a classifier, the last layer is softmax regression from L1-02, applied to learned features instead of raw inputs.

## 3. Activation functions

| Name | $\phi(z)$ | $\phi'(z)$ | Range of $\phi'$ |
|---|---|---|---|
| sigmoid | $\sigma(z) = 1/(1 + e^{-z})$ | $\sigma(z)(1 - \sigma(z))$ | $(0, 0.25]$ |
| tanh | $\tanh z$ | $1 - \tanh^2 z$ | $(0, 1]$ |
| ReLU | $\max(0, z)$ | 1 if $z > 0$, else 0 | $\{0, 1\}$ |
| leaky ReLU | $\max(\alpha z, z)$, small $\alpha$ | 1 or $\alpha$ | $\{\alpha, 1\}$ |
| GELU | $z\,\Phi(z)$ | smooth | |

Here $\Phi$ is the standard normal CDF, so GELU is a smooth curve that behaves like ReLU for large $|z|$.

What matters most for training is **how much gradient passes through.** Backpropagation (section 6) multiplies by $\phi'(z)$ once per layer. The sigmoid's derivative is at most 0.25, and far smaller when $|z|$ is large (the unit is **saturated**), so in a deep sigmoid network the gradient reaching the first layers is a product of many small numbers. Script 6 measures it in a 10-layer network: with sigmoid, the first layer's weight gradient is $5.5 \times 10^{-6}$ times the last layer's, while tanh and ReLU stay within a factor of about 2. ReLU is also cheap, has gradient exactly 1 for active units, and outputs true zeros; [Glorot et al., 2011] showed that deep rectifier networks train well on purely supervised tasks, which helped make it the default hidden activation. Its failure mode is the **dead ReLU**, a unit whose pre-activation is negative for every input and so never receives gradient; leaky ReLU and GELU keep a little gradient there. Keep sigmoid for outputs that must be probabilities.

## 4. What one hidden layer can represent

The **universal approximation theorems** say that MLPs with one hidden layer are, in principle, enough. [Cybenko, 1989] proved that finite sums of sigmoidal units can uniformly approximate any continuous function on the unit hypercube; [Hornik et al., 1989] proved a similar result for one hidden layer with any squashing function; and [Leshno et al., 1993] showed that a network with a (locally bounded, piecewise continuous) activation is a universal approximator exactly when the activation is not a polynomial, provided the units have bias terms. ReLU qualifies.

For ReLU on one input variable, the construction fits in a line. A sum $c_0 + \sum_k a_k\,\mathrm{ReLU}(x - t_k)$ is a piecewise-linear function whose slope changes by $a_k$ at each knot $t_k$. Choose knots on a grid and slopes that join the dots, and $N$ units linearly interpolate $f$ at $N + 1$ points (script 2, part 1). For smooth $f$, linear interpolation with spacing $h = 1/N$ has error at most $h^2 \max|f''|/8$. For $\sin(2\pi x)$ that bound is $4.93/N^2$, which predicts 0.019 at $N = 16$ and 0.0012 at $N = 64$; the script measures 0.0188 and 0.0012.

What the theorems do not tell you:

1. **How many units.** For $\sin(2\pi \cdot 20x)$ the bound is 400 times larger, and the script needs 256 units to reach error 0.03. In $d$ dimensions a grid with spacing $h$ has $(1/h)^d$ cells, so grid-style constructions grow exponentially with dimension.
2. **Whether training finds the weights.** The theorems are about existence; the last experiment below shows the gap.
3. **Whether the function generalizes** beyond a finite training set (L1-01, L2-02).

**Depth.** Width is not the only resource. The "tent map" $g(x) = 2\,\mathrm{ReLU}(x) - 4\,\mathrm{ReLU}(x - \tfrac12)$ folds $[0,1]$ onto itself with 2 units. Composing it $k$ times gives a sawtooth with $2^k$ linear pieces from $2k$ units in $k$ layers (script 2, part 2: 64 pieces from 12 units). A one-hidden-layer ReLU network with $m$ units on a line has at most $m + 1$ pieces, so it needs at least $2^k - 1$ units for the same function. [Telgarsky, 2016] builds on this construction to prove that some functions computed by deep networks cannot be approximated by much shallower ones unless those are exponentially wide. Depth can buy exponential savings, for some functions.

**Representable is not learnable.** Part 3 of script 2 trains networks on the 8-piece sawtooth with Adam from random starts. Three layers of 2 ReLUs can represent it exactly with 19 parameters, yet all three seeds end at the MSE of a constant prediction (0.0837): training never finds the solution that exists. A width-16 net also fails. Networks with 769 or 4,353 parameters fit it well. Networks much larger than strictly necessary are often easier to train; why is still an open research question (L0-05, section 8).

## 5. Output layers and loss functions

Choose the output layer and loss together, as a probabilistic model whose negative log-likelihood is the loss (L0-04, L1-02):

- **Regression.** Linear output, mean squared error. This is maximum likelihood under Gaussian noise.
- **Binary classification.** One logit $z$, $p = \sigma(z)$, binary cross-entropy.
- **$K$ classes.** $K$ logits, softmax, cross-entropy $\mathcal{L} = -\frac1n\sum_i \log p_{i, y_i}$.

With softmax and cross-entropy, the gradient with respect to the logits is $(P - Y)/n$ for a batch mean, where $P$ holds the predicted probabilities and $Y$ the one-hot labels (L1-02 derives it). It is large when the model is confidently wrong. Compare squared error on a sigmoid output: $\partial \tfrac12(p - y)^2/\partial z = (p - y)\,p(1 - p)$. If $y = 1$ and the model says $p = 0.001$, the factor $p(1-p)$ makes the gradient about $10^{-3}$, so a confidently wrong unit barely learns. Cross-entropy cancels that factor, which is one reason it is the standard classification loss.

Two practical rules follow. Compute the loss from logits with a log-softmax (subtract the max first, L0-01), never as `log(softmax(z))`, which overflows or takes the log of 0. And a randomly initialized $K$-class classifier should start near loss $\ln K$, the cross-entropy of a uniform guess; a much larger starting loss means the initial logits are too large (section 8 shows a case).

## 6. Deriving backpropagation

L0-03 built reverse-mode autodiff for scalars: each node passes its adjoint back to its inputs, multiplied by the local derivative, and contributions add where paths meet. **Backpropagation** [Rumelhart et al., 1986] is that algorithm applied to a network's loss. For an MLP we can do the bookkeeping once, in matrix form, and get a few lines of NumPy instead of millions of scalar nodes.

**A scalar warm-up.** One input $x = 2$, one tanh hidden unit with $w_1 = 0.5$, a linear output with $w_2 = 2$, target $y = 1$, loss $\tfrac12(\hat y - y)^2$. Forward: $z = 1$, $h = \tanh 1 = 0.7616$, $\hat y = 1.5232$, loss $0.1369$. Backward, one local derivative at a time:

| Quantity | Rule | Value |
|---|---|---|
| $\partial L/\partial \hat y$ | $\hat y - y$ | $0.5232$ |
| $\partial L/\partial w_2$ | $\partial L/\partial \hat y \cdot h$ | $0.3985$ |
| $\partial L/\partial h$ | $\partial L/\partial \hat y \cdot w_2$ | $1.0464$ |
| $\partial L/\partial z$ | $\partial L/\partial h \cdot (1 - h^2)$ | $0.4395$ |
| $\partial L/\partial w_1$ | $\partial L/\partial z \cdot x$ | $0.8789$ |

Each row reuses the row above, and each needs one value saved from the forward pass. That reuse is why backprop costs a small multiple of one forward pass, instead of one forward pass per parameter as numerical differentiation would.

**Matrix form.** Now take a batch: $X$ is $n \times d$ with one example per row, $H_0 = X$, and for each layer $Z_l = H_{l-1}W_l + \mathbf{1}\mathbf{b}_l^\top$, $H_l = \phi(Z_l)$, with $Z_L$ the logits. Write $G_l = \partial \mathcal{L}/\partial Z_l$, a matrix of the same shape as $Z_l$ ($n \times k_l$).

*Output layer.* From section 5, $G_L = (P - Y)/n$.

*Weights.* Entry $(i, j)$ of $Z_l$ is $\sum_m (H_{l-1})_{im}(W_l)_{mj} + (b_l)_j$, so $W_{mj}$ affects every row $i$ of column $j$, and the chain rule sums over them:

$$\frac{\partial\mathcal{L}}{\partial (W_l)_{mj}} = \sum_i (G_l)_{ij}\,(H_{l-1})_{im} \quad\Longrightarrow\quad \frac{\partial\mathcal{L}}{\partial W_l} = H_{l-1}^\top G_l, \qquad \frac{\partial\mathcal{L}}{\partial \mathbf{b}_l} = G_l^\top\mathbf{1}.$$

The bias gradient sums $G_l$ over the batch, because the bias is shared by every row.

*Inputs to the layer.* By the same argument with the roles swapped, $\partial\mathcal{L}/\partial H_{l-1} = G_l W_l^\top$.

*Through the activation.* $\phi$ acts elementwise, so its Jacobian is diagonal and the product is elementwise:

$$G_{l-1} = \left(G_l W_l^\top\right) \odot \phi'(Z_{l-1}).$$

That is the whole algorithm: start from $G_L$ and, for $l = L, \dots, 1$, compute the parameter gradients and step down a layer. To rederive it, use shapes: each gradient has the shape of its variable, and usually only one product of the available matrices fits. $\partial\mathcal{L}/\partial W_l$ is (fan-in $\times$ fan-out), and from $H_{l-1}$ ($n \times$ fan-in) and $G_l$ ($n \times$ fan-out) only $H_{l-1}^\top G_l$ works.

**Cost and memory.** A forward layer is one matrix product, about $2 n\,k_{l-1} k_l$ floating-point operations. The backward pass does two products of the same size per layer ($H^\top G$ and $G W^\top$), so a training step costs roughly three forward passes. The backward pass needs every $H_{l-1}$ and $Z_l$ (or $H_l$) from the forward pass, so they are cached, and activation memory grows with batch size and depth. This is reverse mode's memory cost from L0-03, at the scale of matrices; L4-02 does the accounting for large models.

`mlp_numpy.py` (script 3) implements these equations: `forward` caches (layer input, pre-activation, output) per layer, and `loss_and_grads` runs the backward recursion. Note that tanh's derivative is computed from the output ($1 - H^2$) and ReLU's from the input ($Z > 0$); confusing the two is a classic bug that script 4 plants on purpose.

## 7. Gradient checking

A wrong backward pass rarely crashes. The network still trains, just worse, and you blame the learning rate. So before trusting a hand-written gradient, compare it with central differences (L0-03, section 8): perturb each parameter by $\pm h$, recompute the loss, and compare $(\mathcal{L}^+ - \mathcal{L}^-)/2h$ with the analytic value using a relative error. The CS231n notes give practical thresholds: above $10^{-2}$ usually means a bug, below $10^{-7}$ is good, and checks should run in double precision with few data points [CS231n, Neural Networks Part 3].

Script 4 checks a network with two hidden layers, 111 parameters, on 8 examples (222 extra forward passes, a fraction of a second). Four lessons come out of it.

1. **The correct implementation passes** for all three activations, with worst absolute errors near $2 \times 10^{-10}$. The sigmoid's worst relative error, $1.7 \times 10^{-6}$, looks borderline, but its absolute error is the same round-off as the others; the gradient involved is tiny ($5 \times 10^{-5}$, in the first layer). Read absolute and relative error together.
2. **Float32 makes the check useless**: relative error 0.035 for a correct gradient. Check in float64.
3. **Real bugs are unmistakable.** $\tanh'$ computed as $1 - Z^2$, or the activation derivative omitted, gives relative error 1.0.
4. **The L0-03 engine agrees.** The same small tanh network built from L0-03's scalar `Value` objects gives the same loss exactly and the same gradients to $6 \times 10^{-17}$: the matrix equations and the scalar engine are the same algorithm.

Two limits. Central differences that straddle a ReLU kink are wrong, so an isolated failure on a ReLU net deserves a second look. And a passing check proves only that the gradient matches the function you wrote; a loss with the wrong sign passes.

## 8. Training on real data, and what to expect

Script 5 trains the MLP with plain minibatch SGD on scikit-learn's 8x8 digits, with the same split and standardization as L1-02's softmax regression (1,257 training and 540 test images).

Two results deserve attention. First, the initial loss is 3.216, not $\ln 10 = 2.303$. The weights use He scaling (variance $2/\text{fan-in}$, explained in L2-02), which is designed for ReLU hidden layers; applied to the output layer as well, it makes the initial logits large enough that random guesses are confidently wrong. Training recovers, but a smaller output layer would start at chance (exercise 6). Second, and more important, **the MLP does not clearly beat the linear model here.** Softmax regression scores 0.972; the MLPs score 0.970 to 0.975 averaged over 5 seeds, with seed-to-seed standard deviations of 0.003 to 0.006. A 540-image test set has a standard error of about 0.007 at this accuracy (L1-01), so these differences are within noise. On 8x8 digits a linear model already gets 97%, and every MLP reaches 100% training accuracy, so the hidden layers have little left to add. Hidden layers pay off when the right features are not linear in the inputs, as with raw images at higher resolution (L2-03) or text (L2-04, L2-05). Always run the baseline.

The detailed run prints test loss to show the curves; in real work you would monitor a validation set and touch the test set once, as L2-02's training loop does. Script 5's settings were fixed before running it, not tuned on test accuracy.

## Hands-on

Run the scripts from `code/L2-01-neural-networks-and-backprop/`, for example `../../.venv/bin/python grad_check.py`. They need NumPy and scikit-learn; `grad_check.py` also imports `engine.py` from `code/L0-03-calculus-and-autodiff/`. Outputs below are from NumPy 2.5.4 and scikit-learn 1.9.1 on Linux. Trailing digits may differ on other platforms. The machine was shared with other jobs during these runs, so the per-run timings in script 5 are only rough.

### Script 1: the perceptron and XOR

```python
"""The perceptron rule: it learns AND and OR, never settles on XOR,
and one hidden layer of two ReLU units fixes that."""
import numpy as np


def perceptron(X, y, max_epochs=100):
    """Labels y in {-1, +1}. On a mistake, move the boundary toward the example."""
    w, b = np.zeros(X.shape[1]), 0.0
    for epoch in range(1, max_epochs + 1):
        mistakes = 0
        for x_i, y_i in zip(X, y):
            if y_i * (x_i @ w + b) <= 0:      # wrong side of (or on) the boundary
                w, b = w + y_i * x_i, b + y_i
                mistakes += 1
        if mistakes == 0:
            return w, b, epoch, True
    return w, b, max_epochs, False


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
for name, target in [("AND", [0, 0, 0, 1]), ("OR", [0, 1, 1, 1]), ("XOR", [0, 1, 1, 0])]:
    y = 2 * np.array(target) - 1
    w, b, epochs, converged = perceptron(X, y)
    pred = (X @ w + b > 0).astype(int)
    print(f"{name}: converged={converged} after {epochs} epochs, w={w}, b={b:+.0f}, "
          f"predictions={pred.tolist()} targets={target}")

# Stacking linear layers does not help: two linear maps compose to one linear map.
rng = np.random.default_rng(0)
W1, W2 = rng.normal(size=(2, 5)), rng.normal(size=(5, 1))
print("\nlinear-linear network equals one linear layer:", np.allclose((X @ W1) @ W2, X @ (W1 @ W2)))

# One hidden layer with a nonlinearity does. Weights chosen by hand:
#   h1 = relu(x1 + x2), h2 = relu(x1 + x2 - 1), output = h1 - 2*h2
W1 = np.array([[1.0, 1.0], [1.0, 1.0]])
b1 = np.array([0.0, -1.0])
w2 = np.array([1.0, -2.0])
H = np.maximum(X @ W1 + b1, 0.0)
print("hidden activations per input:", H.tolist())
print("XOR network output:", (H @ w2).tolist())
```

```text
AND: converged=True after 9 epochs, w=[3. 2.], b=-4, predictions=[0, 0, 0, 1] targets=[0, 0, 0, 1]
OR: converged=True after 6 epochs, w=[2. 2.], b=-1, predictions=[0, 1, 1, 1] targets=[0, 1, 1, 1]
XOR: converged=False after 100 epochs, w=[0. 0.], b=+0, predictions=[0, 0, 0, 0] targets=[0, 1, 1, 0]

linear-linear network equals one linear layer: True
hidden activations per input: [[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [2.0, 1.0]]
XOR network output: [0.0, 1.0, 1.0, 0.0]
```

AND and OR converge, XOR does not, and the hand-built two-unit hidden layer maps XOR's inputs to a linearly separable representation (section 2).

### Script 2: universal approximation, depth, and learnability

```python
"""What one hidden layer can represent, how many units it takes, what depth buys,
and whether gradient-based training actually finds the weights that exist."""
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor

relu = lambda z: np.maximum(z, 0.0)
x = np.linspace(0.0, 1.0, 2 ** 14 + 1)  # dyadic grid: every kink below lands on a grid point


def interpolating_net(f, n_units):
    """One hidden layer of n_units ReLUs that linearly interpolates f at n_units + 1 knots.
    Unit k is relu(x - t_k); its output weight is the change of slope at knot t_k."""
    t = np.linspace(0.0, 1.0, n_units + 1)
    slopes = np.diff(f(t)) / np.diff(t)
    out_w = np.diff(slopes, prepend=0.0)          # first unit sets the initial slope
    return lambda x: f(t[0]) + relu(x[:, None] - t[:-1][None, :]) @ out_w


def count_pieces(y):
    """Number of linear pieces of a piecewise-linear function sampled on the grid x."""
    slope = np.round(np.diff(y) / np.diff(x), 6)
    return 1 + int(np.sum(slope[1:] != slope[:-1]))


# 1. Width needed for a given accuracy, for a smooth and a wiggly target.
targets = {"sin(2 pi x)": lambda x: np.sin(2 * np.pi * x),
           "sin(2 pi 20 x)": lambda x: np.sin(2 * np.pi * 20 * x)}
print(f"{'units':>6} | " + " | ".join(f"max error on {k:<14}" for k in targets))
for n in [4, 16, 64, 256, 1024]:
    errs = [np.abs(interpolating_net(f, n)(x) - f(x)).max() for f in targets.values()]
    print(f"{n:>6} | " + " | ".join(f"{e:>27.2e}" for e in errs))

# 2. Depth: the tent map g(x) = 2 relu(x) - 4 relu(x - 1/2) uses 2 ReLUs.
#    Composing it k times uses 2k units in k layers and makes 2**k linear pieces.
#    One hidden layer with m ReLUs on a line has at most m + 1 pieces.
tent = lambda z: 2 * relu(z) - 4 * relu(z - 0.5)
y = x.copy()
print("\nlayers  units  pieces  (a 1-hidden-layer net needs >= pieces - 1 units)")
for k in range(1, 7):
    y = tent(y)
    print(f"{k:>6} {2 * k:>6} {count_pieces(y):>7}")

# 3. Existence is not learnability. Train on the k = 3 sawtooth (8 pieces). Three layers
#    of 2 ReLUs can represent it exactly; does Adam find those weights from a random start?
x_tr = np.linspace(0, 1, 500)[:, None]
saw = tent(tent(tent(x_tr[:, 0])))
print("\ntraining on the 8-piece sawtooth, full-batch Adam, 2000 steps, 3 seeds")
print(f"  (predicting the mean everywhere gives MSE {saw.var():.4f})")
warnings.filterwarnings("ignore", category=ConvergenceWarning)
for layers in [(2, 2, 2), (16,), (256,), (64, 64)]:
    mses = []
    for seed in range(3):
        net = MLPRegressor(hidden_layer_sizes=layers, activation="relu", learning_rate_init=1e-2,
                           batch_size=len(x_tr), max_iter=2000, n_iter_no_change=2000, tol=0,
                           random_state=seed)
        mses.append(np.mean((net.fit(x_tr, saw).predict(x_tr) - saw) ** 2))
    n_par = sum(a * b + b for a, b in zip((1,) + layers, layers + (1,)))
    print(f"  hidden {str(layers):<10} {n_par:>5} params  training MSE " + "  ".join(f"{m:.4f}" for m in mses))
```

```text
 units | max error on sin(2 pi x)    | max error on sin(2 pi 20 x)
     4 |                    2.11e-01 |                    1.00e+00
    16 |                    1.88e-02 |                    1.61e+00
    64 |                    1.20e-03 |                    4.36e-01
   256 |                    7.53e-05 |                    2.99e-02
  1024 |                    4.71e-06 |                    1.88e-03

layers  units  pieces  (a 1-hidden-layer net needs >= pieces - 1 units)
     1      2       2
     2      4       4
     3      6       8
     4      8      16
     5     10      32
     6     12      64

training on the 8-piece sawtooth, full-batch Adam, 2000 steps, 3 seeds
  (predicting the mean everywhere gives MSE 0.0837)
  hidden (2, 2, 2)     19 params  training MSE 0.0837  0.0837  0.0837
  hidden (16,)         49 params  training MSE 0.0767  0.0837  0.0789
  hidden (256,)       769 params  training MSE 0.0022  0.0018  0.0028
  hidden (64, 64)    4353 params  training MSE 0.0003  0.0002  0.0007
```

Part 1 tracks the interpolation bound (16 times less error per 4 times more units). Part 2 doubles the pieces with every 2 extra units. In part 3 the 19-parameter network that could represent the target exactly never finds it. (`MLPRegressor` adds a small L2 penalty by default, `alpha=1e-4`.)

### Script 3: the MLP

```python
"""A multilayer perceptron in NumPy with a hand-written backward pass.

Shapes: a batch X is (n, d). Layer l has weights W (fan_in, fan_out) and bias
b (fan_out,), so Z = H @ W + b is (n, fan_out). The last layer has no
activation: its outputs are logits that go into softmax cross-entropy.
"""
import numpy as np

# Each activation: (function, derivative). The derivative receives both the
# pre-activation Z and the output H, because some are cheaper from H.
ACTIVATIONS = {
    "relu": (lambda Z: np.maximum(Z, 0.0), lambda Z, H: (Z > 0).astype(Z.dtype)),
    "tanh": (np.tanh, lambda Z, H: 1.0 - H ** 2),
    "sigmoid": (lambda Z: 1.0 / (1.0 + np.exp(-Z)), lambda Z, H: H * (1.0 - H)),
}


def init_params(sizes, rng, dtype=np.float64):
    """Gaussian weights with variance 2 / fan_in (He scaling; L2-02 explains why)."""
    return [[rng.normal(0.0, np.sqrt(2.0 / m), size=(m, k)).astype(dtype), np.zeros(k, dtype)]
            for m, k in zip(sizes[:-1], sizes[1:])]


def log_softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)  # shift for stability (L0-01)
    return Z - np.log(np.exp(Z).sum(axis=1, keepdims=True))


def forward(params, X, act="relu"):
    """Return logits and a cache of (layer input, pre-activation, output) per layer."""
    f, _ = ACTIVATIONS[act]
    cache, H = [], X
    for l, (W, b) in enumerate(params):
        Z = H @ W + b
        H_out = Z if l == len(params) - 1 else f(Z)
        cache.append((H, Z, H_out))
        H = H_out
    return H, cache


def loss_and_grads(params, X, y, act="relu"):
    """Mean cross-entropy over the batch and its gradient for every W and b."""
    logits, cache = forward(params, X, act)
    n = X.shape[0]
    logp = log_softmax(logits)
    loss = -logp[np.arange(n), y].mean()

    dZ = np.exp(logp)                 # softmax probabilities P
    dZ[np.arange(n), y] -= 1.0        # P - Y
    dZ /= n                           # because the loss is a mean over n rows
    _, dact = ACTIVATIONS[act]
    grads = [None] * len(params)
    for l in reversed(range(len(params))):
        H_in, _, _ = cache[l]
        W, _ = params[l]
        grads[l] = [H_in.T @ dZ, dZ.sum(axis=0)]   # (fan_in, n) @ (n, fan_out)
        if l > 0:
            dH = dZ @ W.T                           # gradient wrt this layer's input
            _, Z_prev, H_prev = cache[l - 1]
            dZ = dH * dact(Z_prev, H_prev)          # through the elementwise nonlinearity
    return loss, grads


def predict(params, X, act="relu"):
    return forward(params, X, act)[0].argmax(axis=1)


def train(params, X, y, rng, act="relu", lr=0.1, epochs=50, batch_size=32, log_every=0,
          X_test=None, y_test=None):
    """Plain minibatch SGD (optimizers are L0-05's topic). Updates params in place."""
    n = X.shape[0]
    for epoch in range(1, epochs + 1):
        order = rng.permutation(n)                  # reshuffle every epoch
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            _, grads = loss_and_grads(params, X[idx], y[idx], act)
            for (W, b), (dW, db) in zip(params, grads):
                W -= lr * dW
                b -= lr * db
        if log_every and epoch % log_every == 0:
            train_loss = loss_and_grads(params, X, y, act)[0]
            msg = f"epoch {epoch:3d}  train loss {train_loss:.4f}  train acc {np.mean(predict(params, X, act) == y):.3f}"
            if X_test is not None:
                test_loss = loss_and_grads(params, X_test, y_test, act)[0]
                msg += f"  test loss {test_loss:.4f}  test acc {np.mean(predict(params, X_test, act) == y_test):.3f}"
            print(msg)
    return params
```

This file is a library: running it prints nothing. Scripts 4 to 6 import it.

### Script 4: gradient checking

```python
"""Gradient checking for the NumPy MLP: against central differences, against
the scalar autograd engine from L0-03, and against deliberately broken backward passes."""
import sys
from pathlib import Path

import numpy as np

import mlp_numpy as mlp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "L0-03-calculus-and-autodiff"))
from engine import Value  # noqa: E402  (the L0-03 engine, reused unchanged)


def numerical_grads(params, X, y, act, h=1e-6):
    """Central difference for every parameter: 2 forward passes each."""
    out = []
    for W, b in params:
        pair = []
        for P in (W, b):
            G = np.zeros_like(P)
            for i in np.ndindex(P.shape):
                old = P[i]
                P[i] = old + h
                up = mlp.loss_and_grads(params, X, y, act)[0]
                P[i] = old - h
                down = mlp.loss_and_grads(params, X, y, act)[0]
                P[i] = old
                G[i] = (up - down) / (2 * h)
            pair.append(G)
        out.append(pair)
    return out


def errors(g_a, g_n, floor=1e-8):
    """Worst relative error and worst absolute error over all parameters."""
    pairs = [(a, n) for la, ln in zip(g_a, g_n) for a, n in zip(la, ln)]
    rel = max((np.abs(a - n) / np.maximum(floor, np.abs(a) + np.abs(n))).max() for a, n in pairs)
    return f"worst rel err {rel:.1e}, worst abs err {max(np.abs(a - n).max() for a, n in pairs):.1e}"


rng = np.random.default_rng(0)
sizes = [5, 7, 6, 3]                       # 5 inputs, two hidden layers, 3 classes
X, y = rng.normal(size=(8, 5)), rng.integers(0, 3, size=8)
n_params = sum(W.size + b.size for W, b in mlp.init_params(sizes, rng))
print(f"network {sizes}: {n_params} parameters, so {2 * n_params} forward passes per numerical check")

# 1. Correct backward pass, three activations, float64.
for act in ["relu", "tanh", "sigmoid"]:
    params = mlp.init_params(sizes, np.random.default_rng(1))
    _, g = mlp.loss_and_grads(params, X, y, act)
    print(f"  {act:<8} {errors(g, numerical_grads(params, X, y, act))}")

# 2. The same check in float32: round-off swamps the difference quotient.
params32 = mlp.init_params(sizes, np.random.default_rng(1), dtype=np.float32)
X32 = X.astype(np.float32)
_, g = mlp.loss_and_grads(params32, X32, y, "tanh")
print(f"  tanh, float32, h=1e-3: {errors(g, numerical_grads(params32, X32, y, 'tanh', h=1e-3))}")

# 3. Plant two classic bugs and confirm the check catches them.
good_tanh = mlp.ACTIVATIONS["tanh"]
params = mlp.init_params(sizes, np.random.default_rng(1))
bugs = {"tanh' uses Z instead of H": (np.tanh, lambda Z, H: 1.0 - Z ** 2),
        "activation derivative omitted": (np.tanh, lambda Z, H: np.ones_like(Z))}
for name, fake in bugs.items():
    mlp.ACTIVATIONS["tanh"] = fake
    _, g = mlp.loss_and_grads(params, X, y, "tanh")
    mlp.ACTIVATIONS["tanh"] = good_tanh          # numerical grads use the true forward pass
    print(f"  BUG {name:<30} {errors(g, numerical_grads(params, X, y, 'tanh'))}")


# 4. Same network, same numbers, built from scalar Values: the L0-03 engine must agree.
def engine_loss(params, X, y):
    P = [[[Value(w) for w in row] for row in W] + [[Value(c) for c in b]] for W, b in params]
    total = Value(0.0)
    for x_i, y_i in zip(X, y):
        h = [Value(v) for v in x_i]
        for l, layer in enumerate(P):
            *rows, bias = layer
            z = [sum((h[i] * rows[i][j] for i in range(len(h))), bias[j]) for j in range(len(bias))]
            h = z if l == len(P) - 1 else [v.tanh() for v in z]
        total = total + (sum(v.exp() for v in h).log() - h[y_i])   # -log softmax
    loss = total / len(y)
    loss.backward()
    flat = [v for layer in P for row in layer for v in row]
    return loss.data, np.array([v.grad for v in flat])


small = mlp.init_params([3, 4, 2], np.random.default_rng(2))
Xs, ys = rng.normal(size=(4, 3)), np.array([0, 1, 1, 0])
loss_np, g = mlp.loss_and_grads(small, Xs, ys, "tanh")
g_np = np.concatenate([np.concatenate([dW, db[None, :]]).ravel() for dW, db in g])
loss_eng, g_eng = engine_loss(small, Xs, ys)
print(f"\nL0-03 engine vs vectorized backprop on a [3, 4, 2] tanh net: "
      f"loss diff {abs(loss_np - loss_eng):.1e}, max grad diff {np.abs(g_np - g_eng).max():.1e}")
```

```text
network [5, 7, 6, 3]: 111 parameters, so 222 forward passes per numerical check
  relu     worst rel err 8.3e-08, worst abs err 1.9e-10
  tanh     worst rel err 4.2e-08, worst abs err 1.5e-10
  sigmoid  worst rel err 1.7e-06, worst abs err 1.8e-10
  tanh, float32, h=1e-3: worst rel err 3.5e-02, worst abs err 8.5e-05
  BUG tanh' uses Z instead of H      worst rel err 1.0e+00, worst abs err 7.9e-01
  BUG activation derivative omitted  worst rel err 1.0e+00, worst abs err 2.1e-01

L0-03 engine vs vectorized backprop on a [3, 4, 2] tanh net: loss diff 0.0e+00, max grad diff 5.6e-17
```

Section 7 discusses each line.

### Script 5: training on digits

```python
"""Train the from-scratch MLP on scikit-learn's 8x8 digits and compare it with
softmax regression on the same split as L1-02."""
import time

import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import mlp_numpy as mlp

X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
scaler = StandardScaler().fit(X_tr)                     # training statistics only
X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)
print(f"train {X_tr.shape}, test {X_te.shape}")

baseline = LogisticRegression(C=1.0, max_iter=5000).fit(X_tr, y_tr)
print(f"softmax regression (L1-02 baseline): test accuracy {baseline.score(X_te, y_te):.3f}\n")

# Sanity check before training: with random weights the loss should be near ln(10).
params = mlp.init_params([64, 128, 10], np.random.default_rng(0))
print(f"initial loss {mlp.loss_and_grads(params, X_tr, y_tr)[0]:.3f} (ln 10 = {np.log(10):.3f})")

# One run in detail: watch train and test loss separate.
mlp.train(params, X_tr, y_tr, np.random.default_rng(0), lr=0.05, epochs=60, log_every=10,
          X_test=X_te, y_test=y_te)

# Several architectures, 5 seeds each. Settings were fixed before looking at test results.
print(f"\n{'hidden layers':<16} {'act':<5} {'train acc':>9} {'test acc (5 seeds)':>22} {'s/run':>6}")
for hidden, act in [([32], "relu"), ([128], "relu"), ([128], "tanh"), ([128, 128], "relu")]:
    train_acc, test_acc = [], []
    t0 = time.perf_counter()
    for seed in range(5):
        rng = np.random.default_rng(seed)
        p = mlp.train(mlp.init_params([64, *hidden, 10], rng), X_tr, y_tr, rng, act=act, lr=0.05, epochs=60)
        train_acc.append(np.mean(mlp.predict(p, X_tr, act) == y_tr))
        test_acc.append(np.mean(mlp.predict(p, X_te, act) == y_te))
    secs = (time.perf_counter() - t0) / 5
    print(f"{str(hidden):<16} {act:<5} {np.mean(train_acc):>9.3f} "
          f"{np.mean(test_acc):>14.3f} +/- {np.std(test_acc):.3f} {secs:>6.2f}")
```

```text
train (1257, 64), test (540, 64)
softmax regression (L1-02 baseline): test accuracy 0.972

initial loss 3.216 (ln 10 = 2.303)
epoch  10  train loss 0.0957  train acc 0.987  test loss 0.1659  test acc 0.961
epoch  20  train loss 0.0459  train acc 0.996  test loss 0.1104  test acc 0.970
epoch  30  train loss 0.0278  train acc 0.998  test loss 0.0911  test acc 0.976
epoch  40  train loss 0.0191  train acc 1.000  test loss 0.0826  test acc 0.978
epoch  50  train loss 0.0142  train acc 1.000  test loss 0.0780  test acc 0.978
epoch  60  train loss 0.0112  train acc 1.000  test loss 0.0740  test acc 0.978

hidden layers    act   train acc     test acc (5 seeds)  s/run
[32]             relu      0.999          0.970 +/- 0.005   0.26
[128]            relu      1.000          0.975 +/- 0.003   0.45
[128]            tanh      1.000          0.974 +/- 0.006   0.64
[128, 128]       relu      1.000          0.974 +/- 0.004   1.13
```

Training loss keeps falling long after training accuracy hits 1.000, because cross-entropy keeps rewarding more confident correct answers. Test loss also keeps falling here, so this network is not yet overfitting in the sense of getting worse on new data. See section 8 for why the comparison with the baseline is a tie.

### Script 6: gradients by layer

```python
"""Gradient size per layer at initialization in a 10-layer MLP, for three activations.
Every network gets the same weight scale (variance 1 / fan_in), so only the
activation differs. Fixing this properly (initialization, normalization) is L2-02."""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

import mlp_numpy as mlp

X, y = load_digits(return_X_y=True)
X = StandardScaler().fit_transform(X)[:256]
y = y[:256]
sizes = [64] + [64] * 9 + [10]                  # 9 hidden layers + output = 10 weight matrices

z = np.linspace(-6, 6, 1201)
for act in ["sigmoid", "tanh", "relu"]:
    f, df = mlp.ACTIVATIONS[act]
    print(f"{act:<8} max derivative {df(z, f(z)).max():.3f}")

print(f"\n{'layer':>5} " + " ".join(f"{a:>10}" for a in ["sigmoid", "tanh", "relu"]) + "   (norm of dL/dW)")
norms = {}
for act in ["sigmoid", "tanh", "relu"]:
    rng = np.random.default_rng(0)
    params = [[rng.normal(0, np.sqrt(1.0 / m), size=(m, k)), np.zeros(k)] for m, k in zip(sizes[:-1], sizes[1:])]
    _, grads = mlp.loss_and_grads(params, X, y, act)
    norms[act] = [np.linalg.norm(dW) for dW, _ in grads]
for l in range(len(sizes) - 1):
    print(f"{l + 1:>5} " + " ".join(f"{norms[a][l]:>10.2e}" for a in norms))
print("ratio first/last " + " ".join(f"{norms[a][0] / norms[a][-1]:>10.1e}" for a in norms))
```

```text
sigmoid  max derivative 0.250
tanh     max derivative 1.000
relu     max derivative 1.000

layer    sigmoid       tanh       relu   (norm of dL/dW)
    1   3.13e-06   3.91e-01   2.93e-02
    2   4.51e-06   4.23e-01   2.94e-02
    3   1.66e-05   4.33e-01   2.99e-02
    4   8.19e-05   3.85e-01   3.02e-02
    5   3.65e-04   3.76e-01   2.33e-02
    6   1.58e-03   4.03e-01   2.51e-02
    7   6.53e-03   3.99e-01   2.38e-02
    8   2.79e-02   4.20e-01   2.11e-02
    9   1.39e-01   3.96e-01   1.99e-02
   10   5.66e-01   4.18e-01   1.40e-02
ratio first/last    5.5e-06    9.4e-01    2.1e+00
```

With sigmoid, the first layer's gradient is about $10^5$ times smaller than the last layer's, so with a single learning rate the early layers barely move. The ReLU column shrinks toward the output for another reason: with variance $1/\text{fan-in}$, each ReLU layer roughly halves the signal's second moment. Choosing the scale to prevent that is L2-02's initialization section.

## Common mistakes

- **Forgetting the nonlinearity**, or putting an activation on the output layer of a classifier whose loss expects logits. The first collapses the network into a linear model; the second squashes the logits and slows learning.
- **Putting $1/n$ in the wrong place.** If the loss is a batch mean, the gradient carries $1/n$ once. Doubling it or dropping it changes the effective learning rate by a factor of $n$ and is invisible except in a gradient check, where it shows as a constant ratio between analytic and numerical gradients.
- **Derivative from the wrong variable**: $\tanh'$ from $Z$ instead of $H$, or a sigmoid derivative written as $\sigma'(H)$. Script 4 shows the check catching it.
- **Gradient checks in float32**, with a tiny $h$, or with a relative error and no floor. Use float64, $h \approx 10^{-6}$, few examples, and read the absolute error too.
- **Taking `log(softmax(z))`.** Use a log-softmax; L0-01 shows the overflow and the log of zero.
- **Skipping the baseline.** A network with 100% training accuracy and 97% test accuracy sounds impressive until the linear model gets 97% too.
- **Reading the universal approximation theorem as a promise.** It guarantees that a network exists, not that it is small, that SGD finds it, or that it generalizes.

## Exercises

1. [warm-up] Run the perceptron rule by hand for one pass over the AND data from $\mathbf{w} = 0$, $b = 0$, and check it by printing the weights after epoch 1 in script 1.
2. [warm-up] Find two-unit ReLU weights that compute XNOR (1 when the inputs are equal).
3. [warm-up] For a 64-128-128-10 MLP and a batch of 32, write the shape of every $W_l$, $Z_l$ and $G_l$, and count the parameters.
4. [core] Redo the scalar example of section 6 with ReLU instead of tanh and $w_1 = -0.5$. What happens to the gradients, and what does it mean for a dead unit?
5. [core] Add a `"leaky_relu"` activation (slope 0.01 for negative inputs) to `ACTIVATIONS` and pass the gradient check.
6. [core] Scale the last layer's initial weights in script 5 by 0.01. What happens to the initial loss and to test accuracy?
7. [core] Add an L2 penalty $\frac{\lambda}{2}\sum_l \|W_l\|^2$ (not on biases) to `loss_and_grads`, gradient-check it, and train with a few values of $\lambda$.
8. [core] Replace cross-entropy with squared error on the softmax probabilities, derive and gradient-check the new gradient, and compare training speed.
9. [stretch] Replace plain SGD in `train` with Adam from L0-05 and compare loss curves for the same number of steps.
10. [stretch] Extend `mlp_numpy.py` to regression (linear output, MSE), fit $\sin(2\pi x)$ with 16 hidden ReLUs, and compare with the 16-unit construction in script 2.

### Hints and answers

1. $(0,0)$ is a mistake (score 0), so $b = -1$; $(0,1)$ and $(1,0)$ are then correct; $(1,1)$ scores $-1$, so $\mathbf{w} = (1, 1)$, $b = 0$.
2. Flip XOR: $\hat y = 1 - h_1 + 2h_2$ with the same hidden units.
3. $W$: $64 \times 128$, $128 \times 128$, $128 \times 10$; $Z_l$ and $G_l$: $32 \times 128$, $32 \times 128$, $32 \times 10$. Parameters: $8{,}320 + 16{,}512 + 1{,}290 = 26{,}122$.
4. $z = -1$, so $h = 0$ and $\phi'(z) = 0$: both $\partial L/\partial w_1$ and $\partial L/\partial w_2$ are 0. If that holds for every input, the unit never learns again.
5. The derivative is `np.where(Z > 0, 1.0, 0.01)`; it should pass at ReLU's level.
6. When I ran it, the initial loss fell from 3.216 to 2.305 and mean test accuracy over 5 seeds went from 0.975 to 0.977, within noise. A good start matters more in deep networks (L2-02).
7. The extra gradient is $\lambda W_l$. On digits, small $\lambda$ changes little; large $\lambda$ underfits.
8. With $\mathcal{L} = \frac{1}{2n}\|P - Y\|^2$ and $D = P - Y$, the logit gradient is $\frac1n P \odot (D - \mathbf{s}\mathbf{1}^\top)$ with $s_i = \sum_k D_{ik}P_{ik}$. Expect slower early progress.
9. Keep first- and second-moment arrays per parameter. Compare at equal steps and equal tuning effort.
10. Script 2's construction reaches 0.0188 maximum error with evenly spaced knots; training has to discover where kinks go, so compare over several seeds rather than assuming a match.

## Check your understanding

**1. Why can't a perceptron learn XOR, and what does a hidden layer change?** A perceptron's boundary is a hyperplane, and no line splits XOR (the constraints contradict each other). A hidden layer with a nonlinearity maps the inputs to coordinates where a line works. Without the nonlinearity the layers compose to one linear map.

**2. What does universal approximation guarantee?** With a non-polynomial activation and biases, one hidden layer can approximate any continuous function on a compact set arbitrarily well. It does not bound the width, promise that training finds the weights, or say anything about generalization. Depth can shrink the required size enormously for some functions.

**3. Why pair softmax with cross-entropy rather than squared error?** It is the categorical maximum-likelihood loss, and its logit gradient is $(P - Y)/n$, large when the model is confidently wrong. Squared error on a squashed output adds factors like $p(1-p)$ that vanish exactly then.

**4. Why is $\partial\mathcal{L}/\partial W_l = H_{l-1}^\top G_l$, and why is the bias gradient a sum?** Each weight affects one column of $Z_l$ in every row of the batch, so the chain rule sums over rows, which is the matrix product. The bias is added to every row, so its gradient sums $G_l$ over rows.

**5. A gradient check gives relative error $2 \times 10^{-6}$ on one parameter and $10^{-8}$ elsewhere. Bug?** Probably not. If that gradient is tiny and the absolute error is at round-off level ($10^{-10}$), the numerical side is the noisy one. Real bugs give relative errors near 1. Also check for a ReLU near its kink.

## Further reading

- **Michael Nielsen, "Neural Networks and Deep Learning"** (free online book, 2015). <http://neuralnetworksanddeeplearning.com/>
  *Read for:* chapter 2 derives backpropagation in the same four-equation style as section 6, and chapter 4 gives a visual argument for universal approximation. *Level:* intro.
- **Grant Sanderson (3Blue1Brown), "Backpropagation calculus"** (Neural Networks series, chapter 5). <https://www.3blue1brown.com/lessons/backpropagation-calculus>
  *Read for:* the chain rule through a tiny network, visually, before the matrix version. *Level:* intro.
- **Stanford CS231n, "Neural Networks Part 3: Learning and Evaluation"** (course notes). <https://cs231n.github.io/neural-networks-3/>
  *Read for:* gradient-check practice (centered differences, relative error thresholds, double precision, kinks) and pre-training sanity checks. *Level:* intro.
- **Simon J. D. Prince, "Understanding Deep Learning"** (2023, MIT Press; free PDF from the author). <https://udlbook.github.io/udlbook/>
  *Read for:* chapter 3 on shallow networks and their piecewise-linear regions, and chapter 7 on gradients and initialization. *Level:* intro to intermediate.
- **Ian Goodfellow, Yoshua Bengio, Aaron Courville, "Deep Learning"** (2016, MIT Press; free HTML version). <https://www.deeplearningbook.org/>
  *Read for:* chapter 6, Deep Feedforward Networks: output units, hidden units, universal approximation and back-propagation. *Level:* intermediate.
- **Frank Rosenblatt, "The perceptron: A probabilistic model for information storage and organization in the brain"** (1958, Psychological Review 65(6), 386 to 408). <https://doi.org/10.1037/h0042519>
  *Read for:* the original perceptron paper, framed as a model of the brain. Mainly historical interest. *Level:* intermediate.
- **David E. Rumelhart, Geoffrey E. Hinton, Ronald J. Williams, "Learning representations by back-propagating errors"** (1986, Nature 323, 533 to 536; paywalled at the publisher). <https://doi.org/10.1038/323533a0>
  *Read for:* the short paper that showed backpropagation training hidden units to learn useful features. *Level:* intermediate.
- **George Cybenko, "Approximation by superpositions of a sigmoidal function"** (1989, Mathematics of Control, Signals and Systems 2, 303 to 314). <https://doi.org/10.1007/BF02551274>
  *Read for:* the classic one-hidden-layer universal approximation proof. *Level:* advanced.
- **Kurt Hornik, Maxwell Stinchcombe, Halbert White, "Multilayer feedforward networks are universal approximators"** (1989, Neural Networks 2(5), 359 to 366). <https://doi.org/10.1016/0893-6080(89)90020-8>
  *Read for:* universal approximation for arbitrary squashing functions. *Level:* advanced.
- **Moshe Leshno, Vladimir Ya. Lin, Allan Pinkus, Shimon Schocken, "Multilayer feedforward networks with a nonpolynomial activation function can approximate any function"** (1993, Neural Networks 6(6), 861 to 867). <https://doi.org/10.1016/S0893-6080(05)80131-5>
  *Read for:* the clean "non-polynomial if and only if" condition, which covers ReLU. *Level:* advanced.
- **Matus Telgarsky, "Benefits of depth in neural networks"** (2016, COLT; arXiv:1602.04485). <https://arxiv.org/abs/1602.04485>
  *Read for:* the depth-separation result built on the sawtooth construction of script 2. *Level:* advanced.
- **Xavier Glorot, Antoine Bordes, Yoshua Bengio, "Deep Sparse Rectifier Neural Networks"** (2011, AISTATS, PMLR 15, 315 to 323). <https://proceedings.mlr.press/v15/glorot11a.html>
  *Read for:* evidence that ReLU networks train well without pre-training, and the case for sparse activations. *Level:* intermediate.
