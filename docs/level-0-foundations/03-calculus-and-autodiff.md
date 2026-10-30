---
title: "Calculus and Automatic Differentiation"
---

> **Level 0 · Module 3** · Prerequisites: L0-01, L0-02 · Time: 5 to 7 hours · Code: `code/L0-03-calculus-and-autodiff/`

## What you will be able to do

- Explain derivatives, gradients and Jacobians as local linear approximations, and compute them by hand for small functions.
- Apply the chain rule to compositions, including the multivariable case where it becomes a product of Jacobians.
- Estimate derivatives numerically, pick a sensible step size, and explain why a tiny step makes the answer worse.
- Run forward-mode and reverse-mode differentiation by hand on a computational graph, and say which is cheaper for a given problem.
- Build a scalar reverse-mode autodiff engine and test it against numerical gradients.

## 1. Derivatives as local linear approximations

Training a model means adjusting parameters to make a loss smaller, which requires knowing, for each parameter, how the loss responds to a small nudge. That is what a derivative tells you.

Take $f(x) = x^2$ at $x = 3$, so $f(3) = 9$. Nudge the input to $3.01$ and the output becomes $9.0601$: a change of $0.0601$ for an input change of $0.01$, a ratio of $6.01$. With a nudge of $0.001$ the ratio is $6.001$. The ratio settles on $6$ as the nudge shrinks, and that limit is the derivative:

$$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}.$$

The more useful reading is as an approximation. Near $x$,

$$f(x + h) \approx f(x) + f'(x)\,h.$$

Zoom in far enough on a smooth curve and it looks like a line with slope $f'(x)$. For $x^2$ at $3$, the prediction $f(3.01) \approx 9 + 6 \times 0.01 = 9.06$ is off by only $h^2 = 0.0001$. The error shrinks faster than $h$ itself, which is why small steps along the derivative are trustworthy.

You need only a handful of rules: the derivatives of elementary functions ($x^k \to kx^{k-1}$, $e^x \to e^x$, $\ln x \to 1/x$, $\sin x \to \cos x$, $\tanh x \to 1 - \tanh^2 x$), the sum rule, the product rule $(uv)' = u'v + uv'$, and the chain rule. Automatic differentiation is these rules applied mechanically by a program.

## 2. Partial derivatives and gradients

Losses depend on many parameters. Take $f(x_1, x_2) = x_1^2 x_2 + 3x_2$. A **partial derivative** measures the change when one input moves and the others are held fixed:

$$\frac{\partial f}{\partial x_1} = 2x_1 x_2, \qquad \frac{\partial f}{\partial x_2} = x_1^2 + 3.$$

At $(1, 2)$ both equal $4$. The **gradient** $\nabla f$ stacks the partial derivatives into a vector, here $(4, 4)$, and the linear approximation becomes a dot product (L0-02):

$$f(\mathbf{x} + \mathbf{h}) \approx f(\mathbf{x}) + \nabla f(\mathbf{x}) \cdot \mathbf{h}.$$

Check it: with $\mathbf{h} = (0.01, -0.02)$ the prediction is $8 + 0.04 - 0.08 = 7.96$, and the true value $f(1.01, 1.98)$ is $7.959798$.

The number $\nabla f \cdot \mathbf{v}$ is the **directional derivative**, the rate of change when moving in direction $\mathbf{v}$. A dot product is largest when the vectors align, so the gradient points in the direction of steepest increase and $-\nabla f$ in the direction of steepest decrease. That is the basis of gradient descent; turning it into a good optimizer is L0-05. This module is about computing gradients correctly and cheaply.

## 3. The chain rule and Jacobians

Models are compositions: layer, nonlinearity, layer, loss. If $y = g(u)$ and $u = h(x)$, the **chain rule** says

$$\frac{dy}{dx} = \frac{dy}{du}\cdot\frac{du}{dx}.$$

If $u$ changes 3 times as fast as $x$ and $y$ changes 2 times as fast as $u$, then $y$ changes 6 times as fast as $x$: rates multiply. For $y = \sin(x^2)$, with $u = x^2$, this gives $dy/dx = \cos(x^2) \cdot 2x$.

When a variable reaches the output along several paths, the contributions **add**. If $y = f(u, v)$ with $u$ and $v$ both depending on $x$,

$$\frac{dy}{dx} = \frac{\partial y}{\partial u}\frac{du}{dx} + \frac{\partial y}{\partial v}\frac{dv}{dx}.$$

For $y = x \cdot x$ (so $u = v = x$) this gives $x + x = 2x$, as it should. This sum over paths is the rule most often broken in hand-written backward passes, as section 8 shows.

A function $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$ has an $m \times n$ **Jacobian** $J$ with entries $J_{ij} = \partial f_i / \partial x_j$; row $i$ is the gradient of output $i$. The linear approximation is $\mathbf{f}(\mathbf{x} + \mathbf{h}) \approx \mathbf{f}(\mathbf{x}) + J\mathbf{h}$, so $J$ is the best linear map approximating $\mathbf{f}$ near $\mathbf{x}$. The multivariable chain rule is matrix multiplication: for $\mathbf{y} = \mathbf{g}(\mathbf{u})$ and $\mathbf{u} = \mathbf{h}(\mathbf{x})$,

$$J_{\mathbf{y}\leftarrow\mathbf{x}} = J_{\mathbf{y}\leftarrow\mathbf{u}}\; J_{\mathbf{u}\leftarrow\mathbf{x}}, \qquad (m \times k)(k \times n) = (m \times n).$$

Example used in the code: for $\mathbf{y} = W_2 \tanh(W_1\mathbf{x})$, write $\mathbf{h} = \tanh(W_1\mathbf{x})$. The Jacobians of the three steps are $W_1$, $\mathrm{diag}(1 - \mathbf{h}^2)$ (tanh acts elementwise, so its Jacobian is diagonal) and $W_2$, and the full Jacobian is $W_2\,\mathrm{diag}(1 - \mathbf{h}^2)\,W_1$. A deep network ending in a scalar loss has a gradient that is a long product of such Jacobians. The order in which you multiply that product is the difference between forward and reverse mode (section 6).

## 4. Three ways to get derivatives by computer

[Baydin et al., 2018] distinguish three families.

- **Symbolic differentiation** manipulates formulas, like a computer algebra system. It is exact, but derivative expressions can grow much larger than the original (the survey's term is "expression swell"), and it needs a closed-form formula rather than a program with loops and branches.
- **Numerical differentiation** evaluates the function at nearby points, for example the forward difference $(f(x+h) - f(x))/h$ or the central difference $(f(x+h) - f(x-h))/(2h)$. It works on any black box, but it is approximate, and a gradient over $n$ inputs costs about $2n$ evaluations with central differences. For a million parameters that is two million forward passes per training step. Its real job is testing gradients.
- **Automatic differentiation (AD)** applies the chain rule to the elementary operations the program actually executes. It is exact up to rounding, handles control flow, and in reverse mode delivers a full gradient for a small constant multiple of one function evaluation.

### Why the step size matters

Numerical derivatives have two errors that pull in opposite directions. **Truncation error** comes from $h$ not being zero; a Taylor expansion shows it is proportional to $h$ for the forward difference and to $h^2$ for the central difference, because the $h$ terms cancel. **Round-off error** comes from subtracting two nearly equal floating-point numbers. In double precision the relative rounding error is around $\epsilon \approx 2.2 \times 10^{-16}$ (L0-01), and dividing by $h$ amplifies the subtraction's error to roughly $\epsilon|f|/h$.

Balancing $h$ against $\epsilon/h$ puts the best forward-difference step near $\sqrt{\epsilon} \approx 10^{-8}$. Balancing $h^2$ against $\epsilon/h$ puts the best central-difference step near $\epsilon^{1/3} \approx 6 \times 10^{-6}$, with error around $10^{-11}$. These rules of thumb assume $f$ and its derivatives are of order 1. Script 1 measures this on $f(x) = \sin(x)e^x$: the errors bottom out at $h = 10^{-8}$ (forward) and $h = 10^{-5}$ (central), then grow again. At $h = 10^{-14}$ both are about $2 \times 10^{-2}$, worse than a central difference with $h = 0.1$.

## 5. Computational graphs and the evaluation trace

AD treats a computation as a graph: each node is an elementary operation with a known local derivative, and edges carry values between operations. The running example from [Baydin et al., 2018] is

$$y = f(x_1, x_2) = \ln(x_1) + x_1 x_2 - \sin(x_2), \quad \text{at } (x_1, x_2) = (2, 5).$$

As a sequence of elementary steps (an **evaluation trace**, which the survey notes is also called a Wengert list, after Wengert's 1964 paper):

| Step | Operation | Value |
|---|---|---|
| $v_1$ | $\ln x_1$ | $0.6931$ |
| $v_2$ | $x_1 \times x_2$ | $10.0000$ |
| $v_3$ | $\sin x_2$ | $-0.9589$ |
| $v_4$ | $v_1 + v_2$ | $10.6931$ |
| $v_5 = y$ | $v_4 - v_3$ | $11.6521$ |

Both inputs fan out: $x_1$ feeds $v_1$ and $v_2$, $x_2$ feeds $v_2$ and $v_3$. A computational graph is a directed acyclic graph, so its nodes always have a **topological order**, one in which each node comes after everything it depends on. The table is such an order. By hand, $\partial y/\partial x_1 = 1/x_1 + x_2 = 5.5$ and $\partial y/\partial x_2 = x_1 - \cos x_2 = 1.7163$.

## 6. Forward mode and reverse mode

Both modes apply the chain rule to the same trace; they accumulate in opposite directions.

### Forward mode: push tangents along with values

Attach to every intermediate $v_i$ a **tangent** $\dot v_i$, its derivative with respect to a chosen input direction. Seed $\dot x_1 = 1, \dot x_2 = 0$ and compute each tangent alongside its value:

| Step | Tangent rule | Tangent |
|---|---|---|
| $v_1 = \ln x_1$ | $\dot x_1 / x_1$ | $0.5$ |
| $v_2 = x_1 x_2$ | $\dot x_1 x_2 + x_1 \dot x_2$ | $5$ |
| $v_3 = \sin x_2$ | $\cos(x_2)\,\dot x_2$ | $0$ |
| $v_4 = v_1 + v_2$ | $\dot v_1 + \dot v_2$ | $5.5$ |
| $v_5 = v_4 - v_3$ | $\dot v_4 - \dot v_3$ | $5.5$ |

One pass gives $\partial y/\partial x_1$; $\partial y/\partial x_2$ needs a second pass seeded with $(0, 1)$. In general a seed $\mathbf{v}$ yields $J\mathbf{v}$, a **Jacobian-vector product** (JVP), and a full Jacobian takes $n$ passes, one per input.

A compact implementation uses **dual numbers** $a + b\varepsilon$ with the rule $\varepsilon^2 = 0$. Then $(a + b\varepsilon)(c + d\varepsilon) = ac + (ad + bc)\varepsilon$: the product rule falls out of the arithmetic. Overload the operators, run ordinary code, and the $\varepsilon$ part carries the derivative (script 2).

### Reverse mode: pull adjoints back from the output

Reverse mode evaluates the trace forward once, storing the values, then walks it backward computing each node's **adjoint** $\bar v_i = \partial y / \partial v_i$. Start from $\bar y = 1$. Each node sends its adjoint to its inputs multiplied by the local derivative, and a node that feeds several others sums what it receives:

| Step (reverse order) | Adjoint |
|---|---|
| $\bar v_5 = \bar y$ | $1$ |
| $\bar v_4 = \bar v_5 \cdot 1$, $\;\bar v_3 = \bar v_5 \cdot (-1)$ | $1$, $-1$ |
| $\bar v_1 = \bar v_4$, $\;\bar v_2 = \bar v_4$ | $1$, $1$ |
| $\bar x_2 = \bar v_2 \cdot x_1 + \bar v_3 \cos x_2$ | $2 - 0.2837 = 1.7163$ |
| $\bar x_1 = \bar v_1 / x_1 + \bar v_2 \cdot x_2$ | $0.5 + 5 = 5.5$ |

One backward pass produced both partial derivatives. A seed $\mathbf{u}$ on the output yields $\mathbf{u}^\top J$, a **vector-Jacobian product** (VJP). A full Jacobian takes $m$ passes, one per output, and the gradient of a scalar loss takes exactly one.

### Which is cheaper

Let $\mathrm{ops}(f)$ be the cost of evaluating $f$ once. [Baydin et al., 2018], citing Griewank and Walther's textbook on algorithmic differentiation, give the cost of the full $m \times n$ Jacobian as about $n \cdot c \cdot \mathrm{ops}(f)$ in forward mode and $m \cdot c \cdot \mathrm{ops}(f)$ in reverse mode, with $c$ guaranteed below 6 and typically between 2 and 3. So reverse mode wins when there are many inputs and few outputs (training: millions of parameters, one loss), and forward mode wins in the opposite case. The JAX documentation says the same thing in matrix terms: forward mode suits "tall" Jacobians, reverse mode "wide" ones [JAX Autodiff Cookbook].

In the language of section 3, a loss gradient is $\mathbf{1}^\top J_L \cdots J_2 J_1$. Reverse mode multiplies from the left, so every intermediate result is a row vector and every step is a cheap vector-matrix product. Forward mode multiplies from the right, one input direction at a time, and repeats the whole chain $n$ times.

Reverse mode pays in **memory**. The backward pass needs forward-pass values (the $x_1$ in $\bar v_2 \cdot x_1$, the $\mathbf{h}$ in $\mathrm{diag}(1 - \mathbf{h}^2)$), so they must be stored; the survey notes that storage can grow in proportion to the number of operations, and that checkpointing (recomputing some values instead of storing them) is one remedy. This is why training uses far more memory than inference (L4-02 does the accounting).

**History.** Backpropagation, popularized for neural networks by [Rumelhart et al., 1986], is reverse-mode AD applied to a loss. The ideas were found several times: [Baydin et al., 2018] credit forward mode to Wengert (1964) and note that Linnainmaa's 1970 work is often cited as the first published description of reverse mode, with related ideas earlier in control theory. L2-01 applies backprop to full networks.

## 7. Building a scalar reverse-mode engine

`engine.py` implements reverse mode in about 140 lines, following the design of Karpathy's micrograd [Karpathy, micrograd]:

1. **A `Value` node** holds a number (`data`), its adjoint (`grad`), the nodes it was computed from (`_parents`), and a closure `_backward` that applies the local derivative of the operation that created it.
2. **Operator overloading** records the graph as ordinary arithmetic runs. `c = a * b` creates a `Value` whose `_backward` adds `b.data * c.grad` to `a.grad` and `a.data * c.grad` to `b.grad`. Recording the graph while code runs is called **define-by-run**; PyTorch works this way, and its documentation states that the graph is recreated from scratch at every iteration, which is what lets ordinary Python control flow work [PyTorch, Autograd mechanics].
3. **`backward()`** sorts the graph topologically, sets the output's `grad` to 1, and calls each `_backward` in reverse order. That order guarantees a node has received gradient from all its consumers before it passes gradient on.

Three details carry most of the correctness:

- **Accumulate with `+=`.** A node used twice must receive gradient from both uses (the sum over paths). Assignment keeps only one.
- **Reset gradients.** Because of `+=`, a second backward call would add onto stale values, so this engine zeroes every node in the graph first. PyTorch makes the opposite choice: `Tensor.backward` accumulates into the leaves' `.grad`, and you zero them yourself, usually once per training step [PyTorch, Tensor.backward] (L2-02).
- **Save what the backward pass needs.** `tanh` and `exp` keep their forward result in the closure, which is the memory cost of reverse mode in miniature.

Eight operations are primitive (`+`, `*`, `**` by a constant, `exp`, `log`, `sin`, `tanh`, `relu`); subtraction, division and negation are composed from them and get correct gradients for free. Production frameworks are organized the same way, at the level of tensors instead of scalars.

## 8. Testing gradients against numerical derivatives

A gradient bug rarely crashes anything: the model trains a bit worse, and you blame the learning rate. So every derivative rule should be compared with a central difference at $h \approx 10^{-6}$, a practice called **gradient checking**, using a relative error with a floor $\tau$:

$$\text{err} = \frac{|g_{\text{auto}} - g_{\text{num}}|}{\max(\tau,\; |g_{\text{auto}}| + |g_{\text{num}}|)}.$$

A relative measure is needed because gradients span many orders of magnitude. The floor matters too, which I learned while writing `test_engine.py`. With $\tau = 10^{-8}$, one of the 200 random expressions failed. The failure was an input the expression never used: the true gradient was exactly 0, the engine returned 0, and the central difference returned round-off noise near $10^{-10}$, which a tiny floor turns into a large "relative error". With $\tau = 10^{-4}$, gradients that small are judged on absolute error and the false alarm disappears. When a check fails, first check the check.

I also tested the tests by breaking the engine on purpose, replacing `+=` with `=` in the backward rule of `+`. My original fork test, $x \cdot x + x$, **still passed**. There the `+` is the last operation, so its backward rule runs first, while `x.grad` is still 0, and overwriting 0 is harmless. In $(x + x) \cdot x$ the `+` runs after the multiplication has already deposited gradient in `x`, and the overwrite destroys it; that test fails with relative error 0.6 on the broken engine, and only 97 of the 200 random expressions pass. Both fork tests are in the final script. A test only counts once you have seen it fail on broken code.

The last line of script 4 shows an expected disagreement: `relu` at exactly 0. The function has a kink there and no derivative. The engine uses the convention "0 at the kink", while the central difference averages the two sides and gets 0.5. Gradient checks should use random points, which almost never land on a kink.

## Hands-on

Run the scripts from inside `code/L0-03-calculus-and-autodiff/` with the course environment. Each finishes in a few seconds.

### Script 1: finite differences and the step-size trade-off

```python
"""Numerical derivatives: the step size h trades truncation error against round-off error."""
import numpy as np


def f(x):
    return np.sin(x) * np.exp(x)


def df_exact(x):
    return (np.cos(x) + np.sin(x)) * np.exp(x)  # product rule


x0 = 1.0
exact = df_exact(x0)
print(f"exact f'({x0}) = {exact:.15f}\n")
print(f"{'h':>8} | {'forward error':>14} | {'central error':>14}")
print("-" * 42)
for k in range(1, 15):
    h = 10.0 ** (-k)
    e_fwd = abs((f(x0 + h) - f(x0)) / h - exact)
    e_cen = abs((f(x0 + h) - f(x0 - h)) / (2 * h) - exact)
    print(f"{h:8.0e} | {e_fwd:14.3e} | {e_cen:14.3e}")
```

```text
exact f'(1.0) = 3.756049227094727

       h |  forward error |  central error
------------------------------------------
   1e-01 |      1.437e-01 |      2.741e-03
   1e-02 |      1.466e-02 |      2.729e-05
   1e-03 |      1.468e-03 |      2.729e-07
   1e-04 |      1.469e-04 |      2.729e-09
   1e-05 |      1.469e-05 |      2.866e-11
   1e-06 |      1.469e-06 |      1.046e-10
   1e-07 |      1.511e-07 |      1.046e-10
   1e-08 |      2.210e-08 |      1.046e-10
   1e-09 |      4.664e-07 |      2.231e-08
   1e-10 |      4.218e-07 |      4.218e-07
   1e-11 |      1.290e-05 |      9.304e-06
   1e-12 |      5.014e-04 |      5.731e-05
   1e-13 |      3.495e-03 |      1.275e-03
   1e-14 |      1.871e-02 |      1.871e-02
```

The forward-difference error falls in proportion to $h$ and the central-difference error like $h^2$, until round-off takes over below $h = 10^{-8}$ and $h = 10^{-5}$ respectively, as section 4 predicted.

### Script 2: forward mode with dual numbers

```python
"""Forward-mode autodiff with dual numbers a + b*eps, where eps**2 = 0.

The 'tan' part carries the derivative along a chosen input direction.
"""
import math


class Dual:
    def __init__(self, val, tan=0.0):
        self.val, self.tan = val, tan  # value and tangent

    def __add__(self, o):
        o = o if isinstance(o, Dual) else Dual(o)
        return Dual(self.val + o.val, self.tan + o.tan)

    def __sub__(self, o):
        o = o if isinstance(o, Dual) else Dual(o)
        return Dual(self.val - o.val, self.tan - o.tan)

    def __mul__(self, o):
        o = o if isinstance(o, Dual) else Dual(o)
        # (a + b eps)(c + d eps) = ac + (ad + bc) eps: the product rule, for free
        return Dual(self.val * o.val, self.val * o.tan + self.tan * o.val)

    __radd__, __rmul__ = __add__, __mul__


def sin(x):
    return Dual(math.sin(x.val), math.cos(x.val) * x.tan)


def log(x):
    return Dual(math.log(x.val), x.tan / x.val)


def f(x1, x2):
    return log(x1) + x1 * x2 - sin(x2)  # the worked example from section 5


x1, x2 = 2.0, 5.0
for seed in [(1.0, 0.0), (0.0, 1.0), (3.0, -1.0)]:
    y = f(Dual(x1, seed[0]), Dual(x2, seed[1]))
    print(f"seed {seed}: value = {y.val:.6f}, tangent = {y.tan:.6f}")

grad = (1 / x1 + x2, x1 - math.cos(x2))
print(f"analytic gradient = ({grad[0]:.6f}, {grad[1]:.6f}); "
      f"gradient . (3, -1) = {3 * grad[0] - grad[1]:.6f}")
```

```text
seed (1.0, 0.0): value = 11.652071, tangent = 5.500000
seed (0.0, 1.0): value = 11.652071, tangent = 1.716338
seed (3.0, -1.0): value = 11.652071, tangent = 14.783662
analytic gradient = (5.500000, 1.716338); gradient . (3, -1) = 14.783662
```

Each pass gives one directional derivative. Two passes recover the hand-computed gradient, and the seed $(3, -1)$ gives $J\mathbf{v}$ directly without ever forming $J$.

### Script 3: the reverse-mode engine

```python
"""A tiny scalar reverse-mode autodiff engine (in the spirit of Karpathy's micrograd).

Each Value records the Values it was computed from and a closure that pushes
its gradient back to them using the local derivative (the chain rule).
"""
import math


class Value:
    def __init__(self, data, parents=(), op=""):
        self.data = float(data)
        self.grad = 0.0
        self._parents = parents
        self._op = op
        self._backward = lambda: None  # set by the operation that created this node

    def __repr__(self):
        return f"Value(data={self.data:.6g}, grad={self.grad:.6g})"

    @staticmethod
    def lift(x):
        return x if isinstance(x, Value) else Value(x)

    # ---- elementary operations: value plus local derivative ----
    def __add__(self, other):
        other = Value.lift(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            # d(a+b)/da = 1, d(a+b)/db = 1. Use += so that a node used twice
            # (a fork in the graph) accumulates gradient from both paths.
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = Value.lift(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, k):
        assert isinstance(k, (int, float)), "only constant exponents"
        out = Value(self.data ** k, (self,), f"**{k}")

        def _backward():
            self.grad += k * self.data ** (k - 1) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        e = math.exp(self.data)
        out = Value(e, (self,), "exp")

        def _backward():
            self.grad += e * out.grad  # reuse the forward result
        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), "log")

        def _backward():
            self.grad += out.grad / self.data
        out._backward = _backward
        return out

    def sin(self):
        out = Value(math.sin(self.data), (self,), "sin")

        def _backward():
            self.grad += math.cos(self.data) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t * t) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,), "relu")

        def _backward():
            # Convention: derivative 0 at exactly 0 (the kink).
            self.grad += (self.data > 0) * out.grad
        out._backward = _backward
        return out

    # ---- derived operations, built from the ones above ----
    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-Value.lift(other))

    def __truediv__(self, other):
        return self * Value.lift(other) ** -1

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return Value.lift(other) - self

    def __rmul__(self, other):
        return self * other

    def __rtruediv__(self, other):
        return Value.lift(other) / self

    # ---- reverse pass ----
    def backward(self):
        # Topological order: every node appears after all the nodes it depends on.
        order, seen = [], set()

        def visit(v):
            if id(v) not in seen:
                seen.add(id(v))
                for p in v._parents:
                    visit(p)
                order.append(v)
        visit(self)

        for v in order:
            v.grad = 0.0  # clear stale gradients from earlier calls
        self.grad = 1.0  # dL/dL = 1
        for v in reversed(order):
            v._backward()
        return order  # handy for counting nodes
```

This file only defines `Value`; script 4 exercises it.

### Script 4: testing the engine

```python
"""Check the reverse-mode engine against central-difference numerical gradients."""
import random

from engine import Value


def numerical_grad(fn, xs, h=1e-6):
    """Central differences on plain floats: 2 evaluations of fn per input."""
    f = lambda *a: fn(*[Value(x) for x in a]).data
    return [(f(*xs[:i], x + h, *xs[i + 1:]) - f(*xs[:i], x - h, *xs[i + 1:])) / (2 * h)
            for i, x in enumerate(xs)]


def engine_grad(fn, xs):
    vs = [Value(x) for x in xs]
    nodes = fn(*vs).backward()
    return [v.grad for v in vs], len(nodes)


def rel_err(a, b, floor=1e-4):
    # The floor matters: when the true gradient is exactly 0, central differences
    # return round-off noise near 1e-10, which a pure ratio would call a 100% error.
    return abs(a - b) / max(floor, abs(a) + abs(b))


def worst_error(fn, xs):
    g_auto, n_nodes = engine_grad(fn, xs)
    return max(rel_err(a, b) for a, b in zip(g_auto, numerical_grad(fn, xs))), n_nodes


def check(name, fn, xs, tol=1e-6):
    err, n_nodes = worst_error(fn, xs)
    print(f"[{'PASS' if err < tol else 'FAIL'}] {name:<26} nodes={n_nodes:<3} worst rel err={err:.1e}")
    return err < tol


def tiny_net(*p):
    # 2 inputs -> 3 tanh units -> 1 output, squared error against a target.
    x, target = (0.5, -1.2), 0.3
    hidden = [(p[2 * j] * x[0] + p[2 * j + 1] * x[1]).tanh() for j in range(3)]
    return (sum(p[6 + j] * hidden[j] for j in range(3)) - target) ** 2


UNARY = [lambda v: v.tanh(), lambda v: v.sin(), lambda v: (v * 0.3).exp(), lambda v: v ** 2]
BINARY = [lambda a, b: a + b, lambda a, b: a * b, lambda a, b: a - b]


def random_expression(seed, n_inputs=3, n_ops=12):
    """Draw a random 'program' once, so every call builds the same graph."""
    rng = random.Random(seed)
    plan = []
    for k in range(n_ops):
        size = n_inputs + k  # values available at step k
        if rng.random() < 0.5:
            plan.append((UNARY[rng.randrange(4)], rng.randrange(size), None))
        else:
            plan.append((BINARY[rng.randrange(3)], rng.randrange(size), rng.randrange(size)))

    def fn(*inputs):
        pool = list(inputs)
        for op, i, j in plan:
            pool.append(op(pool[i]) if j is None else op(pool[i], pool[j]))
        return pool[-1] + pool[-2]
    return fn


g, _ = engine_grad(lambda a, b: a.log() + a * b - b.sin(), [2.0, 5.0])
print(f"worked example: dy/dx1={g[0]:.6f}, dy/dx2={g[1]:.6f} (hand: 5.500000, 1.716338)")

random.seed(0)
results = [
    check("ln(a) + a*b - sin(b)", lambda a, b: a.log() + a * b - b.sin(), [2.0, 5.0]),
    # Forks: x is used more than once, so contributions from all paths must add.
    check("x*x + x (fork)", lambda x: x * x + x, [3.0]),
    check("(x + x) * x (fork)", lambda x: (x + x) * x, [3.0]),
    check("(1 + exp(-a)) ** -1 / b", lambda a, b: (1 + (-a).exp()) ** -1 / b, [0.7, 2.5]),
    check("tiny tanh net, 9 params", tiny_net, [random.uniform(-1, 1) for _ in range(9)]),
]

passed = sum(worst_error(random_expression(s), [random.uniform(-1.5, 1.5) for _ in range(3)])[0] < 1e-5
             for s in range(200))
print(f"[{'PASS' if passed == 200 else 'FAIL'}] random expressions          {passed}/200 within 1e-5")
results.append(passed == 200)

# A known disagreement: relu has a kink at 0, so it has no derivative there.
g, _ = engine_grad(lambda x: x.relu(), [0.0])
print(f"[NOTE] relu at 0: engine grad={g[0]}, numerical grad={numerical_grad(lambda x: x.relu(), [0.0])[0]:.3f}")
print(f"\n{sum(results)}/{len(results)} checks passed")
```

```text
worked example: dy/dx1=5.500000, dy/dx2=1.716338 (hand: 5.500000, 1.716338)
[PASS] ln(a) + a*b - sin(b)       nodes=9   worst rel err=1.3e-10
[PASS] x*x + x (fork)             nodes=3   worst rel err=1.3e-10
[PASS] (x + x) * x (fork)         nodes=3   worst rel err=7.0e-11
[PASS] (1 + exp(-a)) ** -1 / b    nodes=10  worst rel err=5.4e-11
[PASS] tiny tanh net, 9 params    nodes=39  worst rel err=1.1e-09
[PASS] random expressions          200/200 within 1e-5
[NOTE] relu at 0: engine grad=0.0, numerical grad=0.500

6/6 checks passed
```

Relative errors of $10^{-9}$ to $10^{-11}$ are what central differences at $h = 10^{-6}$ can resolve, so the remaining discrepancy belongs to the numerical side. The tiny network built 39 nodes, and one backward pass produced all 9 parameter gradients; the numerical check needed 18 extra forward passes for the same information.

### Script 5: forward vs reverse mode for a Jacobian

```python
"""Forward vs reverse mode on f(x) = W2 @ tanh(W1 @ x), f: R^n -> R^m.

A JVP pushes a tangent v forward and returns J @ v; a VJP pulls a cotangent u
backward and returns u @ J. The full m x n Jacobian takes n JVPs or m VJPs.
"""
import time

import numpy as np

HIDDEN = 300


def jvp(W1, W2, x, v):
    a = W1 @ x
    return W2 @ ((1 - np.tanh(a) ** 2) * (W1 @ v))  # tangents flow with the values


def vjp(W1, W2, h, u):
    return W1.T @ ((1 - h ** 2) * (W2.T @ u))  # uses h = tanh(W1 x) saved from the forward pass


def jacobian_forward(W1, W2, x):
    return np.stack([jvp(W1, W2, x, e) for e in np.eye(x.size)], axis=1)  # one pass per input


def jacobian_reverse(W1, W2, x):
    h = np.tanh(W1 @ x)  # forward pass, stored for the backward passes
    return np.stack([vjp(W1, W2, h, e) for e in np.eye(W2.shape[0])])  # one pass per output


def best_time(fn, *args, repeats=5):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn(*args)
        times.append(time.perf_counter() - t0)
    return out, 1e3 * min(times)


print(f"{'n':>5} {'m':>5} | {'fwd passes':>10} {'fwd ms':>7} | {'rev passes':>10} {'rev ms':>7} | max diff")
for n, m in [(1000, 1), (1000, 10), (100, 100), (10, 1000), (1, 1000)]:
    rng = np.random.default_rng(0)
    W1 = rng.normal(size=(HIDDEN, n)) / np.sqrt(n)
    W2 = rng.normal(size=(m, HIDDEN)) / np.sqrt(HIDDEN)
    x = rng.normal(size=n)
    J_f, t_f = best_time(jacobian_forward, W1, W2, x)
    J_r, t_r = best_time(jacobian_reverse, W1, W2, x)
    print(f"{n:>5} {m:>5} | {n:>10} {t_f:>7.2f} | {m:>10} {t_r:>7.2f} | {np.abs(J_f - J_r).max():.1e}")
```

Output from one run on a 13th-generation Intel Core i7 laptop (timings will differ on your machine; the pass counts and the pattern will not):

```text
    n     m | fwd passes  fwd ms | rev passes  rev ms | max diff
 1000     1 |       1000   86.59 |          1    0.08 | 5.6e-17
 1000    10 |       1000   87.80 |         10    0.50 | 5.6e-17
  100   100 |        100    1.16 |        100    0.76 | 1.9e-16
   10  1000 |         10    0.37 |       1000   41.08 | 8.9e-16
    1  1000 |          1    0.04 |       1000   39.27 | 1.8e-15
```

With 1000 inputs and one output, reverse mode needs one pass instead of a thousand and is hundreds of times faster (about a thousand in this run). With one input and 1000 outputs the situation reverses. For a square Jacobian the two are within a small factor. Both produce the same Jacobian to about $10^{-15}$, since they compute the same derivatives in a different order.

## Common mistakes

- **Overwriting instead of accumulating gradients.** `self.grad = ...` in a backward rule gives correct answers on graphs without shared nodes, and sometimes even with them, depending on execution order (section 8). Test several forks and random expressions, and confirm the tests fail when you break the engine deliberately.
- **Forgetting to zero gradients.** In PyTorch, gradients accumulate across `backward()` calls by design [PyTorch, Tensor.backward]. A training loop without a reset adds every earlier step's gradient into the current one, and training behaves erratically.
- **A gradient check with a bad step size or precision.** $h = 10^{-12}$ "for precision" makes the check worse (script 1). In float32, $\epsilon \approx 1.2 \times 10^{-7}$, so at $h = 10^{-6}$ the round-off term $\epsilon/h$ alone is of order 0.1. Check in float64.
- **Comparing with absolute error only, or relative error with no floor.** The first misses errors in small gradients; the second flags correct zero gradients. Look at the actual numbers when a check fails.
- **Checking at a kink** of `relu`, `max`, `abs` or a sort, where no derivative exists. Use random inputs.
- **Treating a passing gradient check as proof the model is right.** It proves the derivative matches the function you wrote. A loss with the wrong sign or a mask in the wrong place passes happily.
- **Deep recursion.** `engine.py` sorts with recursive depth-first search, which hits Python's default recursion limit of 1000 on long chains of operations. Fine here; exercise 9 fixes it.

## Exercises

1. [warm-up] Differentiate $f(x) = e^{-x^2}$ with the chain rule and evaluate at $x = 1$. Check with a central difference.
2. [warm-up] For $f(x_1, x_2) = x_1 \sin(x_2) + x_2^2$, find $\nabla f(1, 0)$ and the directional derivative in direction $(1, 1)/\sqrt{2}$.
3. [warm-up] Write the Jacobian of $\mathbf{f}(x_1, x_2) = (x_1 x_2,\; x_1 + x_2,\; x_2^2)$ and state its shape.
4. [core] Draw the graph of $y = (x_1 + x_2)\tanh(x_1)$ and run reverse mode by hand at $(0.5, 1)$. Confirm with `engine.py`.
5. [core] Break the engine: change `+=` to `=` in the backward rule of `__add__`. Which checks in `test_engine.py` fail, which pass, and why?
6. [core] Add `sigmoid` to `engine.py` as a primitive with backward rule $\sigma'(x) = \sigma(x)(1 - \sigma(x))$, add a test, and compare with sigmoid built from `exp`, `+` and `**-1`.
7. [core] Add division and `exp` to `dual_numbers.py` and differentiate $x / (1 + e^{-x})$ at $x = 2$. Check against the engine.
8. [core] A model has $10^6$ parameters and one forward pass takes 10 ms. Estimate the time for one gradient by central differences and by reverse mode, using the cost model of section 6.
9. [stretch] Make `Value.backward` iterative so it handles a chain of 100,000 additions. How does memory grow with the chain length, and why?
10. [stretch] For $L(\mathbf{x}) = \tfrac12\|W_2\tanh(W_1\mathbf{x})\|^2$, compute a Hessian-vector product $H\mathbf{v}$ by central-differencing the reverse-mode gradient, and verify it on a small problem against a full finite-difference Hessian.
11. [stretch] Use the engine to fit $y = 2x - 1$ from 20 noisy points with plain gradient descent (200 steps, step size 0.1), then fit $y = \sin x$ on $[-3, 3]$ with a hidden layer of 4 tanh units. Leave optimizer tuning to L0-05.

### Hints and answers

1. $f'(x) = -2xe^{-x^2}$, so $f'(1) = -2/e \approx -0.7358$.
2. $\nabla f = (\sin x_2,\; x_1\cos x_2 + 2x_2) = (0, 1)$ at $(1, 0)$; the directional derivative is $1/\sqrt2 \approx 0.7071$.
3. $\begin{pmatrix} x_2 & x_1 \\ 1 & 1 \\ 0 & 2x_2 \end{pmatrix}$, shape $3 \times 2$ (outputs by inputs).
4. $\partial y/\partial x_1 = \tanh x_1 + (x_1 + x_2)(1 - \tanh^2 x_1) \approx 0.4621 + 1.5 \times 0.7864 \approx 1.6418$ and $\partial y/\partial x_2 = \tanh x_1 \approx 0.4621$.
5. When I ran it, the $(x + x) \cdot x$ fork and the random expressions failed (97 of 200 passed); the rest passed, because in each of them the broken `+` either feeds a node used once or runs before anything else has written to that node. Section 8 explains the ordering.
6. They agree to round-off. The primitive builds one node instead of four. Watch for overflow of `math.exp(-x)` at very negative `x`.
7. $(a + b\varepsilon)/(c + d\varepsilon) = a/c + \frac{bc - ad}{c^2}\varepsilon$ and $e^{a + b\varepsilon} = e^a + e^a b\,\varepsilon$. The derivative at 2 is about $1.0908$.
8. Central differences: $2 \times 10^6$ passes, about 20,000 s (5.6 hours). Reverse mode: one forward and one backward pass, a few multiples of 10 ms.
9. Use an explicit stack, or Kahn's algorithm on in-degrees. Memory grows linearly, because every node and its closure stay alive until the backward pass finishes.
10. $H\mathbf{v} \approx (\nabla L(\mathbf{x} + h\mathbf{v}) - \nabla L(\mathbf{x} - h\mathbf{v}))/(2h)$, where $\nabla L$ is `vjp(W1, W2, h, y)` with `y` the output. It costs two gradients regardless of dimension. JAX computes it exactly by applying forward mode to a reverse-mode gradient, which its documentation calls typically the most efficient composition [JAX Autodiff Cookbook].
11. The linear fit should reach a loss near the noise variance; the tanh model should beat any straight line on $\sin x$. Exact numbers depend on your seed.

## Check your understanding

**1. Why not train networks with numerical gradients?** A central-difference gradient needs two evaluations per parameter, two million forward passes for a million parameters, and it is only approximate. Reverse mode gives the exact gradient for a small constant multiple of one forward pass. Numerical gradients remain the standard way to test a gradient implementation on small cases.

**2. What does one pass of each mode compute?** A forward pass seeded with $\mathbf{v}$ computes $J\mathbf{v}$, one directional derivative. A reverse pass seeded with $\mathbf{u}$ computes $\mathbf{u}^\top J$; for a scalar output with $u = 1$ that is the full gradient. A full Jacobian costs $n$ forward passes or $m$ reverse passes.

**3. Why must a node used twice have its gradients summed?** The multivariable chain rule makes the total derivative the sum of contributions along every path to the output. Nudging a shared variable changes the output through every operation that uses it, and the effects add. Assignment keeps only one path's contribution.

**4. What does reverse mode cost besides arithmetic?** Memory. The backward pass needs intermediate values from the forward pass, so they are stored until used, and storage can grow with the number of operations. That is why training needs much more memory than inference, and why checkpointing trades recomputation for memory.

**5. A gradient check shows relative error 0.4 on one parameter and $10^{-10}$ elsewhere. What do you check first?** The check itself. If both numbers are tiny (say $10^{-11}$ and 0), it is round-off and the floor is too small. If the input sits on a kink, no derivative exists there. Only if both gradients are sizable and disagree is a backward rule likely wrong, probably in an operation only that parameter passes through.

## Further reading

- **Atilim Gunes Baydin, Barak A. Pearlmutter, Alexey Andreyevich Radul, Jeffrey Mark Siskind, "Automatic Differentiation in Machine Learning: a Survey"** (2018, Journal of Machine Learning Research 18(153); arXiv:1502.05767). <https://arxiv.org/abs/1502.05767>
  *Read for:* forward and reverse mode, the evaluation-trace notation used here, cost results and history, all in one free survey. *Level:* intermediate.
- **Andrej Karpathy, "micrograd"** (GitHub repository, MIT license). <https://github.com/karpathy/micrograd>
  *Read for:* a compact scalar autograd engine with a small neural network library, tested against PyTorch; compare its design with `engine.py`. *Level:* intro.
- **Andrej Karpathy, "The spelled-out intro to neural networks and backpropagation: building micrograd"** (2022, lecture 1 of the Neural Networks: Zero to Hero course). <https://karpathy.ai/zero-to-hero.html>
  *Read for:* about two and a half hours building such an engine live, assuming basic Python and high-school calculus. *Level:* intro.
- **Grant Sanderson (3Blue1Brown), "Visualizing the chain rule and product rule"** (Essence of Calculus, chapter 5). <https://3blue1brown.com/lessons/chain-rule-and-product-rule>
  *Read for:* a visual intuition for why rates multiply. *Level:* intro.
- **Marc Peter Deisenroth, A. Aldo Faisal, Cheng Soon Ong, "Mathematics for Machine Learning"** (2020, Cambridge University Press; free PDF from the authors). <https://mml-book.github.io/>
  *Read for:* chapter 5, Vector Calculus, including gradients of vector- and matrix-valued functions and section 5.6 on backpropagation and automatic differentiation. *Level:* intro to intermediate.
- **Stanford CS231n, "Backpropagation, Intuitions"** (course notes). <https://cs231n.github.io/optimization-2/>
  *Read for:* backprop as local gates in a circuit, gradient accumulation at forks, and vectorized gradients. *Level:* intro.
- **Ian Goodfellow, Yoshua Bengio, Aaron Courville, "Deep Learning"** (2016, MIT Press; free HTML version). <https://www.deeplearningbook.org/>
  *Read for:* section 6.5, Back-Propagation and Other Differentiation Algorithms. *Level:* intermediate.
- **JAX authors, "The Autodiff Cookbook"** (JAX documentation). <https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html>
  *Read for:* JVPs, VJPs, `jacfwd` versus `jacrev`, and Hessian-vector products in a system that exposes both modes. *Level:* intermediate.
- **PyTorch, "Autograd mechanics"** and **"torch.Tensor.backward"** (PyTorch documentation). <https://docs.pytorch.org/docs/stable/notes/autograd.html> and <https://docs.pytorch.org/docs/stable/generated/torch.Tensor.backward.html>
  *Read for:* how a production define-by-run engine records graphs and saves tensors, and the contract that gradients accumulate in leaves. *Level:* intermediate.
- **David E. Rumelhart, Geoffrey E. Hinton, Ronald J. Williams, "Learning representations by back-propagating errors"** (1986, Nature 323, 533 to 536; paywalled at the publisher). <https://doi.org/10.1038/323533a0>
  *Read for:* the short paper that popularized backpropagation for networks with hidden units. *Level:* intermediate.
