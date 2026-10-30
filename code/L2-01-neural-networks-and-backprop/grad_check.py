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
