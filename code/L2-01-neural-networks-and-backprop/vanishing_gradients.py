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
