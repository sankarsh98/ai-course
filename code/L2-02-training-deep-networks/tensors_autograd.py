"""PyTorch tensors and autograd, checked against the hand-written NumPy MLP from L2-01."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "L2-01-neural-networks-and-backprop"))
import mlp_numpy  # noqa: E402

print("torch", torch.__version__)

# 1. Tensors: NumPy-like, float32 by default, and from_numpy shares memory.
a = np.arange(6.0).reshape(2, 3)
t = torch.from_numpy(a)                 # float64, because a is float64; no copy
t[0, 0] = 100.0
print("a[0, 0] after editing the tensor:", a[0, 0], "| default dtype:", torch.ones(1).dtype)
print("broadcast (2,3) + (3,) ->", tuple((t + torch.ones(3, dtype=t.dtype)).shape))

# 2. Autograd: leaves with requires_grad record a graph; backward() fills .grad.
w = torch.tensor([1.0, -2.0, 3.0], requires_grad=True)
loss = (w ** 2).sum()                   # d/dw = 2w
loss.backward()
print("\ngrad after one backward:", w.grad.tolist())
(w ** 2).sum().backward()               # a second backward ADDS to .grad
print("grad after a second backward (accumulated):", w.grad.tolist())
w.grad = None                           # what optimizer.zero_grad() does by default
with torch.no_grad():                   # no graph: for evaluation and manual updates
    print("inside no_grad, result requires grad?", (w * 2).requires_grad)
print("detach() cuts the graph:", w.detach().requires_grad)

loss = (w ** 3).sum()
loss.backward()
try:
    loss.backward()                     # the graph's saved tensors were freed
except RuntimeError as err:
    print("second backward on the same graph ->", str(err).split(".")[0])

# 3. Same weights, same batch: torch autograd vs the L2-01 manual backward pass.
rng = np.random.default_rng(0)
params = mlp_numpy.init_params([64, 32, 32, 10], rng)
X, y = rng.normal(size=(16, 64)), rng.integers(0, 10, size=16)
loss_np, grads_np = mlp_numpy.loss_and_grads(params, X, y, "tanh")

tparams = [(torch.tensor(W, requires_grad=True), torch.tensor(b, requires_grad=True)) for W, b in params]
h = torch.tensor(X)
for l, (W, b) in enumerate(tparams):
    h = h @ W + b
    if l < len(tparams) - 1:
        h = torch.tanh(h)
loss_t = torch.nn.functional.cross_entropy(h, torch.tensor(y))
loss_t.backward()
diff = max(np.abs(W.grad.numpy() - dW).max() for (W, _), (dW, _) in zip(tparams, grads_np))
print(f"\nloss numpy {loss_np:.12f} vs torch {loss_t.item():.12f}; max |grad diff| {diff:.1e}")
