"""Batch norm and layer norm: rescuing a deep net that will not train,
and the train/eval-mode trap that comes with batch norm."""
import copy

import torch
from torch import nn

from common import digits, evaluate, fit, mlp

data = digits()

# 1. The 20-layer ReLU net with PyTorch's default init, which did not train in initialization.py.
print("20 hidden layers of width 128, default init, SGD(momentum=0.9), 20 epochs")
models = {}
for name, norm in [("no normalization", None), ("BatchNorm1d", nn.BatchNorm1d), ("LayerNorm", nn.LayerNorm)]:
    for lr in [0.01, 0.001]:
        torch.manual_seed(0)
        model = mlp([64] + [128] * 20 + [10], norm=norm)
        hist = fit(model, data, epochs=20, opt=torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9))
        models[name, lr] = model
        print(f"  {name:<17} lr {lr:<6} train loss epoch 1 {hist[0][0]:.3f} -> epoch 20 {hist[-1][0]:.3f}"
              f"   val acc {hist[-1][2]:.3f}")

# 2. Batch norm behaves differently in train and eval mode.
bn_model = models["BatchNorm1d", 0.01]
bn = bn_model[1]
print(f"\nfirst BatchNorm1d layer: running_mean[:3] {[round(v, 3) for v in bn.running_mean[:3].tolist()]}, "
      f"momentum {bn.momentum}")
X_te, y_te = data["test"]


def accuracy_in_train_mode(model, X, y, batch_size, sort_by_label):
    """Predict in train mode (the bug): each batch is normalized with its own statistics."""
    model = copy.deepcopy(model)         # train-mode forward passes would also update running stats
    model.train()
    order = torch.argsort(y, stable=True) if sort_by_label else torch.randperm(len(y), generator=torch.Generator().manual_seed(0))
    correct = 0
    with torch.no_grad():
        for i in range(0, len(y), batch_size):
            idx = order[i:i + batch_size]
            correct += (model(X[idx]).argmax(1) == y[idx]).sum().item()
    return correct / len(y)


print(f"test acc, eval mode (running statistics):          {evaluate(bn_model, X_te, y_te)[1]:.3f}")
for bs in [256, 16]:
    print(f"test acc, train mode, random batches of {bs:<4}:      {accuracy_in_train_mode(bn_model, X_te, y_te, bs, False):.3f}")
print(f"test acc, train mode, batches of 16 sorted by label: {accuracy_in_train_mode(bn_model, X_te, y_te, 16, True):.3f}")
try:
    bn_model.train()
    bn_model(X_te[:1])
except ValueError as err:
    print("train mode, batch of 1 ->", err)

ln_model = models["LayerNorm", 0.001]
ln_model.eval()
with torch.no_grad():
    alone = ln_model(X_te[:1])
    in_batch = ln_model(X_te[:64])[:1]
print(f"LayerNorm: one example alone vs inside a batch, max output diff {(alone - in_batch).abs().max().item():.1e}")
