"""Debugging a network that will not train: the initial-loss check, overfitting one
batch, and the symptoms of five planted bugs."""
import math
import warnings

import torch
from torch import nn

from common import digits, evaluate, fit, mlp

data = digits()
X, y = data["train"]
xb, yb = X[:32], y[:32]                 # the single batch we try to memorize


def overfit_one_batch(bug=None, steps=200):
    torch.manual_seed(0)
    model = mlp([64, 128, 10])
    opt = torch.optim.SGD(model.parameters(), lr=3.0 if bug == "lr = 3.0" else 0.1, momentum=0.9)
    targets = yb[torch.randperm(32)] if bug == "labels shuffled" else yb
    curve = []
    for step in range(steps):
        if bug != "no zero_grad":
            opt.zero_grad()
        logits = model(xb)
        if bug == "softmax before CrossEntropyLoss":
            logits = torch.softmax(logits, dim=1)    # the loss applies log-softmax itself
        loss = nn.functional.cross_entropy(logits, targets)
        loss.backward()
        opt.step()
        curve.append(loss.item())
    return curve


# 1. At initialization, a 10-class classifier should be unsure: loss about ln(10).
torch.manual_seed(0)
model = mlp([64, 128, 10])
print(f"initial loss {evaluate(model, X, y)[0]:.3f}, ln(10) = {math.log(10):.3f}")

# 2. A correct pipeline drives the loss on one batch to ~0. Bugs show up as distinct curves.
print("\noverfitting one batch of 32, SGD(lr=0.1, momentum=0.9); loss at steps 0, 10, 50, 100, 199:")
for bug in [None, "softmax before CrossEntropyLoss", "no zero_grad", "lr = 3.0", "labels shuffled"]:
    c = overfit_one_batch(bug)
    print(f"  {str(bug or 'correct'):<32} " + "  ".join(f"{c[i]:9.3g}" for i in [0, 10, 50, 100, 199]))
print(f"  floor for double softmax: ln(1 + 9/e) = {math.log(1 + 9 / math.e):.4f}")

# 3. The shuffled-label bug passed the single-batch test (a big net memorizes anything),
#    so check it on held-out data, where it is obvious.
torch.manual_seed(0)
perm = torch.randperm(len(y))
for name, labels in [("correct labels", y), ("labels shuffled", y[perm])]:
    torch.manual_seed(0)
    model = mlp([64, 128, 10])
    hist = fit(model, {**data, "train": (X, labels)}, epochs=20, opt=torch.optim.Adam(model.parameters(), lr=1e-3))
    print(f"full training, {name:<16} train loss {hist[-1][0]:.3f}  val acc {hist[-1][2]:.3f}")

# 4. A silent broadcasting bug in regression: predictions (n, 1) vs targets (n,).
target = torch.tensor([1.0, 2.0, 3.0, 4.0])
pred = (target + 0.1).unsqueeze(1)          # nearly perfect predictions, shape (4, 1)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    wrong = nn.functional.mse_loss(pred, target)
print(f"\nMSE with shapes (4, 1) vs (4,): {wrong.item():.3f}; correct: "
      f"{nn.functional.mse_loss(pred.squeeze(1), target).item():.3f}")
print("  warning:", str(caught[0].message)[:95], "...")

# 5. Find where a NaN gradient comes from: sqrt has an infinite derivative at 0,
#    and the next backward step multiplies that inf by 0.
w = torch.tensor([1.0, 1.0], requires_grad=True)
dist = torch.sqrt(((w - torch.tensor([1.0, 1.0])) ** 2).sum())
dist.backward()
print(f"\ngradient of ||w - w0|| at w = w0: {w.grad.tolist()}")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")      # anomaly mode also prints a long traceback as a warning
    with torch.autograd.detect_anomaly():
        try:
            torch.sqrt(((w - torch.tensor([1.0, 1.0])) ** 2).sum()).backward()
        except RuntimeError as err:
            print("detect_anomaly ->", str(err)[:70])
