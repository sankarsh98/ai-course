"""Attribution maps on a small CNN for 8x8 digits, and two checks they can fail.

Methods: gradient (saliency), gradient x input, integrated gradients (IG),
guided backpropagation. Checks:
  * completeness: IG attributions should sum to f(x) - f(baseline);
  * model randomization (Adebayo et al., 2018): re-initialize the layers from
    the top down; a map that barely changes cannot be explaining the learned model;
  * deletion: zero the most-attributed pixels and watch the logit fall.
"""
import copy

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

torch.manual_seed(0)
X, y = load_digits(return_X_y=True)
X = torch.tensor(X / 16.0, dtype=torch.float32).view(-1, 1, 8, 8)
y = torch.tensor(y)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)


class GuidedReLU(torch.autograd.Function):
    """ReLU whose backward pass also zeroes negative incoming gradients."""
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.clamp_min(0)

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return g * (x > 0) * (g > 0)


class Act(nn.Module):
    guided = False          # class-wide switch for guided backprop

    def forward(self, x):
        return GuidedReLU.apply(x) if Act.guided else torch.relu(x)


def make_cnn():
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1), Act(),
        nn.Conv2d(16, 32, 3, padding=1), Act(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(32 * 16, 64), Act(), nn.Linear(64, 10))


model = make_cnn()
opt = torch.optim.Adam(model.parameters(), lr=2e-3)
for epoch in range(15):
    for idx in torch.randperm(len(Xtr)).split(64):
        loss = nn.functional.cross_entropy(model(Xtr[idx]), ytr[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
with torch.no_grad():
    print(f"CNN test accuracy: {(model(Xte).argmax(1) == yte).float().mean():.3f}")


def grad_of_logit(net, x, cls):
    x = x.clone().requires_grad_()
    net(x).gather(1, cls[:, None]).sum().backward()
    return x.grad.detach()


def attributions(net, x, cls, steps=64):
    out = {"gradient": grad_of_logit(net, x, cls)}
    out["grad x input"] = out["gradient"] * x
    # IG with a black-image baseline: average gradient along the straight path.
    total = torch.zeros_like(x)
    for a in (torch.arange(steps) + 0.5) / steps:
        total += grad_of_logit(net, a * x, cls)
    out["integrated grad"] = total / steps * x
    Act.guided = True
    out["guided backprop"] = grad_of_logit(net, x, cls)
    Act.guided = False
    return out


x, cls = Xte[:200], yte[:200]
maps = attributions(model, x, cls)

# ---- completeness of IG --------------------------------------------------------
with torch.no_grad():
    f_x = model(x).gather(1, cls[:, None]).squeeze(1)
    f_0 = model(torch.zeros_like(x)).gather(1, cls[:, None]).squeeze(1)
gap = (maps["integrated grad"].flatten(1).sum(1) - (f_x - f_0)).abs() / (f_x - f_0).abs()
print(f"IG completeness: median relative error {gap.median():.4f} with 64 steps")


def similarity(a, b, on_only=False):
    """Mean Spearman rank correlation of |attribution| maps, per image.
    on_only: use only pixels where the input is nonzero (about half of them)."""
    rs = []
    for i in range(len(a)):
        keep = (x[i].flatten() > 0) if on_only else torch.ones(64, dtype=torch.bool)
        rs.append(spearmanr(a[i].abs().flatten()[keep], b[i].abs().flatten()[keep])[0])
    return np.mean(rs)


# ---- cascading model randomization -------------------------------------------
layers = [i for i, m in enumerate(model) if isinstance(m, (nn.Conv2d, nn.Linear))]
print("\nRank correlation with the trained model's maps after re-initializing layers")
print("from the top down (1.0 = map unchanged; low = map depends on learned weights)")
print(f"{'layers randomized':18s}" + "".join(f"{k:>17s}" for k in maps))
rand_model = copy.deepcopy(model)
torch.manual_seed(1)
for depth, li in enumerate(reversed(layers), start=1):
    rand_model[li].reset_parameters()
    rmaps = attributions(rand_model, x, cls)
    print(f"top {depth} of {len(layers):<10d}" +
          "".join(f"{similarity(maps[k], rmaps[k]):17.3f}" for k in maps))
print(f"{'all 4, on pixels':18s}" +
      "".join(f"{similarity(maps[k], rmaps[k], on_only=True):17.3f}" for k in maps))

# What does the input alone look like as an "explanation"?
edges = x.abs()
print(f"\nrank correlation of each trained map with the raw input |x|:")
print("".join(f"{k:>17s}" for k in maps))
print("".join(f"{similarity(maps[k], edges):17.3f}" for k in maps))

# ---- deletion test ---------------------------------------------------------------
# Zero the k most-attributed pixels (by signed attribution) and measure the
# drop in the class logit; compare with deleting k random "on" pixels.
print("\nmean drop in class logit after zeroing the top-8 pixels")
with torch.no_grad():
    for k_name, m in list(maps.items()) + [("random", torch.rand_like(x) * (x > 0))]:
        top = m.flatten(1).topk(8, dim=1).indices
        xd = x.flatten(1).clone().scatter_(1, top, 0).view_as(x)
        drop = f_x - model(xd).gather(1, cls[:, None]).squeeze(1)
        print(f"  {k_name:16s} {drop.mean():6.2f}")
