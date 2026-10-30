"""Why the weight scale at initialization matters: signal size through 50 layers,
and what that does to training a 20-layer MLP."""
import math

import torch
from torch import nn

from common import digits, fit, mlp


def init_(model, scheme):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            if scheme == "N(0, 0.01^2)":
                nn.init.normal_(m.weight, std=0.01)
            elif scheme == "Xavier":
                nn.init.xavier_normal_(m.weight)            # var = 2 / (fan_in + fan_out)
            elif scheme == "He":
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")  # var = 2 / fan_in
            # "PyTorch default": leave nn.Linear's own reset_parameters() init alone
            if scheme != "PyTorch default":
                nn.init.zeros_(m.bias)
    return model


SCHEMES = ["N(0, 0.01^2)", "PyTorch default", "Xavier", "He"]

# 1. Push a random batch through 50 layers of width 256 and watch the activation scale.
torch.manual_seed(0)
x = torch.randn(512, 256)
print("per-unit std across the batch after layer 1 / 10 / 50 (input std 1.0)")
for act in [nn.Tanh, nn.ReLU]:
    for scheme in SCHEMES:
        net = init_(mlp([256] * 52, act=act), scheme)
        h, stds = x, []
        with torch.no_grad():
            for layer in list(net)[:-1]:
                h = layer(h)
                if isinstance(layer, act):
                    stds.append(h.std(dim=0).mean().item())  # spread ACROSS inputs
        print(f"  {act.__name__:<5} {scheme:<16} " + "  ".join(f"{stds[i]:.1e}" for i in [0, 9, 49]))

w = nn.Linear(256, 256).weight
print(f"\nnn.Linear default weight std {w.std().item():.4f}; 1/sqrt(3*256) = {1 / math.sqrt(3 * 256):.4f}; "
      f"He std sqrt(2/256) = {math.sqrt(2 / 256):.4f}")

# 2. Train a 20-hidden-layer ReLU MLP with SGD + momentum under each scheme.
data = digits()
print("\n20 hidden layers of width 128, SGD(lr=0.01, momentum=0.9), 20 epochs")
for scheme in SCHEMES:
    torch.manual_seed(0)
    model = init_(mlp([64] + [128] * 20 + [10]), scheme)
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    hist = fit(model, data, epochs=20, opt=opt)
    print(f"  {scheme:<16} train loss epoch 1 {hist[0][0]:.3f} -> epoch 20 {hist[-1][0]:.3f}   "
          f"val acc {hist[-1][2]:.3f}")
