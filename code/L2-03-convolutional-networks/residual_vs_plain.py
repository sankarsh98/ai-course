"""Plain deep CNNs vs residual CNNs of the same depth on digits.
Same layers, same optimizer, same budget; the only difference is the identity shortcut."""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

torch.set_num_threads(4)  # fixed thread count, so runs on a shared machine are comparable

X, y = load_digits(return_X_y=True)
X = (X / 16.0).astype(np.float32).reshape(-1, 1, 8, 8)
X_tr, X_te, y_tr, y_te = map(torch.tensor, train_test_split(X, y, test_size=0.3, random_state=0,
                                                            stratify=y))
C = 16  # channels everywhere; all layers work at 8x8 resolution


def conv_bn(relu=True):
    layers = [nn.Conv2d(C, C, 3, padding=1, bias=False), nn.BatchNorm2d(C)]
    return nn.Sequential(*layers, nn.ReLU()) if relu else nn.Sequential(*layers)


class PlainPair(nn.Module):        # two conv layers, no shortcut
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(conv_bn(), conv_bn(relu=False))

    def forward(self, x):
        return F.relu(self.f(x))


class ResidualPair(nn.Module):     # the same two conv layers, plus the identity: relu(x + F(x))
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(conv_bn(), conv_bn(relu=False))

    def forward(self, x):
        return F.relu(x + self.f(x))


def net(depth, block):
    """depth = number of 3x3 conv layers in the body (2 per block)."""
    return nn.Sequential(nn.Conv2d(1, C, 3, padding=1), nn.ReLU(),
                         *[block() for _ in range(depth // 2)],
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(C, 10))


def first_layer_grad_norm(model):
    model.train()
    loss = F.cross_entropy(model(X_tr[:256]), y_tr[:256])
    model.zero_grad()
    loss.backward()
    return model[0].weight.grad.norm().item()


def train(model, seed, epochs=10, bs=128):
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(X_tr), generator=gen)
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            loss = F.cross_entropy(model(X_tr[idx]), y_tr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        train_loss = F.cross_entropy(model(X_tr), y_tr).item()
        test_acc = (model(X_te).argmax(1) == y_te).float().mean().item()
    return train_loss, test_acc


print(f"{'body depth':>10s} {'type':>9s} {'grad norm, layer 1, init':>25s} "
      f"{'final train loss':>17s} {'test acc':>9s} {'time':>6s}")
for depth in [4, 16, 40]:
    for name, block in [("plain", PlainPair), ("residual", ResidualPair)]:
        t0 = time.time()
        g, losses, accs = [], [], []
        for seed in range(2):
            torch.manual_seed(seed)
            model = net(depth, block)
            g.append(first_layer_grad_norm(model))
            tl, ta = train(model, seed)
            losses.append(tl)
            accs.append(ta)
        print(f"{depth:10d} {name:>9s} {np.mean(g):25.4f} {np.mean(losses):17.3f} "
              f"{np.mean(accs):9.3f} {time.time() - t0:5.1f}s")
