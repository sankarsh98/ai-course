"""Adversarial examples on a small image classifier (educational demo).

Data: scikit-learn's bundled 8x8 digits (1,797 images), pixels scaled to [0, 1].
We compare, at each L-infinity budget eps:
  - random sign noise of the same size (a control: is it just "noise"?),
  - FGSM, one gradient-sign step [Goodfellow et al., 2014],
  - PGD, 20 projected gradient steps [Madry et al., 2017],
for a standard model and an adversarially trained (PGD) model.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

torch.set_num_threads(1)
torch.manual_seed(0)

X, y = load_digits(return_X_y=True)
X = X / 16.0  # pixel values are 0..16
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
Xtr, Xte = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(Xte, dtype=torch.float32)
ytr, yte = torch.tensor(ytr), torch.tensor(yte)


def make_model():
    return torch.nn.Sequential(torch.nn.Linear(64, 128), torch.nn.ReLU(),
                               torch.nn.Linear(128, 10))


def pgd(model, x, y, eps, steps=20, alpha=None):
    """Maximize the loss within the box ||delta||_inf <= eps, keeping pixels in [0, 1].
    steps=1 with alpha=eps and no random start is FGSM."""
    alpha = alpha or 2.5 * eps / steps
    delta = torch.zeros_like(x) if steps == 1 else (torch.rand_like(x) * 2 - 1) * eps
    for _ in range(steps):
        delta.requires_grad_(True)
        loss = F.cross_entropy(model((x + delta).clamp(0, 1)), y)
        grad, = torch.autograd.grad(loss, delta)
        delta = (delta + alpha * grad.sign()).clamp(-eps, eps).detach()
    return (x + delta).clamp(0, 1)


def fgsm(model, x, y, eps):
    return pgd(model, x, y, eps, steps=1, alpha=eps)


def train(adv_eps=0.0, epochs=60):
    model = make_model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 64):
            b = perm[i:i + 64]
            xb, yb = Xtr[b], ytr[b]
            if adv_eps > 0:  # train on worst-case perturbed inputs (PGD, 7 steps)
                xb = pgd(model, xb, yb, adv_eps, steps=7)
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def acc(model, x):
    return (model(x).argmax(1) == yte).float().mean().item()


models = {"standard": train(), "PGD-trained (eps=0.1)": train(adv_eps=0.1)}
print(f"{'model':22} {'eps':>5} {'clean':>6} {'random':>7} {'FGSM':>6} {'PGD-20':>7}")
for name, m in models.items():
    for eps in [0.0, 0.05, 0.1, 0.2]:
        g = torch.Generator().manual_seed(1)
        noise = (torch.randint(0, 2, Xte.shape, generator=g) * 2 - 1).float() * eps
        rand_acc = acc(m, (Xte + noise).clamp(0, 1))
        print(f"{name:22} {eps:5.2f} {acc(m, Xte):6.3f} {rand_acc:7.3f} "
              f"{acc(m, fgsm(m, Xte, yte, eps)):6.3f} {acc(m, pgd(m, Xte, yte, eps)):7.3f}")

# Does an attack crafted on one model fool another trained separately? (transfer)
other = train()
torch.manual_seed(2)
x_adv = pgd(models["standard"], Xte, yte, 0.1)
print(f"\nTransfer: PGD (eps=0.1) examples from 'standard' -> independently trained "
      f"standard model: accuracy {acc(other, x_adv):.3f} (clean {acc(other, Xte):.3f})")
print(f"Transfer -> PGD-trained model: accuracy {acc(models['PGD-trained (eps=0.1)'], x_adv):.3f}")
