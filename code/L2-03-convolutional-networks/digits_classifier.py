"""A small image classifier on CPU: scikit-learn's 8x8 digits. Compares logistic regression,
an MLP, a CNN, and a CNN trained with shift augmentation, on clean and shifted test images."""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

torch.set_num_threads(4)  # fixed thread count, so runs on a shared machine are comparable

X, y = load_digits(return_X_y=True)
X = (X / 16.0).astype(np.float32).reshape(-1, 1, 8, 8)   # pixel values 0..16 -> 0..1
# Same split as L1-02 so the logistic-regression number is comparable.
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
X_tr, X_te = torch.tensor(X_tr), torch.tensor(X_te)
y_tr, y_te = torch.tensor(y_tr), torch.tensor(y_te)


def shift(x, gen, max_shift=1):
    """Shift each image by up to max_shift pixels in each direction, filling with zeros."""
    n = x.shape[0]
    padded = F.pad(x, (max_shift,) * 4)
    d = torch.randint(0, 2 * max_shift + 1, (n, 2), generator=gen)
    return torch.stack([padded[i, :, d[i, 0]:d[i, 0] + 8, d[i, 1]:d[i, 1] + 8] for i in range(n)])


def mlp():
    return nn.Sequential(nn.Flatten(), nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 10))


def cnn():
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),     # 8x8 -> 4x4
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),     # 4x4 -> 2x2
        nn.Flatten(), nn.Linear(64 * 2 * 2, 10))


def train(model, augment, seed, epochs=40, bs=64):
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
    for _ in range(epochs):
        perm = torch.randperm(len(X_tr), generator=gen)
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            xb = shift(X_tr[idx], gen) if augment else X_tr[idx]
            loss = F.cross_entropy(model(xb), y_tr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model.eval()


@torch.no_grad()
def acc(model, x):
    return (model(x).argmax(1) == y_te).float().mean().item()


# A fixed "shifted" test set: every test image moved by up to 1 pixel.
X_te_shift = shift(X_te, torch.Generator().manual_seed(123))

lr = LogisticRegression(max_iter=5000).fit(X_tr.reshape(len(X_tr), -1), y_tr)
print(f"train {len(X_tr)} images, test {len(X_te)} images")
print(f"{'model':22s} {'params':>7s} {'test acc':>16s} {'shifted test acc':>18s} {'time/seed':>9s}")
print(f"{'logistic regression':22s} {650:7d} {lr.score(X_te.reshape(len(X_te), -1), y_te):16.3f} "
      f"{lr.score(X_te_shift.reshape(len(X_te), -1), y_te):18.3f}")
for name, make, augment in [("MLP", mlp, False), ("MLP + shift aug", mlp, True),
                            ("CNN", cnn, False), ("CNN + shift aug", cnn, True)]:
    clean, shifted = [], []
    t0 = time.time()
    for seed in range(3):
        torch.manual_seed(seed)
        model = train(make(), augment, seed)
        clean.append(acc(model, X_te))
        shifted.append(acc(model, X_te_shift))
    n_par = sum(p.numel() for p in model.parameters())
    print(f"{name:22s} {n_par:7,d} {np.mean(clean):9.3f} +- {np.std(clean):.3f} "
          f"{np.mean(shifted):11.3f} +- {np.std(shifted):.3f} {(time.time() - t0) / 3:8.1f}s")
