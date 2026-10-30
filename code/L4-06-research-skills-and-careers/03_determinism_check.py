"""Reproducibility check: is the training run bit-for-bit repeatable, and what breaks it?

Trains the same small model under several conditions and compares a hash of the
final weights and the test accuracy with a reference run.
"""
import hashlib
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.datasets import load_digits

X, y = load_digits(return_X_y=True)
X = torch.tensor(X / 16.0, dtype=torch.float32)
y = torch.tensor(y)
Xtr, ytr, Xte, yte = X[:1200], y[:1200], X[1200:], y[1200:]


def train(seed=0, threads=1, batch=32, dtype=torch.float32, shuffle_seed=None):
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(64, 512), torch.nn.ReLU(),
                                torch.nn.Linear(512, 10)).to(dtype)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    g = torch.Generator().manual_seed(seed if shuffle_seed is None else shuffle_seed)
    for _ in range(10):
        perm = torch.randperm(len(Xtr), generator=g)
        for i in range(0, len(Xtr), batch):
            b = perm[i:i + batch]
            loss = F.cross_entropy(model(Xtr[b].to(dtype)), ytr[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        acc = (model(Xte.to(dtype)).argmax(1) == yte).float().mean().item()
        flat = torch.cat([p.detach().double().flatten() for p in model.parameters()])
    digest = hashlib.sha256(flat.numpy().tobytes()).hexdigest()[:12]
    return digest, acc, flat


ref_hash, ref_acc, ref_w = train()
print(f"torch {torch.__version__}, reference run: weights {ref_hash}, test acc {ref_acc:.4f}\n")
print(f"{'condition':40} {'weights hash':>13} {'same?':>6} {'max |dw|':>9} {'test acc':>8}")
conditions = [
    ("same seed, same settings (rerun)", {}),
    ("same seed, 8 threads", {"threads": 8}),
    ("same init, different data order", {"shuffle_seed": 1}),
    ("different seed (init and order)", {"seed": 1}),
    ("batch size 64 instead of 32", {"batch": 64}),
    ("float64 instead of float32", {"dtype": torch.float64}),
]
for name, kw in conditions:
    h, a, w = train(**kw)
    print(f"{name:40} {h:>13} {str(h == ref_hash):>6} {(w - ref_w).abs().max().item():9.2e} {a:8.4f}")

# Across 10 seeds, how much does test accuracy move? That spread is the noise
# floor any claimed improvement must clear.
accs = [train(seed=s)[1] for s in range(10)]
print(f"\ntest accuracy over 10 seeds: mean {np.mean(accs):.4f}, sd {np.std(accs, ddof=1):.4f}, "
      f"range [{min(accs):.4f}, {max(accs):.4f}]")
