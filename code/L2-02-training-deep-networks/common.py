"""Shared helpers for the L2-02 scripts: digits as tensors, an MLP builder,
and a compact version of the training loop written out in full in training_loop.py."""
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

torch.set_num_threads(4)  # tiny models: more threads mostly add overhead


def digits(n_train=None):
    """Same test split as L1-02 and L2-01; 20% of the rest is held out for validation."""
    X, y = load_digits(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=0.2, random_state=0, stratify=y_tr)
    if n_train is not None:
        X_tr, y_tr = X_tr[:n_train], y_tr[:n_train]
    scaler = StandardScaler().fit(X_tr)          # fit on training rows only
    as_x = lambda a: torch.tensor(scaler.transform(a), dtype=torch.float32)
    as_y = lambda a: torch.tensor(a, dtype=torch.long)
    return {"train": (as_x(X_tr), as_y(y_tr)), "val": (as_x(X_va), as_y(y_va)), "test": (as_x(X_te), as_y(y_te))}


def mlp(sizes, act=nn.ReLU, norm=None, dropout=0.0):
    """Linear -> [norm] -> act -> [dropout] for each hidden layer, then a linear output."""
    layers = []
    for m, k in zip(sizes[:-2], sizes[1:-1]):
        layers.append(nn.Linear(m, k))
        if norm is not None:
            layers.append(norm(k))
        layers.append(act())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(sizes[-2], sizes[-1]))
    return nn.Sequential(*layers)


@torch.no_grad()
def evaluate(model, X, y):
    model.eval()                                   # dropout off, batch norm uses running stats
    logits = model(X)
    return nn.functional.cross_entropy(logits, y).item(), (logits.argmax(1) == y).float().mean().item()


def fit(model, data, epochs, opt, scheduler=None, batch_size=64, seed=0):
    """Train; return per-epoch (train loss, val loss, val acc). scheduler steps once per batch."""
    X, y = data["train"]
    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    history = []
    for _ in range(epochs):
        model.train()
        total = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
            if scheduler is not None:
                scheduler.step()
            total += loss.item() * len(xb)
        history.append((total / len(X), *evaluate(model, *data["val"])))
    return history
