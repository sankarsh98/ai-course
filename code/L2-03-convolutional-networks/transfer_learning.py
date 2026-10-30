"""Transfer learning in miniature. Pretrain a CNN on digits 0-4, then learn digits 5-9 from
only k labelled examples per class: from scratch, as a linear probe on frozen features,
or by fine-tuning everything. Each setting is repeated over 5 random labelled subsets."""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

torch.set_num_threads(4)  # fixed thread count, so runs on a shared machine are comparable

X, y = load_digits(return_X_y=True)
X = (X / 16.0).astype(np.float32).reshape(-1, 1, 8, 8)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
src_tr = y_tr < 5                      # source task: digits 0-4
tgt_tr, tgt_te = y_tr >= 5, y_te >= 5  # target task: digits 5-9, relabelled 0-4
Xs, ys = torch.tensor(X_tr[src_tr]), torch.tensor(y_tr[src_tr])
Xt_pool, yt_pool = X_tr[tgt_tr], y_tr[tgt_tr] - 5
Xt_te, yt_te = torch.tensor(X_te[tgt_te]), torch.tensor(y_te[tgt_te] - 5)


def features():
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten())


def fit(model, params, x, y, steps, lr):
    opt = torch.optim.Adam(params, lr=lr)
    for _ in range(steps):
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


@torch.no_grad()
def acc(model):
    return (model(Xt_te).argmax(1) == yt_te).float().mean().item()


# Pretrain on the source task (about 630 images).
torch.manual_seed(0)
pre = nn.Sequential(features(), nn.Linear(256, 5))
g = torch.Generator().manual_seed(0)
opt = torch.optim.Adam(pre.parameters(), lr=2e-3)
for step in range(600):
    idx = torch.randint(0, len(Xs), (64,), generator=g)
    loss = F.cross_entropy(pre(Xs[idx]), ys[idx])
    opt.zero_grad()
    loss.backward()
    opt.step()
print(f"pretrained on digits 0-4: {len(Xs)} images, final batch loss {loss.item():.3f}")
print(f"target task digits 5-9: test set {len(Xt_te)} images\n")

print(f"{'labels/class':>12s} {'pixels+logreg':>14s} {'scratch CNN':>12s} {'linear probe':>13s} "
      f"{'fine-tune':>10s}")
for k in [1, 5, 20, 100]:
    res = {m: [] for m in ["logreg", "scratch", "probe", "finetune"]}
    for seed in range(5):
        rng = np.random.default_rng(seed)
        idx = np.concatenate([rng.choice(np.flatnonzero(yt_pool == c), k, replace=False)
                              for c in range(5)])
        xk, yk = torch.tensor(Xt_pool[idx]), torch.tensor(yt_pool[idx])
        lr_model = LogisticRegression(max_iter=5000).fit(xk.reshape(len(xk), -1), yk)
        res["logreg"].append(lr_model.score(Xt_te.reshape(len(Xt_te), -1), yt_te))

        torch.manual_seed(seed)
        scratch = nn.Sequential(features(), nn.Linear(256, 5))
        res["scratch"].append(acc(fit(scratch, scratch.parameters(), xk, yk, 200, 2e-3)))

        probe = nn.Sequential(copy.deepcopy(pre[0]), nn.Linear(256, 5))
        for p in probe[0].parameters():
            p.requires_grad_(False)                  # frozen pretrained features
        res["probe"].append(acc(fit(probe, probe[1].parameters(), xk, yk, 200, 2e-3)))

        ft = nn.Sequential(copy.deepcopy(pre[0]), nn.Linear(256, 5))
        res["finetune"].append(acc(fit(ft, ft.parameters(), xk, yk, 200, 5e-4)))  # smaller lr
    print(f"{k:12d} " + " ".join(f"{np.mean(v):7.3f}+-{np.std(v):.3f}" for v in res.values()))
