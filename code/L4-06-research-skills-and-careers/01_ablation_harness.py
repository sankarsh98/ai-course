"""Ablation harness template: one config per run, several seeds per config, every
run logged with tracker.py. Analysis lives in 02_analyze_runs.py.

Task: classify scikit-learn's bundled 8x8 digits with a small MLP trained on only
400 images, so that seed-to-seed variation is visible. The "full" method has five
components; each ablation removes one. Usage:
    python 01_ablation_harness.py [--root runs/ablation] [--seeds 5]
"""
import argparse
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import time
from dataclasses import asdict, dataclass, replace

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.datasets import load_digits

from tracker import Run

torch.set_num_threads(1)


@dataclass(frozen=True)
class Config:
    name: str = "full"
    seed: int = 0
    data_seed: int = 0         # fixed: every config sees the same split
    n_train: int = 400
    epochs: int = 40
    lr: float = 3e-3
    layernorm: bool = True
    dropout: float = 0.2
    weight_decay: float = 1e-3
    augment: bool = True       # random 1-pixel shifts
    cosine_lr: bool = True


def get_data(cfg):
    X, y = load_digits(return_X_y=True)
    X = (X / 16.0).astype(np.float32)
    idx = np.random.default_rng(cfg.data_seed).permutation(len(X))
    tr, va, te = idx[:cfg.n_train], idx[cfg.n_train:cfg.n_train + 400], idx[cfg.n_train + 400:]
    t = lambda a: torch.tensor(a)
    return (t(X[tr]), t(y[tr])), (t(X[va]), t(y[va])), (t(X[te]), t(y[te]))


def shift(xb, g):
    """Random shift by up to one pixel in each direction (zero padding)."""
    imgs = F.pad(xb.view(-1, 1, 8, 8), (1, 1, 1, 1))
    dx, dy = torch.randint(0, 3, (2,), generator=g).tolist()
    return imgs[:, :, dy:dy + 8, dx:dx + 8].reshape(-1, 64)


def build(cfg):
    layers, d = [], 64
    for _ in range(2):
        layers += [torch.nn.Linear(d, 256)]
        layers += [torch.nn.LayerNorm(256)] if cfg.layernorm else []
        layers += [torch.nn.ReLU(), torch.nn.Dropout(cfg.dropout)]
        d = 256
    return torch.nn.Sequential(*layers, torch.nn.Linear(d, 10))


@torch.no_grad()
def accuracy(model, X, y):
    model.eval()
    a = (model(X).argmax(1) == y).float().mean().item()
    model.train()
    return a


def run_one(cfg, root):
    torch.manual_seed(cfg.seed)              # init and dropout
    g = torch.Generator().manual_seed(cfg.seed)  # data order and augmentation
    (Xtr, ytr), (Xva, yva), (Xte, yte) = get_data(cfg)
    model = build(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps = cfg.epochs * int(np.ceil(cfg.n_train / 32))
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps) if cfg.cosine_lr
             else torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0))
    with Run(root, asdict(cfg), __file__) as run:
        for ep in range(cfg.epochs):
            perm = torch.randperm(cfg.n_train, generator=g)
            for i in range(0, cfg.n_train, 32):
                b = perm[i:i + 32]
                xb = shift(Xtr[b], g) if cfg.augment else Xtr[b]
                loss = F.cross_entropy(model(xb), ytr[b])
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
            if ep % 10 == 9:
                run.log(ep + 1, loss=loss.item(), val_acc=accuracy(model, Xva, yva))
        # The test set is evaluated once, at the end, and never used for choices.
        run.finish(val_acc=accuracy(model, Xva, yva), test_acc=accuracy(model, Xte, yte))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/ablation")
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    full = Config()
    configs = [full,
               replace(full, name="- layernorm", layernorm=False),
               replace(full, name="- dropout", dropout=0.0),
               replace(full, name="- weight decay", weight_decay=0.0),
               replace(full, name="- augmentation", augment=False),
               replace(full, name="- cosine lr", cosine_lr=False),
               replace(full, name="none (plain MLP)", layernorm=False, dropout=0.0,
                       weight_decay=0.0, augment=False, cosine_lr=False)]
    t0 = time.perf_counter()
    for cfg in configs:
        for s in range(args.seeds):
            run_one(replace(cfg, seed=s), args.root)
    print(f"{len(configs) * args.seeds} runs written to {args.root} "
          f"in {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
