"""SimCLR-style contrastive learning on the 8x8 digits, with no labels.

Two random augmentations of each image form a positive pair; the other
images in the batch are negatives. The InfoNCE / NT-Xent loss pulls
positives together and pushes everything else apart. We then measure:
  - kNN and linear-probe accuracy (labels used only for evaluation),
  - accuracy on test images shifted by one pixel (the invariance we asked for),
  - alignment and uniformity (Wang and Isola, 2020),
  - the mean cosine between different images and the effective rank,
and compare temperatures and a "positives only" loss, which collapses.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

torch.manual_seed(0)
X, y = load_digits(return_X_y=True)
X = (X / 16.0).astype(np.float32)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
Xtr_t, Xte_t = torch.from_numpy(Xtr), torch.from_numpy(Xte)


def shift(imgs, dx, dy):
    """Translate 8x8 images by (dx, dy) pixels, filling with zeros."""
    im = F.pad(imgs.view(-1, 1, 8, 8), (1, 1, 1, 1))
    return im[:, :, 1 - dy:9 - dy, 1 - dx:9 - dx].reshape(-1, 64)


def augment(x):
    """Random shift of up to 1 pixel, random pixel dropout, brightness, noise."""
    out = torch.empty_like(x)
    d = torch.randint(-1, 2, (len(x), 2))
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            m = (d[:, 0] == dx) & (d[:, 1] == dy)
            out[m] = shift(x[m], dx, dy)
    out = out * (torch.rand_like(out) > 0.2)
    out = out * torch.empty(len(x), 1).uniform_(0.7, 1.3)
    return out + 0.05 * torch.randn_like(out)


def make_encoder():
    return nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 64))


def nt_xent(z1, z2, tau):
    """InfoNCE over 2N views: for each view, its partner is the positive."""
    z = F.normalize(torch.cat([z1, z2]), dim=1)
    sim = z @ z.T / tau
    sim.fill_diagonal_(float("-inf"))            # a view is not its own negative
    n = len(z1)
    target = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)])
    return F.cross_entropy(sim, target)


def train(tau=0.2, positives_only=False, epochs=60, seed=0):
    torch.manual_seed(seed)
    enc = make_encoder()
    head = nn.Sequential(nn.ReLU(), nn.Linear(64, 32))   # projection head
    opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()), lr=1e-3)
    for _ in range(epochs):
        for b in torch.randperm(len(Xtr_t)).split(256):
            x = Xtr_t[b]
            z1, z2 = head(enc(augment(x))), head(enc(augment(x)))
            if positives_only:   # no negatives: only pull pairs together
                loss = (F.normalize(z1, dim=1) - F.normalize(z2, dim=1)).pow(2).sum(1).mean()
            else:
                loss = nt_xent(z1, z2, tau)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return enc


def evaluate(name, embed):
    with torch.no_grad():
        h_tr, h_te = embed(Xtr_t).numpy(), embed(Xte_t).numpy()
        h_sh = embed(shift(Xte_t, 1, 0)).numpy()          # every test image moved right
        # Alignment and uniformity on L2-normalized vectors.
        a = F.normalize(embed(augment(Xte_t)), dim=1)
        b = F.normalize(embed(augment(Xte_t)), dim=1)
        align = (a - b).pow(2).sum(1).mean().item()
        unif = torch.pdist(a).pow(2).mul(-2).exp().mean().log().item()
    norm = lambda M: M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    knn = KNeighborsClassifier(5).fit(norm(h_tr), ytr)
    lin = LogisticRegression(max_iter=3000).fit(h_tr, ytr)
    hn = norm(h_te)
    mean_cos = (hn @ hn.T)[~np.eye(len(hn), dtype=bool)].mean()
    # Effective rank (participation ratio of the covariance eigenvalues):
    # 64 if variance is spread evenly over 64 directions, 1 if all in one.
    ev = np.linalg.svd(h_tr - h_tr.mean(0), compute_uv=False) ** 2
    eff_rank = ev.sum() ** 2 / (ev ** 2).sum()
    print(f"{name:<24} {knn.score(hn, yte):>5.3f} {knn.score(norm(h_sh), yte):>8.3f}"
          f" {lin.score(h_te, yte):>7.3f} {align:>7.3f} {unif:>7.3f}"
          f" {mean_cos:>8.3f} {eff_rank:>6.1f}")


print(f"{'representation':<24} {'kNN':>5} {'kNN,shift':>8} {'linear':>7}"
      f" {'align':>7} {'unif':>7} {'mean cos':>8} {'eff.rk':>6}")
evaluate("raw pixels", lambda x: x)
torch.manual_seed(0)
evaluate("random encoder", make_encoder())
for tau in (0.05, 0.2, 1.0):
    evaluate(f"contrastive, tau={tau}", train(tau))
evaluate("positives only", train(positives_only=True))
