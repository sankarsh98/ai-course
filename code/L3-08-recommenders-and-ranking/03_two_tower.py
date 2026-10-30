"""A two-tower retrieval model trained with in-batch softmax, with and without the
log-frequency ("logQ") correction, and a cold-start test on items never seen in
training. Compared with implicit ALS, which has no vector for an unseen item.
"""
import numpy as np
import torch
import torch.nn as nn

from recdata import ials, make_world, organic_clicks, split_holdout, topk_metrics

torch.manual_seed(0)
torch.set_num_threads(4)
world = make_world()
X = organic_clicks(world)
train, test = split_holdout(X)
cp = world["click_prob"]
n_u, n_i = X.shape

# Hold out 20% of items completely: no training clicks at all (new catalogue items).
rng = np.random.default_rng(0)
cold = rng.choice(n_i, n_i // 5, replace=False)
warm = np.setdiff1d(np.arange(n_i), cold)
train_warm = train.copy()
train_warm[:, cold] = 0
pairs = np.argwhere(train_warm > 0)                    # (user, item) positives
q = train_warm.sum(0) / train_warm.sum()               # item sampling frequency
print(f"{len(pairs)} training clicks on {len(warm)} warm items; {len(cold)} cold items")


class Tower(nn.Module):
    def __init__(self, d_in, d_out=16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 64), nn.ReLU(), nn.Linear(64, d_out))

    def forward(self, x):
        return nn.functional.normalize(self.net(x), dim=-1)


def train_two_tower(logq, epochs=40, batch=256, temp=0.1):
    uf, itf = torch.tensor(world["user_feat"]), torch.tensor(world["item_feat"])
    uf, itf = (uf - uf.mean(0)) / uf.std(0), (itf - itf.mean(0)) / itf.std(0)
    ut, it = Tower(uf.shape[1]), Tower(itf.shape[1])
    opt = torch.optim.Adam(list(ut.parameters()) + list(it.parameters()), lr=3e-3)
    logq_t = torch.log(torch.tensor(q, dtype=torch.float32) + 1e-12)
    g = torch.Generator().manual_seed(0)
    for _ in range(epochs):
        for idx in torch.randperm(len(pairs), generator=g).split(batch):
            u, i = torch.tensor(pairs[idx.numpy(), 0]), torch.tensor(pairs[idx.numpy(), 1])
            logits = ut(uf[u]) @ it(itf[i]).T / temp   # every other item in the batch
            if logq:                                   # is a negative; popular items
                logits = logits - logq_t[i][None, :]   # appear more often, so correct
            loss = nn.functional.cross_entropy(logits, torch.arange(len(u)))
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        return (ut(uf) @ it(itf).T).numpy()


def true_prec(scores, items, k=10):
    """Mean true click probability of the top-k among `items` (excluding train clicks)."""
    s = np.where(train[:, items] > 0, -np.inf, scores[:, items])
    top = items[np.argsort(-s, axis=1)[:, :k]]
    return float(np.take_along_axis(cp, top, axis=1).mean())


U, V = ials(train_warm, k=8, alpha=10, reg=1)
models = {"iALS (warm items only)": U @ V.T,
          "two-tower, no logQ": train_two_tower(logq=False),
          "two-tower, logQ": train_two_tower(logq=True)}
top20 = np.argsort(-world["pop"])[:20]
print("\n                          warm items:                         cold items:")
print("   model                   Recall@10  true prec@10  top-20 share   true prec@10")
for name, s in models.items():
    sw = np.where(np.isin(np.arange(n_i), cold)[None, :], -np.inf, s)   # rank warm only
    r, _ = topk_metrics(sw, train_warm, test * np.isin(np.arange(n_i), warm)[None, :])
    top = np.argsort(-np.where(train > 0, -np.inf, sw), axis=1)[:, :10]
    share = np.isin(top, top20).mean()
    print(f"   {name:24s} {r:8.3f}  {true_prec(s, warm):11.3f}  {share:11.0%}"
          f"   {true_prec(s, cold):12.3f}")
print(f"   {'random':24s} {'':8s}  {true_prec(rng.random((n_u, n_i)), warm):11.3f}  {'':11s}"
      f"   {true_prec(rng.random((n_u, n_i)), cold):12.3f}")
print(f"   {'oracle':24s} {'':8s}  {true_prec(cp, warm):11.3f}  {'':11s}"
      f"   {true_prec(cp, cold):12.3f}")
