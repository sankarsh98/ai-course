"""Membership inference on a small overfit model (educational demo, synthetic data).

The attacker sees a record and the model's output on it, and guesses whether the
record was in the training set. Two attacks:
  1. loss threshold [Yeom et al., 2018]: low loss => "member";
  2. difficulty-calibrated, in the spirit of offline LiRA [Carlini et al., 2022]:
     compare the target model's confidence on x with how confident reference
     models that never saw x are on x. Easy examples are confident everywhere;
     only an unusually confident output suggests membership.
We report TPR at low FPR, as Carlini et al. recommend, not just AUC.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import torch
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score, roc_curve

torch.set_num_threads(1)

# One synthetic population, split into disjoint pools.
X_all, y_all = make_classification(n_samples=40_000, n_features=40, n_informative=10,
                                   flip_y=0.15, class_sep=1.0, random_state=0)
X_all = torch.tensor(X_all, dtype=torch.float32)
y_all = torch.tensor(y_all, dtype=torch.long)
MEMBERS = torch.arange(0, 10_000)          # the target model trains on a prefix of these
NONMEMBERS = torch.arange(10_000, 20_000)  # same distribution, never trained on
REFERENCE = torch.arange(20_000, 36_000)   # attacker's own data from the population
TEST = torch.arange(36_000, 40_000)


def train(idx, weight_decay, epochs, seed=0):
    torch.manual_seed(seed)
    Xtr, ytr = X_all[idx], y_all[idx]
    model = torch.nn.Sequential(torch.nn.Linear(40, 256), torch.nn.ReLU(),
                                torch.nn.Linear(256, 256), torch.nn.ReLU(),
                                torch.nn.Linear(256, 2))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=weight_decay)
    for _ in range(epochs):
        perm = torch.randperm(len(idx))
        for i in range(0, len(idx), 64):
            b = perm[i:i + 64]
            loss = torch.nn.functional.cross_entropy(model(Xtr[b]), ytr[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def margin(model, idx):
    """logit(true class) - logit(other class) = log(p_y / (1 - p_y)).
    Unlike the float32 loss it does not saturate at 0 for very confident outputs."""
    z = model(X_all[idx])
    y = y_all[idx]
    return (z.gather(1, y[:, None]) - z.gather(1, 1 - y[:, None])).squeeze(1).numpy()


@torch.no_grad()
def accuracy(model, idx):
    return (model(X_all[idx]).argmax(1) == y_all[idx]).float().mean().item()


def roc_summary(score_m, score_nm):
    s = np.concatenate([score_m, score_nm])
    lab = np.concatenate([np.ones(len(score_m)), np.zeros(len(score_nm))])
    fpr, tpr, _ = roc_curve(lab, s)
    tpr_at = lambda f: tpr[np.searchsorted(fpr, f, side="right") - 1]
    return roc_auc_score(lab, s), tpr_at(0.01), tpr_at(0.001)


configs = [  # (name, n_train, weight_decay, epochs)
    ("overfit: n=500, no reg, 300 ep", 500, 0.0, 300),
    ("n=500, weight decay 1e-2, 20 ep", 500, 1e-2, 20),
    ("n=10,000, no reg, 30 ep", 10_000, 0.0, 30),
    ("n=10,000, weight decay 1e-2, 5 ep", 10_000, 1e-2, 5),
]
N_REF = 4
targets_m_all, targets_nm = MEMBERS, NONMEMBERS
print(f"{'target model':34} {'train':>5} {'test':>5} | {'loss attack':^20} | "
      f"{'calibrated attack':^20}")
print(f"{'':34} {'acc':>5} {'acc':>5} | {'AUC':>5} {'TPR@1%':>6} {'@0.1%':>6} | "
      f"{'AUC':>5} {'TPR@1%':>6} {'@0.1%':>6}")
for name, n_train, wd, ep in configs:
    tr_idx = MEMBERS[:n_train]
    model = train(tr_idx, wd, ep)
    m_t, nm_t = margin(model, tr_idx), margin(model, targets_nm)

    # 1. Loss threshold. Loss = log(1 + exp(-margin)) is decreasing in the margin,
    #    so ranking by margin is the same attack without float32 saturation.
    loss_res = roc_summary(m_t, nm_t)

    # 2. Calibrated: reference models trained the same way on attacker data
    #    (none of the targets), then score = (target margin - mean ref margin) / sd.
    g = torch.Generator().manual_seed(1)
    ref_m, ref_nm = [], []
    for k in range(N_REF):
        ref_idx = REFERENCE[torch.randperm(len(REFERENCE), generator=g)[:n_train]]
        ref = train(ref_idx, wd, ep, seed=100 + k)
        ref_m.append(margin(ref, tr_idx))
        ref_nm.append(margin(ref, targets_nm))
    ref_m, ref_nm = np.mean(ref_m, 0), np.mean(ref_nm, 0)
    sd = np.std(nm_t - ref_nm)  # one global scale; few reference models
    cal_res = roc_summary((m_t - ref_m) / sd, (nm_t - ref_nm) / sd)

    print(f"{name:34} {accuracy(model, tr_idx):5.3f} {accuracy(model, TEST):5.3f} | "
          f"{loss_res[0]:5.3f} {loss_res[1]:6.3f} {loss_res[2]:6.3f} | "
          f"{cal_res[0]:5.3f} {cal_res[1]:6.3f} {cal_res[2]:6.3f}")

# What differential privacy would guarantee. For an (eps, delta)-DP training
# algorithm, any membership test has TPR <= exp(eps) * FPR + delta.
print("\nDP ceiling on attack TPR at FPR = 1% (delta = 1e-5):")
for eps in [0.5, 1, 2, 4, 8]:
    print(f"  eps = {eps:3}: TPR <= {min(1.0, np.exp(eps) * 0.01 + 1e-5):.3f}")
