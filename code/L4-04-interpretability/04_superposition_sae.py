"""Superposition in a toy model, then a sparse autoencoder on data whose true
features we know, so we can check what the SAE recovers instead of guessing.

Part 1 follows the setup of Elhage et al. (2022), "Toy Models of Superposition":
n sparse features, squeezed through m < n hidden dimensions, x' = ReLU(W^T W x + b).
"""
import time

import torch

torch.set_num_threads(4)


# ---------------------------------------------------------------- part 1
def toy_superposition(p_active, n=20, m=5, steps=4000, seed=0):
    """Train x' = ReLU(W^T W x + b) with importance-weighted squared error.
    Each feature is nonzero with probability p_active, uniform on [0, 1]."""
    g = torch.Generator().manual_seed(seed)
    W = (torch.randn(m, n, generator=g) * 0.3).requires_grad_()
    b = torch.zeros(n, requires_grad=True)
    imp = 0.8 ** torch.arange(n)            # feature i matters 0.8^i as much as feature 0
    opt = torch.optim.Adam([W, b], lr=3e-3)
    for _ in range(steps):
        x = torch.rand(1024, n, generator=g) * (torch.rand(1024, n, generator=g) < p_active)
        out = torch.relu(x @ W.T @ W + b)
        loss = (imp * (out - x) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    W = W.detach()
    norms = W.norm(dim=0)                     # ||W_i||: is feature i represented?
    unit = W / norms.clamp_min(1e-9)
    overlap = (unit.T @ W) ** 2               # (W_i_hat . W_j)^2
    interference = overlap.sum(1) - overlap.diagonal()  # sum over j != i
    return norms, interference


print("Part 1: 20 features, 5 hidden dimensions")
print(" P(feature active)  features with |W_i|>0.5  mean interference of those")
for p in [1.0, 0.3, 0.1, 0.03]:
    norms, interf = toy_superposition(p)
    rep = norms > 0.5
    print(f"   {p:5.2f}             {int(rep.sum()):2d}                      {interf[rep].mean():.3f}")


# ---------------------------------------------------------------- part 2
D, N_TRUE, N_LAT = 32, 96, 192   # activation width, true features, SAE latents


def make_data(n_samples, seed):
    """Activations built from 96 known unit directions in 32 dimensions:
    each sample uses about 3 of them with random positive magnitudes."""
    g = torch.Generator().manual_seed(seed)
    dirs = torch.randn(N_TRUE, D, generator=g)
    dirs /= dirs.norm(dim=1, keepdim=True)
    active = torch.rand(n_samples, N_TRUE, generator=g) < 3 / N_TRUE
    coef = active * (0.5 + torch.rand(n_samples, N_TRUE, generator=g))
    return coef @ dirs, dirs, active


def train_sae(acts, l1, seed, steps=3000):
    """ReLU SAE: f = ReLU(W_e (a - b_d) + b_e), a_hat = W_d f + b_d.
    Loss = ||a - a_hat||^2 + l1 * sum_k f_k ||W_d[:, k]||."""
    g = torch.Generator().manual_seed(seed)
    W_d = torch.randn(D, N_LAT, generator=g)
    W_d = (W_d / W_d.norm(dim=0)).requires_grad_()
    W_e = W_d.detach().T.clone().requires_grad_()      # common init: encoder = decoder^T
    b_e = torch.zeros(N_LAT, requires_grad=True)
    b_d = acts.mean(0).clone().requires_grad_()
    opt = torch.optim.Adam([W_d, W_e, b_e, b_d], lr=2e-3)
    for _ in range(steps):
        a = acts[torch.randint(len(acts), (1024,), generator=g)]
        f = torch.relu((a - b_d) @ W_e.T + b_e)
        a_hat = f @ W_d.T + b_d
        loss = ((a - a_hat) ** 2).sum(1).mean() + l1 * (f * W_d.norm(dim=0)).sum(1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        f = torch.relu((acts - b_d) @ W_e.T + b_e)
        a_hat = f @ W_d.T + b_d
        fvu = ((acts - a_hat) ** 2).sum() / ((acts - acts.mean(0)) ** 2).sum()
        l0 = (f > 0).float().sum(1).mean()
        dead = int(((f > 0).sum(0) == 0).sum())
    return (W_d / W_d.norm(dim=0)).detach(), fvu.item(), l0.item(), dead


t0 = time.time()
acts, dirs, active = make_data(20000, seed=0)
print(f"\nPart 2: {N_TRUE} true features in {D} dimensions, SAE with {N_LAT} latents")
print(f"true L0 (features active per sample): {active.float().sum(1).mean():.2f}")
print("   l1    FVU    L0   dead  true feats matched  mean best cos  latents near a true feat")
for l1 in [0.0, 0.03, 0.1, 0.3, 1.0]:
    dec, fvu, l0, dead = train_sae(acts, l1, seed=0)
    sims = dirs @ dec                           # (true feature, latent) cosines
    best = sims.max(1).values                   # best latent for each true feature
    near = int((sims.max(0).values > 0.9).sum())  # latents that match some feature
    print(f"  {l1:4.2f}  {fvu:5.3f}  {l0:5.1f}  {dead:3d}    {int((best > 0.9).sum()):3d} / {N_TRUE}"
          f"          {best.mean():.3f}        {near:3d} / {N_LAT}")

# PCA can only give 32 orthogonal directions for 96 features.
_, _, Vt = torch.linalg.svd(acts - acts.mean(0), full_matrices=False)
best_pca = (dirs @ Vt.T).abs().max(1).values
print(f"  PCA (32 components): true features matched {int((best_pca > 0.9).sum())} / {N_TRUE}, "
      f"mean best cos {best_pca.mean():.3f}")

# Same data, same l1, different seed: do two SAEs find the same latents?
decA, *_ = train_sae(acts, 0.3, seed=1)
decB, *_ = train_sae(acts, 0.3, seed=2)
cross = (decA.T @ decB).max(1).values
print(f"\nseed 1 vs seed 2 at l1=0.3: {int((cross > 0.9).sum())} / {N_LAT} latents of A "
      f"have a twin in B (cos>0.9)")
print(f"\npart 2 time {time.time() - t0:.1f} s")
