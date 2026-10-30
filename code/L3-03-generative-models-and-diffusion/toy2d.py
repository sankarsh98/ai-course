"""Shared pieces for the 2D generative-model toys (imported by the scripts; prints nothing).

Data: a mixture of 8 Gaussians on a circle of radius 2, each with std SIGMA.
Because the modes are known, we can measure what a sample-quality number hides:
  in_mode  fraction of samples within 3 SIGMA of some mode centre (real data: 0.989)
  modes    how many of the 8 modes get at least a quarter of their fair share
  mode_kl  KL(mode histogram || uniform) in nats; 0 means perfectly balanced
  spread   RMS distance to the mode centre for in-mode samples, relative to real data
"""
import math
import torch
import torch.nn as nn

# These models are tiny; beyond a few threads PyTorch spends more time coordinating than computing.
torch.set_num_threads(4)

SIGMA = 0.1
K = 8
CENTERS = 2.0 * torch.stack([torch.cos(torch.arange(K) * 2 * math.pi / K),
                             torch.sin(torch.arange(K) * 2 * math.pi / K)], dim=1)
# E[r^2 | r < 3 sigma] / sigma^2 for a 2D Gaussian, so real data has spread 1.00.
_U_TRUNC = 2 - 9 * math.exp(-4.5) / (1 - math.exp(-4.5))


def sample_data(n, gen=None, return_labels=False):
    k = torch.randint(0, K, (n,), generator=gen)
    x = CENTERS[k] + SIGMA * torch.randn(n, 2, generator=gen)
    return (x, k) if return_labels else x


def evaluate(x):
    d = torch.cdist(x, CENTERS)                      # (n, 8) distances to centres
    dmin, k = d.min(dim=1)
    ok = dmin < 3 * SIGMA
    counts = torch.bincount(k[ok], minlength=K).float()
    p = counts / counts.sum().clamp(min=1)
    mode_kl = (p[p > 0] * (p[p > 0] * K).log()).sum().item()
    spread = (dmin[ok].pow(2).mean() / (SIGMA**2 * _U_TRUNC)).sqrt().item() if ok.any() else float("nan")
    return dict(in_mode=ok.float().mean().item(),
                modes=int((counts >= len(x) / (4 * K)).sum()),
                mode_kl=mode_kl, spread=spread)


def fmt(m):
    return f"in_mode {m['in_mode']:.3f}  modes {m['modes']}/8  mode_kl {m['mode_kl']:.3f}  spread {m['spread']:.2f}"


# ---- Noise schedule (DDPM: T = 1000, beta linear from 1e-4 to 0.02) -------------
T = 1000
BETAS = torch.linspace(1e-4, 0.02, T)
ALPHAS = 1 - BETAS
ABAR = torch.cumprod(ALPHAS, dim=0)                  # ABAR[i] is alpha-bar at t = i + 1


def time_features(t01, n_freq=16):
    """Sinusoidal features of a time in [0, 1], so the MLP can tell noise levels apart."""
    f = torch.exp(torch.linspace(0, math.log(200.0), n_freq))
    a = t01[:, None] * f[None, :]
    return torch.cat([a.sin(), a.cos()], dim=1)


class Net(nn.Module):
    """MLP that maps (x_t, t, optional class) to a 2D output: noise or velocity."""

    def __init__(self, n_classes=0, hidden=128):
        super().__init__()
        self.emb = nn.Embedding(n_classes + 1, 16) if n_classes else None   # last index = "no class"
        d_in = 2 + 32 + (16 if n_classes else 0)
        self.mlp = nn.Sequential(nn.Linear(d_in, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, 2))

    def forward(self, x, t01, c=None):
        h = [x, time_features(t01)]
        if self.emb is not None:
            h.append(self.emb(c))
        return self.mlp(torch.cat(h, dim=1))


def exact_eps(x, i, centers=CENTERS, sigma0=SIGMA):
    """E[noise | x_t] for Gaussian-mixture data with equal weights, in closed form.

    q_t is again a mixture: components N(a * mu_k, v I) with a = sqrt(abar), v = a^2 sigma0^2 + 1 - abar.
    Its score is sum_k r_k(x) (a mu_k - x) / v, and E[eps | x_t] = -sqrt(1 - abar) * score.
    With sigma0 = 0 and centers = training points this is the optimum for an empirical dataset.
    """
    a, s2 = ABAR[i].sqrt(), 1 - ABAR[i]
    v = a**2 * sigma0**2 + s2
    logits = -torch.cdist(x, a * centers).pow(2) / (2 * v)
    r = torch.softmax(logits, dim=1)                 # responsibilities r_k(x)
    score = (r @ (a * centers) - x) / v
    return -s2.sqrt() * score


@torch.no_grad()
def ddpm_sample(eps_fn, n, gen):
    """Ancestral sampling, T steps, sigma_t^2 = beta_t."""
    x = torch.randn(n, 2, generator=gen)
    for i in reversed(range(T)):
        eps = eps_fn(x, i)
        x = (x - BETAS[i] / (1 - ABAR[i]).sqrt() * eps) / ALPHAS[i].sqrt()
        if i > 0:
            x = x + BETAS[i].sqrt() * torch.randn(n, 2, generator=gen)
    return x


@torch.no_grad()
def ddim_sample(eps_fn, n, gen, steps, idx=None):
    """Deterministic DDIM (eta = 0) on an evenly spaced subset of the T noise levels,
    or on the noise levels given by the decreasing index list idx."""
    idx = idx or torch.linspace(T - 1, 0, steps).round().long().tolist()
    x = torch.randn(n, 2, generator=gen)
    for j, i in enumerate(idx):
        eps = eps_fn(x, i)
        x0_hat = (x - (1 - ABAR[i]).sqrt() * eps) / ABAR[i].sqrt()
        abar_prev = ABAR[idx[j + 1]] if j + 1 < len(idx) else torch.tensor(1.0)
        x = abar_prev.sqrt() * x0_hat + (1 - abar_prev).sqrt() * eps
    return x


def train_eps_model(steps=3000, batch=1024, n_classes=0, p_uncond=0.1, seed=0, label_fn=None):
    """Train the noise predictor with the DDPM 'simple' loss ||eps - eps_theta(x_t, t)||^2."""
    torch.manual_seed(seed)
    net = Net(n_classes)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        x0, c = sample_data(batch, return_labels=True)
        i = torch.randint(0, T, (batch,))
        eps = torch.randn_like(x0)
        xt = ABAR[i].sqrt()[:, None] * x0 + (1 - ABAR[i]).sqrt()[:, None] * eps
        if n_classes:   # classifier-free guidance training: sometimes hide the label
            c = label_fn(c) if label_fn else c
            c = torch.where(torch.rand(batch) < p_uncond, torch.full_like(c, n_classes), c)
            pred = net(xt, i.float() / T, c)
        else:
            pred = net(xt, i.float() / T)
        loss = (pred - eps).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return net, loss.item()
