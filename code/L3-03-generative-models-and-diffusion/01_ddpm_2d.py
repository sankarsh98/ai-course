"""DDPM on a 2D mixture of 8 Gaussians, trained on CPU in seconds.

1. The forward process: closed form versus running the noising chain.
2. A sampler check with the exact (analytic) noise predictor, before any learning.
3. Train an MLP noise predictor; compare it with the exact one.
4. Sample with DDPM (1000 stochastic steps) and DDIM (deterministic, fewer steps).
"""
import time
import torch
from toy2d import (ABAR, ALPHAS, T, evaluate, exact_eps, fmt, sample_data,
                   ddpm_sample, ddim_sample, train_eps_model)

torch.manual_seed(0)
gen = torch.Generator().manual_seed(1)
N = 10000

# ---- 1. Forward process ------------------------------------------------------
print("alpha_bar and signal-to-noise ratio along the schedule:")
for t in (1, 100, 250, 500, 750, 1000):
    ab = ABAR[t - 1].item()
    print(f"  t={t:4d}  alpha_bar={ab:.5f}  SNR={ab / (1 - ab):.2e}")

x0 = sample_data(100000, gen)
x = x0.clone()
for i in range(500):                  # run the chain x_t = sqrt(1-b) x_{t-1} + sqrt(b) z for 500 steps
    x = ALPHAS[i].sqrt() * x + (1 - ALPHAS[i]).sqrt() * torch.randn_like(x)
resid = x - ABAR[499].sqrt() * x0     # closed form says this is N(0, (1 - alpha_bar) I)
print(f"chain vs closed form at t=500 (100,000 points): residual variance per axis"
      f" {resid.var(0).numpy().round(4)}, formula {1 - ABAR[499]:.4f}")
xT = ABAR[-1].sqrt() * x0 + (1 - ABAR[-1]).sqrt() * torch.randn_like(x0)
c = torch.cov(xT.T)
print(f"x_T: mean {xT.mean(0).numpy().round(3)}, var {c.diag().numpy().round(3)}, cov_xy {c[0, 1]:.3f}")

# ---- 2. Reference rows ---------------------------------------------------------
print("\nreference rows (10,000 samples each):")
print(f"  real data                    {fmt(evaluate(sample_data(N, gen)))}")
print(f"  Gaussian, same mean and cov  {fmt(evaluate(2**0.5 * torch.randn(N, 2, generator=gen)))}")
t0 = time.time()
x_exact = ddpm_sample(exact_eps, N, gen)
print(f"  DDPM with the exact eps      {fmt(evaluate(x_exact))}  ({time.time() - t0:.1f}s)")
for steps in (20, 10, 5):
    print(f"  DDIM {steps:2d} steps, exact eps   {fmt(evaluate(ddim_sample(exact_eps, N, gen, steps)))}")

# ---- 3. Train -------------------------------------------------------------------
t0 = time.time()
net, last_loss = train_eps_model(steps=3000)
print(f"\ntrained 3000 steps in {time.time() - t0:.1f}s, last batch loss {last_loss:.3f}")
eps_net = lambda x, i: net(x, torch.full((len(x),), i / T))

with torch.no_grad():
    print("learned vs exact noise prediction, E||eps_net - eps_exact||^2 at x_t ~ q_t:")
    for t in (10, 100, 300, 600, 1000):
        i = t - 1
        xt = ABAR[i].sqrt() * sample_data(N, gen) + (1 - ABAR[i]).sqrt() * torch.randn(N, 2, generator=gen)
        err = (eps_net(xt, i) - exact_eps(xt, i)).pow(2).sum(1).mean().item()
        print(f"  t={t:4d}  error {err:.4f}")

# ---- 4. Samplers -----------------------------------------------------------------
print("\nsamples from the trained network:")
t0 = time.time()
x = ddpm_sample(eps_net, N, gen)
print(f"  DDPM, 1000 steps   {fmt(evaluate(x))}  ({time.time() - t0:.1f}s)")
for steps in (100, 20, 10, 5):
    t0 = time.time()
    x = ddim_sample(eps_net, N, gen, steps)
    print(f"  DDIM, {steps:4d} steps   {fmt(evaluate(x))}  ({time.time() - t0:.2f}s)")
