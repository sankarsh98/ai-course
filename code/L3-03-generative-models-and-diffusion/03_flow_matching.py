"""Flow matching (rectified-flow path) versus DDIM on the same 8-mode toy.

Flow matching: x_t = (1 - t) * z + t * x1, z ~ N(0, I), x1 ~ data; regress v_theta(x_t, t)
onto x1 - z. Sampling integrates dx/dt = v_theta(x, t) from t = 0 (noise) to t = 1 (data)
with Euler steps. Both models use the same MLP, optimizer and number of training steps.
DDIM is run twice: on evenly spaced DDPM steps, and on the DDPM steps whose signal-to-noise
ratio matches the flow's evenly spaced times (alpha_bar / (1 - alpha_bar) = t^2 / (1 - t)^2).
"""
import time
import torch
from toy2d import ABAR, T, Net, evaluate, fmt, sample_data, ddim_sample, train_eps_model


def train_flow(steps=3000, batch=1024, seed=0):
    torch.manual_seed(seed)
    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        x1, z = sample_data(batch), torch.randn(batch, 2)
        t = torch.rand(batch)
        xt = (1 - t[:, None]) * z + t[:, None] * x1
        loss = (net(xt, t) - (x1 - z)).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return net, loss.item()


@torch.no_grad()
def euler_sample(net, n, gen, steps):
    x = torch.randn(n, 2, generator=gen)
    for j in range(steps):
        x = x + net(x, torch.full((n,), j / steps)) / steps
    return x


def snr_matched_steps(steps):
    """DDPM indices whose SNR is closest to that of the flow times 0, 1/steps, 2/steps, ..."""
    log_snr = (ABAR / (1 - ABAR)).log()
    idx = [T - 1]                                   # t = 0 is pure noise: start at the last step
    for j in range(1, steps):
        t = j / steps
        idx.append(int((log_snr - 2 * torch.log(torch.tensor(t / (1 - t)))).abs().argmin()))
    return idx


gen = torch.Generator().manual_seed(3)
N = 10000
t0 = time.time()
fm, fm_loss = train_flow()
print(f"flow matching: trained in {time.time() - t0:.1f}s, last batch loss {fm_loss:.3f}"
      " (cannot reach 0: many (z, x1) pairs pass through the same x_t)")
t0 = time.time()
dm, _ = train_eps_model()
print(f"DDPM noise model: trained in {time.time() - t0:.1f}s")
eps_fn = lambda x, i: dm(x, torch.full((len(x),), i / T))

print("\nsteps  sampler")
for steps in (2, 5, 10, 20, 100):
    print(f"  {steps:3d}  flow + Euler       {fmt(evaluate(euler_sample(fm, N, gen, steps)))}")
    print(f"  {steps:3d}  DDIM, even steps   {fmt(evaluate(ddim_sample(eps_fn, N, gen, steps)))}")
    idx = snr_matched_steps(steps)
    print(f"  {steps:3d}  DDIM, matched SNR  {fmt(evaluate(ddim_sample(eps_fn, N, gen, steps, idx)))}")
print("\nDDPM indices used by 'matched SNR' at 5 steps:", snr_matched_steps(5))
