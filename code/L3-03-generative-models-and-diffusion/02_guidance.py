"""Classifier-free guidance on the 8-mode toy.

The label is the parity of the mode: class 0 = the 4 even modes, class 1 = the 4 odd ones,
so each class is itself a 4-mode distribution. One network is trained both conditionally
and unconditionally (the label is replaced by a "no class" token 10% of the time).
At sampling time
    eps = (1 + w) * eps(x, t, c) - w * eps(x, t, none)        (Ho and Salimans' convention)
so w = 0 is the plain conditional model. We ask for class 0 and measure:
  hit     fraction of samples whose nearest mode belongs to class 0
  in_mode fraction within 3 sigma of some mode
  kl4     KL(histogram over the 4 requested modes || uniform), mode balance
  spread  within-mode RMS distance relative to real data (diversity)
  radius  median distance from the origin (the real modes sit at 2.00)
"""
import torch
from toy2d import CENTERS, T, ddpm_sample, evaluate, train_eps_model

gen = torch.Generator().manual_seed(2)
NONE = 2                                          # index of the "no class" token
net, _ = train_eps_model(steps=1500, n_classes=2, p_uncond=0.1, label_fn=lambda k: k % 2)
N = 4000


def guided(w, c):
    def eps_fn(x, i):
        t01 = torch.full((len(x),), i / T)
        e_c = net(x, t01, torch.full((len(x),), c))
        if w == 0:
            return e_c
        return (1 + w) * e_c - w * net(x, t01, torch.full((len(x),), NONE))
    return eps_fn


x = ddpm_sample(lambda x, i: net(x, torch.full((len(x),), i / T), torch.full((len(x),), NONE)), N, gen)
m = evaluate(x)
print(f"unconditional branch: in_mode {m['in_mode']:.3f}  modes {m['modes']}/8  mode_kl {m['mode_kl']:.3f}\n")
print("asking for class 0 (modes 0, 2, 4, 6):")
print(f"{'w':>5} {'hit':>6} {'in_mode':>8} {'kl4':>6} {'spread':>7} {'radius':>7}")
for w in (0.0, 0.5, 1.0, 3.0, 10.0):
    x = ddpm_sample(guided(w, 0), N, gen)
    near = torch.cdist(x, CENTERS).argmin(1)
    hit = near % 2 == 0
    p = torch.bincount(near[hit] // 2, minlength=4).float()
    p = p / p.sum()
    kl4 = (p[p > 0] * (4 * p[p > 0]).log()).sum()
    m = evaluate(x)
    print(f"{w:>5.1f} {hit.float().mean():>6.3f} {m['in_mode']:>8.3f} {kl4:>6.3f} {m['spread']:>7.2f} {x.norm(dim=1).median():>7.3f}")
