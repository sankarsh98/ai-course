"""A VAE and a GAN on the same 8-mode toy, scored with the same metrics as the diffusion model.

VAE: 2D latent, Gaussian decoder with a FIXED output std sigma_x. The negative ELBO is
    ||x - decoder(z)||^2 / (2 sigma_x^2) + KL(q(z|x) || N(0, I))   (+ a constant),
so sigma_x sets how much reconstruction counts relative to the KL term.
GAN: non-saturating loss, same training budget for 3 seeds, checked during training.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from toy2d import evaluate, fmt, sample_data


def mlp(d_in, d_out, h=128):
    return nn.Sequential(nn.Linear(d_in, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, d_out))


def train_vae(sigma_x, steps=3000, batch=512, seed=0):
    torch.manual_seed(seed)
    enc, dec = mlp(2, 4), mlp(2, 2)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=1e-3)
    for _ in range(steps):
        x = sample_data(batch)
        mu, logvar = enc(x).chunk(2, dim=1)
        z = mu + (0.5 * logvar).exp() * torch.randn_like(mu)        # reparameterization trick
        rec = (x - dec(z)).pow(2).sum(1) / (2 * sigma_x**2)
        kl = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar).sum(1)  # KL to N(0, I), closed form
        loss = (rec + kl).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        x = sample_data(10000)
        mu, logvar = enc(x).chunk(2, dim=1)
        kl = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar).sum(1).mean().item()
        rmse = (x - dec(mu)).pow(2).sum(1).mean().sqrt().item()
        samples = dec(torch.randn(10000, 2))     # decoder means; adding sigma_x noise only blurs more
    return kl, rmse, samples


def train_gan(seed, steps=3000, batch=256, checkpoints=(500, 1000, 2000, 3000)):
    torch.manual_seed(seed)
    G, D = mlp(2, 2), mlp(2, 1)
    opt_g = torch.optim.Adam(G.parameters(), lr=1e-3, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=1e-3, betas=(0.5, 0.999))
    report = []
    for step in range(1, steps + 1):
        real, fake = sample_data(batch), G(torch.randn(batch, 2))
        # Discriminator: real -> 1, fake -> 0.
        d_loss = (F.binary_cross_entropy_with_logits(D(real), torch.ones(batch, 1))
                  + F.binary_cross_entropy_with_logits(D(fake.detach()), torch.zeros(batch, 1)))
        opt_d.zero_grad()
        d_loss.backward()
        opt_d.step()
        # Generator, non-saturating form: maximize log D(G(z)).
        g_loss = F.binary_cross_entropy_with_logits(D(fake), torch.ones(batch, 1))
        opt_g.zero_grad()
        g_loss.backward()
        opt_g.step()
        if step in checkpoints:
            with torch.no_grad():
                m = evaluate(G(torch.randn(10000, 2)))
            report.append(f"{step}: {m['modes']}/8 modes, in_mode {m['in_mode']:.2f}")
    return report


print("VAE (decoder std sigma_x fixed):")
for sigma_x in (0.05, 0.1, 0.3, 1.0):
    kl, rmse, x = train_vae(sigma_x)
    print(f"  sigma_x={sigma_x:<4}  KL {kl:5.2f} nats  recon RMSE {rmse:.3f}  samples: {fmt(evaluate(x))}")

print("\nGAN (modes covered and in_mode at training checkpoints):")
for seed in range(3):
    print(f"  seed {seed}:  " + " | ".join(train_gan(seed)))
