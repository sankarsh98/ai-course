"""What a Frechet distance (the 'FD' in FID) can and cannot see, on the 8-mode toy.

FD between two sample sets = ||mu1 - mu2||^2 + Tr(S1 + S2 - 2 (S1 S2)^(1/2)), computed on
some feature vector of each sample. FID uses Inception-v3 features of images; here we use
either the raw 2D coordinates or 64 fixed random Fourier features (a stand-in extractor).
"""
import numpy as np
import torch
from scipy.linalg import sqrtm
from toy2d import CENTERS, SIGMA, K, ddpm_sample, evaluate, exact_eps, sample_data

gen = torch.Generator().manual_seed(5)
W = torch.randn(2, 64, generator=gen) / 0.3      # frequencies: sensitive to structure at scale ~0.3
B = 2 * np.pi * torch.rand(64, generator=gen)


def features(x, kind):
    return x if kind == "raw" else torch.cos(x @ W + B)


def fd(x, y, kind):
    fx, fy = features(x, kind).double().numpy(), features(y, kind).double().numpy()
    m1, m2 = fx.mean(0), fy.mean(0)
    s1, s2 = np.cov(fx, rowvar=False), np.cov(fy, rowvar=False)
    covmean = sqrtm(s1 @ s2).real
    return float(((m1 - m2) ** 2).sum() + np.trace(s1 + s2 - 2 * covmean))


def mixture(n, sigma=SIGMA, modes=range(K)):
    modes = torch.tensor(list(modes))
    k = modes[torch.randint(0, len(modes), (n,), generator=gen)]
    return CENTERS[k] + sigma * torch.randn(n, 2, generator=gen)


n = 2000
test = sample_data(10000, gen)                   # the 'real' reference set
train = sample_data(500, gen)                    # a small training set
copies = train[torch.randint(0, len(train), (n,), generator=gen)]
exact_empirical = ddpm_sample(lambda x, i: exact_eps(x, i, centers=train, sigma0=0.0), n, gen)

candidates = {
    "fresh real data": sample_data(n, gen),
    "Gaussian, matched mean/cov": torch.randn(n, 2, generator=gen) * 2.005**0.5,
    "7 of 8 modes (one dropped)": mixture(n, modes=range(7)),
    "modes 2x too wide": mixture(n, sigma=2 * SIGMA),
    "modes 2x too narrow": mixture(n, sigma=SIGMA / 2),
    "copies of 500 train points": copies,
    "diffusion, exact score of train set": exact_empirical,
}
print(f"{'samples (n=2000) vs 10,000 test points':<38} {'FD raw':>8} {'FD feat':>8} {'in_mode':>8}"
      f" {'modes':>6} {'NN dist':>8}")
for name, x in candidates.items():
    nn_dist = torch.cdist(x, train).min(1).values.median().item()   # distance to closest train point
    m = evaluate(x)
    print(f"{name:<38} {fd(x, test, 'raw'):>8.4f} {fd(x, test, 'feat'):>8.4f} {m['in_mode']:>8.3f}"
          f" {m['modes']:>4}/8 {nn_dist:>8.4f}")

print("\nFD between two independent sets of REAL data, by sample size (features):")
for m_size in (50, 200, 1000, 5000):
    vals = [fd(sample_data(m_size, gen), sample_data(m_size, gen), "feat") for _ in range(5)]
    print(f"  n={m_size:<5} mean {np.mean(vals):.4f}  (min {min(vals):.4f}, max {max(vals):.4f})")
