"""Autoencoders on the 8x8 digits: linear AE vs PCA, nonlinear AE, denoising AE.

1. A linear autoencoder trained with squared error learns the PCA subspace
   (Baldi and Hornik, 1989), though not the PCA axes themselves.
2. A nonlinear autoencoder reconstructs better at the same code size.
3. Lower reconstruction error does not guarantee a more useful code:
   we check each code with a linear probe (labels used only for evaluation).
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

torch.manual_seed(0)
X, y = load_digits(return_X_y=True)
X = (X / 16.0).astype(np.float32)               # pixels in [0, 1]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
mean = Xtr.mean(0)
K = 8                                           # code size: 64 pixels -> 8 numbers


def train_ae(encoder, decoder, noise=0.0, epochs=300):
    params = list(encoder.parameters()) + list(decoder.parameters())
    opt = torch.optim.Adam(params, lr=3e-3)
    xt = torch.from_numpy(Xtr - mean)
    for epoch in range(epochs):
        for b in torch.randperm(len(xt)).split(128):
            x = xt[b]
            # Denoising: corrupt the input, reconstruct the clean target.
            x_in = x * (torch.rand_like(x) > noise) if noise else x
            loss = ((decoder(encoder(x_in)) - x) ** 2).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return encoder, decoder


def recon_error(encoder, decoder, X_):
    with torch.no_grad():
        x = torch.from_numpy(X_ - mean)
        return ((decoder(encoder(x)) - x) ** 2).sum(1).mean().item()


def probe(Z_tr, Z_te):
    """Linear probe: logistic regression on the frozen code."""
    clf = LogisticRegression(max_iter=3000).fit(Z_tr, ytr)
    return clf.score(Z_te, yte)


def codes(encoder):
    with torch.no_grad():
        return [encoder(torch.from_numpy(A - mean)).numpy() for A in (Xtr, Xte)]


# ---- PCA baseline ---------------------------------------------------------------
pca = PCA(n_components=K).fit(Xtr)
pca_err = (((pca.inverse_transform(pca.transform(Xte)) - Xte) ** 2).sum(1)).mean()

# ---- 1. Linear autoencoder ------------------------------------------------------
lin_enc, lin_dec = train_ae(nn.Linear(64, K, bias=False), nn.Linear(K, 64, bias=False))
# Compare the subspace spanned by the decoder's columns with PCA's subspace:
# cosines of the principal angles between them are the singular values of Q1^T Q2.
Q1, _ = np.linalg.qr(lin_dec.weight.detach().numpy())
Q2, _ = np.linalg.qr(pca.components_.T)
cosines = np.linalg.svd(Q1.T @ Q2, compute_uv=False)
axis_cos = np.abs(Q1.T @ pca.components_.T).max(0)    # best match per PCA axis

# ---- 2. Nonlinear and denoising autoencoders ------------------------------------
def mlp_ae():
    enc = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, K))
    dec = nn.Sequential(nn.Linear(K, 128), nn.ReLU(), nn.Linear(128, 64))
    return enc, dec


torch.manual_seed(1)
mlp_enc, mlp_dec = train_ae(*mlp_ae())
torch.manual_seed(1)
den_enc, den_dec = train_ae(*mlp_ae(), noise=0.3)

print(f"test reconstruction error (sum of squared pixel errors), code size {K}")
print(f"  PCA                    {pca_err:.3f}")
print(f"  linear AE              {recon_error(lin_enc, lin_dec, Xte):.3f}")
print(f"  nonlinear AE           {recon_error(mlp_enc, mlp_dec, Xte):.3f}")
print(f"  denoising AE, 30% drop {recon_error(den_enc, den_dec, Xte):.3f}")
print(f"  predict the mean image {((Xte - mean) ** 2).sum(1).mean():.3f}")
print(f"\nlinear AE vs PCA: cosines of principal angles between subspaces:\n  "
      f"{np.round(cosines, 3)}")
print(f"  but individual axes do not match PCA's: best |cos| per PCA axis\n  "
      f"{np.round(axis_cos, 2)}")

print("\nlinear probe accuracy on the 8-number code (labels only for evaluation)")
print(f"  raw pixels (64 numbers) {probe(Xtr, Xte):.3f}")
print(f"  PCA                     {probe(pca.transform(Xtr), pca.transform(Xte)):.3f}")
print(f"  linear AE               {probe(*codes(lin_enc)):.3f}")
print(f"  nonlinear AE            {probe(*codes(mlp_enc)):.3f}")
print(f"  denoising AE            {probe(*codes(den_enc)):.3f}")
torch.manual_seed(2)
print(f"  random untrained MLP    {probe(*codes(mlp_ae()[0])):.3f}")
