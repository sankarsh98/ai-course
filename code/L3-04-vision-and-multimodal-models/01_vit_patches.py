"""Patch embeddings and a tiny vision transformer on the 8x8 digits (bundled with scikit-learn).

1. Patch embedding = a linear map on flattened patches = a convolution with stride = kernel.
2. Token counts and attention cost for common image and patch sizes.
3. Tiny ViT (with and without position embeddings) vs a small CNN, at two training-set sizes.
4. Without position embeddings a ViT cannot tell where a patch came from: shuffling patches
   leaves its prediction unchanged.
"""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

torch.set_num_threads(4)
torch.manual_seed(0)

# ---- 1. Patch embedding three ways ------------------------------------------------
P, C, D = 4, 3, 32
img = torch.randn(2, C, 16, 16)
conv = nn.Conv2d(C, D, kernel_size=P, stride=P)
patches = img.unfold(2, P, P).unfold(3, P, P)                  # (B, C, 4, 4, P, P)
patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(2, 16, C * P * P)
linear_out = patches @ conv.weight.reshape(D, -1).T + conv.bias
conv_out = conv(img).flatten(2).transpose(1, 2)                # (B, 16 tokens, D)
print(f"patchify + linear vs conv(stride=kernel): max abs diff {(linear_out - conv_out).abs().max():.2e}")

# ---- 2. Token counts --------------------------------------------------------------
print("\nimage side, patch -> tokens, attention scores per head per layer")
for side, p in ((224, 16), (224, 14), (336, 14), (224, 8), (1024, 16)):
    n = (side // p) ** 2
    print(f"  {side:4d}px, {p:2d}px patches -> {n:5d} tokens, {n * n:>12,d} scores")

# ---- 3. Tiny ViT vs CNN on digits -------------------------------------------------
X, y = load_digits(return_X_y=True)
X = torch.tensor(X / 16.0, dtype=torch.float32).view(-1, 1, 8, 8)
y = torch.tensor(y)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)


class TinyViT(nn.Module):
    def __init__(self, use_pos=True, patch=2, dim=64, depth=2, heads=4):
        super().__init__()
        n = (8 // patch) ** 2
        self.embed = nn.Conv2d(1, dim, patch, stride=patch)    # patch embedding
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))         # [class] token, as in ViT
        self.pos = nn.Parameter(0.02 * torch.randn(1, n + 1, dim)) if use_pos else None
        layer = nn.TransformerEncoderLayer(dim, heads, 2 * dim, dropout=0.0,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 10))

    def tokens(self, x):
        return self.embed(x).flatten(2).transpose(1, 2)        # (B, N, dim)

    def forward(self, x=None, tok=None):
        tok = self.tokens(x) if tok is None else tok
        h = torch.cat([self.cls.expand(len(tok), -1, -1), tok], dim=1)
        if self.pos is not None:
            h = h + self.pos
        return self.head(self.encoder(h)[:, 0])


def small_cnn():
    return nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Flatten(), nn.Linear(64 * 4, 10))


def train(model, Xs, ys, epochs, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.05)
    steps = epochs * ((len(Xs) + 63) // 64)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps)
    for _ in range(epochs):
        for b in torch.randperm(len(Xs)).split(64):
            loss = F.cross_entropy(model(Xs[b]), ys[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    return model


def acc(model, Xs, ys):
    with torch.no_grad():
        return (model(Xs).argmax(1) == ys).float().mean().item()


n_params = lambda m: sum(p.numel() for p in m.parameters())
print(f"\nparameters: ViT {n_params(TinyViT()):,}, CNN {n_params(small_cnn()):,}")
print(f"{'train images':>12} {'ViT + pos':>10} {'ViT, no pos':>12} {'CNN':>6}   (test accuracy, mean of 3 seeds)")
models = {}
t0 = time.time()
for n_train in (200, len(Xtr)):
    epochs = 60 if n_train == 200 else 30                       # similar number of updates
    row = []
    for make in (lambda: TinyViT(True), lambda: TinyViT(False), small_cnn):
        scores = []
        for seed in range(3):
            torch.manual_seed(seed)
            m = train(make(), Xtr[:n_train], ytr[:n_train], epochs, seed)
            scores.append(acc(m, Xte, yte))
        row.append(np.mean(scores))
        models.setdefault(n_train, []).append(m)
    print(f"{n_train:>12} {row[0]:>10.3f} {row[1]:>12.3f} {row[2]:>6.3f}")
print(f"(training took {time.time() - t0:.0f}s)")

# ---- 4. Shuffle the 16 patch tokens with one fixed permutation ---------------------
vit_pos, vit_nopos, _ = models[len(Xtr)]
perm = torch.randperm(16)
with torch.no_grad():
    for name, m in (("ViT + pos", vit_pos), ("ViT, no pos", vit_nopos)):
        tok = m.tokens(Xte)
        same = (m(tok=tok).argmax(1) == m(tok=tok[:, perm]).argmax(1)).float().mean()
        acc_shuf = (m(tok=tok[:, perm]).argmax(1) == yte).float().mean()
        print(f"{name:<12} (seed 2) accuracy {acc(m, Xte, yte):.3f}; patches shuffled:"
              f" same prediction {same:.3f}, accuracy {acc_shuf:.3f}")
