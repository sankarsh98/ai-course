"""Does a model recognise a shape at a position it never saw in training?
Train on shapes in the LEFT part of the image, test on the left and on the RIGHT part."""
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(4)  # fixed thread count, so runs on a shared machine are comparable

SIZE, NOISE = 24, 0.3
T = {  # 5x5 templates, four classes
    "plus":   ["..#..", "..#..", "#####", "..#..", "..#.."],
    "cross":  ["#...#", ".#.#.", "..#..", ".#.#.", "#...#"],
    "square": ["#####", "#...#", "#...#", "#...#", "#####"],
    "tee":    ["#####", "..#..", "..#..", "..#..", "..#.."],
}
TEMPLATES = [np.array([[c == "#" for c in row] for row in t], dtype=np.float32) for t in T.values()]


def make_data(n, cols, rng):
    """n noisy images; each shape's left column is drawn from `cols`, its top row anywhere."""
    X = rng.normal(0, NOISE, size=(n, 1, SIZE, SIZE)).astype(np.float32)
    y = rng.integers(0, len(TEMPLATES), size=n)
    for i in range(n):
        r, c = rng.integers(0, SIZE - 5 + 1), rng.choice(cols)
        X[i, 0, r:r + 5, c:c + 5] += TEMPLATES[y[i]]
    return torch.tensor(X), torch.tensor(y)


def mlp():
    return nn.Sequential(nn.Flatten(), nn.Linear(SIZE * SIZE, 128), nn.ReLU(), nn.Linear(128, 4))


def convs():
    return [nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2)]


def cnn_flatten():   # convolutional features, but a dense head that sees absolute positions
    return nn.Sequential(*convs(), nn.Flatten(), nn.Linear(32 * 12 * 12, 4))


def cnn_global_pool():  # global max pool: "is the feature anywhere?" then classify
    return nn.Sequential(*convs(), nn.AdaptiveMaxPool2d(1), nn.Flatten(), nn.Linear(32, 4))


def train(model, X, y, steps=1500, seed=0):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(steps):
        idx = torch.randint(0, len(X), (64,), generator=g)
        loss = nn.functional.cross_entropy(model(X[idx]), y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


@torch.no_grad()
def accuracy(model, X, y):
    return (model(X).argmax(1) == y).float().mean().item()


rng = np.random.default_rng(0)
left, right = np.arange(0, 9), np.arange(12, SIZE - 5 + 1)
Xtr, ytr = make_data(3000, left, rng)
Xl, yl = make_data(1000, left, rng)
Xr, yr = make_data(1000, right, rng)
print(f"train: shapes start in columns {left.min()}-{left.max()}; "
      f"right test: columns {right.min()}-{right.max()}")
print(f"{'model':18s} {'params':>7s} {'train acc':>9s} {'left test':>9s} {'right test':>10s}")
for name, make in [("MLP", mlp), ("CNN + flatten", cnn_flatten), ("CNN + global pool", cnn_global_pool)]:
    torch.manual_seed(0)
    model = train(make(), Xtr, ytr)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"{name:18s} {n_par:7,d} {accuracy(model, Xtr, ytr):9.3f} {accuracy(model, Xl, yl):9.3f} "
          f"{accuracy(model, Xr, yr):10.3f}")

# Equivariance check: shifting the input shifts the conv feature map by the same amount.
conv = nn.Sequential(*convs()[:4])
x = Xl[:1]
shifted = torch.roll(x, shifts=3, dims=3)
f, fs = conv(x), conv(shifted)
inner = (slice(None), slice(None), slice(None), slice(6, -6))  # skip columns near the wrap and the padding
err = (torch.roll(f, 3, dims=3) - fs)[inner].abs().max().item()
print(f"\nconv(shift(x)) vs shift(conv(x)), interior columns: max abs difference {err:.1e}")
