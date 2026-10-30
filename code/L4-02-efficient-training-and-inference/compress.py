"""Distillation and pruning on scikit-learn's digits (8x8 images, 10 classes).

1. Distillation: a small student learns from a big teacher's soft targets
   [Hinton et al., 2015], on the labeled set alone or on a larger transfer
   set of unlabeled images, compared with training on labels, over 5 seeds.
2. Magnitude pruning of the teacher, with and without fine-tuning.
3. Zeros in a dense matrix cost the same as any other number: dense vs sparse matmul.
Timings depend on the machine and its load.
"""
import time
import warnings

import torch
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

torch.set_num_threads(4)
warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta")
X, y = load_digits(return_X_y=True)
X = torch.tensor(X / 16.0, dtype=torch.float32)
y = torch.tensor(y)
# 300 labeled images, 800 unlabeled ones (labels thrown away), 697 for testing.
Xl, Xrest, yl, yrest = train_test_split(X, y, train_size=300, random_state=0, stratify=y)
Xu, Xte, _, yte = train_test_split(Xrest, yrest, train_size=800, random_state=0, stratify=yrest)
Xall = torch.cat([Xl, Xu])
n_lab = len(Xl)


def mlp(width):
    return torch.nn.Sequential(torch.nn.Linear(64, width), torch.nn.ReLU(),
                               torch.nn.Linear(width, width), torch.nn.ReLU(),
                               torch.nn.Linear(width, 10))


def train(model, Xs, loss_fn, epochs=150, lr=3e-3, seed=0):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        for idx in torch.randperm(len(Xs), generator=g).split(64):
            aug = Xs[idx] + 0.05 * torch.randn(len(idx), 64, generator=g)   # light noise
            opt.zero_grad()
            loss_fn(model(aug), idx).backward()
            opt.step()
    return model


@torch.no_grad()
def acc(model):
    return (model(Xte).argmax(1) == yte).float().mean().item()


def nparams(model):
    return sum((p != 0).sum().item() for p in model.parameters())


def label_loss(out, idx):          # indices < n_lab are labeled examples
    return F.cross_entropy(out, yl[idx])


torch.manual_seed(0)
teacher = train(mlp(512), Xl, label_loss)
with torch.no_grad():
    t_logits = teacher(Xall)       # the teacher labels everything, including Xu
print(f"teacher: {nparams(teacher):,} parameters, trained on {n_lab} labels,"
      f" test accuracy {acc(teacher):.3f}")


def kd_loss(T, alpha=0.9):
    def loss(out, idx):
        soft = F.kl_div(F.log_softmax(out / T, 1), F.softmax(t_logits[idx] / T, 1),
                        reduction="batchmean") * T * T      # T^2 keeps gradient scale
        lab = idx < n_lab                                   # hard loss only where labels exist
        hard = F.cross_entropy(out[lab], yl[idx[lab]]) if lab.any() else 0.0
        return alpha * soft + (1 - alpha) * hard
    return loss


def pseudo_label_loss(out, idx):   # teacher's argmax as if it were a label
    return F.cross_entropy(out, t_logits[idx].argmax(1))


print("\n1. student with 16 hidden units per layer, 5 seeds: mean (min-max)")
for name, Xs, fn in [("labels only (300)", Xl, label_loss),
                     ("distilled, T=4, same 300", Xl, kd_loss(4.0)),
                     ("hard pseudo-labels, 1100", Xall, pseudo_label_loss),
                     ("distilled, T=1, 1100", Xall, kd_loss(1.0)),
                     ("distilled, T=4, 1100", Xall, kd_loss(4.0))]:
    accs = []
    for seed in range(5):
        torch.manual_seed(seed)
        accs.append(acc(train(mlp(16), Xs, fn, seed=seed)))
    t = torch.tensor(accs)
    print(f"   {name:<26} {t.mean():.3f}  ({t.min():.3f}-{t.max():.3f})")
print(f"   student size: {nparams(mlp(16)):,} parameters")

print("\n2. global magnitude pruning of the teacher's weight matrices")
state = {k: v.clone() for k, v in teacher.state_dict().items()}
weights = [m.weight for m in teacher if isinstance(m, torch.nn.Linear)]
for frac in (0.5, 0.8, 0.9, 0.95, 0.98, 0.99):
    teacher.load_state_dict(state)
    allw = torch.cat([w.detach().abs().flatten() for w in weights])
    cut = allw.kthvalue(int(frac * allw.numel())).values
    masks = [(w.detach().abs() > cut).float() for w in weights]
    with torch.no_grad():
        for w, mk in zip(weights, masks):
            w.mul_(mk)
    before = acc(teacher)

    # fine-tune 20 epochs, re-applying the mask after every step via a hook
    hooks = [w.register_hook(lambda g, mk=mk: g * mk) for w, mk in zip(weights, masks)]
    train(teacher, Xl, label_loss, epochs=20, lr=1e-3)
    for h in hooks:
        h.remove()
    print(f"   {frac:4.0%} of weights removed: accuracy {before:.3f} -> {acc(teacher):.3f}"
          f" after fine-tuning   ({nparams(teacher):,} nonzero parameters)")

print("\n3. 2048 x 2048 weight times 2048 x 64 activations (median of 5)")


def timeit(f):
    ts = []
    for _ in range(5):
        t0 = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t0)
    return sorted(ts)[2] * 1e3


torch.manual_seed(0)
W, A = torch.randn(2048, 2048), torch.randn(2048, 64)
print(f"   dense, no zeros            {timeit(lambda: W @ A):7.2f} ms")
for frac in (0.5, 0.9, 0.99):
    Wz = W * (torch.rand_like(W) > frac)
    Ws = Wz.to_sparse_csr()
    print(f"   {frac:4.0%} zeros: dense {timeit(lambda: Wz @ A):6.2f} ms   "
          f"sparse CSR {timeit(lambda: Ws @ A):6.2f} ms")
