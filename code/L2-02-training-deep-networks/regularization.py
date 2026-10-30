"""Dropout, weight decay and early stopping with a large MLP on 300 training digits,
with clean labels and with 30% of training and validation labels replaced at random.
The test labels are always clean, so test accuracy measures what we actually want."""
import copy

import torch
from torch import nn

from common import digits, evaluate, fit, mlp

# Dropout itself: in train mode zero a fraction p and scale survivors by 1/(1-p).
drop = nn.Dropout(p=0.5)
torch.manual_seed(0)
print("train mode:", drop(torch.ones(10)).tolist())
drop.eval()
print("eval mode: ", drop(torch.ones(10)).tolist())


def corrupt(y, rate, seed):
    g = torch.Generator().manual_seed(seed)
    y = y.clone()
    flip = torch.rand(len(y), generator=g) < rate
    y[flip] = torch.randint(0, 10, (int(flip.sum()),), generator=g)   # may redraw the true label
    return y


def run(data, dropout, wd, seed, epochs=80):
    torch.manual_seed(seed)
    model = mlp([64, 256, 256, 10], dropout=dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=wd)
    best_loss, best_state, best_epoch = float("inf"), None, 0
    for epoch in range(1, epochs + 1):
        train_loss, val_loss, _ = fit(model, data, epochs=1, opt=opt, batch_size=32, seed=1000 * seed + epoch)[0]
        if val_loss < best_loss:                       # early stopping: keep the best-val checkpoint
            best_loss, best_state, best_epoch = val_loss, copy.deepcopy(model.state_dict()), epoch
    final_acc = evaluate(model, *data["test"])[1]
    model.load_state_dict(best_state)
    return train_loss, final_acc, evaluate(model, *data["test"])[1], best_epoch


print("\nMLP 64-256-256-10, AdamW(lr=1e-3), 80 epochs, batch 32, 3 seeds (mean test accuracy)")
print(f"{'labels':<10} {'setting':<21} {'final train loss':>16} {'final test acc':>15} "
      f"{'early-stop test acc':>20} {'stop epochs':>13}")
for rate, settings in [(0.0, ["none", "dropout 0.5 + wd 1.0"]),
                       (0.3, ["none", "weight decay 1.0", "dropout 0.5", "dropout 0.5 + wd 1.0"])]:
    data = digits(n_train=300)
    for split, seed in [("train", 1), ("val", 2)]:
        X, y = data[split]
        data[split] = (X, corrupt(y, rate, seed))
    for name in settings:
        dropout = 0.5 if "dropout" in name else 0.0
        wd = 1.0 if "wd" in name or "weight decay" in name else 0.0
        rows = [run(data, dropout, wd, seed) for seed in range(3)]
        avg = [sum(r[i] for r in rows) / 3 for i in range(3)]
        label = "clean" if rate == 0 else f"{rate:.0%} noisy"
        print(f"{label:<10} {name:<21} {avg[0]:>16.3f} {avg[1]:>15.3f} {avg[2]:>20.3f} "
              f"{str([r[3] for r in rows]):>13}")
