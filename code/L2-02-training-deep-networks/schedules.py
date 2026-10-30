"""Learning-rate schedules on a fixed budget, and a learning-rate range test."""
import math

import torch
from torch.optim.lr_scheduler import LambdaLR

from common import digits, fit, mlp

data = digits()
EPOCHS, BATCH = 15, 64
STEPS = EPOCHS * math.ceil(len(data["train"][1]) / BATCH)   # the scheduler steps once per batch
WARMUP = STEPS // 10


def cosine(step):
    return 0.5 * (1 + math.cos(math.pi * min(step, STEPS) / STEPS))


SCHEDULES = {                         # multiplier on the base learning rate at each step
    "constant": lambda s: 1.0,
    "step (x0.1 at 50%, 75%)": lambda s: 0.1 ** ((s >= STEPS // 2) + (s >= 3 * STEPS // 4)),
    "cosine": cosine,
    "warmup 10% + cosine": lambda s: s / WARMUP if s < WARMUP else cosine((s - WARMUP) * STEPS / (STEPS - WARMUP)),
}

print(f"MLP 64-128-128-10, SGD(momentum=0.9), {EPOCHS} epochs = {STEPS} steps, mean of 3 seeds")
print(f"{'schedule':<25} {'base lr':>7} {'final train loss':>17} {'val loss':>9} {'val acc':>8}")
for base_lr in [0.03, 0.1, 0.3]:
    for name, f in SCHEDULES.items():
        results = []
        for seed in range(3):
            torch.manual_seed(seed)
            model = mlp([64, 128, 128, 10])
            opt = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=0.9)
            results.append(fit(model, data, EPOCHS, opt, scheduler=LambdaLR(opt, f), batch_size=BATCH, seed=seed)[-1])
        avg = [sum(r[i] for r in results) / 3 for i in range(3)]
        print(f"{name:<25} {base_lr:>7} {avg[0]:>17.4f} {avg[1]:>9.4f} {avg[2]:>8.3f}")

# Range test: raise the learning rate exponentially every step and watch the loss.
torch.manual_seed(0)
model = mlp([64, 128, 128, 10])
opt = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
n_steps = 120
sched = LambdaLR(opt, lambda s: 10 ** (5 * s / n_steps))      # 1e-4 -> 10 over 120 steps
X, y = data["train"]
g = torch.Generator().manual_seed(0)
smooth = None
print("\nrange test (loss smoothed with an exponential moving average, factor 0.9)")
for step in range(n_steps):
    idx = torch.randint(0, len(y), (BATCH,), generator=g)
    lr = opt.param_groups[0]["lr"]
    opt.zero_grad()
    loss = torch.nn.functional.cross_entropy(model(X[idx]), y[idx])
    loss.backward()
    opt.step()
    sched.step()
    smooth = loss.item() if smooth is None else 0.9 * smooth + 0.1 * loss.item()
    if step % 12 == 0 or not math.isfinite(smooth) or smooth > 10:
        print(f"  lr {lr:9.2e}  loss {smooth:.3f}")
        if not math.isfinite(smooth) or smooth > 10:
            print("  diverged: stop")
            break
