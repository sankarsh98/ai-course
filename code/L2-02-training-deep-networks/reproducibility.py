"""Reproducibility: same seed, same numbers; hidden consumers of the global RNG;
thread count; and how much results move between seeds anyway."""
import statistics

import torch

from common import digits, evaluate, fit, mlp

data = digits()


def train_once(seed, epochs=5, threads=4, extra_random_call=False):
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    model = mlp([64, 128, 10], dropout=0.2)        # dropout draws random numbers every step
    if extra_random_call:
        torch.rand(1)                              # e.g. a debugging line someone added
    hist = fit(model, data, epochs, torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9), seed=seed)
    torch.set_num_threads(4)
    return [h[0] for h in hist], torch.cat([p.detach().flatten() for p in model.parameters()])


losses_a, w_a = train_once(0)
losses_b, w_b = train_once(0)
losses_c, w_c = train_once(1)
print("same seed twice, bitwise-identical weights:", torch.equal(w_a, w_b))
print("seed 0 vs seed 1, max weight difference:   ", f"{(w_a - w_c).abs().max().item():.3f}")
_, w_d = train_once(0, extra_random_call=True)
print("one extra torch.rand(1) after init, identical weights:", torch.equal(w_a, w_d))
_, w_e = train_once(0, threads=1)
print("4 threads vs 1 thread, identical weights:", torch.equal(w_a, w_e),
      f"(max diff {(w_a - w_e).abs().max().item():.1e})")

# DataLoader shuffling: global RNG vs a dedicated generator.
ds = torch.utils.data.TensorDataset(torch.arange(10))
for label, make_gen in [("global RNG", lambda: None), ("own generator", lambda: torch.Generator().manual_seed(0))]:
    orders = []
    for extra in [False, True]:
        torch.manual_seed(0)
        if extra:
            torch.rand(1)
        loader = torch.utils.data.DataLoader(ds, batch_size=10, shuffle=True, generator=make_gen())
        orders.append(next(iter(loader))[0].tolist())
    print(f"shuffle order with {label:<13}: {orders[0]} vs after an extra draw {orders[1]}")

# Seed-to-seed spread: the noise floor for comparing two training setups.
accs = []
for seed in range(10):
    torch.manual_seed(seed)
    model = mlp([64, 128, 10])
    fit(model, data, 20, torch.optim.AdamW(model.parameters(), lr=1e-3), seed=seed)
    accs.append(evaluate(model, *data["val"])[1])
print(f"\nval accuracy over 10 seeds: mean {statistics.mean(accs):.3f}, std {statistics.stdev(accs):.3f}, "
      f"min {min(accs):.3f}, max {max(accs):.3f}")
