"""Data, tensor and pipeline parallelism, small enough to run on one CPU.

1. Data parallelism with real processes (torch.distributed, gloo backend):
   4 workers each take a quarter of the batch, all-reduce their gradients,
   and must end up with exactly the gradient of the full batch.
2. Tensor parallelism (Megatron-style MLP split) in one process: a column
   split of the first matrix and a row split of the second need a single
   sum (all-reduce) of partial outputs.
3. Pipeline parallelism: simulate a GPipe schedule and measure the bubble.
"""
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

WORLD, D, HID, BATCH = 4, 64, 256, 32


def make_model():
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(D, HID), torch.nn.GELU(), torch.nn.Linear(HID, 1))


def make_data():
    g = torch.Generator().manual_seed(1)
    return torch.randn(BATCH, D, generator=g), torch.randn(BATCH, 1, generator=g)


def worker(rank, queue):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29517")
    dist.init_process_group("gloo", rank=rank, world_size=WORLD)
    torch.set_num_threads(1)
    model = make_model()                      # same seed: identical replicas
    x, y = make_data()
    shard = slice(rank * BATCH // WORLD, (rank + 1) * BATCH // WORLD)
    F.mse_loss(model(x[shard]), y[shard]).backward()
    flat = torch.cat([p.grad.flatten() for p in model.parameters()])
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)   # every rank gets the sum
    flat /= WORLD                                  # mean of shard means = full-batch mean
    if rank == 0:
        queue.put(flat)
    dist.destroy_process_group()


def data_parallel():
    model = make_model()
    x, y = make_data()
    F.mse_loss(model(x), y).backward()
    ref = torch.cat([p.grad.flatten() for p in model.parameters()])
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(r, queue)) for r in range(WORLD)]
    for p in procs:
        p.start()
    got = queue.get()
    for p in procs:
        p.join()
    n = ref.numel()
    print(f"1. data parallel, {WORLD} processes, {n:,} gradient values")
    print(f"   max |all-reduced grad - full-batch grad| = {(got - ref).abs().max():.1e}")
    print(f"   ring all-reduce traffic per worker: 2(W-1)/W x {n * 4:,} bytes"
          f" = {2 * (WORLD - 1) / WORLD * n * 4:,.0f} bytes per step")


def tensor_parallel(t=4, tokens=8):
    torch.manual_seed(0)
    x = torch.randn(tokens, D)
    A, B = torch.randn(D, HID) / D**0.5, torch.randn(HID, D) / HID**0.5
    full = F.gelu(x @ A) @ B
    cols = A.chunk(t, dim=1)                  # device i holds a block of A's columns ...
    rows = B.chunk(t, dim=0)                  # ... and the matching block of B's rows
    partial = [F.gelu(x @ a) @ b for a, b in zip(cols, rows)]   # no communication
    out = sum(partial)                        # the one all-reduce in the forward pass
    print(f"\n2. tensor parallel MLP over {t} devices")
    print(f"   max |sum of partials - full| = {(out - full).abs().max():.1e}")
    # Splitting A by rows instead gives partial pre-activations that must be
    # summed BEFORE the GELU, an extra synchronization point:
    xs, As = x.chunk(t, dim=1), A.chunk(t, dim=0)
    wrong = sum(F.gelu(xi @ ai) for xi, ai in zip(xs, As))
    print(f"   row-split A, GELU applied per device: max error {(wrong - F.gelu(x @ A)).abs().max():.2f}")


def pipeline(p, m, show=False):
    """GPipe: all forwards, then all backwards; backward takes 2 time units."""
    busy = {}                                 # (stage, time) -> label
    t_free = [0] * p
    fwd_done = {}
    for j in range(m):                        # forward of micro-batch j through stages
        for s in range(p):
            start = max(t_free[s], fwd_done.get((s - 1, j), 0))
            busy.update({(s, start): f"F{j}"})
            t_free[s] = fwd_done[(s, j)] = start + 1
    bwd_done = {}
    for j in range(m):                        # backward runs from the last stage down
        for s in reversed(range(p)):
            ready = fwd_done[(p - 1, m - 1)] if s == p - 1 else bwd_done[(s + 1, j)]
            start = max(t_free[s], ready)
            busy.update({(s, start): f"B{j}", (s, start + 1): f"B{j}"})
            t_free[s] = bwd_done[(s, j)] = start + 2
    total = max(t_free)
    idle = 1 - (3 * m * p) / (total * p)
    if show:
        for s in range(p):
            print("   stage", s, " ".join(busy.get((s, t), "..").rjust(3) for t in range(total)))
    return total, idle


if __name__ == "__main__":
    data_parallel()
    tensor_parallel()
    print("\n3. pipeline (GPipe) schedule, p=4 stages, m=4 micro-batches; F=1 unit, B=2 units")
    pipeline(4, 4, show=True)
    for p_, m_ in [(4, 1), (4, 4), (4, 16), (8, 16), (8, 64)]:
        total, idle = pipeline(p_, m_)
        print(f"   p={p_} m={m_:>2}: time {total:>3} units, idle {idle:5.1%},"
              f" formula (p-1)/(m+p-1) = {(p_ - 1) / (m_ + p_ - 1):5.1%}")
