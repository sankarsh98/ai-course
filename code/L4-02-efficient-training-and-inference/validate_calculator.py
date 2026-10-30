"""Check calculator.py against a real model: the small GPT from L2-05.

1. Parameter counts for several shapes, against model.parameters().
2. Weight, gradient and Adam-state bytes, measured after one optimizer step.
3. Activation bytes, measured with saved-tensor hooks: every tensor autograd
   stores for the backward pass passes through `pack`, so we can add up their
   storage (counting shared storage once and skipping the parameters).
   Done for fp32, bf16 autocast, and per-block activation checkpointing.
4. FLOPs, against PyTorch's FlopCounterMode.
5. The time cost of checkpointing. Timings depend on the machine and its load.
"""
import sys
import time
from pathlib import Path

import torch
from torch.utils.checkpoint import checkpoint
from torch.utils.flop_counter import FlopCounterMode

sys.dont_write_bytecode = True   # leave no __pycache__ in the L2-05 folder
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "L2-05-transformers"))
from gpt import GPT, GPTConfig  # noqa: E402  (the L2-05 model, unchanged)

from calculator import Shape, activation_bytes, flops_per_token, params, state_bytes

torch.manual_seed(0)
torch.set_num_threads(4)


def shape_of(cfg):
    return Shape(n_layer=cfg.n_layer, d_model=cfg.n_embd, n_head=cfg.n_head,
                 vocab=cfg.vocab_size, seq_len=cfg.block_size)


class Checkpointed(torch.nn.Module):
    """Run a block without saving its internals; recompute them in backward.
    use_reentrant=True saves the block input with save_for_backward, so our
    hook sees it; the non-reentrant variant hides it from outer hooks."""
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x):
        return checkpoint(self.block, x, use_reentrant=True)


def saved_bytes(model, x, y, autocast=False):
    param_ptrs = {p.untyped_storage().data_ptr() for p in model.parameters()}
    store = {}

    def pack(t):
        s = t.untyped_storage()
        if s.data_ptr() not in param_ptrs:       # parameters are not activations
            store[s.data_ptr()] = s.nbytes()     # views share storage: count once
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
            _, loss = model(x, y)
    return sum(store.values()), loss


def state_measured(model, opt):
    w = sum(p.numel() * p.element_size() for p in model.parameters())
    g = sum(p.grad.numel() * p.grad.element_size() for p in model.parameters())
    o = sum(t.numel() * t.element_size() for st in opt.state.values()
            for t in st.values() if torch.is_tensor(t))
    return w, g, o


print("1. parameter counts")
for kw in [dict(vocab_size=66, block_size=128, n_layer=4, n_head=4, n_embd=96),
           dict(vocab_size=1000, block_size=64, n_layer=2, n_head=8, n_embd=256),
           dict(vocab_size=50257, block_size=1024, n_layer=12, n_head=12, n_embd=768)]:
    cfg = GPTConfig(**kw)
    real = sum(p.numel() for p in GPT(cfg).parameters())
    print(f"  L={cfg.n_layer:>2} d={cfg.n_embd:>3} V={cfg.vocab_size:>5}: model {real:>11,}"
          f"  calculator {params(shape_of(cfg))['total']:>11,}")

cfg = GPTConfig(vocab_size=66, block_size=128, n_layer=4, n_head=4, n_embd=96)
s, B = shape_of(cfg), 8
x = torch.randint(0, 66, (B, cfg.block_size))
y = torch.randint(0, 66, (B, cfg.block_size))

print("\n2. model state after one AdamW step (fp32), bytes")
torch.manual_seed(0)
model = GPT(cfg)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
_, loss = model(x, y)
loss.backward()
opt.step()
w, g, o = state_measured(model, opt)
n = params(s)["total"]
print(f"  weights {w:,}  grads {g:,}  Adam state {o:,}  total {w + g + o:,}")
print(f"  calculator: {state_bytes(n, 'fp32 weights, fp32 Adam'):,.0f}"
      f"  (the difference is Adam's per-tensor step counters: {o - 8 * n} bytes)")

print("\n3. activation bytes saved for backward, B=8, T=128")
grads = {}
for mode in ("fp32", "bf16 autocast", "checkpointed"):
    torch.manual_seed(0)
    model = GPT(cfg)
    if mode == "checkpointed":
        model.blocks = torch.nn.ModuleList(Checkpointed(b) for b in model.blocks)
    measured, loss = saved_bytes(model, x, y, autocast=(mode == "bf16 autocast"))
    if mode == "fp32":
        pred = activation_bytes(s, B, act=4)["saved"]
    elif mode == "bf16 autocast":   # residual stream stays fp32; matmul tensors are bf16
        pred = activation_bytes(s, B, act=2, res=4, autocast_copies=True)["saved"]
    else:
        pred = activation_bytes(s, B, act=4, checkpoint=True)["saved"]
    loss.backward()
    grads[mode] = torch.cat([p.grad.flatten() for p in model.parameters()])
    print(f"  {mode:<14} measured {measured:>11,}  calculator {pred:>11,.0f}  loss {loss.item():.6f}")
diff = (grads["fp32"] - grads["checkpointed"]).abs().max().item()
print(f"  max |grad difference| fp32 vs checkpointed: {diff:.1e}")

print("\n4. FLOPs for one training step (forward + backward), B=8, T=128")
torch.manual_seed(0)
model = GPT(cfg)
with FlopCounterMode(display=False) as fc:
    _, loss = model(x, y)
    loss.backward()
counts = {str(k): v for k, v in fc.get_flop_counts()["Global"].items()}
for k, v in sorted(counts.items()):
    print(f"  {k:<58}{v:>14,}")
tokens = B * cfg.block_size
print(f"  total measured {fc.get_total_flops():,}")
print(f"  calculator     {flops_per_token(s) * tokens:,.0f}")
print(f"  6 N D          {6 * n * tokens:,}")

print("\n5. wall time per training step, B=32, T=128 (median of 5)")
xb = torch.randint(0, 66, (32, cfg.block_size))
yb = torch.randint(0, 66, (32, cfg.block_size))
for mode in ("plain", "checkpointed"):
    torch.manual_seed(0)
    model = GPT(cfg)
    if mode == "checkpointed":
        model.blocks = torch.nn.ModuleList(Checkpointed(b) for b in model.blocks)
    times = []
    for _ in range(6):
        t0 = time.perf_counter()
        _, loss = model(xb, yb)
        loss.backward()
        times.append(time.perf_counter() - t0)
    print(f"  {mode:<13}{sorted(times[1:])[2] * 1e3:7.1f} ms")
