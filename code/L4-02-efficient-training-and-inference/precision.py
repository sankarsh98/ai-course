"""Numeric formats and why mixed precision is built the way it is.

1. Range and precision of fp32, fp16, bf16 and the two fp8 formats.
2. Small gradients underflow in fp16; loss scaling rescues them.
3. Small updates vanish when weights are stored in 16 bits.
4. Long sums need a wide accumulator.
5. A late-training regime (small learning rate) with Adam, three ways:
   fp32; bf16 compute with fp32 master weights; everything in bf16.
6. Is bf16 faster? Only on hardware that supports it.
"""
import time

import torch

torch.manual_seed(0)
torch.set_num_threads(4)

print("1. formats")
print(f"  {'dtype':<16}{'bits':>5}{'eps':>11}{'max':>11}{'min normal':>12}{'min subnormal':>15}")
for dt in (torch.float32, torch.float16, torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2):
    fi = torch.finfo(dt)
    mant = round(-torch.log2(torch.tensor(fi.eps)).item())   # mantissa bits
    sub = fi.smallest_normal * 2.0 ** -mant                   # smallest subnormal
    print(f"  {str(dt)[6:]:<16}{fi.bits:>5}{fi.eps:>11.2e}{fi.max:>11.2e}"
          f"{fi.smallest_normal:>12.2e}{sub:>15.2e}")

print("\n2. gradient underflow: 1M values log-uniform in [1e-12, 1e-2]")
g = 10 ** (torch.rand(1_000_000, dtype=torch.float64) * -10 - 2)
for name, conv in [("fp16", lambda t: t.half()),
                   ("fp16, loss scale 2^16", lambda t: (t * 2**16).half().double() / 2**16),
                   ("bf16", lambda t: t.bfloat16())]:
    q = conv(g).double()
    zero = (q == 0).float().mean().item()
    rel = ((q - g).abs() / g)[q != 0].median().item()
    print(f"  {name:<24} flushed to zero {zero:6.1%}   median relative error of the rest {rel:.1e}")
big = torch.tensor([1e-1, 1.0, 10.0], dtype=torch.float64)
print(f"  but scaled by 2^16, gradients of 1e-1, 1, 10 become {(big * 2**16).half().tolist()}"
      " in fp16 (overflow)")

print("\n3. adding an update of 1e-4 to a weight of 1.0, 1000 times")
for dt in (torch.float32, torch.bfloat16, torch.float16):
    w = torch.tensor(1.0, dtype=dt)
    for _ in range(1000):
        w = w + torch.tensor(1e-4, dtype=dt)
    print(f"  stored in {str(dt)[6:]:<9} final weight {w.item():.6f}   (exact: 1.100000)")

print("\n4. sum of 4096 products x_i*y_i, x, y ~ N(0,1)")
x, y = torch.randn(4096, dtype=torch.float64), torch.randn(4096, dtype=torch.float64)
exact = (x * y).sum().item()
xb, yb = x.bfloat16(), y.bfloat16()
acc = torch.tensor(0.0, dtype=torch.bfloat16)
for a, b in zip(xb, yb):        # every partial sum rounded to bf16
    acc = acc + a * b
wide = (xb.double() * yb.double()).sum().item()   # bf16 inputs, wide accumulator
print(f"  exact {exact:.4f}  bf16 inputs + fp64 accumulator {wide:.4f}"
      f"  bf16 accumulator {acc.item():.4f}")

print("\n5. Adam, lr 1e-4, on a regression whose weights are near 1 (2000 steps)")
X = torch.randn(512, 64)
w_true = 1 + 0.1 * torch.randn(64, 1)
Y = X @ w_true
w_start = w_true + 0.01 * torch.randn(64, 1)     # already close: the late phase of training


def run(mode, steps=2000, lr=1e-4):
    dt = torch.bfloat16 if mode == "pure bf16" else torch.float32
    w = w_start.clone().to(dt).requires_grad_()
    opt = torch.optim.Adam([w], lr=lr)            # its state has w's dtype
    for _ in range(steps):
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=(mode != "fp32")):
            loss = ((X.to(dt) @ w - Y.to(dt)) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return ((X @ w.detach().float() - Y) ** 2).mean().item()   # judged in fp32


print(f"  starting loss {((X @ w_start - Y) ** 2).mean().item():.2e}")
for mode in ("fp32", "bf16 compute, fp32 master", "pure bf16"):
    print(f"  {mode:<26} final loss {run(mode):.2e}")

print("\n6. 1024x1024 matmul on this CPU (median of 5)")
for dt in (torch.float32, torch.bfloat16):
    a, b = torch.randn(1024, 1024).to(dt), torch.randn(1024, 1024).to(dt)
    ts = []
    for _ in range(5):
        t0 = time.perf_counter()
        a @ b
        ts.append(time.perf_counter() - t0)
    ms = sorted(ts)[2] * 1e3
    print(f"  {str(dt)[6:]:<9}{ms:7.2f} ms   {2 * 1024**3 / (ms / 1e3) / 1e9:6.1f} GFLOP/s")
