"""What attention costs: FLOP accounting and measured CPU time versus length.

Per layer and per sequence of T tokens with model width d (forward pass,
counting a multiply-add as 2 FLOPs):
  Q, K, V and output projections: 4 matrices of d x d -> 8 T d^2
  MLP with hidden width 4d:       2 matrices of d x 4d -> 16 T d^2
  scores Q K^T and weights @ V:   2 products of T x T x d -> 4 T^2 d
So the T^2 part overtakes the rest when 4 T^2 d > 24 T d^2, that is T > 6d.
"""
import math
import time

import torch
import torch.nn.functional as F

torch.manual_seed(0)


def layer_flops(T, d):
    linear = 24 * T * d * d
    attn = 4 * T * T * d
    return linear, attn


print("1. forward FLOPs per layer, width d = 768 (crossover at T = 6d = 4608)")
print(f"{'T':>7} {'linear GFLOP':>13} {'T^2 GFLOP':>10} {'T^2 share':>10}"
      f" {'scores MB (12 heads, fp32)':>27}")
for T in (512, 1024, 2048, 4096, 8192, 32768):
    lin, att = layer_flops(T, 768)
    score_mb = 12 * T * T * 4 / 1e6           # one T x T matrix per head
    print(f"{T:>7} {lin / 1e9:>13.1f} {att / 1e9:>10.1f} {att / (lin + att):>10.0%}"
          f" {score_mb:>27,.0f}")


def naive(q, k, v):
    s = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    mask = torch.ones(s.shape[-2:], dtype=torch.bool).tril()
    return s.masked_fill(~mask, float("-inf")).softmax(-1) @ v


def best_time(fn, *args, reps=5):
    fn(*args)                                   # warm up
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t0)
    return min(times)                           # min is least affected by load


print(f"\n2. measured causal attention, 4 heads x 64 dims, batch 1, "
      f"{torch.get_num_threads()} threads")
print(f"{'T':>6} {'naive ms':>9} {'x prev':>7} {'SDPA ms':>8} {'x prev':>7}")
prev = None
with torch.no_grad():
    for T in (256, 512, 1024, 2048, 4096):
        q, k, v = (torch.randn(1, 4, T, 64) for _ in range(3))
        tn = best_time(naive, q, k, v)
        ts = best_time(lambda a, b, c: F.scaled_dot_product_attention(
            a, b, c, is_causal=True), q, k, v)
        ratio = "" if prev is None else f"{tn / prev[0]:.1f}"
        ratio_s = "" if prev is None else f"{ts / prev[1]:.1f}"
        print(f"{T:>6} {tn * 1e3:>9.2f} {ratio:>7} {ts * 1e3:>8.2f} {ratio_s:>7}")
        prev = (tn, ts)

# Exactness: the fused kernel computes the same function, not an approximation.
q, k, v = (torch.randn(1, 4, 1024, 64) for _ in range(3))
diff = (naive(q, k, v) - F.scaled_dot_product_attention(q, k, v, is_causal=True)).abs().max()
print(f"\nmax difference naive vs SDPA at T=1024: {diff:.1e}")
