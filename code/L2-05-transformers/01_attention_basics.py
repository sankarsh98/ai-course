"""Self-attention from first principles: a hand-sized example, the sqrt(d_k)
scale, a check against PyTorch, permutation equivariance, the causal mask."""
import math

import torch
import torch.nn.functional as F

torch.manual_seed(0)
torch.set_printoptions(precision=3, sci_mode=False)


def attention(q, k, v, causal=False):
    """softmax(q k^T / sqrt(d_k)) v for tensors shaped (..., T, d)."""
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
    if causal:
        T = q.size(-2)
        mask = torch.ones(T, T, dtype=torch.bool).tril()
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = scores.softmax(dim=-1)
    return weights @ v, weights


# ---- 1. Worked example: W_Q = W_K = identity, W_V swaps the coordinates. ----
X = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
W_v = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
Q, K, Vv = X, X, X @ W_v
out, w = attention(Q, K, Vv)
print("1. worked example")
print("raw scores Q K^T:\n", Q @ K.T)
print("weights (rows sum to 1):\n", w)
print("outputs:\n", out)

# ---- 2. Why divide by sqrt(d_k)? q.k has variance d_k for unit-variance entries.
print("\n2. score scale vs head width d_k (64 keys, 2000 random queries)")
print(f"{'d_k':>5} {'std(q.k)':>9} | {'max weight':>10} {'grad norm':>10}"
      f" | {'max w, scaled':>13} {'grad, scaled':>12}")
n_keys = 64
for d in (4, 16, 64, 256, 1024):
    q = torch.randn(2000, d)
    k = torch.randn(n_keys, d)
    raw = q @ k.T
    row = []
    for s in (raw, raw / math.sqrt(d)):
        s = s.clone().requires_grad_(True)
        p = s.softmax(dim=-1)
        # Stand-in for the loss gradient: a random linear function of p.
        (p * torch.randn_like(p)).sum().backward()
        row += [p.max(dim=-1).values.mean().item(), s.grad.norm(dim=-1).mean().item()]
    print(f"{d:>5} {raw.std().item():>9.2f} | {row[0]:>10.3f} {row[1]:>10.4f}"
          f" | {row[2]:>13.3f} {row[3]:>12.4f}")

# ---- 3. Check against PyTorch ------------------------------------------------
B, H, T, D = 2, 4, 10, 16
q, k, v = (torch.randn(B, H, T, D) for _ in range(3))
mine, _ = attention(q, k, v, causal=True)
ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
print(f"\n3. max |ours - F.scaled_dot_product_attention| = "
      f"{(mine - ref).abs().max().item():.2e}")

# ---- 4. Permutation equivariance ----------------------------------------------
d_model = 8
Wq, Wk, Wv = (torch.randn(d_model, d_model) for _ in range(3))
x = torch.randn(5, d_model)
perm = torch.tensor([3, 0, 4, 1, 2])


def self_attn(x):
    return attention(x @ Wq, x @ Wk, x @ Wv)[0]


diff = (self_attn(x[perm]) - self_attn(x)[perm]).abs().max().item()
print(f"\n4. shuffle inputs, outputs shuffle the same way: max diff {diff:.2e}")
print("   so 'dog bites man' and 'man bites dog' get the same set of vectors")

# ---- 5. Causal mask -------------------------------------------------------------
x2 = x.clone()
x2[-1] = torch.randn(d_model)          # change only the last token
for causal in (False, True):
    a = attention(x @ Wq, x @ Wk, x @ Wv, causal)[0]
    b = attention(x2 @ Wq, x2 @ Wk, x2 @ Wv, causal)[0]
    changed = (a - b).abs().amax(dim=-1) > 1e-6
    print(f"5. causal={causal!s:5}: outputs changed at positions "
          f"{changed.nonzero().flatten().tolist()}")
