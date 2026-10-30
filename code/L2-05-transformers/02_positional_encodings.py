"""Sinusoidal (added) and rotary (RoPE) positional encodings, checked numerically:
1. PE(t + k) is a rotation of PE(t) that does not depend on t.
2. With added encodings, a q.k score depends on absolute positions.
3. With RoPE, it depends only on the offset m - n."""
import torch

torch.manual_seed(0)
d = 16                                    # must be even: dimensions come in pairs
freqs = 10000.0 ** (-torch.arange(0, d, 2) / d)   # theta_i, one per pair


def sinusoidal(positions):
    """Rows are PE(t): sin in even dims, cos in odd dims (Vaswani et al., 2017)."""
    ang = positions[:, None].float() * freqs[None, :]       # (T, d/2)
    pe = torch.zeros(len(positions), d)
    pe[:, 0::2], pe[:, 1::2] = ang.sin(), ang.cos()
    return pe


def rotate_pairs(x, angles):
    """Rotate each (even, odd) coordinate pair of x by the given angles."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c, s = angles.cos(), angles.sin()
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * c - x2 * s
    out[..., 1::2] = x1 * s + x2 * c
    return out


def rope(x, pos):
    """RoPE (Su et al., 2021): rotate pair i of a vector at position pos by pos * theta_i."""
    return rotate_pairs(x, pos * freqs)


# ---- 1. Sinusoidal: shifting by k is a rotation that does not depend on t.
pos = torch.arange(0, 200)
pe = sinusoidal(pos)
k = 7
# Pairs are stored as (sin, cos), so in rotate_pairs' convention the shift by
# k is a rotation by -k * theta_i, the same for every t.
shifted = torch.stack([rotate_pairs(pe[t], -k * freqs) for t in range(len(pos) - k)])
err = (shifted - pe[k:]).abs().max().item()
print(f"1. max |R_k PE(t) - PE(t+k)| over t < {len(pos) - k}: {err:.1e}")
for t in (0, 50, 120):
    print(f"   PE({t:3d}) . PE({t + 5:3d}) = {(pe[t] @ pe[t + 5]).item():.4f}")

# ---- 2. Added encodings: the score depends on absolute position. -----------
Wq, Wk = torch.randn(d, d) / d ** 0.5, torch.randn(d, d) / d ** 0.5
a, b = torch.randn(d), torch.randn(d)      # two token embeddings ("cat", "sat")
print("\n2. added PE: score of the same token pair at offset 2, different places")
for m in (3, 103, 1003):
    xm = a + sinusoidal(torch.tensor([m]))[0]
    xn = b + sinusoidal(torch.tensor([m - 2]))[0]
    print(f"   positions ({m:4d}, {m - 2:4d}): {((xm @ Wq) @ (xn @ Wk)).item():.4f}")

# ---- 3. RoPE: the score depends only on m - n. -------------------------------
q, kk = a @ Wq, b @ Wk
print("\n3. RoPE: same pair, offset 2, different places")
for m in (3, 103, 1003):
    s = rope(q, m) @ rope(kk, m - 2)
    print(f"   positions ({m:4d}, {m - 2:4d}): {s.item():.4f}")
print("   offsets 0..5 at position 1000:",
      [round((rope(q, 1000) @ rope(kk, 1000 - j)).item(), 3) for j in range(6)])
# Rotations preserve length, so RoPE never changes |q| or |k|.
print(f"   |q| before {q.norm():.4f}, after rotation {rope(q, 1003).norm():.4f}")
