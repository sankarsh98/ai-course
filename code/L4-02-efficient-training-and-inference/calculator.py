"""Memory, FLOPs and communication calculator for decoder-only transformers.

Pure arithmetic: no model is built. validate_calculator.py checks these formulas
against a real PyTorch model (the GPT from L2-05).

Symbols: L layers, d model width, H heads, V vocabulary, T sequence length,
B sequences per batch, N parameters. Bytes, not bits, everywhere.
"""
from dataclasses import dataclass


@dataclass
class Shape:
    n_layer: int
    d_model: int
    n_head: int
    vocab: int
    seq_len: int
    d_ff: int = 0             # 0 means 4 * d_model
    n_kv_head: int = 0        # 0 means n_head (no grouped-query attention)
    gated_mlp: bool = False   # SwiGLU-style MLP: three matrices instead of two
    bias: bool = True         # biases in linear layers
    layernorm: bool = True    # LayerNorm (weight + bias) vs RMSNorm (weight only)
    tied: bool = True         # output head shares the token-embedding matrix
    learned_pos: bool = True  # learned position table (False for rotary)

    def __post_init__(self):
        self.d_ff = self.d_ff or 4 * self.d_model
        self.n_kv_head = self.n_kv_head or self.n_head


def params(s):
    """Parameter count by component. 'matmul' = weights used in a matmul per token."""
    d, hd = s.d_model, s.d_model // s.n_head
    kv = 2 * d * s.n_kv_head * hd                    # key and value projections
    attn = d * d + kv + d * d                        # query, key/value, output
    mlp = (3 if s.gated_mlp else 2) * d * s.d_ff
    norm = 2 * d if s.layernorm else d
    biases = (d + 2 * s.n_kv_head * hd + d + s.d_ff + d) if s.bias else 0
    layer = attn + mlp + 2 * norm + biases
    emb = s.vocab * d + (s.seq_len * d if s.learned_pos else 0)
    head = 0 if s.tied else s.vocab * d
    total = emb + s.n_layer * layer + norm + head
    matmul = s.n_layer * (attn + mlp) + s.vocab * d  # head matmul runs either way
    return dict(embeddings=emb, per_layer=layer, total=total, matmul=matmul)


# Bytes per parameter for model state (weights + gradients + optimizer state).
RECIPES = {
    "fp32 weights, fp32 Adam": dict(weights=4, grads=4, optim=8),
    "bf16 weights+grads, fp32 master+Adam": dict(weights=2, grads=2, optim=12),
    "bf16 everything (no fp32 master)": dict(weights=2, grads=2, optim=4),
    "fp32 SGD with momentum": dict(weights=4, grads=4, optim=4),
}


def state_bytes(n, recipe, zero_stage=0, n_dp=1):
    """Model-state bytes per device. ZeRO shards optimizer (1), +grads (2), +weights (3)."""
    r = RECIPES[recipe]
    w, g, o = r["weights"], r["grads"], r["optim"]
    if zero_stage >= 1:
        o /= n_dp
    if zero_stage >= 2:
        g /= n_dp
    if zero_stage >= 3:
        w /= n_dp
    return n * (w + g + o)


def activation_bytes(s, batch, act=2, res=None, flash=True, dropout=False,
                     checkpoint=False, autocast_copies=False):
    """Bytes autograd keeps between forward and backward.

    act: bytes per element of matmul inputs/outputs; res: bytes of the residual
    stream (layer-norm inputs), defaults to act. Per token per layer we save:
    two norm inputs (2d at res bytes) plus their mean and 1/std (4 fp32 numbers),
    and 14d at act bytes: norm-1 output, q, k, v, attention output, norm-2
    output, MLP hidden before and after the nonlinearity (4d each).
    flash=True: the attention kernel keeps one fp32 log-sum-exp per head and
    row instead of the T x T probability matrix.
    """
    res = act if res is None else res
    tok = batch * s.seq_len
    d = s.d_model
    hd = d // s.n_head
    # q (d) plus k and v (shrunk by grouped-query attention) instead of 3d
    qkv = d + 2 * s.n_kv_head * hd
    mlp = (3 if s.gated_mlp else 2) * s.d_ff     # SwiGLU keeps gate and up outputs
    per_tok = 2 * d * res + 16 + act * (d + qkv + d + d + mlp)
    if flash:
        per_tok += 4 * s.n_head                     # fp32 log-sum-exp per head
    else:
        per_tok += s.n_head * s.seq_len * act       # softmax probabilities
    if dropout:   # 1-byte masks after attention and MLP, as in Korthikanti et al.
        per_tok += 2 * d
        if not flash:   # attention-probability dropout: mask + its 16-bit output
            per_tok += s.n_head * s.seq_len * (1 + act)
    layer = per_tok * tok
    # After the last block: final-norm input + stats, head input, fp32 logits,
    # plus the int64 token ids, targets and positions, and one fp32 scalar
    # kept by the loss reduction.
    tail = tok * (d * res + 8 + d * act + 4 * s.vocab) + 8 * (2 * tok + s.seq_len) + 4
    if checkpoint:   # keep only each block's input; recompute the rest
        saved = s.n_layer * tok * d * res + tail
        peak = saved + layer                       # one block re-materialized
    else:
        saved = s.n_layer * layer + tail
        peak = saved
    if autocast_copies:  # autocast saves a 16-bit copy of every matmul weight
        saved += 2 * params(s)["matmul"]
        peak += 2 * params(s)["matmul"]
    return dict(saved=saved, peak=peak, per_layer=layer)


def korthikanti_per_layer(s, batch):
    """Eq. (1) of Korthikanti et al. (2022): s*b*h*(34 + 5*a*s/h) bytes, 16-bit."""
    t, b, h, a = s.seq_len, batch, s.d_model, s.n_head
    return t * b * h * (34 + 5 * a * t / h)


def flops_per_token(s, train=True, recompute=False):
    """Matmul FLOPs per token: 2 per multiply-add. Attention scores and the
    weighted sum add 4*T*d per layer forward (no saving from the causal mask)."""
    fwd = 2 * params(s)["matmul"] + 4 * s.n_layer * s.seq_len * s.d_model
    if not train:
        return fwd
    return fwd * (4 if recompute else 3)   # backward ~ 2x forward, +1 forward if recomputing


def allreduce_bytes(payload, n):
    """Bytes each device sends in a ring all-reduce (reduce-scatter + all-gather)."""
    return 2 * (n - 1) / n * payload


def pipeline_bubble(p, m):
    """Idle fraction of a GPipe-style schedule with p stages and m micro-batches."""
    return (p - 1) / (m + p - 1)


def gb(x):
    return f"{x / 1e9:8.2f} GB"


SHAPES = {
    # GPT-2 small layout (Radford et al., 2019): learned positions, tied head.
    "GPT-2-small shape": Shape(n_layer=12, d_model=768, n_head=12, vocab=50257, seq_len=1024),
    # Table 1 of Jiang et al. (2023), Mistral 7B. Assumed (not in the table):
    # SwiGLU MLP, RMSNorm, no biases, untied head, rotary positions.
    "Mistral-7B shape": Shape(n_layer=32, d_model=4096, n_head=32, n_kv_head=8, vocab=32000,
                              seq_len=4096, d_ff=14336, gated_mlp=True, bias=False,
                              layernorm=False, tied=False, learned_pos=False),
}

if __name__ == "__main__":
    PEAK = 1e15        # illustrative accelerator peak, FLOP/s (not a specific chip)
    LINK = 100e9       # illustrative per-device network bandwidth, bytes/s
    for name, s in SHAPES.items():
        p = params(s)
        n = p["total"]
        print(f"== {name}: L={s.n_layer} d={s.d_model} H={s.n_head} V={s.vocab} T={s.seq_len}")
        print(f"parameters {n:,}  (embeddings {p['embeddings']:,}, matmul {p['matmul']:,})")
        print("model state per device (weights + grads + optimizer):")
        for r in RECIPES:
            print(f"  {r:<38}{gb(state_bytes(n, r))}")
        mp = "bf16 weights+grads, fp32 master+Adam"
        for z in (1, 2, 3):
            print(f"  same, ZeRO stage {z} over 8 devices      {gb(state_bytes(n, mp, z, 8))}")
        print("activations for ONE sequence, bf16, fused attention:")
        a = activation_bytes(s, 1)
        ac = activation_bytes(s, 1, checkpoint=True)
        an = activation_bytes(s, 1, flash=False, dropout=True)
        print(f"  stored, no recompute{gb(a['saved'])}   with per-layer checkpointing"
              f"{gb(ac['saved'])} (peak {gb(ac['peak']).strip()})")
        print(f"  unfused attention + dropout{gb(an['saved'])}"
              f"   one layer: {an['per_layer'] / 1e6:.1f} MB vs Korthikanti eq. 1:"
              f" {korthikanti_per_layer(s, 1) / 1e6:.1f} MB")
        f = flops_per_token(s)
        print(f"training FLOPs per token {f:.3e}  vs 6N = {6 * n:.3e}  (ratio {f / (6 * n):.3f})")
        tokens = 20 * n
        days = tokens * f / (PEAK * 0.4) / 86400
        print(f"{tokens:.2e} tokens (20 per parameter): {tokens * f:.2e} FLOPs,"
              f" {days:,.1f} device-days at 40% of {PEAK:.0e} FLOP/s")
        grad = 2 * n   # bf16 gradients
        for dp in (8, 64):
            t = allreduce_bytes(grad, dp) / LINK
            print(f"  data-parallel all-reduce of bf16 grads over {dp:>2}: "
                  f"{allreduce_bytes(grad, dp) / 1e9:6.2f} GB sent per device, {t * 1e3:5.0f} ms at 100 GB/s")
        print()
    print("pipeline bubble (p-1)/(m+p-1):")
    for p_ in (4, 8):
        print(f"  p={p_}: " + "  ".join(f"m={m}: {pipeline_bubble(p_, m):.0%}" for m in (1, 4, 16, 64)))
