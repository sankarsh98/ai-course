"""A small decoder-only transformer (GPT) in plain PyTorch.

Shapes: B = batch, T = sequence length, C = model width,
H = number of heads, D = C // H = head width.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 128      # maximum context length
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128          # C
    pre_ln: bool = True        # False gives the original post-LN layout


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)  # all heads at once
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)     # W_O, mixes heads

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, H, T, D): each head attends independently.
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
                   for t in (q, k, v))
        # softmax(q k^T / sqrt(D)) v with the causal mask, as a fused kernel.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # concatenate heads
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.out = nn.Linear(4 * cfg.n_embd, cfg.n_embd)

    def forward(self, x):
        return self.out(F.gelu(self.fc(x)))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.pre_ln = cfg.pre_ln
        self.ln1, self.ln2 = nn.LayerNorm(cfg.n_embd), nn.LayerNorm(cfg.n_embd)
        self.attn, self.mlp = CausalSelfAttention(cfg), MLP(cfg)

    def forward(self, x):
        if self.pre_ln:   # normalize sublayer inputs; residual path untouched
            x = x + self.attn(self.ln1(x))
            return x + self.mlp(self.ln2(x))
        x = self.ln1(x + self.attn(x))   # post-LN: normalize after the sum
        return self.ln2(x + self.mlp(x))


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)  # learned
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd) if cfg.pre_ln else nn.Identity()
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight   # weight tying
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)
        # GPT-2 style: shrink layers that write into the residual stream by
        # 1/sqrt(number of residual layers) so its variance does not grow.
        for name, p in self.named_parameters():
            if name.endswith(("proj.weight", "out.weight")):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def forward(self, idx, targets=None):
        T = idx.shape[1]
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T))
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))          # (B, T, vocab)
        loss = None
        if targets is not None:   # every position predicts the next token
            loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, n_new, temperature=1.0, top_k=None):
        for _ in range(n_new):
            logits, _ = self(idx[:, -self.cfg.block_size:])  # crop context
            logits = logits[:, -1] / temperature              # last position
            if top_k is not None:
                kth = torch.topk(logits, top_k).values[:, [-1]]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            nxt = torch.multinomial(logits.softmax(-1), num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx


def count_params(model):
    return sum(p.numel() for p in model.parameters())  # tied weights once
