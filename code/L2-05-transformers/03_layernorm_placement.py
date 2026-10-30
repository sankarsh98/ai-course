"""Pre-LN versus post-LN: gradient scale at initialization, and a training race.

Xiong et al. (2020) show that at initialization, the gradient of the last
layer's FFN weights stays roughly constant with depth in post-LN
transformers but shrinks with depth in pre-LN ones, and argue that the
large post-LN gradients near the output are why post-LN needs learning-rate
warmup. Part 1 repeats that measurement on our model. Part 2 trains both
layouts with and without warmup.
"""
import math
import pathlib
import time

import torch
import torch.nn as nn

from gpt import GPT, GPTConfig

text = (pathlib.Path(__file__).parent / "data" / "romeo_and_juliet.txt").read_text(encoding="utf-8")
chars = sorted(set(text))
data = torch.tensor([chars.index(c) for c in text])
V, T = len(chars), 64


def batch(gen, size=32):
    ix = torch.randint(len(data) - T - 1, (size,), generator=gen)
    return (torch.stack([data[i:i + T] for i in ix]),
            torch.stack([data[i + 1:i + T + 1] for i in ix]))


def make(pre_ln, n_layer, seed=0):
    torch.manual_seed(seed)
    return GPT(GPTConfig(vocab_size=V, block_size=T, n_layer=n_layer,
                         n_head=4, n_embd=128, pre_ln=pre_ln))


# ---- 1. Gradient norm of the last FFN output matrix at init, vs depth. -----
def xavier(model):
    # Xavier-uniform matrices, the setting Xiong et al. analyze.
    for name, p in model.named_parameters():
        if p.dim() == 2 and "emb" not in name:
            nn.init.xavier_uniform_(p)
    return model


print("1. |grad| of the last block's MLP output weights at init (mean of 3 batches)")
print(f"   {'init':<12} {'layout':<8} " + " ".join(f"{f'L={L}':>7}" for L in (3, 6, 12, 24)))
for init_name, init in (("Xavier", xavier), ("GPT-2 style", lambda m: m)):
    for pre_ln in (False, True):
        row = []
        for L in (3, 6, 12, 24):
            model = init(make(pre_ln, L))
            gen = torch.Generator().manual_seed(1)
            g = 0.0
            for _ in range(3):
                model.zero_grad()
                model(*batch(gen))[1].backward()
                g += model.blocks[-1].mlp.out.weight.grad.norm().item() / 3
            row.append(g)
        name = "pre-LN" if pre_ln else "post-LN"
        print(f"   {init_name:<12} {name:<8} " + " ".join(f"{g:>7.3f}" for g in row))

# ---- 2. Training race: 6 layers, Adam at peak lr 1e-3, 200 steps, 3 seeds. ---
STEPS, PEAK = 200, 1e-3


def train(pre_ln, warmup, seed):
    model = make(pre_ln, 6, seed)
    opt = torch.optim.Adam(model.parameters(), lr=PEAK)
    gen = torch.Generator().manual_seed(seed + 10)
    losses = []
    for step in range(STEPS):
        lr = PEAK * min(1.0, (step + 1) / warmup) if warmup else PEAK
        for g in opt.param_groups:
            g["lr"] = lr
        loss = model(*batch(gen, size=16))[1]
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return sum(losses[-25:]) / 25          # mean of the last 25 steps


unigram = torch.bincount(data, minlength=V).float() / len(data)
print(f"\n2. training loss after {STEPS} steps (mean of last 25), per seed")
print(f"   uniform baseline {math.log(V):.2f}, unigram baseline "
      f"{-(unigram * unigram.log()).sum():.2f}")
t0 = time.time()
for pre_ln, warmup in ((False, 0), (False, 100), (True, 0)):
    r = [train(pre_ln, warmup, seed) for seed in range(3)]
    name = "pre-LN" if pre_ln else "post-LN"
    print(f"   {name:<8} warmup {warmup:>3}: " + "  ".join(f"{x:.3f}" for x in r))
print(f"   ({time.time() - t0:.0f} s)")
