"""Is a capability 'emergent', or is the metric sharp? A small test.

Trains small L2-05 GPTs on 3-digit addition and, at checkpoints along
training (the compute axis), scores each on held-out problems three ways:
loss on the answer tokens and per-digit accuracy (smooth, partial credit),
and exact match of the whole 4-digit answer (all or nothing).
"""
import math
import pathlib
import sys
import time

import torch
import torch.nn.functional as F

sys.dont_write_bytecode = True  # don't leave a __pycache__ in L2-05's folder
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "L2-05-transformers"))
from gpt import GPT, GPTConfig, count_params  # noqa: E402

torch.set_num_threads(8)
PLUS, EQ = 10, 11            # tokens 0-9 are digits
ANS = 4                      # answer digits, written least significant first


def is_test(a, b):           # 1 in 7 of all (a, b) pairs is held out
    return (a * 1000 + b) % 7 == 0


def make_problems(n, g, test=False):
    a = torch.randint(0, 1000, (n,), generator=g)
    b = torch.randint(0, 1000, (n,), generator=g)
    if test:
        b = b - (a * 1000 + b) % 7          # move each pair onto a test pair
        b = torch.where(b < 0, b + 7, b)    # (stays a test pair, mod 7)
    else:
        b = torch.where(is_test(a, b), (b + 1) % 1000, b)  # off the test set
    return a, b


def encode(a, b):
    """'123+456=' then the sum reversed ('9750' for 579), as digit tokens."""
    digits = lambda x, k: torch.stack([(x // 10 ** i) % 10 for i in range(k)], 1)
    # Operands most significant digit first, answer least significant first:
    # then each answer digit depends only on digits already seen plus a carry.
    return torch.cat([digits(a, 3).flip(1), torch.full_like(a, PLUS)[:, None],
                      digits(b, 3).flip(1), torch.full_like(a, EQ)[:, None],
                      digits(a + b, ANS)], 1)   # length 3+1+3+1+4 = 12


def split(seq):
    x, y = seq[:, :-1], seq[:, 1:].clone()
    y[:, :7] = -100           # no loss on predicting the question itself
    return x, y


def evaluate(model, test_seq):
    model.eval()
    with torch.no_grad():
        x, y = split(test_seq)
        logits, _ = model(x)
        ans_loss = F.cross_entropy(logits.flatten(0, 1), y.flatten(), ignore_index=-100)
        # Teacher-forced greedy digits at the 4 answer positions. For exact
        # match this equals free generation: the first wrong digit fails both.
        correct = logits[:, 7:].argmax(-1) == test_seq[:, 8:]
    model.train()
    return (ans_loss.item(), correct.float().mean().item(),
            correct.all(1).float().mean().item(), correct.float().mean(0))


g = torch.Generator().manual_seed(0)
a, b = make_problems(4000, g, test=True)
assert is_test(a, b).all()
test_seq = encode(a, b)
STEPS, BATCH, LR, EVERY = 1500, 128, 3e-3, 150

for width in [24, 48]:
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=12, block_size=11, n_layer=2,
                          n_head=2, n_embd=width))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    print(f"\nwidth {width}, {count_params(model):,} parameters; "
          f"{BATCH} fresh training problems per step")
    print("  step  answer loss  digit acc  exact match  digit_acc^4")
    t0 = time.time()
    for step in range(STEPS + 1):
        if step % EVERY == 0:
            loss_, dacc, exact, by_pos = evaluate(model, test_seq)
            print(f"{step:6d}  {loss_:11.3f}  {dacc:9.3f}  {exact:11.3f}  {dacc ** 4:11.3f}")
        if step == STEPS:
            break
        for gr in opt.param_groups:
            gr["lr"] = LR * 0.5 * (1 + math.cos(math.pi * step / STEPS))
        x, y = split(encode(*make_problems(BATCH, g)))
        logits, _ = model(x)
        loss = F.cross_entropy(logits.flatten(0, 1), y.flatten(), ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    print("  final accuracy by answer digit (ones, tens, hundreds, thousands):",
          " ".join(f"{v:.3f}" for v in by_pos.tolist()), f"({time.time() - t0:.1f} s)")
