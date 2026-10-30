"""Train a small character-level GPT on Romeo and Juliet, on CPU.

Prints baselines, losses during training, samples before and after, and a
check of whether the samples follow the play's format.
"""
import math
import pathlib
import re
import time

import torch

from gpt import GPT, GPTConfig, count_params

torch.manual_seed(0)
text = (pathlib.Path(__file__).parent / "data" / "romeo_and_juliet.txt").read_text(
    encoding="utf-8")

# Tokenize: one token per character. The last 10% of the play is validation.
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
encode = lambda s: torch.tensor([stoi[c] for c in s])
decode = lambda t: "".join(chars[i] for i in t.tolist())
data = encode(text)
n_train = int(0.9 * len(data))
train_data, val_data = data[:n_train], data[n_train:]
V = len(chars)
print(f"corpus: {len(text):,} chars, vocab {V}, "
      f"train {len(train_data):,} / val {len(val_data):,}")

# Baselines a model must beat (nats per character).
unigram = torch.bincount(train_data, minlength=V).float() + 1
unigram /= unigram.sum()
print(f"uniform baseline {math.log(V):.3f}, "
      f"unigram baseline {-unigram.log()[val_data].mean():.3f}")

cfg = GPTConfig(vocab_size=V, block_size=128, n_layer=4, n_head=4, n_embd=96)
model = GPT(cfg)
print(f"parameters: {count_params(model):,}")
BATCH, STEPS, LR, WARMUP = 32, 800, 4e-3, 50


def get_batch(src):
    ix = torch.randint(len(src) - cfg.block_size - 1, (BATCH,))
    x = torch.stack([src[i:i + cfg.block_size] for i in ix])
    y = torch.stack([src[i + 1:i + cfg.block_size + 1] for i in ix])  # shifted
    return x, y


@torch.no_grad()
def estimate_loss(src, n_batches=10):
    return sum(model(*get_batch(src))[1].item() for _ in range(n_batches)) / n_batches


def lr_at(step):   # linear warmup, then cosine decay to 10% of the peak
    if step < WARMUP:
        return LR * (step + 1) / WARMUP
    progress = (step - WARMUP) / (STEPS - WARMUP)
    return LR * (0.1 + 0.45 * (1 + math.cos(math.pi * progress)))


def sample(n, seed=1):
    torch.manual_seed(seed)
    start = encode("ROMEO.\n").unsqueeze(0)
    return decode(model.generate(start, n, temperature=0.8, top_k=20)[0])


print("\n--- sample before training ---\n" + sample(120))

opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.99), weight_decay=0.1)
t0 = time.time()
for step in range(STEPS + 1):
    if step % 200 == 0:
        print(f"step {step:4d}  train {estimate_loss(train_data):.3f}  "
              f"val {estimate_loss(val_data):.3f}  ({time.time() - t0:5.1f} s)")
    if step == STEPS:
        break
    for g in opt.param_groups:
        g["lr"] = lr_at(step)
    _, loss = model(*get_batch(train_data))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
print(f"training time {time.time() - t0:.1f} s, {torch.get_num_threads()} threads")

generated = sample(600)
print("\n--- sample after training ---\n" + generated)

# Did it learn the format? A speaker tag is a line like "JULIET." (the
# prompt "ROMEO." counts as one).
tag = re.compile(r"^([A-Z][A-Z ]+)\.$", re.M)
real_speakers = set(tag.findall(text))
gen_tags = tag.findall(generated)
words = re.findall(r"[a-z]+", generated)
real_words = set(re.findall(r"[a-z]+", text))
print(f"\nspeaker tags: {gen_tags}, real: {sum(t in real_speakers for t in gen_tags)}")
print(f"lowercase words: {len(words)}, "
      f"found in the play: {sum(w in real_words for w in words) / len(words):.0%}")
