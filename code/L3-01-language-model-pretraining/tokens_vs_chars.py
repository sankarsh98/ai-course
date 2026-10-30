"""Same model, same compute, two tokenizers: which predicts the text better?

Trains the L2-05 GPT twice on Romeo and Juliet, once on characters and once
on BPE tokens, with the same number of steps, batch and context length in
tokens (so about the same training FLOPs). Loss per token is not comparable
across tokenizers, so both are converted to bits per byte of text.
"""
import math
import pathlib
import sys
import time

import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.dont_write_bytecode = True  # don't leave a __pycache__ in L2-05's folder
sys.path.insert(0, str(HERE.parent / "L2-05-transformers"))  # reuse L2-05's model
from gpt import GPT, GPTConfig, count_params  # noqa: E402
from bpe import CORPUS, BPETokenizer, train_bpe  # noqa: E402

torch.set_num_threads(8)
text = CORPUS.read_text(encoding="utf-8")
n_train = int(0.9 * len(text))
train_text, val_text = text[:n_train], text[n_train:]
val_bytes = len(val_text.encode("utf-8"))

STEPS, BATCH, BLOCK, LR = 1000, 32, 64, 3e-3


def char_tokenizer():
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    return len(chars), lambda s: [stoi[c] for c in s]


def bpe_tokenizer(n_merges=500):
    tok = BPETokenizer(train_bpe(train_text, n_merges))  # trained on train only
    return 256 + n_merges, tok.encode


def run(name, vocab_size, encode):
    torch.manual_seed(0)
    tr = torch.tensor(encode(train_text))
    va = torch.tensor(encode(val_text))
    model = GPT(GPTConfig(vocab_size=vocab_size, block_size=BLOCK,
                          n_layer=3, n_head=4, n_embd=64))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)

    def batch(src, g=None):
        ix = torch.randint(len(src) - BLOCK - 1, (BATCH,), generator=g)
        return (torch.stack([src[i:i + BLOCK] for i in ix]),
                torch.stack([src[i + 1:i + BLOCK + 1] for i in ix]))

    t0 = time.time()
    for step in range(STEPS):
        for gr in opt.param_groups:      # cosine decay, no warmup needed here
            gr["lr"] = LR * 0.5 * (1 + math.cos(math.pi * step / STEPS))
        _, loss = model(*batch(tr))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    # Validation: nats per token, averaged over fixed batches.
    model.eval()
    g = torch.Generator().manual_seed(123)
    with torch.no_grad():
        nats = sum(model(*batch(va, g))[1].item() for _ in range(40)) / 40
    # Total bits for the validation text = bits/token * tokens; divide by bytes.
    bpb = nats / math.log(2) * len(va) / val_bytes
    print(f"{name:5s} vocab {vocab_size:4d}  params {count_params(model):7,}  "
          f"val tokens {len(va):6,}  bytes/token {val_bytes / len(va):.2f}  "
          f"context {BLOCK} tok = {BLOCK * val_bytes / len(va):5.0f} bytes  "
          f"loss {nats:.3f} nats/token = {bpb:.3f} bits/byte  ({time.time() - t0:.0f} s)")


print(f"{STEPS} steps x {BATCH} sequences x {BLOCK} tokens for both runs\n")
run("char", *char_tokenizer())
run("bpe", *bpe_tokenizer())
