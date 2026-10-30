"""A CLIP-style contrastive image-text model on synthetic data.

Each 3x8x16 image holds two coloured shapes (left half, right half); its caption is the
5-token sequence "<colour> <shape> and <colour> <shape>", left object first.
The object "red x" never appears in training (held out to test composition).
Both encoders are trained with the symmetric InfoNCE loss and a learnable temperature.
Three variants:
  bag-of-words text encoder (mean of token embeddings: word order is invisible),
  transformer text encoder (with position embeddings),
  transformer + hard negatives (each image also sees its caption with colours swapped
  and with the objects swapped, as extra wrong answers).
"""
import itertools
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(4)
gen = torch.Generator().manual_seed(0)
COLOURS = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0],
                        [0, 1, 1], [1, 0, 1], [1, 1, 1], [1, .5, 0]], dtype=torch.float32)
NAMES = ["red", "green", "blue", "yellow", "cyan", "magenta", "white", "orange",
         "square", "plus", "x", "dot", "hbar", "vbar", "ell", "tee", "and"]
NC = NS = 8
AND = 16
S = torch.zeros(NS, 5, 5)
S[0, [0, 4], :] = 1; S[0, :, [0, 4]] = 1                       # square outline
S[1, 2, :] = 1; S[1, :, 2] = 1                                 # plus
S[2, range(5), range(5)] = 1; S[2, range(5), range(4, -1, -1)] = 1   # x
S[3, 1:4, 1:4] = 1                                             # dot (filled 3x3)
S[4, 1:3, :] = 1; S[5, :, 1:3] = 1                             # thick horizontal / vertical bar
S[6, :, 0] = 1; S[6, 4, :] = 1                                 # L
S[7, 0, :] = 1; S[7, :, 2] = 1                                 # T
RED, X = 0, 2


def make_pairs(n, allow_red_x):
    objs = [(c, s) for c in range(NC) for s in range(NS) if allow_red_x or (c, s) != (RED, X)]
    imgs, caps = torch.zeros(n, 3, 8, 16), torch.zeros(n, 5, dtype=torch.long)
    for i in range(n):
        o = torch.randint(0, len(objs), (2,), generator=gen)
        (c1, s1), (c2, s2) = objs[o[0]], objs[o[1]]
        for side, (c, s) in enumerate(((c1, s1), (c2, s2))):
            r, q = torch.randint(0, 4, (2,), generator=gen).tolist()   # position jitter
            imgs[i, :, r:r + 5, 8 * side + q:8 * side + q + 5] = COLOURS[c][:, None, None] * S[s]
        caps[i] = torch.tensor([c1, NC + s1, AND, c2, NC + s2])
    return imgs + 0.1 * torch.randn(imgs.shape, generator=gen), caps


def colour_swap(caps):
    return caps[:, [3, 1, 2, 0, 4]]          # same words, colours exchanged between objects


def order_swap(caps):
    return caps[:, [3, 4, 2, 0, 1]]          # same words, objects exchanged (left <-> right)


class ImageEncoder(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                 nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                 nn.Flatten(), nn.Linear(64 * 2 * 4, d))

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class TextEncoder(nn.Module):
    def __init__(self, bag_of_words, d=64):
        super().__init__()
        self.bow = bag_of_words
        self.tok = nn.Embedding(len(NAMES), d)
        self.pos = nn.Parameter(0.02 * torch.randn(1, 5, d))
        layer = nn.TransformerEncoderLayer(d, 4, 2 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.out = nn.Linear(d, d)

    def forward(self, caps):
        h = self.tok(caps)
        h = h.mean(1) if self.bow else self.tf(h + self.pos).mean(1)
        return F.normalize(self.out(h), dim=-1)


def train(bag_of_words, hard_negatives, Xtr, Ctr, epochs=20, seed=0):
    torch.manual_seed(seed)
    img_enc, txt_enc = ImageEncoder(), TextEncoder(bag_of_words)
    logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))   # CLIP's initial temperature
    params = list(img_enc.parameters()) + list(txt_enc.parameters()) + [logit_scale]
    opt = torch.optim.Adam(params, lr=1e-3)
    for _ in range(epochs):
        for b in torch.randperm(len(Xtr)).split(128):
            zi, zt = img_enc(Xtr[b]), txt_enc(Ctr[b])
            scale = logit_scale.exp().clamp(max=100)
            logits = scale * zi @ zt.T                     # (B, B): image i vs caption j
            target = torch.arange(len(b))
            i2t = logits
            if hard_negatives:                             # extra wrong captions for each image
                zc, zo = txt_enc(colour_swap(Ctr[b])), txt_enc(order_swap(Ctr[b]))
                i2t = torch.cat([logits, scale * (zi * zc).sum(1, keepdim=True),
                                 scale * (zi * zo).sum(1, keepdim=True)], dim=1)
            loss = (F.cross_entropy(i2t, target) + F.cross_entropy(logits.T, target)) / 2
            opt.zero_grad()
            loss.backward()
            opt.step()
    return img_enc, txt_enc, logit_scale.exp().item()


def pairwise_win(zi, zt_true, zt_alt):
    """Fraction of images whose true caption beats the altered one (ties count 1/2)."""
    d = (zi * zt_true).sum(1) - (zi * zt_alt).sum(1)
    return ((d > 1e-5).float() + 0.5 * (d.abs() <= 1e-5).float()).mean().item()


Xtr, Ctr = make_pairs(10000, allow_red_x=False)
Xte, Cte = make_pairs(3000, allow_red_x=True)
has_red_x = ((Cte[:, 0] == RED) & (Cte[:, 1] == NC + X)) | ((Cte[:, 3] == RED) & (Cte[:, 4] == NC + X))
all_caps = torch.tensor([[c1, NC + s1, AND, c2, NC + s2]
                         for c1, s1, c2, s2 in itertools.product(range(8), repeat=4)])
print(f"test images: {len(Xte)}, of which {int(has_red_x.sum())} contain the unseen 'red x'")
print("zero-shot = pick the best of all 4,096 captions; swap tests compare true vs altered caption\n")
print(f"{'text encoder':<26} {'temp':>5} {'0-shot seen':>11} {'0-shot red x':>12}"
      f" {'colour swap':>11} {'order swap':>10}")
for name, bow, hard in (("bag of words", True, False), ("transformer", False, False),
                        ("transformer + hard negs", False, True)):
    img_enc, txt_enc, scale = train(bow, hard, Xtr, Ctr)
    with torch.no_grad():
        zi, zt = img_enc(Xte), txt_enc(Cte)
        pred = (zi @ txt_enc(all_caps).T).argmax(1)
        correct = (all_caps[pred] == Cte).all(1).float()
        diff_colour = Cte[:, 0] != Cte[:, 3]
        diff_obj = (Cte[:, [0, 1]] != Cte[:, [3, 4]]).any(1)
        cs = pairwise_win(zi[diff_colour], zt[diff_colour], txt_enc(colour_swap(Cte[diff_colour])))
        os_ = pairwise_win(zi[diff_obj], zt[diff_obj], txt_enc(order_swap(Cte[diff_obj])))
    print(f"{name:<26} {1 / scale:>5.3f} {correct[~has_red_x].mean():>11.3f}"
          f" {correct[has_red_x].mean():>12.3f} {cs:>11.3f} {os_:>10.3f}")
