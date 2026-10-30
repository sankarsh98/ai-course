---
title: "Vision and Multimodal Models"
---

> **Level 3 · Module 4** · Prerequisites: L2-03, L2-05, L2-06 · Time: 7 to 9 hours · Code: `code/L3-04-vision-and-multimodal-models/`

## What you will be able to do

- Turn an image into a token sequence, explain what a vision transformer gives up relative to a CNN, and measure it on small data.
- Explain the bookkeeping of detection and segmentation (IoU, non-maximum suppression, average precision) and how set-prediction and promptable models changed it.
- Train a CLIP-style contrastive image-text model and test the claims usually made for it, including zero-shot transfer and compositional understanding.
- Describe the main ways vision-language models connect an image encoder to a language model, and their documented weaknesses.
- Explain how speech becomes a sequence a transformer can read, and estimate the token budget of images, audio and video.

## 1. Images as token sequences: the vision transformer

L2-05 built transformers over sequences of token vectors. To apply one to an image, cut the image into non-overlapping $P \times P$ patches, flatten each to a vector of length $P^2 C$ (for $C$ channels), and multiply by a learned matrix to get a $D$-dimensional **patch embedding**. A $224 \times 224$ image with $P = 16$ becomes $14 \times 14 = 196$ tokens, hence the title of the vision transformer (ViT) paper, "An Image is Worth 16x16 Words" [Dosovitskiy et al., 2021]. ViT then follows BERT: it prepends a learnable `[class]` token whose final state feeds the classifier, and adds **learned 1D position embeddings**. The authors reported no significant gain from 2D-aware position embeddings.

The patch embedding is exactly a convolution whose stride equals its kernel size: each output is a dot product of the kernel with one patch, and patches do not overlap. Script 01 checks this; the two computations agree to $5 \times 10^{-7}$. So the only convolution left in a ViT is this first one, and after it every token can attend to every other from the first layer.

**What is given up.** A CNN builds in locality and translation equivariance (L2-03). A ViT has neither beyond the patch grid; its position embeddings must learn which patches are neighbours. Dosovitskiy et al. were explicit about the consequence: trained on mid-sized data such as ImageNet (1.3M images) without strong regularization, ViTs came out a few percentage points below comparable ResNets. Pretrained on ImageNet-21k (14M images) or the internal JFT-300M (303M images) and then transferred, they matched or beat state-of-the-art CNNs, with ViT-H/14 reaching 88.55% on ImageNet. Model sizes were 86M (Base), 307M (Large) and 632M (Huge) parameters. L2-03 noted that modernized CNNs remain competitive; which family wins depends on data, compute and task.

Script 01 shows the same effect at toy scale on the 8×8 digits, with 2×2 patches (16 tokens) and a 2-layer ViT (69,194 parameters) against a small CNN (21,386 parameters), mean of 3 seeds:

| training images | ViT | ViT, no position embeddings | CNN |
|---|---|---|---|
| 200 | 0.778 | 0.509 | 0.935 |
| 1,257 | 0.970 | 0.741 | 0.985 |

With 200 images the CNN wins by 16 points; with all 1,257 the gap is 1.5 points. And without position embeddings a ViT is **permutation invariant** over patches (self-attention is permutation equivariant, L2-05, and the `[class]` readout ignores order): shuffling all 16 patches with a fixed permutation leaves its predictions 100% unchanged, while the model with position embeddings drops from 0.972 to 0.352 accuracy on shuffled inputs. That a bag of patches still classifies 74% of digits says more about this easy dataset than about vision.

**Cost.** Attention cost grows with the square of the token count (L2-05). Halving the patch size quadruples the tokens and multiplies attention scores by 16: script 01 prints 38,416 scores per head per layer at 224 px with 16 px patches, 614,656 with 8 px patches, and 16.8 million at 1024 px. This is why high-resolution vision models merge tokens, use windowed attention, or restrict resolution, and why the number of visual tokens is a central design choice in multimodal models (section 6).

## 2. Detection and segmentation

Classification assigns one label per image. **Object detection** outputs a set of boxes, each with a class and a confidence. **Semantic segmentation** labels every pixel with a class; **instance segmentation** separates individual objects; **panoptic segmentation** does both. The bookkeeping is shared:

- **Intersection over union (IoU)** of two boxes or masks is the area of their overlap divided by the area of their union. Shifting a 10×10 box by half its width gives IoU 0.333; shifting by one pixel diagonally gives 0.681 (script 03). IoU is unforgiving for small objects.
- **Non-maximum suppression (NMS)**: detectors fire many overlapping boxes per object. Greedy NMS keeps the highest-scoring box, deletes those overlapping it above a threshold, and repeats.
- **Average precision (AP)**: sort detections by score; a detection is a true positive if it matches a not-yet-matched ground-truth box with IoU above a threshold, otherwise a false positive (a second detection of the same object counts as false). AP is the area under the resulting precision-recall curve (L1-01). COCO-style **AP@[.5:.95]** averages AP over IoU thresholds from 0.5 to 0.95, rewarding tight localization; mAP averages over classes.

Script 03 simulates a detector that fires 2 to 5 jittered boxes per object, misses 10% of objects and adds background false positives:

| NMS threshold | boxes kept | AP@0.5 | AP@0.75 | AP@[.5:.95] |
|---|---|---|---|---|
| none | 1756 | 0.663 | 0.461 | 0.401 |
| 0.3 | 608 | 0.880 | 0.428 | 0.478 |
| 0.5 | 648 | 0.902 | 0.440 | 0.492 |
| 0.7 | 1041 | 0.770 | 0.502 | 0.449 |

Without NMS, duplicates are false positives and AP@0.5 is 0.663; NMS at 0.5 raises it to 0.902. But AP@0.75 is *highest with little suppression*: in this simulation scores are unrelated to localization quality, so NMS often keeps a confident but sloppy box and deletes a better-placed one. Real detectors learn scores that partly track localization, but the lesson holds: NMS is a hand-tuned heuristic whose threshold interacts with the metric, and crowded scenes (overlapping true objects) make it worse.

**Set prediction.** Classic detectors (two-stage designs such as Faster R-CNN that propose regions and then classify them, and one-stage designs that predict from a dense grid of **anchor** boxes) all need NMS. **DETR** recast detection as direct set prediction [Carion et al., 2020]: a transformer decoder takes a fixed set of learned object queries, predicts all boxes in parallel, and is trained with a loss that matches predictions to ground-truth objects one-to-one (bipartite matching), so duplicates are penalized during training and NMS and anchors are no longer needed. The authors reported accuracy and speed comparable to a Faster R-CNN baseline on COCO, and extended it to panoptic segmentation.

**Promptable segmentation.** Segment Anything (SAM) defined a task in which a model returns a mask for a *prompt* (points, a box, or a rough mask), trained a model for it, and built the SA-1B dataset of over 1 billion masks on 11 million licensed images using the model in its own annotation loop [Kirillov et al., 2023]. They reported zero-shot performance often competitive with, and sometimes better than, fully supervised results on new image distributions. The broader shift: instead of a fixed list of classes, detection and segmentation increasingly take open-ended prompts, often text embedded with a CLIP-style encoder (section 3).

## 3. Contrastive image-text pretraining (CLIP)

Labelled datasets fix the vocabulary of concepts in advance. CLIP instead learns from (image, caption) pairs collected from the web [Radford et al., 2021]. An image encoder and a text encoder each map their input to a normalized vector. For a batch of $N$ pairs with image vectors $u_i$ and text vectors $v_j$, the logits are $s_{ij} = u_i \cdot v_j / \tau$, and the loss is the average of two cross-entropies, image-to-text (row $i$ should pick column $i$) and text-to-image:

$$\mathcal{L} = \frac{1}{2N}\sum_{i=1}^N \Big[-\log \frac{e^{s_{ii}}}{\sum_j e^{s_{ij}}} - \log \frac{e^{s_{ii}}}{\sum_j e^{s_{ji}}}\Big].$$

This is the InfoNCE loss of L2-06 across two modalities; each caption is a positive for its image and the other $N - 1$ captions are negatives. The temperature $\tau$ is learned: CLIP initialized it to the equivalent of 0.07 and clipped the logit scale at 100. Training used 400 million pairs (the WIT dataset) and a minibatch of 32,768. The authors found that predicting the exact caption words learned slowly; predicting a bag of words was about 3 times faster, and swapping that predictive objective for the contrastive one gave a further 4 times improvement in the rate of zero-shot transfer to ImageNet.

**Zero-shot classification.** To classify into new classes, embed a prompt such as "A photo of a {label}." for each class and pick the class whose text vector is closest to the image vector. The best CLIP model (ViT-L/14 at 336 px) reached 76.2% zero-shot on ImageNet, matching the original ResNet-50 without using any of its 1.28M training labels. Prompt wording matters: the single template above added 1.3 points on ImageNet, and an ensemble of 80 prompts another 3.5. Radford et al. also listed limitations: weak performance on systematic tasks such as counting objects, and only 88% on MNIST handwritten digits, a reminder that "zero-shot" works when web captions cover the concept.

**What contrastive training does not guarantee.** A caption only has to be closer to its image than the other captions in the batch. If the negatives rarely differ by word order or attribute binding, the model is not forced to encode them. Yuksekgonul et al. built a benchmark of over 50,000 test cases and found that leading models handled relations poorly, often attached attributes to the wrong objects, and were insensitive to word order; they argued that standard retrieval benchmarks can be solved without composition, and proposed composition-aware hard negatives as a fix [Yuksekgonul et al., 2023].

Script 02 tests these ideas on synthetic data where the truth is known. Each image has two coloured shapes (left and right); the caption is `<colour> <shape> and <colour> <shape>`, left first, from 8 colours and 8 shapes (4,096 possible captions). The object "red x" never appears in training. Results on 3,000 test images:

| text encoder | zero-shot, seen objects | zero-shot, contains "red x" | colour swap | order swap |
|---|---|---|---|---|
| bag of words | 0.315 | 0.247 | 0.500 | 0.500 |
| transformer | 0.903 | 0.558 | 0.999 | 0.988 |
| transformer + hard negatives | 0.999 | 0.701 | 1.000 | 1.000 |

"Zero-shot" picks the best of all 4,096 captions; the swap tests ask whether the true caption scores above the same words with colours exchanged between objects, or with the objects' order exchanged (0.5 is chance).

- **A bag-of-words text encoder cannot bind.** "red x and blue dot" and "blue x and red dot" contain the same words, so their embeddings are identical and the swap tests are exactly at chance. This is an architectural limit, not a training problem.
- **An order-aware encoder learned binding here without help.** In this clean toy every caption describes its image exactly, in a fixed format, with a small vocabulary, so negatives that share most of their words with the true caption are common and the plain contrastive loss already forced binding (0.999). Do not read this as contradicting Yuksekgonul et al.: their point concerns web data, where captions are loose and such near-miss negatives are rare. The toy shows the mechanism, not the real-world frequency.
- **Hard negatives still helped**, raising exact-caption accuracy from 0.903 to 0.999.
- **Composition is partial.** Images containing the never-seen "red x" are described exactly only 56% to 70% of the time, against 90% to 100% for seen objects, although "red" and "x" were each seen often.

## 4. Vision-language models

A **vision-language model (VLM)** takes images and text and generates text: answer questions, read documents, describe scenes. Three ways of connecting vision to a language model dominate the published work:

1. **Encoder, projector, language model.** A pretrained image encoder (often CLIP-style) turns the image into patch vectors; a small projector maps them into the language model's embedding space; the visual tokens are placed in the sequence alongside text tokens. LLaVA connected a CLIP vision encoder to the Vicuna language model this way and trained on instruction-following data generated by the text-only GPT-4 from image descriptions [Liu et al., 2023]. Its recipe, as summarized by Hugging Face, first trains only the projector with both networks frozen and then unfreezes the language model [merve and Beeching, 2024].
2. **Cross-attention into a frozen language model.** Flamingo (Alayrac et al., NeurIPS 2022, arXiv:2204.14198) kept a vision encoder and a language model frozen, compressed visual features into a fixed set of 64 visual outputs with a "Perceiver Resampler", and inserted new gated cross-attention layers between the frozen language-model layers, so text tokens attend to the image without visual tokens entering the sequence [Huyen, 2023]. This handles interleaved images and text and supports few-shot prompting.
3. **Early fusion.** Chameleon converts images into discrete tokens and trains a single transformer on interleaved image and text tokens from the start, so the same model can both read and generate images [Chameleon Team, 2024]. The authors report that this required changes to keep training stable.

**As of October 2026**, the most detailed public description of a leading open VLM family is the Qwen3-VL report [Qwen Team, 2025]: a SigLIP-2 vision encoder that accepts images at their native resolution (so the number of visual tokens varies with image size), a two-layer MLP that merges each 2×2 block of patch features into one token, an LLM backbone (dense models from 2B to 32B parameters and mixture-of-experts models up to 235B total with 22B active), visual features injected from three encoder levels into the first LLM layers ("DeepStack"), and an interleaved multi-axis rotary position scheme for time, height and width, with contexts up to 256K tokens. That is pattern 1 with engineering refinements. Some newer families describe early-fusion training instead: Qwen's Qwen3.5 model cards (February 2026) state "early fusion training on multimodal tokens", but with few published details, so treat such descriptions as claims until a technical report documents them. Closed commercial models publish little about their architectures.

**Documented weaknesses.** Tong et al. collected "CLIP-blind" pairs, images that CLIP embeds as similar although they differ visibly, and built the MMVP benchmark from nine basic visual patterns. State-of-the-art systems including GPT-4V often answered these simple questions wrongly and gave hallucinated explanations, and the patterns that fooled CLIP also tended to fool the VLMs built on it [Tong et al., 2024]. The general lessons: a VLM can only reason about what its visual tokens encode; fluent text is not evidence that the model saw the detail; and VLM benchmarks (multiple-choice question sets, leaderboards) inherit every weakness discussed in L4-03, including contamination and answerability from text alone.

## 5. Speech and audio

Raw audio at 16 kHz is 16,000 numbers per second, too long for attention. The standard front end turns it into a **log-mel spectrogram**: slide a short window along the signal, take the power spectrum of each window with an FFT, pool the frequencies with triangular filters spaced on the perceptual mel scale, and take logs. Whisper uses 25 ms windows every 10 ms with 80 mel channels [Radford et al., 2023]. Script 04 implements this in NumPy: 2 seconds of audio become a 198×80 array, and the loudest band follows a 440 Hz tone (452 Hz band) and then a rising chirp (732, 1316 and 2139 Hz bands at 1.2, 1.5 and 1.9 s, against true frequencies of 700, 1300 and 2100 Hz). A spectrogram is an image-like grid, so the patch and convolution ideas of section 1 carry over.

Three approaches to speech recognition matter conceptually:

- **CTC.** Connectionist temporal classification (Graves, Fernández, Gomez and Schmidhuber, ICML 2006) trains a network to emit, for every frame, a character or a blank, and sums over all alignments consistent with the transcript, so no frame-level segmentation is needed.
- **Self-supervised pretraining.** wav2vec 2.0 (Baevski et al., 2020, arXiv:2006.11477) learns speech representations from unlabeled audio with a contrastive task, then fine-tunes with little labelled data; it reported word error rates of 1.8 and 3.3 on the LibriSpeech clean and other test sets with all labelled data, and still 4.8 and 8.2 with only 10 minutes of labels after pretraining on 53,000 hours of unlabelled audio.
- **Large weakly supervised encoder-decoders.** Whisper trained an encoder-decoder transformer on 680,000 hours of multilingual transcribed web audio [Radford et al., 2023]. The encoder starts with two width-3 convolutions, the second with stride 2, so a 30-second chunk (3,000 frames) becomes 1,500 encoder positions. The decoder is a language model that writes the transcript, and special tokens select the language, the task (transcribe or translate) and timestamps. The authors document failure modes including transcripts entirely unrelated to the audio ("hallucination") and repetition loops, which decoding heuristics reduce but do not remove.

Generating audio usually reverses the picture: models produce spectrograms or discrete audio codes that a separate decoder turns into waveforms, using the same diffusion, flow or autoregressive machinery as images (L3-03) and text (L3-01).

## 6. Putting modalities in one sequence

Every multimodal system answers the same three questions: how to turn each modality into vectors, how many vectors to spend, and where in the network they meet the text.

**Token budgets.** Script 04 does the arithmetic: a 336×336 image in 14 px patches is 576 tokens (144 after 2×2 merging); 30 s of audio is 1,500 Whisper encoder positions; one minute of video sampled at a hypothetical 2 frames per second at 576 tokens per frame is 69,120 tokens. Text is cheap by comparison. This is why VLMs merge tokens, compress video frames, and choose resolution adaptively: visual tokens dominate cost and context length.

**Continuous versus discrete.** Most understanding models feed *continuous* encoder outputs to the language model through a projector. Early-fusion models that also *generate* images need *discrete* image tokens from a learned tokenizer, so that "predict the next token" applies to every modality; the cost is that tokenization loses detail.

**Positions.** Text positions are one-dimensional. Images need two dimensions and video three; Qwen3-VL's interleaved rotary scheme is one answer, and the ViT result that 1D learned embeddings sufficed for classification is a reminder that the right choice depends on the task.

**Where fusion happens.** Late (separate encoders, combined only through a similarity, as in CLIP: ideal for retrieval and zero-shot classification), middle (visual tokens or cross-attention inside a language model, as in LLaVA, Qwen3-VL and Flamingo), or early (one model over all tokens, as in Chameleon). Later fusion reuses strong pretrained parts cheaply; earlier fusion lets modalities interact at every layer but needs expensive joint training. Published evidence comparing them at equal cost is limited, so be wary of strong claims either way.

## Hands-on

Run each script from the repository root, for example `.venv/bin/python code/L3-04-vision-and-multimodal-models/01_vit_patches.py`. Outputs come from Python 3.12, PyTorch 2.15.0 (CPU, 4 threads), NumPy 2.5.4 and scikit-learn 1.9.1 on Linux (WSL2), on a machine shared with other jobs. Wall times: script 01 took 31 to 52 s and script 02 31 to 47 s depending on load; 03 and 04 take about a second. Seeded results reproduced exactly across repeated runs. The digits data ships with scikit-learn; everything else is synthetic.

### Script 01: patch embeddings and a tiny ViT

```python
"""Patch embeddings and a tiny vision transformer on the 8x8 digits (bundled with scikit-learn).

1. Patch embedding = a linear map on flattened patches = a convolution with stride = kernel.
2. Token counts and attention cost for common image and patch sizes.
3. Tiny ViT (with and without position embeddings) vs a small CNN, at two training-set sizes.
4. Without position embeddings a ViT cannot tell where a patch came from: shuffling patches
   leaves its prediction unchanged.
"""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

torch.set_num_threads(4)
torch.manual_seed(0)

# ---- 1. Patch embedding three ways ------------------------------------------------
P, C, D = 4, 3, 32
img = torch.randn(2, C, 16, 16)
conv = nn.Conv2d(C, D, kernel_size=P, stride=P)
patches = img.unfold(2, P, P).unfold(3, P, P)                  # (B, C, 4, 4, P, P)
patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(2, 16, C * P * P)
linear_out = patches @ conv.weight.reshape(D, -1).T + conv.bias
conv_out = conv(img).flatten(2).transpose(1, 2)                # (B, 16 tokens, D)
print(f"patchify + linear vs conv(stride=kernel): max abs diff {(linear_out - conv_out).abs().max():.2e}")

# ---- 2. Token counts --------------------------------------------------------------
print("\nimage side, patch -> tokens, attention scores per head per layer")
for side, p in ((224, 16), (224, 14), (336, 14), (224, 8), (1024, 16)):
    n = (side // p) ** 2
    print(f"  {side:4d}px, {p:2d}px patches -> {n:5d} tokens, {n * n:>12,d} scores")

# ---- 3. Tiny ViT vs CNN on digits -------------------------------------------------
X, y = load_digits(return_X_y=True)
X = torch.tensor(X / 16.0, dtype=torch.float32).view(-1, 1, 8, 8)
y = torch.tensor(y)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)


class TinyViT(nn.Module):
    def __init__(self, use_pos=True, patch=2, dim=64, depth=2, heads=4):
        super().__init__()
        n = (8 // patch) ** 2
        self.embed = nn.Conv2d(1, dim, patch, stride=patch)    # patch embedding
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))         # [class] token, as in ViT
        self.pos = nn.Parameter(0.02 * torch.randn(1, n + 1, dim)) if use_pos else None
        layer = nn.TransformerEncoderLayer(dim, heads, 2 * dim, dropout=0.0,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 10))

    def tokens(self, x):
        return self.embed(x).flatten(2).transpose(1, 2)        # (B, N, dim)

    def forward(self, x=None, tok=None):
        tok = self.tokens(x) if tok is None else tok
        h = torch.cat([self.cls.expand(len(tok), -1, -1), tok], dim=1)
        if self.pos is not None:
            h = h + self.pos
        return self.head(self.encoder(h)[:, 0])


def small_cnn():
    return nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Flatten(), nn.Linear(64 * 4, 10))


def train(model, Xs, ys, epochs, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.05)
    steps = epochs * ((len(Xs) + 63) // 64)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps)
    for _ in range(epochs):
        for b in torch.randperm(len(Xs)).split(64):
            loss = F.cross_entropy(model(Xs[b]), ys[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    return model


def acc(model, Xs, ys):
    with torch.no_grad():
        return (model(Xs).argmax(1) == ys).float().mean().item()


n_params = lambda m: sum(p.numel() for p in m.parameters())
print(f"\nparameters: ViT {n_params(TinyViT()):,}, CNN {n_params(small_cnn()):,}")
print(f"{'train images':>12} {'ViT + pos':>10} {'ViT, no pos':>12} {'CNN':>6}   (test accuracy, mean of 3 seeds)")
models = {}
t0 = time.time()
for n_train in (200, len(Xtr)):
    epochs = 60 if n_train == 200 else 30                       # similar number of updates
    row = []
    for make in (lambda: TinyViT(True), lambda: TinyViT(False), small_cnn):
        scores = []
        for seed in range(3):
            torch.manual_seed(seed)
            m = train(make(), Xtr[:n_train], ytr[:n_train], epochs, seed)
            scores.append(acc(m, Xte, yte))
        row.append(np.mean(scores))
        models.setdefault(n_train, []).append(m)
    print(f"{n_train:>12} {row[0]:>10.3f} {row[1]:>12.3f} {row[2]:>6.3f}")
print(f"(training took {time.time() - t0:.0f}s)")

# ---- 4. Shuffle the 16 patch tokens with one fixed permutation ---------------------
vit_pos, vit_nopos, _ = models[len(Xtr)]
perm = torch.randperm(16)
with torch.no_grad():
    for name, m in (("ViT + pos", vit_pos), ("ViT, no pos", vit_nopos)):
        tok = m.tokens(Xte)
        same = (m(tok=tok).argmax(1) == m(tok=tok[:, perm]).argmax(1)).float().mean()
        acc_shuf = (m(tok=tok[:, perm]).argmax(1) == yte).float().mean()
        print(f"{name:<12} (seed 2) accuracy {acc(m, Xte, yte):.3f}; patches shuffled:"
              f" same prediction {same:.3f}, accuracy {acc_shuf:.3f}")
```

```text
patchify + linear vs conv(stride=kernel): max abs diff 4.77e-07

image side, patch -> tokens, attention scores per head per layer
   224px, 16px patches ->   196 tokens,       38,416 scores
   224px, 14px patches ->   256 tokens,       65,536 scores
   336px, 14px patches ->   576 tokens,      331,776 scores
   224px,  8px patches ->   784 tokens,      614,656 scores
  1024px, 16px patches ->  4096 tokens,   16,777,216 scores

parameters: ViT 69,194, CNN 21,386
train images  ViT + pos  ViT, no pos    CNN   (test accuracy, mean of 3 seeds)
         200      0.778        0.509  0.935
        1257      0.970        0.741  0.985
(training took 49s)
ViT + pos    (seed 2) accuracy 0.972; patches shuffled: same prediction 0.356, accuracy 0.352
ViT, no pos  (seed 2) accuracy 0.754; patches shuffled: same prediction 1.000, accuracy 0.754
```

### Script 02: a CLIP-style toy

```python
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
```

```text
test images: 3000, of which 77 contain the unseen 'red x'
zero-shot = pick the best of all 4,096 captions; swap tests compare true vs altered caption

text encoder                temp 0-shot seen 0-shot red x colour swap order swap
bag of words               0.038       0.315        0.247       0.500      0.500
transformer                0.036       0.903        0.558       0.999      0.988
transformer + hard negs    0.039       0.999        0.701       1.000      1.000
```

### Script 03: IoU, NMS and average precision

```python
"""Detection bookkeeping from scratch: IoU, non-maximum suppression and average precision.

A simulated detector fires several jittered boxes around each true object (duplicates) and
some boxes on background, each with a confidence score. We then measure how the NMS
threshold and the IoU threshold used for matching change average precision (AP).
Boxes are (x1, y1, x2, y2).
"""
import numpy as np

rng = np.random.default_rng(0)


def iou(a, b):
    """IoU between every box in a (n, 4) and every box in b (m, 4)."""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter)


def nms(boxes, scores, thresh):
    """Greedy NMS: keep the best box, drop boxes overlapping it by more than thresh, repeat."""
    order, keep = np.argsort(-scores), []
    while len(order):
        i = order[0]
        keep.append(i)
        order = order[1:][iou(boxes[[i]], boxes[order[1:]])[0] <= thresh]
    return np.array(keep, dtype=int)


def average_precision(dets, gts, iou_thr):
    """dets: list over images of (boxes, scores); gts: list of true boxes. One class."""
    records, n_gt = [], sum(len(g) for g in gts)
    for (boxes, scores), g in zip(dets, gts):
        matched = np.zeros(len(g), bool)
        for j in np.argsort(-scores):                  # greedy, highest score first
            ious = iou(boxes[[j]], g)[0] if len(g) else np.array([])
            k = int(ious.argmax()) if len(ious) else -1
            tp = k >= 0 and ious[k] >= iou_thr and not matched[k]
            if tp:
                matched[k] = True                      # a second hit on the same object is a FP
            records.append((scores[j], tp))
    records.sort(key=lambda r: -r[0])
    tp = np.cumsum([r[1] for r in records]); fp = np.cumsum([not r[1] for r in records])
    recall, precision = tp / n_gt, tp / (tp + fp)
    # Area under the precision-recall curve, with precision made monotone (as in VOC/COCO).
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    r = np.concatenate([[0], recall])
    return float(np.sum((r[1:] - r[:-1]) * precision))


def simulate(n_images=200):
    gts, dets = [], []
    for _ in range(n_images):
        n_obj = rng.integers(1, 5)
        xy = rng.uniform(0, 80, (n_obj, 2)); wh = rng.uniform(10, 30, (n_obj, 2))
        g = np.hstack([xy, xy + wh])
        boxes, scores = [], []
        for b in g:
            if rng.random() < 0.1:                      # the detector misses 10% of objects
                continue
            for _ in range(rng.integers(2, 6)):        # several overlapping firings per object
                jitter = rng.normal(0, 0.08, 4) * np.r_[b[2:] - b[:2], b[2:] - b[:2]]
                boxes.append(b + jitter); scores.append(rng.uniform(0.5, 1.0))
        for _ in range(rng.integers(0, 3)):            # background false positives
            xy0 = rng.uniform(0, 80, 2)
            boxes.append(np.r_[xy0, xy0 + rng.uniform(10, 30, 2)]); scores.append(rng.uniform(0, 0.8))
        gts.append(g); dets.append((np.array(boxes).reshape(-1, 4), np.array(scores)))
    return gts, dets


a = np.array([[0, 0, 10, 10.]]); b = np.array([[5, 0, 15, 10.], [1, 1, 11, 11.], [20, 20, 30, 30.]])
print("IoU of a 10x10 box with: shifted half its width, shifted by 1 px diagonally, disjoint:",
      iou(a, b)[0].round(3))

gts, raw = simulate()
print(f"\n{len(gts)} images, {sum(len(g) for g in gts)} objects, {sum(len(d[0]) for d in raw)} raw detections")
print(f"{'NMS threshold':>14} {'boxes kept':>10} {'AP@0.5':>7} {'AP@0.75':>8} {'AP@[.5:.95]':>12}")
for thr in (None, 0.3, 0.5, 0.7, 0.9):
    dets = raw if thr is None else [(bx[nms(bx, sc, thr)], sc[nms(bx, sc, thr)]) for bx, sc in raw]
    aps = [average_precision(dets, gts, t) for t in np.arange(0.5, 0.96, 0.05)]
    print(f"{'none' if thr is None else thr:>14} {sum(len(d[0]) for d in dets):>10} {aps[0]:>7.3f}"
          f" {aps[5]:>8.3f} {np.mean(aps):>12.3f}")
```

```text
IoU of a 10x10 box with: shifted half its width, shifted by 1 px diagonally, disjoint: [0.333 0.681 0.   ]

200 images, 495 objects, 1756 raw detections
 NMS threshold boxes kept  AP@0.5  AP@0.75  AP@[.5:.95]
          none       1756   0.663    0.461        0.401
           0.3        608   0.880    0.428        0.478
           0.5        648   0.902    0.440        0.492
           0.7       1041   0.770    0.502        0.449
           0.9       1735   0.669    0.466        0.405
```

### Script 04: audio front end and token budgets

```python
"""From waveform to tokens: a log-mel spectrogram with Whisper's front-end settings.

16 kHz audio, 25 ms windows (400 samples) every 10 ms (160 samples), 80 mel bands.
Whisper's encoder then applies two convolutions, the second with stride 2, so 30 s of
audio becomes 3000 frames and then 1500 encoder positions. NumPy only.
"""
import numpy as np

SR, WIN, HOP, N_MELS, N_FFT = 16000, 400, 160, 80, 512


def mel_filterbank(n_mels, n_fft, sr):
    hz_to_mel = lambda f: 2595 * np.log10(1 + f / 700)          # one common mel formula
    mel_to_hz = lambda m: 700 * (10 ** (m / 2595) - 1)
    edges = mel_to_hz(np.linspace(0, hz_to_mel(sr / 2), n_mels + 2))
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    fb = np.zeros((n_mels, len(freqs)))
    for m in range(n_mels):                                     # triangular filters
        lo, mid, hi = edges[m], edges[m + 1], edges[m + 2]
        fb[m] = np.clip(np.minimum((freqs - lo) / (mid - lo), (hi - freqs) / (hi - mid)), 0, None)
    return fb, edges


def log_mel(wave):
    n_frames = 1 + (len(wave) - WIN) // HOP
    frames = np.stack([wave[i * HOP:i * HOP + WIN] for i in range(n_frames)]) * np.hanning(WIN)
    power = np.abs(np.fft.rfft(frames, n=N_FFT)) ** 2           # (frames, 257)
    fb, edges = mel_filterbank(N_MELS, N_FFT, SR)
    return np.log10(power @ fb.T + 1e-10), edges               # (frames, 80)


rng = np.random.default_rng(0)
t = np.arange(int(2.0 * SR)) / SR                               # 2 seconds
wave = np.where(t < 1.0, np.sin(2 * np.pi * 440 * t),           # 1 s of a 440 Hz tone, then
                np.sin(2 * np.pi * (300 * t + 1000 * (t - 1) ** 2)))  # a rising chirp
wave = wave + 0.01 * rng.standard_normal(len(t))
spec, edges = log_mel(wave)
centres = edges[1:-1]
print(f"2.0 s of audio = {len(wave)} samples -> log-mel {spec.shape} (frames x mel bands)")
print(f"mel band centres: first {centres[:3].round(0)} Hz ... last {centres[-2:].round(0)} Hz")
for sec in (0.5, 1.2, 1.5, 1.9):
    f = int(sec * SR / HOP)
    print(f"  t={sec:.1f}s  loudest band centre {centres[spec[f].argmax()]:6.0f} Hz")

print("\nsequence lengths a model has to attend over (arithmetic, not measured):")
whisper_frames = 30 * SR // HOP
print(f"  30 s of audio: {whisper_frames} frames -> {whisper_frames // 2} positions after the stride-2 conv")
print(f"  a 336x336 image in 14 px patches: {(336 // 14) ** 2} tokens; "
      f"after merging 2x2 patches: {(336 // 28) ** 2}")
fps, minutes = 2, 1
print(f"  {minutes} min of video at {fps} frames/s, 576 tokens per frame: {minutes * 60 * fps * 576:,} tokens")
```

```text
2.0 s of audio = 32000 samples -> log-mel (198, 80) (frames x mel bands)
mel band centres: first [22. 45. 68.] Hz ... last [7475. 7734.] Hz
  t=0.5s  loudest band centre    452 Hz
  t=1.2s  loudest band centre    732 Hz
  t=1.5s  loudest band centre   1316 Hz
  t=1.9s  loudest band centre   2139 Hz

sequence lengths a model has to attend over (arithmetic, not measured):
  30 s of audio: 3000 frames -> 1500 positions after the stride-2 conv
  a 336x336 image in 14 px patches: 576 tokens; after merging 2x2 patches: 144
  1 min of video at 2 frames/s, 576 tokens per frame: 69,120 tokens
```

## Common mistakes

- **Forgetting position embeddings, or breaking them.** Without them a ViT sees a bag of patches (script 01). Changing resolution changes the patch grid, so learned embeddings must be interpolated.
- **Training a ViT from scratch on a small dataset** and concluding transformers do not work for vision. Use a pretrained model, a CNN, or strong augmentation.
- **Comparing detection numbers with different AP definitions.** AP@0.5, AP@0.75 and AP@[.5:.95] answer different questions; so do VOC-style and COCO-style evaluation.
- **Tuning the NMS threshold on the test set**, or ignoring that it trades duplicates against crowded objects.
- **Trusting zero-shot accuracy as general understanding.** It depends on prompt wording, and swap tests or benchmarks like ARO and MMVP probe what retrieval scores hide.
- **Assuming the text encoder sees word order.** Check whether swapped captions get different scores.
- **Believing a fluent VLM answer.** Ask about details that would change the answer, and compare with the image.
- **Ignoring the token budget.** High-resolution images and video can dominate latency and context; measure tokens per input.

## Exercises

1. [warm-up] How many tokens does a ViT produce for a 384×384 image with 16 px patches? How many attention scores per head per layer, counting the `[class]` token?
2. [warm-up] Compute by hand the IoU of boxes (0, 0, 4, 4) and (2, 2, 6, 6).
3. [warm-up] Why do the colour-swap and order-swap tests give exactly 0.5 for the bag-of-words encoder?
4. [core] In script 01, replace learned position embeddings with fixed 2D sinusoidal ones. Compare accuracy at 200 training images.
5. [core] In script 01, use 4×4 patches (4 tokens) and 1×1 patches (64 tokens). How do accuracy and training time change?
6. [core] In script 03, make scores correlate with localization quality (for example, score = IoU with the true box plus noise). Does NMS still hurt AP@0.75?
7. [core] In script 02, train the transformer encoder on 2,000 pairs instead of 10,000. Which column degrades first?
8. [core] In script 02, describe objects in random order in the captions (so order no longer encodes position). What happens to the order-swap test, and why is that the correct behaviour?
9. [stretch] Add zero-shot "attribute probes" to script 02: score prompts like `red <any shape> and <anything>` by averaging embeddings over the matching captions, and measure left-colour accuracy.
10. [stretch] Implement DETR-style matching for script 03: given a fixed number of predictions per image, find the one-to-one assignment to ground truth (use `scipy.optimize.linear_sum_assignment` on a cost of $1 - \mathrm{IoU}$) and count unmatched predictions as "no object".
11. [stretch] Build a log-mel spectrogram for 3 s of audio, cut it into 16×16 patches, and count tokens. Compare with the Whisper encoder's positions for the same audio.

### Hints and answers

1. $24^2 = 576$ patch tokens, 577 with `[class]`, so $577^2 = 332{,}929$ scores.
2. Intersection 2×2 = 4, union 16 + 16 − 4 = 28, IoU = 1/7 ≈ 0.143.
3. Mean-pooled token embeddings depend only on the multiset of tokens; the swapped captions have the same multiset, so the scores tie and ties count 1/2.
4. Fixed 2D embeddings encode neighbourhood for free, which may help with 200 images; ViT found no gain at large scale, which is a different regime.
5. Fewer, larger patches are cheaper but coarser; 64 tokens make attention 16 times costlier than 16. Measure rather than guess the accuracy.
6. If scores track localization, NMS tends to keep the better box, and the AP@0.75 penalty should shrink.
7. Expect the held-out "red x" column and exact zero-shot to fall before the swap tests, but measure it.
8. The order-swap test should fall toward 0.5, because the caption no longer says which object is on the left; a model that "fails" here is right not to guess.
9. Average the text embeddings of all captions with left colour red, and similarly for the other colours; classify each image by the closest average.
10. `linear_sum_assignment` returns the minimum-cost matching; predictions left unmatched are trained toward a "no object" class in DETR.
11. With 10 ms hops, 3 s is about 300 frames × 80 bands; 16×16 patches give $\lfloor 300/16 \rfloor \times 5 = 90$ tokens, against 150 Whisper encoder positions.

## Check your understanding

1. **Why does a ViT need more data than a CNN?** It lacks the CNN's built-in locality and translation equivariance, so it must learn from data which patches relate. With little data the CNN's assumptions win (0.935 against 0.778 with 200 digits); with enough pretraining data the ViT's flexibility pays off, as Dosovitskiy et al. found with ImageNet-21k and JFT-300M.

2. **What problem did DETR remove, and how?** Duplicate detections, which classic detectors removed afterwards with hand-tuned NMS. DETR predicts a fixed set of boxes and trains with a one-to-one matching to ground truth, so duplicates are penalized by the loss and no NMS or anchors are needed.

3. **How does CLIP classify an image into classes it was never trained on?** It embeds a text prompt for each class name and picks the class whose text embedding is most similar to the image embedding. This works when web captions covered the concept, depends on prompt wording, and fails for systematic skills like counting.

4. **Why can a contrastive image-text model ignore word order?** The loss only requires each caption to beat the other captions in the batch. If no negative differs from the true caption only in order or binding, nothing forces the model to encode them, and a bag-of-words encoder cannot encode them at all. Hard negatives with swapped attributes or order supply the missing pressure.

5. **Name the three main ways to connect vision to a language model and one trade-off of each.** Projecting encoder outputs into the LLM's input sequence (simple and cheap to train, but costs context tokens); cross-attention into a frozen LLM (keeps the sequence short and the LLM intact, but adds new layers); early fusion over discrete tokens from all modalities (can generate images and lets modalities interact everywhere, but needs costly joint training and a lossy image tokenizer).

## Further reading

- **Chip Huyen, "Multimodality and Large Multimodal Models (LMMs)"** (2023, blog). https://huyenchip.com/2023/10/10/multimodal.html
  *Read for:* an accessible overview of multimodal tasks, CLIP and Flamingo, and research directions. *Level:* intro.
- **merve and Edward Beeching, "Vision Language Models Explained"** (Hugging Face blog, 2024). https://huggingface.co/blog/vlms
  *Read for:* the encoder, projector and decoder structure of open VLMs, LLaVA-style training stages, and VLM benchmarks. *Level:* intro.
- **Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby, "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"** (2021, ICLR; arXiv:2010.11929). https://arxiv.org/abs/2010.11929
  *Read for:* the ViT architecture and the evidence on data scale versus inductive bias. *Level:* intermediate.
- **Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, Sergey Zagoruyko, "End-to-End Object Detection with Transformers"** (2020, ECCV; arXiv:2005.12872). https://arxiv.org/abs/2005.12872
  *Read for:* detection as set prediction with bipartite matching, without NMS or anchors. *Level:* intermediate.
- **Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, Ross Girshick, "Segment Anything"** (2023, ICCV; arXiv:2304.02643). https://arxiv.org/abs/2304.02643
  *Read for:* promptable segmentation and building a billion-mask dataset with a model in the loop. *Level:* intermediate.
- **Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever, "Learning Transferable Visual Models From Natural Language Supervision"** (2021, ICML; arXiv:2103.00020). https://arxiv.org/abs/2103.00020
  *Read for:* the CLIP objective, zero-shot transfer, prompt engineering, and an unusually thorough limitations section. *Level:* intermediate.
- **Mert Yuksekgonul, Federico Bianchi, Pratyusha Kalluri, Dan Jurafsky, James Zou, "When and why vision-language models behave like bags-of-words, and what to do about it?"** (2023, ICLR; arXiv:2210.01936). https://arxiv.org/abs/2210.01936
  *Read for:* the ARO benchmark and composition-aware hard negatives. *Level:* intermediate.
- **Haotian Liu, Chunyuan Li, Qingyang Wu, Yong Jae Lee, "Visual Instruction Tuning"** (2023, NeurIPS; arXiv:2304.08485). https://arxiv.org/abs/2304.08485
  *Read for:* LLaVA, connecting a vision encoder to an LLM and generating instruction data with a text-only model. *Level:* intermediate.
- **Chameleon Team, "Chameleon: Mixed-Modal Early-Fusion Foundation Models"** (2024, arXiv:2405.09818). https://arxiv.org/abs/2405.09818
  *Read for:* early fusion with discrete image tokens and the training-stability issues it raises. *Level:* advanced.
- **Qwen Team, "Qwen3-VL Technical Report"** (2025, arXiv:2511.21631). https://arxiv.org/abs/2511.21631
  *Read for:* a detailed, recent account of a large open VLM: native-resolution encoder, token merging, multi-level feature injection and positions for video. *Level:* advanced.
- **Shengbang Tong, Zhuang Liu, Yuexiang Zhai, Yi Ma, Yann LeCun, Saining Xie, "Eyes Wide Shut? Exploring the Visual Shortcomings of Multimodal LLMs"** (2024, CVPR; arXiv:2401.06209). https://arxiv.org/abs/2401.06209
  *Read for:* CLIP-blind image pairs and the MMVP benchmark of basic visual failures in VLMs. *Level:* intermediate.
- **Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, Ilya Sutskever, "Robust Speech Recognition via Large-Scale Weak Supervision"** (2023, ICML; arXiv:2212.04356). https://arxiv.org/abs/2212.04356
  *Read for:* Whisper's data, front end, multitask token format and its documented failure modes. *Level:* intermediate.
