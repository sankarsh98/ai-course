# Convolutional Networks

> **Level 2 · Module 3** · Prerequisites: L0-02, L1-01, L2-01, L2-02 · Time: 6 to 8 hours · Code: `code/L2-03-convolutional-networks/`

## What you will be able to do

- Compute a convolution by hand and in code, predict output shapes, parameter counts and cost, and derive its backward pass.
- Explain translation equivariance, why pooling gives only approximate invariance, and test both experimentally.
- Calculate receptive fields and explain why the effective receptive field is smaller.
- Trace the main ideas from LeNet to ResNet, and show why residual connections make deep networks trainable.
- Train a small CNN on CPU and judge when augmentation and transfer learning help.

## 1. Why images need a different layer

Map a 32 by 32 colour image (3 channels) to 16 same-size feature maps with a dense layer (L2-01): about 50 million weights. A $3 \times 3$ convolution doing the same job has 448 (Script 1 prints both). It gets away with this because it builds in two assumptions:

- **Locality.** Edges, corners and textures are local, so each output only looks at a small window.
- **Stationarity.** A pattern means the same thing anywhere, so the same weights are reused at every position (**weight sharing**).

Together they give **translation equivariance**: shift the input and the feature map shifts by the same amount, $f(Sx) = Sf(x)$ for a shift $S$, away from the borders. A dense layer must learn "edge at (3, 5)" and "edge at (20, 11)" separately. These assumptions suit images and other grid data, not tabular data where column order means nothing (L1-03).

## 2. The convolution operation

**1-D first.** Slide the kernel $w = [-1, 0, 1]$ along $x = [0, 0, 1, 2, 3, 0, 0]$, multiplying and summing at each position: $[0, 0, 1] \to 1$, $[0, 1, 2] \to 2$, $[1, 2, 3] \to 2$, $[2, 3, 0] \to -2$, $[3, 0, 0] \to -3$. The output $[1, 2, 2, -2, -3]$ is a discrete derivative. A CNN learns such kernels by gradient descent.

**2-D with channels.** The input has $C_{in}$ channels; a layer has $C_{out}$ kernels of shape $C_{in} \times k \times k$ and one bias each. With stride $s$ (step between windows) and an input already zero-padded by $p$ pixels,

$$y_{o,i,j} = b_o + \sum_{c=1}^{C_{in}} \sum_{u=0}^{k-1} \sum_{v=0}^{k-1} w_{o,c,u,v}\, x_{c,\, si + u,\, sj + v}.$$

Each kernel spans all input channels. A $1 \times 1$ convolution is a dense layer applied at every pixel. Strictly this is **cross-correlation**; true convolution flips the kernel. Libraries compute cross-correlation (PyTorch's `Conv2d` documentation says so), and since kernels are learned the flip doesn't matter.

**Output size** along each axis, with input $n$, kernel $k$, padding $p$, stride $s$, dilation $d$ (spacing between taps, usually 1):

$$n_{out} = \left\lfloor \frac{n + 2p - d(k-1) - 1}{s} \right\rfloor + 1.$$

For $n = 9$, $k = 3$: $p = 0$ gives 7, $p = 1$ gives 9 ("same" padding), $p = 1, s = 2$ gives 5.

**Cost.** Parameters: $C_{out}(C_{in}k^2 + 1)$, independent of image size. Multiply-adds: $C_{in}k^2$ for each of the $C_{out}H_{out}W_{out}$ outputs. Convolutions are cheap in parameters and expensive in compute, mostly in the early high-resolution layers.

**As a matrix multiply.** Copy each $C_{in} \times k \times k$ patch into a column (**im2col**), giving a $(C_{in}k^2) \times (H_{out}W_{out})$ matrix; reshape the kernels to $C_{out} \times C_{in}k^2$; one matrix product computes the layer. This is a common implementation strategy, at the cost of duplicating each input value up to $k^2$ times.

## 3. Backpropagation through a convolution

In 1-D, $y_i = \sum_u w_u x_{i+u}$. With upstream gradient $g_i = \partial L/\partial y_i$:

- $\partial L/\partial w_u = \sum_i g_i x_{i+u}$: the input correlated with $g$. Weight sharing sums the gradient from every position into one weight.
- $\partial L/\partial x_j = \sum_u g_{j-u} w_u$: a *full* convolution (padding $k-1$) of $g$ with the kernel rotated 180 degrees. With channels, also swap the kernel's in and out axes.

So the backward pass of a convolution is another convolution. Used as a layer for upsampling, this operation is called a **transposed convolution**. Script 1 checks both formulas against autograd.

## 4. Pooling and (approximate) invariance

**Max pooling** keeps the largest value in each window, **average pooling** the mean; $2 \times 2$ with stride 2 halves each side. Strided convolution downsamples with learned weights. Downsampling cuts compute, grows the receptive field, and adds some shift tolerance. **Global pooling** reduces each whole feature map to one number, so the classifier asks "is this feature anywhere?" Convolutions plus global pooling are close to **translation invariant**.

Two things break that:

1. **A flatten head.** Flattening feature maps into a dense layer brings absolute position back. In Script 2, shapes appear only in the left part of training images. A CNN with a flatten head is at chance on the right side; the same convolutions with global max pooling get 100%.
2. **Aliasing.** Striding and pooling sample on a grid, so a one-pixel shift changes which values survive. Zhang (2019) showed common CNNs are shift-sensitive for this reason, and that blurring before downsampling improves shift robustness and ImageNet accuracy [Zhang, 2019]. In Script 3 a CNN at 98% accuracy on digits drops to 58% when test images move by one pixel.

## 5. Receptive fields

The **receptive field** $r$ of a unit is the input region that can affect it. Track it with the **jump** $j$, the input distance between neighbouring units. Start with $r = 1$, $j = 1$; for each layer with kernel $k$, stride $s$, dilation $d$:

$$r \leftarrow r + (k-1)\,d\,j, \qquad j \leftarrow j\,s.$$

For conv3, pool2, conv3, pool2, conv3: $r = 3$; pool: $4$, $j = 2$; conv: $4 + 2 \cdot 2 = 8$; pool: $10$, $j = 4$; conv: $10 + 2 \cdot 4 = 18$. Dilation grows $r$ without losing resolution (dilations 1, 2, 4 reach 15). Script 1 checks these by backpropagating from one output unit.

Three $3 \times 3$ layers have the $7 \times 7$ field of one $7 \times 7$ layer, but use $27C^2$ weights instead of $49C^2$ (for $C$ channels) and three nonlinearities instead of one. That argument underlies the VGG design [Simonyan & Zisserman, 2015], and small kernels have been the default since.

**The effective field is smaller.** Luo et al. (2016) measured the gradient of a central output with respect to each input pixel: it is roughly Gaussian, covers only a fraction of the theoretical field, and its width grows like the square root of depth [Luo et al., 2016]. A quick way to see it: with uniform weights and no nonlinearity, $n$ stacked $3 \times 3$ layers give a gradient profile that is the $n$-fold convolution of $[1, 1, 1]$, which tends to a Gaussian with standard deviation $\sqrt{2n/3}$. At $n = 32$ the theoretical field is 65 pixels but the standard deviation is 4.6. Units are dominated by the centre of what they "see".

## 6. Classic architectures: LeNet to ResNet

| Network | Key ideas |
|---|---|
| LeNet [LeCun et al., 1998] | Convolution, subsampling and a small classifier trained end to end on handwritten characters. CNNs beat the other methods compared on digit recognition, and a check-reading system built on them was in commercial use. |
| AlexNet [Krizhevsky et al., 2012] | The recipe scaled to ImageNet (1.3 million training images): five convolutional layers and fully connected layers, 60 million parameters, ReLUs, GPU training, dropout, shift and flip augmentation. Error rates considerably better than the previous state of the art; widely seen as the start of the deep learning era in vision. |
| VGG [Simonyan & Zisserman, 2015] | Uniform stacks of $3 \times 3$ convolutions, 16 to 19 weight layers, then dense layers of width 4096, 4096 and 1000; 133 to 144 million parameters. |
| ResNet [He et al., 2016a] | Residual connections, batch normalization after every convolution (L2-02), global average pooling and one dense layer. Up to 152 layers; an ensemble reached 3.57% top-5 error on ImageNet test and won ILSVRC 2015 classification. |

Depth went up and the big dense head gave way to global pooling. As of October 2026, vision transformers (L3-04) compete with or beat CNNs at large scale, but Liu et al. (2022) showed a modernised pure CNN, ConvNeXt, at 87.8% ImageNet top-1 and competitive with Swin transformers on detection and segmentation [Liu et al., 2022]. Which family is better depends on data, compute and task.

## 7. Residual connections

A deeper network could copy a shallower one by making its extra layers the identity, so it should never be worse. Yet He et al. found that plain networks beyond some depth get *higher training error*: an optimization failure, not overfitting. Their plain nets used batch normalization, and they argued vanishing gradients were unlikely to be the cause [He et al., 2016a]. Script 4 reproduces this: a 40-layer plain CNN ends at training loss 1.9 (24% test accuracy); the residual version of the same layers reaches 0.17 (92%).

The fix is to let a block learn a residual $F$ and add its input back:

$$x_{l+1} = x_l + F(x_l).$$

Doing nothing now only needs $F \approx 0$. In ResNet, $F$ is two $3 \times 3$ conv layers with batch norm; when shapes change, the shortcut is a $1 \times 1$ convolution.

**Why gradients flow.** $\partial x_{l+1}/\partial x_l = I + \partial F/\partial x_l$, so across blocks the Jacobian is a product of $(I + \partial F_i/\partial x_i)$ terms, and expanding it leaves a pure identity term: a direct gradient path to every layer. With "pre-activation" blocks (batch norm and ReLU inside $F$), He et al. (2016b) get exactly $x_L = x_l + \sum_{i=l}^{L-1} F(x_i)$, hence

$$\frac{\partial L}{\partial x_l} = \frac{\partial L}{\partial x_L}\Big(I + \frac{\partial}{\partial x_l}\sum_{i=l}^{L-1} F(x_i)\Big),$$

and they trained a 1001-layer network on CIFAR-10 [He et al., 2016b]. The same additive path appears in the LSTM cell (L2-04) and in transformers (L2-05).

In Script 4 the plain 40-layer net's first-layer gradient at initialization is not vanishing but *large* (13, against 0.7 with shortcuts): deep plain stacks with batch norm can be badly conditioned in the other direction. At depth 4 shortcuts gave no benefit. They fix depth; they are not free accuracy.

## 8. Data augmentation

**Data augmentation** trains on randomly transformed images (shifts, crops, flips, colour changes) to encode invariances you believe the task has. Transformations must preserve the label: flips suit photos of cats, not digits or text, and a 180-degree rotation turns 6 into 9.

In Script 3, random one-pixel shifts lift the CNN's accuracy on shifted test digits from 58% to 97% at no clean-accuracy cost. The MLP with the same augmentation reaches 92% shifted but falls from 97.8% to 93.5% clean: it must memorise every shifted copy, while the CNN's equivariance makes a shifted digit nearly the same problem. Augmentation is regularization (L2-02) with a prior attached, only as good as the match between its transformations and real test-time variation.

## 9. Transfer learning and fine-tuning

Early CNN layers learn edges, colour blobs and textures that look similar across datasets. Yosinski et al. (2014) measured this: early features transfer well, later ones are task-specific, transfer weakens as tasks diverge, and transferred initializations improved generalization even after fine-tuning [Yosinski et al., 2014]. Hence **transfer learning**: start from a network trained on a large source dataset and adapt it. From cheapest to most flexible:

1. **Linear probe.** Freeze the backbone, replace the last layer, train only that.
2. **Partial fine-tuning.** Also unfreeze the last few blocks.
3. **Full fine-tuning.** Train everything with a smaller learning rate than from scratch (3 to 10 times smaller is a common rule of thumb), so features are adjusted rather than destroyed.

Preprocess inputs exactly as the pretrained model expects.

**Transfer is not guaranteed to help.** He, Girshick and Dollár (2019) found COCO detectors trained from scratch matched ImageNet-pretrained ones given longer training, even with 10% of the data; pretraining mostly sped up convergence [He et al., 2019]. Script 5 pretrains on digits 0 to 4 and learns 5 to 9 from 1 to 100 labels per class. Pretraining gave no measurable benefit. Likely reasons, worth asking in real projects: the source is tiny (630 images), adding little beyond the CNN's built-in prior; training to zero source loss specialised the top features; and the target is easy (pixel logistic regression gets 77% from one example per class). In practice transfer is the default for small image datasets because the source is something like ImageNet, which would need downloaded weights this environment lacks.

## Hands-on

Run from the repository root, for example `.venv/bin/python code/L2-03-convolutional-networks/conv_from_scratch.py`. Data are synthetic or scikit-learn's bundled 8 by 8 digits (1,797 images, no download). Outputs are from PyTorch 2.15.0 (CPU) and scikit-learn 1.9.1 on Linux, on a machine busy with other jobs, so times are rough; each script took under 45 seconds. Last digits may vary by platform and thread count.

### Script 1: convolution from scratch

```python
"""Convolution from scratch (im2col), its backward pass, parameter counts and receptive
fields, all checked against PyTorch."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

rng = np.random.default_rng(0)


def out_size(n, k, p, s, d=1):
    return (n + 2 * p - d * (k - 1) - 1) // s + 1


def conv2d(x, w, b, stride=1, pad=0):
    """x: (C_in, H, W), w: (C_out, C_in, k, k). Copy every patch into a column (im2col),
    then compute the whole layer as ONE matrix multiply. Cross-correlation, like PyTorch."""
    C_out, C_in, k, _ = w.shape
    x = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    H_out, W_out = out_size(x.shape[1], k, 0, stride), out_size(x.shape[2], k, 0, stride)
    cols = np.empty((C_in * k * k, H_out * W_out))
    for i in range(H_out):
        for j in range(W_out):
            cols[:, i * W_out + j] = x[:, i * stride:i * stride + k, j * stride:j * stride + k].ravel()
    return (w.reshape(C_out, -1) @ cols + b[:, None]).reshape(C_out, H_out, W_out)


# 1. The 1-D worked example: a derivative filter.
print("1-D:", np.correlate([0., 0., 1., 2., 3., 0., 0.], [-1., 0., 1.], mode="valid"))

# 2. Forward pass against PyTorch, for several strides and paddings.
x, w, b = rng.normal(size=(3, 9, 9)), rng.normal(size=(4, 3, 3, 3)), rng.normal(size=4)
for stride, pad in [(1, 0), (1, 1), (2, 1)]:
    ours = conv2d(x, w, b, stride, pad)
    ref = F.conv2d(torch.tensor(x)[None], torch.tensor(w), torch.tensor(b), stride=stride,
                   padding=pad)[0].numpy()
    print(f"stride={stride} pad={pad}: shape {ours.shape}, formula {out_size(9, 3, pad, stride)}, "
          f"max|ours-torch| {np.abs(ours - ref).max():.1e}")

# 3. Backward pass, with upstream gradient g = dL/dy:
#    dL/dw[o, c] = correlate(x[c], g[o]);  dL/dx = full convolution of g with the kernel
#    rotated 180 degrees and its in/out channels swapped.
xt, wt = torch.tensor(x, requires_grad=True), torch.tensor(w, requires_grad=True)
y = F.conv2d(xt[None], wt)[0]
g = rng.normal(size=tuple(y.shape))
y.backward(torch.tensor(g))
dw = np.stack([np.stack([conv2d(x[c:c + 1], g[o][None, None], np.zeros(1))[0] for c in range(3)])
               for o in range(4)])
dx = conv2d(g, w[:, :, ::-1, ::-1].transpose(1, 0, 2, 3), np.zeros(3), pad=2)  # pad k-1
print(f"backward: max|dw-autograd| {np.abs(dw - wt.grad.numpy()).max():.1e}, "
      f"max|dx-autograd| {np.abs(dx - xt.grad.numpy()).max():.1e}")

# 4. Parameters and multiply-adds on a 32x32x3 image -> 16 maps of 32x32.
print(f"conv 3x3: {16 * 27 + 16:,} params, {16 * 32 * 32 * 27:,} multiply-adds; "
      f"dense: {3072 * 16384 + 16384:,} params")


# 5. Receptive field: formula vs the input pixels the centre output actually depends on.
def rf_formula(layers):            # layers: (kernel, stride, dilation)
    r, jump = 1, 1
    for k, s, d in layers:
        r, jump = r + (k - 1) * d * jump, jump * s
    return r


def rf_measured(layers, size=64):
    net = nn.Sequential(*[nn.Conv2d(1, 1, k, stride=s, dilation=d, bias=False) for k, s, d in layers])
    for m in net:
        nn.init.ones_(m.weight)    # positive weights: no accidental cancellation
    xin = torch.ones(1, 1, size, size, requires_grad=True)
    out = net(xin)
    out[0, 0, out.shape[2] // 2, out.shape[3] // 2].backward()
    rows = xin.grad[0, 0].sum(1).nonzero()
    return int(rows.max() - rows.min() + 1)


for name, layers in {"3 x conv3": [(3, 1, 1)] * 3,
                     "conv3 pool2 conv3 pool2 conv3": [(3, 1, 1), (2, 2, 1)] * 2 + [(3, 1, 1)],
                     "conv3 with dilation 1, 2, 4": [(3, 1, 1), (3, 1, 2), (3, 1, 4)]}.items():
    print(f"receptive field, {name}: formula {rf_formula(layers)}, measured {rf_measured(layers)}")
```

```
1-D: [ 1.  2.  2. -2. -3.]
stride=1 pad=0: shape (4, 7, 7), formula 7, max|ours-torch| 8.9e-16
stride=1 pad=1: shape (4, 9, 9), formula 9, max|ours-torch| 8.9e-16
stride=2 pad=1: shape (4, 5, 5), formula 5, max|ours-torch| 8.9e-16
backward: max|dw-autograd| 7.1e-15, max|dx-autograd| 7.1e-15
conv 3x3: 448 params, 442,368 multiply-adds; dense: 50,348,032 params
receptive field, 3 x conv3: formula 7, measured 7
receptive field, conv3 pool2 conv3 pool2 conv3: formula 18, measured 18
receptive field, conv3 with dilation 1, 2, 4: formula 15, measured 15
```

The im2col version and the hand-derived gradients match PyTorch to rounding error, and the receptive fields match the formula (a $2 \times 2$ stride-2 window stands in for pooling).

### Script 2: shifted test positions

```python
"""Does a model recognise a shape at a position it never saw in training?
Train on shapes in the LEFT part of the image, test on the left and on the RIGHT part."""
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(4)  # fixed thread count, so runs on a shared machine are comparable

SIZE, NOISE = 24, 0.3
T = {  # 5x5 templates, four classes
    "plus":   ["..#..", "..#..", "#####", "..#..", "..#.."],
    "cross":  ["#...#", ".#.#.", "..#..", ".#.#.", "#...#"],
    "square": ["#####", "#...#", "#...#", "#...#", "#####"],
    "tee":    ["#####", "..#..", "..#..", "..#..", "..#.."],
}
TEMPLATES = [np.array([[c == "#" for c in row] for row in t], dtype=np.float32) for t in T.values()]


def make_data(n, cols, rng):
    """n noisy images; each shape's left column is drawn from `cols`, its top row anywhere."""
    X = rng.normal(0, NOISE, size=(n, 1, SIZE, SIZE)).astype(np.float32)
    y = rng.integers(0, len(TEMPLATES), size=n)
    for i in range(n):
        r, c = rng.integers(0, SIZE - 5 + 1), rng.choice(cols)
        X[i, 0, r:r + 5, c:c + 5] += TEMPLATES[y[i]]
    return torch.tensor(X), torch.tensor(y)


def mlp():
    return nn.Sequential(nn.Flatten(), nn.Linear(SIZE * SIZE, 128), nn.ReLU(), nn.Linear(128, 4))


def convs():
    return [nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2)]


def cnn_flatten():   # convolutional features, but a dense head that sees absolute positions
    return nn.Sequential(*convs(), nn.Flatten(), nn.Linear(32 * 12 * 12, 4))


def cnn_global_pool():  # global max pool: "is the feature anywhere?" then classify
    return nn.Sequential(*convs(), nn.AdaptiveMaxPool2d(1), nn.Flatten(), nn.Linear(32, 4))


def train(model, X, y, steps=1500, seed=0):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(steps):
        idx = torch.randint(0, len(X), (64,), generator=g)
        loss = nn.functional.cross_entropy(model(X[idx]), y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


@torch.no_grad()
def accuracy(model, X, y):
    return (model(X).argmax(1) == y).float().mean().item()


rng = np.random.default_rng(0)
left, right = np.arange(0, 9), np.arange(12, SIZE - 5 + 1)
Xtr, ytr = make_data(3000, left, rng)
Xl, yl = make_data(1000, left, rng)
Xr, yr = make_data(1000, right, rng)
print(f"train: shapes start in columns {left.min()}-{left.max()}; "
      f"right test: columns {right.min()}-{right.max()}")
print(f"{'model':18s} {'params':>7s} {'train acc':>9s} {'left test':>9s} {'right test':>10s}")
for name, make in [("MLP", mlp), ("CNN + flatten", cnn_flatten), ("CNN + global pool", cnn_global_pool)]:
    torch.manual_seed(0)
    model = train(make(), Xtr, ytr)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"{name:18s} {n_par:7,d} {accuracy(model, Xtr, ytr):9.3f} {accuracy(model, Xl, yl):9.3f} "
          f"{accuracy(model, Xr, yr):10.3f}")

# Equivariance check: shifting the input shifts the conv feature map by the same amount.
conv = nn.Sequential(*convs()[:4])
x = Xl[:1]
shifted = torch.roll(x, shifts=3, dims=3)
f, fs = conv(x), conv(shifted)
inner = (slice(None), slice(None), slice(None), slice(6, -6))  # skip columns near the wrap and the padding
err = (torch.roll(f, 3, dims=3) - fs)[inner].abs().max().item()
print(f"\nconv(shift(x)) vs shift(conv(x)), interior columns: max abs difference {err:.1e}")
```

```
train: shapes start in columns 0-8; right test: columns 12-19
model               params train acc left test right test
MLP                 74,372     1.000     0.390      0.255
CNN + flatten       23,236     1.000     0.912      0.263
CNN + global pool    4,932     1.000     1.000      1.000

conv(shift(x)) vs shift(conv(x)), interior columns: max abs difference 0.0e+00
```

All three fit the training set. The MLP fails even on new positions inside the training region (39%) and is at chance (25%) outside it. The flatten head generalizes only within the region it saw. Global pooling, with a fifteenth of the MLP's parameters, is perfect everywhere. The last line confirms equivariance away from the padding and the wrap-around of `torch.roll`.

### Script 3: a small image classifier on CPU

```python
"""A small image classifier on CPU: scikit-learn's 8x8 digits. Compares logistic regression,
an MLP, a CNN, and a CNN trained with shift augmentation, on clean and shifted test images."""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

torch.set_num_threads(4)  # fixed thread count, so runs on a shared machine are comparable

X, y = load_digits(return_X_y=True)
X = (X / 16.0).astype(np.float32).reshape(-1, 1, 8, 8)   # pixel values 0..16 -> 0..1
# Same split as L1-02 so the logistic-regression number is comparable.
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
X_tr, X_te = torch.tensor(X_tr), torch.tensor(X_te)
y_tr, y_te = torch.tensor(y_tr), torch.tensor(y_te)


def shift(x, gen, max_shift=1):
    """Shift each image by up to max_shift pixels in each direction, filling with zeros."""
    n = x.shape[0]
    padded = F.pad(x, (max_shift,) * 4)
    d = torch.randint(0, 2 * max_shift + 1, (n, 2), generator=gen)
    return torch.stack([padded[i, :, d[i, 0]:d[i, 0] + 8, d[i, 1]:d[i, 1] + 8] for i in range(n)])


def mlp():
    return nn.Sequential(nn.Flatten(), nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 10))


def cnn():
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),     # 8x8 -> 4x4
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),     # 4x4 -> 2x2
        nn.Flatten(), nn.Linear(64 * 2 * 2, 10))


def train(model, augment, seed, epochs=40, bs=64):
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
    for _ in range(epochs):
        perm = torch.randperm(len(X_tr), generator=gen)
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            xb = shift(X_tr[idx], gen) if augment else X_tr[idx]
            loss = F.cross_entropy(model(xb), y_tr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model.eval()


@torch.no_grad()
def acc(model, x):
    return (model(x).argmax(1) == y_te).float().mean().item()


# A fixed "shifted" test set: every test image moved by up to 1 pixel.
X_te_shift = shift(X_te, torch.Generator().manual_seed(123))

lr = LogisticRegression(max_iter=5000).fit(X_tr.reshape(len(X_tr), -1), y_tr)
print(f"train {len(X_tr)} images, test {len(X_te)} images")
print(f"{'model':22s} {'params':>7s} {'test acc':>16s} {'shifted test acc':>18s} {'time/seed':>9s}")
print(f"{'logistic regression':22s} {650:7d} {lr.score(X_te.reshape(len(X_te), -1), y_te):16.3f} "
      f"{lr.score(X_te_shift.reshape(len(X_te), -1), y_te):18.3f}")
for name, make, augment in [("MLP", mlp, False), ("MLP + shift aug", mlp, True),
                            ("CNN", cnn, False), ("CNN + shift aug", cnn, True)]:
    clean, shifted = [], []
    t0 = time.time()
    for seed in range(3):
        torch.manual_seed(seed)
        model = train(make(), augment, seed)
        clean.append(acc(model, X_te))
        shifted.append(acc(model, X_te_shift))
    n_par = sum(p.numel() for p in model.parameters())
    print(f"{name:22s} {n_par:7,d} {np.mean(clean):9.3f} +- {np.std(clean):.3f} "
          f"{np.mean(shifted):11.3f} +- {np.std(shifted):.3f} {(time.time() - t0) / 3:8.1f}s")
```

```
train 1257 images, test 540 images
model                   params         test acc   shifted test acc time/seed
logistic regression        650            0.970              0.415
MLP                      9,610     0.978 +- 0.002       0.430 +- 0.005      0.5s
MLP + shift aug          9,610     0.935 +- 0.005       0.919 +- 0.005      0.9s
CNN                     25,866     0.984 +- 0.001       0.576 +- 0.010      1.5s
CNN + shift aug         25,866     0.986 +- 0.003       0.971 +- 0.005      2.2s
```

The CNN beats logistic regression (97.0%, the L1-02 split) and the MLP on clean digits by a little; the big differences are on shifted digits, a harsh test since one pixel is 12.5% of the width (Section 8). Hyperparameters were fixed in advance; to tune them, use a validation split (L1-01).

### Script 4: plain versus residual

```python
"""Plain deep CNNs vs residual CNNs of the same depth on digits.
Same layers, same optimizer, same budget; the only difference is the identity shortcut."""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

torch.set_num_threads(4)  # fixed thread count, so runs on a shared machine are comparable

X, y = load_digits(return_X_y=True)
X = (X / 16.0).astype(np.float32).reshape(-1, 1, 8, 8)
X_tr, X_te, y_tr, y_te = map(torch.tensor, train_test_split(X, y, test_size=0.3, random_state=0,
                                                            stratify=y))
C = 16  # channels everywhere; all layers work at 8x8 resolution


def conv_bn(relu=True):
    layers = [nn.Conv2d(C, C, 3, padding=1, bias=False), nn.BatchNorm2d(C)]
    return nn.Sequential(*layers, nn.ReLU()) if relu else nn.Sequential(*layers)


class PlainPair(nn.Module):        # two conv layers, no shortcut
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(conv_bn(), conv_bn(relu=False))

    def forward(self, x):
        return F.relu(self.f(x))


class ResidualPair(nn.Module):     # the same two conv layers, plus the identity: relu(x + F(x))
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(conv_bn(), conv_bn(relu=False))

    def forward(self, x):
        return F.relu(x + self.f(x))


def net(depth, block):
    """depth = number of 3x3 conv layers in the body (2 per block)."""
    return nn.Sequential(nn.Conv2d(1, C, 3, padding=1), nn.ReLU(),
                         *[block() for _ in range(depth // 2)],
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(C, 10))


def first_layer_grad_norm(model):
    model.train()
    loss = F.cross_entropy(model(X_tr[:256]), y_tr[:256])
    model.zero_grad()
    loss.backward()
    return model[0].weight.grad.norm().item()


def train(model, seed, epochs=10, bs=128):
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(X_tr), generator=gen)
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            loss = F.cross_entropy(model(X_tr[idx]), y_tr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        train_loss = F.cross_entropy(model(X_tr), y_tr).item()
        test_acc = (model(X_te).argmax(1) == y_te).float().mean().item()
    return train_loss, test_acc


print(f"{'body depth':>10s} {'type':>9s} {'grad norm, layer 1, init':>25s} "
      f"{'final train loss':>17s} {'test acc':>9s} {'time':>6s}")
for depth in [4, 16, 40]:
    for name, block in [("plain", PlainPair), ("residual", ResidualPair)]:
        t0 = time.time()
        g, losses, accs = [], [], []
        for seed in range(2):
            torch.manual_seed(seed)
            model = net(depth, block)
            g.append(first_layer_grad_norm(model))
            tl, ta = train(model, seed)
            losses.append(tl)
            accs.append(ta)
        print(f"{depth:10d} {name:>9s} {np.mean(g):25.4f} {np.mean(losses):17.3f} "
              f"{np.mean(accs):9.3f} {time.time() - t0:5.1f}s")
```

```
body depth      type  grad norm, layer 1, init  final train loss  test acc   time
         4     plain                    0.0894             0.075     0.970   1.4s
         4  residual                    0.1198             0.186     0.919   0.7s
        16     plain                    0.2937             1.484     0.564   2.5s
        16  residual                    0.3760             0.011     0.981   2.6s
        40     plain                   13.1087             1.905     0.235   6.1s
        40  residual                    0.6813             0.165     0.922   6.3s
```

At depth 16 and 40, plain nets fail to fit the training data; residual nets of the same depth do. Each cell averages only two seeds, so small gaps are noise.

### Script 5: transfer learning in miniature

```python
"""Transfer learning in miniature. Pretrain a CNN on digits 0-4, then learn digits 5-9 from
only k labelled examples per class: from scratch, as a linear probe on frozen features,
or by fine-tuning everything. Each setting is repeated over 5 random labelled subsets."""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

torch.set_num_threads(4)  # fixed thread count, so runs on a shared machine are comparable

X, y = load_digits(return_X_y=True)
X = (X / 16.0).astype(np.float32).reshape(-1, 1, 8, 8)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
src_tr = y_tr < 5                      # source task: digits 0-4
tgt_tr, tgt_te = y_tr >= 5, y_te >= 5  # target task: digits 5-9, relabelled 0-4
Xs, ys = torch.tensor(X_tr[src_tr]), torch.tensor(y_tr[src_tr])
Xt_pool, yt_pool = X_tr[tgt_tr], y_tr[tgt_tr] - 5
Xt_te, yt_te = torch.tensor(X_te[tgt_te]), torch.tensor(y_te[tgt_te] - 5)


def features():
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten())


def fit(model, params, x, y, steps, lr):
    opt = torch.optim.Adam(params, lr=lr)
    for _ in range(steps):
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


@torch.no_grad()
def acc(model):
    return (model(Xt_te).argmax(1) == yt_te).float().mean().item()


# Pretrain on the source task (about 630 images).
torch.manual_seed(0)
pre = nn.Sequential(features(), nn.Linear(256, 5))
g = torch.Generator().manual_seed(0)
opt = torch.optim.Adam(pre.parameters(), lr=2e-3)
for step in range(600):
    idx = torch.randint(0, len(Xs), (64,), generator=g)
    loss = F.cross_entropy(pre(Xs[idx]), ys[idx])
    opt.zero_grad()
    loss.backward()
    opt.step()
print(f"pretrained on digits 0-4: {len(Xs)} images, final batch loss {loss.item():.3f}")
print(f"target task digits 5-9: test set {len(Xt_te)} images\n")

print(f"{'labels/class':>12s} {'pixels+logreg':>14s} {'scratch CNN':>12s} {'linear probe':>13s} "
      f"{'fine-tune':>10s}")
for k in [1, 5, 20, 100]:
    res = {m: [] for m in ["logreg", "scratch", "probe", "finetune"]}
    for seed in range(5):
        rng = np.random.default_rng(seed)
        idx = np.concatenate([rng.choice(np.flatnonzero(yt_pool == c), k, replace=False)
                              for c in range(5)])
        xk, yk = torch.tensor(Xt_pool[idx]), torch.tensor(yt_pool[idx])
        lr_model = LogisticRegression(max_iter=5000).fit(xk.reshape(len(xk), -1), yk)
        res["logreg"].append(lr_model.score(Xt_te.reshape(len(Xt_te), -1), yt_te))

        torch.manual_seed(seed)
        scratch = nn.Sequential(features(), nn.Linear(256, 5))
        res["scratch"].append(acc(fit(scratch, scratch.parameters(), xk, yk, 200, 2e-3)))

        probe = nn.Sequential(copy.deepcopy(pre[0]), nn.Linear(256, 5))
        for p in probe[0].parameters():
            p.requires_grad_(False)                  # frozen pretrained features
        res["probe"].append(acc(fit(probe, probe[1].parameters(), xk, yk, 200, 2e-3)))

        ft = nn.Sequential(copy.deepcopy(pre[0]), nn.Linear(256, 5))
        res["finetune"].append(acc(fit(ft, ft.parameters(), xk, yk, 200, 5e-4)))  # smaller lr
    print(f"{k:12d} " + " ".join(f"{np.mean(v):7.3f}+-{np.std(v):.3f}" for v in res.values()))
```

```
pretrained on digits 0-4: 630 images, final batch loss 0.000
target task digits 5-9: test set 269 images

labels/class  pixels+logreg  scratch CNN  linear probe  fine-tune
           1   0.765+-0.063   0.772+-0.067   0.731+-0.046   0.751+-0.060
           5   0.897+-0.032   0.915+-0.031   0.891+-0.029   0.907+-0.034
          20   0.949+-0.006   0.958+-0.015   0.938+-0.012   0.952+-0.015
         100   0.988+-0.003   0.990+-0.004   0.977+-0.006   0.984+-0.004
```

Pretraining never helps, and the frozen probe is the weakest CNN option. Spreads across subsets are as large as most gaps at 1 and 5 labels, so the fair summary is "no measurable benefit", not "transfer hurts".

## Common mistakes

- **Shape errors at the first dense layer.** Compute shapes with the formula, or push a dummy tensor through and print each shape.
- **Channels in the wrong place.** PyTorch wants `(batch, channels, height, width)`; image files give `(height, width, channels)`. A wrong reshape often runs and trains on scrambled data. Plot inputs.
- **Assuming shift invariance.** Test robustness to the transformations you care about, as in Scripts 2 and 3.
- **Label-changing augmentation**, or augmenting the test set by accident. Look at augmented samples.
- **Fine-tuning with a from-scratch learning rate**, or preprocessing differently from the pretrained model. Both fail silently.
- **One-seed comparisons.** Gaps of a point or two on small data are often seed noise.

## Exercises

1. [warm-up] A $64 \times 64 \times 3$ input goes through 32 filters of $5 \times 5$, stride 2, padding 2. Give the output shape, parameters and multiply-adds.
2. [warm-up] By hand, find the receptive field and jump of conv7 stride 2, maxpool3 stride 2, then three conv3 layers. Check with `rf_formula`.
3. [warm-up] What do the 1-D kernels $[1, 2, 1]/4$ and $[1, -2, 1]$ do?
4. [core] Add dilation to `conv2d` in Script 1 and check it against `F.conv2d(..., dilation=2)`.
5. [core] Derive the gradient of $2 \times 2$ max pooling, implement it in NumPy, and check it against autograd. What happens on ties?
6. [core] In Script 2, swap global max pooling for global average pooling, then add a stride-2 `MaxPool2d` after each conv. How do right-side accuracy and shift sensitivity change?
7. [core] In Script 3, add horizontal-flip augmentation. Which digit classes suffer, and why?
8. [core] In Script 4, train 3 times longer, then remove batch norm from both variants. What changes? Does zero-initializing the last conv of each residual branch help?
9. [stretch] Make transfer win in Script 5 (for example: more varied source data via augmentation, a harder target with noise or shifts, earlier stopping of pretraining). Report 5 seeds and say which change mattered.

### Hints and answers

1. $\lfloor (64 + 4 - 4 - 1)/2 \rfloor + 1 = 32$, so $32 \times 32 \times 32$. Parameters $32(75 + 1) = 2432$. Multiply-adds $32 \cdot 32 \cdot 32 \cdot 75 = 2{,}457{,}600$.
2. $r = 7, j = 2$; pool: $r = 11, j = 4$; each conv3 adds 8: $r = 35$, $j = 4$.
3. A blur; a second derivative (responds to peaks and dips, zero on straight ramps).
4. Slice patches with step $d$; the effective kernel width is $d(k-1) + 1$.
5. Only each window's maximum gets gradient; on ties PyTorch picks one index.
6. Average pooling should also generalize across positions. Stride-2 pooling makes feature maps change by more than a shift (aliasing).
7. 2, 3, 4, 5, 7 and 9 become non-digits; 0, 1 and 8 survive. Expect lower clean accuracy.
8. The plain net should improve slowly but stay behind. Without batch norm, residual activations can grow with depth; zero-initialized branches start each block as the identity, a widely used fix.
9. Transfer tends to win when the source is richer than the architectural prior and target labels are scarce. Check the linear probe against pixel logistic regression first.

## Check your understanding

1. **Why does a conv layer need far fewer parameters than a dense layer, and what does it give up?** Locality and weight sharing make its parameter count independent of image size. It gives up treating positions differently; when position matters, something else must supply it.

2. **Is a CNN translation invariant?** Its conv layers are equivariant up to borders. Pooling gives only approximate invariance (strided sampling aliases), and a flatten head reintroduces position. Script 3's CNN fell from 98% to 58% under one-pixel shifts.

3. **Four $3 \times 3$ convs with a stride-2 pool after the second: receptive field, and is it used evenly?** $3, 5$, pool $6$ (jump 2), then $10, 14$. Not evenly: the effective field is roughly Gaussian and concentrated in the centre [Luo et al., 2016].

4. **Why can a 40-layer plain net have higher training loss than a 4-layer one, and how do shortcuts help?** Optimization, not overfitting: the identity solution exists but gradient descent doesn't find it. With $x + F(x)$ the identity is $F = 0$, and the Jacobian $I + \partial F/\partial x$ gives gradients a direct path back.

5. **You have 300 labelled images of a new defect type. What do you try?** A CNN pretrained on a large varied dataset, preprocessed as it expects: linear probe first, then full fine-tuning at a small learning rate, with label-preserving augmentation matched to real variation. Compare with a from-scratch baseline, choose on a validation split, test once, and report spread over seeds (L1-01). Transfer usually helps at this size, but measure it [He et al., 2019].

## Further reading

- **CS231n course staff, "Convolutional Neural Networks (CNNs / ConvNets)"** (Stanford CS231n notes, accessed October 2026). https://cs231n.github.io/convolutional-networks/
  *Read for:* conv arithmetic, parameter sharing, pooling, and case studies from LeNet to ResNet. *Level:* intro.
- **Aston Zhang, Zachary C. Lipton, Mu Li, Alexander J. Smola, "Dive into Deep Learning"** (2023, arXiv:2106.11342; free online). https://d2l.ai/chapter_convolutional-neural-networks/
  *Read for:* CNN chapters with runnable PyTorch LeNet, VGG and ResNet. *Level:* intro.
- **Yann LeCun, Léon Bottou, Yoshua Bengio, Patrick Haffner, "Gradient-Based Learning Applied to Document Recognition"** (1998, Proceedings of the IEEE 86(11):2278-2324). https://doi.org/10.1109/5.726791
  *Read for:* LeNet and the case for learned features. Publisher copy may need a subscription; free copies are linked from https://bottou.org/papers/lecun-98h. *Level:* intermediate.
- **Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton, "ImageNet Classification with Deep Convolutional Neural Networks"** (2012, NeurIPS 25). https://proceedings.neurips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html
  *Read for:* the network that made large GPU-trained CNNs standard. *Level:* intermediate.
- **Karen Simonyan, Andrew Zisserman, "Very Deep Convolutional Networks for Large-Scale Image Recognition"** (2015, ICLR; arXiv:1409.1556). https://arxiv.org/abs/1409.1556
  *Read for:* the case for $3 \times 3$ stacks and a careful study of depth. *Level:* intermediate.
- **Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, "Deep Residual Learning for Image Recognition"** (2016, CVPR; arXiv:1512.03385). https://arxiv.org/abs/1512.03385
  *Read for:* the degradation problem and residual blocks. Cited as [He et al., 2016a]. *Level:* intermediate.
- **Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, "Identity Mappings in Deep Residual Networks"** (2016, ECCV; arXiv:1603.05027). https://arxiv.org/abs/1603.05027
  *Read for:* gradient propagation through shortcuts and pre-activation blocks. Cited as [He et al., 2016b]. *Level:* advanced.
- **Wenjie Luo, Yujia Li, Raquel Urtasun, Richard Zemel, "Understanding the Effective Receptive Field in Deep Convolutional Neural Networks"** (2016, NeurIPS 29; arXiv:1701.04128). https://arxiv.org/abs/1701.04128
  *Read for:* why the effective receptive field is Gaussian and small. *Level:* advanced.
- **Richard Zhang, "Making Convolutional Networks Shift-Invariant Again"** (2019, ICML; arXiv:1904.11486). https://arxiv.org/abs/1904.11486
  *Read for:* how pooling and striding break shift invariance, and the anti-aliasing fix. *Level:* intermediate.
- **Jason Yosinski, Jeff Clune, Yoshua Bengio, Hod Lipson, "How transferable are features in deep neural networks?"** (2014, NeurIPS 27; arXiv:1411.1792). https://arxiv.org/abs/1411.1792
  *Read for:* layer-by-layer measurements of feature generality. *Level:* intermediate.
- **Kaiming He, Ross Girshick, Piotr Dollár, "Rethinking ImageNet Pre-training"** (2019, ICCV; arXiv:1811.08883). https://arxiv.org/abs/1811.08883
  *Read for:* when training from scratch matches pretraining. Cited as [He et al., 2019]. *Level:* intermediate.
- **Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, Saining Xie, "A ConvNet for the 2020s"** (2022, CVPR; arXiv:2201.03545). https://arxiv.org/abs/2201.03545
  *Read for:* how far a modernised CNN goes against vision transformers. *Level:* advanced.
