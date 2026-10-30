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
